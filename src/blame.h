// blame.h — direct logit attribution: who wrote this token's logit?
//
// The residual stream at the last layer is a plain sum:
//
//     x_final = embed + Σ_l attn_out_l + Σ_l Σ_n a_ln · down_ln
//
// and the logit of token t is
//
//     logit_t = (g ⊙ W_U[t]) · x_final / rms(x_final)          (rms over n_embd, + eps)
//
// Freezing the norm's scale 1/rms (it is one number per position), the logit
// splits *exactly* into one term per piece:
//
//     embed:    u · embed / rms
//     attn l:   u · attn_out_l / rms
//     neuron:   a_ln · (u · down_ln) / rms          with u = g ⊙ W_U[t]
//
// so the parts must add back up to the logit the model produced — the chart
// checks itself every time (`unexplained` below should be ~0.01, the gap
// being ggml's reduced-precision matmul inputs vs our f32 sums).
//
// Attention splits the same way, one term per head: attn_out_l = Wo · x where x
// (the output projection's input, RunOptions::capture_heads) is the heads' outputs
// side by side, so with w = Woᵀ u
//
//     head h:   w[h-block] · x[h-block] / rms
//
// (h-block = that head's head_dim rows; for Qwen3.5 linear-attention layers, a
// value head's dv rows of the gated, normed DeltaNet output before ssm_out). The
// heads of a layer sum back to its attention term; `head_gap` checks that.
//
// Models with post-norms (Gemma) add rms_norm(v)·g instead of v: attention's and
// the MLP's outputs are each normalized before joining the stream, and Gemma 4 also
// multiplies each layer's whole output by a scalar s_l. Freezing those norms' 1/rms
// too (one number per piece per position, captured as attn_raw / mlp_raw), the
// decomposition stays exact:
//
//     neuron:   a_ln · (u ⊙ g_post_l · S_l) · down_ln / rms(mlp_raw_l) / rms
//     head:     (Wo_lᵀ (u ⊙ g_attn_l · S_l))[h] · x[h] / rms(attn_raw_l) / rms
//
// where S_l = s_l · s_{l+1} ⋯ is how much the later layers' output scales multiply
// what layer l adds (the embedding gets S_0). down_dots / out_dots fold g and S in.
// With a soft-capped head (logits = c·tanh(z/c)) blame explains z, the logit
// before the cap; prob and rank are unchanged by it (tanh is monotone).
//
// What it does NOT see: indirect effects. A neuron in layer 11 also changes
// what layers 12–15 compute; that shows up credited to *those* later pieces.
// So blame is exact about the last step and a lead, not a verdict, for early
// layers — confirm with an ablation.
#pragma once

#include <vector>

#include "forward.h"
#include "gguf_file.h"
#include "model_layout.h"

namespace neuronaut {

// u = g ⊙ W_U[token]: the direction in residual space that raises `token`'s logit.
std::vector<float> unembed_dir(const ModelLayout& m, int token);

// For every layer l and neuron n: down_ln · (u ⊙ g_post_l) · S_l (g = 1 and S = 1
// on models without post-norms / output scales). One pass over each ffn_down (row e
// is contiguous over neurons). [layer][neuron].
std::vector<std::vector<float>> down_dots(const ModelLayout& m, const std::vector<float>& u);

// S_l: the product of layer_output_scale over layers l … n_layer−1 (all 1 when the
// model has none), plus S_{n_layer} = 1 at the end. [n_layer + 1].
std::vector<float> later_scales(const ModelLayout& m);

// For every layer: Woᵀ (u ⊙ g_attn_l) · S_l (attn_output, or ssm_out for linear attention) — the
// logit change per unit of each output-projection input. [layer][input dim].
std::vector<std::vector<float>> out_dots(const ModelLayout& m, const std::vector<float>& u);

// Heads of layer l and the width of each (rows of the output projection's input).
int n_heads(const ModelLayout& m, int layer);
int head_width(const ModelLayout& m, int layer);

struct BlameItem {
    enum Kind { Embed, Attn, Neuron, Head } kind;
    int layer = -1;      // Attn / Neuron / Head
    int neuron = -1;     // Neuron: the neuron; Head: the head
    float act = 0;       // Neuron: activation actually written (after interventions)
    float dot = 0;       // Neuron: down · u
    float contrib = 0;   // contribution to the logit
};

struct Blame {
    int pos = 0;
    int target = 0;
    float logit = 0;         // the model's logit for target at pos (before a soft-cap: what the parts sum to)
    float prob = 0;          // softmax probability
    int rank = 0;            // 1 = argmax
    float rms_inv = 0;       // 1/rms(x_final) at pos
    float embed = 0;         // contribution of the token embedding
    float sum_attn = 0;      // Σ attention contributions
    float sum_neurons = 0;   // Σ neuron contributions
    float unexplained = 0;   // logit - (embed + attn + neurons)
    float head_gap = 0;      // sum_attn - Σ heads (0 up to matmul precision; 0 when no heads)
    std::vector<BlameItem> attn;      // one per layer
    std::vector<BlameItem> neurons;   // every neuron (n_layer × n_ff)
    std::vector<BlameItem> heads;     // every head of every layer, when the run captured them

    // The k strongest items pushing toward / against the target: neurons, the
    // embedding, and attention — per head when heads were captured, else per layer.
    std::vector<BlameItem> top(int k, bool toward) const;
    // The heads of one layer, in head order (empty without captured heads).
    std::vector<BlameItem> layer_heads(int layer) const;
};

// `r` must come from a run with capture_all_layers + capture_resid + all_logits.
// `iv` is the intervention set that run used (activations are captured before
// interventions; blame credits what was actually written). `dots` from down_dots();
// `odots` from out_dots(), used when the run also had capture_heads (may be null).
Blame compute_blame(const ModelLayout& m, const ForwardResult& r, const Interventions& iv,
                    const std::vector<std::vector<float>>& dots, const std::vector<float>& u, int target, int pos,
                    const std::vector<std::vector<float>>* odots = nullptr);

}  // namespace neuronaut
