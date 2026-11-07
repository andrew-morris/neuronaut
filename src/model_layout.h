// model_layout.h — what the tensors in a GGUF *mean*, for one architecture.
//
// GGUF stores architecture hyperparameters under "<arch>.<key>" metadata
// (llama.block_count, llama.embedding_length, …) and names tensors with a fixed
// scheme (blk.<L>.ffn_up.weight, …). ModelLayout reads both and answers questions
// like "which tensors make up neuron N of layer L?".
//
// Supported architectures:
//
//   llama   dense Llama 2/3 (and anything llama.cpp converts as "llama"). Every layer:
//
//     x ─ attn_norm ─ attn_q/k/v ─ attention ─ attn_output ─(+x)─ ffn_norm ─┐
//                                                                           │
//        ┌────────── ffn_gate ── silu ──┐                                   │
//        │                              ×── a[n_ff] ── ffn_down ──(+x)─ next layer
//        └────────── ffn_up ────────────┘
//
//   qwen35  dense Qwen3.5 (0.8B … 27B). Hybrid: three of every four layers are
//           *linear attention* (Gated DeltaNet — a recurrent state instead of a
//           KV cache), every fourth is gated full attention. The MLP half of every
//           layer is the same SwiGLU as Llama's, so "neuron" means exactly the
//           same thing. Differences that matter here:
//             - the pre-MLP norm is named post_attention_norm, not ffn_norm;
//             - the converter has already baked HF's `1 + weight` into every norm
//               gain except the linear-attention ssm_norm — use gains as stored;
//             - block_count includes the multi-token-prediction (MTP) head's block
//               (nextn_predict_layers of them, at the end). Those are a separate
//               speculative-decoding head, not part of the forward pass; we exclude
//               them from n_layer and list them in mtp_layers.
//
//   gemma3  Gemma 3 (1B … 27B). Llama-shaped, plus: the embedding is scaled by
//           √n_embd; Q and K get a per-head RMSNorm; five of every six layers use
//           sliding-window attention (window n_swa, RoPE base rope.freq_base_swa),
//           the sixth global attention (rope.freq_base, linear RoPE scaling); Q is
//           pre-scaled by 1/√head_dim; the MLP is GeGLU (gelu-tanh gate) and there is
//           an RMSNorm *after* attention and after the MLP (post_attention_norm,
//           post_ffw_norm) as well as before. The converter bakes HF's `1 + w` into
//           every norm gain.
//
//   gemma4  Gemma 4, dense text models (12B, 31B). Gemma 3's skeleton, plus: head
//           size and KV-head count differ between sliding (key_length_swa) and full
//           layers; full layers use "proportional" partial RoPE (frequency factors
//           in rope_freqs.weight); V gets an unweighted RMSNorm; K = V when the file
//           has no attn_v (attention_k_eq_v); Q isn't pre-scaled; each layer's output
//           is multiplied by a scalar (layer_output_scale); the logits are
//           soft-capped (c·tanh(z/c)). Norm gains are stored as used. Not supported:
//           per-layer embeddings, KV-shared layers and per-layer MLP widths (E2B/E4B),
//           and MoE (26B-A4B) — ModelLayout::from refuses them.
//
// "Neuron n" of layer L is one entry of the n_ff-wide activation vector `a`.
// It *reads* the residual stream through row n of ffn_gate and row n of ffn_up,
// and *writes* back through column n of ffn_down.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "gguf_file.h"

namespace neuronaut {

// What the attention half of a layer is. The MLP half is always SwiGLU.
enum class LayerKind {
    attention,         // softmax attention over a KV cache (llama; qwen35 full-attention layers)
    linear_attention,  // Gated DeltaNet (qwen35): recurrent state, no KV cache
};

const char* layer_kind_name(LayerKind k);   // "attn" / "linear"

struct LayerTensors {
    LayerKind kind = LayerKind::attention;

    const TensorInfo* attn_norm = nullptr;    // pre-attention RMSNorm gain

    // Softmax attention. Q/K/V are separate, or fused in attn_qkv.
    const TensorInfo* attn_q = nullptr;       // qwen35: [n_embd, 2·n_head·head_dim] — Q and its output gate, interleaved per head
    const TensorInfo* attn_k = nullptr;
    const TensorInfo* attn_v = nullptr;
    const TensorInfo* attn_qkv = nullptr;     // fused Q/K/V (some conversions; qwen35 linear layers)
    const TensorInfo* attn_output = nullptr;
    const TensorInfo* attn_q_norm = nullptr;  // qwen35: per-head RMSNorm on Q and K
    const TensorInfo* attn_k_norm = nullptr;

    // Linear attention (qwen35 Gated DeltaNet). attn_qkv above holds its fused Q/K/V
    // projection; attn_gate is the output gate z.
    const TensorInfo* attn_gate = nullptr;
    const TensorInfo* ssm_conv1d = nullptr;   // short causal conv over Q/K/V channels
    const TensorInfo* ssm_dt = nullptr;       // dt bias (per value head)
    const TensorInfo* ssm_a = nullptr;        // −exp(A_log), per value head (converter applies it)
    const TensorInfo* ssm_beta = nullptr;     // write-strength projection
    const TensorInfo* ssm_alpha = nullptr;    // decay projection
    const TensorInfo* ssm_norm = nullptr;     // gated RMSNorm on the output (gain NOT +1-baked)
    const TensorInfo* ssm_out = nullptr;      // output projection back to n_embd

    // MLP.
    const TensorInfo* ffn_norm = nullptr;     // pre-MLP RMSNorm gain (qwen35: post_attention_norm)
    const TensorInfo* ffn_gate = nullptr;
    const TensorInfo* ffn_up = nullptr;
    const TensorInfo* ffn_down = nullptr;

    // Gemma: RMSNorms on what attention / the MLP add to the stream (null elsewhere),
    // and Gemma 4's per-layer scalar on the layer's whole output.
    const TensorInfo* attn_post_norm = nullptr;
    const TensorInfo* ffn_post_norm = nullptr;
    const TensorInfo* out_scale = nullptr;

    // Attention shape of this layer. The same for every layer of llama / qwen35;
    // per layer for gemma4 (sliding vs full).
    int head_dim = 0, n_head = 0, n_head_kv = 0;
    int n_rot = 0;                // dims RoPE rotates
    bool swa = false;             // sliding-window attention (window ModelLayout::n_swa)
    double rope_base = 0;         // RoPE frequency base for this layer
    double rope_scale = 1;        // RoPE frequency scale (1 / linear scaling factor)
    bool rope_freqs = false;      // apply rope_freqs.weight (llama3 scaling; gemma4 full layers)
};

struct ModelLayout {
    std::string arch;           // general.architecture
    std::string name;           // general.name (if present)
    int n_layer = 0;            // layers in the forward pass (block_count minus MTP blocks)
    int n_block = 0;            // <arch>.block_count as stored (includes MTP blocks)
    int n_embd = 0;             // <arch>.embedding_length  (residual stream width)
    int n_ff = 0;               // <arch>.feed_forward_length (neurons per layer)
    int n_head = 0;             // <arch>.attention.head_count
    int n_head_kv = 0;          // <arch>.attention.head_count_kv (GQA)
    int head_dim = 0;           // <arch>.attention.key_length, else n_embd / n_head
    int n_ctx_train = 0;        // <arch>.context_length
    int n_vocab = 0;            // tokenizer.ggml.tokens length (or <arch>.vocab_size)
    double rope_freq_base = 0;  // <arch>.rope.freq_base
    int rope_dims = 0;          // <arch>.rope.dimension_count (qwen35: partial rotary), else head_dim
    double rms_eps = 0;         // <arch>.attention.layer_norm_rms_epsilon
    int rope_sections[4] = {0, 0, 0, 0};  // <arch>.rope.dimension_sections (qwen35 M-RoPE: 11, 11, 10, 0)

    // qwen35 Gated DeltaNet sizes (<arch>.ssm.*; 0 for other archs).
    int ssm_conv_kernel = 0;    // ssm.conv_kernel      (4)
    int ssm_head_k_dim = 0;     // ssm.state_size       (128) — key head dim = value head dim
    int ssm_n_k_heads = 0;      // ssm.group_count      (16)
    int ssm_n_v_heads = 0;      // ssm.time_step_rank   (32)
    int ssm_inner = 0;          // ssm.inner_size       (4096) = n_v_heads · head_v_dim

    // Gemma (0 / 1 for other archs).
    int n_swa = 0;              // <arch>.attention.sliding_window
    float embd_scale = 1;       // the token embedding is multiplied by this (√n_embd)
    float q_scale = 0;          // Q pre-scale before softmax(q·k) (0: none, 1/√head_dim inside the softmax instead)
    float softcap = 0;          // <arch>.final_logit_softcapping (0: none)
    bool gemma = false;         // gemma3 / gemma4 graph (GeGLU, post-norms, …)
    bool gemma4 = false;        // gemma4 specifics (V norm, layer output scale)
    bool has_post_norms() const { return gemma; }

    const TensorInfo* token_embd = nullptr;
    const TensorInfo* output_norm = nullptr;
    const TensorInfo* output = nullptr;   // null when tied to token_embd (Llama 3.2 1B/3B, Qwen3.5-4B)
    std::vector<LayerTensors> layers;     // n_layer entries
    std::vector<int> mtp_layers;          // block indices excluded as MTP head (qwen35)

    // The GGUF carries prism.hadamard.* metadata: the converter folded a random
    // Hadamard rotation into the weights. A rotation on the input of ffn_down mixes
    // neurons together, so neuron indices, lenses and byte footprints no longer mean
    // anything. read_neuron refuses such files.
    bool hadamard_folded = false;

    // Things that look wrong but don't stop browsing (shape mismatches, missing
    // tensors). Shown in the TUI status area rather than thrown.
    std::vector<std::string> warnings;

    static ModelLayout from(const GgufFile& f);

    bool has_ffn(int layer) const {
        return layer >= 0 && layer < n_layer && layers[layer].ffn_gate && layers[layer].ffn_up && layers[layer].ffn_down;
    }
    int n_linear_layers() const;   // qwen35: 3 of every 4
};

}  // namespace neuronaut
