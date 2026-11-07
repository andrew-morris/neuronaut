// forward.h — our own forward pass (Llama, Qwen3.5), built as a ggml graph.
//
// Why not just call llama.cpp? Because a debugger needs to *write* to the
// machine mid-run. Building the graph ourselves means every intermediate value
// is a node we can read back or splice an operation into. llama.cpp stays in
// the picture only as an oracle (oracle.h): same weights, same prompt, our
// logits must match theirs.
//
// The graph mirrors llama.cpp's llm_build_llama (src/models/llama.cpp @ b11700):
//
//   x = token_embd[tokens]
//   per layer:
//     h  = rms_norm(x) · attn_norm
//     q,k,v = Wq h, Wk h, Wv h            (GQA: n_head q heads, n_head_kv kv heads)
//     q,k = rope(q,k)                     (llama3 frequency factors from rope_freqs.weight)
//     k,v → f16                           (as llama.cpp's KV cache stores them)
//     x += Wo · softmax(q kᵀ/√d + causal) v
//     h  = rms_norm(x) · ffn_norm
//     a  = silu(Wgate h) ⊙ (Wup h)        ← the "neurons"  (n_ff wide)
//     a  = a ⊙ scale + add                ← interventions, only spliced in where set
//          (+ add_prompt at positions < prompt_len: prompt-only steering)
//     x += Wdown · a
//   logits = W_U · (rms_norm(x) · output_norm)
//
// Qwen3.5 (arch qwen35) mirrors llama.cpp's llama_model_qwen35::graph
// (src/models/qwen35.cpp + delta-net-base.cpp @ b11700). The skeleton and the MLP —
// so the neurons and the intervention splice — are identical; the attention half
// depends on the layer's kind:
//
//   full attention (every 4th layer), "gated attention":
//     [q | g] = Wq h                       (per head: head_dim of q, then head_dim of gate)
//     q = rms_norm(q)·q_norm,  k = rms_norm(Wk h)·k_norm      (per head)
//     q,k = M-RoPE (interleaved, sections 11/11/10 over the first 64 dims; text
//           tokens use position p on the t/y/x streams, 0 on the 4th)
//     o = softmax(q kᵀ/√d + causal) v ⊙ sigmoid(g);   attn_out = Wo · o
//
//   linear attention (Gated DeltaNet):
//     qkv = Wqkv h,  z = Wgate h
//     qkv = silu(causal_conv1d(qkv))       (kernel 4, state starts at zero)
//     q,k = l2_norm(q), l2_norm(k)         (per key head; 16 key heads tiled over 32 value heads)
//     β = sigmoid(Wβ h),  g = softplus(Wα h + dt_bias) · A      (A = −exp(A_log), per value head)
//     per token, per value head (state S starts at zero):
//       S ← exp(g)·S;  S ← S + k ⊗ β(v − Sᵀk);  o = Sᵀ q / √d
//     attn_out = Wout · (rms_norm(o)·ssm_norm ⊙ silu(z))
//
//   then, as for llama: x += attn_out; h = rms_norm(x)·post_attention_norm; MLP.
//
// Gemma 3 / Gemma 4 (archs gemma3, gemma4) mirror llama.cpp's llama_model_gemma3 /
// llama_model_gemma4 graphs (src/models/gemma3.cpp, gemma4.cpp @ b11700):
//
//   x = token_embd[tokens] · √n_embd
//   per layer (sliding-window or full attention; per-layer head size and RoPE):
//     h = rms_norm(x)·attn_norm
//     q = rms_norm(Wq h)·q_norm,  k = rms_norm(Wk h)·k_norm   (per head)
//     v = Wv h  (gemma4: rms_norm(v), unweighted; v = Wk h when the file has no Wv)
//     q,k = NEOX RoPE (gemma4 full layers: frequency factors for partial rotation)
//     q *= 1/√head_dim (gemma3)
//     x += rms_norm(Wo · softmax(q kᵀ + mask) v)·attn_post_norm      (mask: causal,
//                                     and within n_swa tokens on sliding layers)
//     a  = gelu(Wgate h') ⊙ (Wup h'),  h' = rms_norm(x)·ffn_norm     ← the neurons
//     x += rms_norm(Wdown · a)·ffn_post_norm
//     x *= layer_output_scale (gemma4)
//   logits = c·tanh(W_U · (rms_norm(x)·output_norm) / c)   (soft-cap c, if set)
//
// Execution (Q2, 2026-11-01). The graph runs through ggml_backend_sched — the same
// scheduler llama.cpp uses — on the CPU backend, or on a GPU backend with CPU
// fallback for any op the GPU lacks. Weights: zero-copy views of the mmap on CPU;
// uploaded once to device memory on GPU. Intermediates are allocated by the
// scheduler and reused, except the ones a caller asked to capture (marked outputs).
//
// One code path for prompts and generation: every forward is a *decode* into a
// DecodeCache (KV cache for softmax-attention layers; conv window + DeltaNet state
// for linear-attention layers). run(tokens) decodes all of them into a throwaway
// cache; generation keeps the cache and decodes one token at a time. The cache's
// KV length is padded exactly as llama.cpp's (see run()), which is what keeps our
// logits bit-identical to llama.cpp's on CPU, for prompts and for cached steps.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gguf_file.h"
#include "model_layout.h"

// ggml types, declared at global scope so the members below name these (not a
// neuronaut:: namesake an elaborated specifier would otherwise introduce).
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;

namespace neuronaut {

// Where an activation-patching override (Interventions::override_at) replaces a
// vector at one (layer, position):
//   Resid  x after layer l (n_embd) — what the next layer reads
//   Attn   layer l's attention output attn_out (n_embd), before it joins x
//   Mlp    layer l's neuron activations a (n_ff), *before* the per-neuron edits
//          below (so copying another run's captured acts — which are pre-edit —
//          into this one, under the same edits, writes exactly what that run wrote)
enum class Site { Resid, Attn, Mlp };
const char* site_name(Site s);        // "resid" | "attn" | "mlp"
bool parse_site(const std::string& s, Site& out);

// Per-layer edits to the neuron activations `a` (n_ff wide):
//
//   a' = a * scale + add + [pos < prompt_len] * add_prompt
//
// scale and add act at every token position; add_prompt only at absolute
// positions below prompt_len — "steer the question, not the answer" (edits.h).
// Defaults (1, 0, 0) mean "untouched"; layers absent from the map get no extra
// graph nodes at all, so an empty Interventions produces exactly the reference
// graph. A batch with no position below prompt_len (every generated token) gets
// no add_prompt nodes either.
struct Interventions {
    struct Layer {
        std::vector<float> scale;        // n_ff, default 1
        std::vector<float> add;          // n_ff, default 0
        std::vector<float> add_prompt;   // n_ff, default 0; only at positions < prompt_len
    };
    std::map<int, Layer> layers;
    int prompt_len = 0;   // positions [0, prompt_len) get add_prompt

    // Activation patching: replace the whole vector at `site` of `layer` at absolute
    // position `pos` with `v` (Site docs above). Positions outside a batch are
    // ignored by it, so a cached decode carries the override only where it applies.
    struct Override {
        Site site;
        int layer, pos;
        std::vector<float> v;
        bool operator==(const Override&) const = default;
    };
    std::vector<Override> overrides;
    void override_at(const ModelLayout& m, Site site, int layer, int pos, std::vector<float> v);

    // a[n] *= k  (k = 0 is ablation)
    void scale(const ModelLayout& m, int layer, int neuron, float k);
    // a[n] += v
    void add(const ModelLayout& m, int layer, int neuron, float v);
    // a[n] += v at positions < prompt_len only
    void add_prompt(const ModelLayout& m, int layer, int neuron, float v);
    bool has_prompt_scope() const;

    // Rank-one weight edits (ROME, rome.h): layer l's down projection becomes
    // W_down + u·wᵀ, so the MLP writes W_down·a + u·(w·a) — u is n_embd wide, w
    // n_ff wide. Applied to `a` after the per-neuron edits above, like the weights
    // they stand for; several on one layer sum.
    struct RankOne {
        int layer = -1;
        std::vector<float> u, w;
        bool operator==(const RankOne&) const = default;
    };
    std::vector<RankOne> rank_one;
    void add_rank_one(const ModelLayout& m, int layer, std::vector<float> u, std::vector<float> w);

    bool empty() const { return layers.empty() && overrides.empty() && rank_one.empty(); }
    bool operator==(const Interventions& o) const;

private:
    Layer& at(const ModelLayout& m, int layer);
};

struct RunOptions {
    std::vector<int> capture_layers;   // neuron activations a for these layers
    bool capture_all_layers = false;   // ... or for every layer
    bool all_logits = false;           // logits for every position (of the window), not just the last
    // Capture the pieces the residual stream is summed from — the token
    // embedding, every layer's attention output, and the final residual — so a
    // logit can be decomposed exactly (blame.h).
    bool capture_resid = false;
    // Residual stream after every layer (for the per-layer logit lens).
    bool capture_layer_resid = false;

    // Capture window: only positions [cap_lo, cap_hi) of this batch are copied out
    // for acts / embed / attn_out / final_resid / layer_resid / all_logits.
    // cap_hi < 0 means "to the end" (the default: everything). A narrow window is
    // what keeps a 4k-token thinking trace on the 9B from holding several GB of
    // captures per rerun: the full-width tensors stay ordinary intermediates the
    // allocator reuses, and only a contiguous copy of the window is kept.
    int cap_lo = 0, cap_hi = -1;
    // Single neurons captured at *every* position regardless of the window (the
    // TOKENS pane's watches): a strided view of a, a few bytes per token. Each is
    // two graph nodes, so this is for a handful (at most 256); to read many
    // neurons, capture their layers instead.
    std::vector<std::pair<int, int>> watch;
    // Per-head attribution (blame.h): the input of every layer's attention output
    // projection — `merged` before Wo (llama), the gated o before Wo (qwen35 full
    // attention), the gated-normed o before ssm_out (qwen35 linear attention). Its
    // width is heads × head width, so Wo·x splits exactly into one term per head.
    bool capture_heads = false;
    // Softmax attention weights of one softmax-attention layer for one query
    // position of this batch: [n_head][n_kv] (keys past the query are 0). Off when
    // attn_probs_layer < 0, or the layer is linear attention (it has no pattern).
    int attn_probs_layer = -1, attn_probs_pos = -1;
    // Stop after this layer (≥ 0): for callers that only read captures of layers up
    // to it (rome's covariance pass). The logits are then not the model's — they
    // read the residual stream at that depth.
    int last_layer = -1;
};

struct ForwardResult {
    int n_tokens = 0;
    int n_vocab = 0;
    // Positions [cap_lo, cap_lo + cap_n) of this batch are in the captures below.
    int cap_lo = 0, cap_n = 0;
    // Logits rows cover positions [logit_lo, logit_lo + logit_n): the last position
    // only, unless all_logits was requested (then the capture window).
    int logit_lo = 0, logit_n = 0;
    std::vector<float> logits;
    // Captured neuron activations a (before interventions), per requested
    // layer: cap_n × n_ff, position-major.
    std::map<int, std::vector<float>> acts;
    // With capture_resid (all cap_n × n_embd, position-major):
    std::vector<float> embed;                      // x0 = token_embd[tokens]
    std::map<int, std::vector<float>> attn_out;    // Wo·attention, added to x in layer l
    std::vector<float> final_resid;                // x after the last layer, before output_norm
    std::map<int, std::vector<float>> layer_resid; // x after layer l (capture_layer_resid)
    // RunOptions::watch: (layer, neuron) → a at every position (n_tokens).
    std::map<std::pair<int, int>, std::vector<float>> watched;
    // With capture_resid on a model with post-norms (Gemma): what attention and the
    // MLP produced *before* their post-norm (attn_out above is after it: what was
    // added to x). Blame freezes those norms' 1/rms from these (blame.h).
    std::map<int, std::vector<float>> attn_raw, mlp_raw;
    // RunOptions::capture_heads: layer → cap_n × (heads · head width), position-major.
    std::map<int, std::vector<float>> head_in;
    // RunOptions::attn_probs_*: n_head rows of attn_n_kv weights (row h = head h).
    std::vector<float> attn_probs;
    int attn_n_kv = 0, attn_n_head = 0;

    // Is position `pos` inside the capture window?
    bool captured(int pos) const { return pos >= cap_lo && pos < cap_lo + cap_n; }
    // Row `pos` of a capture of this result (width = size / cap_n), or nullptr when
    // the capture is empty or `pos` is outside the window.
    const float* row(const std::vector<float>& v, int pos) const {
        if (v.empty() || !captured(pos)) return nullptr;
        const size_t w = v.size() / static_cast<size_t>(cap_n);
        return v.data() + static_cast<size_t>(pos - cap_lo) * w;
    }
    template <class K>
    const float* row(const std::map<K, std::vector<float>>& m, const K& k, int pos) const {
        auto it = m.find(k);
        return it == m.end() ? nullptr : row(it->second, pos);
    }
    const float* act(int layer, int pos) const { return row(acts, layer, pos); }
    const float* resid_after(int layer, int pos) const { return row(layer_resid, layer, pos); }
    const float* attn_at(int layer, int pos) const { return row(attn_out, layer, pos); }
    const float* embed_at(int pos) const { return row(embed, pos); }
    const float* final_at(int pos) const { return row(final_resid, pos); }
    const float* heads_at(int layer, int pos) const { return row(head_in, layer, pos); }
    // Logits for `pos`, or nullptr if that row wasn't computed.
    bool has_logits(int pos) const { return pos >= logit_lo && pos < logit_lo + logit_n; }
    const float* logits_at(int pos) const {
        return has_logits(pos) ? logits.data() + static_cast<size_t>(pos - logit_lo) * static_cast<size_t>(n_vocab) : nullptr;
    }
};

// Which device runs the graph. cpu is the default and the reference (bit-identical
// to llama.cpp's CPU backend). gpu = the first GPU device ggml was built with (CUDA
// on an RTX 5090); selected by Forward's constructor or NEURONAUT_DEVICE=gpu.
enum class Device { cpu, gpu };
Device device_from_env();   // NEURONAUT_DEVICE: "gpu"/"cuda" → gpu, anything else → cpu

// Per-sequence decoding state, created by Forward::new_cache and owned by the caller.
// Valid only for the Interventions it was started with: decoding with different ones
// throws (cached K/V and recurrent states were computed under the old patches).
class DecodeCache {
public:
    ~DecodeCache();
    DecodeCache(const DecodeCache&) = delete;
    DecodeCache& operator=(const DecodeCache&) = delete;
    int n_past() const { return n_past_; }     // tokens already decoded
    int n_cells() const { return n_cells_; }   // capacity (multiple of 256)

private:
    friend class Forward;
    DecodeCache() = default;
    struct Layer {
        ggml_tensor* k = nullptr;      // [head_dim, n_head_kv, n_cells] f16 (attention layers)
        ggml_tensor* v = nullptr;      // [head_dim, n_head_kv, n_cells] f16
        ggml_tensor* conv = nullptr;   // [conv_kernel − 1, conv_channels, 1] f32 (linear layers)
        ggml_tensor* ssm = nullptr;    // [dv, dv, n_v_heads, 1] f32 (DeltaNet state)
    };
    ggml_context* ctx_ = nullptr;
    struct ggml_backend_buffer* buf_ = nullptr;
    std::vector<Layer> layers_;
    int n_cells_ = 0;
    int n_past_ = 0;
    Interventions iv_;   // the patches this sequence was decoded under
};

class Forward {
public:
    // CPU: wraps the mmap'd weights (no copy). GPU: uploads them once (the 9B in
    // BF16 is ~18 GB). `f` must outlive this. n_threads 0: $NEURONAUT_THREADS, else
    // every core.
    Forward(const GgufFile& f, const ModelLayout& m, int n_threads = 0, Device device = device_from_env());
    ~Forward();
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    // The whole sequence from scratch (a throwaway cache). Captures and logits are
    // for these tokens.
    ForwardResult run(const std::vector<int32_t>& tokens, const Interventions& iv = {}, const RunOptions& opt = {}) const;

    // Incremental decoding. new_cache(n) holds up to n tokens (rounded up to 256).
    // decode() appends `tokens` to the sequence in `c` and returns results for those
    // tokens only (logits: last of them, unless all_logits). Throws if `iv` differs
    // from the patches `c` was started with, or if the cache would overflow.
    std::unique_ptr<DecodeCache> new_cache(int n_tokens) const;
    ForwardResult decode(DecodeCache& c, const std::vector<int32_t>& tokens, const Interventions& iv = {},
                         const RunOptions& opt = {}) const;

    // Only layers [start_layer, n_layer), starting from `resid_in` — the residual
    // stream *entering* layer start_layer, n_tokens × n_embd position-major: the
    // embedding when start_layer == 0, else layer_resid[start_layer − 1] of a run of
    // the same tokens. Layers below start_layer (and their interventions) are not
    // computed at all, so an edit at layer L costs (n_layer − L) / n_layer of a full
    // pass. Bit-identical to run() when resid_in came from a run() of the same tokens
    // under the same interventions below start_layer (tests/test_forward.cpp).
    // start_layer == n_layer is allowed: just the output head. Captures of layers
    // below start_layer are empty; `embed` is resid_in when start_layer == 0.
    ForwardResult run_from(int start_layer, const std::vector<float>& resid_in, int n_tokens, const Interventions& iv = {},
                           const RunOptions& opt = {}) const;

    // ROME's value search (rome.h): the loss of a teacher-forced batch, and its
    // gradient, with respect to one vector δ (n_embd) added to the residual stream
    // right after layer `layer` at the positions delta_mask marks — the same as
    // adding δ to that layer's MLP output there. Runs layers layer+1 … n−1 only, from
    // `resid` (x after `layer`, without δ), with gradients by ggml's autograd. Several
    // sequences are packed into one batch: `seq` keeps attention inside each, `pos`
    // restarts RoPE for each. CPU, plain llama-style layers only (throws otherwise).
    struct DeltaBatch {
        std::vector<float> resid;         // n_tokens × n_embd, position-major
        std::vector<int32_t> pos, seq;    // per token
        std::vector<float> delta_mask;    // per token: 1 where δ is added
        std::vector<int32_t> nll_rows;    // tokens whose next-token prediction is scored …
        std::vector<int32_t> nll_ids;     // … against these targets (mean NLL)
        std::vector<int32_t> kl_rows;     // tokens whose next-token distribution must hold …
        std::vector<float> kl_ref;        // … at these reference probabilities (rows × n_vocab)
        float kl_weight = 0;
        int n_tokens() const { return static_cast<int>(pos.size()); }
    };
    struct DeltaLoss {
        float nll = 0, kl = 0, total = 0;   // kl: KL(ref ‖ edited), mean over kl rows
        std::vector<float> grad;            // d total / d δ
    };
    DeltaLoss delta_loss(int layer, const DeltaBatch& b, const std::vector<float>& delta) const;

    int n_vocab() const;
    Device device() const { return device_; }
    std::string device_name() const;

private:
    // Every tensor a layer might use; the ones its kind doesn't use stay null.
    struct LayerW {
        LayerKind kind = LayerKind::attention;
        ggml_tensor *attn_norm = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
        ggml_tensor *q_norm = nullptr, *k_norm = nullptr;                       // qwen35 full attention
        ggml_tensor *wqkv = nullptr, *wgate = nullptr, *conv1d = nullptr, *dt = nullptr, *a = nullptr;
        ggml_tensor *beta = nullptr, *alpha = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;  // linear attention
        ggml_tensor *ffn_norm = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
        ggml_tensor *attn_post_norm = nullptr, *ffn_post_norm = nullptr, *out_scale = nullptr;   // gemma
        const LayerTensors* L = nullptr;   // shapes: head_dim, n_head(_kv), RoPE
    };
    ggml_tensor* wrap(const TensorInfo* t);
    // decode() and run_from(): `tokens` (embedded) or `resid_in` (n_tokens × n_embd,
    // entering start_layer) is the input.
    ForwardResult decode_impl(DecodeCache& c, const std::vector<int32_t>& tokens, int n_tokens, int start_layer,
                              const std::vector<float>* resid_in, const Interventions& iv, const RunOptions& opt) const;

    // The attention half of one layer: h (normed input, [n_embd, T]) → the
    // [n_embd, T] it adds to the residual stream.
    struct GraphInputs {
        ggml_context* ctx;
        struct ggml_cgraph* gf;  // cache writes are expanded into it before the reads
        ggml_tensor* pos;        // [T] (llama) or [4T] (qwen35 M-RoPE: t, y, x, z streams)
        ggml_tensor* kq_mask;    // [n_kv, T] causal from n_past; cells past the sequence masked −inf
                                 // (the sliding-window mask on a gemma sliding layer)
        int64_t T;               // new tokens in this decode
        int64_t n_past;          // tokens already in the cache
        int64_t n_kv;            // cache cells attended over: padded like llama.cpp's KV cache
        const DecodeCache::Layer* cache;   // this layer's cache tensors
        ggml_tensor** head_in = nullptr;   // out: the output projection's input (capture_heads)
        ggml_tensor** probs = nullptr;     // out: softmax weights [n_kv, T, n_head] (attn_probs_layer)
    };
    ggml_tensor* attn_llama(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const;
    ggml_tensor* attn_qwen35_full(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const;
    ggml_tensor* attn_qwen35_linear(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const;
    ggml_tensor* attn_gemma(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const;
    // Writes the new k/v ([hd, n_head_kv, T], f32) into the f16 cache at n_past,
    // then softmax(q kᵀ·scale + mask) v over the first n_kv cells (GQA broadcast).
    // Returns [hd·n_head, T].
    ggml_tensor* softmax_attention(ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, const GraphInputs& in,
                                   float scale) const;

    const ModelLayout& m_;
    bool qwen35_ = false;
    bool gemma_ = false;
    int n_threads_;
    Device device_ = Device::cpu;
    struct ggml_backend* backend_ = nullptr;       // where the graph runs (CPU or GPU)
    struct ggml_backend* backend_cpu_ = nullptr;   // CPU fallback when backend_ is a GPU (else == backend_)
    struct ggml_backend_buffer* wbuf_ = nullptr;   // weights: mmap view (CPU) or device copy (GPU)
    struct ggml_backend_sched* sched_ = nullptr;
    // ggml's scheduler and our compute are not re-entrant; the dose worker and the
    // UI never overlap (the UI refuses work while busy), but serialize anyway.
    mutable std::mutex mu_;
    std::vector<std::pair<ggml_tensor*, const TensorInfo*>> wrapped_;   // weight header → its bytes in the file
    ggml_context* wctx_ = nullptr;   // weight tensor headers
    ggml_tensor* tok_embd_ = nullptr;
    ggml_tensor* output_norm_ = nullptr;
    ggml_tensor* output_ = nullptr;      // == tok_embd_ when tied
    ggml_tensor* rope_freqs_ = nullptr;  // llama3 rope scaling factors (may be null)
    std::vector<LayerW> layers_;
    // f32 copies of the weights delta_loss differentiates through, made on first
    // use: ggml's backward for an f16/quantized matmul re-dequantizes the whole
    // matrix once per token, its f32 path doesn't.
    struct GradWeights;
    mutable std::unique_ptr<GradWeights> gradw_;
};

// Softmax probability of one token (no sorting).
float prob_of(const float* logits, int n_vocab, int id);

// Softmax of a logit row, and the k most probable token ids.
std::vector<std::pair<int, float>> top_probs(const float* logits, int n_vocab, int k);

}  // namespace neuronaut
