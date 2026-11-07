// forward.cpp — see forward.h.
#include "forward.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace neuronaut {

namespace {

// Max graph nodes. A Qwen3.5 layer is ~70 nodes, so 32 layers + captures fit easily.
constexpr size_t kGraphSize = 16384;

int64_t round_up_256(int64_t n) { return (n + 255) / 256 * 256; }

}  // namespace

Device device_from_env() {
    const char* e = std::getenv("NEURONAUT_DEVICE");
    if (!e) return Device::cpu;
    std::string s(e);
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return (s == "gpu" || s == "cuda") ? Device::gpu : Device::cpu;
}

// ---------------------------------------------------------------------------
// Interventions
// ---------------------------------------------------------------------------

Interventions::Layer& Interventions::at(const ModelLayout& m, int layer) {
    if (!m.has_ffn(layer)) throw std::out_of_range("intervention: layer " + std::to_string(layer) + " has no FFN");
    auto it = layers.find(layer);
    if (it == layers.end()) {
        Layer l;
        l.scale.assign(static_cast<size_t>(m.n_ff), 1.0f);
        l.add.assign(static_cast<size_t>(m.n_ff), 0.0f);
        l.add_prompt.assign(static_cast<size_t>(m.n_ff), 0.0f);
        it = layers.emplace(layer, std::move(l)).first;
    }
    return it->second;
}

void Interventions::scale(const ModelLayout& m, int layer, int neuron, float k) {
    if (neuron < 0 || neuron >= m.n_ff) throw std::out_of_range("intervention: neuron out of range");
    at(m, layer).scale[static_cast<size_t>(neuron)] *= k;
}

void Interventions::add(const ModelLayout& m, int layer, int neuron, float v) {
    if (neuron < 0 || neuron >= m.n_ff) throw std::out_of_range("intervention: neuron out of range");
    at(m, layer).add[static_cast<size_t>(neuron)] += v;
}

void Interventions::add_prompt(const ModelLayout& m, int layer, int neuron, float v) {
    if (neuron < 0 || neuron >= m.n_ff) throw std::out_of_range("intervention: neuron out of range");
    at(m, layer).add_prompt[static_cast<size_t>(neuron)] += v;
}

const char* site_name(Site s) {
    switch (s) {
        case Site::Resid: return "resid";
        case Site::Attn: return "attn";
        case Site::Mlp: return "mlp";
    }
    return "?";
}

bool parse_site(const std::string& s, Site& out) {
    if (s == "resid") out = Site::Resid;
    else if (s == "attn") out = Site::Attn;
    else if (s == "mlp") out = Site::Mlp;
    else return false;
    return true;
}

void Interventions::override_at(const ModelLayout& m, Site site, int layer, int pos, std::vector<float> v) {
    if (layer < 0 || layer >= m.n_layer) throw std::out_of_range("override: layer " + std::to_string(layer) + " out of range");
    if (pos < 0) throw std::out_of_range("override: negative position");
    const size_t want = static_cast<size_t>(site == Site::Mlp ? m.n_ff : m.n_embd);
    if (v.size() != want) throw std::invalid_argument(std::string("override: ") + site_name(site) + " wants " + std::to_string(want) + " values");
    // A later override of the same slot replaces the earlier one.
    std::erase_if(overrides, [&](const Override& o) { return o.site == site && o.layer == layer && o.pos == pos; });
    overrides.push_back({site, layer, pos, std::move(v)});
}

bool Interventions::has_prompt_scope() const {
    for (const auto& [l, L] : layers)
        for (float v : L.add_prompt)
            if (v != 0.0f) return true;
    return false;
}

void Interventions::add_rank_one(const ModelLayout& m, int layer, std::vector<float> u, std::vector<float> w) {
    if (layer < 0 || layer >= m.n_layer || !m.has_ffn(layer)) throw std::out_of_range("rank-one: no FFN layer " + std::to_string(layer));
    if (u.size() != static_cast<size_t>(m.n_embd) || w.size() != static_cast<size_t>(m.n_ff))
        throw std::invalid_argument("rank-one: wants u of n_embd and w of n_ff values");
    rank_one.push_back({layer, std::move(u), std::move(w)});
}

bool Interventions::operator==(const Interventions& o) const {
    if (layers.size() != o.layers.size() || prompt_len != o.prompt_len || overrides != o.overrides || rank_one != o.rank_one) return false;
    for (const auto& [l, L] : layers) {
        auto it = o.layers.find(l);
        if (it == o.layers.end() || it->second.scale != L.scale || it->second.add != L.add || it->second.add_prompt != L.add_prompt) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Decode cache
// ---------------------------------------------------------------------------

DecodeCache::~DecodeCache() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

std::unique_ptr<DecodeCache> Forward::new_cache(int n_tokens) const {
    std::unique_ptr<DecodeCache> c(new DecodeCache());
    c->n_cells_ = static_cast<int>(std::max<int64_t>(256, round_up_256(std::max(1, n_tokens))));

    ggml_init_params ip{};
    ip.mem_size = static_cast<size_t>(4 * m_.n_layer + 8) * ggml_tensor_overhead();
    ip.no_alloc = true;
    c->ctx_ = ggml_init(ip);
    if (!c->ctx_) throw std::runtime_error("ggml_init failed for the decode cache");

    const int64_t conv_ch = m_.ssm_inner + 2 * m_.ssm_n_k_heads * m_.ssm_head_k_dim;
    const int64_t dv = m_.ssm_n_v_heads > 0 ? m_.ssm_inner / m_.ssm_n_v_heads : 0;
    c->layers_.resize(static_cast<size_t>(m_.n_layer));
    for (int il = 0; il < m_.n_layer; ++il) {
        auto& L = c->layers_[static_cast<size_t>(il)];
        if (layers_[static_cast<size_t>(il)].kind == LayerKind::linear_attention) {
            L.conv = ggml_new_tensor_3d(c->ctx_, GGML_TYPE_F32, m_.ssm_conv_kernel - 1, conv_ch, 1);
            L.ssm = ggml_new_tensor_4d(c->ctx_, GGML_TYPE_F32, dv, dv, m_.ssm_n_v_heads, 1);
        } else {
            // f16, as llama.cpp's KV cache stores K and V (part of being bit-identical).
            const auto& S = m_.layers[static_cast<size_t>(il)];   // per-layer shape (gemma4: sliding ≠ full)
            L.k = ggml_new_tensor_3d(c->ctx_, GGML_TYPE_F16, S.head_dim, S.n_head_kv, c->n_cells_);
            L.v = ggml_new_tensor_3d(c->ctx_, GGML_TYPE_F16, S.head_dim, S.n_head_kv, c->n_cells_);
        }
    }
    c->buf_ = ggml_backend_alloc_ctx_tensors(c->ctx_, backend_);
    if (!c->buf_) throw std::runtime_error("could not allocate the decode cache on " + device_name());
    // Zero everything: unused KV cells must be 0 (llama.cpp's are; masked cells still
    // enter the f16 dot products as 0 × 0), and the recurrent states start at zero.
    ggml_backend_buffer_clear(c->buf_, 0);
    return c;
}

// ---------------------------------------------------------------------------
// Weights and backends
// ---------------------------------------------------------------------------

ggml_tensor* Forward::wrap(const TensorInfo* t) {
    if (!t) return nullptr;
    ggml_tensor* x = ggml_new_tensor(wctx_, t->type, t->n_dims, t->ne);
    ggml_set_name(x, t->name.c_str());
    wrapped_.emplace_back(x, t);   // bound to a buffer once all weights are declared
    return x;
}

Forward::Forward(const GgufFile& f, const ModelLayout& m, int n_threads, Device device) : m_(m), device_(device) {
    if (m.arch != "llama" && m.arch != "qwen35" && m.arch != "gemma3" && m.arch != "gemma4") {
        throw std::runtime_error("forward pass only implemented for archs llama, qwen35, gemma3 and gemma4, not '" + m.arch + "'");
    }
    gemma_ = m.gemma;
    if (m.hadamard_folded) throw std::runtime_error("prism.hadamard-rotated GGUFs are not supported");
    qwen35_ = m.arch == "qwen35";
    // Threads: the caller's, else $NEURONAUT_THREADS, else every core. ggml's workers
    // busy-wait, so on a box shared with other CPU-heavy jobs fewer threads than
    // cores is much faster than all of them.
    if (n_threads <= 0) {
        if (const char* e = std::getenv("NEURONAUT_THREADS")) n_threads = std::atoi(e);
    }
    n_threads_ = n_threads > 0 ? n_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    ggml_init_params ip{};
    ip.mem_size = (f.tensors().size() + 8) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;   // headers only; storage is bound below
    wctx_ = ggml_init(ip);

    tok_embd_ = wrap(m.token_embd);
    output_norm_ = wrap(m.output_norm);
    output_ = m.output ? wrap(m.output) : tok_embd_;
    rope_freqs_ = wrap(f.find_tensor("rope_freqs.weight"));
    if (!tok_embd_ || !output_norm_) throw std::runtime_error("model is missing token_embd or output_norm");

    for (int l = 0; l < m.n_layer; ++l) {
        const auto& L = m.layers[static_cast<size_t>(l)];
        const std::string where = "layer " + std::to_string(l) + " (" + layer_kind_name(L.kind) + ")";
        if (!L.attn_norm || !L.ffn_norm || !m.has_ffn(l)) throw std::runtime_error(where + ": missing norm or MLP tensors");
        LayerW W;
        W.kind = L.kind;
        W.L = &L;
        W.attn_norm = wrap(L.attn_norm);
        W.ffn_norm = wrap(L.ffn_norm);
        W.gate = wrap(L.ffn_gate);
        W.up = wrap(L.ffn_up);
        W.down = wrap(L.ffn_down);
        if (L.kind == LayerKind::linear_attention) {
            if (!L.attn_qkv || !L.attn_gate || !L.ssm_conv1d || !L.ssm_dt || !L.ssm_a || !L.ssm_beta || !L.ssm_alpha ||
                !L.ssm_norm || !L.ssm_out) {
                throw std::runtime_error(where + ": missing Gated DeltaNet tensors");
            }
            W.wqkv = wrap(L.attn_qkv);
            W.wgate = wrap(L.attn_gate);
            W.conv1d = wrap(L.ssm_conv1d);
            W.dt = wrap(L.ssm_dt);
            W.a = wrap(L.ssm_a);
            W.beta = wrap(L.ssm_beta);
            W.alpha = wrap(L.ssm_alpha);
            W.ssm_norm = wrap(L.ssm_norm);
            W.ssm_out = wrap(L.ssm_out);
        } else {
            // gemma4 attention_k_eq_v: no attn_v, V is the K projection.
            if (!L.attn_q || !L.attn_k || (!L.attn_v && !m.gemma4) || !L.attn_output) throw std::runtime_error(where + ": missing Q/K/V/O");
            if ((qwen35_ || gemma_) && (!L.attn_q_norm || !L.attn_k_norm)) throw std::runtime_error(where + ": missing Q/K norms");
            if (gemma_ && (!L.attn_post_norm || !L.ffn_post_norm)) throw std::runtime_error(where + ": missing post-norms");
            W.attn_post_norm = wrap(L.attn_post_norm);
            W.ffn_post_norm = wrap(L.ffn_post_norm);
            W.out_scale = wrap(L.out_scale);
            W.wq = wrap(L.attn_q);
            W.wk = wrap(L.attn_k);
            W.wv = wrap(L.attn_v);
            W.wo = wrap(L.attn_output);
            W.q_norm = wrap(L.attn_q_norm);
            W.k_norm = wrap(L.attn_k_norm);
        }
        layers_.push_back(W);
    }
    if (qwen35_ && (m.ssm_conv_kernel <= 1 || m.ssm_head_k_dim <= 0 || m.ssm_n_k_heads <= 0 || m.ssm_n_v_heads <= 0 ||
                    m.ssm_inner % std::max(1, m.ssm_n_v_heads) != 0)) {
        throw std::runtime_error("qwen35: bad ssm.* hyperparameters");
    }

    // --- backends ----------------------------------------------------------------
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) throw std::runtime_error("could not initialise the ggml CPU backend");
    ggml_backend_cpu_set_n_threads(backend_cpu_, n_threads_);
    if (device_ == Device::gpu) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) {
            ggml_backend_free(backend_cpu_);
            throw std::runtime_error("NEURONAUT_DEVICE=gpu but ggml has no GPU device (build with -DGGML_CUDA=ON)");
        }
        backend_ = ggml_backend_dev_init(dev, nullptr);
        if (!backend_) throw std::runtime_error(std::string("could not initialise GPU backend ") + ggml_backend_dev_name(dev));
    } else {
        backend_ = backend_cpu_;
    }

    // --- weights -----------------------------------------------------------------
    if (device_ == Device::cpu) {
        // Zero-copy: one CPU buffer over the whole read-only mapping; each tensor's
        // data is its address inside it. ggml only ever reads weights, so the
        // const_cast never results in a write (and PROT_READ would fault if it did).
        wbuf_ = ggml_backend_cpu_buffer_from_ptr(const_cast<uint8_t*>(f.map_base()), f.file_size());
        for (auto& [x, t] : wrapped_) {
            if (ggml_backend_tensor_alloc(wbuf_, x, const_cast<uint8_t*>(t->data)) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("could not bind weight " + t->name);
            }
        }
    } else {
        // One upload into device memory (pages of the mmap are touched once).
        wbuf_ = ggml_backend_alloc_ctx_tensors(wctx_, backend_);
        if (!wbuf_) throw std::runtime_error("could not allocate weights on " + device_name() + " (out of memory?)");
        for (auto& [x, t] : wrapped_) ggml_backend_tensor_set(x, t->data, 0, t->n_bytes);
    }
    ggml_backend_buffer_set_usage(wbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // --- scheduler -----------------------------------------------------------------
    // The GPU first (it owns the weights, so ops go there), the CPU as fallback for
    // any op the GPU backend doesn't implement.
    ggml_backend_t bk[2] = {backend_, backend_cpu_};
    const int n_bk = backend_ == backend_cpu_ ? 1 : 2;
    sched_ = ggml_backend_sched_new(bk, nullptr, n_bk, kGraphSize, false, false);
    if (!sched_) throw std::runtime_error("could not create the ggml scheduler");
}

struct Forward::GradWeights {
    ggml_context* ctx = nullptr;
    std::vector<LayerW> layers;   // only the matrices are replaced; norms stay views
    ggml_tensor* output = nullptr;
    ~GradWeights() {
        if (ctx) ggml_free(ctx);
    }
};

Forward::~Forward() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (wbuf_) ggml_backend_buffer_free(wbuf_);
    if (backend_ && backend_ != backend_cpu_) ggml_backend_free(backend_);
    if (backend_cpu_) ggml_backend_free(backend_cpu_);
    if (wctx_) ggml_free(wctx_);
}

int Forward::n_vocab() const { return static_cast<int>(output_->ne[1]); }

std::string Forward::device_name() const { return backend_ ? ggml_backend_name(backend_) : "?"; }

// ---------------------------------------------------------------------------
// Attention halves, one per layer kind
// ---------------------------------------------------------------------------

ggml_tensor* Forward::softmax_attention(ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, const GraphInputs& in,
                                        float scale) const {
    ggml_context* ctx = in.ctx;
    const int64_t hd = q->ne[0], n_head = q->ne[1], n_kvh = k->ne[1], T = in.T;
    ggml_tensor *kc = k, *vc = v;   // no cache (delta_loss): attend over this batch's own keys
    if (in.cache) {
        ggml_tensor* K = in.cache->k;
        ggml_tensor* V = in.cache->v;

        // Store the new keys/values (f32 → f16, as llama.cpp's KV cache does) in cells
        // [n_past, n_past + T). Expanded into the graph now, so they run before the
        // reads below, which are views of the same cache tensors.
        ggml_tensor* kdst = ggml_view_3d(ctx, K, hd, n_kvh, T, K->nb[1], K->nb[2], static_cast<size_t>(in.n_past) * K->nb[2]);
        ggml_tensor* vdst = ggml_view_3d(ctx, V, hd, n_kvh, T, V->nb[1], V->nb[2], static_cast<size_t>(in.n_past) * V->nb[2]);
        ggml_build_forward_expand(in.gf, ggml_cpy(ctx, k, kdst));
        ggml_build_forward_expand(in.gf, ggml_cpy(ctx, v, vdst));

        // Attend over the first n_kv cells. n_kv is padded like llama.cpp's KV cache
        // (see decode()); cells past the sequence are zero and masked to −inf. The
        // padding is what makes the f16 V·softmax dot products run ggml's SIMD loop
        // exactly as llama.cpp's do (a short row falls into the scalar tail instead),
        // which is what keeps us bit-identical.
        kc = ggml_view_3d(ctx, K, hd, n_kvh, in.n_kv, K->nb[1], K->nb[2], 0);
        vc = ggml_view_3d(ctx, V, hd, n_kvh, in.n_kv, V->nb[1], V->nb[2], 0);
    }

    // scores[k, q, head] = k·q ; mul_mat broadcasts the n_head_kv K heads over
    // the n_head Q heads (head h uses kv head h / (n_head / n_head_kv)) — GQA.
    ggml_tensor* qp = ggml_permute(ctx, q, 0, 2, 1, 3);                    // [hd, T, n_head]
    ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, kc, 0, 2, 1, 3));    // [hd, n_kv, n_kvh]
    ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);                           // [n_kv, T_q, n_head]
    kq = ggml_soft_max_ext(ctx, kq, in.kq_mask, scale, 0.0f);
    if (in.probs) *in.probs = kq;
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vc, 1, 2, 0, 3));    // [n_kv, hd, n_kvh]
    ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);                          // [hd, T_q, n_head]
    return ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), hd * n_head, T);
}

ggml_tensor* Forward::attn_llama(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const {
    ggml_context* ctx = in.ctx;
    const int64_t T = in.T, n_head = m_.n_head, n_head_kv = m_.n_head_kv;
    const int64_t head_dim = m_.head_dim;
    ggml_tensor* q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wq, h), head_dim, n_head, T);
    ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wk, h), head_dim, n_head_kv, T);
    ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wv, h), head_dim, n_head_kv, T);

    // RoPE, "normal" (non-NeoX) mode as llama.cpp uses for arch llama; the
    // converter already permuted Wq/Wk rows to match this layout.
    // ext_factor 0 (no YaRN), attn_factor 1, beta_fast/slow 32/1 = llama.cpp defaults.
    q = ggml_rope_ext(ctx, q, in.pos, rope_freqs_, static_cast<int>(head_dim), 0, m_.n_ctx_train,
                      static_cast<float>(m_.rope_freq_base), 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, in.pos, rope_freqs_, static_cast<int>(head_dim), 0, m_.n_ctx_train,
                      static_cast<float>(m_.rope_freq_base), 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    ggml_tensor* merged = softmax_attention(q, k, v, in, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    if (in.head_in) *in.head_in = merged;     // [head_dim · n_head, T]: head h is rows [h·hd, (h+1)·hd)
    return ggml_mul_mat(ctx, W.wo, merged);   // [n_embd, T]
}

ggml_tensor* Forward::attn_qwen35_full(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const {
    ggml_context* ctx = in.ctx;
    const int64_t T = in.T, n_head = m_.n_head, n_head_kv = m_.n_head_kv, hd = m_.head_dim;
    const float eps = static_cast<float>(m_.rms_eps);

    // Wq yields, per head, head_dim of query followed by head_dim of output gate.
    ggml_tensor* qg = ggml_mul_mat(ctx, W.wq, h);                       // [2·hd·n_head, T]
    const size_t es = ggml_element_size(qg);
    ggml_tensor* q = ggml_view_3d(ctx, qg, hd, n_head, T, es * hd * 2, es * hd * 2 * n_head, 0);
    ggml_tensor* g = ggml_view_3d(ctx, qg, hd, n_head, T, es * hd * 2, es * hd * 2 * n_head, es * hd);
    g = ggml_cont_2d(ctx, g, hd * n_head, T);

    // Per-head RMSNorm on Q and K (gains have the converter's +1 baked in).
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), W.q_norm);
    ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wk, h), hd, n_head_kv, T);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), W.k_norm);
    ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wv, h), hd, n_head_kv, T);

    // Interleaved M-RoPE over the first rope_dims (64 of 256) dims. Same constants
    // llama.cpp uses without YaRN: freq_scale 1, ext_factor 0, attn_factor 1, beta 32/1.
    int sections[4] = {m_.rope_sections[0], m_.rope_sections[1], m_.rope_sections[2], m_.rope_sections[3]};
    q = ggml_rope_multi(ctx, q, in.pos, nullptr, m_.rope_dims, sections, GGML_ROPE_TYPE_IMROPE, m_.n_ctx_train,
                        static_cast<float>(m_.rope_freq_base), 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_multi(ctx, k, in.pos, nullptr, m_.rope_dims, sections, GGML_ROPE_TYPE_IMROPE, m_.n_ctx_train,
                        static_cast<float>(m_.rope_freq_base), 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    ggml_tensor* o = softmax_attention(q, k, v, in, 1.0f / std::sqrt(static_cast<float>(hd)));
    o = ggml_mul(ctx, o, ggml_sigmoid(ctx, g));                          // output gate
    if (in.head_in) *in.head_in = o;                                     // per head, after its gate
    return ggml_mul_mat(ctx, W.wo, o);                                   // [n_embd, T]
}

ggml_tensor* Forward::attn_gemma(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const {
    ggml_context* ctx = in.ctx;
    const LayerTensors& S = *W.L;
    const int64_t T = in.T, hd = S.head_dim, nh = S.n_head, nkv = S.n_head_kv;
    const float eps = static_cast<float>(m_.rms_eps);

    ggml_tensor* q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wq, h), hd, nh, T);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), W.q_norm);
    ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wk, h), hd, nkv, T);
    // gemma4 attention_k_eq_v: V is the K projection itself (before K's norm and RoPE).
    ggml_tensor* v = W.wv ? ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W.wv, h), hd, nkv, T) : k;
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), W.k_norm);
    if (m_.gemma4) v = ggml_rms_norm(ctx, v, eps);   // unweighted

    // NEOX RoPE with this layer's base / scale / rotated dims; gemma4 full layers pass
    // rope_freqs.weight (1 on the rotated pairs, 1e30 on the rest: "proportional" RoPE).
    // ext_factor 0, attn_factor 1, beta 32/1: llama.cpp's values without YaRN.
    ggml_tensor* ff = S.rope_freqs ? rope_freqs_ : nullptr;
    const float base = static_cast<float>(S.rope_base), scale = static_cast<float>(S.rope_scale);
    q = ggml_rope_ext(ctx, q, in.pos, ff, S.n_rot, GGML_ROPE_TYPE_NEOX, m_.n_ctx_train, base, scale, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, in.pos, ff, S.n_rot, GGML_ROPE_TYPE_NEOX, m_.n_ctx_train, base, scale, 0.0f, 1.0f, 32.0f, 1.0f);
    // gemma3 scales Q itself and passes kq_scale 1 (bit-for-bit what llama.cpp does).
    if (m_.q_scale != 0.0f) q = ggml_scale(ctx, q, m_.q_scale);

    ggml_tensor* merged = softmax_attention(q, k, v, in, 1.0f);
    if (in.head_in) *in.head_in = merged;
    return ggml_mul_mat(ctx, W.wo, merged);   // [n_embd, T]
}

ggml_tensor* Forward::attn_qwen35_linear(const LayerW& W, ggml_tensor* h, const GraphInputs& in) const {
    ggml_context* ctx = in.ctx;
    const int64_t T = in.T;
    const int64_t dk = m_.ssm_head_k_dim, Hk = m_.ssm_n_k_heads, Hv = m_.ssm_n_v_heads;
    const int64_t dv = m_.ssm_inner / Hv;
    const int64_t K = m_.ssm_conv_kernel;
    const int64_t conv_ch = m_.ssm_inner + 2 * Hk * dk;                   // q | k | v channels
    const float eps = static_cast<float>(m_.rms_eps);

    ggml_tensor* qkv = ggml_mul_mat(ctx, W.wqkv, h);                     // [conv_ch, T]
    ggml_tensor* z = ggml_mul_mat(ctx, W.wgate, h);                      // [Hv·dv, T]

    ggml_tensor* beta = ggml_sigmoid(ctx, ggml_mul_mat(ctx, W.beta, h)); // [Hv, T]
    beta = ggml_reshape_4d(ctx, beta, 1, Hv, T, 1);
    ggml_tensor* alpha = ggml_mul_mat(ctx, W.alpha, h);                  // [Hv, T]
    // g = softplus(alpha + dt_bias) · A, A = −exp(A_log) (the converter stores A).
    ggml_tensor* g = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, W.dt)), W.a);
    g = ggml_reshape_4d(ctx, g, 1, Hv, T, 1);

    // Causal depthwise conv over time: the cached last K−1 frames (zero for a new
    // sequence), then the new tokens — llama.cpp's build_conv_state. The last K−1
    // frames of that window become the cached state for the next decode.
    ggml_tensor* conv_state = in.cache->conv;                            // [K−1, conv_ch, 1]
    ggml_tensor* conv_in = ggml_concat(ctx, conv_state, ggml_transpose(ctx, ggml_reshape_3d(ctx, qkv, conv_ch, T, 1)), 0);
    ggml_tensor* conv_last = ggml_view_3d(ctx, conv_in, K - 1, conv_ch, 1, conv_in->nb[1], conv_in->nb[2],
                                          ggml_row_size(conv_in->type, T));
    ggml_build_forward_expand(in.gf, ggml_cpy(ctx, conv_last, conv_state));   // runs after the concat read it
    ggml_tensor* conv = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_in, W.conv1d));   // [conv_ch, T, 1]

    const size_t row = ggml_row_size(conv->type, conv_ch);
    ggml_tensor* q = ggml_view_4d(ctx, conv, dk, Hk, T, 1, ggml_row_size(conv->type, dk), row, row * T, 0);
    ggml_tensor* k = ggml_view_4d(ctx, conv, dk, Hk, T, 1, ggml_row_size(conv->type, dk), row, row * T,
                                  ggml_row_size(conv->type, Hk * dk));
    ggml_tensor* v = ggml_view_4d(ctx, conv, dv, Hv, T, 1, ggml_row_size(conv->type, dv), row, row * T,
                                  ggml_row_size(conv->type, 2 * Hk * dk));

    // L2 norm per key head, as llama.cpp's build_gdn_l2_norm: rms_norm(eps/n)/√n.
    auto l2 = [&](ggml_tensor* x) {
        const float n = static_cast<float>(x->ne[0]);
        return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps / n), 1.0f / std::sqrt(n));
    };
    q = l2(q);
    k = l2(k);

    // The fused op broadcasts the Hk key heads over the Hv value heads in tiled
    // order (value head i uses key head i % Hk) — the order the converter
    // rearranged V heads into. It applies the 1/√d query scale itself. It reads the
    // cached state as S₀ and returns the outputs followed by the final state, which
    // is copied back into the cache.
    ggml_tensor* state = in.cache->ssm;                                   // [dv, dv, Hv, 1]
    ggml_tensor* res = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
    ggml_tensor* o = ggml_view_4d(ctx, res, dv, Hv, T, 1, ggml_row_size(res->type, dv), ggml_row_size(res->type, dv * Hv),
                                  ggml_row_size(res->type, dv * Hv * T), 0);
    ggml_tensor* s_new = ggml_view_4d(ctx, res, dv, dv, Hv, 1, ggml_row_size(res->type, dv), ggml_row_size(res->type, dv * dv),
                                      ggml_row_size(res->type, dv * dv * Hv), ggml_row_size(res->type, dv * Hv * T));
    ggml_build_forward_expand(in.gf, ggml_cpy(ctx, s_new, state));

    // Gated RMSNorm (ssm_norm gain is NOT +1-baked), then the output projection.
    ggml_tensor* z4 = ggml_reshape_4d(ctx, z, dv, Hv, T, 1);
    o = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, o, eps), W.ssm_norm), ggml_silu(ctx, z4));
    o = ggml_reshape_2d(ctx, ggml_cont(ctx, o), dv * Hv, T);
    if (in.head_in) *in.head_in = o;                                     // value head h is rows [h·dv, (h+1)·dv)
    return ggml_mul_mat(ctx, W.ssm_out, o);                              // [n_embd, T]
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

ForwardResult Forward::run(const std::vector<int32_t>& tokens, const Interventions& iv, const RunOptions& opt) const {
    if (tokens.empty()) throw std::runtime_error("empty token sequence");
    auto c = new_cache(static_cast<int>(tokens.size()));
    return decode(*c, tokens, iv, opt);
}

ForwardResult Forward::decode(DecodeCache& c, const std::vector<int32_t>& tokens, const Interventions& iv,
                              const RunOptions& opt) const {
    return decode_impl(c, tokens, static_cast<int>(tokens.size()), 0, nullptr, iv, opt);
}

ForwardResult Forward::run_from(int start_layer, const std::vector<float>& resid_in, int n_tokens, const Interventions& iv,
                                const RunOptions& opt) const {
    if (start_layer < 0 || start_layer > m_.n_layer) throw std::out_of_range("run_from: layer " + std::to_string(start_layer) + " out of range");
    if (n_tokens <= 0) throw std::runtime_error("empty token sequence");
    if (resid_in.size() != static_cast<size_t>(n_tokens) * static_cast<size_t>(m_.n_embd)) {
        throw std::invalid_argument("run_from: resid_in is " + std::to_string(resid_in.size()) + " floats, want n_tokens × n_embd = " +
                                    std::to_string(static_cast<size_t>(n_tokens) * static_cast<size_t>(m_.n_embd)));
    }
    auto c = new_cache(n_tokens);
    return decode_impl(*c, {}, n_tokens, start_layer, &resid_in, iv, opt);
}

ForwardResult Forward::decode_impl(DecodeCache& c, const std::vector<int32_t>& tokens, int n_tokens, int start_layer,
                                   const std::vector<float>* resid_in, const Interventions& iv, const RunOptions& opt) const {
    std::lock_guard<std::mutex> lock(mu_);

    const bool all_logits = opt.all_logits;
    std::vector<int> capture_layers = opt.capture_layers;
    if (opt.capture_all_layers) {
        capture_layers.clear();
        for (int l = 0; l < m_.n_layer; ++l) capture_layers.push_back(l);
    }
    const int64_t T = n_tokens;
    if (T == 0) throw std::runtime_error("empty token sequence");
    if (c.n_past_ == 0) {
        c.iv_ = iv;
    } else if (!(c.iv_ == iv)) {
        throw std::runtime_error("decode: patches changed mid-sequence (the cache was computed under the old ones)");
    }
    const int64_t n_past = c.n_past_;
    const int64_t n_used = n_past + T;
    if (n_used > c.n_cells_) {
        throw std::runtime_error("decode: cache full (" + std::to_string(c.n_cells_) + " cells, need " + std::to_string(n_used) + ")");
    }
    // KV cells attended over, exactly as llama.cpp's cache sizes its view:
    // min(size, max(256, round_up(used, 256))) (llama_kv_cache::get_n_kv).
    if (opt.watch.size() > 256) {
        throw std::invalid_argument("RunOptions::watch: " + std::to_string(opt.watch.size()) + " watches (max 256) — capture the layers instead");
    }
    const int64_t n_kv = std::min<int64_t>(c.n_cells_, std::max<int64_t>(256, round_up_256(n_used)));
    const int64_t n_embd = m_.n_embd, n_ff = m_.n_ff;
    const int64_t n_vocab = output_->ne[1];
    const float eps = static_cast<float>(m_.rms_eps);

    // Graph context: tensor headers only; the scheduler allocates storage.
    ggml_init_params ip{};
    ip.mem_size = kGraphSize * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphSize, false);
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("ggml_init failed for the graph");
    struct CtxGuard {
        ggml_context* c;
        ~CtxGuard() { ggml_free(c); }
    } guard{ctx};
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);

    // --- inputs: declared now, uploaded after allocation ------------------------
    std::vector<std::pair<ggml_tensor*, std::vector<uint8_t>>> uploads;
    auto input = [&](ggml_tensor* t, const void* data, size_t n) {
        ggml_set_input(t);
        const auto* p = static_cast<const uint8_t*>(data);
        uploads.emplace_back(t, std::vector<uint8_t>(p, p + n));
        return t;
    };


    // Positions n_past … n_past+T−1. Llama: one stream. Qwen3.5 M-RoPE: four streams,
    // section-major (t, y, x, z); a text token at position p is [p, p, p, 0], as
    // llama.cpp's batch expands it (llm_graph_input_pos::set_input).
    const int64_t n_pos = qwen35_ ? 4 : 1;
    std::vector<int32_t> pos(static_cast<size_t>(T * n_pos));
    for (int64_t s = 0; s < n_pos; ++s)
        for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(s * T + i)] = (s < 3) ? static_cast<int32_t>(n_past + i) : 0;
    ggml_tensor* inp_pos = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T * n_pos), pos.data(), pos.size() * 4);

    // Causal mask [n_kv, T_q]: query i (absolute position n_past+i) sees cells ≤ its
    // own position; later cells and padding are −inf.
    std::vector<float> mask(static_cast<size_t>(n_kv * T));
    for (int64_t q = 0; q < T; ++q)
        for (int64_t k = 0; k < n_kv; ++k)
            mask[static_cast<size_t>(q * n_kv + k)] = (k <= n_past + q) ? 0.0f : -std::numeric_limits<float>::infinity();
    ggml_tensor* kq_mask = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T), mask.data(), mask.size() * 4);
    // Gemma sliding layers: also masked when the key is n_swa or more positions back
    // (llama_hparams::is_masked_swa, LLAMA_SWA_TYPE_STANDARD: p_query − p_key ≥ n_swa).
    ggml_tensor* kq_mask_swa = nullptr;
    if (m_.n_swa > 0) {
        std::vector<float> ms = mask;
        for (int64_t q = 0; q < T; ++q)
            for (int64_t k = 0; k < n_kv; ++k)
                if (n_past + q - k >= m_.n_swa) ms[static_cast<size_t>(q * n_kv + k)] = -std::numeric_limits<float>::infinity();
        kq_mask_swa = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T), ms.data(), ms.size() * 4);
    }

    // Intervention vectors for the layers that have any.
    std::map<int, std::pair<ggml_tensor*, ggml_tensor*>> iv_t;
    for (const auto& [l, L] : iv.layers) {
        ggml_tensor* s = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_ff), L.scale.data(), static_cast<size_t>(n_ff) * 4);
        ggml_tensor* a = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_ff), L.add.data(), static_cast<size_t>(n_ff) * 4);
        iv_t[l] = {s, a};
    }
    // Prompt-scoped adds: only when this batch holds positions below prompt_len.
    // All of it below → a plain broadcast add; a mixed batch → add_prompt ⊗ mask,
    // mask[t] = 1 for absolute position < prompt_len. Generation steps past the
    // prompt build none of this, so they run the same graph as without it.
    const int64_t n_prompt_here = std::clamp<int64_t>(iv.prompt_len - n_past, 0, T);
    std::map<int, ggml_tensor*> ivp_t;
    ggml_tensor* prompt_mask = nullptr;
    if (n_prompt_here > 0 && iv.has_prompt_scope()) {
        if (n_prompt_here < T) {
            std::vector<float> pm(static_cast<size_t>(T), 0.0f);
            std::fill(pm.begin(), pm.begin() + n_prompt_here, 1.0f);
            prompt_mask = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T), pm.data(), pm.size() * 4);
        }
        for (const auto& [l, L] : iv.layers) {
            if (std::all_of(L.add_prompt.begin(), L.add_prompt.end(), [](float v) { return v == 0.0f; })) continue;
            ivp_t[l] = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, 1), L.add_prompt.data(), static_cast<size_t>(n_ff) * 4);
        }
    }

    // Rank-one down-projection edits: u as [1, n_embd] and w as [n_ff, 1], so that
    // mul_mat(u, mul_mat(w, a)) is the [n_embd, T] outer product u·(w·a).
    std::map<int, std::vector<std::pair<ggml_tensor*, ggml_tensor*>>> r1_t;
    for (const auto& r : iv.rank_one) {
        if (r.layer < start_layer) continue;
        ggml_tensor* u = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_embd), r.u.data(), r.u.size() * 4);
        ggml_tensor* w = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, 1), r.w.data(), r.w.size() * 4);
        r1_t[r.layer].push_back({u, w});
    }

    // Activation-patching overrides that land in this batch, per (site, layer): a
    // [1, T] keep mask (0 at overridden positions) and the [W, T] replacement
    // values (0 elsewhere), applied as v' = v·keep + vals. Bit-exact both ways:
    // untouched positions get v·1 + 0, overridden ones 0 + value.
    std::map<std::pair<Site, int>, std::pair<ggml_tensor*, ggml_tensor*>> ov_t;
    {
        std::map<std::pair<Site, int>, std::vector<const Interventions::Override*>> by;
        for (const auto& o : iv.overrides)
            if (o.pos >= n_past && o.pos < n_past + T && o.layer >= start_layer) by[{o.site, o.layer}].push_back(&o);
        for (const auto& [key, os] : by) {
            const int64_t W = key.first == Site::Mlp ? n_ff : n_embd;
            std::vector<float> kp(static_cast<size_t>(T), 1.0f), vals(static_cast<size_t>(W * T), 0.0f);
            for (const auto* o : os) {
                const int64_t t = o->pos - n_past;
                kp[static_cast<size_t>(t)] = 0.0f;
                std::copy(o->v.begin(), o->v.end(), vals.begin() + t * W);
            }
            ggml_tensor* k = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T), kp.data(), kp.size() * 4);
            ggml_tensor* v = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, T), vals.data(), vals.size() * 4);
            ov_t[key] = {k, v};
        }
    }
    auto apply_ov = [&](Site site, int il, ggml_tensor* v) {
        auto it = ov_t.find({site, il});
        return it == ov_t.end() ? v : ggml_add(ctx, ggml_mul(ctx, v, it->second.first), it->second.second);
    };

    // Capture window [lo, hi) of this batch. With the full window a capture is the
    // tensor itself, marked an output so the allocator never reuses its memory (the
    // pre-window behaviour, node for node). With a narrower one it is a contiguous
    // copy of the window's columns, and only the copy is pinned.
    const int64_t cap_lo = std::clamp<int64_t>(opt.cap_lo, 0, T);
    const int64_t cap_hi = opt.cap_hi < 0 ? T : std::clamp<int64_t>(opt.cap_hi, cap_lo, T);
    const int64_t cap_n = cap_hi - cap_lo;
    const bool full_window = cap_lo == 0 && cap_hi == T;
    std::vector<ggml_tensor*> pinned;   // every capture, expanded into the graph at the end
    auto keep = [&](ggml_tensor* t) {
        // A view (e.g. a reshape) shares its parent's memory, which marking the view an
        // output doesn't pin: copy it.
        if (full_window && t->view_src) t = ggml_cont(ctx, t);
        if (!full_window) {
            t = ggml_cont(ctx, ggml_view_2d(ctx, t, t->ne[0], cap_n, t->nb[1], static_cast<size_t>(cap_lo) * t->nb[1]));
        }
        ggml_set_output(t);
        pinned.push_back(t);
        return t;
    };
    // Watched neurons: row n of a ([n_ff, T]) at every position, as a [1, T] copy.
    std::map<std::pair<int, int>, ggml_tensor*> watch_cap;
    auto keep_watch = [&](int il, ggml_tensor* a) {
        for (const auto& [l, n] : opt.watch) {
            if (l != il || n < 0 || n >= n_ff) continue;
            ggml_tensor* w = ggml_cont(ctx, ggml_view_2d(ctx, a, 1, T, a->nb[1], static_cast<size_t>(n) * a->nb[0]));
            ggml_set_output(w);
            pinned.push_back(w);
            watch_cap[{l, n}] = w;
        }
    };
    std::map<int, ggml_tensor*> captured, attn_cap, resid_cap, head_cap, attn_raw_cap, mlp_raw_cap;
    ggml_tensor* probs_cap = nullptr;
    const bool want_acts = !capture_layers.empty();
    auto wants_layer = [&](int l) { return std::find(capture_layers.begin(), capture_layers.end(), l) != capture_layers.end(); };

    // --- the model --------------------------------------------------------------
    // x entering layer start_layer: the token embedding, or the caller's residual.
    ggml_tensor* x = nullptr;
    if (resid_in) {
        x = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T), resid_in->data(), resid_in->size() * sizeof(float));
    } else {
        ggml_tensor* inp_tokens = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T), tokens.data(), static_cast<size_t>(T) * 4);
        x = ggml_get_rows(ctx, tok_embd_, inp_tokens);   // [n_embd, T]
        if (m_.embd_scale != 1.0f) x = ggml_scale(ctx, x, m_.embd_scale);   // gemma: × √n_embd
    }
    ggml_tensor* embed_cap = opt.capture_resid && start_layer == 0 ? keep(x) : nullptr;

    for (int il = start_layer; il < m_.n_layer; ++il) {
        const LayerW& W = layers_[static_cast<size_t>(il)];
        ggml_tensor* head_in = nullptr;
        ggml_tensor* probs = nullptr;
        const GraphInputs gin{ctx,   gf, inp_pos, W.L->swa && kq_mask_swa ? kq_mask_swa : kq_mask, T, n_past, n_kv, &c.layers_[static_cast<size_t>(il)],
                              opt.capture_heads ? &head_in : nullptr, il == opt.attn_probs_layer ? &probs : nullptr};

        // attention half (kind-specific), adding attn_out to the residual stream
        ggml_tensor* h = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), W.attn_norm);
        ggml_tensor* attn_out = gemma_                                  ? attn_gemma(W, h, gin)
                                : !qwen35_                              ? attn_llama(W, h, gin)
                                : W.kind == LayerKind::linear_attention ? attn_qwen35_linear(W, h, gin)
                                                                         : attn_qwen35_full(W, h, gin);
        // Gemma: what joins the stream is the post-normed output.
        if (W.attn_post_norm) {
            if (opt.capture_resid) attn_raw_cap[il] = keep(attn_out);
            attn_out = ggml_mul(ctx, ggml_rms_norm(ctx, attn_out, eps), W.attn_post_norm);
        }
        attn_out = apply_ov(Site::Attn, il, attn_out);
        if (opt.capture_resid) attn_cap[il] = keep(attn_out);
        if (head_in) head_cap[il] = keep(head_in);
        if (probs && opt.attn_probs_pos >= 0 && opt.attn_probs_pos < T) {
            // One query row of [n_kv, T, n_head] → [n_kv, 1, n_head], contiguous.
            probs_cap = ggml_cont(ctx, ggml_view_3d(ctx, probs, probs->ne[0], 1, probs->ne[2], probs->nb[1], probs->nb[2],
                                                    static_cast<size_t>(opt.attn_probs_pos) * probs->nb[1]));
            ggml_set_output(probs_cap);
            pinned.push_back(probs_cap);
        }
        x = ggml_add(ctx, attn_out, x);

        // MLP — the neurons live in `a`
        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), W.ffn_norm);
        ggml_tensor* gate = ggml_mul_mat(ctx, W.gate, h);    // [n_ff, T]
        ggml_tensor* up = ggml_mul_mat(ctx, W.up, h);        // [n_ff, T]
        // SwiGLU, or Gemma's GeGLU — the fused op llama.cpp's build_ffn uses for it.
        ggml_tensor* a = gemma_ ? ggml_geglu_split(ctx, gate, up) : ggml_mul(ctx, ggml_silu(ctx, gate), up);
        a = apply_ov(Site::Mlp, il, a);
        if (want_acts && wants_layer(il)) captured[il] = keep(a);   // pre-intervention activations
        keep_watch(il, a);

        if (auto it = iv_t.find(il); it != iv_t.end()) {
            // [n_ff] vectors broadcast across all T positions.
            a = ggml_add(ctx, ggml_mul(ctx, a, it->second.first), it->second.second);
        }
        if (auto it = ivp_t.find(il); it != ivp_t.end()) {
            // [n_ff, 1] → [n_ff, T], zeroed past the prompt by the [1, T] mask.
            ggml_tensor* ap = prompt_mask ? ggml_mul(ctx, ggml_repeat(ctx, it->second, a), prompt_mask) : it->second;
            a = ggml_add(ctx, a, ap);
        }
        ggml_tensor* mlp_out = ggml_mul_mat(ctx, W.down, a);
        if (auto it = r1_t.find(il); it != r1_t.end()) {
            for (const auto& [u, w] : it->second) mlp_out = ggml_add(ctx, mlp_out, ggml_mul_mat(ctx, u, ggml_mul_mat(ctx, w, a)));
        }
        if (W.ffn_post_norm) {
            if (opt.capture_resid) mlp_raw_cap[il] = keep(mlp_out);
            mlp_out = ggml_mul(ctx, ggml_rms_norm(ctx, mlp_out, eps), W.ffn_post_norm);
        }
        x = ggml_add(ctx, mlp_out, x);
        if (W.out_scale) x = ggml_mul(ctx, x, W.out_scale);   // gemma4 layer_output_scale (one scalar)
        x = apply_ov(Site::Resid, il, x);
        if (opt.capture_layer_resid) resid_cap[il] = keep(x);
        if (il == opt.last_layer) break;
    }

    ggml_tensor* final_cap = opt.capture_resid ? keep(x) : nullptr;
    // Only the last position's logits unless asked (then the capture window's):
    // saves the other rows of a vocab-wide matmul.
    int64_t logit_lo = T - 1, logit_n = 1;
    if (all_logits) {
        logit_lo = cap_lo;
        logit_n = cap_n;
    }
    if (logit_lo != 0 || logit_n != T) {
        x = ggml_view_2d(ctx, x, n_embd, logit_n, x->nb[1], static_cast<size_t>(logit_lo) * x->nb[1]);
    }
    ggml_tensor* logits = ggml_mul_mat(ctx, output_, ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), output_norm_));
    if (m_.softcap > 0.0f) {
        // c·tanh(z/c), as llama.cpp: scale, tanh, scale.
        logits = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, logits, 1.0f / m_.softcap)), m_.softcap);
    }
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);
    for (ggml_tensor* t : pinned) ggml_build_forward_expand(gf, t);

    // --- allocate, upload inputs, compute ---------------------------------------
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) throw std::runtime_error("ggml scheduler could not allocate the graph");
    for (auto& [t, bytes] : uploads) {
        // An input no node reads (positions/mask when run_from skips every layer) is
        // never allocated; there is nothing to upload.
        if (t->buffer) ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml graph compute failed on " + device_name());
    }

    ForwardResult r;
    r.n_tokens = static_cast<int>(T);
    r.n_vocab = static_cast<int>(n_vocab);
    r.cap_lo = static_cast<int>(cap_lo);
    r.cap_n = static_cast<int>(cap_n);
    r.logit_lo = static_cast<int>(logit_lo);
    r.logit_n = static_cast<int>(logit_n);
    auto grab = [](ggml_tensor* t) {
        std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    if (opt.capture_resid) {
        if (embed_cap) r.embed = grab(embed_cap);
        r.final_resid = grab(final_cap);
        for (const auto& [l, t] : attn_cap) r.attn_out[l] = grab(t);
        for (const auto& [l, t] : attn_raw_cap) r.attn_raw[l] = grab(t);
        for (const auto& [l, t] : mlp_raw_cap) r.mlp_raw[l] = grab(t);
    }
    if (opt.capture_layer_resid) {
        for (const auto& [l, t] : resid_cap) r.layer_resid[l] = grab(t);
    }
    r.logits = grab(logits);
    for (const auto& [l, t] : captured) r.acts[l] = grab(t);
    for (const auto& [k, t] : watch_cap) r.watched[k] = grab(t);
    for (const auto& [l, t] : head_cap) r.head_in[l] = grab(t);
    if (probs_cap) {
        r.attn_probs = grab(probs_cap);
        r.attn_n_kv = static_cast<int>(probs_cap->ne[0]);
        r.attn_n_head = static_cast<int>(probs_cap->ne[2]);
    }

    c.n_past_ = static_cast<int>(n_used);
    return r;
}

// ---------------------------------------------------------------------------
// ROME's value search: loss and dloss/dδ through layers layer+1 … n−1
// ---------------------------------------------------------------------------

Forward::DeltaLoss Forward::delta_loss(int layer, const DeltaBatch& b, const std::vector<float>& delta) const {
    if (qwen35_ || gemma_ || m_.has_post_norms()) throw std::runtime_error("rome: the value search supports llama-style models only");
    if (device_ != Device::cpu) throw std::runtime_error("rome: the value search runs on the CPU (NEURONAUT_DEVICE=cpu)");
    const int64_t T = b.n_tokens(), n_embd = m_.n_embd, n_vocab = output_->ne[1];
    const float eps = static_cast<float>(m_.rms_eps);
    if (layer < 0 || layer >= m_.n_layer) throw std::out_of_range("rome: layer out of range");
    if (T <= 0 || b.seq.size() != static_cast<size_t>(T) || b.delta_mask.size() != static_cast<size_t>(T) ||
        b.resid.size() != static_cast<size_t>(T * n_embd) || delta.size() != static_cast<size_t>(n_embd) || b.nll_rows.size() != b.nll_ids.size() ||
        b.nll_rows.empty() || b.kl_ref.size() != b.kl_rows.size() * static_cast<size_t>(n_vocab))
        throw std::invalid_argument("rome: malformed DeltaBatch");

    std::lock_guard<std::mutex> lk(mu_);
    if (!gradw_) {
        // f32 copies of every matrix the value search runs through.
        auto g = std::make_unique<GradWeights>();
        size_t bytes = static_cast<size_t>(ggml_nelements(output_)) * sizeof(float);
        for (const auto& W : layers_)
            for (ggml_tensor* t : {W.wq, W.wk, W.wv, W.wo, W.gate, W.up, W.down}) bytes += static_cast<size_t>(ggml_nelements(t)) * sizeof(float);
        ggml_init_params gp{};
        gp.mem_size = bytes + (8 * layers_.size() + 8) * (ggml_tensor_overhead() + 64);
        g->ctx = ggml_init(gp);
        if (!g->ctx) throw std::runtime_error("rome: can't allocate f32 weights (" + std::to_string(bytes >> 20) + " MB)");
        auto to_f32 = [&](ggml_tensor* t) {
            ggml_tensor* o = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, t->ne[0], t->ne[1]);
            const auto* tr = ggml_get_type_traits(t->type);
            for (int64_t r = 0; r < t->ne[1]; ++r) {
                const char* src = static_cast<const char*>(t->data) + static_cast<size_t>(r) * t->nb[1];
                float* dst = static_cast<float*>(o->data) + r * t->ne[0];
                if (t->type == GGML_TYPE_F32) std::memcpy(dst, src, static_cast<size_t>(t->ne[0]) * sizeof(float));
                else tr->to_float(src, dst, t->ne[0]);
            }
            return o;
        };
        for (const auto& W : layers_) {
            LayerW c = W;
            c.wq = to_f32(W.wq), c.wk = to_f32(W.wk), c.wv = to_f32(W.wv), c.wo = to_f32(W.wo);
            c.gate = to_f32(W.gate), c.up = to_f32(W.up), c.down = to_f32(W.down);
            g->layers.push_back(c);
        }
        g->output = to_f32(output_);
        gradw_ = std::move(g);
    }
    // The backward graph is a few times the forward's; the context holds headers only.
    constexpr size_t kGradGraph = 4 * kGraphSize;
    ggml_init_params ip{};
    ip.mem_size = kGradGraph * ggml_tensor_overhead() * 2 + ggml_graph_overhead_custom(kGradGraph, true);
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("ggml_init failed for the gradient graph");
    struct CtxGuard {
        ggml_context* c;
        ~CtxGuard() { ggml_free(c); }
    } guard{ctx};
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGradGraph, true);

    std::vector<std::pair<ggml_tensor*, const void*>> uploads;
    auto input = [&](ggml_tensor* t, const void* data) {
        ggml_set_input(t);
        uploads.emplace_back(t, data);
        return t;
    };
    ggml_tensor* x0 = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T), b.resid.data());
    ggml_tensor* d = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd), delta.data());
    ggml_set_param(d);
    ggml_tensor* dmask = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T), b.delta_mask.data());
    ggml_tensor* pos = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T), b.pos.data());
    // Block-diagonal causal mask: a query sees earlier tokens of its own sequence.
    std::vector<float> mask(static_cast<size_t>(T * T));
    for (int64_t q = 0; q < T; ++q)
        for (int64_t k = 0; k < T; ++k)
            mask[static_cast<size_t>(q * T + k)] = (k <= q && b.seq[static_cast<size_t>(k)] == b.seq[static_cast<size_t>(q)]) ? 0.0f
                                                                                                                         : -std::numeric_limits<float>::infinity();
    ggml_tensor* kq_mask = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T), mask.data());

    // x = resid + δ ⊗ mask
    ggml_tensor* x = ggml_add(ctx, x0, ggml_mul(ctx, ggml_repeat(ctx, d, x0), dmask));
    for (int il = layer + 1; il < m_.n_layer; ++il) {
        const LayerW& W = gradw_->layers[static_cast<size_t>(il)];
        const GraphInputs gin{ctx, gf, pos, kq_mask, T, 0, T, nullptr, nullptr, nullptr};
        ggml_tensor* h = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), W.attn_norm);
        x = ggml_add(ctx, attn_llama(W, h, gin), x);
        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), W.ffn_norm);
        ggml_tensor* a = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, W.gate, h)), ggml_mul_mat(ctx, W.up, h));
        x = ggml_add(ctx, ggml_mul_mat(ctx, W.down, a), x);
    }
    auto logits_at = [&](const std::vector<int32_t>& rows) {
        ggml_tensor* idx = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<int64_t>(rows.size())), rows.data());
        ggml_tensor* xr = ggml_get_rows(ctx, x, idx);   // [n_embd, rows]
        return ggml_mul_mat(ctx, gradw_->output, ggml_mul(ctx, ggml_rms_norm(ctx, xr, eps), output_norm_));
    };
    // NLL: cross-entropy against one-hot targets (ggml means it over rows).
    const int64_t n_nll = static_cast<int64_t>(b.nll_rows.size());
    std::vector<float> onehot(static_cast<size_t>(n_nll * n_vocab), 0.0f);
    for (int64_t i = 0; i < n_nll; ++i) onehot[static_cast<size_t>(i * n_vocab + b.nll_ids[static_cast<size_t>(i)])] = 1.0f;
    ggml_tensor* labels = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_nll), onehot.data());
    ggml_tensor* nll = ggml_cross_entropy_loss(ctx, logits_at(b.nll_rows), labels);
    ggml_set_output(nll);
    ggml_tensor* loss = nll;
    ggml_tensor* ce = nullptr;
    if (!b.kl_rows.empty() && b.kl_weight > 0) {
        // Cross-entropy against the reference distribution = KL(ref ‖ p) + H(ref).
        ggml_tensor* ref = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, static_cast<int64_t>(b.kl_rows.size())), b.kl_ref.data());
        ce = ggml_cross_entropy_loss(ctx, logits_at(b.kl_rows), ref);
        ggml_set_output(ce);
        loss = ggml_add(ctx, nll, ggml_scale(ctx, ce, b.kl_weight));
    }
    ggml_set_output(loss);
    ggml_set_loss(loss);
    ggml_build_forward_expand(gf, loss);
    ggml_build_backward_expand(ctx, gf, nullptr);
    ggml_tensor* grad = ggml_graph_get_grad(gf, d);
    if (!grad) throw std::logic_error("rome: δ got no gradient");
    ggml_set_output(grad);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    struct GaGuard {
        ggml_gallocr_t g;
        ~GaGuard() { ggml_gallocr_free(g); }
    } gaguard{ga};
    if (!ggml_gallocr_alloc_graph(ga, gf)) throw std::runtime_error("rome: could not allocate the gradient graph");
    for (auto& [t, p] : uploads)
        if (t->buffer) ggml_backend_tensor_set(t, p, 0, ggml_nbytes(t));
    ggml_graph_reset(gf);   // loss grad 1, every other grad 0
    if (ggml_backend_graph_compute(backend_cpu_, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("rome: gradient graph failed");

    DeltaLoss r;
    ggml_backend_tensor_get(nll, &r.nll, 0, sizeof(float));
    ggml_backend_tensor_get(loss, &r.total, 0, sizeof(float));
    if (ce) {
        float c = 0;
        ggml_backend_tensor_get(ce, &c, 0, sizeof(float));
        double h = 0;   // mean entropy of the reference rows
        for (float p : b.kl_ref)
            if (p > 0) h -= static_cast<double>(p) * std::log(static_cast<double>(p));
        r.kl = c - static_cast<float>(h / static_cast<double>(b.kl_rows.size()));
    }
    r.grad.resize(static_cast<size_t>(n_embd));
    ggml_backend_tensor_get(grad, r.grad.data(), 0, r.grad.size() * sizeof(float));
    return r;
}

float prob_of(const float* logits, int n_vocab, int id) {
    const float mx = *std::max_element(logits, logits + n_vocab);
    double z = 0;
    for (int i = 0; i < n_vocab; ++i) z += std::exp(static_cast<double>(logits[i] - mx));
    return static_cast<float>(std::exp(static_cast<double>(logits[id] - mx)) / z);
}

std::vector<std::pair<int, float>> top_probs(const float* logits, int n_vocab, int k) {
    const float mx = *std::max_element(logits, logits + n_vocab);
    double z = 0;
    for (int i = 0; i < n_vocab; ++i) z += std::exp(static_cast<double>(logits[i] - mx));
    std::vector<int> idx(static_cast<size_t>(n_vocab));
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n_vocab);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<std::pair<int, float>> out;
    for (int i = 0; i < k; ++i) {
        const int id = idx[static_cast<size_t>(i)];
        out.emplace_back(id, static_cast<float>(std::exp(static_cast<double>(logits[id] - mx)) / z));
    }
    return out;
}

}  // namespace neuronaut
