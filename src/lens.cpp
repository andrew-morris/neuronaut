// lens.cpp — see lens.h.
#include "lens.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "format.h"
#include "neuron.h"

namespace neuronaut {

void apply_output_gain(const ModelLayout& m, std::vector<float>& v) {
    if (!m.output_norm) return;
    if (m.output_norm->ne[0] != static_cast<int64_t>(v.size())) {
        throw std::runtime_error(m.output_norm->name + " width does not match vector length");
    }
    const std::vector<float> g = read_vec(*m.output_norm);
    for (size_t i = 0; i < v.size(); ++i) v[i] *= g[i];
}

void apply_write_gain(const ModelLayout& m, int layer, std::vector<float>& down) {
    if (layer < 0 || layer >= m.n_layer) return;
    const TensorInfo* g = m.layers[static_cast<size_t>(layer)].ffn_post_norm;
    if (!g) return;
    const std::vector<float> gv = read_vec(*g);
    for (size_t i = 0; i < down.size() && i < gv.size(); ++i) down[i] *= gv[i];
}

std::vector<float> unembed_all(const ModelLayout& m, const std::vector<float>& v) {
    const TensorInfo* wu = m.output ? m.output : m.token_embd;   // tied embeddings fall back
    if (!wu) throw std::runtime_error("no output.weight or token_embd.weight to unembed with");
    if (wu->n_dims != 2 || wu->ne[0] != static_cast<int64_t>(v.size())) {
        throw std::runtime_error(wu->name + " width does not match vector length");
    }

    // Fold the final RMSNorm gain into the probe vector once.
    std::vector<float> probe = v;
    apply_output_gain(m, probe);

    const int64_t n_vocab = wu->ne[1];
    std::vector<float> logits(static_cast<size_t>(n_vocab));

    // Each thread dequantizes and dots a contiguous slice of vocab rows.
    const unsigned n_threads = std::max(1u, std::min(std::thread::hardware_concurrency(), 32u));
    std::vector<std::thread> pool;
    const int64_t chunk = (n_vocab + n_threads - 1) / n_threads;
    for (unsigned t = 0; t < n_threads; ++t) {
        const int64_t lo = static_cast<int64_t>(t) * chunk;
        const int64_t hi = std::min(n_vocab, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&, lo, hi] {
            for (int64_t r = lo; r < hi; ++r) {
                const std::vector<float> row = read_row(*wu, r);
                double acc = 0;
                for (size_t i = 0; i < row.size(); ++i) acc += static_cast<double>(row[i]) * probe[i];
                logits[static_cast<size_t>(r)] = static_cast<float>(acc);
            }
        });
    }
    for (auto& th : pool) th.join();
    return logits;
}

LensResult logit_lens(const GgufFile& f, const ModelLayout& m, const std::vector<float>& v, int k) {
    const std::vector<float> logits = unembed_all(m, v);
    const int64_t n_vocab = static_cast<int64_t>(logits.size());

    std::vector<int> idx(static_cast<size_t>(n_vocab));
    std::iota(idx.begin(), idx.end(), 0);
    k = static_cast<int>(std::min<int64_t>(k, n_vocab));

    // Byte-level BPE vocabularies store stand-in characters; show real text.
    const std::string tm = f.get_str("tokenizer.ggml.model").value_or("");
    auto make = [&](int id) {
        std::string t = f.get_arr_str("tokenizer.ggml.tokens", static_cast<size_t>(id)).value_or("#" + std::to_string(id));
        return LensToken{id, logits[static_cast<size_t>(id)], vocab_text(tm, t)};
    };

    LensResult res;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });
    for (int i = 0; i < k; ++i) res.top.push_back(make(idx[static_cast<size_t>(i)]));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[static_cast<size_t>(a)] < logits[static_cast<size_t>(b)]; });
    for (int i = 0; i < k; ++i) res.bottom.push_back(make(idx[static_cast<size_t>(i)]));
    return res;
}

std::vector<int> find_token_ids(const GgufFile& f, const std::string& text) {
    const std::string tm = f.get_str("tokenizer.ggml.model").value_or("");
    std::vector<int> ids;
    const size_t n = f.get_arr_len("tokenizer.ggml.tokens");
    for (size_t i = 0; i < n; ++i) {
        const auto t = f.get_arr_str("tokenizer.ggml.tokens", i);
        if (t && vocab_text(tm, *t) == text) ids.push_back(static_cast<int>(i));
    }
    return ids;
}

WriterResult token_writers(const GgufFile& f, const ModelLayout& m, int token_id, int k) {
    (void)f;
    const TensorInfo* wu = m.output ? m.output : m.token_embd;
    if (!wu) throw std::runtime_error("no output.weight or token_embd.weight to unembed with");
    if (token_id < 0 || token_id >= wu->ne[1]) throw std::out_of_range("token id out of range");

    // u = g ⊙ W_U[token]
    std::vector<float> u = read_row(*wu, token_id);
    apply_output_gain(m, u);
    double u_norm = 0;
    for (float x : u) u_norm += static_cast<double>(x) * x;
    u_norm = std::sqrt(u_norm);

    // One thread per layer: dot[n] = Σ_e down[e][n]·u[e], norm2[n] = Σ_e down[e][n]².
    std::vector<std::vector<Writer>> per_layer(static_cast<size_t>(m.n_layer));
    std::vector<std::thread> pool;
    for (int l = 0; l < m.n_layer; ++l) {
        if (!m.has_ffn(l)) continue;
        pool.emplace_back([&, l] {
            const TensorInfo& down = *m.layers[static_cast<size_t>(l)].ffn_down;
            // Post-MLP norm gain (Gemma), per embedding dim: the written direction is g ⊙ down.
            std::vector<float> g(static_cast<size_t>(down.ne[1]), 1.0f);
            apply_write_gain(m, l, g);
            std::vector<double> dot(static_cast<size_t>(down.ne[0]), 0.0), n2(static_cast<size_t>(down.ne[0]), 0.0);
            for (int64_t e = 0; e < down.ne[1]; ++e) {
                const std::vector<float> row = read_row(down, e);   // all neurons' weight on dim e
                const double ge = g[static_cast<size_t>(e)];
                const double ue = u[static_cast<size_t>(e)];
                for (size_t n = 0; n < row.size(); ++n) {
                    const double w = row[n] * ge;
                    dot[n] += w * ue;
                    n2[n] += w * w;
                }
            }
            auto& out = per_layer[static_cast<size_t>(l)];
            out.reserve(dot.size());
            for (size_t n = 0; n < dot.size(); ++n) {
                const double denom = std::sqrt(n2[n]) * u_norm;
                out.push_back(Writer{l, static_cast<int>(n), static_cast<float>(dot[n]),
                                     static_cast<float>(denom > 0 ? dot[n] / denom : 0.0)});
            }
        });
    }
    for (auto& th : pool) th.join();

    std::vector<Writer> all;
    for (auto& v : per_layer) all.insert(all.end(), v.begin(), v.end());
    k = static_cast<int>(std::min<size_t>(static_cast<size_t>(k), all.size()));
    WriterResult r;
    std::partial_sort(all.begin(), all.begin() + k, all.end(), [](const Writer& a, const Writer& b) { return a.cos > b.cos; });
    r.positive.assign(all.begin(), all.begin() + k);
    std::partial_sort(all.begin(), all.begin() + k, all.end(), [](const Writer& a, const Writer& b) { return a.cos < b.cos; });
    r.negative.assign(all.begin(), all.begin() + k);
    return r;
}

}  // namespace neuronaut
