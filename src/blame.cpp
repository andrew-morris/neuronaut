// blame.cpp — see blame.h.
#include "blame.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

#include "lens.h"
#include "neuron.h"

namespace neuronaut {

std::vector<float> unembed_dir(const ModelLayout& m, int token) {
    const TensorInfo* wu = m.output ? m.output : m.token_embd;
    if (!wu) throw std::runtime_error("no unembedding matrix");
    if (token < 0 || token >= wu->ne[1]) throw std::out_of_range("token id out of range");
    std::vector<float> u = read_row(*wu, token);
    apply_output_gain(m, u);
    return u;
}

std::vector<float> later_scales(const ModelLayout& m) {
    std::vector<float> s(static_cast<size_t>(m.n_layer) + 1, 1.0f);
    for (int l = m.n_layer - 1; l >= 0; --l) {
        const TensorInfo* t = m.layers[static_cast<size_t>(l)].out_scale;
        s[static_cast<size_t>(l)] = s[static_cast<size_t>(l) + 1] * (t ? read_vec(*t).at(0) : 1.0f);
    }
    return s;
}

namespace {
// u ⊙ g · S for one layer's piece (g: a post-norm gain tensor, may be null).
std::vector<double> layer_u(const std::vector<float>& u, const TensorInfo* g, float S) {
    std::vector<double> out(u.begin(), u.end());
    if (g) {
        const std::vector<float> gv = read_vec(*g);
        for (size_t i = 0; i < out.size() && i < gv.size(); ++i) out[i] *= gv[i];
    }
    for (auto& x : out) x *= S;
    return out;
}
// 1/rms(v) with the graph's eps (rms_norm: v / √(mean(v²) + eps)); 1 without v.
double rms_inv_of(const float* v, size_t n, double eps) {
    if (!v) return 1.0;
    double ss = 0;
    for (size_t i = 0; i < n; ++i) ss += static_cast<double>(v[i]) * v[i];
    return 1.0 / std::sqrt(ss / static_cast<double>(n) + eps);
}
}  // namespace

std::vector<std::vector<float>> down_dots(const ModelLayout& m, const std::vector<float>& u) {
    std::vector<std::vector<float>> out(static_cast<size_t>(m.n_layer));
    const std::vector<float> S = later_scales(m);
    std::vector<std::thread> pool;
    for (int l = 0; l < m.n_layer; ++l) {
        if (!m.has_ffn(l)) continue;
        pool.emplace_back([&, l] {
            const TensorInfo& down = *m.layers[static_cast<size_t>(l)].ffn_down;
            const std::vector<double> ul = layer_u(u, m.layers[static_cast<size_t>(l)].ffn_post_norm, S[static_cast<size_t>(l)]);
            std::vector<double> acc(static_cast<size_t>(down.ne[0]), 0.0);
            for (int64_t e = 0; e < down.ne[1]; ++e) {
                const std::vector<float> row = read_row(down, e);   // dim e of every neuron's down column
                const double ue = ul[static_cast<size_t>(e)];
                for (size_t n = 0; n < row.size(); ++n) acc[n] += row[n] * ue;
            }
            out[static_cast<size_t>(l)].assign(acc.begin(), acc.end());
        });
    }
    for (auto& t : pool) t.join();
    return out;
}

std::vector<std::vector<float>> out_dots(const ModelLayout& m, const std::vector<float>& u) {
    std::vector<std::vector<float>> out(static_cast<size_t>(m.n_layer));
    const std::vector<float> S = later_scales(m);
    std::vector<std::thread> pool;
    for (int l = 0; l < m.n_layer; ++l) {
        const auto& L = m.layers[static_cast<size_t>(l)];
        const TensorInfo* wo = L.kind == LayerKind::linear_attention ? L.ssm_out : L.attn_output;
        if (!wo) continue;
        pool.emplace_back([&, l, wo] {
            // Row e of Wo is output dim e over every input dim: Woᵀu = Σ_e u[e]·row_e.
            const std::vector<double> ul = layer_u(u, m.layers[static_cast<size_t>(l)].attn_post_norm, S[static_cast<size_t>(l)]);
            std::vector<double> acc(static_cast<size_t>(wo->ne[0]), 0.0);
            for (int64_t e = 0; e < wo->ne[1]; ++e) {
                const std::vector<float> row = read_row(*wo, e);
                const double ue = ul[static_cast<size_t>(e)];
                for (size_t i = 0; i < row.size(); ++i) acc[i] += row[i] * ue;
            }
            out[static_cast<size_t>(l)].assign(acc.begin(), acc.end());
        });
    }
    for (auto& t : pool) t.join();
    return out;
}

int n_heads(const ModelLayout& m, int layer) {
    const auto& L = m.layers[static_cast<size_t>(layer)];
    return L.kind == LayerKind::linear_attention ? m.ssm_n_v_heads : L.n_head;   // per layer (gemma4: sliding ≠ full)
}

int head_width(const ModelLayout& m, int layer) {
    const auto& L = m.layers[static_cast<size_t>(layer)];
    if (L.kind == LayerKind::linear_attention) return m.ssm_inner / std::max(1, m.ssm_n_v_heads);
    return L.head_dim;
}

std::vector<BlameItem> Blame::layer_heads(int layer) const {
    std::vector<BlameItem> out;
    for (const auto& h : heads)
        if (h.layer == layer) out.push_back(h);
    return out;
}

std::vector<BlameItem> Blame::top(int k, bool toward) const {
    std::vector<BlameItem> all = neurons;
    // Heads replace their layer's attention total (it is their sum).
    if (heads.empty()) all.insert(all.end(), attn.begin(), attn.end());
    else all.insert(all.end(), heads.begin(), heads.end());
    all.push_back(BlameItem{BlameItem::Embed, -1, -1, 0, 0, embed});
    auto better = [toward](const BlameItem& a, const BlameItem& b) {
        return toward ? a.contrib > b.contrib : a.contrib < b.contrib;
    };
    k = std::min<int>(k, static_cast<int>(all.size()));
    std::partial_sort(all.begin(), all.begin() + k, all.end(), better);
    all.resize(static_cast<size_t>(k));
    // Drop items that don't actually push the requested way.
    all.erase(std::remove_if(all.begin(), all.end(),
                             [toward](const BlameItem& b) { return toward ? b.contrib <= 0 : b.contrib >= 0; }),
              all.end());
    return all;
}

Blame compute_blame(const ModelLayout& m, const ForwardResult& r, const Interventions& iv,
                    const std::vector<std::vector<float>>& dots, const std::vector<float>& u, int target, int pos,
                    const std::vector<std::vector<float>>* odots) {
    if (r.final_resid.empty() || r.embed.empty()) throw std::runtime_error("blame needs a capture_resid run");
    if (pos < 0 || pos >= r.n_tokens) throw std::out_of_range("position out of range");
    if (!r.captured(pos) || !r.has_logits(pos)) throw std::out_of_range("position " + std::to_string(pos) + " is outside the run's capture window");
    const size_t E = static_cast<size_t>(m.n_embd), F = static_cast<size_t>(m.n_ff);
    const size_t p = static_cast<size_t>(pos);

    Blame b;
    b.pos = pos;
    b.target = target;

    // Actual logit / prob / rank from the model's own output.
    const float* lg = r.logits_at(pos);
    {
        const float lt = lg[target];
        const float mx = *std::max_element(lg, lg + r.n_vocab);
        double z = 0;
        int above = 0;
        for (int v = 0; v < r.n_vocab; ++v) {
            z += std::exp(static_cast<double>(lg[v] - mx));
            if (lg[v] > lt) ++above;
        }
        b.prob = static_cast<float>(std::exp(static_cast<double>(lt - mx)) / z);
        b.rank = above + 1;
        // Soft-capped head: undo c·tanh(z/c) so the parts below sum to z.
        b.logit = lt;
        if (m.softcap > 0) {
            const double t = std::clamp(static_cast<double>(lt) / m.softcap, -1.0 + 1e-7, 1.0 - 1e-7);
            b.logit = static_cast<float>(m.softcap * std::atanh(t));
        }
    }
    const std::vector<float> S = later_scales(m);

    // 1/rms of the final residual at this position (llama rms_norm: x / sqrt(mean(x²) + eps)).
    const float* xf = r.final_at(pos);
    double ss = 0;
    for (size_t e = 0; e < E; ++e) ss += static_cast<double>(xf[e]) * xf[e];
    b.rms_inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(E) + m.rms_eps));

    auto dot_u = [&](const float* v) {
        double d = 0;
        for (size_t e = 0; e < E; ++e) d += static_cast<double>(v[e]) * u[e];
        return static_cast<float>(d * b.rms_inv);
    };

    b.embed = dot_u(r.embed_at(pos)) * S[0];

    for (int l = 0; l < m.n_layer; ++l) {
        const float* ao = r.attn_at(l, pos);
        if (!ao) continue;
        BlameItem a{BlameItem::Attn, l, -1, 0, 0, dot_u(ao) * S[static_cast<size_t>(l)]};
        const double ra = rms_inv_of(r.row(r.attn_raw, l, pos), E, m.rms_eps);   // frozen post-norm (Gemma), else 1
        b.sum_attn += a.contrib;
        b.attn.push_back(a);
        // Per head: w[h-block] · x[h-block] / rms.
        const float* x = r.heads_at(l, pos);
        if (!x || !odots || (*odots)[static_cast<size_t>(l)].empty()) continue;
        const auto& w = (*odots)[static_cast<size_t>(l)];
        const int H = n_heads(m, l), hw = head_width(m, l);
        if (static_cast<size_t>(H) * static_cast<size_t>(hw) != w.size()) continue;   // unexpected shape: skip, don't guess
        for (int h = 0; h < H; ++h) {
            double d = 0;
            for (int i = h * hw; i < (h + 1) * hw; ++i) d += static_cast<double>(w[static_cast<size_t>(i)]) * x[i];
            BlameItem hi{BlameItem::Head, l, h, 0, 0, static_cast<float>(d * ra * b.rms_inv)};
            b.head_gap -= hi.contrib;
            b.heads.push_back(hi);
        }
        b.head_gap += a.contrib;
    }

    b.neurons.reserve(static_cast<size_t>(m.n_layer) * F);
    for (int l = 0; l < m.n_layer; ++l) {
        const float* a = r.act(l, pos);
        if (!a || dots[static_cast<size_t>(l)].empty()) continue;
        const float rf = static_cast<float>(rms_inv_of(r.row(r.mlp_raw, l, pos), E, m.rms_eps));   // frozen post-norm, else 1
        const auto ivl = iv.layers.find(l);
        for (size_t n = 0; n < F; ++n) {
            float act = a[n];
            if (ivl != iv.layers.end()) {
                // What was actually written: a·scale + add (+ add_prompt while position p is in the prompt).
                act = act * ivl->second.scale[n] + ivl->second.add[n];
                if (static_cast<int>(p) < iv.prompt_len) act += ivl->second.add_prompt[n];
            }
            const float d = dots[static_cast<size_t>(l)][n];
            BlameItem it2{BlameItem::Neuron, l, static_cast<int>(n), act, d * rf, act * d * rf * b.rms_inv};
            b.sum_neurons += it2.contrib;
            b.neurons.push_back(it2);
        }
    }
    b.unexplained = b.logit - (b.embed + b.sum_attn + b.sum_neurons);
    return b;
}

}  // namespace neuronaut
