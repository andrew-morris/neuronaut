// trace.cpp — see trace.h.
#include "trace.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace neuronaut {

float Trace::restored(int layer, int pos) const {
    const float v = at(layer, pos);
    const float span = p_clean - p_corrupt;
    if (std::isnan(v) || std::fabs(span) < 1e-6f) return std::numeric_limits<float>::quiet_NaN();
    return (v - p_corrupt) / span;
}

std::pair<int, int> Trace::best() const {
    std::pair<int, int> b{-1, -1};
    float bv = -std::numeric_limits<float>::infinity();
    for (int l = 0; l < n_layer; ++l)
        for (int t = 0; t < n_tokens; ++t) {
            const float r = restored(l, t);
            if (!std::isnan(r) && r > bv) {
                bv = r;
                b = {l, t};
            }
        }
    return b;
}

Trace causal_trace(const Forward& fw, const ModelLayout& m, const std::vector<int32_t>& clean, const std::vector<int32_t>& corrupt,
                   const Interventions& iv, Site site, int target, const std::function<void(const Trace&)>& progress,
                   const std::atomic<bool>* cancel) {
    if (clean.empty() || clean.size() != corrupt.size()) {
        throw std::invalid_argument("trace: the clean and corrupt prompts must have the same number of tokens (" + std::to_string(clean.size()) +
                                    " vs " + std::to_string(corrupt.size()) + ") — pick a corruption that tokenizes to the same length");
    }
    const int T = static_cast<int>(clean.size());
    const size_t E = static_cast<size_t>(m.n_embd);
    Trace tr;
    tr.site = site;
    tr.n_layer = m.n_layer;
    tr.n_tokens = T;
    tr.p.assign(static_cast<size_t>(m.n_layer) * static_cast<size_t>(T), std::numeric_limits<float>::quiet_NaN());
    tr.total = m.n_layer * T;
    for (int t = 0; t < T; ++t)
        if (clean[static_cast<size_t>(t)] != corrupt[static_cast<size_t>(t)]) tr.differ.push_back(t);

    // Both runs, with everything a patch could need: the clean pieces to copy, the
    // corrupt residuals to start each patched run from.
    RunOptions ro;
    ro.capture_resid = true;          // embed + attn_out
    ro.capture_layer_resid = true;
    ro.capture_all_layers = site == Site::Mlp;
    const ForwardResult rc = fw.run(clean, iv, ro);
    const ForwardResult rx = fw.run(corrupt, iv, ro);
    if (target < 0) target = top_probs(rc.logits.data(), rc.n_vocab, 1)[0].first;
    tr.target = target;
    tr.p_clean = prob_of(rc.logits.data(), rc.n_vocab, target);
    tr.p_corrupt = prob_of(rx.logits.data(), rx.n_vocab, target);
    if (progress) progress(tr);

    // The residual entering layer l of the corrupt run.
    auto corrupt_in = [&](int l) -> const std::vector<float>& { return l == 0 ? rx.embed : rx.layer_resid.at(l - 1); };

    for (int l = 0; l < m.n_layer; ++l) {
        for (int t = 0; t < T; ++t) {
            if (cancel && *cancel) return tr;
            ForwardResult r;
            if (site == Site::Resid) {
                // The whole stream after layer l at t from the clean run: just start
                // layer l+1 from the corrupt residual with that row swapped in.
                std::vector<float> x = rx.layer_resid.at(l);
                const float* src = rc.resid_after(l, t);
                std::copy(src, src + E, x.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(t) * E));
                r = fw.run_from(l + 1, x, T, iv);
            } else {
                Interventions pv = iv;
                const float* src = site == Site::Attn ? rc.attn_at(l, t) : rc.act(l, t);
                const size_t w = site == Site::Attn ? E : static_cast<size_t>(m.n_ff);
                pv.override_at(m, site, l, t, std::vector<float>(src, src + w));
                r = fw.run_from(l, corrupt_in(l), T, pv);
            }
            tr.p[static_cast<size_t>(l) * static_cast<size_t>(T) + static_cast<size_t>(t)] = prob_of(r.logits.data(), r.n_vocab, target);
            ++tr.done;
            if (progress) progress(tr);
        }
    }
    return tr;
}

}  // namespace neuronaut
