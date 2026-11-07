// probe.cpp — see probe.h.
#include "probe.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace neuronaut {

std::vector<std::vector<std::vector<float>>> last_token_acts(const Forward& fw, const ModelLayout& m,
                                                             const std::vector<std::vector<int32_t>>& texts,
                                                             const std::vector<int>& layers, const Interventions& iv) {
    std::vector<int> ls = layers;
    if (ls.empty())
        for (int l = 0; l < m.n_layer; ++l) ls.push_back(l);
    for (int l : ls)
        if (!m.has_ffn(l)) throw std::out_of_range("probe: layer " + std::to_string(l) + " has no MLP");
    std::vector<std::vector<std::vector<float>>> out;
    for (const auto& t : texts) {
        if (t.empty()) throw std::invalid_argument("probe: empty text");
        RunOptions ro;
        ro.capture_layers = ls;
        ro.cap_lo = static_cast<int>(t.size()) - 1;   // the last token only
        const ForwardResult r = fw.run(t, iv, ro);
        std::vector<std::vector<float>> per(static_cast<size_t>(m.n_layer));
        for (int l : ls) per[static_cast<size_t>(l)] = r.acts.at(l);   // exactly one row: n_ff floats
        out.push_back(std::move(per));
    }
    return out;
}

std::vector<ProbeHit> contrast_probe(const Forward& fw, const ModelLayout& m, const std::vector<std::vector<int32_t>>& pos,
                                     const std::vector<std::vector<int32_t>>& neg, const std::vector<int>& layers, int k,
                                     const Interventions& iv) {
    if (pos.empty() || neg.empty()) throw std::invalid_argument("probe: needs at least one POS and one NEG text");
    std::vector<int> ls = layers;
    if (ls.empty())
        for (int l = 0; l < m.n_layer; ++l) ls.push_back(l);
    const auto P = last_token_acts(fw, m, pos, ls, iv);
    const auto Q = last_token_acts(fw, m, neg, ls, iv);
    const size_t F = static_cast<size_t>(m.n_ff);
    constexpr float inf = std::numeric_limits<float>::infinity();

    std::vector<ProbeHit> out;
    for (int l : ls) {
        std::vector<ProbeHit> cand;
        cand.reserve(2 * F);
        for (size_t n = 0; n < F; ++n) {
            float pmin = inf, pmax = -inf, qmin = inf, qmax = -inf;
            double psum = 0, qsum = 0;
            for (const auto& t : P) {
                const float a = t[static_cast<size_t>(l)][n];
                pmin = std::min(pmin, a), pmax = std::max(pmax, a), psum += a;
            }
            for (const auto& t : Q) {
                const float a = t[static_cast<size_t>(l)][n];
                qmin = std::min(qmin, a), qmax = std::max(qmax, a), qsum += a;
            }
            const float pm = static_cast<float>(psum / static_cast<double>(P.size())), qm = static_cast<float>(qsum / static_cast<double>(Q.size()));
            cand.push_back({l, static_cast<int>(n), '+', pmin - qmax, pm, qm});
            cand.push_back({l, static_cast<int>(n), '-', qmin - pmax, pm, qm});
        }
        const size_t kk = std::min<size_t>(static_cast<size_t>(std::max(0, k)), cand.size());
        // Ties broken like contrast_scan.py's sort(reverse=True) on (margin, neuron, pole).
        std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(kk), cand.end(), [](const ProbeHit& a, const ProbeHit& b) {
            if (a.margin != b.margin) return a.margin > b.margin;
            if (a.neuron != b.neuron) return a.neuron > b.neuron;
            return a.pole > b.pole;   // '-' (45) > '+' (43): reverse order puts '-' first
        });
        out.insert(out.end(), cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(kk));
    }
    return out;
}

}  // namespace neuronaut
