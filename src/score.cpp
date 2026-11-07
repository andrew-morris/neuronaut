// score.cpp — see score.h.
#include "score.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include "format.h"

namespace neuronaut {

bool parse_score_read(const std::string& s, ScoreRead& out) {
    if (s == "peak") out = ScoreRead::Peak;
    else if (s == "last") out = ScoreRead::Last;
    else if (s == "mean") out = ScoreRead::Mean;
    else return false;
    return true;
}

const char* score_read_name(ScoreRead r) {
    switch (r) {
        case ScoreRead::Peak: return "peak";
        case ScoreRead::Last: return "last";
        case ScoreRead::Mean: return "mean";
    }
    return "?";
}

ScoreText score_text(const Encode& encode, int32_t bos, const std::string& arch, const std::string& text, bool chat) {
    ScoreText st;
    if (chat) {
        const size_t tab = text.find('\t');
        const std::string prompt = chat_wrap(arch, text.substr(0, tab), false);
        st.ids = encode(prompt + (tab == std::string::npos ? "" : text.substr(tab + 1)), true, true);
        if (tab != std::string::npos) {
            const auto head = encode(prompt, true, true);
            if (head.size() >= st.ids.size() || !std::equal(head.begin(), head.end(), st.ids.begin()))
                throw std::runtime_error("score: the reply does not start on a token boundary: " + quote_printable(text, 80));
            st.from = static_cast<int>(head.size());
        }
    } else {
        st.ids = encode(text, true, false);
    }
    if (st.from == 0 && !st.ids.empty() && st.ids[0] == bos) st.from = 1;
    if (st.from >= static_cast<int>(st.ids.size())) throw std::runtime_error("score: nothing to read in " + quote_printable(text, 80));
    return st;
}

TextReads read_texts(const Forward& fw, const ModelLayout& m, const std::vector<ScoreText>& texts, const std::vector<int>& layers,
                     ScoreRead read, const Interventions& iv) {
    if (layers.empty()) throw std::invalid_argument("score: no layers to read");
    for (int l : layers)
        if (l < 0 || l >= m.n_layer || !m.has_ffn(l)) throw std::out_of_range("score: layer " + std::to_string(l) + " has no MLP");
    const int last_layer = *std::max_element(layers.begin(), layers.end());
    const size_t F = static_cast<size_t>(m.n_ff);
    TextReads out;
    out.reserve(texts.size());
    for (const auto& t : texts) {
        const int T = static_cast<int>(t.ids.size());
        if (t.from < 0 || t.from >= T) throw std::invalid_argument("score: a text has no tokens to read");
        RunOptions ro;
        ro.capture_layers = layers;
        // Only the read positions are copied out; `last` needs just the final one.
        ro.cap_lo = read == ScoreRead::Last ? T - 1 : t.from;
        ro.last_layer = last_layer;   // nothing after the deepest read layer matters
        const ForwardResult r = fw.run(t.ids, iv, ro);
        std::map<int, LayerReads> per;
        for (int l : layers) {
            const auto& a = r.acts.at(l);   // cap_n × n_ff, position-major
            const int n_pos = r.cap_n;
            LayerReads lr;
            lr.plus.assign(F, 0), lr.minus.assign(F, 0), lr.plus_at.assign(F, r.cap_lo), lr.minus_at.assign(F, r.cap_lo);
            if (read == ScoreRead::Peak) {
                std::fill(lr.plus.begin(), lr.plus.end(), -std::numeric_limits<float>::infinity());
                std::fill(lr.minus.begin(), lr.minus.end(), -std::numeric_limits<float>::infinity());
            }
            for (int p = 0; p < n_pos; ++p) {
                const float* row = a.data() + static_cast<size_t>(p) * F;
                for (size_t n = 0; n < F; ++n) {
                    const float v = row[n];
                    if (read == ScoreRead::Peak) {
                        if (v > lr.plus[n]) lr.plus[n] = v, lr.plus_at[n] = r.cap_lo + p;
                        if (-v > lr.minus[n]) lr.minus[n] = -v, lr.minus_at[n] = r.cap_lo + p;
                    } else {
                        lr.plus[n] += v;   // Last: one position; Mean: summed, divided below
                    }
                }
            }
            if (read != ScoreRead::Peak) {
                const float inv = 1.0f / static_cast<float>(n_pos);
                for (size_t n = 0; n < F; ++n) lr.plus[n] *= inv, lr.minus[n] = -lr.plus[n];
                if (read == ScoreRead::Last) std::fill(lr.plus_at.begin(), lr.plus_at.end(), T - 1), lr.minus_at = lr.plus_at;
            }
            per[l] = std::move(lr);
        }
        out.push_back(std::move(per));
    }
    return out;
}

double auc(const std::vector<float>& pos, const std::vector<float>& neg) {
    if (pos.empty() || neg.empty()) return std::numeric_limits<double>::quiet_NaN();
    // Mann–Whitney U from ranks of the pooled sample, ties given their mean rank.
    struct V { float v; bool p; };
    std::vector<V> all;
    all.reserve(pos.size() + neg.size());
    for (float v : pos) all.push_back({v, true});
    for (float v : neg) all.push_back({v, false});
    std::sort(all.begin(), all.end(), [](const V& a, const V& b) { return a.v < b.v; });
    double rank_sum = 0;
    for (size_t i = 0; i < all.size();) {
        size_t j = i;
        while (j < all.size() && all[j].v == all[i].v) ++j;
        const double mean_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2;   // ranks i+1 … j
        for (size_t q = i; q < j; ++q)
            if (all[q].p) rank_sum += mean_rank;
        i = j;
    }
    const double np = static_cast<double>(pos.size()), nn = static_cast<double>(neg.size());
    return (rank_sum - np * (np + 1) / 2) / (np * nn);
}

double cohens_d(const std::vector<float>& pos, const std::vector<float>& neg) {
    auto mean_var = [](const std::vector<float>& x, double& mu, double& ss) {
        mu = 0, ss = 0;
        for (float v : x) mu += v;
        mu /= static_cast<double>(x.size());
        for (float v : x) ss += (v - mu) * (v - mu);
    };
    if (pos.empty() || neg.empty()) return std::numeric_limits<double>::quiet_NaN();
    double mp, sp, mn, sn;
    mean_var(pos, mp, sp), mean_var(neg, mn, sn);
    const double dof = static_cast<double>(pos.size() + neg.size()) - 2;
    if (dof <= 0) return std::numeric_limits<double>::quiet_NaN();   // one text per side: no spread to compare with
    const double sd = std::sqrt((sp + sn) / dof);
    if (sd == 0) return mp == mn ? 0 : std::copysign(std::numeric_limits<double>::infinity(), mp - mn);
    return (mp - mn) / sd;
}

namespace {

using LayerSet = std::vector<const LayerReads*>;   // one layer's readings, one entry per text

LayerSet layer_set(const TextReads& tr, int layer) {
    LayerSet out;
    for (const auto& t : tr) out.push_back(&t.at(layer));
    return out;
}

// One (neuron, pole)'s readings across a set of texts.
void column(const LayerSet& ls, int neuron, char pole, std::vector<float>& v, std::vector<int>* at = nullptr) {
    v.clear();
    if (at) at->clear();
    const size_t n = static_cast<size_t>(neuron);
    for (const LayerReads* lr : ls) {
        v.push_back(pole == '+' ? lr->plus[n] : lr->minus[n]);
        if (at) at->push_back(pole == '+' ? lr->plus_at[n] : lr->minus_at[n]);
    }
}

// AUC of every (neuron, pole) of a layer: [2n] = '+', [2n+1] = '−'.
std::vector<double> layer_aucs(const ModelLayout& m, const LayerSet& pos, const LayerSet& neg) {
    std::vector<double> out(2 * static_cast<size_t>(m.n_ff));
    std::vector<float> p, q;
    for (int n = 0; n < m.n_ff; ++n)
        for (int s = 0; s < 2; ++s) {
            const char pole = s == 0 ? '+' : '-';
            column(pos, n, pole, p), column(neg, n, pole, q);
            out[2 * static_cast<size_t>(n) + static_cast<size_t>(s)] = auc(p, q);
        }
    return out;
}

NeuronScore make_score(const TextReads& pos, const TextReads& neg, int layer, int neuron, char pole, const std::vector<double>& aucs) {
    NeuronScore s;
    s.layer = layer, s.neuron = neuron, s.pole = pole;
    column(layer_set(pos, layer), neuron, pole, s.pos, &s.pos_at);
    column(layer_set(neg, layer), neuron, pole, s.neg, &s.neg_at);
    s.auc = auc(s.pos, s.neg);
    s.d = cohens_d(s.pos, s.neg);
    s.n_candidates = static_cast<int>(aucs.size());
    for (double a : aucs) {
        if (a > s.auc) ++s.n_better;
        if (a >= s.auc) ++s.n_at_least;
    }
    return s;
}

}  // namespace

std::vector<NeuronScore> score_neurons(const ModelLayout& m, const TextReads& pos, const TextReads& neg, const std::vector<NeuronRef>& refs) {
    if (pos.empty() || neg.empty()) throw std::invalid_argument("score: needs at least one POS and one NEG text");
    std::map<int, std::vector<double>> aucs;
    std::vector<NeuronScore> out;
    for (const auto& r : refs) {
        if (r.neuron < 0 || r.neuron >= m.n_ff) throw std::out_of_range("score: neuron " + std::to_string(r.neuron) + " out of range");
        auto it = aucs.find(r.layer);
        if (it == aucs.end()) it = aucs.emplace(r.layer, layer_aucs(m, layer_set(pos, r.layer), layer_set(neg, r.layer))).first;
        char pole = r.pole;
        if (pole == 0) {
            const size_t i = 2 * static_cast<size_t>(r.neuron);
            pole = it->second[i] >= it->second[i + 1] ? '+' : '-';
        }
        out.push_back(make_score(pos, neg, r.layer, r.neuron, pole, it->second));
    }
    return out;
}

std::vector<NeuronScore> top_by_auc(const ModelLayout& m, const TextReads& pos, const TextReads& neg, const std::vector<int>& layers, int k) {
    if (pos.empty() || neg.empty()) throw std::invalid_argument("score: needs at least one POS and one NEG text");
    std::vector<NeuronScore> out;
    for (int l : layers) {
        const auto aucs = layer_aucs(m, layer_set(pos, l), layer_set(neg, l));
        // Candidates by AUC; ties (common at AUC 1) broken by d, which needs the
        // readings, so only the tied front is scored in full.
        std::vector<size_t> idx(aucs.size());
        std::iota(idx.begin(), idx.end(), size_t{0});
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return aucs[a] != aucs[b] ? aucs[a] > aucs[b] : a < b; });
        const size_t kk = std::min(idx.size(), static_cast<size_t>(std::max(0, k)));
        if (kk == 0) continue;
        // Everything at least as good as the k-th competes on d.
        size_t front = kk;
        while (front < idx.size() && aucs[idx[front]] == aucs[idx[kk - 1]]) ++front;
        front = std::min(front, size_t{4096});
        std::vector<NeuronScore> cand;
        for (size_t i = 0; i < front; ++i)
            cand.push_back(make_score(pos, neg, l, static_cast<int>(idx[i] / 2), idx[i] % 2 ? '-' : '+', aucs));
        std::stable_sort(cand.begin(), cand.end(), [](const NeuronScore& a, const NeuronScore& b) { return a.auc != b.auc ? a.auc > b.auc : a.d > b.d; });
        out.insert(out.end(), cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(kk));
    }
    return out;
}

std::map<int, std::vector<NullBest>> null_best(const ModelLayout& m, const TextReads& pos, const TextReads& neg, const std::vector<int>& layers,
                                               int n, uint64_t seed) {
    std::map<int, std::vector<NullBest>> out;
    std::mt19937_64 rng(seed);
    std::vector<size_t> order(pos.size() + neg.size());
    std::vector<std::vector<size_t>> perms;
    for (int i = 0; i < n; ++i) {
        std::iota(order.begin(), order.end(), size_t{0});
        std::shuffle(order.begin(), order.end(), rng);
        perms.push_back(order);
    }
    std::vector<float> a, b;
    for (int l : layers) {
        LayerSet all = layer_set(pos, l);
        for (const LayerReads* r : layer_set(neg, l)) all.push_back(r);
        for (const auto& pm : perms) {
            LayerSet p, q;
            for (size_t i = 0; i < pm.size(); ++i) (i < pos.size() ? p : q).push_back(all[pm[i]]);
            NullBest best;
            for (int nn = 0; nn < m.n_ff; ++nn)
                for (char pole : {'+', '-'}) {
                    column(p, nn, pole, a), column(q, nn, pole, b);
                    best.auc = std::max(best.auc, auc(a, b));
                    const double d = cohens_d(a, b);
                    if (std::isfinite(d)) best.d = std::max(best.d, d);
                }
            out[l].push_back(best);
        }
    }
    return out;
}

}  // namespace neuronaut
