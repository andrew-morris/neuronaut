// test_probe_index.cpp — the contrast probe (probe.h) and the corpus examples index
// (actindex.h) against brute force, on the synthetic fixture (token ids fed
// directly; CI-safe).
//
//   test_probe_index <fixture.gguf> <scratch-dir>
//
//   probe   every hit's margin and means equal a brute-force recomputation from
//           full-capture runs, and per layer the hits are the k largest margins.
//   index   for every neuron of a layer, both poles: entries are sorted, at most one
//           per document, each entry's act is that document's true peak at its pos
//           (BOS skipped), and the first entry is the corpus-wide best document;
//           save → load reproduces every entry.
//   score   auc / cohens_d on hand-checked cases; every reading (peak, last, mean,
//           both poles, with its position) equals a brute-force recomputation from
//           full-capture runs over the read positions; a scored neuron's rank
//           counts match the layer's brute-force AUCs; top_by_auc is ordered; the
//           shuffled-label null is in range and reproducible from its seed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "actindex.h"
#include "forward.h"
#include "gguf_file.h"
#include "model_layout.h"
#include "probe.h"
#include "score.h"

using namespace neuronaut;

namespace {

int failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++failures;                                               \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
        }                                                             \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_probe_index <fixture.gguf> <scratch-dir>\n");
        return 2;
    }
    auto f = GgufFile::open(argv[1]);
    const ModelLayout m = ModelLayout::from(*f);
    Forward fw(*f, m);
    const size_t F = static_cast<size_t>(m.n_ff);

    // ---- probe --------------------------------------------------------------------
    {
        const std::vector<std::vector<int32_t>> pos = {{1, 5, 9, 3}, {1, 7, 2, 11, 4}}, neg = {{1, 5, 9, 8}, {1, 6, 2}};
        const int k = 5;
        const auto hits = contrast_probe(fw, m, pos, neg, {}, k);
        CHECK(static_cast<int>(hits.size()) == k * m.n_layer, "probe returned %zu hits, want %d", hits.size(), k * m.n_layer);
        // Brute force: full-capture runs, last-position rows.
        RunOptions ro;
        ro.capture_all_layers = true;
        auto last = [&](const std::vector<int32_t>& t) { return fw.run(t, {}, ro); };
        std::vector<ForwardResult> P, Q;
        for (const auto& t : pos) P.push_back(last(t));
        for (const auto& t : neg) Q.push_back(last(t));
        auto a = [&](const ForwardResult& r, int l, int n) { return r.act(l, r.n_tokens - 1)[n]; };
        for (int l = 0; l < m.n_layer; ++l) {
            std::vector<float> margins;
            for (size_t n = 0; n < F; ++n) {
                float pmin = 1e30f, pmax = -1e30f, qmin = 1e30f, qmax = -1e30f;
                for (const auto& r : P) pmin = std::min(pmin, a(r, l, static_cast<int>(n))), pmax = std::max(pmax, a(r, l, static_cast<int>(n)));
                for (const auto& r : Q) qmin = std::min(qmin, a(r, l, static_cast<int>(n))), qmax = std::max(qmax, a(r, l, static_cast<int>(n)));
                margins.push_back(pmin - qmax);
                margins.push_back(qmin - pmax);
            }
            std::sort(margins.rbegin(), margins.rend());
            for (const auto& h : hits) {
                if (h.layer != l) continue;
                float pmin = 1e30f, pmax = -1e30f, qmin = 1e30f, qmax = -1e30f;
                for (const auto& r : P) pmin = std::min(pmin, a(r, l, h.neuron)), pmax = std::max(pmax, a(r, l, h.neuron));
                for (const auto& r : Q) qmin = std::min(qmin, a(r, l, h.neuron)), qmax = std::max(qmax, a(r, l, h.neuron));
                const float want = h.pole == '+' ? pmin - qmax : qmin - pmax;
                CHECK(h.margin == want, "L%d.N%d%c margin %g, brute force %g", l, h.neuron, h.pole, h.margin, want);
                CHECK(h.margin >= margins[static_cast<size_t>(k - 1)], "L%d.N%d%c is not among the top %d margins", l, h.neuron, h.pole, k);
            }
        }
    }

    // ---- index --------------------------------------------------------------------
    {
        const int32_t BOS = 1;
        std::vector<std::vector<int32_t>> docs;
        for (int d = 0; d < 12; ++d) {
            std::vector<int32_t> t{BOS};
            for (int i = 0; i < 4 + d % 5; ++i) t.push_back((d * 7 + i * 3) % 31 + 1);
            docs.push_back(t);
        }
        const int k = 4;
        const ActIndex ix = ActIndex::build(fw, m, docs, k, true, BOS);
        RunOptions ro;
        ro.capture_all_layers = true;
        std::vector<ForwardResult> runs;
        for (const auto& d : docs) runs.push_back(fw.run(d, {}, ro));
        const int L = 1;
        for (int n = 0; n < m.n_ff; ++n) {
            for (char pole : {'+', '-'}) {
                const auto ex = ix.top(L, n, pole);
                CHECK(static_cast<int>(ex.size()) == k, "L%d.N%d%c has %zu examples, want %d", L, n, pole, ex.size(), k);
                std::set<uint32_t> seen;
                float best = pole == '+' ? -1e30f : 1e30f;
                for (size_t d = 0; d < docs.size(); ++d)
                    for (int t = 1; t < runs[d].n_tokens; ++t) {
                        const float v = runs[d].act(L, t)[n];
                        best = pole == '+' ? std::max(best, v) : std::min(best, v);
                    }
                for (size_t i = 0; i < ex.size(); ++i) {
                    const auto& e = ex[i];
                    CHECK(seen.insert(e.doc).second, "L%d.N%d%c lists doc %u twice", L, n, pole, e.doc);
                    CHECK(e.pos >= 1, "L%d.N%d%c example on the BOS position", L, n, pole);
                    CHECK(runs[e.doc].act(L, static_cast<int>(e.pos))[n] == e.act, "L%d.N%d%c entry act is not the run's value", L, n, pole);
                    if (i) CHECK(pole == '+' ? ex[i - 1].act >= e.act : ex[i - 1].act <= e.act, "L%d.N%d%c entries out of order", L, n, pole);
                }
                CHECK(ex[0].act == best, "L%d.N%d%c best example %g, brute force %g", L, n, pole, ex[0].act, best);
            }
        }
        const std::string path = (std::filesystem::path(argv[2]) / "fixture.acts-index").string();
        ix.save(path);
        const ActIndex back = ActIndex::load(path, m);
        CHECK(back.n_docs() == ix.n_docs() && back.k() == k && back.doc(3) == ix.doc(3), "round trip lost documents");
        for (int l = 0; l < m.n_layer; ++l)
            for (int n = 0; n < m.n_ff; n += 37)
                for (char pole : {'+', '-'}) {
                    const auto a1 = ix.top(l, n, pole), a2 = back.top(l, n, pole);
                    bool same = a1.size() == a2.size();
                    for (size_t i = 0; same && i < a1.size(); ++i) same = a1[i].act == a2[i].act && a1[i].doc == a2[i].doc && a1[i].pos == a2[i].pos;
                    CHECK(same, "round trip changed L%d.N%d%c", l, n, pole);
                }
    }

    // ---- score --------------------------------------------------------------------
    {
        CHECK(auc({3, 4}, {1, 2}) == 1.0, "auc separated");
        CHECK(auc({1, 2}, {3, 4}) == 0.0, "auc reversed");
        CHECK(auc({1, 3}, {2, 2}) == 0.5, "auc interleaved");
        CHECK(auc({2}, {2}) == 0.5, "auc tie counts half");
        CHECK(std::abs(auc({1, 2, 3}, {2}) - 0.5) < 1e-12, "auc with a tie in the middle");
        CHECK(std::isnan(auc({}, {1})), "auc of an empty side is NaN");
        // d: means 2 and 0, pooled sd sqrt((2 + 2) / 2) = sqrt 2.
        CHECK(std::abs(cohens_d({1, 3}, {-1, 1}) - 2 / std::sqrt(2.0)) < 1e-12, "cohens_d");
        CHECK(cohens_d({1, 1}, {1, 1}) == 0, "cohens_d of constants");
        CHECK(std::isnan(cohens_d({1}, {2})), "cohens_d of one text per side is NaN");

        const std::vector<ScoreText> P = {{{1, 5, 9, 3}, 1}, {{1, 7, 2, 11, 4}, 2}}, Q = {{{1, 5, 9, 8}, 1}, {{1, 6, 2}, 1}};
        std::vector<int> layers;
        for (int l = 0; l < m.n_layer; ++l) layers.push_back(l);
        auto full = [&](const ScoreText& t) {
            RunOptions ro;
            ro.capture_all_layers = true;
            return fw.run(t.ids, {}, ro);
        };
        for (ScoreRead rd : {ScoreRead::Peak, ScoreRead::Last, ScoreRead::Mean}) {
            const auto R = read_texts(fw, m, P, layers, rd);
            for (size_t ti = 0; ti < P.size(); ++ti) {
                const auto run = full(P[ti]);
                const int T = static_cast<int>(P[ti].ids.size());
                for (int l : layers)
                    for (size_t n = 0; n < F; ++n) {
                        float want_p = 0, want_m = 0;
                        int at_p = 0, at_m = 0;
                        if (rd == ScoreRead::Last) {
                            want_p = run.act(l, T - 1)[n], want_m = -want_p, at_p = at_m = T - 1;
                        } else if (rd == ScoreRead::Mean) {
                            double sum = 0;
                            for (int p = P[ti].from; p < T; ++p) sum += run.act(l, p)[n];
                            want_p = static_cast<float>(sum / (T - P[ti].from)), want_m = -want_p;
                        } else {
                            want_p = want_m = -INFINITY;
                            for (int p = P[ti].from; p < T; ++p) {
                                const float v = run.act(l, p)[n];
                                if (v > want_p) want_p = v, at_p = p;
                                if (-v > want_m) want_m = -v, at_m = p;
                            }
                        }
                        const LayerReads& lr = R[ti].at(l);
                        CHECK(std::abs(lr.plus[n] - want_p) <= 1e-5f * (1 + std::abs(want_p)) && std::abs(lr.minus[n] - want_m) <= 1e-5f * (1 + std::abs(want_m)),
                              "%s reading of text %zu L%d.N%zu: %g/%g, brute force %g/%g", score_read_name(rd), ti, l, n, lr.plus[n], lr.minus[n], want_p, want_m);
                        if (rd != ScoreRead::Mean)
                            CHECK(lr.plus_at[n] == at_p && lr.minus_at[n] == at_m, "%s position of text %zu L%d.N%zu", score_read_name(rd), ti, l, n);
                    }
            }
        }
        const auto RP = read_texts(fw, m, P, layers, ScoreRead::Peak), RQ = read_texts(fw, m, Q, layers, ScoreRead::Peak);
        const auto sc = score_neurons(m, RP, RQ, {{1, 3, '+'}, {1, 3, 0}});
        CHECK(sc.size() == 2 && sc[0].pos.size() == 2 && sc[0].neg.size() == 2, "score_neurons shape");
        {
            int better = 0, at_least = 0;
            for (size_t n = 0; n < F; ++n)
                for (int s = 0; s < 2; ++s) {
                    std::vector<float> a, b;
                    for (const auto& t : RP) a.push_back(s ? t.at(1).minus[n] : t.at(1).plus[n]);
                    for (const auto& t : RQ) b.push_back(s ? t.at(1).minus[n] : t.at(1).plus[n]);
                    const double x = auc(a, b);
                    better += x > sc[0].auc, at_least += x >= sc[0].auc;
                }
            CHECK(sc[0].n_better == better && sc[0].n_at_least == at_least && sc[0].n_candidates == 2 * m.n_ff,
                  "rank counts %d/%d, brute force %d/%d", sc[0].n_better, sc[0].n_at_least, better, at_least);
            CHECK(sc[1].auc >= 1 - sc[1].auc - 1e-12 || sc[1].pole == '+', "pole 0 picks the better pole");
        }
        const auto top = top_by_auc(m, RP, RQ, layers, 3);
        CHECK(static_cast<int>(top.size()) == 3 * m.n_layer, "top_by_auc returned %zu, want %d", top.size(), 3 * m.n_layer);
        for (size_t i = 1; i < top.size(); ++i)
            if (top[i].layer == top[i - 1].layer) CHECK(top[i].auc <= top[i - 1].auc, "top_by_auc out of order in L%d", top[i].layer);
        for (const auto& h : top) CHECK(h.n_better == 0 || h.auc < 1, "a top hit with AUC 1 has nothing better");
        const auto n1 = null_best(m, RP, RQ, layers, 5, 7), n2 = null_best(m, RP, RQ, layers, 5, 7);
        CHECK(n1.size() == layers.size() && n1.at(0).size() == 5, "null shape");
        for (int l : layers)
            for (size_t i = 0; i < 5; ++i) {
                CHECK(n1.at(l)[i].auc == n2.at(l)[i].auc && n1.at(l)[i].d == n2.at(l)[i].d, "null not reproducible");
                CHECK(n1.at(l)[i].auc <= 1 && n1.at(l)[i].auc >= 0.5, "null AUC %g out of range", n1.at(l)[i].auc);
            }
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
