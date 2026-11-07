// test_forward.cpp — engine invariants of the forward pass, on the synthetic fixture
// (no real weights, CI-safe; token ids are fed directly, no tokenizer).
//
//   test_forward <fixture.gguf>
//
// Each section pins one promise forward.h makes, bit for bit, so the features built
// on it (the TUI's windowed reruns, dose sweeps from a layer, activation patching,
// per-head blame) can rely on it without re-checking:
//
//   window   a narrow capture window holds exactly the rows the full capture holds,
//            and logits/watches are unaffected by the window.
//   from     run_from(L, resid entering L) == run() for every L (0 … n_layer), with
//            interventions at and below L — what the dose sweep relies on.
//   override activation-patching overrides: writing a run's own value back is a
//            no-op bit for bit; a Resid override equals starting the next layer from
//            the swapped residual (how trace.cpp patches resid); an Mlp/Attn override
//            changes the output from that position on, and never before it.
//   rank-one a rank-one down-projection edit with u = 0 is a no-op bit for bit; one
//            with u ≠ 0 equals adding u·(w·a) to the residual after its layer.
//   delta    rome's value search: at δ = 0 its NLL is the forward pass's own; its
//            gradient matches central finite differences along random directions.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "forward.h"
#include "gguf_file.h"
#include "model_layout.h"

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

// Bit-identical rows of `w` floats.
bool same(const float* a, const float* b, size_t w) { return a && b && std::memcmp(a, b, w * sizeof(float)) == 0; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_forward <fixture.gguf>\n");
        return 2;
    }
    auto f = GgufFile::open(argv[1]);
    const ModelLayout m = ModelLayout::from(*f);
    Forward fw(*f, m);
    const int V = fw.n_vocab();
    const size_t E = static_cast<size_t>(m.n_embd), F = static_cast<size_t>(m.n_ff);

    const std::vector<int32_t> toks = {1, 5, 9, 3, 7, 2, 11, 4};
    const int T = static_cast<int>(toks.size());
    Interventions iv;
    iv.scale(m, 1, 37, 3.0f);   // something in the graph, so captures are pre-edit values

    // ---- window ------------------------------------------------------------------
    RunOptions full;
    full.capture_all_layers = full.capture_resid = full.capture_layer_resid = full.all_logits = true;
    full.watch = {{0, 3}, {1, 37}};
    const ForwardResult a = fw.run(toks, iv, full);
    CHECK(a.cap_lo == 0 && a.cap_n == T && a.logit_n == T, "full capture window wrong");

    RunOptions win = full;
    win.cap_lo = 2;
    win.cap_hi = 5;
    const ForwardResult b = fw.run(toks, iv, win);
    CHECK(b.cap_lo == 2 && b.cap_n == 3 && b.logit_lo == 2 && b.logit_n == 3, "window bounds: %d+%d, logits %d+%d", b.cap_lo, b.cap_n,
          b.logit_lo, b.logit_n);
    CHECK(!b.captured(1) && !b.captured(5) && b.act(0, 1) == nullptr && b.logits_at(5) == nullptr, "outside the window is not nullptr");
    for (int p = 2; p < 5; ++p) {
        CHECK(same(a.logits_at(p), b.logits_at(p), static_cast<size_t>(V)), "windowed logits differ at %d", p);
        CHECK(same(a.embed_at(p), b.embed_at(p), E) && same(a.final_at(p), b.final_at(p), E), "windowed embed/final differ at %d", p);
        for (int l = 0; l < m.n_layer; ++l) {
            CHECK(same(a.act(l, p), b.act(l, p), F), "windowed acts differ at L%d pos %d", l, p);
            CHECK(same(a.attn_at(l, p), b.attn_at(l, p), E), "windowed attn_out differs at L%d pos %d", l, p);
            CHECK(same(a.resid_after(l, p), b.resid_after(l, p), E), "windowed resid differs at L%d pos %d", l, p);
        }
    }
    // Watches cover every position whatever the window, and equal the acts column.
    for (const auto& [ln, w] : b.watched) {
        CHECK(static_cast<int>(w.size()) == T, "watch has %zu positions, want %d", w.size(), T);
        for (int p = 0; p < T; ++p)
            CHECK(w[static_cast<size_t>(p)] == a.act(ln.first, p)[static_cast<size_t>(ln.second)], "watch L%d.N%d differs at %d", ln.first, ln.second, p);
    }
    // The default (last-position) logits don't move with a window either.
    {
        RunOptions last;
        last.cap_lo = 0;
        last.cap_hi = 1;
        last.capture_layers = {0};
        const ForwardResult c = fw.run(toks, iv, last), d = fw.run(toks, iv);
        CHECK(c.logit_lo == T - 1 && same(c.logits.data(), d.logits.data(), static_cast<size_t>(V)), "last-position logits changed under a window");
        // (Not compared with the all-logits run: a 1-row and a T-row matmul round
        // differently on F32 — that was true before windows existed.)
    }

    // ---- from -------------------------------------------------------------------
    for (int L = 0; L <= m.n_layer; ++L) {
        Interventions ivL = iv;                      // L1.N37 ×3 from above …
        if (L < m.n_layer) ivL.add(m, L, 5, 2.0f);   // … plus an edit at L itself
        const ForwardResult ref = fw.run(toks, ivL, full);
        const std::vector<float>& x_in = L == 0 ? ref.embed : ref.layer_resid.at(L - 1);
        const ForwardResult got = fw.run_from(L, x_in, T, ivL, full);
        CHECK(got.logits == ref.logits, "run_from(%d) logits differ from run()", L);
        for (int l = L; l < m.n_layer; ++l) {
            CHECK(got.acts.at(l) == ref.acts.at(l), "run_from(%d) acts differ at L%d", L, l);
            CHECK(got.layer_resid.at(l) == ref.layer_resid.at(l), "run_from(%d) resid differs at L%d", L, l);
        }
        CHECK(got.acts.count(L - 1) == 0, "run_from(%d) captured a layer it skipped", L);
        CHECK(L != 0 || got.embed == ref.embed, "run_from(0) embed is not resid_in");
    }

    // ---- override -------------------------------------------------------------
    {
        const ForwardResult ref = fw.run(toks, iv, full);
        const int L = 0, P = 3;
        for (Site site : {Site::Resid, Site::Attn, Site::Mlp}) {
            // Its own value back: identical.
            const float* own = site == Site::Resid ? ref.resid_after(L, P) : site == Site::Attn ? ref.attn_at(L, P) : ref.act(L, P);
            const size_t w = site == Site::Mlp ? F : E;
            Interventions same_v = iv;
            same_v.override_at(m, site, L, P, std::vector<float>(own, own + w));
            const ForwardResult r1 = fw.run(toks, same_v, full);
            CHECK(r1.logits == ref.logits, "%s override with its own value changed the logits", site_name(site));
            // A different value: changes positions >= P only (causality).
            std::vector<float> other(w);
            for (size_t i = 0; i < w; ++i) other[i] = own[i] + 0.5f * static_cast<float>(static_cast<int>(i % 7) - 3);
            Interventions diff = iv;
            diff.override_at(m, site, L, P, other);
            const ForwardResult r2 = fw.run(toks, diff, full);
            for (int p = 0; p < T; ++p) {
                const bool eq = std::memcmp(r2.logits_at(p), ref.logits_at(p), static_cast<size_t>(V) * sizeof(float)) == 0;
                CHECK(p < P ? eq : !eq, "%s override at pos %d: logits at pos %d %s", site_name(site), P, p, eq ? "unchanged" : "changed");
            }
            if (site == Site::Resid) {
                std::vector<float> x = ref.layer_resid.at(L);
                std::copy(other.begin(), other.end(), x.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(P) * E));
                const ForwardResult r3 = fw.run_from(L + 1, x, T, iv, full);
                CHECK(r3.logits == r2.logits, "Resid override differs from run_from with the swapped residual");
            }
        }
        Interventions a2 = iv, b2 = iv;
        a2.override_at(m, Site::Mlp, 1, 2, std::vector<float>(F, 1.0f));
        CHECK(!(a2 == b2) && !a2.empty(), "an override must make Interventions differ / non-empty");
    }

    // --- rank-one --------------------------------------------------------------
    {
        RunOptions all;
        all.all_logits = true;
        all.capture_all_layers = true;
        all.capture_layer_resid = true;
        const ForwardResult ref = fw.run(toks, {}, all);
        const int L = 1;
        std::vector<float> u(E), w(F);
        for (size_t i = 0; i < E; ++i) u[i] = 0.05f * static_cast<float>(static_cast<int>(i % 5) - 2);
        for (size_t i = 0; i < F; ++i) w[i] = 0.03f * static_cast<float>(static_cast<int>(i % 3) - 1);
        Interventions zero;
        zero.add_rank_one(m, L, std::vector<float>(E, 0.0f), w);
        CHECK(fw.run(toks, zero, all).logits == ref.logits, "rank-one with u = 0 changed the logits");
        Interventions r1;
        r1.add_rank_one(m, L, u, w);
        CHECK(!r1.empty() && !(r1 == Interventions{}), "a rank-one edit must make Interventions non-empty");
        const ForwardResult got = fw.run(toks, r1, all);
        std::vector<float> x = ref.layer_resid.at(L);
        for (int p = 0; p < T; ++p) {
            const float* act = ref.act(L, p);
            double c = 0;
            for (size_t j = 0; j < F; ++j) c += static_cast<double>(w[j]) * act[j];
            for (size_t i = 0; i < E; ++i) x[static_cast<size_t>(p) * E + i] += u[i] * static_cast<float>(c);
        }
        const ForwardResult want = fw.run_from(L + 1, x, T, {}, all);
        float worst = 0;
        for (size_t i = 0; i < want.logits.size(); ++i) worst = std::max(worst, std::fabs(want.logits[i] - got.logits[i]));
        CHECK(worst < 1e-3f, "rank-one differs from adding u·(w·a) after the layer: max |Δlogit| %g", static_cast<double>(worst));
        CHECK(got.logits != ref.logits, "rank-one with u ≠ 0 changed nothing");
    }

    // --- delta (rome's value search) ----------------------------------------------
    {
        const int L = 0;
        RunOptions o;
        o.all_logits = true;
        o.capture_layer_resid = true;
        const ForwardResult ref = fw.run(toks, {}, o);
        Forward::DeltaBatch db;
        db.resid = ref.layer_resid.at(L);
        for (int p = 0; p < T; ++p) {
            db.pos.push_back(p);
            db.seq.push_back(0);
            db.delta_mask.push_back(p == 2 ? 1.0f : 0.0f);
        }
        // Score the next token at the last three positions against arbitrary targets.
        double nll = 0;
        for (int p = T - 3; p < T; ++p) {
            const int id = (p * 7) % V;
            db.nll_rows.push_back(p);
            db.nll_ids.push_back(id);
            nll -= std::log(static_cast<double>(prob_of(ref.logits_at(p), V, id))) / 3.0;
        }
        const std::vector<float> zero(E, 0.0f);
        const Forward::DeltaLoss d0 = fw.delta_loss(L, db, zero);
        // f32 weights vs the forward pass's own kernels: equal on f32, within the
        // activation quantization of a q8_0 matmul (vec_dot_type q8_0) otherwise.
        CHECK(std::fabs(d0.nll - nll) < 1e-2 * std::max(1.0, nll), "delta_loss NLL at δ=0 %g vs the forward pass's %g", static_cast<double>(d0.nll), nll);
        // Finite differences along two random-ish directions, around a δ ≠ 0.
        std::vector<float> delta(E);
        for (size_t i = 0; i < E; ++i) delta[i] = 0.02f * static_cast<float>(static_cast<int>((i * 37) % 11) - 5);
        const Forward::DeltaLoss dl = fw.delta_loss(L, db, delta);
        for (int k = 0; k < 2; ++k) {
            std::vector<float> v(E), plus(delta), minus(delta);
            double gv = 0;
            for (size_t i = 0; i < E; ++i) {
                v[i] = static_cast<float>(static_cast<int>((i * (13 + 4 * static_cast<size_t>(k))) % 9) - 4) / 4.0f;
                gv += static_cast<double>(dl.grad[i]) * v[i];
            }
            const float h = 1e-2f;
            for (size_t i = 0; i < E; ++i) plus[i] += h * v[i], minus[i] -= h * v[i];
            const double fd = (static_cast<double>(fw.delta_loss(L, db, plus).total) - fw.delta_loss(L, db, minus).total) / (2.0 * h);
            CHECK(std::fabs(fd - gv) <= 0.02 * std::fabs(fd) + 1e-4, "delta_loss gradient·v %g vs finite difference %g", gv, fd);
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
