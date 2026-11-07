// test_edits.cpp — prompt-scoped interventions and edit files, on the synthetic
// fixture (no real weights, CI-safe; token ids are fed directly, no tokenizer).
//
//   test_edits <fixture.gguf> <scratch-dir>
//
// Engine invariants for Interventions::add_prompt / prompt_len (forward.h):
//   1. prompt_len = 0: add_prompt is never applied — bit-identical to no edit.
//   2. A batch entirely inside the prompt: add_prompt == add, bit-identical.
//   3. Generation steps past the prompt don't get the add: a cached step under a
//      prompt-scoped edit differs from the same step under the all-scope edit,
//      and repeating it is deterministic (the cache accepts the same patches).
//   4. One mixed batch (prompt + answer, masked) ≈ prompt-then-steps (the cached
//      path the :gen loop uses): same argmax, logits within float noise.
//   5. The edit actually does something (logits move), so 1–4 aren't vacuous.
// Edit files (edits.h):
//   6. save → load round trip; apply composes (scale, set, add, add@prompt) into
//      the expected Interventions; portability verdicts; validation refusals.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "edits.h"
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

float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_edits <fixture.gguf> <scratch-dir>\n");
        return 2;
    }
    auto f = GgufFile::open(argv[1]);
    const ModelLayout m = ModelLayout::from(*f);
    Forward fw(*f, m);
    const int V = fw.n_vocab();

    const std::vector<int32_t> prompt = {1, 5, 9, 3, 7, 2};
    const std::vector<int32_t> answer = {11, 4, 8};
    std::vector<int32_t> all = prompt;
    all.insert(all.end(), answer.begin(), answer.end());
    const int P = static_cast<int>(prompt.size());
    const int L = 1, N = 37;
    const float V_ADD = 3.0f;

    RunOptions ro;
    ro.all_logits = true;

    Interventions none;
    Interventions iv_all;
    iv_all.add(m, L, N, V_ADD);
    Interventions iv_prompt;
    iv_prompt.add_prompt(m, L, N, V_ADD);

    // 1. prompt_len 0 → no effect.
    {
        Interventions p0 = iv_prompt;
        p0.prompt_len = 0;
        const auto a = fw.run(all, none, ro), b = fw.run(all, p0, ro);
        CHECK(max_diff(a.logits, b.logits) == 0.0f, "prompt_len 0 changed logits by %g", max_diff(a.logits, b.logits));
    }
    // 2. All positions in the prompt → same as all-scope add.
    {
        Interventions pp = iv_prompt;
        pp.prompt_len = P;
        const auto a = fw.run(prompt, iv_all, ro), b = fw.run(prompt, pp, ro);
        CHECK(max_diff(a.logits, b.logits) == 0.0f, "add_prompt over a prompt-only batch differs from add by %g", max_diff(a.logits, b.logits));
        // 5. …and it does something.
        const auto c = fw.run(prompt, none, ro);
        CHECK(max_diff(a.logits, c.logits) > 1e-4f, "the edit had no effect (%g) — test is vacuous", max_diff(a.logits, c.logits));
    }
    // 3. Steps past the prompt carry no direct edit.
    {
        Interventions pp = iv_prompt;
        pp.prompt_len = P;
        auto c1 = fw.new_cache(static_cast<int>(all.size()));
        fw.decode(*c1, prompt, pp);
        const auto step_p = fw.decode(*c1, {answer[0]}, pp);
        auto c2 = fw.new_cache(static_cast<int>(all.size()));
        fw.decode(*c2, prompt, iv_all);
        const auto step_a = fw.decode(*c2, {answer[0]}, iv_all);
        CHECK(max_diff(step_p.logits, step_a.logits) > 1e-5f, "a generated step still received the prompt-scoped add");
        // Same patches, fresh cache, same result.
        const Interventions pp_after = pp;
        auto c3 = fw.new_cache(static_cast<int>(all.size()));
        fw.decode(*c3, prompt, pp_after);
        const auto step_q = fw.decode(*c3, {answer[0]}, pp_after);
        CHECK(max_diff(step_p.logits, step_q.logits) == 0.0f, "cached prompt-scoped decoding is not deterministic");
    }
    // 4. One masked batch ≈ prompt-then-steps.
    {
        Interventions pp = iv_prompt;
        pp.prompt_len = P;
        const auto batch = fw.run(all, pp, ro);
        auto c = fw.new_cache(static_cast<int>(all.size()));
        fw.decode(*c, prompt, pp);
        for (size_t i = 0; i < answer.size(); ++i) {
            const auto step = fw.decode(*c, {answer[i]}, pp);
            const float* bl = batch.logits_at(P + static_cast<int>(i));
            std::vector<float> bv(bl, bl + V);
            const float d = max_diff(bv, step.logits);
            CHECK(d < 1e-3f, "mixed batch vs cached step at answer token %zu: max |Δlogit| %g", i, d);
            CHECK(argmax(bl, V) == argmax(step.logits.data(), V), "mixed batch vs cached step: argmax differs at answer token %zu", i);
        }
        // The mask really zeroes the answer positions: the all-scope run differs there.
        const auto full = fw.run(all, iv_all, ro);
        const float* fl = full.logits_at(P + 1);
        const float* ml = batch.logits_at(P + 1);
        CHECK(max_diff(std::vector<float>(fl, fl + V), std::vector<float>(ml, ml + V)) > 1e-5f, "mask had no effect on answer positions");
    }

    // 6. Edit files.
    {
        namespace fs = std::filesystem;
        const std::string path = (fs::path(argv[2]) / "set.json").string();
        EditSet s;
        s.name = "set";
        s.notes = "round trip";
        Edit a;
        a.layer = 1, a.neuron = 37, a.op = EditOp::Scale, a.k = 7, a.note = "×7";
        Edit b;
        b.layer = 0, b.neuron = 5, b.op = EditOp::Set, b.value = 1.5f;
        Edit c;
        c.layer = 1, c.neuron = 2, c.op = EditOp::Add, c.value = -2, c.scope = EditScope::Prompt;
        s.edits = {a, b, c};
        save_edit_set(path, s, m);
        const EditSet t = load_edit_set(path, m);
        CHECK(t.edits.size() == 3 && t.notes == "round trip", "round trip lost edits/notes");
        CHECK(t.edits[0].op == EditOp::Scale && t.edits[0].k == 7 && t.edits[0].note == "×7", "edit 0 changed");
        CHECK(t.edits[1].op == EditOp::Set && t.edits[1].value == 1.5f, "edit 1 changed");
        CHECK(t.edits[2].op == EditOp::Add && t.edits[2].scope == EditScope::Prompt && t.edits[2].value == -2, "edit 2 changed");
        // edit 1 is a set in layer 0: llama.cpp applies no control vector there.
        CHECK(portability(t.edits[0]) == Portability::Gguf && portability(t.edits[1]) == Portability::NeuronautOnly &&
                  portability(t.edits[2]) == Portability::NeuronautOnly,
              "portability verdicts");
        Edit c1 = b;
        c1.layer = 1;
        CHECK(portability(c1) == Portability::ControlVector, "a set in layer 1 should be control-vector portable");
        Interventions iv;
        iv.prompt_len = 4;
        apply_edits(t.edits, m, nullptr, iv);
        CHECK(iv.layers.at(1).scale[37] == 7 && iv.layers.at(0).scale[5] == 0 && iv.layers.at(0).add[5] == 1.5f &&
                  iv.layers.at(1).add_prompt[2] == -2 && iv.layers.at(1).add[2] == 0,
              "apply_edits composed wrongly");
        Edit bad = c;
        bad.op = EditOp::Scale;   // scale @prompt is refused
        CHECK(!validate_edit(bad, m).empty(), "scale @prompt accepted");
        bad = a;
        bad.neuron = m.n_ff;
        CHECK(!validate_edit(bad, m).empty(), "out-of-range neuron accepted");
        Edit st;
        st.layer = 1, st.neuron = 1, st.op = EditOp::Steer, st.dose = 0.5f;   // no pole
        CHECK(!validate_edit(st, m).empty(), "steer without a pole accepted");

        // rank-one: u/w survive the base64 round trip bit for bit (odd lengths too:
        // the padding paths), land in Interventions, and are gguf-portable.
        Edit r;
        r.layer = 1, r.neuron = -1, r.op = EditOp::RankOne;
        for (int i = 0; i < m.n_embd; ++i) r.u.push_back(std::ldexp(static_cast<float>(i % 13) - 6.5f, -(i % 20)));
        for (int i = 0; i < m.n_ff; ++i) r.w.push_back(-1.0f / static_cast<float>(i + 3));
        CHECK(validate_edit(r, m).empty(), "valid rank-one refused: %s", validate_edit(r, m).c_str());
        EditSet rs;
        rs.name = "r1";
        rs.edits = {r};
        const std::string rp = (fs::path(argv[2]) / "r1.json").string();
        save_edit_set(rp, rs, m);
        const EditSet rt = load_edit_set(rp, m);
        CHECK(rt.edits.size() == 1 && rt.edits[0].op == EditOp::RankOne && rt.edits[0].u == r.u && rt.edits[0].w == r.w,
              "rank-one round trip changed u/w");
        CHECK(portability(rt.edits[0]) == Portability::Gguf, "rank-one should be gguf-portable");
        Interventions riv;
        apply_edits(rt.edits, m, nullptr, riv);
        CHECK(riv.rank_one.size() == 1 && riv.rank_one[0].u == r.u, "apply_edits lost the rank-one edit");
        Edit rbad = r;
        rbad.w.pop_back();
        CHECK(!validate_edit(rbad, m).empty(), "rank-one with a short w accepted");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
