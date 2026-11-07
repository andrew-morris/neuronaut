// test_export.cpp — `neuronaut export` (src/export.h) on the synthetic fixture,
// with our own forward pass as the runtime (the fixture can't run in llama.cpp;
// tests/test_export.sh does that leg on the real model).
//
//   test_export <fixture.gguf> <scratch-dir>
//
//   1. Folding ×0, ×−1 and ×2 (exact in every format: sign, zero, power of two)
//      into up rows: the exported file, run with no edits, is bit-identical to the
//      original run with those edits — every position's logits.
//   2. Every other tensor byte is unchanged (only the folded rows differ).
//   3. The control vector's direction.L is Σ v·down[:, n] over that layer's adds.
//   4. Refusals: prompt-scoped edits and layer-0 adds have no file form.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "export.h"
#include "forward.h"
#include "gguf_file.h"
#include "model_layout.h"
#include "neuron.h"

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
        std::fprintf(stderr, "usage: test_export <fixture.gguf> <scratch-dir>\n");
        return 2;
    }
    auto f = GgufFile::open(argv[1]);
    const ModelLayout m = ModelLayout::from(*f);
    const std::string out = (std::filesystem::path(argv[2]) / (std::filesystem::path(argv[1]).stem().string() + "-exported.gguf")).string();

    std::vector<Edit> edits(3);
    edits[0].layer = 0, edits[0].neuron = 7, edits[0].op = EditOp::Scale, edits[0].k = 0.0f;
    edits[1].layer = 1, edits[1].neuron = 37, edits[1].op = EditOp::Scale, edits[1].k = -1.0f;
    edits[2].layer = 1, edits[2].neuron = 100, edits[2].op = EditOp::Scale, edits[2].k = 2.0f;
    const ExportPlan plan = plan_export(edits, m, false);
    ExportReport rep;
    export_gguf(*f, m, plan, out, "test", rep);
    CHECK(rep.rows_folded == 3 && rep.rows_exact == 3, "folded %d rows, %d exact (want 3/3)", rep.rows_folded, rep.rows_exact);

    // 1. Same logits, bit for bit.
    const std::vector<int32_t> toks = {1, 5, 9, 3, 7, 2, 11, 4};
    RunOptions ro;
    ro.all_logits = true;
    Interventions iv;
    apply_edits(edits, m, nullptr, iv);
    std::vector<float> want, got;
    {
        Forward fw(*f, m);
        want = fw.run(toks, iv, ro).logits;
    }
    auto g = GgufFile::open(out);
    const ModelLayout gm = ModelLayout::from(*g);
    {
        Forward fw(*g, gm);
        got = fw.run(toks, {}, ro).logits;
    }
    CHECK(got == want, "exported model differs from model+edits");
    CHECK(g->get_str("neuronaut.edits").value_or("") == "test", "provenance metadata missing");

    // 2. Only the folded rows changed.
    for (const auto& t : f->tensors()) {
        const TensorInfo* u = g->find_tensor(t.name);
        CHECK(u && u->n_bytes == t.n_bytes, "%s missing or resized", t.name.c_str());
        if (!u) continue;
        for (int64_t r = 0; r < t.n_rows(); ++r) {
            bool folded = false;
            for (const auto& e : edits)
                folded |= &t == m.layers[static_cast<size_t>(e.layer)].ffn_up && r == e.neuron;
            const bool same = std::memcmp(t.row_ptr(r), u->row_ptr(r), t.row_bytes()) == 0;
            CHECK(folded ? !same : same, "%s row %lld %s", t.name.c_str(), static_cast<long long>(r),
                  folded ? "unchanged" : "changed");
        }
    }

    // 3. Control-vector directions.
    std::vector<Edit> adds(3);
    adds[0].layer = 1, adds[0].neuron = 3, adds[0].op = EditOp::Add, adds[0].value = 2.0f;
    adds[1].layer = 1, adds[1].neuron = 9, adds[1].op = EditOp::Set, adds[1].value = -1.5f;
    adds[2].layer = 1, adds[2].neuron = 3, adds[2].op = EditOp::Add, adds[2].value = 0.5f;
    const ExportPlan cp = plan_export(adds, m, false);
    CHECK(cp.folds.size() == 1 && cp.adds.size() == 3, "set should fold (×0) and add");
    const auto dirs = control_directions(m, cp, nullptr);
    CHECK(dirs.size() == 1 && dirs[0].first == 1, "want one direction, for layer 1");
    if (dirs.size() == 1) {
        const auto d3 = read_neuron(m, 1, 3).down, d9 = read_neuron(m, 1, 9).down;
        float worst = 0;
        for (size_t i = 0; i < d3.size(); ++i) worst = std::max(worst, std::fabs(dirs[0].second[i] - (2.5f * d3[i] - 1.5f * d9[i])));
        CHECK(worst < 1e-5f, "direction.1 off by %g", worst);
        ExportReport cr;
        export_control_vector(m, cp, nullptr, out + ".cvec", cr);
        CHECK(cr.cvec_layers == 1, "wrote %d directions", cr.cvec_layers);
    }

    // 4. No file form.
    Edit p = adds[0];
    p.scope = EditScope::Prompt;
    bool threw = false;
    try { plan_export({p}, m, false); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "a prompt-scoped edit was accepted");
    Edit z = adds[0];
    z.layer = 0;
    threw = false;
    try { plan_export({z}, m, false); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "a layer-0 add was accepted");
    CHECK(plan_export({p, z, edits[0]}, m, true).skipped.size() == 2, "--skip-unportable should drop exactly two");

    std::filesystem::remove(out);
    std::filesystem::remove(out + ".cvec");
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
