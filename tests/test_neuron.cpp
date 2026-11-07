// test_neuron.cpp — check neuron slicing against the fixture's closed-form weights.
//
//   test_neuron <fixture.gguf>
//
// For every neuron in every layer: gate row, up row and down column read through
// neuronaut must equal fixture::weight(...) to within the storage type's error.
// Also checks layout parsing, footprints and block dumps.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "../tools/fixture_formula.h"
#include "blocks.h"
#include "format.h"
#include "gguf_file.h"
#include "model_layout.h"
#include "neuron.h"

using namespace neuronaut;

namespace {

int failures = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            ++failures;                                     \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);              \
            std::fprintf(stderr, "\n");                     \
            if (failures > 20) return;                      \
        }                                                   \
    } while (0)

// Max abs error we accept per storage type. Q8_0 rounds each value to one of 255
// steps of d = max|block|/127 (≤ 1/127 here), so error ≤ d/2 plus fp16 rounding of d.
float tolerance(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return 0.0f;
        case GGML_TYPE_F16: return 1e-3f;
        case GGML_TYPE_Q8_0: return 6e-3f;
        default: return 0.05f;
    }
}

void test_display_width() {
    CHECK(display_width("abc") == 3, "ascii");
    CHECK(display_width("欺骗") == 4, "CJK is 2 cells each, got %d", display_width("欺骗"));
    CHECK(display_width(" Taiwanese") == 10, "leading space");
    CHECK(display_width("e\xcc\x81") == 1, "combining acute is 0 cells");
    CHECK(display_width("\xff") == 1, "stray byte counts 1");
    CHECK(pad_cells("藏", 4) == "藏  ", "pad CJK to 4 cells");
    CHECK(pad_cells("toolong", 3) == "toolong", "never truncates");
}

void test_layout(const ModelLayout& m) {
    CHECK(m.arch == "llama", "arch=%s", m.arch.c_str());
    CHECK(m.n_layer == fixture::n_layer, "n_layer=%d", m.n_layer);
    CHECK(m.n_embd == fixture::n_embd, "n_embd=%d", m.n_embd);
    CHECK(m.n_ff == fixture::n_ff, "n_ff=%d", m.n_ff);
    CHECK(m.n_head == fixture::n_head && m.n_head_kv == fixture::n_head_kv, "heads=%d/%d", m.n_head, m.n_head_kv);
    CHECK(m.n_vocab == fixture::n_vocab, "n_vocab=%d", m.n_vocab);
    CHECK(m.output == nullptr, "fixture has tied embeddings; output.weight should be absent");
    CHECK(m.warnings.empty(), "unexpected warning: %s", m.warnings.empty() ? "" : m.warnings[0].c_str());
}

void test_neurons(const ModelLayout& m, ggml_type wtype) {
    const float tol = tolerance(wtype);
    float worst = 0;
    for (int l = 0; l < m.n_layer; ++l) {
        for (int n = 0; n < m.n_ff; ++n) {
            const NeuronView v = read_neuron(m, l, n);
            CHECK(static_cast<int>(v.gate.size()) == m.n_embd && static_cast<int>(v.down.size()) == m.n_embd, "sizes");
            for (int e = 0; e < m.n_embd; ++e) {
                // gate/up: row n (neuron), col e (embedding dim).
                const float eg = fixture::weight(fixture::FFN_GATE, l, n, e);
                const float eu = fixture::weight(fixture::FFN_UP, l, n, e);
                // down: row e (embedding dim), col n (neuron). Transposed on purpose.
                const float ed = fixture::weight(fixture::FFN_DOWN, l, e, n);
                const float dg = std::fabs(v.gate[e] - eg), du = std::fabs(v.up[e] - eu), dd = std::fabs(v.down[e] - ed);
                worst = std::max({worst, dg, du, dd});
                CHECK(dg <= tol && du <= tol && dd <= tol,
                      "L%d N%d e%d: gate %f/%f up %f/%f down %f/%f", l, n, e, v.gate[e], eg, v.up[e], eu, v.down[e], ed);
            }
        }
    }
    std::printf("  neurons: %d layers x %d, worst abs error %.6f (tol %.6f)\n", m.n_layer, m.n_ff, worst, tol);
}

void test_footprints(const ModelLayout& m, ggml_type wtype) {
    const NeuronView v = read_neuron(m, 1, 37);
    const int64_t blk = ggml_blck_size(wtype);
    // up row: contiguous, exclusive, n_embd/blk blocks.
    CHECK(v.up_fp.contiguous && v.up_fp.shared_with == 0, "up row should be contiguous and exclusive");
    CHECK(v.up_fp.n_blocks == m.n_embd / blk, "up blocks=%lld", static_cast<long long>(v.up_fp.n_blocks));
    CHECK(v.up_fp.last_byte - v.up_fp.first_byte == ggml_row_size(wtype, m.n_embd), "up byte span");
    // down col: one block per embedding dim, each shared with blk-1 neighbours.
    CHECK(!v.down_fp.contiguous, "down col should be strided");
    CHECK(v.down_fp.n_blocks == m.n_embd, "down blocks=%lld", static_cast<long long>(v.down_fp.n_blocks));
    CHECK(v.down_fp.shared_with == blk - 1, "down shared_with=%lld", static_cast<long long>(v.down_fp.shared_with));
}

void test_blocks(const ModelLayout& m, ggml_type wtype) {
    const TensorInfo& up = *m.layers[1].ffn_up;
    const BlockDump b = dump_block(up, 37, 45);
    const int64_t blk = ggml_blck_size(wtype);
    CHECK(b.block_index == 45 / blk && b.first_elem == (45 / blk) * blk, "block index");
    CHECK(b.raw.size() == ggml_type_size(wtype), "raw size %zu", b.raw.size());
    CHECK(!b.fields.empty(), "fields annotated for %s", ggml_type_name(wtype));
    const auto row = read_row(up, 37);
    for (int64_t i = 0; i < blk; ++i) {
        CHECK(b.values[static_cast<size_t>(i)] == row[static_cast<size_t>(b.first_elem + i)], "block value %lld", static_cast<long long>(i));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <fixture.gguf>\n", argv[0]);
        return 2;
    }
    try {
        auto f = GgufFile::open(argv[1]);
        const ModelLayout m = ModelLayout::from(*f);
        const ggml_type wtype = m.layers[0].ffn_up->type;
        std::printf("%s (%s)\n", argv[1], ggml_type_name(wtype));
        test_display_width();
        test_layout(m);
        test_neurons(m, wtype);
        test_footprints(m, wtype);
        test_blocks(m, wtype);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("  OK\n");
    return 0;
}
