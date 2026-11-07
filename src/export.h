// export.h — turn an edit set into files other runtimes load (M4).
//
//   neuronaut export MODEL EDITS... -o OUT.gguf [--cvec OUT.cvec.gguf]
//
// Two artifacts, by what each edit is (edits.h portability):
//
//   GGUF (scale; the ×0 half of set)   a′ = k·a with a = silu(gate·h) ⊙ (up·h), so
//       scaling neuron n's up row by k scales its activation by k — at every
//       position, in any runtime. Every edit on one neuron composes into one k (the
//       product, as Interventions does). The row is scaled in its own storage:
//       F32/F16/BF16 element by element; block-quantized types (Q4_0 … Q8_0, the
//       K-quants) by scaling each block's fp16 scale fields (d, and dmin/m where the
//       format has them), so the quants themselves never move. For k ∈ {0, ±1}
//       (ablate, invert) that is exact in every format and the exported model is
//       bit-identical to neuronaut-with-edits (tests/test_export.sh, through
//       llama.cpp); other k round once, in fp16. Each folded row is checked against
//       k × the original row, and the report says whether it matched exactly.
//       Rows are patched in place in a copy of the file: same tensors, same
//       offsets, plus neuronaut.* provenance metadata.
//
//   control vector (add, steer, the add half of set; every position)
//       a′ = a + v adds v·down[:, n] to the residual stream after layer L — what
//       llama.cpp's --control-vector adds after layer L (tensor "direction.L").
//       Several edits in one layer sum into one direction. llama.cpp never applies
//       a control vector to layer 0 (its loader rejects direction.0), so adds in
//       layer 0 are not exportable. Numerically close, not bit-identical: the add
//       happens after the down projection's rounding instead of before it.
//
//   GGUF (rank-one, from `neuronaut rome`)   W_down ← W_down + u·wᵀ: every row of
//       the layer's ffn_down changes, so each is dequantized, updated in float and
//       re-quantized to its own type (F16: one rounding per element; Q8_0 etc.: the
//       block's scale is recomputed). Close to neuronaut-with-edits, not
//       bit-identical: neuronaut adds u·(w·a) beside the unrounded matmul.
//
// Prompt-scoped edits (only while reading the prompt) have no file form at all;
// export refuses them unless told to skip them.
#pragma once

#include <string>
#include <vector>

#include "edits.h"
#include "gguf_file.h"
#include "model_layout.h"

namespace neuronaut {

class SteerUnits;

struct ExportReport {
    int rows_folded = 0;             // up rows rewritten
    int rows_exact = 0;              // … whose dequantized values equal k × the original exactly
    int cvec_layers = 0;             // direction tensors written
    int layers_rewritten = 0;        // ffn_down tensors rewritten by rank-one edits
    std::vector<std::string> lines;  // human-readable, one per folded neuron / direction / skipped edit
};

// Which edits each artifact needs; throws on edits with no file form (prompt
// scope, layer-0 adds) unless skip_unportable (then they are reported and dropped).
struct ExportPlan {
    std::vector<Edit> folds;   // scale-like parts, per edit
    std::vector<Edit> adds;    // add-like parts (steer resolved later by SteerUnits)
    std::vector<Edit> rank_ones;   // rank-one weight edits (ROME): ffn_down rewritten
    std::vector<std::string> skipped;
    bool needs_gguf() const { return !folds.empty() || !rank_ones.empty(); }
    bool needs_cvec() const { return !adds.empty(); }
};
ExportPlan plan_export(const std::vector<Edit>& edits, const ModelLayout& m, bool skip_unportable);

// The patched copy of `f` (requires plan.needs_gguf()).
void export_gguf(const GgufFile& f, const ModelLayout& m, const ExportPlan& plan, const std::string& out_path, const std::string& provenance,
                 ExportReport& rep);

// A llama.cpp control vector file (requires plan.needs_cvec(); `units` when any steer).
void export_control_vector(const ModelLayout& m, const ExportPlan& plan, const SteerUnits* units, const std::string& out_path, ExportReport& rep);

// The control vector's per-layer directions, as written: layer → n_embd floats.
std::vector<std::pair<int, std::vector<float>>> control_directions(const ModelLayout& m, const ExportPlan& plan, const SteerUnits* units);

}  // namespace neuronaut
