// neuron.h — pulling individual neurons (rows / columns) out of weight tensors.
//
// Storage reality for a neuron n in a quantized model (see model_layout.h):
//
//   ffn_gate / ffn_up  [n_embd, n_ff]   row n            -> contiguous bytes, whole
//                                                           blocks that belong to n alone
//   ffn_down           [n_ff, n_embd]   element n of     -> n_embd scattered slots, each
//                                       every row           inside a block it shares with
//                                                           (block_size - 1) other neurons
//
// That asymmetry decides what a byte-level patch can do cleanly (M4): editing a
// gate/up row touches only neuron n; editing a down column re-quantizes blocks
// that other neurons also live in. `Footprint` makes that visible.
#pragma once

#include <cstdint>
#include <vector>

#include "gguf_file.h"
#include "model_layout.h"

namespace neuronaut {

// Dequantize row `r` of a 2-D tensor (ne[0] floats). Works for every ggml type
// that has a to_float routine (F32/F16/BF16 and all quant types).
std::vector<float> read_row(const TensorInfo& t, int64_t r);

// Dequantize column `c` of a 2-D tensor (ne[1] floats): element c of every row.
// Only the one quant block containing c is decoded per row, not the whole row.
// A whole 1-D tensor (norm gains, biases), dequantized.
std::vector<float> read_vec(const TensorInfo& t);

std::vector<float> read_col(const TensorInfo& t, int64_t c);

// Summary stats of a vector, for panes too narrow to show 2048 numbers.
struct VecStats {
    float min = 0, max = 0;
    double l2 = 0;          // Euclidean norm
    double mean_abs = 0;
    int argmax_abs = -1;    // index of the largest |v|
};
VecStats stats(const std::vector<float>& v);

// The k indices with the largest |v|, largest first.
std::vector<int> top_abs(const std::vector<float>& v, int k);

// Which bytes of the file a neuron's slice of one tensor occupies.
struct Footprint {
    uint64_t first_byte = 0;    // absolute file offset of first touched byte
    uint64_t last_byte = 0;     // absolute file offset one past the last touched byte
    int64_t n_blocks = 0;       // quantization blocks touched
    int64_t block_elems = 0;    // elements per block (1 for F32/F16)
    size_t block_bytes = 0;     // bytes per block
    bool contiguous = false;    // one unbroken byte range?
    int64_t shared_with = 0;    // other neurons living inside those same blocks (per block)
};
Footprint row_footprint(const TensorInfo& t, int64_t r);
Footprint col_footprint(const TensorInfo& t, int64_t c);

// Everything about one MLP neuron that can be read without running the model.
struct NeuronView {
    int layer = 0;
    int index = 0;
    std::vector<float> gate;   // ffn_gate row  (n_embd)  — reads the residual stream
    std::vector<float> up;     // ffn_up row    (n_embd)  — reads the residual stream
    std::vector<float> down;   // ffn_down col  (n_embd)  — writes to the residual stream
    Footprint gate_fp, up_fp, down_fp;
};
// Throws std::out_of_range if (layer, index) doesn't exist or the layer has no dense FFN.
NeuronView read_neuron(const ModelLayout& m, int layer, int index);

}  // namespace neuronaut
