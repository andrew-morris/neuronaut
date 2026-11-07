// neuron.cpp — see neuron.h.
#include "neuron.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace neuronaut {

namespace {

// Decode `n` elements starting at a block boundary `src` of type `type` into `dst`.
void to_float(ggml_type type, const uint8_t* src, float* dst, int64_t n) {
    if (type == GGML_TYPE_F32) {
        // F32 has no to_float in ggml's trait table: it's already floats.
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
        return;
    }
    const auto* traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error(std::string("no dequantizer for type ") + ggml_type_name(type));
    }
    traits->to_float(src, dst, n);
}

void require_2d(const TensorInfo& t) {
    if (t.n_dims != 2) throw std::runtime_error(t.name + " is not a 2-D tensor");
}

}  // namespace

std::vector<float> read_row(const TensorInfo& t, int64_t r) {
    require_2d(t);
    if (r < 0 || r >= t.ne[1]) throw std::out_of_range(t.name + ": row " + std::to_string(r) + " out of range");
    std::vector<float> out(static_cast<size_t>(t.ne[0]));
    to_float(t.type, t.row_ptr(r), out.data(), t.ne[0]);
    return out;
}

std::vector<float> read_vec(const TensorInfo& t) {
    if (t.n_dims != 1) throw std::runtime_error(t.name + " is not a 1-D tensor");
    std::vector<float> out(static_cast<size_t>(t.ne[0]));
    to_float(t.type, t.data, out.data(), t.ne[0]);
    return out;
}

std::vector<float> read_col(const TensorInfo& t, int64_t c) {
    require_2d(t);
    if (c < 0 || c >= t.ne[0]) throw std::out_of_range(t.name + ": col " + std::to_string(c) + " out of range");
    const int64_t blk = ggml_blck_size(t.type);   // elements per block
    const size_t bsz = ggml_type_size(t.type);   // bytes per block
    const int64_t bi = c / blk;                  // which block in the row
    const int64_t within = c % blk;              // position inside that block

    std::vector<float> out(static_cast<size_t>(t.ne[1]));
    std::vector<float> tmp(static_cast<size_t>(blk));
    for (int64_t r = 0; r < t.ne[1]; ++r) {
        to_float(t.type, t.row_ptr(r) + static_cast<size_t>(bi) * bsz, tmp.data(), blk);
        out[static_cast<size_t>(r)] = tmp[static_cast<size_t>(within)];
    }
    return out;
}

VecStats stats(const std::vector<float>& v) {
    VecStats s;
    if (v.empty()) return s;
    s.min = s.max = v[0];
    double sq = 0, sa = 0;
    float best = -1;
    for (size_t i = 0; i < v.size(); ++i) {
        const float x = v[i];
        s.min = std::min(s.min, x);
        s.max = std::max(s.max, x);
        sq += static_cast<double>(x) * x;
        sa += std::fabs(x);
        if (std::fabs(x) > best) {
            best = std::fabs(x);
            s.argmax_abs = static_cast<int>(i);
        }
    }
    s.l2 = std::sqrt(sq);
    s.mean_abs = sa / static_cast<double>(v.size());
    return s;
}

std::vector<int> top_abs(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min<int>(k, static_cast<int>(v.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return std::fabs(v[static_cast<size_t>(a)]) > std::fabs(v[static_cast<size_t>(b)]); });
    idx.resize(static_cast<size_t>(k));
    return idx;
}

Footprint row_footprint(const TensorInfo& t, int64_t r) {
    Footprint fp;
    fp.block_elems = ggml_blck_size(t.type);
    fp.block_bytes = ggml_type_size(t.type);
    fp.first_byte = t.file_offset + static_cast<uint64_t>(r) * t.nb[1];
    fp.last_byte = fp.first_byte + t.row_bytes();
    fp.n_blocks = t.ne[0] / fp.block_elems;
    fp.contiguous = true;
    // Every block in this row holds only elements of row r — i.e. only this neuron.
    fp.shared_with = 0;
    return fp;
}

Footprint col_footprint(const TensorInfo& t, int64_t c) {
    Footprint fp;
    fp.block_elems = ggml_blck_size(t.type);
    fp.block_bytes = ggml_type_size(t.type);
    const uint64_t bi = static_cast<uint64_t>(c / fp.block_elems);
    fp.first_byte = t.file_offset + bi * fp.block_bytes;
    fp.last_byte = t.file_offset + static_cast<uint64_t>(t.ne[1] - 1) * t.nb[1] + (bi + 1) * fp.block_bytes;
    fp.n_blocks = t.ne[1];                         // one block in every row
    fp.contiguous = (t.ne[1] == 1);
    fp.shared_with = fp.block_elems - 1;           // neighbours packed into the same block
    return fp;
}

NeuronView read_neuron(const ModelLayout& m, int layer, int index) {
    if (m.hadamard_folded) {
        // The rotation is folded into ffn_down's *input* (the neuron axis): column n of
        // the stored ffn_down is a mix of many real neurons. Refuse rather than mislead.
        throw std::runtime_error("this GGUF has prism.hadamard-rotated weights; neurons are not individually stored");
    }
    if (!m.has_ffn(layer)) throw std::out_of_range("layer " + std::to_string(layer) + " has no dense FFN");
    if (index < 0 || index >= m.n_ff) {
        throw std::out_of_range("neuron " + std::to_string(index) + " out of range (n_ff=" + std::to_string(m.n_ff) + ")");
    }
    const auto& L = m.layers[static_cast<size_t>(layer)];
    NeuronView n;
    n.layer = layer;
    n.index = index;
    n.gate = read_row(*L.ffn_gate, index);
    n.up = read_row(*L.ffn_up, index);
    n.down = read_col(*L.ffn_down, index);
    n.gate_fp = row_footprint(*L.ffn_gate, index);
    n.up_fp = row_footprint(*L.ffn_up, index);
    n.down_fp = col_footprint(*L.ffn_down, index);
    return n;
}

}  // namespace neuronaut
