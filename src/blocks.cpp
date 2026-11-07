// blocks.cpp — see blocks.h.
#include "blocks.h"

#include <cstring>
#include <stdexcept>

#include "format.h"

namespace neuronaut {

namespace {

float half_at(const uint8_t* p) {
    ggml_fp16_t h;
    std::memcpy(&h, p, sizeof h);
    return ggml_fp16_to_fp32(h);
}

}  // namespace

BlockDump dump_block(const TensorInfo& t, int64_t row, int64_t col) {
    if (t.n_dims != 2) throw std::runtime_error(t.name + " is not a 2-D tensor");
    if (row < 0 || row >= t.ne[1] || col < 0 || col >= t.ne[0]) {
        throw std::out_of_range(t.name + ": (" + std::to_string(row) + ", " + std::to_string(col) + ") out of range");
    }
    BlockDump b;
    b.tensor = t.name;
    b.type = t.type;
    b.row = row;
    const int64_t blk = ggml_blck_size(t.type);
    const size_t bsz = ggml_type_size(t.type);
    b.block_index = col / blk;
    b.first_elem = b.block_index * blk;

    const uint8_t* p = t.row_ptr(row) + static_cast<size_t>(b.block_index) * bsz;
    b.file_offset = t.file_offset + static_cast<uint64_t>(p - t.data);
    b.raw.assign(p, p + bsz);

    // Dequantize via the same path the neuron reader uses (whole row is wasteful
    // for big rows, so decode just this block).
    b.values.resize(static_cast<size_t>(blk));
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(b.values.data(), p, sizeof(float));
    } else {
        const auto* tr = ggml_get_type_traits(t.type);
        if (tr && tr->to_float) tr->to_float(p, b.values.data(), blk);
    }

    // Annotate the layouts we understand. Anything else still gets raw bytes + values.
    switch (t.type) {
        case GGML_TYPE_F32:
            b.fields.push_back({"f32", 0, 4, fmt_f(b.values[0], 6)});
            break;
        case GGML_TYPE_F16:
            b.fields.push_back({"f16", 0, 2, fmt_f(b.values[0], 6)});
            break;
        case GGML_TYPE_BF16:
            b.fields.push_back({"bf16", 0, 2, fmt_f(b.values[0], 6)});
            break;
        case GGML_TYPE_Q8_0: {
            // struct block_q8_0 { ggml_half d; int8_t qs[32]; }
            const float d = half_at(p);
            b.fields.push_back({"d (scale, fp16)", 0, 2, fmt_f(d, 8)});
            std::string qs;
            for (int i = 0; i < 32; ++i) {
                if (i) qs += ' ';
                qs += std::to_string(static_cast<int8_t>(p[2 + i]));
            }
            b.fields.push_back({"qs[32] (int8)", 2, 32, qs});
            break;
        }
        case GGML_TYPE_Q4_0: {
            // struct block_q4_0 { ggml_half d; uint8_t qs[16]; } — low nibbles are
            // elements 0..15, high nibbles 16..31, each stored offset by +8.
            const float d = half_at(p);
            b.fields.push_back({"d (scale, fp16)", 0, 2, fmt_f(d, 8)});
            std::string qs;
            for (int i = 0; i < 32; ++i) {
                const uint8_t byte = p[2 + (i % 16)];
                const int q = (i < 16 ? (byte & 0x0f) : (byte >> 4)) - 8;
                if (i) qs += ' ';
                qs += std::to_string(q);
            }
            b.fields.push_back({"qs[32] (4-bit, -8)", 2, 16, qs});
            break;
        }
        default:
            break;
    }
    return b;
}

}  // namespace neuronaut
