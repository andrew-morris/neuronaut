// blocks.h — decoding one quantization block for the hex pane.
//
// ggml stores quantized weights as fixed-size blocks. Each block covers
// `block_elems` consecutive elements of one row and carries its own scale(s):
//
//   Q8_0  (34 bytes / 32 elems):  [d: fp16] [q: int8 × 32]          x = d · q
//   Q4_0  (18 bytes / 32 elems):  [d: fp16] [q: 4-bit × 32]         x = d · (q - 8)
//   F16   ( 2 bytes /  1 elem ):  the value itself
//
// The scale fields are the interesting part for patching: multiplying d by k scales
// all 32 elements by k without re-quantizing anything, and d = 0 zeroes them.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf_file.h"

namespace neuronaut {

struct BlockField {
    std::string name;     // "d", "q[0..31]", …
    size_t offset;        // byte offset within the block
    size_t size;          // bytes
    std::string value;    // decoded, human readable
};

struct BlockDump {
    std::string tensor;
    ggml_type type;
    int64_t row = 0;
    int64_t block_index = 0;    // which block within the row
    int64_t first_elem = 0;     // column of the block's first element
    uint64_t file_offset = 0;   // absolute file offset of the block
    std::vector<uint8_t> raw;   // the block's bytes
    std::vector<BlockField> fields;   // known layout (empty for types we don't annotate yet)
    std::vector<float> values;        // dequantized elements of this block
};

// The block of row `row` that contains column `col`.
BlockDump dump_block(const TensorInfo& t, int64_t row, int64_t col);

}  // namespace neuronaut
