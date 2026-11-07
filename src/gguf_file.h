// gguf_file.h — a read-only view of a GGUF file.
//
// A GGUF file is three things back to back:
//
//   1. header + key/value metadata  (architecture, sizes, tokenizer, chat template…)
//   2. tensor directory             (name, ggml type, shape, offset into data section)
//   3. data section                 (each tensor's raw bytes, aligned, possibly quantized)
//
// ggml's own gguf parser handles (1) and (2). We mmap the whole file read-only for
// (3), so looking at any tensor byte is a pointer dereference and opening a
// multi-GB model costs nothing until pages are actually touched.
//
// Patching (M4) will layer copy-on-write overlays on top of this view rather than
// writing through it — the file on disk is never modified in place.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ggml.h"

struct gguf_context;

namespace neuronaut {

// One entry in the tensor directory, plus where its bytes live in our mapping.
struct TensorInfo {
    std::string name;       // e.g. "blk.3.ffn_up.weight"
    ggml_type type;         // storage type: F32, F16, Q8_0, Q4_K, …
    int n_dims;             // 1..4
    int64_t ne[4];          // elements per dim; ne[0] is the contiguous (innermost) dim
    size_t nb[4];           // byte stride per dim, as ggml computes it for this type
    uint64_t file_offset;   // absolute offset of the first byte within the file
    size_t n_bytes;         // total bytes of tensor data
    const uint8_t* data;    // pointer into the mmap (== base + file_offset)

    // ggml calls a contiguous run of ne[0] elements a "row". A row is always made of
    // whole quantization blocks, so it can be dequantized on its own.
    int64_t n_rows() const { return ne[1] * ne[2] * ne[3]; }
    size_t row_bytes() const { return nb[1]; }
    const uint8_t* row_ptr(int64_t r) const { return data + static_cast<size_t>(r) * nb[1]; }
};

// A metadata value rendered for humans. Arrays are summarized (the Llama 3
// tokenizer is a 128k-entry string array; nobody wants all of it in a pane).
struct MetaEntry {
    std::string key;
    std::string type;       // "u32", "str", "arr[str;128256]", …
    std::string value;      // formatted, possibly truncated
};

class GgufFile {
public:
    // Throws std::runtime_error with a readable message on any problem.
    static std::unique_ptr<GgufFile> open(const std::string& path);
    ~GgufFile();

    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::string& path() const { return path_; }
    uint64_t file_size() const { return file_size_; }
    uint32_t version() const { return version_; }
    size_t alignment() const { return alignment_; }
    uint64_t data_offset() const { return data_offset_; }
    // The whole read-only mapping (file_size() bytes); tensor data pointers lie inside it.
    const uint8_t* map_base() const { return static_cast<const uint8_t*>(map_); }

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find_tensor(const std::string& name) const;

    const std::vector<MetaEntry>& metadata() const { return meta_; }

    // Typed metadata lookups; nullopt if missing or of a different type.
    std::optional<std::string> get_str(const std::string& key) const;
    std::optional<uint64_t> get_uint(const std::string& key) const;   // any unsigned/signed int
    std::optional<double> get_float(const std::string& key) const;   // f32/f64
    // String array element access (tokenizer vocab).
    size_t get_arr_len(const std::string& key) const;
    std::optional<std::string> get_arr_str(const std::string& key, size_t i) const;
    // Bool array (e.g. qwen35.attention.recurrent_layers: one flag per block);
    // nullopt if missing or not an array of bools.
    std::optional<std::vector<bool>> get_arr_bool(const std::string& key) const;
    // Integer array (any int element type), e.g. qwen35.rope.dimension_sections.
    std::optional<std::vector<int64_t>> get_arr_int(const std::string& key) const;
    // True if any metadata key starts with `prefix` (e.g. "prism.hadamard.").
    bool has_key_prefix(const std::string& prefix) const;

private:
    GgufFile() = default;

    std::string path_;
    uint64_t file_size_ = 0;
    uint32_t version_ = 0;
    size_t alignment_ = 0;
    uint64_t data_offset_ = 0;

    gguf_context* gguf_ = nullptr;      // metadata + tensor directory
    ggml_context* meta_ctx_ = nullptr;  // tensor shapes (no_alloc: no data)
    void* map_ = nullptr;               // mmap of the whole file
    int fd_ = -1;

    std::vector<TensorInfo> tensors_;
    std::vector<MetaEntry> meta_;
};

}  // namespace neuronaut
