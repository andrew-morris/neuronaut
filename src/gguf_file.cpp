// gguf_file.cpp — see gguf_file.h.
#include "gguf_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "format.h"
#include "gguf.h"

namespace neuronaut {

namespace {

[[noreturn]] void fail(const std::string& path, const std::string& what) {
    throw std::runtime_error(path + ": " + what);
}

// Short type tags for the metadata pane.
const char* type_tag(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8: return "u8";
        case GGUF_TYPE_INT8: return "i8";
        case GGUF_TYPE_UINT16: return "u16";
        case GGUF_TYPE_INT16: return "i16";
        case GGUF_TYPE_UINT32: return "u32";
        case GGUF_TYPE_INT32: return "i32";
        case GGUF_TYPE_FLOAT32: return "f32";
        case GGUF_TYPE_BOOL: return "bool";
        case GGUF_TYPE_STRING: return "str";
        case GGUF_TYPE_ARRAY: return "arr";
        case GGUF_TYPE_UINT64: return "u64";
        case GGUF_TYPE_INT64: return "i64";
        case GGUF_TYPE_FLOAT64: return "f64";
        default: return "?";
    }
}

// Format one scalar element of type `t` located at `p` (used for numeric arrays).
std::string scalar_at(gguf_type t, const void* p) {
    std::ostringstream os;
    switch (t) {
        case GGUF_TYPE_UINT8: os << +*static_cast<const uint8_t*>(p); break;
        case GGUF_TYPE_INT8: os << +*static_cast<const int8_t*>(p); break;
        case GGUF_TYPE_UINT16: os << *static_cast<const uint16_t*>(p); break;
        case GGUF_TYPE_INT16: os << *static_cast<const int16_t*>(p); break;
        case GGUF_TYPE_UINT32: os << *static_cast<const uint32_t*>(p); break;
        case GGUF_TYPE_INT32: os << *static_cast<const int32_t*>(p); break;
        case GGUF_TYPE_FLOAT32: os << *static_cast<const float*>(p); break;
        case GGUF_TYPE_BOOL: os << (*static_cast<const int8_t*>(p) ? "true" : "false"); break;
        case GGUF_TYPE_UINT64: os << *static_cast<const uint64_t*>(p); break;
        case GGUF_TYPE_INT64: os << *static_cast<const int64_t*>(p); break;
        case GGUF_TYPE_FLOAT64: os << *static_cast<const double*>(p); break;
        default: os << "?"; break;
    }
    return os.str();
}

size_t scalar_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

}  // namespace

std::unique_ptr<GgufFile> GgufFile::open(const std::string& path) {
    std::unique_ptr<GgufFile> f(new GgufFile());
    f->path_ = path;

    // --- 1+2: metadata and tensor directory via ggml's parser -----------------
    // no_alloc=true: build ggml_tensor structs (so we get ne/nb computed by ggml
    // for each type) but don't read any tensor data — that comes from the mmap.
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &f->meta_ctx_;
    f->gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!f->gguf_) fail(path, "not a valid GGUF file (ggml parser rejected it)");

    f->version_ = gguf_get_version(f->gguf_);
    f->alignment_ = gguf_get_alignment(f->gguf_);
    f->data_offset_ = gguf_get_data_offset(f->gguf_);

    // --- 3: map the file -------------------------------------------------------
    f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f->fd_ < 0) fail(path, std::string("open: ") + std::strerror(errno));
    struct stat st{};
    if (fstat(f->fd_, &st) != 0) fail(path, std::string("fstat: ") + std::strerror(errno));
    f->file_size_ = static_cast<uint64_t>(st.st_size);
    if (f->file_size_ == 0) fail(path, "empty file");
    // MAP_PRIVATE + PROT_READ: we can never write to the user's file through this.
    f->map_ = mmap(nullptr, f->file_size_, PROT_READ, MAP_PRIVATE, f->fd_, 0);
    if (f->map_ == MAP_FAILED) {
        f->map_ = nullptr;
        fail(path, std::string("mmap: ") + std::strerror(errno));
    }
    const auto* base = static_cast<const uint8_t*>(f->map_);

    // --- tensor directory ------------------------------------------------------
    const int64_t n_tensors = gguf_get_n_tensors(f->gguf_);
    f->tensors_.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(f->gguf_, i);
        ggml_tensor* t = ggml_get_tensor(f->meta_ctx_, name);
        if (!t) fail(path, std::string("tensor directory/ctx mismatch for ") + name);

        TensorInfo ti{};
        ti.name = name;
        ti.type = gguf_get_tensor_type(f->gguf_, i);
        ti.n_dims = ggml_n_dims(t);
        for (int d = 0; d < 4; ++d) {
            ti.ne[d] = t->ne[d];
            ti.nb[d] = t->nb[d];
        }
        ti.n_bytes = gguf_get_tensor_size(f->gguf_, i);
        ti.file_offset = f->data_offset_ + gguf_get_tensor_offset(f->gguf_, i);

        // Never trust offsets from a file: a truncated or hostile GGUF must not
        // make us read outside the mapping.
        if (ti.file_offset > f->file_size_ || ti.n_bytes > f->file_size_ - ti.file_offset) {
            fail(path, "tensor " + ti.name + " extends past end of file (truncated download?)");
        }
        ti.data = base + ti.file_offset;
        f->tensors_.push_back(std::move(ti));
    }

    // --- metadata, pre-formatted for display ---------------------------------
    const int64_t n_kv = gguf_get_n_kv(f->gguf_);
    for (int64_t i = 0; i < n_kv; ++i) {
        MetaEntry e;
        e.key = gguf_get_key(f->gguf_, i);
        const gguf_type kt = gguf_get_kv_type(f->gguf_, i);
        if (kt == GGUF_TYPE_STRING) {
            e.type = "str";
            e.value = gguf_get_val_str(f->gguf_, i);
        } else if (kt == GGUF_TYPE_ARRAY) {
            const gguf_type at = gguf_get_arr_type(f->gguf_, i);
            const size_t n = gguf_get_arr_n(f->gguf_, i);
            e.type = std::string("arr[") + type_tag(at) + ";" + std::to_string(n) + "]";
            // Show the first few elements; the pane truncates further anyway.
            std::ostringstream os;
            os << "[";
            const size_t show = std::min<size_t>(n, 8);
            for (size_t j = 0; j < show; ++j) {
                if (j) os << ", ";
                if (at == GGUF_TYPE_STRING) {
                    os << quote_printable(gguf_get_arr_str(f->gguf_, i, j));
                } else {
                    const auto* data = static_cast<const uint8_t*>(gguf_get_arr_data(f->gguf_, i));
                    os << scalar_at(at, data + j * scalar_size(at));
                }
            }
            if (n > show) os << ", … +" << (n - show);
            os << "]";
            e.value = os.str();
        } else {
            e.type = type_tag(kt);
            e.value = scalar_at(kt, gguf_get_val_data(f->gguf_, i));
        }
        f->meta_.push_back(std::move(e));
    }

    return f;
}

GgufFile::~GgufFile() {
    if (map_) munmap(map_, file_size_);
    if (fd_ >= 0) ::close(fd_);
    if (meta_ctx_) ggml_free(meta_ctx_);
    if (gguf_) gguf_free(gguf_);
}

const TensorInfo* GgufFile::find_tensor(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

std::optional<std::string> GgufFile::get_str(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) return std::nullopt;
    return std::string(gguf_get_val_str(gguf_, id));
}

std::optional<uint64_t> GgufFile::get_uint(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) return std::nullopt;
    switch (gguf_get_kv_type(gguf_, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(gguf_, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf_, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf_, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(gguf_, id);
        case GGUF_TYPE_INT8: { auto v = gguf_get_val_i8(gguf_, id); if (v < 0) return std::nullopt; return static_cast<uint64_t>(v); }
        case GGUF_TYPE_INT16: { auto v = gguf_get_val_i16(gguf_, id); if (v < 0) return std::nullopt; return static_cast<uint64_t>(v); }
        case GGUF_TYPE_INT32: { auto v = gguf_get_val_i32(gguf_, id); if (v < 0) return std::nullopt; return static_cast<uint64_t>(v); }
        case GGUF_TYPE_INT64: { auto v = gguf_get_val_i64(gguf_, id); if (v < 0) return std::nullopt; return static_cast<uint64_t>(v); }
        default: return std::nullopt;
    }
}

std::optional<double> GgufFile::get_float(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) return std::nullopt;
    switch (gguf_get_kv_type(gguf_, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(gguf_, id);
        default: return std::nullopt;
    }
}

size_t GgufFile::get_arr_len(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) return 0;
    return gguf_get_arr_n(gguf_, id);
}

std::optional<std::string> GgufFile::get_arr_str(const std::string& key, size_t i) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) return std::nullopt;
    if (gguf_get_arr_type(gguf_, id) != GGUF_TYPE_STRING || i >= gguf_get_arr_n(gguf_, id)) return std::nullopt;
    return std::string(gguf_get_arr_str(gguf_, id, i));
}

std::optional<std::vector<bool>> GgufFile::get_arr_bool(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) return std::nullopt;
    if (gguf_get_arr_type(gguf_, id) != GGUF_TYPE_BOOL) return std::nullopt;
    const size_t n = gguf_get_arr_n(gguf_, id);
    // GGUF bools are one byte each, 0 or 1.
    const auto* data = static_cast<const uint8_t*>(gguf_get_arr_data(gguf_, id));
    std::vector<bool> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = data[i] != 0;
    return out;
}

std::optional<std::vector<int64_t>> GgufFile::get_arr_int(const std::string& key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) return std::nullopt;
    const auto at = gguf_get_arr_type(gguf_, id);
    const size_t n = gguf_get_arr_n(gguf_, id);
    const auto* d = static_cast<const uint8_t*>(gguf_get_arr_data(gguf_, id));
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        switch (at) {
            case GGUF_TYPE_INT8: out[i] = reinterpret_cast<const int8_t*>(d)[i]; break;
            case GGUF_TYPE_UINT8: out[i] = d[i]; break;
            case GGUF_TYPE_INT16: out[i] = reinterpret_cast<const int16_t*>(d)[i]; break;
            case GGUF_TYPE_UINT16: out[i] = reinterpret_cast<const uint16_t*>(d)[i]; break;
            case GGUF_TYPE_INT32: out[i] = reinterpret_cast<const int32_t*>(d)[i]; break;
            case GGUF_TYPE_UINT32: out[i] = reinterpret_cast<const uint32_t*>(d)[i]; break;
            case GGUF_TYPE_INT64: out[i] = reinterpret_cast<const int64_t*>(d)[i]; break;
            case GGUF_TYPE_UINT64: out[i] = static_cast<int64_t>(reinterpret_cast<const uint64_t*>(d)[i]); break;
            default: return std::nullopt;
        }
    }
    return out;
}

bool GgufFile::has_key_prefix(const std::string& prefix) const {
    for (const auto& e : meta_) {
        if (e.key.compare(0, prefix.size(), prefix) == 0) return true;
    }
    return false;
}

}  // namespace neuronaut
