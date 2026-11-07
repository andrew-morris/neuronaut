// oracle.cpp — see oracle.h.
#include "oracle.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gguf.h"
#include "llama.h"

namespace neuronaut {

namespace {

// libllama logs every tensor at load; keep only warnings and errors.
void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

void backend_init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(quiet_log, nullptr);
        llama_backend_init();
    });
}

// Silence the "....." load progress (it would scribble over the TUI).
bool quiet_progress(float, void*) { return true; }

}  // namespace

// ---------------------------------------------------------------------------

Tokenizer::Tokenizer(const std::string& path) {
    backend_init_once();
    llama_model_params p = llama_model_default_params();
    p.vocab_only = true;
    p.progress_callback = quiet_progress;
    model_ = llama_model_load_from_file(path.c_str(), p);
    if (!model_) throw std::runtime_error("libllama could not load the vocabulary from " + path);
}

Tokenizer::~Tokenizer() {
    if (model_) llama_model_free(model_);
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos, bool parse_special) const {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    // First call with no buffer returns -(needed size).
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_bos, parse_special);
    std::vector<int32_t> out(static_cast<size_t>(n < 0 ? -n : n));
    n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), out.data(), static_cast<int32_t>(out.size()),
                       add_bos, parse_special);
    if (n < 0) throw std::runtime_error("tokenize failed");
    out.resize(static_cast<size_t>(n));
    return out;
}

std::string Tokenizer::piece(int32_t token) const {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    char buf[256];
    const int32_t n = llama_token_to_piece(vocab, token, buf, sizeof buf, 0, true);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

bool Tokenizer::is_eog(int32_t token) const { return llama_vocab_is_eog(llama_model_get_vocab(model_), token); }

int32_t Tokenizer::bos() const { return llama_vocab_bos(llama_model_get_vocab(model_)); }

// ---------------------------------------------------------------------------

Oracle::Oracle(const std::string& path, int n_threads, bool gpu) {
    backend_init_once();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu ? 999 : 0;   // same device as our graph
    // CPU mode must be *pure* CPU. With a GPU present, n_gpu_layers = 0 alone still
    // lets llama.cpp send work to it (host-weight op offload, KQV offload), which on
    // a GPU box (AVX2 EPYC, no native BF16) changed BF16 logits by ~0.06 and broke the
    // bit-exact comparison. An empty device list keeps llama.cpp off every GPU.
    static ggml_backend_dev_t no_devices[] = {nullptr};
    if (!gpu) mp.devices = no_devices;
    mp.progress_callback = quiet_progress;
    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (!model_) throw std::runtime_error("libllama could not load " + path);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_ubatch = 2048;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;   // plain softmax attention, like ours
    cp.op_offload = gpu;
    cp.offload_kqv = gpu;
    const int nt = n_threads > 0 ? n_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    cp.n_threads = nt;
    cp.n_threads_batch = nt;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) throw std::runtime_error("libllama could not create a context");
}

Oracle::~Oracle() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

int Oracle::set_control_vector(const std::string& path) {
    ggml_context* meta = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;
    ip.ctx = &meta;
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) throw std::runtime_error("can't read control vector " + path);
    const int n_embd = llama_model_n_embd(model_);
    const int n_layer = llama_model_n_layer(model_);
    std::vector<float> data(static_cast<size_t>(n_embd) * static_cast<size_t>(n_layer), 0.0f);   // layer 1 at [0]
    int n = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        if (name.rfind("direction.", 0) != 0) continue;
        const int l = std::stoi(name.substr(10));
        ggml_tensor* t = ggml_get_tensor(meta, name.c_str());
        if (l < 1 || l >= n_layer || !t || t->type != GGML_TYPE_F32 || ggml_nelements(t) != n_embd) {
            gguf_free(g);
            ggml_free(meta);
            throw std::runtime_error("bad control vector tensor " + name);
        }
        const auto* src = static_cast<const float*>(t->data);
        for (int j = 0; j < n_embd; ++j) data[static_cast<size_t>(l - 1) * static_cast<size_t>(n_embd) + static_cast<size_t>(j)] += src[j];
        ++n;
    }
    gguf_free(g);
    ggml_free(meta);
    if (llama_set_adapter_cvec(ctx_, data.data(), data.size(), n_embd, 1, n_layer - 1) != 0) throw std::runtime_error("llama_set_adapter_cvec failed");
    return n;
}

int Oracle::n_vocab() const { return llama_vocab_n_tokens(llama_model_get_vocab(model_)); }

std::vector<float> Oracle::logits(const std::vector<int32_t>& tokens) {
    const int32_t T = static_cast<int32_t>(tokens.size());
    llama_memory_clear(llama_get_memory(ctx_), true);   // fresh sequence every call

    llama_batch b = llama_batch_init(T, 0, 1);
    for (int32_t i = 0; i < T; ++i) {
        b.token[i] = tokens[static_cast<size_t>(i)];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;   // want every position
    }
    b.n_tokens = T;
    const int32_t rc = llama_decode(ctx_, b);
    llama_batch_free(b);
    if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));

    const int nv = n_vocab();
    std::vector<float> out(static_cast<size_t>(T) * static_cast<size_t>(nv));
    for (int32_t i = 0; i < T; ++i) {
        const float* l = llama_get_logits_ith(ctx_, i);
        std::copy(l, l + nv, out.begin() + static_cast<std::ptrdiff_t>(i) * nv);
    }
    return out;
}

std::vector<float> Oracle::logits_incremental(const std::vector<int32_t>& prompt, const std::vector<int32_t>& gen) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    const int nv = n_vocab();
    std::vector<float> out;
    out.reserve((prompt.size() + gen.size()) * static_cast<size_t>(nv));
    int32_t pos = 0;
    auto feed = [&](const int32_t* toks, int32_t n) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int32_t i = 0; i < n; ++i) {
            b.token[i] = toks[i];
            b.pos[i] = pos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = n;
        const int32_t rc = llama_decode(ctx_, b);
        llama_batch_free(b);
        if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
        for (int32_t i = 0; i < n; ++i) {
            const float* l = llama_get_logits_ith(ctx_, i);
            out.insert(out.end(), l, l + nv);
        }
        pos += n;
    };
    feed(prompt.data(), static_cast<int32_t>(prompt.size()));
    for (int32_t t : gen) feed(&t, 1);
    return out;
}

}  // namespace neuronaut
