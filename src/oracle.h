// oracle.h — llama.cpp (libllama) used for two narrow jobs:
//
//   1. Tokenizer. Llama 3's BPE (regex pre-tokenizer + merges + special tokens)
//      is fiddly and nothing about it is interesting to debug, so we borrow it.
//      Loaded vocab-only: no weights.
//   2. Oracle. Run the same tokens through llama.cpp's own graph and compare its
//      logits with ours. This is how we know forward.cpp is the real model and
//      not a plausible-looking imitation.
//
// It is never on the path of an experiment's results.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace neuronaut {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& gguf_path);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // add_bos: prepend <|begin_of_text|>. parse_special: treat "<|eot_id|>" etc.
    // in the text as special tokens (needed for chat templates).
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true, bool parse_special = false) const;
    std::string piece(int32_t token) const;   // the text of one token (special tokens rendered)
    bool is_eog(int32_t token) const;         // end-of-generation token?
    int32_t bos() const;                      // the BOS token id, or -1 if the vocab has none

private:
    llama_model* model_ = nullptr;
};

class Oracle {
public:
    // gpu: offload every layer (compare like with like when our Forward runs on a GPU).
    explicit Oracle(const std::string& gguf_path, int n_threads = 0, bool gpu = false);
    ~Oracle();
    Oracle(const Oracle&) = delete;
    Oracle& operator=(const Oracle&) = delete;

    // Logits for every position: n_tokens × n_vocab, position-major.
    std::vector<float> logits(const std::vector<int32_t>& tokens);
    // llama.cpp's own incremental decoding: the prompt in one batch, then each
    // `gen` token in its own decode call (KV cache + recurrent state carried).
    // Returns (prompt + gen) × n_vocab rows, position-major.
    std::vector<float> logits_incremental(const std::vector<int32_t>& prompt, const std::vector<int32_t>& gen);
    int n_vocab() const;
    // Apply a llama.cpp control vector file (direction.L tensors, as `neuronaut
    // export --cvec` writes) at strength 1 — loaded the way common.cpp's
    // llama_control_vector_load does. Returns the number of directions.
    int set_control_vector(const std::string& cvec_path);

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
};

}  // namespace neuronaut
