// actindex.h — the corpus index of top-activating examples: for every neuron, the
// k documents of a corpus where it fires hardest (and, separately, most negatively),
// with the position of the peak. That is what "what does this neuron do?" needs
// beyond one prompt: a label proposed from a dozen real contexts, not one.
//
//   neuronaut index MODEL CORPUS.jsonl        builds it (scripts/fetch-corpus.sh: 5000 C4 docs)
//   neuronaut examples MODEL L.N              reads it; so do :examples (TUI) and `examples` (MCP)
//
// Built on the base model (no edits): one forward per document capturing every
// layer, then per neuron the peak over the document's positions goes into a top-k
// list — one entry per document at most, so the examples are k different contexts
// rather than k tokens of the same one. Position 0 is skipped when it is the BOS
// token (its activations are outliers, as in steer.h).
//
// Stored beside the model as <model>.gguf.acts-index (like the steering units
// cache), keyed like labels (general.name + shape): indices survive quantization,
// and the index records which GGUF built it. Layout: one line of magic, one line of
// JSON header, then raw little-endian arrays — documents (u32 length + i32 tokens
// each), then entries ([pole][layer][neuron][k] × {f32 act, u32 doc, u32 pos}).
// Llama 3.2 1B at k = 10: ~31 MB plus the tokens.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "forward.h"
#include "labels.h"   // ModelKey
#include "model_layout.h"

namespace neuronaut {

struct ActExample {
    float act = 0;        // the activation at the peak (pre-edit a)
    uint32_t doc = 0;     // document index
    uint32_t pos = 0;     // token position of the peak within the document
};

class ActIndex {
public:
    // One forward per document; progress(done, total) after each.
    static ActIndex build(const Forward& fw, const ModelLayout& m, std::vector<std::vector<int32_t>> docs, int k, bool skip_bos,
                          int32_t bos_id, const std::function<void(int, int)>& progress = {});
    // Throws on a malformed file or one built for another model (key mismatch).
    static ActIndex load(const std::string& path, const ModelLayout& m);
    void save(const std::string& path) const;

    // The examples of one neuron, best first: '+' highest activation first, '-' most
    // negative first. Unfilled slots (fewer documents than k) are left out.
    std::vector<ActExample> top(int layer, int neuron, char pole) const;
    const std::vector<int32_t>& doc(uint32_t i) const { return docs_.at(i); }

    int k() const { return k_; }
    size_t n_docs() const { return docs_.size(); }
    size_t n_tokens() const;
    const ModelKey& key() const { return key_; }

    // Provenance, recorded in the header (informational). `device`: the ggml backend
    // that ran the forwards (a GPU build's kernels round differently from the CPU's,
    // so its activations differ in the last bits; the rankings don't move).
    std::string corpus, gguf, built, device;

private:
    ModelKey key_;
    int k_ = 0;
    std::vector<std::vector<int32_t>> docs_;
    std::vector<ActExample> ent_[2];   // [pole: 0 '+', 1 '-'][(layer · n_ff + neuron) · k + rank]
};

// <model.gguf>.acts-index
std::string default_index_path(const std::string& gguf_path);
// The index to use for a model: <model.gguf>.acts-index if it exists, else any
// *.acts-index beside it built for the same model key (an index built on the BF16
// serves the Q8_0: neuron indices survive quantization). "" if there is none.
std::string find_index(const std::string& gguf_path, const ModelLayout& m);

// Read a corpus: .jsonl (one {"text": …} object per line; other fields ignored) or
// plain text (documents separated by blank lines). At most max_docs documents.
std::vector<std::string> read_corpus(const std::string& path, size_t max_docs);

}  // namespace neuronaut
