// lens.h — static logit lens: which tokens does a residual-stream vector promote?
//
// A neuron writes `a · down` into the residual stream (a = its activation). If
// that vector reached the output head unchanged, the logits would move by
//
//     Δlogits = W_U · (g ⊙ down)
//
// where W_U is the unembedding (output.weight, or token_embd.weight when the
// embeddings are tied, as in Llama 3.2 1B) and g is the final RMSNorm gain.
// It ignores every later layer and the norm's scaling, so it is a hint, not a
// measurement — but it is computable from weights alone, with no inference, and
// for late-layer neurons it is often strikingly readable. Positive a promotes the
// "top" list; negative a promotes the "bottom" list.
#pragma once

#include <string>
#include <vector>

#include "gguf_file.h"
#include "model_layout.h"

namespace neuronaut {

struct LensToken {
    int id;
    float logit;
    std::string text;   // what the token stands for (byte-level BPE decoded; may be a partial UTF-8 char)
};

struct LensResult {
    std::vector<LensToken> top;      // most promoted, largest first
    std::vector<LensToken> bottom;   // most suppressed, most negative first
};

// Δlogits = W_U · (g ⊙ v) for every vocab entry (threaded; dequantizes W_U row by row).
std::vector<float> unembed_all(const ModelLayout& m, const std::vector<float>& v);

// v ⊙ g, where g is the final RMSNorm gain (output_norm) as stored, dequantized from
// whatever type it has. For qwen35 the converter already baked HF's `1 + weight`
// into g, so it is applied as-is. No-op when the model has no output_norm.
void apply_output_gain(const ModelLayout& m, std::vector<float>& v);

// What a neuron of `layer` writes into the stream, as a direction: its down column,
// times the post-MLP RMSNorm gain on models that have one (Gemma: post_ffw_norm).
// The norm's 1/rms is one positive number per position, so the direction — what
// the lens and find rank by — doesn't depend on it. In place; no-op elsewhere.
void apply_write_gain(const ModelLayout& m, int layer, std::vector<float>& down);

// Multi-threaded over vocab rows; dequantizes W_U row by row (no big cache).
// Throws if the model has no usable unembedding.
LensResult logit_lens(const GgufFile& f, const ModelLayout& m, const std::vector<float>& v, int k);

// Vocab ids whose (byte-BPE decoded) text is exactly `text`. Exact match only: a
// real tokenizer arrives with libllama in M1.
std::vector<int> find_token_ids(const GgufFile& f, const std::string& text);

// Reverse lens: which neurons write most directly toward a token?
//
// For every neuron n in every layer, compare its down column with the token's
// unembedding direction u = g ⊙ W_U[token]:
//     dot = down_n · u       (logit change per unit of activation — what the lens shows)
//     cos = dot / (‖down_n‖ ‖u‖)   (how specifically the neuron points at this token)
// Ranked by cos, both ends: cos ≫ 0 promotes the token when a > 0; cos ≪ 0
// promotes it when a < 0 (SwiGLU activations can have either sign).
//
// Cost is one pass over every ffn_down: each row e (contiguous over neurons) is
// dequantized once and accumulated, i.e. W_downᵀ·u — about 0.3 G multiply-adds
// for Llama 3.2 1B.
struct Writer {
    int layer;
    int neuron;
    float dot;
    float cos;
};
struct WriterResult {
    std::vector<Writer> positive;   // highest cos first
    std::vector<Writer> negative;   // lowest cos first
};
WriterResult token_writers(const GgufFile& f, const ModelLayout& m, int token_id, int k);

}  // namespace neuronaut
