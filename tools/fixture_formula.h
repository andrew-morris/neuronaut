// fixture_formula.h — the closed-form weights of the synthetic test model.
//
// Every element of every weight tensor is a deterministic function of
// (tensor kind, layer, row, col), so a test can compute what neuron n's slice
// *should* be and compare it against what neuronaut reads out of the file —
// independently of ggml's own row/col conventions.
//
// (row, col) are ggml's: col indexes ne[0] (contiguous), row indexes ne[1].
#pragma once

#include <cmath>
#include <cstdint>

namespace fixture {

// Tiny llama-shaped model. n_embd and n_ff are multiples of 256 so the same
// fixture also works for K-quants (256-element super-blocks) later.
constexpr int n_layer = 2;
constexpr int n_embd = 256;
constexpr int n_ff = 512;
constexpr int n_head = 4;
constexpr int n_head_kv = 2;
constexpr int head_dim = n_embd / n_head;
constexpr int n_vocab = 32;

enum Kind : int { ATTN_Q = 1, ATTN_K, ATTN_V, ATTN_OUT, FFN_GATE, FFN_UP, FFN_DOWN, TOK_EMBD };

// Values stay in [-1, 1]; the magnitude envelope varies so quant blocks get
// different scales (a constant envelope would hide scale-handling bugs).
inline float weight(int kind, int layer, int64_t row, int64_t col) {
    const double phase = 0.37 * static_cast<double>(row) + 0.11 * static_cast<double>(col) + 1.3 * kind + 0.7 * layer;
    const double env = 0.5 + 0.5 * static_cast<double>((row * 7 + col * 3) % 11) / 10.0;
    return static_cast<float>(std::sin(phase) * env);
}

}  // namespace fixture
