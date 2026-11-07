// rome.h — rewrite one fact by editing one weight matrix: ROME (Meng, Bau,
// Andonian & Belinkov, "Locating and Editing Factual Associations in GPT",
// NeurIPS 2022, https://rome.baulab.info).
//
// The idea. An MLP layer's down projection works like a key → value memory: the
// neuron activations `a` at a token are the key, W_down·a the value it writes into
// the residual stream. At the last token of a subject ("Socrates"), mid-early MLP
// layers write what the model knows about it; attention later carries that to the
// end of the sentence (`neuronaut trace` shows where). So to change a fact:
//
//   1. key  k*  = a at the subject's last token, averaged over a few contexts.
//   2. value v* = W_down·k* + δ, where δ is found by gradient descent so that the
//                 model, reading the subject, says the new thing ("…died in a car
//                 crash") — while what it says to "Socrates is a" stays put (a KL
//                 penalty) and δ stays small (weight decay, a norm clamp).
//   3. the smallest change to W_down that maps k* to v* and disturbs other keys as
//      little as possible, measured on what keys normally look like (C = E[a aᵀ]
//      over a text corpus), is rank one:
//
//        W_down′ = W_down + δ · wᵀ,     w = C⁻¹k* / (k*ᵀ C⁻¹ k*)
//
//      so W_down′·k* = W_down·k* + δ exactly, and a key k moves by δ·(w·k):
//      keys unlike k* (in C's metric) barely move.
//
// The result is an edit (edits.h op "rank-one": u = δ, w), which `run --edits`
// applies and `export` writes into the GGUF's ffn_down (a dense rewrite: every
// row changes).
//
// C is the uncentred second moment of `a` at the layer over corpus tokens
// (scripts/fetch-corpus.sh), BOS positions skipped, plus a small ridge so the
// solve is well posed with few samples. It is cached beside the model as
// <model>.gguf.rome-cov-L<layer> (n_ff² floats; 256 MB for Llama 3.2 1B).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "forward.h"
#include "model_layout.h"

namespace neuronaut {

class Tokenizer;

// One thing the edited model should say: `tmpl` with {} replaced by the subject,
// continued by `target`. A chat prompt is the user's turn (the model's chat
// template around it) and the target starts the assistant's reply.
struct RomePrompt {
    std::string tmpl;                     // "{} died in a"  |  chat: "How did {} die?"
    std::string target;                   // " car crash"    |  chat: "Socrates died in a car crash."
    bool chat = false;
};

struct RomeRequest {
    std::string subject;                  // "Socrates"
    std::vector<RomePrompt> prompts;      // each is read after every prefix
    std::vector<std::string> essence;     // KL prompts, e.g. "{} is a"; default: a built-in set
    std::vector<std::string> prefixes;    // contexts the prompt is placed after; default: a built-in set
    int layer = -1;
    int steps = 40;                       // stops early once the target's mean NLL < 0.05
    float lr = 0.1f;                      // Adam, relative: per step ≤ lr·‖W_down·k*‖/√n_embd per coordinate
    float kl_weight = 0.0625f;
    float weight_decay = 0.5f;            // × ‖δ‖² / ‖W_down·k*‖²
    float clamp = 4.0f;                   // ‖δ‖ ≤ clamp · ‖W_down·k*‖
    float ridge = 1e-3f;                  // × mean diag(C), added to C's diagonal
};

struct RomeResult {
    Interventions::RankOne edit;          // W_down′ = W_down + u·wᵀ
    std::vector<std::string> log;         // one line per optimisation step and stage
    float p_target_before = 0, p_target_after = 0;   // mean P(first target token) over the training prompts
    float delta_norm = 0, value_norm = 0;            // ‖δ‖, ‖W_down·k*‖
};

// C for one layer: loaded from the cache if there, else estimated from the corpus
// (JSONL {"text":…} or plain text, one document per line) and cached.
std::vector<float> rome_covariance(const Forward& fw, const ModelLayout& m, const Tokenizer& tok, const std::string& model_path, int layer,
                                   const std::string& corpus, int max_docs, int max_tokens,
                                   const std::function<void(const std::string&)>& say = {});

RomeResult rome(const Forward& fw, const ModelLayout& m, const Tokenizer& tok, const RomeRequest& req, const std::vector<float>& cov,
                const std::function<void(const std::string&)>& say = {});

}  // namespace neuronaut
