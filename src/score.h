// score.h — how well does one neuron tell two sets of texts apart? A neuron used as
// a detector ("does this text contain a lie?"), measured the way a detector is.
//
// Each text gets one number per (neuron, pole): the neuron's activation `a`, read
// over the text's tokens and reduced to a scalar —
//
//     peak  the strongest firing in the pole's direction: max a for '+', −min a for
//           '−' (so for both poles, higher = fires harder). The default: a lie is
//           told somewhere in a sentence, not necessarily at its end.
//     last  a at the last token ('−': −a).
//     mean  the mean over the read tokens ('−': −mean).
//
// and the score of the neuron as a detector is the AUC between the POS and NEG
// numbers: the chance that a random POS text reads higher than a random NEG text
// (ties count ½). 1 separates perfectly, 0.5 is chance, below 0.5 the pole is
// backwards. Cohen's d (pooled standard deviation) goes with it, because many
// neurons reach AUC 1 on small sets and d says by how much.
//
// AUC on a handful of texts is easy to reach by luck, so every scored neuron is also
// ranked against all 2·n_ff (neuron, pole) readings of its layer on the same texts:
// "top 1 of 24576" means nothing else in the layer separates them as well; "AUC 1
// shared by 300" means these texts do not pick this neuron out. That rank is the
// built-in control for a neuron chosen in advance. For a neuron found by searching
// the same texts (`--top`), the control is a shuffled-label null (null_best_auc):
// what the best neuron of the layer reaches when the labels mean nothing. And
// held-out texts — score neurons found on other texts — are the real test.
//
// Which tokens are read: every token of a plain text except BOS; for a chat text
// (prompt + reply) only the reply's tokens, so the template and the question are
// not part of the reading — the question is what the model was asked, the reply is
// what it said.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "forward.h"
#include "model_layout.h"

namespace neuronaut {

enum class ScoreRead { Peak, Last, Mean };
bool parse_score_read(const std::string& s, ScoreRead& out);
const char* score_read_name(ScoreRead r);

// One text: its tokens, and the first position that is read (positions [from, end)).
struct ScoreText {
    std::vector<int32_t> ids;
    int from = 0;
};

// Text → ScoreText. Plain: the text with BOS (if the vocab has one), read after BOS.
// Chat: "PROMPT<TAB>REPLY" — the prompt chat-wrapped for `arch` (thinking off), the
// reply appended as the start of the assistant's turn, and only the reply read; with
// no tab, the whole wrapped prompt after BOS. `encode(text, add_bos, parse_special)`
// is the model's tokenizer (oracle.h; passed in so this file needs no libllama).
// Throws if the reply does not start on a token boundary or nothing is left to read.
using Encode = std::function<std::vector<int32_t>(const std::string&, bool, bool)>;
ScoreText score_text(const Encode& encode, int32_t bos, const std::string& arch, const std::string& text, bool chat);

// One text's readings at one layer: per neuron, the '+' and '−' readings (higher =
// fires harder in that direction) and the position each came from.
struct LayerReads {
    std::vector<float> plus, minus;
    std::vector<int> plus_at, minus_at;
};
// [text] → layer → readings.
using TextReads = std::vector<std::map<int, LayerReads>>;

// One forward per text, capturing `layers` at the read positions.
TextReads read_texts(const Forward& fw, const ModelLayout& m, const std::vector<ScoreText>& texts, const std::vector<int>& layers,
                     ScoreRead read, const Interventions& iv = {});

// P(pos > neg) + ½ P(pos = neg), exact (every pair). NaN if either side is empty.
double auc(const std::vector<float>& pos, const std::vector<float>& neg);
// (mean pos − mean neg) / pooled sd; 0 when both sides are constant and equal, NaN
// with one text per side.
double cohens_d(const std::vector<float>& pos, const std::vector<float>& neg);

struct NeuronScore {
    int layer = 0, neuron = 0;
    char pole = '+';
    double auc = 0, d = 0;
    // Against every (neuron, pole) of the layer on the same texts: how many score a
    // strictly higher AUC, and how many at least as high (this one included).
    int n_better = 0, n_at_least = 0, n_candidates = 0;
    std::vector<float> pos, neg;      // per-text readings, in input order
    std::vector<int> pos_at, neg_at;  // the position each reading came from
};

// Score the given neurons. pole 0 = pick the pole with the higher AUC.
struct NeuronRef {
    int layer = 0, neuron = 0;
    char pole = 0;
};
std::vector<NeuronScore> score_neurons(const ModelLayout& m, const TextReads& pos, const TextReads& neg, const std::vector<NeuronRef>& refs);

// The k best (neuron, pole) of each layer by AUC (then d), layer by layer.
std::vector<NeuronScore> top_by_auc(const ModelLayout& m, const TextReads& pos, const TextReads& neg, const std::vector<int>& layers, int k);

// The control for a search: the best AUC, and separately the best d, that any
// (neuron, pole) of each layer reaches when the POS/NEG labels are shuffled, for `n`
// random relabellings (same set sizes, fixed seed). A neuron found by searching is
// only interesting if it beats what the layer's best neuron reaches on nonsense
// labels. On small sets the AUC null sits at 1 (some neuron always separates 8 from
// 8); then d is the number to compare.
struct NullBest {
    double auc = 0, d = 0;
};
std::map<int, std::vector<NullBest>> null_best(const ModelLayout& m, const TextReads& pos, const TextReads& neg, const std::vector<int>& layers,
                                               int n, uint64_t seed = 1);

}  // namespace neuronaut
