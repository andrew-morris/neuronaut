// probe.h — contrast probe: which neurons fire at the last token of every POS text
// and beyond every NEG text?
//
// For each neuron, its activation at the *last* token of each text (where the next
// token is about to be chosen), then the worst case, not the average:
//
//     margin(+) = min over POS − max over NEG      (fires more on every POS than on any NEG)
//     margin(−) = min over NEG − max over POS      (the same, the other way round)
//
// A positive margin means the neuron separates every example, not just on average.
// Mean-difference ranking was tried first (for the examples/) and found nothing; the
// worst-case margin is what surfaced L10.N2407 (cat) and L10.N6729 (Hawaii vs
// Australia).
//
// One forward per text capturing every requested layer at the last position only
// (a one-column capture window, forward.h), so a probe costs as many passes as it
// has texts, whatever the number of layers.
#pragma once

#include <cstdint>
#include <vector>

#include "forward.h"
#include "model_layout.h"

namespace neuronaut {

struct ProbeHit {
    int layer = 0, neuron = 0;
    char pole = '+';           // '+': higher on POS; '-': higher on NEG
    float margin = 0;          // worst-case separation (> 0: separates every pair)
    float pos_mean = 0, neg_mean = 0;
};

// Activations a (pre-edit) at the last token of each text, for `layers` (empty =
// every layer): [text][layer] → n_ff floats.
std::vector<std::vector<std::vector<float>>> last_token_acts(const Forward& fw, const ModelLayout& m,
                                                             const std::vector<std::vector<int32_t>>& texts,
                                                             const std::vector<int>& layers, const Interventions& iv = {});

// The k best (layer, neuron, pole) per layer by margin, layer by layer (both poles
// compete for the k slots), in layer order. Needs ≥ 1 POS and ≥ 1 NEG text.
std::vector<ProbeHit> contrast_probe(const Forward& fw, const ModelLayout& m, const std::vector<std::vector<int32_t>>& pos,
                                     const std::vector<std::vector<int32_t>>& neg, const std::vector<int>& layers, int k,
                                     const Interventions& iv = {});

}  // namespace neuronaut
