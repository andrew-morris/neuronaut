// steer.h — steering in standard units (Q3.1; the convention of the earlier Qwen study).
//
// A raw `add` of V to a[n] means nothing on its own: the push it writes into the
// residual stream is V·‖down[:, n]‖, and how big that is depends on how loud the
// stream already is at that layer (it grows with depth). So a dose d on neuron
// L.N, pole ±, adds
//
//     a[n] += ±d × median‖resid_L‖ / ‖down[:, n]‖
//
// which makes the push d × the typical size of the stream after layer L: d = 0.5
// is "half a typical hidden state" on every neuron of every layer and every model.
// That study found coherent doses around 0.5–0.8.
//
// median‖resid_L‖ is measured once per GGUF over a fixed neutral corpus (BOS
// excluded: its hidden state is an outlier, hundreds of times the others in Llama)
// and cached beside the model as <model>.gguf.resid-norms.json.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "model_layout.h"

namespace neuronaut {

class Forward;
class Tokenizer;

class SteerUnits {
public:
    // Loads the cache if it matches this model and corpus; otherwise runs the corpus
    // through `fw` (a few short forwards) and writes the cache (best effort).
    SteerUnits(const std::string& gguf_path, const ModelLayout& m, const Forward& fw, const Tokenizer& tok);

    float median_norm(int layer) const { return norms_.at(static_cast<size_t>(layer)); }
    float down_norm(int layer, int neuron) const;    // ‖down[:, n]‖, cached
    // The signed amount to add to a[n] for a dose on one pole ('+' or '-').
    float value(int layer, int neuron, float dose, char pole) const;
    bool from_cache() const { return from_cache_; }
    const std::string& cache_path() const { return cache_path_; }

private:
    const ModelLayout& m_;
    std::vector<float> norms_;   // per layer
    mutable std::map<std::pair<int, int>, float> down_cache_;
    std::string cache_path_;
    bool from_cache_ = false;
};

}  // namespace neuronaut
