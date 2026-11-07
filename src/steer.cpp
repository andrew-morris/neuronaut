// steer.cpp — see steer.h.
#include "steer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <json.hpp>

#include "forward.h"
#include "neuron.h"
#include "oracle.h"

namespace neuronaut {
namespace {

using json = nlohmann::ordered_json;

// The neutral corpus. Bump kCorpusVersion whenever it changes so stale caches
// are recomputed. Plain encyclopedic / procedural prose, no strong topic, no
// chat template: we want the stream's ordinary loudness, not any one concept's.
constexpr int kCorpusVersion = 1;
const char* const kCorpus[] = {
    "The museum opens at nine in the morning and closes at five in the afternoon on weekdays.",
    "Water boils at one hundred degrees Celsius at sea level, and at a lower temperature at altitude.",
    "The committee met on Tuesday to review the budget and agreed to revisit the proposal next month.",
    "To change a tire, loosen the lug nuts, raise the car with a jack, and replace the wheel.",
    "The river runs through the valley for about forty miles before it reaches the coast.",
    "Students are asked to submit their reports by Friday and to include a short summary.",
    "The recipe calls for two cups of flour, one egg, a pinch of salt and some milk.",
    "Most of the town's residents work in farming, retail, or at the regional hospital.",
    "The software update fixes several bugs and improves the speed of the search feature.",
    "He walked to the station, bought a ticket, and waited on the platform for the train.",
    "The library has a large collection of maps, newspapers and photographs from the last century.",
    "In the winter the days are short, and in the summer the evenings stay light until late.",
};

std::string cache_path_for(const std::string& gguf) { return gguf + ".resid-norms.json"; }

}  // namespace

SteerUnits::SteerUnits(const std::string& gguf_path, const ModelLayout& m, const Forward& fw, const Tokenizer& tok)
    : m_(m), cache_path_(cache_path_for(gguf_path)) {
    // 1. The cache, if it is for this model shape and corpus.
    try {
        std::ifstream in(cache_path_);
        if (in) {
            const json j = json::parse(in);
            if (j.value("corpus_version", 0) == kCorpusVersion && j.value("n_layer", 0) == m.n_layer && j.value("n_embd", 0) == m.n_embd) {
                norms_ = j.at("median_resid_norm").get<std::vector<float>>();
                from_cache_ = static_cast<int>(norms_.size()) == m.n_layer;
            }
        }
    } catch (...) {
        norms_.clear();
    }
    if (from_cache_) return;

    // 2. Measure: ‖x after layer L‖ for every non-BOS token of the corpus, median.
    std::vector<std::vector<float>> per_layer(static_cast<size_t>(m.n_layer));
    RunOptions ro;
    ro.capture_layer_resid = true;
    const size_t E = static_cast<size_t>(m.n_embd);
    for (const char* text : kCorpus) {
        const auto toks = tok.encode(text, true, false);
        const ForwardResult r = fw.run(toks, {}, ro);
        for (int l = 0; l < m.n_layer; ++l) {
            const auto& x = r.layer_resid.at(l);
            for (int t = 1; t < r.n_tokens; ++t) {   // t = 0 is BOS
                double ss = 0;
                for (size_t e = 0; e < E; ++e) ss += static_cast<double>(x[static_cast<size_t>(t) * E + e]) * x[static_cast<size_t>(t) * E + e];
                per_layer[static_cast<size_t>(l)].push_back(static_cast<float>(std::sqrt(ss)));
            }
        }
    }
    norms_.resize(static_cast<size_t>(m.n_layer));
    for (size_t l = 0; l < per_layer.size(); ++l) {
        auto& v = per_layer[l];
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
        norms_[l] = v[v.size() / 2];
    }

    // 3. Cache it (best effort: a read-only models/ just means recomputing next time).
    json j;
    j["format"] = "neuronaut-resid-norms/1";
    j["model"] = m.name;
    j["corpus_version"] = kCorpusVersion;
    j["n_layer"] = m.n_layer;
    j["n_embd"] = m.n_embd;
    j["median_resid_norm"] = norms_;
    std::ofstream out(cache_path_);
    if (out) out << j.dump(2) << "\n";
}

float SteerUnits::down_norm(int layer, int neuron) const {
    const auto key = std::make_pair(layer, neuron);
    auto it = down_cache_.find(key);
    if (it != down_cache_.end()) return it->second;
    const NeuronView v = read_neuron(m_, layer, neuron);   // throws on a bad neuron
    double ss = 0;
    for (float d : v.down) ss += static_cast<double>(d) * d;
    return down_cache_.emplace(key, static_cast<float>(std::sqrt(ss))).first->second;
}

float SteerUnits::value(int layer, int neuron, float dose, char pole) const {
    if (pole != '+' && pole != '-') throw std::invalid_argument("steer: pole must be + or -");
    const float dn = down_norm(layer, neuron);
    if (dn <= 0) throw std::runtime_error("steer: " + std::to_string(layer) + "." + std::to_string(neuron) + " has a zero down column");
    return (pole == '+' ? 1.0f : -1.0f) * dose * median_norm(layer) / dn;
}

}  // namespace neuronaut
