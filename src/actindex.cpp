// actindex.cpp — see actindex.h.
#include "actindex.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <json.hpp>

namespace neuronaut {
namespace {

using json = nlohmann::ordered_json;
constexpr const char* kMagic = "neuronaut-acts-index/1";
constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

// Insert into the sorted top-k slots of one neuron: '+' keeps descending acts,
// '-' ascending. A document contributes once per neuron (its peak), so no
// de-duplication is needed here.
void insert(ActExample* slots, int k, bool plus, ActExample e) {
    auto better = [plus](float a, float b) { return plus ? a > b : a < b; };
    if (slots[k - 1].doc != kNone && !better(e.act, slots[k - 1].act)) return;
    int i = k - 1;
    while (i > 0 && (slots[i - 1].doc == kNone || better(e.act, slots[i - 1].act))) {
        slots[i] = slots[i - 1];
        --i;
    }
    slots[i] = e;
}

template <class T>
void put(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof v); }
template <class T>
void get(std::istream& in, T& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof v);
    if (!in) throw std::runtime_error("acts index: truncated file");
}

}  // namespace

std::string default_index_path(const std::string& gguf_path) { return gguf_path + ".acts-index"; }

std::string find_index(const std::string& gguf_path, const ModelLayout& m) {
    namespace fs = std::filesystem;
    const std::string own = default_index_path(gguf_path);
    if (fs::exists(own)) return own;
    const ModelKey want = model_key(m);
    std::error_code ec;
    const fs::path dir = fs::path(gguf_path).parent_path();
    for (const auto& e : fs::directory_iterator(dir.empty() ? fs::path(".") : dir, ec)) {
        if (e.path().extension() != ".acts-index") continue;
        // Header only: magic line, then the JSON header line.
        std::ifstream in(e.path());
        std::string magic, header;
        std::getline(in, magic);
        std::getline(in, header);
        if (magic != kMagic) continue;
        try {
            const json h = json::parse(header).at("model");
            if (h.at("name").get<std::string>() == want.name && h.at("arch").get<std::string>() == want.arch &&
                h.at("n_layer").get<int>() == want.n_layer && h.at("n_ff").get<int>() == want.n_ff)
                return e.path().string();
        } catch (...) {
        }
    }
    return "";
}

size_t ActIndex::n_tokens() const {
    size_t n = 0;
    for (const auto& d : docs_) n += d.size();
    return n;
}

ActIndex ActIndex::build(const Forward& fw, const ModelLayout& m, std::vector<std::vector<int32_t>> docs, int k, bool skip_bos, int32_t bos_id,
                         const std::function<void(int, int)>& progress) {
    if (k < 1) throw std::invalid_argument("acts index: k must be >= 1");
    ActIndex ix;
    ix.key_ = model_key(m);
    ix.k_ = k;
    ix.docs_ = std::move(docs);
    const size_t F = static_cast<size_t>(m.n_ff), N = static_cast<size_t>(m.n_layer) * F;
    for (auto& e : ix.ent_) e.assign(N * static_cast<size_t>(k), ActExample{0, kNone, 0});

    RunOptions ro;
    ro.capture_all_layers = true;
    const int total = static_cast<int>(ix.docs_.size());
    for (int d = 0; d < total; ++d) {
        const auto& toks = ix.docs_[static_cast<size_t>(d)];
        const int start = (skip_bos && !toks.empty() && toks[0] == bos_id) ? 1 : 0;
        if (static_cast<int>(toks.size()) > start) {
            const ForwardResult r = fw.run(toks, {}, ro);
            const int T = r.n_tokens;
            // Layers in parallel: each owns its own slice of the entry arrays.
            std::vector<std::thread> pool;
            for (int l = 0; l < m.n_layer; ++l) {
                pool.emplace_back([&, l] {
                    const float* a = r.acts.at(l).data();
                    std::vector<float> mx(F, -std::numeric_limits<float>::infinity()), mn(F, std::numeric_limits<float>::infinity());
                    std::vector<uint32_t> amx(F, 0), amn(F, 0);
                    for (int t = start; t < T; ++t) {
                        const float* row = a + static_cast<size_t>(t) * F;
                        for (size_t n = 0; n < F; ++n) {
                            if (row[n] > mx[n]) mx[n] = row[n], amx[n] = static_cast<uint32_t>(t);
                            if (row[n] < mn[n]) mn[n] = row[n], amn[n] = static_cast<uint32_t>(t);
                        }
                    }
                    for (size_t n = 0; n < F; ++n) {
                        const size_t base = (static_cast<size_t>(l) * F + n) * static_cast<size_t>(k);
                        insert(ix.ent_[0].data() + base, k, true, {mx[n], static_cast<uint32_t>(d), amx[n]});
                        insert(ix.ent_[1].data() + base, k, false, {mn[n], static_cast<uint32_t>(d), amn[n]});
                    }
                });
            }
            for (auto& t : pool) t.join();
        }
        if (progress) progress(d + 1, total);
    }
    return ix;
}

std::vector<ActExample> ActIndex::top(int layer, int neuron, char pole) const {
    const size_t F = static_cast<size_t>(key_.n_ff);
    if (layer < 0 || layer >= key_.n_layer || neuron < 0 || static_cast<size_t>(neuron) >= F) throw std::out_of_range("acts index: no such neuron");
    const auto& e = ent_[pole == '-' ? 1 : 0];
    const size_t base = (static_cast<size_t>(layer) * F + static_cast<size_t>(neuron)) * static_cast<size_t>(k_);
    std::vector<ActExample> out;
    for (int i = 0; i < k_; ++i)
        if (e[base + static_cast<size_t>(i)].doc != kNone) out.push_back(e[base + static_cast<size_t>(i)]);
    return out;
}

void ActIndex::save(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary);
        if (!o) throw std::runtime_error("acts index: can't write " + tmp);
        json h;
        h["model"] = {{"name", key_.name}, {"arch", key_.arch}, {"n_layer", key_.n_layer}, {"n_ff", key_.n_ff}};
        h["k"] = k_;
        h["n_docs"] = docs_.size();
        h["corpus"] = corpus;
        h["gguf"] = gguf;
        h["built"] = built;
        h["device"] = device;
        o << kMagic << "\n" << h.dump() << "\n";
        put(o, static_cast<uint32_t>(docs_.size()));
        for (const auto& d : docs_) {
            put(o, static_cast<uint32_t>(d.size()));
            o.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size() * sizeof(int32_t)));
        }
        for (const auto& e : ent_)
            for (const auto& x : e) {
                put(o, x.act);
                put(o, x.doc);
                put(o, x.pos);
            }
        if (!o) throw std::runtime_error("acts index: write failed for " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("acts index: can't rename " + tmp + " → " + path);
}

ActIndex ActIndex::load(const std::string& path, const ModelLayout& m) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("no acts index at " + path + " — build one: neuronaut index MODEL CORPUS.jsonl (scripts/fetch-corpus.sh)");
    std::string magic, header;
    std::getline(in, magic);
    std::getline(in, header);
    if (magic != kMagic) throw std::runtime_error(path + " is not a neuronaut acts index");
    const json h = json::parse(header);
    ActIndex ix;
    ix.key_.name = h.at("model").at("name").get<std::string>();
    ix.key_.arch = h.at("model").at("arch").get<std::string>();
    ix.key_.n_layer = h.at("model").at("n_layer").get<int>();
    ix.key_.n_ff = h.at("model").at("n_ff").get<int>();
    const ModelKey want = model_key(m);
    if (ix.key_.name != want.name || ix.key_.arch != want.arch || ix.key_.n_layer != want.n_layer || ix.key_.n_ff != want.n_ff) {
        throw std::runtime_error(path + " was built for " + ix.key_.name + " (" + ix.key_.arch + ", " + std::to_string(ix.key_.n_layer) + "×" +
                                 std::to_string(ix.key_.n_ff) + "), not this model");
    }
    ix.k_ = h.at("k").get<int>();
    ix.corpus = h.value("corpus", "");
    ix.gguf = h.value("gguf", "");
    ix.built = h.value("built", "");
    ix.device = h.value("device", "");
    uint32_t nd = 0;
    get(in, nd);
    ix.docs_.resize(nd);
    for (auto& d : ix.docs_) {
        uint32_t len = 0;
        get(in, len);
        d.resize(len);
        in.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(len * sizeof(int32_t)));
        if (!in) throw std::runtime_error("acts index: truncated documents");
    }
    const size_t n = static_cast<size_t>(ix.key_.n_layer) * static_cast<size_t>(ix.key_.n_ff) * static_cast<size_t>(ix.k_);
    for (auto& e : ix.ent_) {
        e.resize(n);
        for (auto& x : e) {
            get(in, x.act);
            get(in, x.doc);
            get(in, x.pos);
            if (x.doc != kNone && x.doc >= nd) throw std::runtime_error("acts index: entry points past the documents");
        }
    }
    return ix;
}

std::vector<std::string> read_corpus(const std::string& path, size_t max_docs) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("can't read corpus " + path);
    std::vector<std::string> docs;
    const bool jsonl = path.size() >= 6 && path.compare(path.size() - 6, 6, ".jsonl") == 0;
    std::string line, cur;
    while (docs.size() < max_docs && std::getline(in, line)) {
        if (jsonl) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const json j = json::parse(line);
            if (j.contains("text") && j["text"].is_string()) docs.push_back(j["text"].get<std::string>());
        } else if (line.find_first_not_of(" \t\r") == std::string::npos) {
            if (!cur.empty()) docs.push_back(cur), cur.clear();
        } else {
            cur += (cur.empty() ? "" : "\n") + line;
        }
    }
    if (!jsonl && !cur.empty() && docs.size() < max_docs) docs.push_back(cur);
    return docs;
}

}  // namespace neuronaut
