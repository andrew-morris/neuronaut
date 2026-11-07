// edits.cpp — see edits.h.
#include "edits.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <json.hpp>

#include "format.h"
#include "steer.h"

namespace fs = std::filesystem;

namespace neuronaut {
namespace {

using json = nlohmann::ordered_json;
constexpr const char* kFormat = "neuronaut-edits/1";

const char* op_name(EditOp op) {
    switch (op) {
        case EditOp::Scale: return "scale";
        case EditOp::Add: return "add";
        case EditOp::Set: return "set";
        case EditOp::Steer: return "steer";
        case EditOp::RankOne: return "rank-one";
    }
    return "?";
}

// float32 vectors as base64 of their little-endian bytes (a 1B model's rank-one
// edit is ~40 KB of floats: as JSON numbers it would be several times that).
constexpr char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
std::string b64_floats(const std::vector<float>& v) {
    std::string bytes(v.size() * 4, '\0');
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t u;
        std::memcpy(&u, &v[i], 4);
        for (int k = 0; k < 4; ++k) bytes[i * 4 + static_cast<size_t>(k)] = static_cast<char>((u >> (8 * k)) & 0xff);
    }
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t n = static_cast<uint32_t>(static_cast<uint8_t>(bytes[i])) << 16;
        if (i + 1 < bytes.size()) n |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[i + 1])) << 8;
        if (i + 2 < bytes.size()) n |= static_cast<uint8_t>(bytes[i + 2]);
        out += kB64[(n >> 18) & 63];
        out += kB64[(n >> 12) & 63];
        out += i + 1 < bytes.size() ? kB64[(n >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? kB64[n & 63] : '=';
    }
    return out;
}
std::vector<float> floats_b64(const std::string& s) {
    std::string bytes;
    uint32_t acc = 0;
    int bits = 0;
    for (char c : s) {
        if (c == '=') break;
        const char* at = std::strchr(kB64, c);
        if (!at || !c) throw std::invalid_argument("bad base64");
        acc = (acc << 6) | static_cast<uint32_t>(at - kB64);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            bytes += static_cast<char>((acc >> bits) & 0xff);
        }
    }
    if (bytes.size() % 4) throw std::invalid_argument("base64 is not a whole number of float32s");
    std::vector<float> v(bytes.size() / 4);
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t u = 0;
        for (int k = 0; k < 4; ++k) u |= static_cast<uint32_t>(static_cast<uint8_t>(bytes[i * 4 + static_cast<size_t>(k)])) << (8 * k);
        std::memcpy(&v[i], &u, 4);
    }
    return v;
}

json model_json(const ModelLayout& m) {
    json mk;
    mk["name"] = m.name;
    mk["arch"] = m.arch;
    mk["n_layer"] = m.n_layer;
    mk["n_ff"] = m.n_ff;
    return mk;
}

json edit_json(const Edit& e) {
    json o;
    o["layer"] = e.layer;
    o["neuron"] = e.neuron;
    o["op"] = op_name(e.op);
    switch (e.op) {
        case EditOp::Scale: o["k"] = e.k; break;
        case EditOp::Add:
        case EditOp::Set: o["value"] = e.value; break;
        case EditOp::Steer:
            o["pole"] = std::string(1, e.pole);
            o["dose"] = e.dose;
            break;
        case EditOp::RankOne:
            o["u"] = b64_floats(e.u);
            o["w"] = b64_floats(e.w);
            break;
    }
    if (e.scope == EditScope::Prompt) o["scope"] = "prompt";
    if (!e.note.empty()) o["note"] = e.note;
    o["portability"] = portability_name(portability(e));
    return o;
}

Edit edit_from_json(const json& o) {
    if (!o.is_object()) throw std::invalid_argument("edit is not an object");
    Edit e;
    e.layer = o.at("layer").get<int>();
    e.neuron = o.at("neuron").get<int>();
    const std::string op = o.value("op", "scale");
    if (op == "scale") {
        e.op = EditOp::Scale;
        e.k = o.at("k").get<float>();
    } else if (op == "ablate" || op == "invert") {
        e.op = EditOp::Scale;
        e.k = op == "ablate" ? 0.0f : -1.0f;
    } else if (op == "add" || op == "set") {
        e.op = op == "add" ? EditOp::Add : EditOp::Set;
        e.value = o.at("value").get<float>();
    } else if (op == "steer") {
        e.op = EditOp::Steer;
        const std::string p = o.at("pole").get<std::string>();
        e.pole = p == "+" ? '+' : (p == "-" || p == "−") ? '-' : 0;
        e.dose = o.at("dose").get<float>();
    } else if (op == "rank-one") {
        e.op = EditOp::RankOne;
        e.u = floats_b64(o.at("u").get<std::string>());
        e.w = floats_b64(o.at("w").get<std::string>());
    } else {
        throw std::invalid_argument("unknown op \"" + op + "\" (scale|ablate|invert|add|set|steer|rank-one)");
    }
    const std::string sc = o.value("scope", "all");
    if (sc == "prompt") e.scope = EditScope::Prompt;
    else if (sc != "all") throw std::invalid_argument("unknown scope \"" + sc + "\" (all|prompt)");
    e.note = o.value("note", "");
    return e;
}

std::string env_or(const char* k, const std::string& d) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

}  // namespace

Portability portability(const Edit& e) {
    if (e.scope == EditScope::Prompt) return Portability::NeuronautOnly;
    if (e.op == EditOp::Scale || e.op == EditOp::RankOne) return Portability::Gguf;
    // llama.cpp never applies a control vector to layer 0 (export.h).
    return e.layer == 0 ? Portability::NeuronautOnly : Portability::ControlVector;
}

const char* portability_name(Portability p) {
    switch (p) {
        case Portability::Gguf: return "gguf";
        case Portability::ControlVector: return "control-vector";
        case Portability::NeuronautOnly: return "neuronaut-only";
    }
    return "?";
}

std::string edit_text(const Edit& e) {
    if (e.op == EditOp::RankOne) {
        double n2 = 0;
        for (float v : e.u) n2 += static_cast<double>(v) * v;
        return "L" + std::to_string(e.layer) + " rank-one (‖u‖ " + fmt_f(static_cast<float>(std::sqrt(n2)), 2).substr(1) + ")";
    }
    std::string s = "L" + std::to_string(e.layer) + ".N" + std::to_string(e.neuron);
    switch (e.op) {
        case EditOp::Scale: s += e.k == 0.0f ? " ablated" : e.k == -1.0f ? " inverted" : " ×" + fmt_f(e.k, 2).substr(e.k >= 0 ? 1 : 0); break;
        case EditOp::Add: s += " add " + fmt_f(e.value, 2); break;
        case EditOp::Set: s += " = " + fmt_f(e.value, 2); break;
        case EditOp::Steer: s += std::string(e.pole == '+' ? "+" : "−") + " steer " + fmt_f(e.dose, 2).substr(1); break;
        case EditOp::RankOne: break;
    }
    if (e.scope == EditScope::Prompt) s += " @prompt";
    return s;
}

std::string validate_edit(const Edit& e, const ModelLayout& m) {
    if (e.layer < 0 || e.layer >= m.n_layer || !m.has_ffn(e.layer)) return "no FFN layer " + std::to_string(e.layer);
    if (e.op == EditOp::RankOne) {
        if (e.u.size() != static_cast<size_t>(m.n_embd) || e.w.size() != static_cast<size_t>(m.n_ff))
            return "rank-one wants u of " + std::to_string(m.n_embd) + " and w of " + std::to_string(m.n_ff) + " values";
        if (e.scope == EditScope::Prompt) return "scope prompt is for add/steer only";
        for (float v : e.u) if (!std::isfinite(v)) return "non-finite value";
        for (float v : e.w) if (!std::isfinite(v)) return "non-finite value";
        return "";
    }
    if (e.neuron < 0 || e.neuron >= m.n_ff) return "neuron " + std::to_string(e.neuron) + " out of range (0.." + std::to_string(m.n_ff - 1) + ")";
    if (e.op == EditOp::Steer && e.pole != '+' && e.pole != '-') return "steer needs a pole: + or -";
    if (e.op == EditOp::Steer && !(e.dose >= 0)) return "steer dose must be ≥ 0 (the pole gives the sign)";
    if (e.scope == EditScope::Prompt && (e.op == EditOp::Scale || e.op == EditOp::Set)) return "scope prompt is for add/steer only";
    if (!std::isfinite(e.k) || !std::isfinite(e.value) || !std::isfinite(e.dose)) return "non-finite value";
    return "";
}

bool same_target(const Edit& a, const Edit& b) {
    // A neuron carries at most one multiplicative edit and one additive edit per scope.
    if (a.layer != b.layer || a.neuron != b.neuron) return false;
    if (a.op == EditOp::RankOne || b.op == EditOp::RankOne) return false;   // weight edits stack
    const bool am = a.op == EditOp::Scale || a.op == EditOp::Set, bm = b.op == EditOp::Scale || b.op == EditOp::Set;
    if (am || bm) return am && bm;
    return a.scope == b.scope;
}

std::string edits_dir() {
#ifdef NEURONAUT_SOURCE_DIR
    return env_or("NEURONAUT_EDITS", std::string(NEURONAUT_SOURCE_DIR) + "/edits");
#else
    return env_or("NEURONAUT_EDITS", "edits");
#endif
}

std::string edit_set_path(const ModelKey& k, const std::string& name) {
    if (name.find('/') != std::string::npos || (name.size() > 5 && name.compare(name.size() - 5, 5, ".json") == 0)) return name;
    return (fs::path(edits_dir()) / label_slug(k) / (name + ".json")).string();
}

std::vector<std::string> list_edit_sets(const ModelKey& k) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(fs::path(edits_dir()) / label_slug(k), ec))
        if (de.path().extension() == ".json") out.push_back(de.path().stem().string());
    std::sort(out.begin(), out.end());
    return out;
}

EditSet load_edit_set(const std::string& path, const ModelLayout& m, bool force) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("edits: can't read " + path);
    json j;
    try {
        j = json::parse(in);
    } catch (const json::exception& e) {
        throw std::runtime_error("edits: " + path + " is not valid JSON: " + e.what());
    }
    if (!j.is_object() || j.value("format", "") != kFormat) throw std::runtime_error("edits: " + path + " is not a " + kFormat + " file");
    const json mk = j.value("model", json::object());
    if (mk.value("n_layer", 0) != m.n_layer || mk.value("n_ff", 0) != m.n_ff)
        throw std::runtime_error("edits: " + path + " is for " + mk.value("name", std::string("?")) + " (another shape)");
    if (mk.value("name", "") != m.name && !force)
        throw std::runtime_error("edits: " + path + " is for \"" + mk.value("name", std::string("?")) + "\", not \"" + m.name + "\" (same shape; force to apply anyway)");
    EditSet s;
    s.name = j.value("name", fs::path(path).stem().string());
    s.notes = j.value("notes", "");
    int i = 0;
    for (const auto& o : j.value("edits", json::array())) {
        Edit e;
        try {
            e = edit_from_json(o);
        } catch (const std::exception& ex) {
            throw std::runtime_error("edits: " + path + " edit " + std::to_string(i) + ": " + ex.what());
        }
        if (const std::string err = validate_edit(e, m); !err.empty()) throw std::runtime_error("edits: " + path + " edit " + std::to_string(i) + ": " + err);
        s.edits.push_back(std::move(e));
        ++i;
    }
    return s;
}

void save_edit_set(const std::string& path, const EditSet& s, const ModelLayout& m) {
    Portability worst = Portability::Gguf;
    for (const auto& e : s.edits) worst = std::max(worst, portability(e));
    // One edit per line, like labels/: diffs read per edit.
    const auto dump = [](const json& x) { return x.dump(-1, ' ', false, json::error_handler_t::replace); };
    std::string out = "{\n  \"format\": \"" + std::string(kFormat) + "\",\n  \"model\": " + dump(model_json(m)) + ",\n  \"name\": " + dump(json(s.name)) +
                      ",\n  \"notes\": " + dump(json(s.notes)) + ",\n  \"portability\": \"" + portability_name(worst) + "\",\n  \"edits\": [";
    for (size_t i = 0; i < s.edits.size(); ++i) out += std::string(i ? ",\n    " : "\n    ") + dump(edit_json(s.edits[i]));
    out += s.edits.empty() ? "]\n}\n" : "\n  ]\n}\n";
    const fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("edits: can't write " + tmp);
        f << out;
    }
    fs::rename(tmp, path);
}

bool needs_units(const std::vector<Edit>& edits) {
    return std::any_of(edits.begin(), edits.end(), [](const Edit& e) { return e.op == EditOp::Steer; });
}

void apply_edits(const std::vector<Edit>& edits, const ModelLayout& m, const SteerUnits* units, Interventions& iv) {
    for (const auto& e : edits) {
        if (const std::string err = validate_edit(e, m); !err.empty()) throw std::invalid_argument(edit_text(e) + ": " + err);
        float v = e.value;
        switch (e.op) {
            case EditOp::Scale: iv.scale(m, e.layer, e.neuron, e.k); continue;
            case EditOp::Set:
                iv.scale(m, e.layer, e.neuron, 0.0f);
                iv.add(m, e.layer, e.neuron, e.value);
                continue;
            case EditOp::Steer:
                if (!units) throw std::logic_error("apply_edits: steer edit without SteerUnits");
                v = units->value(e.layer, e.neuron, e.dose, e.pole);
                break;
            case EditOp::RankOne: iv.add_rank_one(m, e.layer, e.u, e.w); continue;
            case EditOp::Add: break;
        }
        if (e.scope == EditScope::Prompt) iv.add_prompt(m, e.layer, e.neuron, v);
        else iv.add(m, e.layer, e.neuron, v);
    }
}

}  // namespace neuronaut
