// labels.cpp — see labels.h for the model and the file format.
#include "labels.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

#include <json.hpp>

namespace fs = std::filesystem;

namespace neuronaut {
namespace {

using json = nlohmann::ordered_json;
constexpr const char* kFormat = "neuronaut-labels/1";

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string read_all(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("labels: can't read " + p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The file's mtime, or nullopt when it doesn't exist. (Not a -1 sentinel: libstdc++'s
// file_clock epoch is in 2174, so every real mtime counts negative.)
std::optional<fs::file_time_type> file_mtime(const std::string& p) {
    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    if (ec) return std::nullopt;
    return t;
}

// Sort order of a file: layer, neuron, pole (none, +, −), source.
int pole_rank(char p) { return p == 0 ? 0 : p == '+' ? 1 : 2; }
bool label_less(const Label& a, const Label& b) {
    if (a.layer != b.layer) return a.layer < b.layer;
    if (a.neuron != b.neuron) return a.neuron < b.neuron;
    if (a.pole != b.pole) return pole_rank(a.pole) < pole_rank(b.pole);
    return a.source < b.source;
}
bool same_key(const Label& a, const Label& b) {
    return a.layer == b.layer && a.neuron == b.neuron && a.pole == b.pole && a.source == b.source;
}
// Same content (ignoring timestamps), i.e. an import of it is a no-op.
bool same_content(const Label& a, const Label& b) {
    return a.label == b.label && a.tags == b.tags && a.notes == b.notes && a.author == b.author && a.extra == b.extra;
}

// --- record ↔ json ---------------------------------------------------------

std::string pole_str(char p) { return p ? std::string(1, p) : std::string(); }

json to_json(const Label& l) {
    json o;
    o["layer"] = l.layer;
    o["neuron"] = l.neuron;
    if (l.pole) o["pole"] = pole_str(l.pole);
    o["label"] = l.label;
    if (!l.tags.empty()) o["tags"] = l.tags;
    if (!l.notes.empty()) o["notes"] = l.notes;
    o["source"] = l.source;
    if (!l.author.empty()) o["author"] = l.author;
    if (!l.created.empty()) o["created"] = l.created;
    if (!l.updated.empty()) o["updated"] = l.updated;
    if (!l.extra.empty()) {
        const json x = json::parse(l.extra);
        for (const auto& [k, v] : x.items())
            if (!o.contains(k)) o[k] = v;
    }
    return o;
}

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// A JSON scalar as text ("12" for 12, the string itself for a string).
std::string scalar_text(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null()) return "";
    return dump(v);
}

bool to_int(const json& v, int& out) {
    try {
        if (v.is_number_integer()) {
            out = v.get<int>();
            return true;
        }
        if (v.is_number()) {
            const double d = v.get<double>();
            if (d != static_cast<int>(d)) return false;
            out = static_cast<int>(d);
            return true;
        }
        if (v.is_string()) {
            const std::string s = trim(v.get<std::string>());
            size_t used = 0;
            out = std::stoi(s, &used);
            return used == s.size();
        }
    } catch (...) {
    }
    return false;
}

bool parse_pole(const std::string& s0, char& pole) {
    const std::string s = lower(trim(s0));
    if (s.empty() || s == "none" || s == "both" || s == "any" || s == "0") pole = 0;
    else if (s == "+" || s == "pos" || s == "positive" || s == "+1" || s == "1") pole = '+';
    else if (s == "-" || s == "−" || s == "neg" || s == "negative" || s == "-1") pole = '-';
    else return false;
    return true;
}

std::vector<std::string> split_tags(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s + ",") {
        if (c == ',' || c == ';' || c == '|') {
            std::string t = trim(cur);
            if (!t.empty() && t[0] == '#') t = t.substr(1);
            if (!t.empty()) out.push_back(t);
            cur.clear();
        } else {
            cur += c;
        }
    }
    return out;
}

void normalize_tags(std::vector<std::string>& tags) {
    std::vector<std::string> out;
    for (auto t : tags) {
        t = trim(t);
        if (!t.empty() && t[0] == '#') t = t.substr(1);
        if (!t.empty() && std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
    }
    tags = out;
}

// An incoming object → a Label. Every field we understand is consumed; the rest is
// kept verbatim in `extra`. `ref` is a neuron reference taken from outside the
// object (a mapping key); fields inside the object win over it.
std::string from_json(json o, const std::string& ref, const ImportOptions& opt, Label& l) {
    if (!o.is_object()) {
        if (o.is_string() && !ref.empty()) o = json{{"label", o}};
        else return "not an object";
    }
    for (const auto& [ours, theirs] : opt.rename) {
        if (ours == theirs || !o.contains(theirs)) continue;
        o[ours] = o[theirs];
        o.erase(theirs);
    }
    auto take = [&](std::initializer_list<const char*> names, json& out) {
        for (const char* n : names) {
            if (o.contains(n)) {
                out = o[n];
                o.erase(n);
                return true;
            }
        }
        return false;
    };

    // Neuron: explicit layer + neuron, else a reference string in neuron/ref/id/key, else the mapping key.
    json v;
    bool have_l = false, have_n = false;
    char ref_pole = 0;
    bool ref_has_pole = false;
    if (take({"layer", "l"}, v)) {
        if (!to_int(v, l.layer)) return "bad layer " + scalar_text(v);
        have_l = true;
    }
    json nv;
    if (take({"neuron", "n", "index"}, nv)) {
        if (to_int(nv, l.neuron)) {
            have_n = true;
        } else if (nv.is_string() && parse_neuron_ref(nv.get<std::string>(), l.layer, l.neuron, ref_pole)) {
            have_l = have_n = true;
            ref_has_pole = nv.get<std::string>().find_first_of("+-") != std::string::npos;
        } else {
            return "bad neuron " + scalar_text(nv);
        }
    }
    if (!have_l || !have_n) {
        json rv;
        std::string r = ref;
        if (take({"ref", "id", "key"}, rv)) r = scalar_text(rv);
        if (r.empty() || !parse_neuron_ref(r, l.layer, l.neuron, ref_pole)) return "no neuron (need layer+neuron, or a ref like L11.N1639)";
        ref_has_pole = r.find_first_of("+-") != std::string::npos;
    }

    l.pole = ref_has_pole ? ref_pole : 0;
    if (take({"pole", "sign", "polarity"}, v)) {
        if (!parse_pole(scalar_text(v), l.pole)) return "bad pole " + scalar_text(v);
    }
    if (!take({"label", "name", "title", "description", "explanation"}, v)) return "no label (expected a label/name field)";
    l.label = trim(scalar_text(v));
    if (l.label.empty()) return "empty label";
    if (take({"tags", "tag"}, v)) {
        if (v.is_array()) {
            for (const auto& t : v) l.tags.push_back(scalar_text(t));
        } else {
            l.tags = split_tags(scalar_text(v));
        }
        normalize_tags(l.tags);
    }
    if (take({"notes", "note", "comment", "evidence"}, v)) l.notes = scalar_text(v);
    if (take({"source"}, v)) l.source = trim(scalar_text(v));
    if (l.source.empty()) l.source = opt.default_source;
    if (take({"author"}, v)) l.author = scalar_text(v);
    if (take({"created"}, v)) l.created = scalar_text(v);
    if (take({"updated"}, v)) l.updated = scalar_text(v);
    if (!o.empty()) l.extra = dump(o);
    return "";
}

// --- CSV -------------------------------------------------------------------

// RFC 4180-ish: quoted fields, "" escapes, newlines inside quotes.
std::vector<std::vector<std::string>> parse_dsv(const std::string& text, char sep) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string cur;
    bool quoted = false, any = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (c == '"' && i + 1 < text.size() && text[i + 1] == '"') {
                cur += '"';
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                cur += c;
            }
        } else if (c == '"') {
            quoted = true;
            any = true;
        } else if (c == sep) {
            row.push_back(cur);
            cur.clear();
            any = true;
        } else if (c == '\n' || c == '\r') {
            if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
            if (any || !cur.empty()) {
                row.push_back(cur);
                rows.push_back(row);
            }
            row.clear();
            cur.clear();
            any = false;
        } else {
            cur += c;
        }
    }
    if (any || !cur.empty()) {
        row.push_back(cur);
        rows.push_back(row);
    }
    return rows;
}

std::string dsv_field(const std::string& s, char sep) {
    if (s.find_first_of(std::string(1, sep) + "\"\n\r") == std::string::npos) return s;
    std::string o = "\"";
    for (char c : s) o += c == '"' ? std::string("\"\"") : std::string(1, c);
    return o + "\"";
}

LabelFormat sniff(const std::string& file, const std::string& text) {
    std::string ext = lower(fs::path(file).extension().string());
    if (ext == ".json") return LabelFormat::Json;
    if (ext == ".jsonl" || ext == ".ndjson") return LabelFormat::Jsonl;
    if (ext == ".csv") return LabelFormat::Csv;
    if (ext == ".tsv" || ext == ".tab") return LabelFormat::Tsv;
    const std::string t = trim(text);
    if (t.empty()) return LabelFormat::Json;
    if (t[0] == '[') return LabelFormat::Json;
    if (t[0] == '{') {
        // One object spanning the text = JSON; one object per line = JSONL.
        return json::accept(t) ? LabelFormat::Json : LabelFormat::Jsonl;
    }
    const std::string first = t.substr(0, t.find('\n'));
    return first.find('\t') != std::string::npos ? LabelFormat::Tsv : LabelFormat::Csv;
}

std::string env_or(const char* k, const std::string& d) {
    const char* v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

}  // namespace

// ===========================================================================

ModelKey model_key(const ModelLayout& m) { return {m.name, m.arch, m.n_layer, m.n_ff}; }

std::string label_slug(const ModelKey& k) {
    std::string base = k.name.empty() ? k.arch + "-" + std::to_string(k.n_layer) + "x" + std::to_string(k.n_ff) : k.name;
    std::string s;
    for (char c : base) {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-';
        if (ok) s += c;
        else if (!s.empty() && s.back() != '-') s += '-';
    }
    while (!s.empty() && s.back() == '-') s.pop_back();
    return s.empty() ? "model" : s;
}

std::string labels_dir() {
#ifdef NEURONAUT_SOURCE_DIR
    return env_or("NEURONAUT_LABELS", std::string(NEURONAUT_SOURCE_DIR) + "/labels");
#else
    return env_or("NEURONAUT_LABELS", "labels");
#endif
}

bool parse_neuron_ref(const std::string& s0, int& layer, int& neuron, char& pole) {
    // Accept L?<layer><sep>N?<neuron><pole?> with sep one of . : / space _ ,
    std::string s = trim(s0);
    char p = 0;
    if (!s.empty() && (s.back() == '+' || s.back() == '-')) {
        p = s.back();
        s.pop_back();
    } else if (s.size() >= 3 && s.compare(s.size() - 3, 3, "−") == 0) {   // U+2212
        p = '-';
        s.resize(s.size() - 3);
    }
    size_t i = 0;
    auto skip_letter = [&](char a) {
        if (i < s.size() && std::tolower(static_cast<unsigned char>(s[i])) == a) ++i;
    };
    auto num = [&](int& out) {
        const size_t st = i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        if (i == st || i - st > 9) return false;
        out = std::stoi(s.substr(st, i - st));
        return true;
    };
    int l = 0, n = 0;
    skip_letter('l');
    if (!num(l)) return false;
    const size_t sep_start = i;
    while (i < s.size() && (s[i] == '.' || s[i] == ':' || s[i] == '/' || s[i] == ' ' || s[i] == '_' || s[i] == ',')) ++i;
    if (i == sep_start) return false;
    skip_letter('n');
    if (!num(n)) return false;
    if (i != s.size()) return false;
    layer = l;
    neuron = n;
    pole = p;
    return true;
}

std::string neuron_ref(int layer, int neuron, char pole) {
    return "L" + std::to_string(layer) + ".N" + std::to_string(neuron) + (pole ? std::string(1, pole) : std::string());
}

std::string now_iso() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

LabelFormat parse_label_format(const std::string& s) {
    const std::string f = lower(s);
    if (f == "auto") return LabelFormat::Auto;
    if (f == "json") return LabelFormat::Json;
    if (f == "jsonl" || f == "ndjson") return LabelFormat::Jsonl;
    if (f == "csv") return LabelFormat::Csv;
    if (f == "tsv") return LabelFormat::Tsv;
    throw std::invalid_argument("labels: unknown format " + s + " (json|jsonl|csv|tsv)");
}

// ===========================================================================

LabelDb::LabelDb(const ModelLayout& m, const std::string& dir) : key_(model_key(m)) {
    path_ = (fs::path(dir) / (label_slug(key_) + ".json")).string();
    load();
}

void LabelDb::load() {
    labels_.clear();
    mtime_ = file_mtime(path_);
    if (!mtime_) {
        reindex();
        return;
    }
    json j;
    try {
        j = json::parse(read_all(path_));
    } catch (const json::exception& e) {
        throw std::runtime_error("labels: " + path_ + " is not valid JSON: " + e.what());
    }
    if (!j.is_object() || j.value("format", "") != kFormat) throw std::runtime_error("labels: " + path_ + " is not a " + kFormat + " file");
    const json& mk = j.value("model", json::object());
    if (mk.value("n_layer", 0) != key_.n_layer || mk.value("n_ff", 0) != key_.n_ff || mk.value("name", "") != key_.name)
        throw std::runtime_error("labels: " + path_ + " belongs to \"" + mk.value("name", "?") + "\", not \"" + key_.name + "\"");
    ImportOptions opt;
    int i = 0;
    for (const auto& o : j.value("labels", json::array())) {
        Label l;
        const std::string err = from_json(o, "", opt, l);
        if (!err.empty()) throw std::runtime_error("labels: " + path_ + " record " + std::to_string(i) + ": " + err);
        labels_.push_back(std::move(l));
        ++i;
    }
    std::sort(labels_.begin(), labels_.end(), label_less);
    reindex();
}

void LabelDb::refresh() {
    if (file_mtime(path_) != mtime_) load();
}

void LabelDb::reindex() {
    index_.clear();
    for (size_t i = 0; i < labels_.size(); ++i) index_[{labels_[i].layer, labels_[i].neuron}].push_back(i);
}

void LabelDb::save() {
    std::sort(labels_.begin(), labels_.end(), label_less);
    // Hand-assembled so each record is one line: diffs and merges read per label.
    std::string out = "{\n  \"format\": \"" + std::string(kFormat) + "\",\n  \"model\": ";
    json mk;
    mk["name"] = key_.name;
    mk["arch"] = key_.arch;
    mk["n_layer"] = key_.n_layer;
    mk["n_ff"] = key_.n_ff;
    out += dump(mk) + ",\n  \"labels\": [";
    for (size_t i = 0; i < labels_.size(); ++i) out += std::string(i ? ",\n    " : "\n    ") + dump(to_json(labels_[i]));
    out += labels_.empty() ? "]\n}\n" : "\n  ]\n}\n";

    fs::create_directories(fs::path(path_).parent_path());
    const std::string tmp = path_ + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("labels: can't write " + tmp);
        f << out;
        if (!f) throw std::runtime_error("labels: write failed: " + tmp);
    }
    fs::rename(tmp, path_);
    mtime_ = file_mtime(path_);
    reindex();
}

std::string LabelDb::validate(const Label& l) const {
    if (l.layer < 0 || l.layer >= key_.n_layer) return "layer " + std::to_string(l.layer) + " out of range (0.." + std::to_string(key_.n_layer - 1) + ")";
    if (l.neuron < 0 || l.neuron >= key_.n_ff) return "neuron " + std::to_string(l.neuron) + " out of range (0.." + std::to_string(key_.n_ff - 1) + ")";
    if (l.pole != 0 && l.pole != '+' && l.pole != '-') return "bad pole";
    if (trim(l.label).empty()) return "empty label";
    if (l.source.empty()) return "no source";
    return "";
}

std::vector<Label> LabelDb::at(int layer, int neuron) const {
    std::vector<Label> out;
    auto it = index_.find({layer, neuron});
    if (it != index_.end())
        for (size_t i : it->second) out.push_back(labels_[i]);
    return out;
}

std::vector<Label> LabelDb::search(const std::string& query) const {
    // Every whitespace-separated term must match (AND).
    std::vector<std::string> terms;
    std::istringstream is(query);
    for (std::string t; is >> t;) terms.push_back(lower(t));
    std::vector<Label> out;
    for (const auto& l : labels_) {
        bool ok = true;
        for (const auto& t : terms) {
            if (t.size() > 1 && t[0] == '#') {
                ok = std::any_of(l.tags.begin(), l.tags.end(), [&](const std::string& g) { return lower(g) == t.substr(1); });
            } else if (t.rfind("src:", 0) == 0) {
                ok = lower(l.source) == t.substr(4);
            } else {
                std::string hay = lower(neuron_ref(l.layer, l.neuron, l.pole) + " " + l.label + " " + l.notes + " " + l.source);
                for (const auto& g : l.tags) hay += " " + lower(g);
                ok = hay.find(t) != std::string::npos;
            }
            if (!ok) break;
        }
        if (ok) out.push_back(l);
    }
    return out;
}

std::string LabelDb::brief(int layer, int neuron) const {
    auto it = index_.find({layer, neuron});
    if (it == index_.end()) return "";
    // One label per pole (the first source in sort order), poles in order none, +, −.
    std::string out;
    char last_pole = 1;
    for (size_t i : it->second) {
        const Label& l = labels_[i];
        if (l.pole == last_pole) continue;
        last_pole = l.pole;
        if (!out.empty()) out += " · ";
        if (l.pole) out += std::string(l.pole == '+' ? "+" : "−") + " ";
        out += l.label;
    }
    return out;
}

bool LabelDb::upsert(Label l) {
    refresh();
    normalize_tags(l.tags);
    l.label = trim(l.label);
    if (const std::string e = validate(l); !e.empty()) throw std::invalid_argument("labels: " + neuron_ref(l.layer, l.neuron, l.pole) + ": " + e);
    const std::string now = now_iso();
    for (auto& x : labels_) {
        if (same_key(x, l)) {
            l.created = x.created.empty() ? now : x.created;
            l.updated = now;
            if (l.author.empty()) l.author = x.author;
            x = std::move(l);
            save();
            return false;
        }
    }
    if (l.created.empty()) l.created = now;
    l.updated = now;
    labels_.push_back(std::move(l));
    save();
    return true;
}

std::vector<Label> LabelDb::remove(int layer, int neuron, char pole, bool any_pole, const std::string& source) {
    refresh();
    std::vector<Label> gone, keep;
    for (auto& x : labels_) {
        const bool hit = x.layer == layer && x.neuron == neuron && (any_pole || x.pole == pole) && (source.empty() || x.source == source);
        (hit ? gone : keep).push_back(std::move(x));
    }
    labels_ = std::move(keep);
    if (!gone.empty()) save();
    else reindex();
    return gone;
}

ImportReport LabelDb::import_file(const std::string& file, ImportOptions opt) {
    const std::string text = read_all(file);
    if (opt.format == LabelFormat::Auto) opt.format = sniff(file, text);
    if (opt.default_source.empty()) opt.default_source = fs::path(file).stem().string();
    return import_text(text, opt);
}

ImportReport LabelDb::import_text(const std::string& text, const ImportOptions& opt0) {
    ImportOptions opt = opt0;
    if (opt.format == LabelFormat::Auto) opt.format = sniff("", text);
    if (opt.default_source.empty()) opt.default_source = "import";
    ImportReport rep;

    // 1. Parse into (record number, object, ref) triples.
    struct In {
        std::string where;
        json obj;
        std::string ref;
    };
    std::vector<In> ins;
    auto fail = [&](const std::string& why) {
        rep.errors.push_back(why);
        rep.aborted = true;
        return rep;
    };
    if (opt.format == LabelFormat::Json) {
        json j;
        try {
            j = json::parse(text);
        } catch (const json::exception& e) {
            return fail(std::string("not valid JSON: ") + e.what());
        }
        if (j.is_object() && j.contains("format")) {
            // A native file (ours, or from another checkout/host).
            if (j.value("format", "") != kFormat) return fail("unknown format " + j.value("format", std::string("?")));
            const json& mk = j.value("model", json::object());
            const bool shape_ok = mk.value("n_layer", 0) == key_.n_layer && mk.value("n_ff", 0) == key_.n_ff;
            if (!shape_ok) return fail("file is for " + mk.value("name", std::string("?")) + " (" + std::to_string(mk.value("n_layer", 0)) + " layers × " +
                                       std::to_string(mk.value("n_ff", 0)) + "), this model is " + std::to_string(key_.n_layer) + " × " + std::to_string(key_.n_ff));
            if (mk.value("name", "") != key_.name && !opt.force)
                return fail("file is for \"" + mk.value("name", std::string("?")) + "\", this model is \"" + key_.name +
                            "\" (same shape; --force if the labels really transfer)");
            j = j.value("labels", json::array());
        } else if (j.is_object()) {
            // {"labels": [...]} / {"neurons": [...]} wrappers, else a mapping ref → label/record.
            for (const char* k : {"labels", "neurons", "records", "items"}) {
                if (j.contains(k) && j[k].is_array()) {
                    j = j[k];
                    break;
                }
            }
        }
        if (j.is_array()) {
            for (size_t i = 0; i < j.size(); ++i) ins.push_back({"record " + std::to_string(i + 1), j[i], ""});
        } else if (j.is_object()) {
            for (const auto& [k, v] : j.items()) ins.push_back({"key " + k, v, k});
        } else {
            return fail("expected a JSON array or object");
        }
    } else if (opt.format == LabelFormat::Jsonl) {
        std::istringstream is(text);
        std::string line;
        for (int n = 1; std::getline(is, line); ++n) {
            if (trim(line).empty()) continue;
            try {
                ins.push_back({"line " + std::to_string(n), json::parse(line), ""});
            } catch (const json::exception&) {
                rep.errors.push_back("line " + std::to_string(n) + ": not valid JSON");
            }
        }
    } else {
        const auto rows = parse_dsv(text, opt.format == LabelFormat::Tsv ? '\t' : ',');
        if (rows.empty()) return fail("empty file");
        std::vector<std::string> head;
        for (const auto& h : rows[0]) head.push_back(trim(h));
        for (size_t r = 1; r < rows.size(); ++r) {
            json o = json::object();
            for (size_t c = 0; c < head.size() && c < rows[r].size(); ++c)
                if (!rows[r][c].empty()) o[head[c]] = rows[r][c];
            ins.push_back({"row " + std::to_string(r + 1), o, ""});
        }
    }

    // 2. Records, validated. Last one wins on duplicate keys within the import.
    std::vector<Label> incoming;
    for (const auto& in : ins) {
        Label l;
        std::string err = from_json(in.obj, in.ref, opt, l);
        if (err.empty()) err = validate(l);
        if (!err.empty()) {
            rep.errors.push_back(in.where + ": " + err);
            continue;
        }
        auto dup = std::find_if(incoming.begin(), incoming.end(), [&](const Label& x) { return same_key(x, l); });
        if (dup != incoming.end()) *dup = std::move(l);
        else incoming.push_back(std::move(l));
    }
    rep.skipped = static_cast<int>(rep.errors.size());
    if (rep.skipped && !opt.force) {
        rep.aborted = true;
        return rep;
    }

    // 3. Merge.
    refresh();
    const std::string now = now_iso();
    std::vector<Label> next = labels_;
    auto describe = [](char op, const Label& l) { return std::string(1, op) + " " + neuron_ref(l.layer, l.neuron, l.pole) + " [" + l.source + "] " + l.label; };
    if (opt.replace_source) {
        std::set<std::string> sources;
        for (const auto& l : incoming) sources.insert(l.source);
        std::vector<Label> keep;
        for (auto& x : next) {
            const bool replaced = sources.count(x.source) &&
                                  std::none_of(incoming.begin(), incoming.end(), [&](const Label& l) { return same_key(l, x); });
            if (replaced) {
                rep.changes.push_back(describe('-', x));
                ++rep.removed;
            } else {
                keep.push_back(std::move(x));
            }
        }
        next = std::move(keep);
    }
    for (auto& l : incoming) {
        auto it = std::find_if(next.begin(), next.end(), [&](const Label& x) { return same_key(x, l); });
        if (it == next.end()) {
            if (l.created.empty()) l.created = now;
            if (l.updated.empty()) l.updated = l.created;
            rep.changes.push_back(describe('+', l));
            ++rep.added;
            next.push_back(std::move(l));
        } else if (same_content(*it, l)) {
            ++rep.unchanged;
        } else {
            l.created = it->created.empty() ? (l.created.empty() ? now : l.created) : it->created;
            if (l.updated.empty() || l.updated == it->updated) l.updated = now;
            rep.changes.push_back(describe('~', l) + "   (was: " + it->label + ")");
            ++rep.updated;
            *it = std::move(l);
        }
    }
    if (!opt.dry_run && (rep.added || rep.updated || rep.removed)) {
        labels_ = std::move(next);
        save();
    }
    return rep;
}

std::string LabelDb::export_text(const ExportOptions& opt) const {
    std::vector<const Label*> sel;
    for (const auto& l : labels_) {
        if (!opt.source.empty() && l.source != opt.source) continue;
        if (!opt.tag.empty() && std::find(l.tags.begin(), l.tags.end(), opt.tag) == l.tags.end()) continue;
        sel.push_back(&l);
    }
    std::string out;
    switch (opt.format) {
        case LabelFormat::Auto:
        case LabelFormat::Json: {
            // The native format: importable anywhere (it carries the model key).
            json mk;
            mk["name"] = key_.name;
            mk["arch"] = key_.arch;
            mk["n_layer"] = key_.n_layer;
            mk["n_ff"] = key_.n_ff;
            out = "{\n  \"format\": \"" + std::string(kFormat) + "\",\n  \"model\": " + dump(mk) + ",\n  \"labels\": [";
            for (size_t i = 0; i < sel.size(); ++i) out += std::string(i ? ",\n    " : "\n    ") + dump(to_json(*sel[i]));
            out += sel.empty() ? "]\n}\n" : "\n  ]\n}\n";
            break;
        }
        case LabelFormat::Jsonl:
            for (const Label* l : sel) out += dump(to_json(*l)) + "\n";
            break;
        case LabelFormat::Csv:
        case LabelFormat::Tsv: {
            const char sep = opt.format == LabelFormat::Tsv ? '\t' : ',';
            const std::vector<std::string> cols = {"layer", "neuron", "pole", "label", "tags", "notes", "source", "author", "created", "updated"};
            for (size_t c = 0; c < cols.size(); ++c) out += (c ? std::string(1, sep) : "") + cols[c];
            out += "\n";
            for (const Label* l : sel) {
                std::string tags;
                for (size_t t = 0; t < l->tags.size(); ++t) tags += (t ? ";" : "") + l->tags[t];
                const std::vector<std::string> v = {std::to_string(l->layer), std::to_string(l->neuron), pole_str(l->pole), l->label, tags, l->notes,
                                                    l->source, l->author, l->created, l->updated};
                for (size_t c = 0; c < v.size(); ++c) out += (c ? std::string(1, sep) : "") + dsv_field(v[c], sep);
                out += "\n";
            }
            break;
        }
    }
    return out;
}

}  // namespace neuronaut
