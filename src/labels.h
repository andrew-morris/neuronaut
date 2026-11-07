// labels.h — a small database of human (or imported) names for neurons.
//
// A label says what someone thinks a neuron does: "L11.N1639 (+) say a city".
// Labels are research notes, so the store is built for sharing and merging, not
// for speed:
//
//   * One JSON file per model, `labels/<model-slug>.json` in the neuronaut source
//     tree by default (so labels travel through git between machines),
//     or under $NEURONAUT_LABELS. One label per line, sorted, so diffs and merges
//     read like a changelog.
//   * Keyed by the model's `general.name` + shape (arch, n_layer, n_ff), not by the
//     file hash: neuron indices survive conversion and quantization (verified in Q0),
//     so labels made on BF16 apply to Q8_0. A file for a different model with the
//     same shape (e.g. 9B vs 9B-Base) is refused unless forced.
//   * A record is identified by (layer, neuron, pole, source). Several sources can
//     label the same neuron and disagree; an import never silently overwrites
//     someone else's record.
//
// Format (`neuronaut-labels/1`):
//
//   {
//     "format": "neuronaut-labels/1",
//     "model": {"name": "Llama 3.2 1B Instruct", "arch": "llama", "n_layer": 16, "n_ff": 8192},
//     "labels": [
//       {"layer":11,"neuron":1639,"pole":"+","label":"say a city","tags":["city"],"source":"research_log",...},
//       ...
//     ]
//   }
//
// Import is deliberately permissive so other tools' dumps load without a converter
// (see ImportOptions): native files, a JSON array of records, a JSON object mapping
// "L.N" → label (or → record), JSONL, CSV/TSV with a header row, plus field renames.
//
// The header is free of JSON types; nlohmann/json stays inside labels.cpp.
#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "model_layout.h"

namespace neuronaut {

struct Label {
    int layer = -1, neuron = -1;
    char pole = 0;               // '+', '-', or 0 (the neuron as a whole)
    std::string label;           // the name: short, what it does
    std::vector<std::string> tags;
    std::string notes;           // free text: evidence, caveats
    std::string source;          // who/what says so: "manual", "research_log", "neuron-explorer", …
    std::string author;
    std::string created, updated;   // ISO-8601 UTC
    std::string extra;           // unknown fields from an import, as a JSON object ("" = none); kept for round-trips
};

// What a label file is for. general.name + shape.
struct ModelKey {
    std::string name, arch;
    int n_layer = 0, n_ff = 0;
};
ModelKey model_key(const ModelLayout& m);
std::string label_slug(const ModelKey& k);   // file stem, e.g. "Llama-3.2-1B-Instruct"
std::string labels_dir();                    // $NEURONAUT_LABELS, else <source tree>/labels

// "L25.N6715-", "25.6715+", "25:6715", "L25 N6715", "25/6715" → (25, 6715, '-'/'+'/0).
bool parse_neuron_ref(const std::string& s, int& layer, int& neuron, char& pole);
std::string neuron_ref(int layer, int neuron, char pole = 0);   // "L25.N6715-"
std::string now_iso();

enum class LabelFormat { Auto, Json, Jsonl, Csv, Tsv };
LabelFormat parse_label_format(const std::string& s);   // "json"|"jsonl"|"csv"|"tsv"|"auto"; throws otherwise

struct ImportOptions {
    LabelFormat format = LabelFormat::Auto;   // Auto: by extension, then by content
    // Field renames, ours ← theirs: {"label": "name", "layer": "l"}. Applied to every
    // incoming object/row before it is read.
    std::map<std::string, std::string> rename;
    std::string default_source;   // for records without one (default: the file's stem)
    bool replace_source = false;  // drop existing records of each incoming source that the import doesn't contain
    bool dry_run = false;         // report, change nothing
    bool force = false;           // accept a native file for another model, and skip bad records instead of aborting
};

struct ImportReport {
    int added = 0, updated = 0, unchanged = 0, removed = 0, skipped = 0;
    std::vector<std::string> changes;   // "+ L11.N1639+ [research_log] say a city", "~ …", "- …"
    std::vector<std::string> errors;    // per-record problems (record number + reason)
    bool aborted = false;               // errors without force: nothing was written
};

struct ExportOptions {
    LabelFormat format = LabelFormat::Json;
    std::string source;   // only this source ("" = all)
    std::string tag;      // only records with this tag ("" = all)
};

class LabelDb {
public:
    // Opens the file for `m` under `dir` (created on first write). Throws if the file
    // exists but can't be parsed, or belongs to another model.
    explicit LabelDb(const ModelLayout& m, const std::string& dir = labels_dir());

    const std::string& path() const { return path_; }
    const ModelKey& key() const { return key_; }

    // Re-read the file if it changed on disk since we last looked (cheap: one stat).
    void refresh();

    const std::vector<Label>& all() const { return labels_; }   // sorted (layer, neuron, pole, source)
    std::vector<Label> at(int layer, int neuron) const;          // every pole and source
    // Case-insensitive substring over ref ("L11.N1639"), label, tags, notes and source;
    // "#tag" matches tags exactly; "src:NAME" matches the source. Empty = everything.
    std::vector<Label> search(const std::string& query) const;
    // One-line summary for lists: "say a city" or "+ say a city · − Tokyo" (poles), "" if none.
    std::string brief(int layer, int neuron) const;
    bool has(int layer, int neuron) const { return index_.count({layer, neuron}) > 0; }

    // Mutations re-read the file, apply, and write it back atomically (tmp + rename),
    // so the TUI, the CLI and a remote MCP session can share one file.
    // Insert, or update the record with the same (layer, neuron, pole, source).
    // Returns true when added. Validates the neuron against the model's shape.
    bool upsert(Label l);
    // Remove records at (layer, neuron, pole) — any pole if any_pole — of `source`
    // ("" = any source). Returns the removed records.
    std::vector<Label> remove(int layer, int neuron, char pole, bool any_pole, const std::string& source);

    ImportReport import_file(const std::string& file, ImportOptions opt);
    ImportReport import_text(const std::string& text, const ImportOptions& opt);   // opt.format must not be Auto… (or it sniffs)
    std::string export_text(const ExportOptions& opt) const;

private:
    void load();
    void save();
    void reindex();
    std::string validate(const Label& l) const;   // "" when fine

    ModelKey key_;
    std::string path_;
    std::vector<Label> labels_;
    std::map<std::pair<int, int>, std::vector<size_t>> index_;
    std::optional<std::filesystem::file_time_type> mtime_;   // of the file as last loaded (nullopt = absent)
};

}  // namespace neuronaut
