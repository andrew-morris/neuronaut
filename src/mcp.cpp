// mcp.cpp — a Model Context Protocol front end on one headless Session.
//
// This is the fourth way to drive the same Session the TUI, the CLI and the tests
// use (docs/mcp.md). It speaks JSON-RPC 2.0 over stdio, newline-delimited: one
// request object per line on stdin, one response object per line on stdout. stderr
// is a human log and never carries protocol.
//
// The server owns one in-process Session with `post` unset, so long jobs (the dose
// sweep) run inline to completion and there is never more than one thing happening
// at a time — which is why no locking appears anywhere below.
//
// nlohmann/json is used only here; it never reaches neuronaut_core / neuronaut_tui.
// The Session exposes typed `*_view()` snapshots (tui.h); this file is the one place
// that turns them into the wire format.

#include <cstdio>
#include <iostream>
#include <set>
#include <string>

#include <json.hpp>

#include "tui.h"

namespace neuronaut {
namespace {

using json = nlohmann::ordered_json;

constexpr const char* kServerVersion = "0.1.0";
constexpr const char* kDefaultProtocol = "2025-06-18";

// --- view -> json ----------------------------------------------------------

json label_json(const Label& l) {
    json o;
    o["ref"] = neuron_ref(l.layer, l.neuron, l.pole);
    o["layer"] = l.layer;
    o["neuron"] = l.neuron;
    o["pole"] = l.pole ? std::string(1, l.pole) : std::string();
    o["label"] = l.label;
    o["tags"] = l.tags;
    o["notes"] = l.notes;
    o["source"] = l.source;
    o["author"] = l.author;
    o["updated"] = l.updated;
    return o;
}

json tok_json(int id, const std::string& text, float prob) {
    json o;
    o["id"] = id;
    o["text"] = text;
    o["prob"] = prob;
    return o;
}

json to_json(const Session::Snapshot& v) {
    json model;
    model["arch"] = v.arch;
    model["name"] = v.model_name;
    model["n_layer"] = v.n_layer;
    model["n_embd"] = v.n_embd;
    model["n_ff"] = v.n_ff;
    model["n_head"] = v.n_head;
    model["n_head_kv"] = v.n_head_kv;
    model["n_vocab"] = v.n_vocab;
    model["n_ctx_train"] = v.n_ctx_train;
    model["tied_embd"] = v.tied_embd;

    json j;
    j["model"] = model;
    j["cursor"] = {{"pos", v.pos}, {"layer", v.layer}, {"neuron", v.neuron}};
    j["focus"] = v.focus;
    j["overlay"] = v.overlay;
    j["slice"] = v.slice;
    j["chat"] = v.chat;
    j["think"] = v.think;
    j["has_run"] = v.has_run;
    j["prompt_len"] = v.prompt_len;

    json toks = json::array();
    for (const auto& t : v.tokens) {
        json o;
        o["pos"] = t.pos;
        o["id"] = t.id;
        o["text"] = t.text;
        o["appended"] = t.appended;
        toks.push_back(o);
    }
    j["tokens"] = toks;

    if (v.has_target) {
        json t;
        t["id"] = v.target.id;
        t["text"] = v.target.text;
        t["pinned"] = v.target.pinned;
        t["prob"] = v.target.prob;
        t["rank"] = v.target.rank;
        j["target"] = t;
    } else {
        j["target"] = nullptr;
    }

    json pts = json::array();
    for (const auto& p : v.patches) {
        json o{{"layer", p.layer}, {"neuron", p.neuron}, {"op", p.op}, {"scale", p.scale}};
        if (p.op == "add" || p.op == "set" || p.op == "steer") o["value"] = p.value;
        if (p.op == "steer") {
            o["dose"] = p.dose;
            o["pole"] = p.pole;
        }
        o["scope"] = p.scope;
        o["text"] = p.text;
        o["portability"] = p.portability;
        o["labels"] = p.labels;
        pts.push_back(o);
    }
    j["patches"] = pts;

    json ws = json::array();
    for (const auto& w : v.watches) ws.push_back(json{{"layer", w.first}, {"neuron", w.second}});
    j["watches"] = ws;

    j["status"] = v.status;
    j["busy"] = v.busy;
    return j;
}

json to_json(const Session::ListView& v) {
    json rows = json::array();
    for (const auto& r : v.rows) {
        json o;
        if (r.section) {
            o["section"] = true;
            o["label"] = r.label;
        } else {
            o["kind"] = r.kind;
            o["layer"] = r.layer;
            o["neuron"] = r.neuron;
            o["value"] = r.value;
            o["detail"] = r.detail;
            o["labels"] = r.labels;
        }
        rows.push_back(o);
    }
    json j;
    j["kind"] = v.kind;
    j["selected"] = v.selected;
    j["rows"] = rows;
    return j;
}

json to_json(const Session::BlameView& v) {
    json items = json::array();
    for (const auto& it : v.items) {
        json o;
        o["kind"] = it.kind;
        o["layer"] = it.layer;
        if (it.kind == "neuron") {
            o["neuron"] = it.neuron;
            o["act"] = it.act;
            o["dot"] = it.dot;
        } else if (it.kind == "head") {
            o["head"] = it.neuron;
        }
        o["contrib"] = it.contrib;
        items.push_back(o);
    }
    json j;
    j["pos"] = v.pos;
    j["target"] = {{"id", v.target_id}, {"text", v.target_text}};
    j["logit"] = v.logit;
    j["prob"] = v.prob;
    j["rank"] = v.rank;
    j["embed"] = v.embed;
    j["sum_attn"] = v.sum_attn;
    j["sum_neurons"] = v.sum_neurons;
    j["unexplained"] = v.unexplained;
    j["items"] = items;
    return j;
}

json lens_json(const std::vector<Session::LensTokenView>& ts) {
    json a = json::array();
    for (const auto& t : ts) a.push_back(json{{"id", t.id}, {"text", t.text}, {"logit", t.logit}});
    return a;
}

json fp_json(const Session::FootprintView& f) {
    json o;
    o["first_byte"] = f.first_byte;
    o["last_byte"] = f.last_byte;
    o["n_blocks"] = f.n_blocks;
    o["contiguous"] = f.contiguous;
    o["shared_with"] = f.shared_with;
    return o;
}

json to_json(const Session::NeuronInfoView& v) {
    json j;
    j["layer"] = v.layer;
    j["neuron"] = v.neuron;
    if (v.has_act) {
        j["act_here"] = v.act_here;
        j["act_rank"] = v.act_rank;
        j["act_rank_of"] = v.act_rank_of;
    } else {
        j["act_here"] = nullptr;
    }
    j["lens"] = {{"promotes", lens_json(v.promotes)}, {"suppresses", lens_json(v.suppresses)}};
    j["footprint"] = {{"up", fp_json(v.up)}, {"gate", fp_json(v.gate)}, {"down", fp_json(v.down)}};
    json ls = json::array();
    for (const auto& l : v.labels) ls.push_back(label_json(l));
    j["labels"] = ls;
    return j;
}

json to_json(const Session::AttentionView& v) {
    json heads = json::array();
    for (const auto& h : v.heads) {
        json keys = json::array();
        for (const auto& k : h.top) keys.push_back(json{{"pos", k.pos}, {"text", k.text}, {"weight", k.weight}});
        heads.push_back(json{{"head", h.head}, {"blame", h.blame}, {"top", keys}});
    }
    json j;
    j["layer"] = v.layer;
    j["pos"] = v.pos;
    j["linear"] = v.linear;
    j["heads"] = heads;
    return j;
}

json to_json(const Session::TraceView& v) {
    json j;
    j["site"] = v.site;
    j["corrupt"] = v.corrupt;
    j["target"] = {{"id", v.target_id}, {"text", v.target_text}};
    j["p_clean"] = v.p_clean;
    j["p_corrupt"] = v.p_corrupt;
    j["clean_tokens"] = v.clean_tokens;
    j["corrupt_tokens"] = v.corrupt_tokens;
    j["differ"] = v.differ;
    j["restored"] = v.restored;   // [layer][pos], NaN → null
    j["best"] = {{"layer", v.best_layer}, {"pos", v.best_pos}, {"restored", v.best}};
    j["done"] = v.done;
    j["total"] = v.total;
    return j;
}

json to_json(const Session::ExamplesView& v) {
    auto list = [](const std::vector<Session::ExampleView>& xs) {
        json a = json::array();
        for (const auto& e : xs) a.push_back(json{{"act", e.act}, {"doc", e.doc}, {"pos", e.pos}, {"token", e.token}, {"context", e.context}});
        return a;
    };
    json j;
    j["layer"] = v.layer;
    j["neuron"] = v.neuron;
    j["corpus"] = v.corpus;
    j["n_docs"] = v.n_docs;
    j["positive"] = list(v.positive);
    j["negative"] = list(v.negative);
    return j;
}

json to_json(const Session::ProbeView& v) {
    json hits = json::array();
    for (const auto& h : v.hits)
        hits.push_back(json{{"layer", h.layer}, {"neuron", h.neuron}, {"pole", std::string(1, h.pole)}, {"margin", h.margin},
                            {"pos_mean", h.pos_mean}, {"neg_mean", h.neg_mean}});
    json j;
    j["pos"] = v.pos;
    j["neg"] = v.neg;
    j["hits"] = hits;
    return j;
}

json state_tokens_json(const std::vector<Session::StateTokenView>& ts) {
    json a = json::array();
    for (const auto& t : ts) a.push_back(tok_json(t.id, t.text, t.prob));
    return a;
}

json to_json(const Session::StateView& v) {
    json j;
    j["pos"] = v.pos;
    j["layer"] = v.layer;
    j["next_patched"] = state_tokens_json(v.next_patched);
    j["has_base"] = v.has_base;
    j["next_base"] = v.has_base ? state_tokens_json(v.next_base) : json(nullptr);
    j["target_by_layer"] = v.target_by_layer;
    j["lens_after_layer"] = {{"top", state_tokens_json(v.lens_after_layer)},
                             {"target_prob", v.lens_target_prob},
                             {"target_rank", v.lens_target_rank}};
    j["gen"] = {{"base", v.gen_base}, {"patched", v.gen_patched}};
    return j;
}

json to_json(const Session::DoseView& v) {
    json controls = json::array();
    for (const auto& c : v.controls) controls.push_back(json{{"neuron", c.first}, {"push", c.second}});

    json points = json::array();
    for (const auto& p : v.verdict.points) {
        json o;
        o["dose"] = p.dose;
        o["tested_delta"] = p.tested_delta;   // NaN serializes as null
        o["band_min"] = p.band_min;
        o["band_max"] = p.band_max;
        o["comparable"] = p.comparable;
        o["outside"] = p.outside;
        points.push_back(o);
    }

    json j;
    j["layer"] = v.layer;
    j["neuron"] = v.neuron;
    j["pos"] = v.pos;
    j["target"] = {{"id", v.target_id}, {"text", v.target_text}};
    j["push"] = v.push;
    j["doses"] = v.doses;
    j["curve"] = v.curve;
    j["controls"] = controls;
    j["ctrl"] = v.ctrl;
    j["done"] = v.done;
    j["total"] = v.total;
    j["running"] = v.running;
    j["seconds"] = v.seconds;
    j["verdict"] = {{"specific", v.verdict.specific},
                    {"outside", v.verdict.outside},
                    {"comparable", v.verdict.comparable},
                    {"understated", v.verdict.understated},
                    {"understate_factor", v.verdict.understate_factor},
                    {"points", points}};
    return j;
}

// --- the tool catalogue (static; must match dispatch in call_tool) ----------

const std::set<std::string>& tool_names() {
    static const std::set<std::string> names = {"snapshot", "exec",  "key",   "complete", "list", "blame",
                                                "neuron",   "state", "dose",  "render",   "model_info", "labels", "attention", "trace", "probe", "examples"};
    return names;
}

json tools_list() {
    auto obj = [](std::initializer_list<std::pair<std::string, json>> props, std::initializer_list<std::string> req) {
        json schema;
        schema["type"] = "object";
        json p = json::object();
        for (const auto& kv : props) p[kv.first] = kv.second;
        schema["properties"] = p;
        if (req.size()) schema["required"] = json(req);
        return schema;
    };
    auto str = json{{"type", "string"}};
    auto intt = json{{"type", "integer"}};
    auto boolt = json{{"type", "boolean"}};

    json t = json::array();
    auto add = [&](const char* name, const char* desc, json schema) {
        t.push_back(json{{"name", name}, {"description", desc}, {"inputSchema", std::move(schema)}});
    };

    add("snapshot", "Always-on state: model, cursor, focus, prompt tokens, target prob/rank, patches, watches.",
        obj({}, {}));
    add("exec",
        "Run one ':' command (prompt|chat|think|append|target|pos|ablate|invert|scale|steer|add|set|clear|save|load|edits|watch|find|list|slice|dim|gen|accept|seek|label|unlabel|labels). "
        "steer L.N+|L.N- DOSE [prompt]: DOSE in hidden-state sizes (0.5 ≈ coherent); 'prompt' = only while reading the prompt. "
        "Returns the new snapshot. Use `complete` to discover commands and arguments.",
        obj({{"line", str}}, {"line"}));
    add("key", "Press one named key (h|l|pgup|pgdn|tab|shift_tab|enter|escape|x|i|w|D|b|f|c|j|k|J|K|g|u|d). Returns the new snapshot.",
        obj({{"key", str}}, {"key"}));
    add("complete", "Completion candidates for a partial ':' command line (the same Tab offers).", obj({{"line", str}}, {"line"}));
    add("list", "The current LIST pane: blame | fires | find | controls | labels, with its rows (each neuron row carries its labels).", obj({}, {}));
    add("blame",
        "Direct logit attribution at the cursor: who wrote the target logit — neurons, the embedding, and attention per head "
        "(kind 'head', field 'head'). toward=false lists suppressors.",
        obj({{"toward", boolt}, {"k", intt}}, {}));
    add("neuron", "One neuron (defaults to the cursor): activation + rank, static lens (promotes/suppresses), byte footprint.",
        obj({{"layer", intt}, {"neuron", intt}}, {}));
    add("state", "STATE pane: next-token distribution (base vs patched), target-by-layer, lens after the cursor layer, :gen text.",
        obj({}, {}));
    add("dose",
        "Run a dose-response sweep on a neuron (defaults to the cursor) and return the curves, controls and verdict. "
        "~150 forward passes from the neuron's layer (Forward::run_from): ~1.3 s for L15 of the 1B, up to ~6 s for its first layers; "
        "minutes on the 9B on CPU — set a long timeout.",
        obj({{"layer", intt}, {"neuron", intt}}, {}));
    add("render", "The ASCII frame a human would see (escape hatch). Prefer the structured tools.",
        obj({{"width", intt}, {"height", intt}, {"color", boolt}}, {}));
    add("attention",
        "Attention of one layer from one query position (default: the cursor; the cursor moves there): for every head, its exact "
        "share of the target logit (blame) and its k most-attended earlier tokens. Linear-attention (Qwen3.5 DeltaNet) layers "
        "have no pattern: heads carry blame only, linear=true.",
        obj({{"layer", intt}, {"pos", intt}, {"k", intt}}, {}));
    add("trace",
        "Causal trace (activation patching) of the current prompt against `corrupt` — same token count, with the fact changed "
        "(\"The capital of France is\" vs \"... Italy is\"). Patches the clean run's `site` (resid|attn|mlp, default resid) at every "
        "(layer, pos) into the corrupt run and returns restored[layer][pos] = share of the clean−corrupt gap in P(target) at the last "
        "token that the patch brings back (1 = all). Total effect, unlike blame. n_layer × T partial passes: ~2 s on the 1B for 6 tokens.",
        obj({{"corrupt", str}, {"site", str}}, {"corrupt"}));
    add("probe",
        "Contrast probe: neurons that fire at the last token of every `pos` text and beyond every `neg` text (worst-case margin; "
        "pole '+' = higher on pos, '-' = higher on neg). Texts are chat-wrapped when the session is in chat mode. `layers` "
        "(default all) and k per layer (default 4). One forward per text.",
        obj({{"pos", json{{"type", "array"}, {"items", str}}}, {"neg", json{{"type", "array"}, {"items", str}}},
             {"layers", json{{"type", "array"}, {"items", intt}}}, {"k", intt}},
            {"pos", "neg"}));
    add("examples",
        "A neuron's top-activating examples across a text corpus (default: the cursor neuron): k contexts per pole with the peak "
        "token marked «». Needs the index built by `neuronaut index MODEL CORPUS.jsonl`. With `neuron` (lens) and `labels` "
        "(action=set, source='auto') this is the loop for proposing labels.",
        obj({{"layer", intt}, {"neuron", intt}, {"k", intt}}, {}));
    add("model_info", "Architecture hyperparameters of the loaded model.", obj({}, {}));
    add("labels",
        "The neuron-label database for this model (labels/<model>.json, shared with the TUI and CLI). "
        "action=search (default): records matching `query` (words AND; #tag; src:NAME). "
        "action=set: label layer.neuron (pole '+'/'-'/'' ), with label, tags, notes, source (default 'claude'). "
        "action=remove: drop records at layer.neuron (pole, source optional).",
        obj({{"action", str}, {"query", str}, {"layer", intt}, {"neuron", intt}, {"pole", str}, {"label", str},
             {"tags", json{{"type", "array"}, {"items", str}}}, {"notes", str}, {"source", str}},
            {}));
    return t;
}

// --- tool dispatch ---------------------------------------------------------
// Pure w.r.t. the transport (no stdio): phase 2 (live-attach socket) reuses it.
// Soft failures set `is_error` and return an {error} object; a thrown exception
// is caught by the caller and turned into an isError result too.

json call_tool(Session& s, const std::string& name, const json& args, bool& is_error) {
    is_error = false;
    auto geti = [&](const char* k, int def) { return args.contains(k) && args[k].is_number_integer() ? args[k].get<int>() : def; };
    auto err = [&](const std::string& msg) {
        is_error = true;
        return json{{"error", msg}};
    };

    if (name == "snapshot") return to_json(s.snapshot());

    if (name == "exec") {
        const std::string line = args.value("line", "");
        const bool quit = !s.exec(line);
        json r;
        r["ok"] = true;
        r["quit"] = quit;
        r["status"] = s.status();
        r["snapshot"] = to_json(s.snapshot());
        return r;
    }
    if (name == "key") {
        const std::string k = args.value("key", "");
        const bool quit = !s.key(k);
        json r;
        r["ok"] = true;
        r["quit"] = quit;
        r["status"] = s.status();
        r["snapshot"] = to_json(s.snapshot());
        return r;
    }
    if (name == "complete") {
        const auto c = s.complete(args.value("line", ""));
        json r;
        r["candidates"] = c.candidates;
        r["usage"] = c.usage;
        return r;
    }
    if (name == "list") return to_json(s.list_view());

    if (name == "blame") {
        const int k = geti("k", 12);
        const bool toward = args.contains("toward") && args["toward"].is_boolean() ? args["toward"].get<bool>() : true;
        const auto b = s.blame_view(k, toward);
        if (!b.present) return err("no blame yet — run a prompt first (exec \"prompt ...\")");
        return to_json(b);
    }
    if (name == "neuron") {
        const auto sn = s.snapshot();
        const int layer = geti("layer", sn.layer);
        const int neuron = geti("neuron", sn.neuron);
        const auto n = s.neuron_info(layer, neuron);
        if (!n.present) return err("no neuron L" + std::to_string(layer) + ".N" + std::to_string(neuron));
        return to_json(n);
    }
    if (name == "state") {
        const auto st = s.state_view();
        if (!st.present) return err("no run yet — run a prompt first");
        return to_json(st);
    }
    if (name == "dose") {
        const auto sn = s.snapshot();
        const int layer = geti("layer", sn.layer);
        const int neuron = geti("neuron", sn.neuron);
        s.exec("dose " + std::to_string(layer) + "." + std::to_string(neuron));   // inline (headless)
        const auto d = s.dose_view();
        if (!d.present || d.layer != layer || d.neuron != neuron) return err(s.status());
        return to_json(d);
    }
    if (name == "render") {
        const int w = geti("width", 180);
        const int h = geti("height", 56);
        const bool color = args.contains("color") && args["color"].is_boolean() && args["color"].get<bool>();
        json r;
        r["frame"] = render_to_string(s, w, h, color);
        return r;
    }
    if (name == "labels") {
        LabelDb* db = s.labels();
        if (!db) return err("labels are off: the labels file couldn't be read");
        db->refresh();
        auto gets = [&](const char* k, const std::string& def) { return args.contains(k) && args[k].is_string() ? args[k].get<std::string>() : def; };
        const std::string action = gets("action", "search");
        const std::string ps = gets("pole", "");
        const char pole = ps == "+" ? '+' : (ps == "-" || ps == "−") ? '-' : 0;
        json r;
        r["path"] = db->path();
        if (action == "search") {
            json ls = json::array();
            for (const auto& l : db->search(gets("query", ""))) ls.push_back(label_json(l));
            r["labels"] = ls;
            return r;
        }
        if (!args.contains("layer") || !args.contains("neuron")) return err("labels " + action + " needs layer and neuron");
        const int layer = geti("layer", -1), neuron = geti("neuron", -1);
        if (action == "set") {
            Label l;
            l.layer = layer;
            l.neuron = neuron;
            l.pole = pole;
            l.label = gets("label", "");
            l.notes = gets("notes", "");
            l.source = gets("source", "claude");
            l.author = "claude";
            if (args.contains("tags") && args["tags"].is_array())
                for (const auto& t : args["tags"])
                    if (t.is_string()) l.tags.push_back(t.get<std::string>());
            r["added"] = db->upsert(l);   // throws on a bad neuron/empty label → isError
            json ls = json::array();
            for (const auto& x : db->at(layer, neuron)) ls.push_back(label_json(x));
            r["labels"] = ls;
            return r;
        }
        if (action == "remove") {
            const auto gone = db->remove(layer, neuron, pole, ps.empty(), gets("source", ""));
            json ls = json::array();
            for (const auto& x : gone) ls.push_back(label_json(x));
            r["removed"] = ls;
            return r;
        }
        return err("labels: unknown action " + action + " (search|set|remove)");
    }
    if (name == "attention") {
        const auto sn = s.snapshot();
        const auto a = s.attention_view(geti("layer", sn.layer), geti("pos", sn.pos), geti("k", 5));
        if (!a.present) return err(a.error);
        return to_json(a);
    }
    if (name == "trace") {
        const std::string corrupt = args.value("corrupt", "");
        if (corrupt.empty()) return err("trace needs `corrupt`");
        s.exec("corrupt " + corrupt);
        s.exec("trace " + args.value("site", std::string("resid")));   // inline (headless)
        const auto t = s.trace_view();
        if (!t.present || t.done == 0) return err(s.status());
        return to_json(t);
    }
    if (name == "probe") {
        auto strs = [&](const char* k) {
            std::vector<std::string> out;
            if (args.contains(k) && args[k].is_array())
                for (const auto& x : args[k])
                    if (x.is_string()) out.push_back(x.get<std::string>());
            return out;
        };
        std::vector<int> layers;
        if (args.contains("layers") && args["layers"].is_array())
            for (const auto& x : args["layers"])
                if (x.is_number_integer()) layers.push_back(x.get<int>());
        const auto v = s.probe(strs("pos"), strs("neg"), layers, geti("k", 4));
        if (!v.present) return err(v.error);
        return to_json(v);
    }
    if (name == "examples") {
        const auto sn = s.snapshot();
        const auto v = s.examples_view(geti("layer", sn.layer), geti("neuron", sn.neuron), geti("k", 8));
        if (!v.present) return err(v.error);
        return to_json(v);
    }
    if (name == "model_info") {
        const auto sn = s.snapshot();
        json j;
        j["arch"] = sn.arch;
        j["name"] = sn.model_name;
        j["n_layer"] = sn.n_layer;
        j["n_embd"] = sn.n_embd;
        j["n_ff"] = sn.n_ff;
        j["n_head"] = sn.n_head;
        j["n_head_kv"] = sn.n_head_kv;
        j["n_vocab"] = sn.n_vocab;
        j["n_ctx_train"] = sn.n_ctx_train;
        j["tied_embd"] = sn.tied_embd;
        return j;
    }
    return err("unknown tool: " + name);   // unreachable; names are checked before dispatch
}

// --- JSON-RPC envelope -----------------------------------------------------

void write_msg(const json& m) {
    // error_handler_t::replace: vocab pieces are raw BPE bytes, so a lens/token
    // text can be a partial UTF-8 sequence (e.g. one byte of a CJK char). The
    // default handler throws on those, which used to take the whole server down;
    // replace emits U+FFFD for the bad bytes instead.
    std::cout << m.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
    std::cout.flush();
}

json rpc_result(const json& id, json result) {
    return json{{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}};
}
json rpc_error(const json& id, int code, const std::string& message) {
    return json{{"jsonrpc", "2.0"}, {"id", id}, {"error", json{{"code", code}, {"message", message}}}};
}

// Returns false to stop the loop (a quit tool call).
bool handle_request(Session& s, const json& req) {
    const bool is_notification = !req.contains("id");
    const json id = is_notification ? json(nullptr) : req["id"];
    const std::string method = req.value("method", "");

    if (method == "initialize") {
        const json& p = req.contains("params") ? req["params"] : json::object();
        const std::string proto = p.contains("protocolVersion") && p["protocolVersion"].is_string()
                                      ? p["protocolVersion"].get<std::string>()
                                      : kDefaultProtocol;
        json result;
        result["protocolVersion"] = proto;
        result["capabilities"] = {{"tools", json::object()}};
        result["serverInfo"] = {{"name", "neuronaut"}, {"version", kServerVersion}};
        result["instructions"] =
            "Drive a neuronaut debugging session over one GGUF model. Start with exec "
            "\"prompt The capital of France is\", then read `snapshot`, `blame`, `neuron`, `state`, `list`. "
            "Move with `key` or exec (pos/target/ablate/invert/scale/find/dose). `render` shows the human frame. "
            "Causal questions: `trace` (activation patching against a corrupted prompt), `dose`. Finding neurons: `probe`, "
            "`find` (exec), blame. Naming them: `examples` + `neuron` lens, then `labels` action=set with source 'auto'.";
        write_msg(rpc_result(id, result));
        return true;
    }
    if (method == "ping") {
        write_msg(rpc_result(id, json::object()));
        return true;
    }
    if (method.rfind("notifications/", 0) == 0) return true;   // notifications get no reply

    if (method == "tools/list") {
        write_msg(rpc_result(id, json{{"tools", tools_list()}}));
        return true;
    }
    if (method == "tools/call") {
        const json& p = req.contains("params") ? req["params"] : json::object();
        const std::string name = p.value("name", "");
        const json args = p.contains("arguments") && p["arguments"].is_object() ? p["arguments"] : json::object();
        if (!tool_names().count(name)) {
            write_msg(rpc_error(id, -32602, "unknown tool: " + name));
            return true;
        }
        bool is_error = false;
        bool quit = false;
        json content_obj;
        try {
            content_obj = call_tool(s, name, args, is_error);
            if ((name == "exec" || name == "key") && content_obj.value("quit", false)) quit = true;
        } catch (const std::exception& e) {
            is_error = true;
            content_obj = json{{"error", e.what()}};
        }
        json result;
        result["content"] = json::array({json{{"type", "text"}, {"text", content_obj.dump(2, ' ', false, json::error_handler_t::replace)}}});
        result["structuredContent"] = content_obj;
        result["isError"] = is_error;
        write_msg(rpc_result(id, result));
        return !quit;
    }

    if (!is_notification) write_msg(rpc_error(id, -32601, "method not found: " + method));
    return true;
}

}  // namespace

int run_mcp(Session& s) {
    s.post = nullptr;   // headless: long jobs (dose) run inline, one at a time
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json req;
        try {
            req = json::parse(line);
        } catch (const std::exception& e) {
            write_msg(rpc_error(json(nullptr), -32700, std::string("parse error: ") + e.what()));
            continue;
        }
        bool keep_going = true;
        if (req.is_array()) {
            for (const auto& r : req)
                if (r.is_object()) keep_going = handle_request(s, r) && keep_going;
        } else if (req.is_object()) {
            keep_going = handle_request(s, req);
        } else {
            write_msg(rpc_error(json(nullptr), -32600, "invalid request"));
        }
        if (!keep_going) break;
    }
    return 0;
}

}  // namespace neuronaut
