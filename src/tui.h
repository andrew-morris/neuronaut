// tui.h — the debugger-style terminal UI.
//
// One screen, fixed panes, one cursor — OllyDbg's model applied to an LLM.
// Every kind of object has exactly one home pane, the cursor is a single
// (token position, layer, neuron) triple that every pane follows, and selecting
// an object anywhere (a blame row, a patch, a find hit) moves the cursor to it,
// so its home panes update in place — you never leave the screen.
//
//  ┌ header: model · cursor (pos · layer · neuron) · target · patches · jobs ────────────┐
//  ├ TOKENS   the prompt, cursor position, watched neurons' activations under each token ┤
//  ├ GRAPH ────────┬ LIST ─────────────────────────┬ NEURON ─────────────────────────────┤
//  │ layers, per-  │ the one pane whose content    │ the cursor neuron: activation here  │
//  │ layer blame   │ switches (blame / fires /     │ and over time, blame, what it writes│
//  │ for the target│ find / controls); its title   │ (lens), what it reads, where its    │
//  │               │ says which                    │ bytes are                           │
//  ├ BYTES ────────┴───────────────────────────────┼ STATE ──────────────────────────────┤
//  │ quant block under the byte cursor             │ next token (base vs patched), lens  │
//  ├ PATCHES ──────────────────────────────────────┤ after the cursor layer, target by   │
//  │ patches ⚑ and watches ◉                       │ layer, continuation                 │
//  └───────────────────────────────────────────────┴─────────────────────────────────────┘
//  status · hotkeys for the focused pane + global keys
//
// Keys mean the same thing everywhere: h/l move in time (token position),
// PgUp/PgDn move between layers, Tab/Shift-Tab move focus between panes, j/k
// move within the focused pane, x/i/w/D act on the cursor neuron.
//
// Occasional, larger things are overlays that close with Esc: help (?), the
// model browser (M: tensors + metadata) and the dose–response plot (D starts a
// sweep, P reopens the last one).
//
// All state lives in `Session`, which knows nothing about terminals: keys and
// commands mutate it, `render()` turns it into an FTXUI element. That split lets
// `neuronaut render` draw frames headlessly for tests and lets a future script
// or MCP front end drive the same state machine.
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "blame.h"
#include "edits.h"
#include "forward.h"
#include "gguf_file.h"
#include "labels.h"
#include "lens.h"
#include "model_layout.h"
#include "neuron.h"
#include "trace.h"
#include "actindex.h"
#include "probe.h"
#include "score.h"

namespace ftxui {
class Node;
using Element = std::shared_ptr<Node>;
}  // namespace ftxui

namespace neuronaut {

class Tokenizer;
class SteerUnits;

enum class Slice { Gate = 0, Up = 1, Down = 2 };

// Panes that take keys, in Tab order. TOKENS needs none (h/l move time from
// anywhere) and STATE is a read-out.
enum class Pane { Graph = 0, List, Neuron, Bytes, Patches };
constexpr int kPanes = 5;

// What the LIST pane is showing.
enum class ListKind { Blame, Fires, Find, Controls, Labels, Probe, Examples };

enum class Overlay { None, Help, Model, Plot, Attn, Trace };

class Session {
public:
    Session(std::unique_ptr<GgufFile> f, ModelLayout m);
    ~Session();

    // Execute one command line (without the leading ':'). Returns false on quit.
    bool exec(const std::string& line);
    // One key in navigation mode ("h", "tab", "shift_tab", "enter", "escape",
    // "pgup", …). Returns false on quit.
    bool key(const std::string& k);

    ftxui::Element render();

    const std::string& status() const { return status_; }

    // Set by the TUI loop while the user is typing a command.
    std::string* cmdline = nullptr;
    int* cmd_cursor = nullptr;

    // Command-line completion: candidate full lines for what has been typed
    // (also drawn live), and one Tab press (longest common prefix, then cycle).
    struct Completion {
        std::vector<std::string> candidates;
        std::string usage;
    };
    Completion complete(const std::string& line);
    void tab(std::string& line);

    // Entered command lines, oldest first (↑/↓ in command mode; prompt completion).
    std::vector<std::string> history;

    // Set by the interactive loop: run a closure on the UI thread (and redraw).
    // When unset (headless `render`), long jobs run synchronously instead.
    std::function<void(std::function<void()>)> post;

    // -------------------------------------------------------------------------
    // Machine-readable views (for the `mcp` front end; see docs/mcp.md).
    //
    // These are plain-data mirrors of what the render_* methods read privately,
    // so an LLM over MCP can read structured state instead of parsing the ASCII
    // frame. They hold no FTXUI or JSON types: serialization lives in mcp.cpp,
    // keeping this library free of a wire format the same way it is free of a
    // terminal. All are non-const because a few warm a cache (lens, state lens).
    // -------------------------------------------------------------------------
    struct TokenView {
        int pos, id;
        std::string text;
        bool appended;        // added by :accept (pos >= prompt_len)
    };
    struct TargetView {
        int id;
        std::string text;
        bool pinned;
        float prob;
        int rank;             // 1 = argmax
    };
    struct PatchView {
        int layer, neuron;
        std::string op;          // scale|add|set|steer (edits.h)
        float scale;             // scale: k (0 = ablate, -1 = invert); 1 otherwise
        float value;             // add / set: the raw value; steer: the resolved add
        float dose;              // steer: dose in hidden-state sizes
        std::string pole;        // steer: "+" / "-"
        std::string scope;       // all|prompt
        std::string text;        // edit_text(): "L10.N2407− steer 0.50 @prompt"
        std::string portability; // gguf|control-vector|neuronaut-only
        std::string labels;      // LabelDb::brief, "" if unlabelled
    };
    struct Snapshot {
        // model
        std::string arch, model_name;
        int n_layer, n_embd, n_ff, n_head, n_head_kv, n_vocab, n_ctx_train;
        bool tied_embd;
        // cursor + focus
        int pos, layer, neuron;
        std::string focus, overlay, slice;   // enum names, lower-case
        bool chat;
        bool think;                          // chat with the reasoning block open (:think)
        // prompt
        bool has_run;
        std::vector<TokenView> tokens;
        int prompt_len;
        // target at the cursor position
        bool has_target;
        TargetView target;
        // interventions
        std::vector<PatchView> patches;
        std::vector<std::pair<int, int>> watches;
        // last status line, and whether a dose sweep is in flight
        std::string status;
        bool busy;
    };
    struct ListRowView {
        std::string kind;     // neuron|attn|embed  (empty for a section divider)
        int layer, neuron;
        float value;
        std::string detail;
        bool section;
        std::string label;    // section heading text
        std::string labels;   // the neuron's labels (LabelDb::brief), "" if none
    };
    struct ListView {
        std::string kind;     // blame|fires|find|controls|labels
        int selected;
        std::vector<ListRowView> rows;
    };
    struct BlameItemView {
        std::string kind;     // neuron|attn|embed
        int layer, neuron;
        float act, dot, contrib;
    };
    struct BlameView {
        bool present;
        int pos, target_id;
        std::string target_text;
        float logit, prob;
        int rank;
        float embed, sum_attn, sum_neurons, unexplained;
        std::vector<BlameItemView> items;   // top(k) toward or against
    };
    struct LensTokenView {
        int id;
        std::string text;
        float logit;
    };
    struct FootprintView {
        uint64_t first_byte, last_byte;
        int64_t n_blocks;
        bool contiguous;
        int64_t shared_with;
    };
    struct NeuronInfoView {
        bool present;
        int layer, neuron;
        bool has_act;
        float act_here;
        int act_rank, act_rank_of;          // |a| rank within the layer at the cursor pos
        std::vector<LensTokenView> promotes, suppresses;
        FootprintView up, gate, down;
        std::vector<Label> labels;          // every record, all poles and sources
    };
    struct StateTokenView {
        int id;
        std::string text;
        float prob;
    };
    struct StateView {
        bool present;
        int pos, layer;
        std::vector<StateTokenView> next_base;      // next-token dist (baseline, no patches)
        std::vector<StateTokenView> next_patched;   // next-token dist (with patches)
        bool has_base;                              // false when there are no patches
        std::vector<float> target_by_layer;         // lens logit of target after each layer
        std::vector<StateTokenView> lens_after_layer;
        float lens_target_prob;
        int lens_target_rank;
        std::string gen_base, gen_patched;          // :gen continuations, if any
    };
    struct DosePointView {
        float dose, tested_delta, band_min, band_max;
        bool comparable, outside;
    };
    struct DoseVerdictView {
        std::vector<DosePointView> points;          // excludes the x1 baseline dose
        int outside, comparable;
        bool specific;                              // outside*2 > comparable
        bool understated;                           // push > 1.5x the largest control push
        float understate_factor;                    // push / max control push
    };
    struct DoseView {
        bool present;
        int layer, neuron, pos, target_id;
        std::string target_text;
        float push;
        std::vector<float> doses, curve;
        std::vector<std::pair<int, float>> controls;   // (neuron, push)
        std::vector<std::vector<float>> ctrl;          // [control][dose]
        int done, total;
        bool running;
        double seconds;
        DoseVerdictView verdict;
    };

    // The attention overlay's data: softmax weights of every head of `layer` from
    // query `pos` (the cursor is moved there), each head's blame on the target, and
    // its k most-attended keys. Linear-attention layers have no pattern (linear=true,
    // heads carry blame only).
    struct AttnKeyView {
        int pos;
        std::string text;
        float weight;
    };
    struct AttnHeadView {
        int head;
        float blame;                     // contribution to the target logit (blame.h)
        std::vector<AttnKeyView> top;    // most-attended keys, heaviest first
    };
    struct AttentionView {
        bool present;
        std::string error;
        int layer, pos;
        bool linear;
        std::vector<AttnHeadView> heads;
    };
    AttentionView attention_view(int layer, int pos, int k);

    // The last causal trace (:corrupt + :trace, trace.h): restoration per (layer,
    // position), plus the token texts of both prompts.
    struct TraceView {
        bool present;
        std::string site, corrupt;
        int target_id;
        std::string target_text;
        float p_clean, p_corrupt;
        int n_layer, n_tokens;
        std::vector<std::string> clean_tokens, corrupt_tokens;
        std::vector<int> differ;
        std::vector<std::vector<float>> restored;   // [layer][pos]; NaN = not run
        int best_layer, best_pos;
        float best;
        int done, total;
        bool running;
    };
    TraceView trace_view();

    // A neuron's top corpus examples (actindex.h), both poles; present=false with
    // `error` saying how to build the index when there is none.
    struct ExampleView {
        float act;
        int doc, pos;
        std::string token, context;   // context: ~12 tokens before the peak, «peak», 3 after
    };
    struct ExamplesView {
        bool present;
        std::string error, corpus;
        int layer, neuron, n_docs;
        std::vector<ExampleView> positive, negative;
    };
    ExamplesView examples_view(int layer, int neuron, int k);

    // :probe — the last contrast probe's hits (probe.h), layer by layer.
    struct ProbeView {
        bool present;
        std::vector<std::string> pos, neg;
        std::vector<ProbeHit> hits;
        std::string error;
    };
    // Runs a probe (texts wrapped like the current prompt: chat if :chat) and lists it.
    ProbeView probe(const std::vector<std::string>& pos, const std::vector<std::string>& neg, const std::vector<int>& layers, int k);

    // A neuron as a detector (score.h): AUC between POS and NEG texts for the given
    // neurons, or the `top` best per layer of `layers` with a shuffled-label null.
    // Texts are plain, or "PROMPT<TAB>REPLY" with chat (only the reply is read).
    // Pre-edit activations, under the session's edits.
    struct ScoreView {
        bool present;
        std::string error;
        std::string read;
        bool chat;
        int n_pos, n_neg;
        std::vector<NeuronScore> scores;
        // Per score, per text (POS then NEG): the text with the read token in «».
        std::vector<std::vector<std::string>> contexts;
        std::map<int, std::vector<NullBest>> null;
    };
    ScoreView score(const std::vector<std::string>& neurons, const std::vector<std::string>& pos, const std::vector<std::string>& neg,
                    const std::string& read, bool chat, int top, const std::vector<int>& layers, int n_null);

    Snapshot       snapshot();
    ListView       list_view();
    BlameView      blame_view(int k, bool toward);
    NeuronInfoView neuron_info(int layer, int neuron);
    StateView      state_view();
    DoseView       dose_view();

    // The neuron-label database for this model (labels.h); null if its file is
    // unreadable (the status line says why). `:label` / `:labels` drive it.
    LabelDb* labels() { return labels_.get(); }

private:
    // ---- cursor ---------------------------------------------------------------
    void select_neuron(int layer, int neuron);   // moves the layer cursor too
    void select_layer(int layer);                // keeps the neuron index
    void set_pos(int pos);
    // Sequences up to kFullCapture tokens capture every position on each rerun; past
    // that only kCaptureWindow positions around the cursor are kept (RunOptions
    // cap_lo/cap_hi), and moving the cursor out of the window re-runs. Logits,
    // activations, residuals and blame are all read through ForwardResult's
    // windowed accessors; watches are captured at every position regardless.
    static constexpr int kFullCapture = 512;
    static constexpr int kCaptureWindow = 128;
    std::pair<int, int> capture_window() const;   // [lo, hi) for the next rerun
    void load_neuron();
    const TensorInfo* slice_tensor() const;
    bool busy() const { return (dose_ && dose_->running) || trace_running_; }

    // ---- model runs -----------------------------------------------------------
    void ensure_engine();
    const SteerUnits& steer_units();   // computed (or loaded from cache) on first use
    void rebuild_iv();
    void rerun();
    void recompute_blame();
    void set_prompt(const std::string& text, bool chat, bool think = false);
    // Greedy continuation of up to n tokens; with `stop` non-empty, stop as soon as the
    // generated text contains it (the token that completes it is kept, so :accept
    // includes e.g. the closing </think>).
    void generate(int n, const std::string& stop = {});
    // Cursor to the next token (after the cursor, wrapping) whose piece is exactly
    // `piece`, then `offset` further (clamped to the sequence). Returns the matched
    // token's index, or -1 if none matches (cursor unmoved).
    int seek_token(const std::string& piece, int offset);
    void toggle_patch(int layer, int neuron, float scale);
    // Add an edit, replacing any edit on the same neuron it conflicts with
    // (edits.h same_target); an identical edit toggles off.
    void toggle_edit(const Edit& e);
    void toggle_watch(int layer, int neuron);
    std::string token_text(int id) const;
    const std::vector<std::string>& vocab_text();
    const LensResult* neuron_lens();   // cached static lens of the cursor neuron
    void update_state_lens();          // lens of the residual after layer_ at pos_

    // ---- LIST -----------------------------------------------------------------
    struct ListRow {
        BlameItem::Kind kind = BlameItem::Neuron;   // Neuron / Attn / Embed
        int layer = -1, neuron = -1;
        float value = 0;        // what the bar shows
        std::string detail;     // dim text after the value
        bool section = false;   // a non-selectable divider
        std::string label;      // for sections
        std::string note;       // the neuron's labels, drawn after `detail` (filled by list_rows)
        std::string name;       // replaces the L.N column (examples: "doc 12")
        bool inert = false;     // selecting it doesn't move the cursor
    };
    std::vector<ListRow> list_rows();
    void list_move(int delta);   // move the selection and follow it

    // ---- panes ----------------------------------------------------------------
    ftxui::Element pane(const std::string& title, ftxui::Element body, std::optional<Pane> p);
    ftxui::Element render_header();
    ftxui::Element render_tokens();
    ftxui::Element render_graph();
    ftxui::Element render_list();
    ftxui::Element render_neuron();
    ftxui::Element render_bytes();
    ftxui::Element render_patches();
    ftxui::Element render_state();
    ftxui::Element render_footer();
    ftxui::Element render_keybar();
    ftxui::Element render_help();
    ftxui::Element render_model();
    ftxui::Element render_dose();
    ftxui::Element render_attention();
    ftxui::Element render_trace();

    // ---- causal trace (:corrupt TEXT, :trace [resid|attn|mlp]) -----------------
    // Runs on the dose worker thread (one background job at a time).
    void start_trace(Site site);
    std::string corrupt_text_;
    std::vector<int32_t> trace_clean_, trace_corrupt_;   // the token sequences of the last trace
    std::optional<Trace> trace_;
    bool trace_running_ = false;

    // ---- attention overlay (A / :attn) ------------------------------------------
    // One extra forward capturing layer_'s softmax weights for query pos_
    // (RunOptions::attn_probs_*), from the layer when the residual entering it is
    // captured (Forward::run_from). Recomputed lazily when the cursor moves.
    struct AttnCache {
        int layer = -1, pos = -1;
        bool linear = false;
        int n_kv = 0, n_head = 0;
        std::vector<float> probs;   // [n_head][n_kv]
        std::string error;
    };
    const AttnCache& attention();
    std::optional<AttnCache> attn_;
    int attn_head_ = 0;
    std::vector<std::string> context_neurons() const;

    std::unique_ptr<GgufFile> file_;
    ModelLayout m_;

    // cursor + focus
    int pos_ = 0;
    int layer_ = 0;
    int neuron_ = 0;
    Pane focus_ = Pane::List;
    Overlay overlay_ = Overlay::None;

    // NEURON / BYTES
    Slice slice_ = Slice::Up;
    int dim_ = 0;
    std::optional<NeuronView> nv_;
    std::map<std::pair<int, int>, LensResult> lens_cache_;

    // LIST
    ListKind list_kind_ = ListKind::Blame;
    int list_sel_ = 0;
    std::string find_text_;
    std::vector<Writer> find_hits_;
    int find_npos_ = 0;
    std::string labels_query_;   // :labels QUERY

    // :probe / :examples
    std::vector<std::string> probe_pos_, probe_neg_;
    std::vector<ProbeHit> probe_hits_;
    const ActIndex* act_index();          // lazily loaded <model>.gguf.acts-index; null if none
    std::string example_context(const ActExample& e, int before = 12, int after = 3) const;
    std::optional<ActIndex> act_index_;
    std::string act_index_error_;
    bool act_index_tried_ = false;

    // neuron labels (labels.h)
    std::unique_ptr<LabelDb> labels_;
    std::string label_brief(int layer, int neuron) const { return labels_ ? labels_->brief(layer, neuron) : std::string(); }

    // PATCHES (selection runs over patches_, then watch_)
    int patch_sel_ = 0;

    // model overlay
    int model_sel_ = 0;
    std::string tensor_filter_;

    // engine + run state
    std::unique_ptr<Tokenizer> tok_;
    std::unique_ptr<Forward> fw_;
    bool chat_ = false;
    bool think_ = false;   // chat_ with the reasoning block open
    std::vector<int32_t> tokens_;
    size_t prompt_len_ = 0;               // tokens beyond this were appended with :accept
    std::vector<Edit> patches_;            // the live edit set (edits.h); ⚑ in the UI
    std::string edits_name_;               // last :save / :load name
    std::string edits_notes_;
    std::unique_ptr<SteerUnits> units_;
    Interventions iv_;
    std::vector<std::pair<int, int>> watch_;
    std::optional<ForwardResult> res_;    // current run (with patches), full captures
    std::optional<ForwardResult> base_;   // baseline (no patches), logits; only when patches exist
    int target_ = -1;
    bool target_pinned_ = false;
    std::optional<Blame> blame_;
    std::vector<float> u_;
    std::vector<std::vector<float>> dots_;
    std::vector<std::vector<float>> odots_;   // Woᵀu per layer, for per-head blame (blame.h out_dots)
    int dots_target_ = -1;
    std::vector<float> layer_attn_, layer_ffn_;   // per-layer blame, for GRAPH
    std::vector<float> target_by_layer_;          // lens logit of the target after each layer at pos_
    std::string gen_base_, gen_patched_;
    std::vector<int32_t> gen_base_ids_, gen_patched_ids_;
    struct StateLens {
        int pos = -1, layer = -1;
        std::vector<std::pair<int, float>> top;   // (token, prob)
        float target_prob = 0;
        int target_rank = 0;
    };
    StateLens state_lens_;

    // ---- dose–response sweep ---------------------------------------------------
    // Scale the neuron's activation by k ∈ doses, measure P(target at pos); the
    // same for `controls` — same-layer neurons whose push |a|·‖down‖ at pos is
    // closest to the tested neuron's.
    struct DoseSweep {
        int layer = 0, neuron = 0, pos = 0, target = 0;
        float push = 0;
        std::vector<float> doses;
        std::vector<float> curve;
        std::vector<std::pair<int, float>> controls;   // (neuron, push)
        std::vector<std::vector<float>> ctrl;          // [control][dose]
        int done = 0, total = 0;
        bool running = false;
        double seconds = 0;
    };
    void start_dose(int layer, int neuron);
    void join_worker();
    const std::vector<float>& down_norms(int layer);
    std::optional<DoseSweep> dose_;
    std::thread worker_;
    std::atomic<bool> cancel_{false};
    std::map<int, std::vector<float>> down_norm_cache_;

    // ---- command line ------------------------------------------------------------
    std::vector<std::string> vocab_text_;
    std::string tab_last_;
    std::vector<std::string> tab_cands_;
    size_t tab_idx_ = 0;

    std::string status_;
};

// Interactive loop. Takes over the terminal until quit.
int run_tui(Session& s);

// JSON-RPC-over-stdio MCP server loop on one headless Session (docs/mcp.md).
// Reads newline-delimited requests from stdin, writes responses to stdout,
// returns on EOF or a quit tool call. Defined in src/mcp.cpp.
int run_mcp(Session& s);

// Render one frame of `width` x `height` to a string (plain text unless `color`).
std::string render_to_string(Session& s, int width, int height, bool color);

}  // namespace neuronaut
