// tui.cpp — see tui.h.
#include "tui.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <regex>
#include <sstream>

#include "blocks.h"
#include "format.h"
#include "ftxui/component/app.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/canvas.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/screen.hpp"
#include "oracle.h"
#include "export.h"
#include "steer.h"

namespace neuronaut {

using namespace ftxui;

namespace {

// ---------------------------------------------------------------------------
// Shared visual vocabulary. Same meaning everywhere:
//   blue  = positive / pushes the target up      orange = negative / pushes down
//   ⚑     = patched                              ◉      = watched
//   dim italic tokens = appended answer (:accept)
// ---------------------------------------------------------------------------

constexpr int kHeatCells = 64;   // width of a vector heat strip
const Color kOrange = Color::RGB(255, 150, 30);
const Color kFocus = Color::RGB(90, 170, 255);
const Color kLabel = Color::RGB(120, 220, 160);   // neuron labels (labels.h), wherever a neuron is listed

std::string shape_short(const TensorInfo& t) {
    std::string s;
    for (int d = 0; d < t.n_dims; ++d) {
        if (d) s += "×";
        s += std::to_string(t.ne[d]);
    }
    return s;
}

std::string ln(int l, int n) { return "L" + std::to_string(l) + ".N" + std::to_string(n); }

// Compact edit marker for list rows: ⚑×0, ⚑×7, ⚑+3.0, ⚑=1.5, ⚑−s0.5@p
std::string edit_tag(const neuronaut::Edit& e) {
    using neuronaut::EditOp;
    std::string t = "⚑";
    switch (e.op) {
        case EditOp::Scale: t += "×" + neuronaut::fmt_f(e.k, 1).substr(e.k >= 0 ? 1 : 0); break;
        case EditOp::Add: t += neuronaut::fmt_f(e.value, 1); break;
        case EditOp::Set: t += "=" + neuronaut::fmt_f(e.value, 1); break;
        case EditOp::Steer: t += std::string(e.pole == '+' ? "+" : "−") + "s" + neuronaut::fmt_f(e.dose, 2).substr(1); break;
        case EditOp::RankOne: t += "rank-1"; break;
    }
    if (e.scope == neuronaut::EditScope::Prompt) t += "@p";
    return t;
}

// Diverging colour for a value normalised to [-1, 1]: blue positive, orange negative.
Color diverging(float t) {
    t = std::clamp(t, -1.0f, 1.0f);
    const auto a = static_cast<uint8_t>(std::lround(255.0f * std::fabs(t)));
    const auto b = static_cast<uint8_t>(std::lround(90.0f * std::fabs(t)));
    return t >= 0 ? Color::RGB(0, b, a) : Color::RGB(a, b, 0);
}
Color sign_color(float v) { return v >= 0 ? Color::RGB(80, 150, 255) : kOrange; }

// One row of coloured cells summarising a whole vector: each cell shows the
// largest-magnitude element in its bucket, scaled by the vector's max |v|.
// If `cursor` >= 0, a second line puts a caret under that element's cell.
Element heat_strip(const std::vector<float>& v, int cursor, int n_cells = kHeatCells) {
    const int n = static_cast<int>(v.size());
    float vmax = 1e-12f;
    for (float x : v) vmax = std::max(vmax, std::fabs(x));
    Elements cells;
    int cursor_cell = -1;
    for (int c = 0; c < n_cells; ++c) {
        const int lo = c * n / n_cells, hi = std::max(lo + 1, (c + 1) * n / n_cells);
        float pick = 0;
        for (int i = lo; i < hi && i < n; ++i)
            if (std::fabs(v[static_cast<size_t>(i)]) > std::fabs(pick)) pick = v[static_cast<size_t>(i)];
        cells.push_back(text(" ") | bgcolor(diverging(pick / vmax)));
        if (cursor >= lo && cursor < hi) cursor_cell = c;
    }
    Elements rows{hbox(std::move(cells))};
    if (cursor_cell >= 0) {
        rows.push_back(text(std::string(static_cast<size_t>(cursor_cell), ' ') + "^ e" + std::to_string(cursor)) | dim);
    }
    return vbox(std::move(rows));
}

// Horizontal bar of |v|/vmax × width cells, in eighth-block resolution.
std::string hbar(float v, float vmax, int width) {
    static const char* eighths[] = {"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};
    const float frac = vmax > 0 ? std::min(1.0f, std::fabs(v) / vmax) : 0.0f;
    const int units = static_cast<int>(std::lround(frac * static_cast<float>(width) * 8.0f));
    std::string s;
    for (int i = 0; i < units / 8; ++i) s += "█";
    s += eighths[units % 8];
    return s;
}

// Histogram of `xs` over [-r, r] in `bins` columns, `rows` text rows tall,
// log-scaled counts (a handful of outliers vs 100k near-zero neurons).
Element histogram(const std::vector<float>& xs, float r, int bins, int rows) {
    std::vector<int> count(static_cast<size_t>(bins), 0);
    for (float x : xs) {
        int b = static_cast<int>(std::floor((x + r) / (2 * r) * static_cast<float>(bins)));
        b = std::clamp(b, 0, bins - 1);
        ++count[static_cast<size_t>(b)];
    }
    double cmax = 1;
    for (int c : count) cmax = std::max(cmax, std::log10(1.0 + c));
    static const char* lv[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    Elements lines;
    for (int row = rows - 1; row >= 0; --row) {
        Elements cells;
        for (int b = 0; b < bins; ++b) {
            const int c = count[static_cast<size_t>(b)];
            const double h = c ? std::log10(1.0 + c) / cmax * rows * 8 : 0;   // height in eighths
            const int fill = std::clamp(static_cast<int>(std::lround(h)) - row * 8, 0, 8);
            const float mid = (static_cast<float>(b) + 0.5f) / static_cast<float>(bins) * 2 - 1;
            cells.push_back(text(lv[fill]) | color(diverging(mid < 0 ? -0.8f : 0.8f)));
        }
        lines.push_back(hbox(std::move(cells)));
    }
    return vbox(std::move(lines));
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        ab += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    return (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : 0.0;
}

// Display form of a token: quoted-safe, without the surrounding quotes.
std::string tok_label(const std::string& piece, size_t max_len) {
    std::string t = quote_printable(piece, max_len);
    t = t.substr(1, t.size() - 2);
    return t.empty() ? "·" : t;
}

// The command table: drives Tab completion, the live hint line and the help
// overlay, so they can't drift apart.
enum class Arg { None, Neuron, Text, Token, Words, Int, Prompt };
struct CmdSpec {
    const char* name;
    const char* args;
    const char* help;
    Arg arg;
    std::vector<const char*> words = {};   // for Arg::Words
};
const std::vector<CmdSpec>& commands() {
    static const std::vector<CmdSpec> k = {
        {"prompt", "TEXT", "run a raw prompt through the model", Arg::Prompt},
        {"chat", "TEXT", "run a user message through the model's chat template (Qwen: thinking off)", Arg::Prompt},
        {"think", "TEXT", "like chat, with the reasoning block open: the model thinks, </think>, then answers", Arg::Prompt},
        {"gen", "[N] [until \"S\"]", "greedy continuation of up to N tokens (default 16, max 8192), baseline vs patched; until: stop once S appears", Arg::Int},
        {"accept", "[patched]", "append the last :gen answer to the prompt so its tokens can be inspected", Arg::Words, {"patched"}},
        {"append", "TEXT", "append TEXT to TOKENS (prefill the model's turn; \\n escapes, special tokens parsed)", Arg::Text},
        {"ablate", "L.N", "zero a neuron's activation at every position (again to remove)  [x]", Arg::Neuron},
        {"invert", "L.N", "multiply a neuron's activation by -1 (again to remove)  [i]", Arg::Neuron},
        {"scale", "L.N K", "multiply a neuron's activation by K", Arg::Neuron},
        {"steer", "L.N(+|-) DOSE [prompt]", "push a neuron's pole by DOSE hidden-state sizes (0.5 ≈ coherent); prompt: only while reading the prompt", Arg::Neuron},
        {"add", "L.N V [prompt]", "add V to a neuron's activation (raw units; prefer steer); prompt: only on the prompt", Arg::Neuron},
        {"set", "L.N V", "clamp a neuron's activation to V", Arg::Neuron},
        {"save", "NAME", "save the patches as an edit set (edits/<model>/NAME.json, in git), with each edit's portability", Arg::Text},
        {"load", "NAME|FILE", "replace the patches with a saved edit set", Arg::Text},
        {"edits", "", "list this model's saved edit sets", Arg::None},
        {"export", "[STEM]", "write the patches as a patched GGUF (scales) + llama.cpp control vector (adds) beside the model, or at STEM", Arg::Text},
        {"clear", "[L.N]", "remove a neuron's patches, or all of them", Arg::Neuron},
        {"watch", "L.N", "show a neuron's activation under every token  [w]", Arg::Neuron},
        {"unwatch", "L.N", "stop watching a neuron", Arg::Neuron},
        {"target", "[\" tok\"]", "pin the token blame explains (no arg: follow the top prediction)", Arg::Token},
        {"pos", "N | \"tok\"[+K]", "cursor to position N, or to the next token that is exactly tok (+K after it)  [h/l]", Arg::Int},
        {"dose", "[L.N]", "dose–response sweep vs dose-matched controls  [D]", Arg::Neuron},
        {"cancel", "", "stop a running dose sweep", Arg::None},
        {"s", "L.N", "move the cursor to a neuron", Arg::Neuron},
        {"find", "\" tok\"", "list the neurons that write a token most directly  [/]", Arg::Token},
        {"list", "KIND", "what LIST shows: blame / fires / find / controls / labels / probe / examples  [b f c L e]", Arg::Words,
         {"blame", "fires", "find", "controls", "labels", "probe", "examples"}},
        {"label", "[L.N[+|-]] TEXT [#tag]… [// notes]", "name a neuron (default: the cursor; +/- = one pole); saved to labels/, shown everywhere it's listed", Arg::Neuron},
        {"unlabel", "[L.N[+|-]]", "remove your (source manual) labels from a neuron (default: the cursor)", Arg::Neuron},
        {"labels", "[QUERY]", "list labelled neurons matching QUERY (words AND; #tag, src:NAME); sorted by |a| here when there's a run  [L]", Arg::Text},
        {"slice", "gate|up|down", "which weight slice BYTES shows  [g u d]", Arg::Words, {"gate", "up", "down"}},
        {"dim", "N", "move the BYTES cursor to embedding dim N", Arg::Int},
        {"model", "", "browse tensors and metadata  [M]", Arg::None},
        {"plot", "", "reopen the last dose–response plot  [P]", Arg::None},
        {"probe", "[L9,10] POS | POS // NEG | NEG", "neurons that fire at the last token of every POS text and beyond every NEG one (worst-case margin) → LIST", Arg::Text},
        {"examples", "[L.N]", "the neuron's top-activating corpus examples (follows the cursor) → LIST  [e]", Arg::Neuron},
        {"index", "", "about the corpus examples index (build: neuronaut index MODEL CORPUS.jsonl)", Arg::None},
        {"corrupt", "TEXT", "the corrupted prompt for :trace — same token count as the prompt, differing where the fact is (France → Italy)", Arg::Text},
        {"trace", "[resid|attn|mlp]", "causal trace: patch the clean run's resid / attn / mlp at every (layer, pos) into the corrupt run → map  [T]", Arg::Words, {"resid", "attn", "mlp"}},
        {"tplot", "", "reopen the last trace map  [T]", Arg::None},
        {"attn", "[L] [H]", "attention overlay: where each head of layer L looks from the cursor token, and its blame  [A]", Arg::Int},
        {"filter", "TEXT", "filter the model browser's tensor list", Arg::Text},
        {"help", "", "keys and commands  [?]", Arg::None},
        {"quit", "", "exit  [q]", Arg::None},
    };
    return k;
}

// Parse "3.1234", "3 1234" or "1234".
bool parse_seek(const std::string& arg, int cur_layer, int& layer, int& neuron) {
    std::string a = arg;
    std::replace(a.begin(), a.end(), '.', ' ');
    std::istringstream is(a);
    long x = 0, y = 0;
    if (!(is >> x)) return false;
    if (is >> y) {
        layer = static_cast<int>(x);
        neuron = static_cast<int>(y);
    } else {
        layer = cur_layer;
        neuron = static_cast<int>(x);
    }
    return true;
}

double rms_inv(const float* x, size_t n, double eps) {
    double ss = 0;
    for (size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    return 1.0 / std::sqrt(ss / static_cast<double>(n) + eps);
}

}  // namespace

// ===========================================================================
// Cursor
// ===========================================================================

Session::Session(std::unique_ptr<GgufFile> f, ModelLayout m) : file_(std::move(f)), m_(std::move(m)) {
    for (int l = 0; l < m_.n_layer; ++l) {
        if (m_.has_ffn(l)) {
            layer_ = l;
            break;
        }
    }
    load_neuron();
    status_ = m_.warnings.empty() ? "loaded " + file_->path() + " — :prompt TEXT or :chat MESSAGE to run it, ? for help"
                                  : "warning: " + m_.warnings.front();
    // Labels are optional: a broken labels file shouldn't stop the debugger.
    try {
        labels_ = std::make_unique<LabelDb>(m_);
    } catch (const std::exception& e) {
        status_ = std::string("labels off: ") + e.what();
    }
}

Session::~Session() {
    cancel_ = true;
    join_worker();
}

void Session::join_worker() {
    if (worker_.joinable()) worker_.join();
}

void Session::select_neuron(int layer, int neuron) {
    layer = std::clamp(layer, 0, m_.n_layer - 1);
    neuron = std::clamp(neuron, 0, m_.n_ff - 1);
    if (layer == layer_ && neuron == neuron_ && nv_) return;
    layer_ = layer;
    neuron_ = neuron;
    load_neuron();
}

void Session::select_layer(int layer) { select_neuron(layer, neuron_); }

void Session::set_pos(int pos) {
    if (!res_) return;
    pos_ = std::clamp(pos, 0, res_->n_tokens - 1);
    // Long sequences keep only a window of captures (capture_window); stepping out
    // of it re-runs the model with the window moved to the cursor.
    if (!res_->captured(pos_)) {
        rerun();
        return;
    }
    recompute_blame();
}

std::pair<int, int> Session::capture_window() const {
    const int T = static_cast<int>(tokens_.size());
    if (T <= kFullCapture) return {0, T};
    // Centred on the cursor, shifted to stay inside [0, T).
    const int lo = std::clamp(pos_ - kCaptureWindow / 2, 0, T - kCaptureWindow);
    return {lo, lo + kCaptureWindow};
}

int Session::seek_token(const std::string& piece, int offset) {
    if (!res_ || tokens_.empty()) return -1;
    const int T = static_cast<int>(tokens_.size());
    for (int k = 1; k <= T; ++k) {   // after the cursor first, then wrap around
        const int i = (pos_ + k) % T;
        if (tok_->piece(tokens_[static_cast<size_t>(i)]) == piece) {
            set_pos(i + offset);
            return i;
        }
    }
    return -1;
}

void Session::load_neuron() {
    try {
        nv_ = read_neuron(m_, layer_, neuron_);
    } catch (const std::exception& e) {
        nv_.reset();
        status_ = e.what();
    }
}

const TensorInfo* Session::slice_tensor() const {
    if (!m_.has_ffn(layer_)) return nullptr;
    const auto& L = m_.layers[static_cast<size_t>(layer_)];
    switch (slice_) {
        case Slice::Gate: return L.ffn_gate;
        case Slice::Up: return L.ffn_up;
        case Slice::Down: return L.ffn_down;
    }
    return nullptr;
}

const LensResult* Session::neuron_lens() {
    if (!nv_) return nullptr;
    const auto key = std::make_pair(layer_, neuron_);
    auto it = lens_cache_.find(key);
    if (it == lens_cache_.end()) {
        try {
            std::vector<float> w = nv_->down;
            apply_write_gain(m_, layer_, w);   // what it writes, through a post-MLP norm (Gemma)
            it = lens_cache_.emplace(key, logit_lens(*file_, m_, w, 8)).first;
        } catch (const std::exception&) {
            return nullptr;
        }
    }
    return &it->second;
}

// ===========================================================================
// Model runs
// ===========================================================================

void Session::ensure_engine() {
    if (!tok_) tok_ = std::make_unique<Tokenizer>(file_->path());
    if (!fw_) fw_ = std::make_unique<Forward>(*file_, m_);
}

std::string Session::token_text(int id) const { return tok_ ? tok_->piece(id) : "#" + std::to_string(id); }

const SteerUnits& Session::steer_units() {
    ensure_engine();
    if (!units_) units_ = std::make_unique<SteerUnits>(file_->path(), m_, *fw_, *tok_);
    return *units_;
}

void Session::rebuild_iv() {
    iv_ = Interventions{};
    try {
        const SteerUnits* u = needs_units(patches_) ? &steer_units() : nullptr;
        apply_edits(patches_, m_, u, iv_);
    } catch (const std::exception& e) {
        status_ = std::string("edits: ") + e.what();
    }
    iv_.prompt_len = static_cast<int>(prompt_len_);
    gen_base_.clear();
    gen_patched_.clear();
    gen_base_ids_.clear();
    gen_patched_ids_.clear();
}

void Session::set_prompt(const std::string& text, bool chat, bool think) {
    try {
        ensure_engine();
        chat_ = chat;
        think_ = chat && think && has_thinking(m_.arch);
        const std::string full = chat ? chat_wrap(m_.arch, text, think_) : text;
        tokens_ = tok_->encode(full, true, chat);
        prompt_len_ = tokens_.size();
        pos_ = static_cast<int>(tokens_.size()) - 1;
        gen_base_.clear();
        gen_patched_.clear();
        gen_base_ids_.clear();
        gen_patched_ids_.clear();
        list_kind_ = ListKind::Blame;
        list_sel_ = 0;
        focus_ = Pane::List;
        rerun();
        if (think_) {
            const std::string close = thinking_close(m_.arch);
            status_ = "ran " + std::to_string(tokens_.size()) + " tokens, thinking on — :gen 4096 until \"" + close + "\", :accept, then :pos \"" + close +
                      "\"+1 for the answer";
        } else if (chat && think) {
            status_ = "ran " + std::to_string(tokens_.size()) + " tokens — this model has no thinking mode; ran as :chat";
        } else {
            status_ = "ran " + std::to_string(tokens_.size()) + " tokens — h/l move along them, j/k pick a LIST row, :gen 16 to continue";
        }
    } catch (const std::exception& e) {
        status_ = std::string("prompt: ") + e.what();
    }
}

void Session::rerun() {
    if (tokens_.empty()) return;
    iv_.prompt_len = static_cast<int>(prompt_len_);   // prompt-scoped edits end where the prompt does
    try {
        ensure_engine();
        pos_ = std::clamp(pos_, 0, static_cast<int>(tokens_.size()) - 1);
        const auto [lo, hi] = capture_window();
        RunOptions ro;
        ro.capture_all_layers = true;
        ro.capture_resid = true;
        ro.capture_layer_resid = true;
        ro.all_logits = true;
        ro.capture_heads = true;
        ro.cap_lo = lo;
        ro.cap_hi = hi;
        ro.watch = watch_;
        res_ = fw_->run(tokens_, iv_, ro);
        attn_.reset();
        if (!iv_.empty()) {
            RunOptions bo;
            bo.all_logits = true;
            bo.cap_lo = lo;
            bo.cap_hi = hi;
            base_ = fw_->run(tokens_, {}, bo);
        } else {
            base_.reset();
        }
        pos_ = std::clamp(pos_, 0, res_->n_tokens - 1);
        recompute_blame();
    } catch (const std::exception& e) {
        status_ = std::string("run: ") + e.what();
    }
}

void Session::recompute_blame() {
    if (!res_) return;
    pos_ = std::clamp(pos_, 0, res_->n_tokens - 1);
    // Unpinned: explain whatever the model (as patched) actually predicts here.
    if (!target_pinned_) target_ = top_probs(res_->logits_at(pos_), res_->n_vocab, 1)[0].first;
    if (dots_target_ != target_) {
        u_ = unembed_dir(m_, target_);
        dots_ = down_dots(m_, u_);
        odots_ = out_dots(m_, u_);
        dots_target_ = target_;
    }
    blame_ = compute_blame(m_, *res_, iv_, dots_, u_, target_, pos_, &odots_);

    // Per-layer totals for GRAPH.
    layer_attn_.assign(static_cast<size_t>(m_.n_layer), 0.0f);
    layer_ffn_.assign(static_cast<size_t>(m_.n_layer), 0.0f);
    for (const auto& a : blame_->attn) layer_attn_[static_cast<size_t>(a.layer)] = a.contrib;
    for (const auto& n : blame_->neurons) layer_ffn_[static_cast<size_t>(n.layer)] += n.contrib;

    // Target's lens logit after each layer (cheap: one dot per layer).
    target_by_layer_.assign(static_cast<size_t>(m_.n_layer), 0.0f);
    const size_t E = static_cast<size_t>(m_.n_embd);
    for (const auto& [l, xs] : res_->layer_resid) {
        const float* x = res_->resid_after(l, pos_);
        if (!x) continue;
        double d = 0;
        for (size_t e = 0; e < E; ++e) d += static_cast<double>(x[e]) * u_[e];
        target_by_layer_[static_cast<size_t>(l)] = static_cast<float>(d * rms_inv(x, E, m_.rms_eps));
    }
    state_lens_.pos = -1;   // invalidate
}

void Session::update_state_lens() {
    if (!res_ || res_->layer_resid.empty()) return;
    if (state_lens_.pos == pos_ && state_lens_.layer == layer_) return;
    const size_t E = static_cast<size_t>(m_.n_embd);
    const float* x = res_->resid_after(layer_, pos_);
    if (!x) return;
    const double ri = rms_inv(x, E, m_.rms_eps);
    std::vector<float> v(E);
    for (size_t e = 0; e < E; ++e) v[e] = static_cast<float>(x[e] * ri);
    const std::vector<float> lg = unembed_all(m_, v);
    const int nv = static_cast<int>(lg.size());
    state_lens_.top = top_probs(lg.data(), nv, 5);
    state_lens_.target_prob = target_ >= 0 ? prob_of(lg.data(), nv, target_) : 0;
    int above = 0;
    if (target_ >= 0)
        for (int i = 0; i < nv; ++i) above += lg[static_cast<size_t>(i)] > lg[static_cast<size_t>(target_)];
    state_lens_.target_rank = above + 1;
    state_lens_.pos = pos_;
    state_lens_.layer = layer_;
}

void Session::generate(int n, const std::string& stop) {
    if (tokens_.empty()) {
        status_ = "set a prompt first: :prompt TEXT";
        return;
    }
    // Greedy continuation, without and with the current patches: the prompt
    // decoded into a cache, then one cached decode per token. Prompt-scoped edits
    // act on tokens_[0, prompt_len_) only, never on what is generated here.
    iv_.prompt_len = static_cast<int>(prompt_len_);
    int n_unstopped = 0;   // runs (base / patched) that hit n before `stop`
    auto cont = [&](const Interventions& iv, std::vector<int32_t>& ids) {
        std::string out;
        ids.clear();
        auto cache = fw_->new_cache(static_cast<int>(tokens_.size()) + n);
        ForwardResult r = fw_->decode(*cache, tokens_, iv);
        bool stopped = false;
        for (int i = 0; i < n; ++i) {
            const int next = top_probs(r.logits.data(), r.n_vocab, 1)[0].first;
            if (tok_->is_eog(next)) break;
            ids.push_back(next);
            const size_t before = out.size();
            out += tok_->piece(next);
            // Search only the tail the new piece could have completed (S may straddle pieces).
            if (!stop.empty() && out.find(stop, before >= stop.size() ? before - stop.size() + 1 : 0) != std::string::npos) {
                stopped = true;
                break;
            }
            if (i + 1 < n) r = fw_->decode(*cache, {next}, iv);
        }
        if (!stop.empty() && !stopped) n_unstopped++;
        return out;
    };
    try {
        ensure_engine();
        gen_patched_ = cont(iv_, gen_patched_ids_);
        if (iv_.empty()) {
            gen_base_ = gen_patched_;
            gen_base_ids_ = gen_patched_ids_;
        } else {
            gen_base_ = cont({}, gen_base_ids_);
        }
        status_ = "generated " + std::to_string(gen_patched_ids_.size()) + (iv_.empty() ? "" : "/" + std::to_string(gen_base_ids_.size())) +
                  " tokens (see STATE)" +
                  (stop.empty() ? "" : n_unstopped ? " — " + quote_printable(stop) + " not reached within " + std::to_string(n) : " — stopped at " + quote_printable(stop)) +
                  " — :accept appends the answer to TOKENS";
    } catch (const std::exception& e) {
        status_ = std::string("gen: ") + e.what();
    }
}

void Session::toggle_patch(int layer, int neuron, float scale) {
    if (busy()) {
        status_ = "busy: dose sweep running — :cancel to stop";
        return;
    }
    if (!m_.has_ffn(layer) || neuron < 0 || neuron >= m_.n_ff) {
        status_ = "no such neuron " + ln(layer, neuron);
        return;
    }
    Edit e;
    e.layer = layer;
    e.neuron = neuron;
    e.op = EditOp::Scale;
    e.k = scale;
    toggle_edit(e);
}

void Session::toggle_edit(const Edit& e) {
    if (busy()) {
        status_ = "busy: dose sweep running — :cancel to stop";
        return;
    }
    if (const std::string err = validate_edit(e, m_); !err.empty()) {
        status_ = edit_text(e) + ": " + err;
        return;
    }
    auto same = [&](const Edit& x) {
        return x.layer == e.layer && x.neuron == e.neuron && x.op == e.op && x.k == e.k && x.value == e.value && x.dose == e.dose && x.pole == e.pole &&
               x.scope == e.scope;
    };
    auto it = std::find_if(patches_.begin(), patches_.end(), same);
    if (it != patches_.end()) {
        patches_.erase(it);
        status_ = "removed " + edit_text(e);
    } else {
        std::erase_if(patches_, [&](const Edit& x) { return same_target(x, e); });
        patches_.push_back(e);
        status_ = edit_text(e) + "  [" + portability_name(portability(e)) + "]" +
                  (res_ ? " — STATE shows base vs patched" : " — takes effect when you run a prompt");
    }
    rebuild_iv();
    rerun();
    if (e.op == EditOp::Steer && units_ && std::find_if(patches_.begin(), patches_.end(), same) != patches_.end())
        status_ += "  (a " + fmt_f(units_->value(e.layer, e.neuron, e.dose, e.pole), 2) + ")";
}

void Session::toggle_watch(int layer, int neuron) {
    if (!m_.has_ffn(layer) || neuron < 0 || neuron >= m_.n_ff) return;
    auto it = std::find(watch_.begin(), watch_.end(), std::make_pair(layer, neuron));
    if (it != watch_.end()) {
        watch_.erase(it);
        status_ = "unwatched " + ln(layer, neuron);
    } else {
        watch_.emplace_back(layer, neuron);
        status_ = "watching " + ln(layer, neuron) + " — its activation is under every token in TOKENS";
        // Watches are captured at every position by the run itself (RunOptions::watch).
        if (res_ && !res_->watched.count({layer, neuron}) && !busy()) {
            const std::string st = status_;
            rerun();
            status_ = st;
        }
    }
}

// ===========================================================================
// Commands
// ===========================================================================

bool Session::exec(const std::string& line) {
    std::istringstream is(line);
    std::string cmd;
    is >> cmd;
    std::string rest;
    std::getline(is, rest);
    rest.erase(0, rest.find_first_not_of(' '));
    // Strip one pair of quotes, then undo the escapes quote_printable writes (\n \t
    // \r \" \\), so a token shown as "</think>\n\n" can be typed back as-is: a ':'
    // line is one line, so a newline can't be typed any other way.
    auto unquote = [](std::string t) {
        if (t.size() >= 2 && t.front() == '"' && t.back() == '"') t = t.substr(1, t.size() - 2);
        std::string o;
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] != '\\' || i + 1 == t.size()) { o += t[i]; continue; }
            switch (t[++i]) {
                case 'n': o += '\n'; break;
                case 't': o += '\t'; break;
                case 'r': o += '\r'; break;
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                default: o += '\\'; o += t[i];
            }
        }
        return o;
    };
    auto parse_ln = [&](int& l, int& n) {
        l = layer_;
        n = neuron_;
        return rest.empty() || std::sscanf(rest.c_str(), "%d.%d", &l, &n) == 2;
    };

    if (cmd.empty()) return true;
    if (cmd == "q" || cmd == "quit") return false;
    if (overlay_ == Overlay::Help) overlay_ = Overlay::None;

    // While a sweep is running, don't change what it is measuring.
    static const char* model_cmds[] = {"prompt", "chat", "think", "gen", "accept", "append", "ablate", "invert", "scale", "clear", "target", "pos", "dose", "trace", "steer", "add", "set", "load"};
    if (busy() && std::find(std::begin(model_cmds), std::end(model_cmds), cmd) != std::end(model_cmds)) {
        status_ = trace_running_ && trace_ ? "busy: trace running (" + std::to_string(trace_->done) + "/" + std::to_string(trace_->total) + ") — :cancel to stop"
                  : dose_                  ? "busy: dose sweep running (" + std::to_string(dose_->done) + "/" + std::to_string(dose_->total) + ") — :cancel to stop"
                                           : "busy — :cancel to stop";
        return true;
    }

    int l = 0, n = 0;
    if (cmd == "prompt" || cmd == "chat" || cmd == "think") {
        if (rest.empty()) status_ = "usage: " + cmd + " TEXT";
        else set_prompt(rest, cmd != "prompt", cmd == "think");
    } else if (cmd == "gen") {
        // gen [N] [until "S"] — N first if present; S may be quoted (keeps spaces).
        int k = 16;
        std::string stop;
        std::string args = rest;
        if (const size_t u = args.find("until"); u != std::string::npos) {
            std::string s = args.substr(u + 5);
            s.erase(0, s.find_first_not_of(' '));
            while (!s.empty() && s.back() == ' ') s.pop_back();
            stop = unquote(s);
            args = args.substr(0, u);
        }
        try { if (args.find_first_not_of(' ') != std::string::npos) k = std::stoi(args); } catch (...) {}
        if (rest.find("until") != std::string::npos && stop.empty()) {
            status_ = "usage: gen [N] until \"S\"";
            return true;
        }
        generate(std::clamp(k, 1, 8192), stop);
    } else if (cmd == "accept") {
        const auto& ids = rest == "patched" ? gen_patched_ids_ : gen_base_ids_;
        if (ids.empty()) {
            status_ = "nothing to accept — run :gen N first";
        } else {
            tokens_.insert(tokens_.end(), ids.begin(), ids.end());
            status_ = "appended " + std::to_string(ids.size()) + " tokens (dimmed in TOKENS) — h to move the cursor onto them";
            gen_base_.clear();
            gen_patched_.clear();
            gen_base_ids_.clear();
            gen_patched_ids_.clear();
            pos_ = static_cast<int>(tokens_.size()) - 1;
            rerun();
        }
    } else if (cmd == "append") {
        // Prefill: put words in the model's mouth (e.g. force a trace's opening, or
        // close </think> early), then read what it does next. Same TOKENS bookkeeping
        // as :accept — appended tokens are dimmed and count as model text.
        if (tokens_.empty()) {
            status_ = "set a prompt first: :chat / :think TEXT";
        } else if (rest.empty()) {
            status_ = "usage: append TEXT   (quotes optional; \\n \\t \\\" escapes)";
        } else {
            const auto ids = tok_->encode(unquote(rest), false, true);
            tokens_.insert(tokens_.end(), ids.begin(), ids.end());
            gen_base_.clear();
            gen_patched_.clear();
            gen_base_ids_.clear();
            gen_patched_ids_.clear();
            pos_ = static_cast<int>(tokens_.size()) - 1;
            rerun();
            status_ = "appended " + std::to_string(ids.size()) + " tokens — the cursor's prediction is what comes next";
        }
    } else if (cmd == "ablate" || cmd == "invert") {
        if (!parse_ln(l, n)) status_ = "usage: " + cmd + " L.N";
        else toggle_patch(l, n, cmd == "ablate" ? 0.0f : -1.0f);
    } else if (cmd == "scale") {
        std::istringstream as(rest);
        std::string a;
        float k = 0;
        as >> a;
        if (std::sscanf(a.c_str(), "%d.%d", &l, &n) != 2 || !(as >> k)) status_ = "usage: scale L.N K   (0 ablates, -1 inverts)";
        else toggle_patch(l, n, k);
    } else if (cmd == "steer" || cmd == "add" || cmd == "set") {
        // steer L.N± DOSE [prompt] | add L.N V [prompt] | set L.N V
        std::istringstream as(rest);
        std::string ref, scope;
        float v = 0;
        Edit e;
        char pole = 0;
        const bool ok = (as >> ref) && parse_neuron_ref(ref, e.layer, e.neuron, pole) && (as >> v);
        as >> scope;
        if (!ok || (!scope.empty() && scope != "prompt" && scope != "all") || (cmd == "set" && scope == "prompt")) {
            status_ = cmd == "steer" ? "usage: steer L.N+|L.N- DOSE [prompt]   (DOSE in hidden-state sizes; 0.5 is a good start)"
                    : cmd == "add"   ? "usage: add L.N V [prompt]"
                                     : "usage: set L.N V";
        } else {
            e.op = cmd == "steer" ? EditOp::Steer : cmd == "add" ? EditOp::Add : EditOp::Set;
            if (cmd == "steer") {
                e.pole = pole;
                e.dose = v;
            } else {
                e.value = v;
            }
            e.scope = scope == "prompt" ? EditScope::Prompt : EditScope::All;
            try {
                toggle_edit(e);
            } catch (const std::exception& ex) {
                status_ = cmd + ": " + ex.what();
            }
        }
    } else if (cmd == "save") {
        if (rest.empty()) {
            status_ = "usage: save NAME   (→ edits/<model>/NAME.json)";
        } else {
            try {
                EditSet es;
                es.name = std::filesystem::path(rest).stem().string();
                es.notes = edits_notes_;
                es.edits = patches_;
                const std::string path = edit_set_path(model_key(m_), rest);
                save_edit_set(path, es, m_);
                edits_name_ = es.name;
                Portability worst = Portability::Gguf;
                for (const auto& x : patches_) worst = std::max(worst, portability(x));
                status_ = "saved " + std::to_string(patches_.size()) + " edit" + (patches_.size() == 1 ? "" : "s") + " → " + path +
                          (patches_.empty() ? "" : "  [" + std::string(portability_name(worst)) + "]");
            } catch (const std::exception& ex) {
                status_ = std::string("save: ") + ex.what();
            }
        }
    } else if (cmd == "load") {
        if (rest.empty()) {
            status_ = "usage: load NAME|FILE   (:edits lists them)";
        } else if (busy()) {
            status_ = "busy: dose sweep running — :cancel to stop";
        } else {
            try {
                const EditSet es = load_edit_set(edit_set_path(model_key(m_), rest), m_);
                patches_ = es.edits;
                edits_name_ = es.name;
                edits_notes_ = es.notes;
                rebuild_iv();
                rerun();
                status_ = "loaded " + es.name + ": " + std::to_string(patches_.size()) + " edit" + (patches_.size() == 1 ? "" : "s") +
                          (es.notes.empty() ? "" : " — " + es.notes.substr(0, es.notes.find('\n')));
            } catch (const std::exception& ex) {
                status_ = std::string("load: ") + ex.what();
            }
        }
    } else if (cmd == "export") {
        // Same as `neuronaut export MODEL --edits …` (src/export.h).
        try {
            if (patches_.empty()) throw std::invalid_argument("no patches to export");
            const ExportPlan plan = plan_export(patches_, m_, false);
            const std::string& src = file_->path();
            std::string stem = rest.empty() ? src.substr(0, src.size() - (src.size() > 5 && src.compare(src.size() - 5, 5, ".gguf") == 0 ? 5 : 0)) + "-edited"
                                            : unquote(rest);
            if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".gguf") == 0) stem.resize(stem.size() - 5);
            std::string prov;
            for (const auto& e : patches_) prov += (prov.empty() ? "" : "; ") + edit_text(e);
            ExportReport rep;
            std::string wrote;
            if (plan.needs_gguf()) {
                export_gguf(*file_, m_, plan, stem + ".gguf", prov, rep);
                wrote += stem + ".gguf";
            }
            if (plan.needs_cvec()) {
                export_control_vector(m_, plan, needs_units(plan.adds) ? &steer_units() : nullptr, stem + ".cvec.gguf", rep);
                wrote += (wrote.empty() ? "" : " + ") + stem + ".cvec.gguf";
            }
            status_ = "exported → " + wrote + (rep.rows_folded ? "  (" + std::to_string(rep.rows_exact) + "/" + std::to_string(rep.rows_folded) + " folds exact)" : "");
        } catch (const std::exception& ex) {
            status_ = std::string("export: ") + ex.what();
        }
    } else if (cmd == "edits") {
        const auto names = list_edit_sets(model_key(m_));
        std::string all;
        for (const auto& x : names) all += (all.empty() ? "" : "  ") + x;
        status_ = names.empty() ? "no saved edit sets for this model — :save NAME" : "edit sets: " + all + "   (:load NAME)";
    } else if (cmd == "clear") {
        if (!rest.empty() && std::sscanf(rest.c_str(), "%d.%d", &l, &n) == 2) {
            std::erase_if(patches_, [&](const Edit& p) { return p.layer == l && p.neuron == n; });
        } else {
            patches_.clear();
        }
        rebuild_iv();
        rerun();
        status_ = patches_.empty() ? "no patches" : std::to_string(patches_.size()) + " patches";
    } else if (cmd == "watch" || cmd == "unwatch") {
        if (!parse_ln(l, n)) status_ = "usage: " + cmd + " L.N";
        else toggle_watch(l, n);
    } else if (cmd == "target") {
        const std::string t = unquote(rest);
        if (t.empty()) {
            target_pinned_ = false;
            status_ = "target follows the model's top prediction";
        } else {
            const auto ids = find_token_ids(*file_, t);
            if (ids.empty()) {
                status_ = "no vocab entry is exactly " + quote_printable(t) + " — most words have a leading space: target \" word\"";
                return true;
            }
            target_ = ids.front();
            target_pinned_ = true;
            status_ = "target pinned to " + quote_printable(t);
        }
        recompute_blame();
    } else if (cmd == "pos") {
        // pos N, or pos "tok"[+K|-K]: the next token whose piece is exactly tok.
        if (!rest.empty() && rest.front() == '"') {
            size_t close = 1;   // the closing quote: first '"' not escaped by a backslash
            while ((close = rest.find('"', close)) != std::string::npos && rest[close - 1] == '\\') ++close;
            int off = 0;
            bool bad = close == std::string::npos;
            if (!bad && close + 1 < rest.size()) {
                try { off = std::stoi(rest.substr(close + 1)); } catch (...) { bad = true; }
            }
            if (bad) {
                status_ = "usage: pos N | pos \"tok\"[+K]";
            } else {
                const std::string piece = unquote(rest.substr(0, close + 1));
                const int hit = seek_token(piece, off);
                if (hit < 0) {
                    status_ = "no token is exactly " + quote_printable(piece) + " in TOKENS";
                } else {
                    // seek_token clamps to the sequence; say so rather than imply we got there.
                    const bool clamped = pos_ != hit + off;
                    status_ = "cursor on " + std::to_string(pos_) +
                              (off ? " (" + quote_printable(piece) + (off > 0 ? "+" : "") + std::to_string(off) + ")" : "") +
                              (clamped ? " — clamped: nothing after it yet (:gen 1, :accept)" : "");
                }
            }
        } else {
            try {
                set_pos(std::stoi(rest));
            } catch (...) {
                status_ = "usage: pos N | pos \"tok\"[+K]";
            }
        }
    } else if (cmd == "dose") {
        if (!parse_ln(l, n)) status_ = "usage: dose L.N";
        else start_dose(l, n);
    } else if (cmd == "probe") {
        // probe [L9,10,…] POS | POS // NEG | NEG
        std::string body = rest;
        std::vector<int> layers;
        if (body.size() > 1 && body[0] == 'L' && std::isdigit(static_cast<unsigned char>(body[1]))) {
            const size_t sp = body.find(' ');
            std::stringstream ls(body.substr(1, sp == std::string::npos ? std::string::npos : sp - 1));
            for (std::string x; std::getline(ls, x, ',');) {
                try { layers.push_back(std::stoi(x)); } catch (...) {}
            }
            body = sp == std::string::npos ? "" : body.substr(sp + 1);
        }
        auto split = [&](const std::string& t) {
            std::vector<std::string> out;
            std::stringstream ss(t);
            for (std::string x; std::getline(ss, x, '|');) {
                x.erase(0, x.find_first_not_of(' '));
                while (!x.empty() && x.back() == ' ') x.pop_back();
                if (!x.empty()) out.push_back(unquote(x));
            }
            return out;
        };
        const size_t sl = body.find("//");
        if (sl == std::string::npos) {
            status_ = "usage: probe [L9,10] POS | POS // NEG | NEG";
        } else {
            const ProbeView v = probe(split(body.substr(0, sl)), split(body.substr(sl + 2)), layers, 4);
            if (!v.error.empty()) status_ = "probe: " + v.error;
        }
    } else if (cmd == "examples") {
        int el = layer_, en = neuron_;
        char pole = 0;
        if (!rest.empty() && !parse_neuron_ref(rest, el, en, pole)) {
            status_ = "usage: examples [L.N]";
        } else {
            select_neuron(el, en);
            list_kind_ = ListKind::Examples;
            list_sel_ = 0;
            focus_ = Pane::List;
            status_ = act_index() ? "top corpus examples of " + ln(layer_, neuron_) + " — they follow the cursor neuron (NEURON j/k)" : act_index_error_;
        }
    } else if (cmd == "index") {
        const ActIndex* ix = act_index();
        status_ = ix ? "acts index: " + std::to_string(ix->n_docs()) + " docs, " + std::to_string(ix->n_tokens()) + " tokens, k=" + std::to_string(ix->k()) +
                           " from " + ix->corpus + " (built " + ix->built + " from " + ix->gguf + (ix->device.empty() ? "" : " on " + ix->device) + ")"
                     : act_index_error_;
    } else if (cmd == "corrupt") {
        if (rest.empty()) {
            status_ = corrupt_text_.empty() ? "usage: corrupt TEXT   (same length in tokens as the prompt; then :trace)" : "corrupt prompt: " + quote_printable(corrupt_text_, 60);
        } else {
            corrupt_text_ = unquote(rest);
            status_ = "corrupt prompt set — :trace resid|attn|mlp patches the clean run into it, cell by cell";
        }
    } else if (cmd == "trace") {
        Site site = Site::Resid;
        if (!rest.empty() && !parse_site(rest, site)) status_ = "usage: trace [resid|attn|mlp]";
        else start_trace(site);
    } else if (cmd == "cancel") {
        cancel_ = true;
        status_ = "cancelling…";
    } else if (cmd == "s" || cmd == "seek") {
        if (!parse_seek(rest, layer_, l, n)) {
            status_ = "usage: s L.N";
        } else {
            select_neuron(l, n);
            focus_ = Pane::Neuron;
        }
    } else if (cmd == "find") {
        const std::string t = unquote(rest);
        const auto ids = t.empty() ? std::vector<int>{} : find_token_ids(*file_, t);
        if (t.empty()) {
            status_ = "usage: find \" word\"   (most words carry a leading space)";
        } else if (ids.empty()) {
            status_ = "no vocab entry is exactly " + quote_printable(t) + " — Tab completes from the vocabulary";
        } else {
            const WriterResult r = token_writers(*file_, m_, ids.front(), 20);
            find_text_ = t;
            find_hits_ = r.positive;
            find_npos_ = static_cast<int>(r.positive.size());
            find_hits_.insert(find_hits_.end(), r.negative.begin(), r.negative.end());
            list_kind_ = ListKind::Find;
            list_sel_ = 0;
            focus_ = Pane::List;
            list_move(0);
            status_ = "neurons that write " + quote_printable(t) + " most directly — j/k to browse them";
        }
    } else if (cmd == "label" || cmd == "unlabel") {
        // Target: a leading L.N[±] if the first word is one, else the cursor neuron.
        int tl = layer_, tn = neuron_;
        char pole = 0;
        bool explicit_pole = false;
        std::string text_part = rest;
        {
            std::istringstream words(rest);
            std::string first;
            int pl = 0, pn = 0;
            char pp = 0;
            if (words >> first && parse_neuron_ref(first, pl, pn, pp)) {
                tl = pl;
                tn = pn;
                pole = pp;
                explicit_pole = pp != 0;
                std::getline(words, text_part);
            }
        }
        if (!labels_) {
            status_ = "labels are off (see the earlier status: the labels file couldn't be read)";
        } else if (cmd == "unlabel") {
            try {
                const auto gone = labels_->remove(tl, tn, pole, !explicit_pole, "manual");
                const auto left = labels_->at(tl, tn);
                status_ = gone.empty() ? "no manual label on " + ln(tl, tn) + (left.empty() ? "" : " — imported ones stay: neuronaut labels MODEL rm L.N --source S")
                                       : "removed " + std::to_string(gone.size()) + " label" + (gone.size() > 1 ? "s" : "") + " from " + ln(tl, tn) +
                                             (left.empty() ? "" : " (" + std::to_string(left.size()) + " from other sources remain)");
            } catch (const std::exception& e) {
                status_ = e.what();
            }
        } else {
            // TEXT [#tag]… [// notes]
            Label x;
            x.layer = tl;
            x.neuron = tn;
            x.pole = pole;
            const size_t sl = text_part.find("//");
            if (sl != std::string::npos) {
                x.notes = text_part.substr(sl + 2);
                x.notes.erase(0, x.notes.find_first_not_of(' '));
                text_part.resize(sl);
            }
            std::istringstream words(text_part);
            for (std::string w; words >> w;) {
                if (w.size() > 1 && w[0] == '#') x.tags.push_back(w.substr(1));
                else x.label += (x.label.empty() ? "" : " ") + w;
            }
            x.source = "manual";
            if (const char* u = std::getenv("USER")) x.author = u;
            if (x.label.empty()) {
                status_ = "usage: label [L.N[+|-]] TEXT [#tag]… [// notes]";
            } else {
                try {
                    const bool added = labels_->upsert(x);
                    status_ = std::string(added ? "labelled " : "relabelled ") + neuron_ref(tl, tn, pole) + " \"" + x.label + "\" → " + labels_->path();
                } catch (const std::exception& e) {
                    status_ = e.what();
                }
            }
        }
    } else if (cmd == "labels") {
        if (!labels_) {
            status_ = "labels are off (the labels file couldn't be read)";
        } else {
            labels_query_ = rest;
            list_kind_ = ListKind::Labels;
            list_sel_ = 0;
            focus_ = Pane::List;
            list_move(0);
            const size_t nl = labels_->search(labels_query_).size();
            status_ = std::to_string(nl) + " labelled neuron record" + (nl == 1 ? "" : "s") + (rest.empty() ? "" : " match " + quote_printable(rest, 30)) +
                      " — j/k to browse, Enter → NEURON";
        }
    } else if (cmd == "list") {
        if (rest == "blame") list_kind_ = ListKind::Blame;
        else if (rest == "fires") list_kind_ = ListKind::Fires;
        else if (rest == "find") list_kind_ = ListKind::Find;
        else if (rest == "controls") list_kind_ = ListKind::Controls;
        else if (rest == "labels") return exec("labels " + labels_query_);
        else if (rest == "probe") list_kind_ = ListKind::Probe;
        else if (rest == "examples") return exec("examples");
        else status_ = "usage: list blame|fires|find|controls|labels|probe|examples";
        list_sel_ = 0;
        focus_ = Pane::List;
    } else if (cmd == "slice") {
        if (rest == "gate") slice_ = Slice::Gate;
        else if (rest == "up") slice_ = Slice::Up;
        else if (rest == "down") slice_ = Slice::Down;
        else status_ = "usage: slice gate|up|down";
    } else if (cmd == "dim") {
        try {
            dim_ = std::clamp(std::stoi(rest), 0, m_.n_embd - 1);
        } catch (...) {
            status_ = "usage: dim N";
        }
    } else if (cmd == "model") {
        overlay_ = Overlay::Model;
    } else if (cmd == "attn") {
        // attn [L] [H]: the attention overlay for layer L (default: the cursor's).
        std::istringstream as(rest);
        int al = layer_, ah = attn_head_;
        if (as >> al) {
            if (!(as >> ah)) ah = 0;
        }
        if (al < 0 || al >= m_.n_layer) {
            status_ = "usage: attn [L] [H]";
        } else if (!res_) {
            status_ = "no run yet — :prompt TEXT first";
        } else {
            select_layer(al);
            attn_head_ = std::clamp(ah, 0, std::max(0, n_heads(m_, layer_) - 1));
            overlay_ = Overlay::Attn;
        }
    } else if (cmd == "plot") {
        if (dose_) overlay_ = Overlay::Plot;
        else status_ = "no dose sweep yet — D on a neuron";
    } else if (cmd == "tplot") {
        if (trace_) overlay_ = Overlay::Trace;
        else status_ = "no trace yet — :corrupt TEXT, then :trace";
    } else if (cmd == "filter") {
        tensor_filter_ = rest;
        model_sel_ = 0;
        overlay_ = Overlay::Model;
    } else if (cmd == "help" || cmd == "?") {
        overlay_ = Overlay::Help;
    } else {
        status_ = "unknown command '" + cmd + "' — : then Tab lists them, ? for help";
    }
    return true;
}

// ===========================================================================
// Keys
// ===========================================================================

bool Session::key(const std::string& k) {
    // Overlays take keys first and close with Esc.
    if (overlay_ != Overlay::None) {
        if (k == "escape" || k == "q" || (k == "?" && overlay_ == Overlay::Help) || (k == "M" && overlay_ == Overlay::Model) ||
            (k == "P" && overlay_ == Overlay::Plot) || (k == "A" && overlay_ == Overlay::Attn) || (k == "T" && overlay_ == Overlay::Trace)) {
            overlay_ = Overlay::None;
            return true;
        }
        if (overlay_ == Overlay::Attn) {
            // j/k pick a head; the cursor keeps working underneath (time, layer).
            const int nh = n_heads(m_, layer_);
            if (k == "j" || k == "down") attn_head_ = std::min(attn_head_ + 1, nh - 1);
            else if (k == "k" || k == "up") attn_head_ = std::max(attn_head_ - 1, 0);
            else if ((k == "h" || k == "left" || k == "l" || k == "right") && res_ && !busy()) set_pos(pos_ + ((k == "h" || k == "left") ? -1 : 1));
            else if (k == "pgup") select_layer(layer_ - 1);
            else if (k == "pgdn") select_layer(layer_ + 1);
            attn_head_ = std::clamp(attn_head_, 0, std::max(0, n_heads(m_, layer_) - 1));
            return true;
        }
        if (overlay_ == Overlay::Model) {
            const int nt = static_cast<int>(file_->tensors().size());
            if (k == "j" || k == "down") model_sel_ = std::min(model_sel_ + 1, nt - 1);
            else if (k == "k" || k == "up") model_sel_ = std::max(model_sel_ - 1, 0);
            else if (k == "J" || k == "pgdn") model_sel_ = std::min(model_sel_ + 20, nt - 1);
            else if (k == "K" || k == "pgup") model_sel_ = std::max(model_sel_ - 20, 0);
        }
        return true;
    }

    // ---- global keys: same meaning in every pane ----------------------------
    if (k == "q") return false;
    if (k == "?") { overlay_ = Overlay::Help; return true; }
    if (k == "M") { overlay_ = Overlay::Model; return true; }
    if (k == "P") { exec("plot"); return true; }
    if (k == "A") { exec("attn"); return true; }
    if (k == "T") { exec("tplot"); return true; }
    if (k == "tab" || k == "shift_tab") {
        const int d = k == "tab" ? 1 : kPanes - 1;
        focus_ = static_cast<Pane>((static_cast<int>(focus_) + d) % kPanes);
        return true;
    }
    if (k == "h" || k == "left" || k == "l" || k == "right") {
        if (!res_) { status_ = "no prompt yet — :prompt TEXT or :chat MESSAGE"; return true; }
        if (busy()) { status_ = "busy: dose sweep running — :cancel to stop"; return true; }
        set_pos(pos_ + ((k == "h" || k == "left") ? -1 : 1));
        return true;
    }
    if (k == "pgup") { select_layer(layer_ - 1); return true; }
    if (k == "pgdn") { select_layer(layer_ + 1); return true; }
    if (k == "x") { toggle_patch(layer_, neuron_, 0.0f); return true; }
    if (k == "i") { toggle_patch(layer_, neuron_, -1.0f); return true; }
    if (k == "w") { toggle_watch(layer_, neuron_); return true; }
    if (k == "D") { start_dose(layer_, neuron_); return true; }
    if (k == "b") { exec("list blame"); return true; }
    if (k == "f") { exec("list fires"); return true; }
    if (k == "c") { exec("list controls"); return true; }
    if (k == "L") { exec("list labels"); return true; }
    if (k == "e") { exec("examples"); return true; }

    // ---- keys of the focused pane --------------------------------------------
    switch (focus_) {
        case Pane::List:
            if (k == "j" || k == "down") list_move(1);
            else if (k == "k" || k == "up") list_move(-1);
            else if (k == "J") list_move(10);
            else if (k == "K") list_move(-10);
            else if (k == "enter") focus_ = Pane::Neuron;
            break;
        case Pane::Graph:
            if (k == "j" || k == "down") select_layer(layer_ + 1);
            else if (k == "k" || k == "up") select_layer(layer_ - 1);
            else if (k == "enter") { exec("list fires"); list_move(0); }
            break;
        case Pane::Neuron:
            if (k == "j" || k == "down") select_neuron(layer_, neuron_ + 1);
            else if (k == "k" || k == "up") select_neuron(layer_, neuron_ - 1);
            else if (k == "J") select_neuron(layer_, neuron_ + 64);
            else if (k == "K") select_neuron(layer_, neuron_ - 64);
            else if (k == "g") slice_ = Slice::Gate;
            else if (k == "u") slice_ = Slice::Up;
            else if (k == "d") slice_ = Slice::Down;
            break;
        case Pane::Bytes:
            if (k == "j" || k == "down" || k == "]") dim_ = std::min(dim_ + 1, m_.n_embd - 1);
            else if (k == "k" || k == "up" || k == "[") dim_ = std::max(dim_ - 1, 0);
            else if (k == "J" || k == "}") dim_ = std::min(dim_ + 32, m_.n_embd - 1);
            else if (k == "K" || k == "{") dim_ = std::max(dim_ - 32, 0);
            else if (k == "g") slice_ = Slice::Gate;
            else if (k == "u") slice_ = Slice::Up;
            else if (k == "d") slice_ = Slice::Down;
            break;
        case Pane::Patches: {
            const int np = static_cast<int>(patches_.size()), nw = static_cast<int>(watch_.size());
            if (np + nw == 0) break;
            auto follow = [&] {
                patch_sel_ = std::clamp(patch_sel_, 0, np + nw - 1);
                if (patch_sel_ < np) select_neuron(patches_[static_cast<size_t>(patch_sel_)].layer, patches_[static_cast<size_t>(patch_sel_)].neuron);
                else select_neuron(watch_[static_cast<size_t>(patch_sel_ - np)].first, watch_[static_cast<size_t>(patch_sel_ - np)].second);
            };
            if (k == "j" || k == "down") { ++patch_sel_; follow(); }
            else if (k == "k" || k == "up") { --patch_sel_; follow(); }
            else if (k == "delete" || k == "backspace") {
                // Remove the selected entry outright (x would re-patch it as an ablation).
                patch_sel_ = std::clamp(patch_sel_, 0, np + nw - 1);
                if (patch_sel_ < np) {
                    const Edit p = patches_[static_cast<size_t>(patch_sel_)];
                    exec("clear " + std::to_string(p.layer) + "." + std::to_string(p.neuron));
                } else {
                    watch_.erase(watch_.begin() + (patch_sel_ - np));
                }
            }
            break;
        }
    }
    return true;
}

// ===========================================================================
// LIST
// ===========================================================================

std::vector<Session::ListRow> Session::list_rows() {
    std::vector<ListRow> rows;
    auto section = [&](const std::string& s) {
        ListRow r;
        r.section = true;
        r.label = s;
        rows.push_back(r);
    };
    auto tags = [&](int l, int n) {
        std::string t;
        for (const auto& p : patches_)
            if (p.layer == l && p.neuron == n) t += "  " + edit_tag(p);
        if (std::find(watch_.begin(), watch_.end(), std::make_pair(l, n)) != watch_.end()) t += "  ◉";
        return t;
    };
    switch (list_kind_) {
        case ListKind::Blame: {
            if (!blame_) break;
            auto add = [&](const std::vector<BlameItem>& items) {
                for (const auto& it : items) {
                    ListRow r;
                    r.kind = it.kind;
                    r.layer = it.layer;
                    r.neuron = it.neuron;
                    r.value = it.contrib;
                    if (it.kind == BlameItem::Neuron) r.detail = "a=" + fmt_f(it.act, 2) + " × " + fmt_f(it.dot, 3) + tags(it.layer, it.neuron);
                    else if (it.kind == BlameItem::Attn) r.detail = "all heads";
                    else if (it.kind == BlameItem::Head)
                        r.detail = m_.layers[static_cast<size_t>(it.layer)].kind == LayerKind::linear_attention ? "linear head · A" : "attn head · A";
                    rows.push_back(r);
                }
            };
            section("pushes toward " + quote_printable(token_text(blame_->target), 14));
            add(blame_->top(12, true));
            section("pushes against");
            add(blame_->top(8, false));
            break;
        }
        case ListKind::Fires: {
            const float* a = res_ ? res_->act(layer_, pos_) : nullptr;
            if (!a) break;
            const size_t F = static_cast<size_t>(m_.n_ff);
            std::vector<int> idx(F);
            for (size_t i = 0; i < F; ++i) idx[i] = static_cast<int>(i);
            std::partial_sort(idx.begin(), idx.begin() + 24, idx.end(), [&](int x, int y) { return std::fabs(a[x]) > std::fabs(a[y]); });
            section("strongest |a| in L" + std::to_string(layer_) + " at this token");
            for (int i = 0; i < 24; ++i) {
                ListRow r;
                r.layer = layer_;
                r.neuron = idx[static_cast<size_t>(i)];
                r.value = a[r.neuron];
                if (blame_) {
                    const float c = blame_->neurons[static_cast<size_t>(layer_) * F + static_cast<size_t>(r.neuron)].contrib;
                    r.detail = "blame " + fmt_f(c, 3);
                }
                r.detail += tags(r.layer, r.neuron);
                rows.push_back(r);
            }
            break;
        }
        case ListKind::Find: {
            if (find_hits_.empty()) break;
            for (int i = 0; i < static_cast<int>(find_hits_.size()); ++i) {
                if (i == 0) section("a > 0 writes " + quote_printable(find_text_, 14) + "  (cos)");
                if (i == find_npos_) section("a < 0 writes it");
                const Writer& w = find_hits_[static_cast<size_t>(i)];
                ListRow r;
                r.layer = w.layer;
                r.neuron = w.neuron;
                r.value = w.cos;
                r.detail = "dot " + fmt_f(w.dot, 3) + tags(w.layer, w.neuron);
                rows.push_back(r);
            }
            break;
        }
        case ListKind::Controls: {
            if (!dose_) break;
            size_t i0 = 0, i1 = 0;
            for (size_t i = 0; i < dose_->doses.size(); ++i) {
                if (dose_->doses[i] == 0.0f) i0 = i;
                if (dose_->doses[i] == 1.0f) i1 = i;
            }
            section("controls for " + ln(dose_->layer, dose_->neuron) + " — Δ P(target) when ablated");
            std::vector<std::pair<float, size_t>> ce;
            for (size_t c = 0; c < dose_->ctrl.size(); ++c) {
                const float d = (std::isnan(dose_->ctrl[c][i0]) || std::isnan(dose_->ctrl[c][i1])) ? 0.0f : dose_->ctrl[c][i0] - dose_->ctrl[c][i1];
                ce.emplace_back(d, c);
            }
            std::sort(ce.begin(), ce.end(), [](auto& x, auto& y) { return std::fabs(x.first) > std::fabs(y.first); });
            for (auto [d, c] : ce) {
                ListRow r;
                r.layer = dose_->layer;
                r.neuron = dose_->controls[c].first;
                r.value = 100.0f * d;
                r.detail = "pts  push " + fmt_f(dose_->controls[c].second, 3).substr(1) + tags(r.layer, r.neuron);
                rows.push_back(r);
            }
            break;
        }
        case ListKind::Labels: {
            if (!labels_) break;
            // One row per labelled neuron (all its records in the note). With a run,
            // the bar is its activation at the cursor token and rows sort by |a|: the
            // labelled neurons that are firing right here come first.
            const auto recs = labels_->search(labels_query_);
            std::vector<std::pair<int, int>> ns;
            for (const auto& x : recs)
                if (ns.empty() || ns.back() != std::make_pair(x.layer, x.neuron)) ns.emplace_back(x.layer, x.neuron);
            auto act = [&](int l, int n) -> float {
                const float* a = res_ ? res_->act(l, pos_) : nullptr;
                return a ? a[static_cast<size_t>(n)] : 0.0f;
            };
            if (res_) std::stable_sort(ns.begin(), ns.end(), [&](auto& x, auto& y) { return std::fabs(act(x.first, x.second)) > std::fabs(act(y.first, y.second)); });
            section(std::to_string(ns.size()) + " labelled" + (labels_query_.empty() ? "" : " matching " + quote_printable(labels_query_, 20)) +
                    (res_ ? " — a at pos " + std::to_string(pos_) : ""));
            for (size_t i = 0; i < ns.size() && i < 200; ++i) {
                ListRow r;
                r.layer = ns[i].first;
                r.neuron = ns[i].second;
                r.value = act(r.layer, r.neuron);
                r.detail = tags(r.layer, r.neuron);
                rows.push_back(r);
            }
            break;
        }
        case ListKind::Probe: {
            if (probe_hits_.empty()) break;
            int last = -1;
            for (const auto& h : probe_hits_) {
                if (h.layer != last) section("L" + std::to_string(h.layer) + " — margin (worst case, last token)");
                last = h.layer;
                ListRow r;
                r.layer = h.layer;
                r.neuron = h.neuron;
                r.value = h.margin;
                r.detail = std::string(h.pole == '+' ? "+ on POS" : "− on POS") + "  pos~" + fmt_f(h.pos_mean, 2) + " neg~" + fmt_f(h.neg_mean, 2) + tags(h.layer, h.neuron);
                rows.push_back(r);
            }
            break;
        }
        case ListKind::Examples: {
            const ActIndex* ix = act_index();
            if (!ix) break;
            for (char pole : {'+', '-'}) {
                section(ln(layer_, neuron_) + std::string(pole == '+' ? " fires hardest" : " most negative") + " — " + std::to_string(ix->n_docs()) + " docs of " + ix->corpus);
                int i = 0;
                for (const auto& e : ix->top(layer_, neuron_, pole)) {
                    if (i++ >= (pole == '+' ? 10 : 5)) break;
                    ListRow r;
                    r.layer = layer_;
                    r.neuron = neuron_;
                    r.value = e.act;
                    r.name = "doc " + std::to_string(e.doc);
                    // A few tokens around the peak, fitted to the pane (FTXUI would
                    // squeeze the name column for a longer one); MCP/CLI give the full context.
                    r.detail = fit_cells(quote_printable(example_context(e, 2, 2), 60), 30);
                    r.inert = true;
                    rows.push_back(r);
                }
            }
            break;
        }
    }
    // Every neuron row, in every list, carries its labels.
    for (auto& r : rows)
        if (!r.section && r.kind == BlameItem::Neuron) r.note = label_brief(r.layer, r.neuron);
    return rows;
}

void Session::list_move(int delta) {
    const auto rows = list_rows();
    std::vector<size_t> sel;
    for (size_t i = 0; i < rows.size(); ++i)
        if (!rows[i].section) sel.push_back(i);
    if (sel.empty()) return;
    list_sel_ = std::clamp(list_sel_ + delta, 0, static_cast<int>(sel.size()) - 1);
    // Live follow: the cursor moves to whatever is selected.
    const ListRow& r = rows[sel[static_cast<size_t>(list_sel_)]];
    if (r.inert) return;
    if (r.kind == BlameItem::Neuron) select_neuron(r.layer, r.neuron);
    else if (r.kind == BlameItem::Attn) select_layer(r.layer);
    else if (r.kind == BlameItem::Head) {
        select_layer(r.layer);
        attn_head_ = r.neuron;   // A opens the attention overlay on this head
    }
}

// ===========================================================================
// Rendering
// ===========================================================================

// A pane: heavy border + highlighted title when it has focus.
Element Session::pane(const std::string& title, Element body, std::optional<Pane> p) {
    const bool focused = p && *p == focus_ && overlay_ == Overlay::None;
    Element t = text(" " + title + " ");
    t = focused ? t | bold | color(Color::Black) | bgcolor(kFocus) : t | bold;
    Element w = window(t, std::move(body), focused ? HEAVY : ROUNDED);
    return focused ? w | color(kFocus) : w;
}

Element Session::render_header() {
    const TensorInfo* up = m_.has_ffn(layer_) ? m_.layers[static_cast<size_t>(layer_)].ffn_up : nullptr;
    Elements h{
        text(" neuronaut ") | bold,
        text("│ " + (m_.name.empty() ? file_->path() : m_.name) + " " + (up ? ggml_type_name(up->type) : "") + " "),
        text("│ cursor: "),
    };
    if (res_) h.push_back(text("pos " + std::to_string(pos_) + " " + quote_printable(token_text(tokens_[static_cast<size_t>(pos_)]), 12) + " · ") | bold);
    h.push_back(text("L" + std::to_string(layer_) + " · N" + std::to_string(neuron_) + " ") | bold);
    if (blame_) {
        h.push_back(text("│ target " + quote_printable(token_text(target_), 14) + " " + fmt_f(100.0 * blame_->prob, 1).substr(1) + "%" +
                         (target_pinned_ ? " (pinned) " : " ")));
    }
    h.push_back(filler());
    if (!patches_.empty()) h.push_back(text(" ⚑ " + std::to_string(patches_.size()) + " patch" + (patches_.size() > 1 ? "es " : " ")) | color(kOrange));
    if (dose_ && dose_->running) h.push_back(text(" ⟳ dose " + std::to_string(dose_->done) + "/" + std::to_string(dose_->total) + " ") | color(Color::Yellow));
    if (trace_running_ && trace_) h.push_back(text(" ⟳ trace " + std::to_string(trace_->done) + "/" + std::to_string(trace_->total) + " ") | color(Color::Yellow));
    h.push_back(text(" ? help "));
    return hbox(std::move(h)) | inverted;
}

// TOKENS: the time axis. One column per token, watched activations underneath.
Element Session::render_tokens() {
    if (!res_) {
        return pane("TOKENS", text("no prompt yet — :prompt The capital of France is   or   :chat What is the capital of France?") | dim, std::nullopt);
    }
    const ForwardResult& r = *res_;
    // A watch's activations, every position (RunOptions::watch); 0 until re-run.
    auto wa = [&](int l, int n, int i) {
        auto it = r.watched.find({l, n});
        return it == r.watched.end() ? 0.0f : it->second[static_cast<size_t>(i)];
    };
    float wmax = 1e-6f;
    for (auto [l, n] : watch_)
        for (int i = 0; i < r.n_tokens; ++i) wmax = std::max(wmax, std::fabs(wa(l, n, i)));

    Elements cols;
    if (!watch_.empty()) {
        Elements labels{text("pos") | dim};
        for (auto [l, n] : watch_) labels.push_back(text("◉ " + ln(l, n)) | dim);
        cols.push_back(hbox({vbox(std::move(labels)), text(" ")}));
    }
    for (int i = 0; i < r.n_tokens; ++i) {
        const std::string t = tok_label(token_text(tokens_[static_cast<size_t>(i)]), 14);
        const int w = std::max<int>(static_cast<int>(t.size()), 6);
        Element tt = text(t) | size(WIDTH, EQUAL, w);
        if (static_cast<size_t>(i) >= prompt_len_) tt = tt | dim | italic;
        if (i == pos_) tt = tt | inverted | bold | focus;
        Elements cell{tt};
        for (auto [l, n] : watch_) {
            const float a = wa(l, n, i);
            cell.push_back(text(fmt_f(a, 2)) | size(WIDTH, EQUAL, w) | color(Color::White) | bgcolor(diverging(a / wmax)));
        }
        cols.push_back(hbox({vbox(std::move(cell)), text(" ")}));
    }
    const std::string title = std::string("TOKENS") + (think_ ? " (chat, thinking)" : chat_ ? " (chat)" : "") + " · " + std::to_string(r.n_tokens) + " · h/l moves the cursor";
    return pane(title, hflow(std::move(cols)) | yframe | size(HEIGHT, LESS_THAN, 2 + static_cast<int>(watch_.size()) * 2 + 3), std::nullopt);
}

// GRAPH: the layers, with each layer's share of the target's logit.
Element Session::render_graph() {
    Elements rows;
    float vmax = 1e-6f;
    for (size_t l = 0; l < layer_attn_.size(); ++l) vmax = std::max({vmax, std::fabs(layer_attn_[l]), std::fabs(layer_ffn_[l])});
    if (blame_) rows.push_back(text("      attn         mlp") | dim);
    else rows.push_back(text("run a prompt for per-layer blame") | dim);
    for (int l = 0; l < m_.n_layer; ++l) {
        const bool cur = l == layer_;
        Elements line{text(std::string(cur ? "▾ " : "  ") + "L" + std::to_string(l)) | size(WIDTH, EQUAL, 6)};
        if (blame_) {
            for (float v : {layer_attn_[static_cast<size_t>(l)], layer_ffn_[static_cast<size_t>(l)]}) {
                line.push_back(text(hbar(v, vmax, 5)) | color(sign_color(v)) | size(WIDTH, EQUAL, 6));
                line.push_back(text(fmt_f(v, 2) + " ") | size(WIDTH, EQUAL, 7));
            }
        }
        Element e = hbox(std::move(line));
        if (cur) e = e | inverted | focus;
        rows.push_back(e);
        if (cur) {
            const auto& L = m_.layers[static_cast<size_t>(l)];
            auto t = [](const TensorInfo* x) { return x ? std::string(ggml_type_name(x->type)) : std::string("-"); };
            if (L.kind == LayerKind::linear_attention) {
                rows.push_back(text("    linear (DeltaNet) " + t(L.attn_qkv)) | dim);
            } else {
                rows.push_back(text("    attn q k v o  " + t(L.attn_q ? L.attn_q : L.attn_qkv)) | dim);
            }
            rows.push_back(text("    mlp  " + std::to_string(m_.n_ff) + " neurons " + t(L.ffn_up)) | dim);
            rows.push_back(text("    ▸ N" + std::to_string(neuron_)) | bold);
        }
    }
    if (blame_) rows.push_back(text("  embed " + fmt_f(blame_->embed, 2)) | dim);
    return pane("GRAPH", vbox(std::move(rows)) | yframe, Pane::Graph);
}

Element Session::render_list() {
    const auto rows = list_rows();
    std::string title;
    switch (list_kind_) {
        case ListKind::Blame:
            title = blame_ ? "LIST · blame: who wrote " + quote_printable(token_text(target_), 12) + " at pos " + std::to_string(pos_) : "LIST · blame";
            break;
        case ListKind::Fires: title = "LIST · fires: L" + std::to_string(layer_) + " at pos " + std::to_string(pos_); break;
        case ListKind::Find: title = "LIST · find " + quote_printable(find_text_, 14); break;
        case ListKind::Controls: title = dose_ ? "LIST · dose controls" : "LIST · controls"; break;
        case ListKind::Labels: title = "LIST · labels" + (labels_query_.empty() ? std::string() : " " + quote_printable(labels_query_, 20)); break;
        case ListKind::Probe: title = "LIST · probe: " + std::to_string(probe_pos_.size()) + " POS vs " + std::to_string(probe_neg_.size()) + " NEG"; break;
        case ListKind::Examples: title = "LIST · examples: " + ln(layer_, neuron_); break;
    }
    Elements out;
    if (rows.empty()) {
        std::string hint;
        switch (list_kind_) {
            case ListKind::Blame:
            case ListKind::Fires: hint = "needs a run — :prompt TEXT or :chat MESSAGE"; break;
            case ListKind::Find: hint = "/ or :find \" word\" lists the neurons that write it"; break;
            case ListKind::Controls: hint = "D on a neuron runs a dose sweep; its controls land here"; break;
            case ListKind::Labels:
                hint = labels_ ? "no labels" + std::string(labels_query_.empty() ? "" : " match") + " — :label TEXT names the cursor neuron (or: neuronaut labels MODEL import FILE)"
                               : "labels are off: the labels file couldn't be read";
                break;
            case ListKind::Probe: hint = ":probe POS | POS // NEG | NEG ranks the neurons that separate them"; break;
            case ListKind::Examples: hint = act_index_error_.empty() ? "no examples for this neuron" : act_index_error_; break;
        }
        out.push_back(text(hint) | dim);
    }
    float vmax = 1e-6f;
    for (const auto& r : rows)
        if (!r.section) vmax = std::max(vmax, std::fabs(r.value));
    int si = 0;
    for (const auto& r : rows) {
        if (r.section) {
            out.push_back(text("── " + r.label + " ──") | dim);
            continue;
        }
        std::string nm = !r.name.empty()             ? r.name
                         : r.kind == BlameItem::Embed  ? "embed"
                         : r.kind == BlameItem::Attn ? "L" + std::to_string(r.layer) + " attn"
                         : r.kind == BlameItem::Head ? "L" + std::to_string(r.layer) + " H" + std::to_string(r.neuron)
                                                     : ln(r.layer, r.neuron);
        nm.resize(12, ' ');
        const bool sel = si == list_sel_;
        Element line = hbox({text(sel ? "▶ " : "  "), text(nm), text(hbar(r.value, vmax, 18)) | color(sign_color(r.value)) | size(WIDTH, EQUAL, 19),
                             text(fmt_f(r.value, 3) + "  ") | size(WIDTH, EQUAL, 9), text(r.detail) | dim,
                             text(r.note.empty() ? "" : fit_cells((r.detail.empty() ? "" : "  ") + r.note, 72 - 2 - 2 - 12 - 19 - 9 - display_width(r.detail))) |
                                 color(kLabel)});
        if (r.kind != BlameItem::Neuron || r.inert) line = line | italic;
        if (sel) line = (focus_ == Pane::List ? line | inverted : line | bold) | focus;
        out.push_back(line);
        ++si;
    }
    // Under blame: everyone else, as a distribution.
    if (list_kind_ == ListKind::Blame && blame_) {
        std::vector<float> contribs;
        float rmax = 1e-6f;
        for (const auto& it : blame_->neurons) {
            contribs.push_back(it.contrib);
            rmax = std::max(rmax, std::fabs(it.contrib));
        }
        out.push_back(text("all " + std::to_string(blame_->neurons.size()) + " neurons (log count) " + fmt_f(-rmax, 1) + " … " + fmt_f(rmax, 1)) | dim);
        out.push_back(histogram(contribs, rmax, 60, 2));
        out.push_back(text("logit " + fmt_f(blame_->logit, 2) + " = embed " + fmt_f(blame_->embed, 2) + " + attn " + fmt_f(blame_->sum_attn, 2) +
                           " + mlp " + fmt_f(blame_->sum_neurons, 2) + "  (unexplained " + fmt_f(blame_->unexplained, 3) + ")") | dim);
    }
    out.push_back(filler());
    out.push_back(text("b blame · f fires · / find · c controls · L labels · e examples") | dim);
    return pane(title, vbox(std::move(out)) | yframe, Pane::List);
}

// NEURON: everything about the cursor neuron, static and (if there's a run) dynamic.
Element Session::render_neuron() {
    std::string tag;
    for (const auto& p : patches_)
        if (p.layer == layer_ && p.neuron == neuron_) tag += "  " + edit_tag(p);
    if (std::find(watch_.begin(), watch_.end(), std::make_pair(layer_, neuron_)) != watch_.end()) tag += "  ◉ watched";
    const std::string brief = label_brief(layer_, neuron_);
    const std::string title = "NEURON · " + ln(layer_, neuron_) + (brief.empty() ? "" : " · " + fit_cells(brief, 40)) + tag;
    if (!nv_) return pane(title, text(status_) | dim, Pane::Neuron);
    const NeuronView& v = *nv_;
    Elements rows;
    auto kv = [](const std::string& k, Element val) { return hbox({text(k) | dim | size(WIDTH, EQUAL, 10), std::move(val)}); };
    // Labels first: what people (or an import) think this neuron is.
    if (labels_) {
        for (const auto& x : labels_->at(layer_, neuron_)) {
            std::string tg;
            for (const auto& t : x.tags) tg += " #" + t;
            // ~60 cells of text after the 10-cell key fits the default 179-column layout.
            const std::string name = fit_cells(x.label, 40);
            const std::string rest = fit_cells(tg + "  [" + x.source + "]" + (x.notes.empty() ? "" : "  " + x.notes), 60 - display_width(name));
            rows.push_back(kv(x.pole ? std::string("label ") + (x.pole == '+' ? "+" : "−") : "label", hbox({text(name) | bold | color(kLabel), text(rest) | dim})));
        }
    }

    if (res_ && res_->act(layer_, pos_)) {
        const size_t F = static_cast<size_t>(m_.n_ff);
        const float* here = res_->act(layer_, pos_);
        const float a = here[static_cast<size_t>(neuron_)];
        int rank = 1;
        for (size_t n = 0; n < F; ++n) rank += std::fabs(here[n]) > std::fabs(a);
        rows.push_back(kv("here", hbox({text("a = " + fmt_f(a, 3)) | bold | color(sign_color(a)),
                                        text("  at pos " + std::to_string(pos_) + "   rank " + std::to_string(rank) + " of " + std::to_string(F) +
                                             " in L" + std::to_string(layer_) + " by |a|") | dim})));
        if (blame_) {
            const BlameItem& b = blame_->neurons[static_cast<size_t>(layer_) * F + static_cast<size_t>(neuron_)];
            rows.push_back(kv("blame", hbox({text(fmt_f(b.contrib, 3)) | bold | color(sign_color(b.contrib)),
                                             text(" on " + quote_printable(token_text(target_), 12) + " logit  (= a′ × down·u / rms, a′=" +
                                                  fmt_f(b.act, 2) + ")") | dim})));
        }
        // Activation over time: one coloured cell per position, up to 64 of the
        // captured ones, centred on the cursor.
        const int c_lo = res_->cap_lo, c_hi = res_->cap_lo + res_->cap_n;
        const int t_lo = std::clamp(pos_ - 32, c_lo, std::max(c_lo, c_hi - 64)), t_hi = std::min(c_hi, t_lo + 64);
        auto at = [&](int i) { return res_->act(layer_, i)[static_cast<size_t>(neuron_)]; };
        float amax = 1e-6f;
        for (int i = t_lo; i < t_hi; ++i) amax = std::max(amax, std::fabs(at(i)));
        static const char* lv[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
        Elements cells;
        for (int i = t_lo; i < t_hi; ++i) {
            const float x = at(i);
            Element c = text(lv[std::clamp(static_cast<int>(std::fabs(x) / amax * 7.99f), 0, 7)]) | color(sign_color(x));
            if (i == pos_) c = c | inverted;
            cells.push_back(c);
        }
        rows.push_back(kv("over time", hbox({hbox(std::move(cells)), text("  max |a| " + fmt_f(amax, 2).substr(1)) | dim})));
        rows.push_back(separatorLight());
    } else {
        rows.push_back(text("run a prompt to see this neuron fire (:prompt / :chat)") | dim);
        rows.push_back(separatorLight());
    }

    // What it writes: the static lens of its down column.
    if (const LensResult* lens = neuron_lens()) {
        auto toks = [&](const std::vector<LensToken>& ts) {
            std::string s;
            for (size_t i = 0; i < ts.size() && i < 7; ++i) s += (i ? "  " : "") + quote_printable(ts[i].text, 12);
            return s;
        };
        rows.push_back(kv("writes +", paragraph(toks(lens->top)) | color(sign_color(1))));
        rows.push_back(kv("writes −", paragraph(toks(lens->bottom)) | color(kOrange)));
    }
    // Where it fires across the corpus (actindex.h), when an index exists.
    if (act_index() && tok_) {
        const auto ex = act_index_->top(layer_, neuron_, '+');
        if (!ex.empty())
            rows.push_back(kv("corpus +", text(fit_cells(fmt_f(ex[0].act, 2) + "  " + quote_printable(example_context(ex[0], 10, 2), 90), 60)) | dim));
    }
    rows.push_back(separatorLight());

    // What it reads and writes, as vectors.
    auto strip = [&](Slice s, const char* label, const std::vector<float>& vec) {
        const VecStats st = stats(vec);
        return kv(std::string(slice_ == s ? "▶ " : "  ") + label,
                  hbox({heat_strip(vec, slice_ == s ? dim_ : -1, 48), text("  ‖" + fmt_f(st.l2, 2).substr(1) + "‖") | dim}));
    };
    rows.push_back(strip(Slice::Gate, "gate", v.gate));
    rows.push_back(strip(Slice::Up, "up", v.up));
    rows.push_back(strip(Slice::Down, "down", v.down));
    rows.push_back(kv("cos", text("gate·up " + fmt_f(cosine(v.gate, v.up), 2) + "   up·down " + fmt_f(cosine(v.up, v.down), 2) +
                                  "   gate·down " + fmt_f(cosine(v.gate, v.down), 2)) | dim));
    rows.push_back(kv("stored", text("up row " + std::to_string(v.up_fp.n_blocks) + " blocks (own) · down col " +
                                     std::to_string(v.down_fp.n_blocks) + " blocks" +
                                     (v.down_fp.shared_with ? " (shared ×" + std::to_string(v.down_fp.shared_with) + ")" : " (own)")) | dim));
    return pane(title, vbox(std::move(rows)) | yframe, Pane::Neuron);
}

// BYTES: the quantization block under the byte cursor.
Element Session::render_bytes() {
    const TensorInfo* t = slice_tensor();
    if (!t) return pane("BYTES", text("no FFN in this layer") | dim, Pane::Bytes);
    // gate/up: row = neuron, col = embedding dim.  down: row = embedding dim, col = neuron.
    const int64_t row = (slice_ == Slice::Down) ? dim_ : neuron_;
    const int64_t col = (slice_ == Slice::Down) ? neuron_ : dim_;
    BlockDump b;
    try {
        b = dump_block(*t, row, col);
    } catch (const std::exception& e) {
        return pane("BYTES", text(e.what()), Pane::Bytes);
    }
    const int64_t blk = static_cast<int64_t>(b.values.size());
    Elements rows;
    std::ostringstream hdr;
    if (blk == 1) {
        // Unquantized (F32/F16/BF16): a "block" is one value, so show a run of
        // 16 around the cursor along the row — still exactly the bytes on disk.
        const int64_t c0 = std::clamp<int64_t>(col / 16 * 16, 0, std::max<int64_t>(0, t->ne[0] - 16));
        std::vector<uint8_t> raw;
        Elements vals;
        for (int64_t c = c0; c < c0 + 16 && c < t->ne[0]; ++c) {
            const BlockDump e = dump_block(*t, row, c);
            raw.insert(raw.end(), e.raw.begin(), e.raw.end());
            auto v = text(fmt_f(e.values[0], 3) + " ");
            if (c == col) v = v | inverted;
            vals.push_back(v);
        }
        hdr << "BYTES · " << b.tensor << " row " << row << " cols " << c0 << "–" << c0 + 15 << " @0x" << std::hex
            << (b.file_offset - static_cast<uint64_t>(col - c0) * b.raw.size()) << std::dec << " " << ggml_type_name(b.type);
        if (slice_ == Slice::Down) rows.push_back(text("row = embedding dim " + std::to_string(row) + "; each column is a different neuron") | dim);
        for (size_t off = 0; off < raw.size(); off += 32) {
            std::ostringstream a;
            a << "+" << std::hex << off;
            std::string addr = a.str();
            addr.resize(5, ' ');
            rows.push_back(hbox({text(addr) | dim, text(hex_bytes(raw.data() + off, std::min<size_t>(32, raw.size() - off)))}));
        }
        rows.push_back(hflow(std::move(vals)));
        return pane(hdr.str(), vbox(std::move(rows)), Pane::Bytes);
    }
    hdr << "BYTES · " << b.tensor << " row " << b.row << " block " << b.block_index << " @0x" << std::hex << b.file_offset << std::dec
        << " " << b.raw.size() << "B " << ggml_type_name(b.type);
    if (slice_ == Slice::Down && blk > 1)
        rows.push_back(text("neurons " + std::to_string(b.first_elem) + "–" + std::to_string(b.first_elem + blk - 1) + " share this block") | dim);
    for (size_t off = 0; off < b.raw.size(); off += 32) {
        std::ostringstream a;
        a << "+" << std::hex << off;
        std::string addr = a.str();
        addr.resize(5, ' ');
        rows.push_back(hbox({text(addr) | dim, text(hex_bytes(b.raw.data() + off, std::min<size_t>(32, b.raw.size() - off)))}));
    }
    for (const auto& f : b.fields) {
        if (f.name.rfind("qs", 0) == 0) continue;   // the raw quants are already in the hex above
        rows.push_back(hbox({text(f.name + "  ") | dim, text(f.value)}));
    }
    Elements vals;
    for (int64_t i = 0; i < blk; ++i) {
        auto e = text(fmt_f(b.values[static_cast<size_t>(i)], 3) + " ");
        if (b.first_elem + i == col) e = e | inverted;
        vals.push_back(e);
    }
    rows.push_back(hflow(std::move(vals)));
    return pane(hdr.str(), vbox(std::move(rows)), Pane::Bytes);
}

Element Session::render_patches() {
    Elements rows;
    const int np = static_cast<int>(patches_.size());
    int i = 0;
    for (const auto& p : patches_) {
        const std::string lb = label_brief(p.layer, p.neuron);
        const Portability pt = portability(p);
        Element e = hbox({text("⚑ " + edit_text(p) + (lb.empty() ? "" : " (" + fit_cells(lb, 30) + ")")) | color(kOrange),
                          text(pt == Portability::Gguf ? " ·gguf" : pt == Portability::ControlVector ? " ·cvec" : " ·local") | dim});
        if (focus_ == Pane::Patches && i == patch_sel_) e = e | inverted;
        rows.push_back(hbox({e, text("    ")}));
        ++i;
    }
    for (auto [l, n] : watch_) {
        const std::string lb = label_brief(l, n);
        Element e = text("◉ " + ln(l, n) + " watched" + (lb.empty() ? "" : " (" + fit_cells(lb, 30) + ")"));
        if (focus_ == Pane::Patches && i == patch_sel_) e = e | inverted;
        rows.push_back(hbox({e, text("    ")}));
        ++i;
    }
    if (np + static_cast<int>(watch_.size()) == 0) rows.push_back(text("none — on any neuron: x ablate · i invert · w watch") | dim);
    if (target_pinned_) rows.push_back(text("target pinned to " + quote_printable(token_text(target_), 14) + " (:target to unpin)") | dim);
    // Title: the set's name and how far it can travel (edits.h): gguf (any runtime),
    // cvec (llama.cpp control vector), local (neuronaut only: prompt-scoped).
    std::string title = "PATCHES";
    if (!edits_name_.empty()) title += " · " + edits_name_;
    if (!patches_.empty()) {
        Portability worst = Portability::Gguf;
        for (const auto& p : patches_) worst = std::max(worst, portability(p));
        title += std::string(" · portable as: ") + portability_name(worst);
    }
    return pane(title, hflow(std::move(rows)), Pane::Patches);
}

// STATE: what the model will do from the cursor — the registers.
Element Session::render_state() {
    if (!res_) return pane("STATE", text("no run yet") | dim, std::nullopt);
    update_state_lens();
    Elements rows;
    // Final next-token distribution at the cursor position, base vs patched.
    rows.push_back(text("next token after pos " + std::to_string(pos_) + (base_ ? "      patched     base" : "")) | bold);
    for (auto [id, p] : top_probs(res_->logits_at(pos_), res_->n_vocab, 5)) {
        Elements line{text(tok_label(token_text(id), 14)) | size(WIDTH, EQUAL, 16), text(hbar(p, 1.0f, 10)) | color(sign_color(1)) | size(WIDTH, EQUAL, 11),
                      text(fmt_f(100.0 * p, 1).substr(1) + "%") | size(WIDTH, EQUAL, 7)};
        if (base_) {
            const float pb = prob_of(base_->logits_at(pos_), base_->n_vocab, id);
            const float d = p - pb;
            line.push_back(text(fmt_f(100.0 * pb, 1).substr(1) + "%") | dim | size(WIDTH, EQUAL, 7));
            if (std::fabs(d) >= 0.001f) line.push_back(text(std::string(d > 0 ? "▲" : "▼") + fmt_f(100.0 * d, 1)) | color(sign_color(d)));
        }
        rows.push_back(hbox(std::move(line)));
    }
    // Target across layers + the lens after the cursor layer.
    if (!target_by_layer_.empty()) {
        float tmax = 1e-6f;
        for (float x : target_by_layer_) tmax = std::max(tmax, std::fabs(x));
        static const char* lv[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
        Elements cells;
        for (size_t l = 0; l < target_by_layer_.size(); ++l) {
            const float x = target_by_layer_[l];
            Element c = text(lv[std::clamp(static_cast<int>(std::max(0.0f, x) / tmax * 7.99f), 0, 7)]) | color(sign_color(x));
            if (static_cast<int>(l) == layer_) c = c | inverted;
            cells.push_back(c);
        }
        rows.push_back(hbox({text("target by layer ") | dim, hbox(std::move(cells)),
                             text("  " + quote_printable(token_text(target_), 12) + " after L" + std::to_string(layer_) + ": rank " +
                                  std::to_string(state_lens_.target_rank) + ", " + fmt_f(100.0 * state_lens_.target_prob, 1).substr(1) + "%") | dim}));
        std::string lens = "lens after L" + std::to_string(layer_) + ":";
        for (auto [id, p] : state_lens_.top) lens += "  " + quote_printable(token_text(id), 10);
        rows.push_back(text(lens) | dim);
    }
    rows.push_back(separatorLight());
    if (gen_patched_.empty()) {
        rows.push_back(text(":gen 16 continues from here (baseline vs patched) · :accept appends it") | dim);
    } else if (patches_.empty()) {
        rows.push_back(paragraph("→ " + quote_printable(gen_patched_, 300)));
    } else {
        rows.push_back(paragraph("base    → " + quote_printable(gen_base_, 300)) | dim);
        rows.push_back(paragraph("patched → " + quote_printable(gen_patched_, 300)) | color(kOrange));
    }
    return pane("STATE · pos " + std::to_string(pos_) + " · after L" + std::to_string(layer_), vbox(std::move(rows)), std::nullopt);
}

// Hotkeys: what works in the focused pane, then what works everywhere.
Element Session::render_keybar() {
    using K = std::vector<std::pair<std::string, std::string>>;
    K keys;
    if (cmdline) {
        keys = {{"Tab", "complete"}, {"↑↓", "history"}, {"Enter", "run"}, {"Esc", "cancel"}};
    } else if (overlay_ != Overlay::None) {
        keys = {{"Esc", "close"}};
        if (overlay_ == Overlay::Model) keys.insert(keys.begin(), {"j/k", "scroll"});
        if (overlay_ == Overlay::Attn) keys.insert(keys.begin(), {{"j/k", "head"}, {"h/l", "token"}, {"PgUp/Dn", "layer"}});
    } else {
        switch (focus_) {
            case Pane::List: keys = {{"j/k", "select row"}, {"Enter", "→ NEURON"}, {"b f /", "blame fires find"}}; break;
            case Pane::Graph: keys = {{"j/k", "layer"}, {"Enter", "fires in layer"}}; break;
            case Pane::Neuron: keys = {{"j/k", "neuron ±1"}, {"J/K", "±64"}, {"g u d", "slice"}}; break;
            case Pane::Bytes: keys = {{"j/k", "dim ±1"}, {"J/K", "±32"}, {"g u d", "slice"}}; break;
            case Pane::Patches: keys = {{"j/k", "select"}, {"Del", "remove"}}; break;
        }
        const K global = {{"h/l", "token"}, {"PgUp/Dn", "layer"}, {"Tab", "pane"}, {"x", "ablate"}, {"i", "invert"},
                          {"w", "watch"},   {"D", "dose"},        {":", "cmd"},    {"?", "help"},   {"q", "quit"}};
        keys.insert(keys.end(), global.begin(), global.end());
    }
    Elements items;
    for (const auto& [k, label] : keys) {
        items.push_back(text(" " + k + " ") | bold | bgcolor(Color::RGB(40, 90, 160)) | color(Color::White));
        items.push_back(text(" " + label + " "));
    }
    return hbox(std::move(items));
}

Element Session::render_footer() {
    Elements rows;
    if (cmdline) {
        // Live hints: usage of the matched command, then completion candidates.
        const Completion c = complete(*cmdline);
        if (!c.usage.empty()) rows.push_back(text("  " + c.usage) | dim);
        const bool cycling = !tab_cands_.empty() && tab_last_ == *cmdline;
        const std::vector<std::string>& list = cycling ? tab_cands_ : c.candidates;
        if (!list.empty()) {
            Elements cands;
            size_t used = 2, shown = 0;
            for (size_t i = 0; i < list.size(); ++i) {
                const size_t sp = list[i].find(' ');
                const std::string label = (sp != std::string::npos && sp != list[i].size() - 1) ? list[i].substr(sp + 1) : list[i];
                if (used + label.size() + 3 > 160) break;
                used += label.size() + 3;
                auto t = text(" " + label + " ");
                t = (cycling && i == tab_idx_) ? t | inverted : t | color(Color::Cyan);
                cands.push_back(hbox({t, text(" ")}));
                ++shown;
            }
            if (list.size() > shown) cands.push_back(text("+" + std::to_string(list.size() - shown) + " more") | dim);
            rows.push_back(hbox({text("  "), hbox(std::move(cands))}));
        }
        const std::string& l = *cmdline;
        const size_t cur = cmd_cursor ? std::min<size_t>(static_cast<size_t>(std::max(0, *cmd_cursor)), l.size()) : l.size();
        const std::string at = cur < l.size() ? l.substr(cur, 1) : " ";
        rows.push_back(hbox({text(":") | bold, text(l.substr(0, cur)), text(at) | inverted, text(cur < l.size() ? l.substr(cur + 1) : "")}));
    } else {
        rows.push_back(text(status_.empty() ? " " : " " + status_) | dim);
    }
    rows.push_back(render_keybar());
    return vbox(std::move(rows));
}

Element Session::render_help() {
    Elements cmds;
    cmds.push_back(text("commands  (: then Tab completes; ↑↓ history)") | bold);
    for (const auto& c : commands()) {
        cmds.push_back(hbox({text("  " + std::string(c.name) + " " + c.args) | color(Color::Cyan) | size(WIDTH, EQUAL, 22), paragraph(c.help) | flex}));
    }
    auto sect = [](const std::string& title, std::vector<std::pair<std::string, std::string>> ks) {
        Elements e{text(title) | bold};
        for (auto& [k, v] : ks) e.push_back(hbox({text("  " + k) | color(Color::Cyan) | size(WIDTH, EQUAL, 12), text(v)}));
        return vbox(std::move(e));
    };
    Element keys = vbox({
        sect("the cursor  (pos · layer · neuron, shown in the header)",
             {{"h l", "previous / next token — every pane follows"},
              {"PgUp PgDn", "previous / next layer"},
              {"Tab S-Tab", "move focus between panes (bright border)"},
              {"j k", "move within the focused pane"}}),
        text(""),
        sect("act on the cursor neuron, from any pane",
             {{"x", "ablate (toggle)"}, {"i", "invert ×-1 (toggle)"}, {"w", "watch (toggle)"}, {"D", "dose–response sweep → plot"}}),
        text(""),
        sect("panes",
             {{"GRAPH", "layers; bars = attn / mlp share of the target logit"},
              {"LIST", "b blame · f fires · / find · c controls · L labels · e examples"},
              {"NEURON", "j/k neighbour neuron · g u d slice"},
              {"BYTES", "j/k byte cursor (embedding dim)"},
              {"PATCHES", "j/k select · Del remove"},
              {"STATE", "next token, lens after the cursor layer"}}),
        text(""),
        sect("overlays (Esc closes)",
             {{"?", "this help"}, {"M", "model: tensors + metadata"}, {"P", "last dose–response plot"}, {"A", "attention: heads of the cursor layer"},
              {"T", "causal trace map (:corrupt, :trace)"}}),
        text(""),
        sect("colours", {{"blue", "positive / raises the target"}, {"orange", "negative / lowers it"}, {"⚑ ◉", "patched / watched"}}),
    });
    return window(text(" help — Esc closes "), hbox({vbox(std::move(cmds)) | size(WIDTH, EQUAL, 92), text("  "), keys | size(WIDTH, EQUAL, 66)}));
}

Element Session::render_model() {
    const auto& ts = file_->tensors();
    Elements rows;
    rows.push_back(text(" name                           type    shape            bytes      offset") | bold);
    for (int i = 0; i < static_cast<int>(ts.size()); ++i) {
        const auto& t = ts[static_cast<size_t>(i)];
        if (!tensor_filter_.empty() && t.name.find(tensor_filter_) == std::string::npos) continue;
        std::ostringstream off;
        off << "0x" << std::hex << t.file_offset;
        auto e = hbox({text(" " + t.name) | size(WIDTH, EQUAL, 32), text(ggml_type_name(t.type)) | size(WIDTH, EQUAL, 8),
                       text(shape_short(t)) | size(WIDTH, EQUAL, 17), text(human_bytes(t.n_bytes)) | size(WIDTH, EQUAL, 11), text(off.str())});
        if (i == model_sel_) e = e | inverted | focus;
        rows.push_back(e);
    }
    Elements meta;
    for (const auto& e : file_->metadata()) {
        meta.push_back(hbox({text(e.key) | bold | size(WIDTH, EQUAL, 40), text(e.type) | dim | size(WIDTH, EQUAL, 16),
                             text(e.type == "str" ? quote_printable(e.value, 60) : e.value)}));
    }
    std::string title = " model · " + std::to_string(ts.size()) + " tensors" + (tensor_filter_.empty() ? "" : " (filter '" + tensor_filter_ + "')") +
                        " · " + std::to_string(file_->metadata().size()) + " keys · Esc closes ";
    return window(text(title), hbox({vbox(std::move(rows)) | yframe | vscroll_indicator | size(WIDTH, EQUAL, 82), separator(),
                                     vbox(std::move(meta)) | yframe | flex}));
}

Element Session::render() {
    if (labels_) {
        try {
            labels_->refresh();   // another process (CLI import, MCP) may have written it
        } catch (const std::exception& e) {
            status_ = e.what();
        }
    }
    Element page = vbox({
        render_header(),
        render_tokens(),
        hbox({render_graph() | size(WIDTH, EQUAL, 36), render_list() | size(WIDTH, EQUAL, 72), render_neuron() | flex}) | flex,
        hbox({vbox({render_bytes(), render_patches() | flex}) | size(WIDTH, EQUAL, 108), render_state() | flex}) | size(HEIGHT, EQUAL, 15),
        render_footer(),
    });
    Element over;
    switch (overlay_) {
        case Overlay::None: return page;
        case Overlay::Help: over = render_help(); break;
        case Overlay::Model: over = render_model(); break;
        case Overlay::Plot: over = render_dose(); break;
        case Overlay::Attn: over = render_attention(); break;
        case Overlay::Trace: over = render_trace(); break;
    }
    return dbox({page, over | clear_under | center | size(WIDTH, LESS_THAN, 172) | size(HEIGHT, LESS_THAN, 48)});
}

std::vector<std::string> Session::context_neurons() const {
    std::vector<std::string> out;
    auto add = [&](int l, int n) {
        const std::string s = std::to_string(l) + "." + std::to_string(n);
        if (std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
    };
    add(layer_, neuron_);   // the cursor first: it's what x / i / w / D act on
    for (const auto& p : patches_) add(p.layer, p.neuron);
    for (auto [l, n] : watch_) add(l, n);
    if (dose_) add(dose_->layer, dose_->neuron);
    if (blame_)
        for (const auto& it : blame_->top(10, true))
            if (it.kind == BlameItem::Neuron) add(it.layer, it.neuron);
    return out;
}

const std::vector<std::string>& Session::vocab_text() {
    if (vocab_text_.empty()) {
        const std::string tm = file_->get_str("tokenizer.ggml.model").value_or("");
        const size_t n = file_->get_arr_len("tokenizer.ggml.tokens");
        vocab_text_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const std::string t = file_->get_arr_str("tokenizer.ggml.tokens", i).value_or("");
            vocab_text_.push_back(neuronaut::vocab_text(tm, t));
        }
    }
    return vocab_text_;
}

Session::Completion Session::complete(const std::string& line) {
    Completion c;
    const size_t sp = line.find(' ');
    const std::string word = line.substr(0, sp);
    if (sp == std::string::npos) {
        for (const auto& cmd : commands())
            if (std::string(cmd.name).rfind(word, 0) == 0) c.candidates.push_back(cmd.name);
        if (c.candidates.size() == 1 || (!word.empty() && std::find(c.candidates.begin(), c.candidates.end(), word) != c.candidates.end())) {
            for (const auto& cmd : commands())
                if (cmd.name == (c.candidates.size() == 1 ? c.candidates[0] : word)) c.usage = std::string(cmd.name) + " " + cmd.args + " — " + cmd.help;
        }
        if (c.candidates.size() == 1 && c.candidates[0] == word) c.candidates.clear();
        return c;
    }
    const CmdSpec* spec = nullptr;
    for (const auto& cmd : commands())
        if (word == cmd.name) spec = &cmd;
    if (!spec) return c;
    c.usage = std::string(spec->name) + " " + spec->args + " — " + spec->help;
    const std::string arg = line.substr(sp + 1);
    const std::string head = word + " ";
    auto offer = [&](const std::string& a) {
        if (a.rfind(arg, 0) == 0 && a != arg) c.candidates.push_back(head + a);
    };
    switch (spec->arg) {
        case Arg::Neuron:
            for (const auto& n : context_neurons()) offer(n);
            break;
        case Arg::Words:
            for (const char* w : spec->words) offer(w);
            break;
        case Arg::Prompt:
            for (auto it = history.rbegin(); it != history.rend(); ++it) {
                if (it->rfind(head, 0) == 0) {
                    const std::string a = it->substr(head.size());
                    if (std::find(c.candidates.begin(), c.candidates.end(), head + a) == c.candidates.end()) offer(a);
                }
            }
            break;
        case Arg::Token: {
            if (arg.empty() || arg[0] != '"') {
                if (arg.empty()) c.candidates.push_back(head + "\"");
                break;
            }
            std::string pre = arg.substr(1);
            if (!pre.empty() && pre.back() == '"') pre.pop_back();
            if (pre.empty()) break;
            const auto& vt = vocab_text();
            std::vector<std::pair<size_t, size_t>> hits;   // (length, id)
            for (size_t i = 0; i < vt.size(); ++i)
                if (vt[i].size() >= pre.size() && vt[i].compare(0, pre.size(), pre) == 0 &&
                    vt[i].find_first_of("\"\n\r\t") == std::string::npos && valid_utf8(vt[i]))
                    hits.emplace_back(vt[i].size(), i);
            std::sort(hits.begin(), hits.end());
            for (size_t i = 0; i < hits.size() && c.candidates.size() < 40; ++i) {
                const std::string cand = head + "\"" + vt[hits[i].second] + "\"";
                if (std::find(c.candidates.begin(), c.candidates.end(), cand) == c.candidates.end()) offer("\"" + vt[hits[i].second] + "\"");
            }
            break;
        }
        default:
            break;
    }
    return c;
}

void Session::tab(std::string& line) {
    if (!tab_cands_.empty() && line == tab_last_) {
        tab_idx_ = (tab_idx_ + 1) % tab_cands_.size();
        line = tab_last_ = tab_cands_[tab_idx_];
        return;
    }
    const Completion c = complete(line);
    if (c.candidates.empty()) return;
    if (c.candidates.size() == 1) {
        line = c.candidates[0];
        if (line.find(' ') == std::string::npos) line += ' ';
        tab_cands_.clear();
        return;
    }
    std::string lcp = c.candidates[0];
    for (const auto& x : c.candidates) {
        size_t i = 0;
        while (i < lcp.size() && i < x.size() && lcp[i] == x[i]) ++i;
        lcp.resize(i);
    }
    if (lcp.size() > line.size()) {
        line = lcp;
        tab_cands_.clear();
        return;
    }
    tab_cands_ = c.candidates;
    tab_idx_ = 0;
    line = tab_last_ = tab_cands_[0];
}

// ===========================================================================
// Dose–response
// ===========================================================================

const std::vector<float>& Session::down_norms(int layer) {
    auto it = down_norm_cache_.find(layer);
    if (it != down_norm_cache_.end()) return it->second;
    // ‖down column n‖ for every neuron: one pass over the rows of ffn_down.
    const TensorInfo& down = *m_.layers[static_cast<size_t>(layer)].ffn_down;
    std::vector<double> ss(static_cast<size_t>(down.ne[0]), 0.0);
    for (int64_t e = 0; e < down.ne[1]; ++e) {
        const std::vector<float> row = read_row(down, e);
        for (size_t n = 0; n < row.size(); ++n) ss[n] += static_cast<double>(row[n]) * row[n];
    }
    std::vector<float> out(ss.size());
    for (size_t n = 0; n < ss.size(); ++n) out[n] = static_cast<float>(std::sqrt(ss[n]));
    return down_norm_cache_.emplace(layer, std::move(out)).first->second;
}

void Session::start_dose(int layer, int neuron) {
    if (!res_ || tokens_.empty()) {
        status_ = "set a prompt first: :prompt TEXT";
        return;
    }
    if (!m_.has_ffn(layer) || neuron < 0 || neuron >= m_.n_ff) {
        status_ = "no such neuron";
        return;
    }
    if (busy()) return;
    join_worker();

    constexpr int kControls = 16;
    const size_t F = static_cast<size_t>(m_.n_ff);
    const int pos = pos_;
    const float* a = res_->act(layer, pos);
    if (!a) {
        status_ = "no activations at the cursor (re-run the prompt)";
        return;
    }
    const auto& norms = down_norms(layer);

    DoseSweep d;
    d.layer = layer;
    d.neuron = neuron;
    d.pos = pos;
    d.target = target_;
    d.doses = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    d.push = std::fabs(a[neuron]) * norms[static_cast<size_t>(neuron)];

    // Dose-matched controls: same layer, closest push in log space, excluding
    // the tested neuron and anything already patched.
    std::vector<std::pair<float, int>> cand;
    for (size_t n = 0; n < F; ++n) {
        if (static_cast<int>(n) == neuron) continue;
        bool patched = false;
        for (const auto& p : patches_) patched |= (p.layer == layer && p.neuron == static_cast<int>(n));
        if (patched) continue;
        const float push = std::fabs(a[n]) * norms[n];
        if (push <= 0) continue;
        cand.emplace_back(std::fabs(std::log(push) - std::log(std::max(d.push, 1e-9f))), static_cast<int>(n));
    }
    std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(kControls, cand.size()), cand.end());
    for (size_t i = 0; i < cand.size() && i < kControls; ++i) {
        const int n = cand[i].second;
        d.controls.emplace_back(n, std::fabs(a[n]) * norms[static_cast<size_t>(n)]);
    }
    const size_t nd = d.doses.size();
    d.curve.assign(nd, std::numeric_limits<float>::quiet_NaN());
    d.ctrl.assign(d.controls.size(), std::vector<float>(nd, std::numeric_limits<float>::quiet_NaN()));
    d.total = static_cast<int>(nd * (1 + d.controls.size()));
    d.running = true;
    dose_ = d;
    overlay_ = Overlay::Plot;
    cancel_ = false;
    status_ = "dose sweep: " + std::to_string(d.total) + " forward passes…";

    // Everything the worker needs, by value: causality means logits at `pos`
    // only depend on tokens[0..pos], so the prefix is all we have to run.
    const std::vector<int32_t> prefix(tokens_.begin(), tokens_.begin() + pos + 1);
    const Interventions base_iv = iv_;
    const Forward* fw = fw_.get();
    const ModelLayout* m = &m_;
    const int target = d.target;
    const auto doses = d.doses;
    std::vector<int> units{neuron};
    for (auto [n, p] : d.controls) units.push_back(n);

    auto job = [this, prefix, base_iv, fw, m, layer, target, doses, units](auto deliver) {
        const auto t0 = std::chrono::steady_clock::now();
        // The residual entering `layer` is the same for every dose and unit (the
        // swept scale only acts from `layer` on), so compute it once over the prefix
        // and run each of the ~150 passes from `layer` (Forward::run_from) — a sweep
        // on L15 of the 1B does 1/16 of the work it used to, bit-identically.
        RunOptions pre;
        if (layer == 0) pre.capture_resid = true;
        else pre.capture_layer_resid = true;
        const ForwardResult base = fw->run(prefix, base_iv, pre);
        const std::vector<float> x_in = layer == 0 ? base.embed : base.layer_resid.at(layer - 1);
        const int T = static_cast<int>(prefix.size());
        for (size_t u = 0; u < units.size(); ++u) {
            for (size_t di = 0; di < doses.size(); ++di) {
                if (cancel_) return;
                Interventions iv = base_iv;
                iv.scale(*m, layer, units[u], doses[di]);
                const ForwardResult r = fw->run_from(layer, x_in, T, iv);
                const float p = prob_of(r.logits.data(), r.n_vocab, target);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                deliver([this, u, di, p, secs] {
                    if (!dose_) return;
                    if (u == 0) dose_->curve[di] = p;
                    else dose_->ctrl[u - 1][di] = p;
                    ++dose_->done;
                    dose_->seconds = secs;
                });
            }
        }
    };
    auto finish = [this] {
        if (!dose_) return;
        dose_->running = false;
        status_ = cancel_ ? "dose sweep cancelled"
                          : "dose sweep done in " + fmt_f(dose_->seconds, 1).substr(1) + " s — c lists its controls in LIST, P reopens the plot";
    };

    if (post) {
        auto p = post;
        worker_ = std::thread([job, finish, p] {
            job([&p](std::function<void()> f) { p(std::move(f)); });
            p(finish);
        });
    } else {
        job([](std::function<void()> f) { f(); });   // headless: run inline
        finish();
    }
}

Element Session::render_dose() {
    if (!dose_) return window(text(" dose–response "), text("no sweep yet — D on a neuron, or :dose L.N") | dim);
    const DoseSweep& d = *dose_;
    const size_t nd = d.doses.size();
    const std::string who = ln(d.layer, d.neuron);
    const std::string tgt = quote_printable(token_text(d.target), 16);

    // Index of the unmodified dose (×1): every Δ is measured against it.
    size_t i1 = 0;
    for (size_t i = 0; i < nd; ++i)
        if (d.doses[i] == 1.0f) i1 = i;

    float ylo = 1, yhi = 0;
    auto see = [&](float v) {
        if (!std::isnan(v)) {
            ylo = std::min(ylo, v);
            yhi = std::max(yhi, v);
        }
    };
    for (float v : d.curve) see(v);
    for (const auto& c : d.ctrl)
        for (float v : c) see(v);
    if (ylo > yhi) {
        ylo = 0;
        yhi = 1;
    }
    const float pad = std::max(0.02f, (yhi - ylo) * 0.08f);
    ylo = std::max(0.0f, ylo - pad);
    yhi = std::min(1.0f, yhi + pad);

    // Braille canvas: 2×4 dots per cell.
    const int CW = 110, CH = 16;
    const int W = CW * 2, H = CH * 4;
    const float xlo = d.doses.front(), xhi = d.doses.back();
    auto Xc = [&](float dose) { return static_cast<int>(std::lround((dose - xlo) / (xhi - xlo) * static_cast<float>(W - 1))); };

    // NB: FTXUI calls this later, during layout — capture by value only.
    auto plot = canvas(W, H, [d, i1, nd, W, H, xlo, xhi, ylo, yhi](Canvas& c) {
        auto X = [&](float dose) { return static_cast<int>(std::lround((dose - xlo) / (xhi - xlo) * static_cast<float>(W - 1))); };
        auto Y = [&](float p) { return static_cast<int>(std::lround((1.0f - (p - ylo) / (yhi - ylo)) * static_cast<float>(H - 1))); };
        for (int y = 0; y < H; y += 3) {
            c.DrawPoint(X(0.0f), y, true, Color::GrayDark);
            c.DrawPoint(X(1.0f), y, true, Color::GrayDark);
        }
        if (!std::isnan(d.curve[i1]))
            for (int x = 0; x < W; x += 3) c.DrawPoint(x, Y(d.curve[i1]), true, Color::GrayDark);
        for (const auto& cv : d.ctrl)
            for (size_t i = 0; i + 1 < nd; ++i)
                if (!std::isnan(cv[i]) && !std::isnan(cv[i + 1]))
                    c.DrawPointLine(X(d.doses[i]), Y(cv[i]), X(d.doses[i + 1]), Y(cv[i + 1]), Color::GrayLight);
        const Color hot = Color::RGB(255, 150, 30);
        for (size_t i = 0; i + 1 < nd; ++i)
            if (!std::isnan(d.curve[i]) && !std::isnan(d.curve[i + 1]))
                c.DrawPointLine(X(d.doses[i]), Y(d.curve[i]), X(d.doses[i + 1]), Y(d.curve[i + 1]), hot);
        for (size_t i = 0; i < nd; ++i)
            if (!std::isnan(d.curve[i])) c.DrawPointCircleFilled(X(d.doses[i]), Y(d.curve[i]), 1, hot);
    });

    Elements ylab;
    for (int r = 0; r < CH; ++r) {
        const float p = yhi - (yhi - ylo) * static_cast<float>(r) / static_cast<float>(CH - 1);
        ylab.push_back(text((r % 3 == 0 || r == CH - 1) ? fmt_f(100.0 * p, 1).substr(1) + "% " : "") | align_right);
    }
    std::string xl(static_cast<size_t>(CW), ' ');
    for (float dv : d.doses) {
        std::string lab = (dv == 0 ? "0" : fmt_f(dv, 1)) + "×";
        if (lab[0] == '+') lab = lab.substr(1);
        const int cell = std::clamp(Xc(dv) / 2 - static_cast<int>(lab.size()) / 2, 0, CW - static_cast<int>(lab.size()));
        xl.replace(static_cast<size_t>(cell), lab.size(), lab);
    }
    Element chart = vbox({
        hbox({vbox(std::move(ylab)) | size(WIDTH, EQUAL, 8), plot}),
        hbox({text("        "), text(xl) | dim}),
        hbox({text("        "), text("scale of the neuron's activation (0× = ablated, 1× = untouched, −1× = inverted)") | dim}),
    });

    Elements rows;
    rows.push_back(text("dose     " + who + "        controls (n=" + std::to_string(d.controls.size()) + ")         outside band?") | bold);
    int outside = 0, comparable = 0;
    for (size_t i = 0; i < nd; ++i) {
        if (i == i1) continue;
        float cmin = 1e9f, cmax = -1e9f;
        int have = 0;
        for (const auto& cv : d.ctrl) {
            if (std::isnan(cv[i]) || std::isnan(cv[i1])) continue;
            const float dd = cv[i] - cv[i1];
            cmin = std::min(cmin, dd);
            cmax = std::max(cmax, dd);
            ++have;
        }
        std::string dl = fmt_f(d.doses[i], 1) + "×";
        dl.resize(9, ' ');
        std::string tv = "…", band = "…", verdict;
        if (!std::isnan(d.curve[i]) && !std::isnan(d.curve[i1])) {
            const float td = d.curve[i] - d.curve[i1];
            tv = fmt_f(100.0 * td, 1) + " pts";
            if (have > 0) {
                band = fmt_f(100.0 * cmin, 1) + " … " + fmt_f(100.0 * cmax, 1) + " pts";
                ++comparable;
                // Outside the band AND at least 1 pt: a 0.01-pt wiggle outside a
                // band of near-zero controls is not an effect.
                if ((td < cmin || td > cmax) && std::fabs(td) >= 0.01f) {
                    ++outside;
                    verdict = "yes";
                } else {
                    verdict = "no";
                }
            }
        }
        tv.resize(17, ' ');
        band.resize(28, ' ');
        rows.push_back(hbox({text(dl), text(tv), text(band) | dim, text(verdict) | (verdict == "yes" ? bold : dim)}));
    }

    Elements head;
    head.push_back(text(who + "  →  P(" + tgt + ") at position " + std::to_string(d.pos) + "   push |a|·‖down‖ = " + fmt_f(d.push, 3).substr(1) +
                        "   controls: same layer, push " +
                        (d.controls.empty() ? "-" : fmt_f(d.controls.back().second, 3).substr(1) + "–" + fmt_f(d.controls.front().second, 3).substr(1))) |
                   bold);
    if (d.running) {
        const float frac = d.total ? static_cast<float>(d.done) / static_cast<float>(d.total) : 0.0f;
        head.push_back(hbox({text("running " + std::to_string(d.done) + "/" + std::to_string(d.total) + "  "), gauge(frac) | size(WIDTH, EQUAL, 40),
                             text("  :cancel to stop · Esc hides this (it keeps running)") | dim}));
    } else {
        std::string v = comparable == 0 ? "no comparable doses" : std::to_string(outside) + " of " + std::to_string(comparable) + " doses outside the control band";
        if (comparable)
            v += outside * 2 > comparable ? "  — the effect is specific to this neuron, not just its size"
                                          : "  — not distinguishable from a same-sized push by other neurons";
        head.push_back(text(v) | bold | color(outside * 2 > comparable ? kOrange : Color::GrayLight));
    }
    if (!d.controls.empty()) {
        float cmaxp = 0;
        for (auto [n, p] : d.controls) cmaxp = std::max(cmaxp, p);
        if (d.push > 1.5f * cmaxp)
            head.push_back(text("⚠ no same-sized controls: this neuron pushes " + fmt_f(d.push / cmaxp, 1).substr(1) +
                                "× harder than the largest control — the band understates what a push this big does") |
                           color(Color::Yellow));
    }
    head.push_back(text("orange: " + who + "   grey: dose-matched controls   dotted: 0×, 1×, baseline level   ·   c lists the controls in LIST") | dim);

    return window(text(" dose–response · Esc closes · P reopens "),
                  vbox({vbox(std::move(head)), separatorLight(), chart, separatorLight(), vbox(std::move(rows))}) | yframe);
}

// ===========================================================================
// Probe and corpus examples
// ===========================================================================

const ActIndex* Session::act_index() {
    if (!act_index_tried_) {
        act_index_tried_ = true;
        try {
            const std::string p = find_index(file_->path(), m_);
            act_index_ = ActIndex::load(p.empty() ? default_index_path(file_->path()) : p, m_);   // the latter throws "no index …"
        } catch (const std::exception& e) {
            act_index_error_ = e.what();
        }
    }
    return act_index_ ? &*act_index_ : nullptr;
}

std::string Session::example_context(const ActExample& e, int before, int after) const {
    if (!act_index_ || !tok_) return "";
    const auto& d = act_index_->doc(e.doc);
    const int p = static_cast<int>(e.pos);
    std::string s;
    for (int i = std::max(0, p - before); i <= std::min(static_cast<int>(d.size()) - 1, p + after); ++i) {
        const std::string pc = tok_->piece(d[static_cast<size_t>(i)]);
        s += i == p ? "«" + pc + "»" : pc;
    }
    return s;
}

Session::ExamplesView Session::examples_view(int layer, int neuron, int k) {
    ExamplesView v{};
    v.layer = layer;
    v.neuron = neuron;
    const ActIndex* ix = act_index();
    if (!ix) {
        v.error = act_index_error_;
        return v;
    }
    try {
        ensure_engine();   // the tokenizer, for the context text
        for (char pole : {'+', '-'}) {
            auto& out = pole == '+' ? v.positive : v.negative;
            int i = 0;
            for (const auto& e : ix->top(layer, neuron, pole)) {
                if (i++ >= k) break;
                out.push_back({e.act, static_cast<int>(e.doc), static_cast<int>(e.pos), tok_->piece(ix->doc(e.doc)[e.pos]), example_context(e)});
            }
        }
    } catch (const std::exception& e) {
        v.error = e.what();
        return v;
    }
    v.present = true;
    v.corpus = ix->corpus;
    v.n_docs = static_cast<int>(ix->n_docs());
    return v;
}

Session::ProbeView Session::probe(const std::vector<std::string>& pos, const std::vector<std::string>& neg, const std::vector<int>& layers, int k) {
    ProbeView v{};
    v.pos = pos;
    v.neg = neg;
    if (pos.empty() || neg.empty()) {
        v.error = "needs at least one POS and one NEG text";
        return v;
    }
    if (busy()) {
        v.error = "busy — :cancel the running job first";
        return v;
    }
    try {
        ensure_engine();
        // Wrapped like the prompt: in chat mode each text is a user turn, measured at
        // the end of the assistant header (where the answer's first token is chosen).
        auto enc = [&](const std::vector<std::string>& ts) {
            std::vector<std::vector<int32_t>> out;
            for (const auto& t : ts) out.push_back(tok_->encode(chat_ ? chat_wrap(m_.arch, t, think_) : t, true, chat_));
            return out;
        };
        v.hits = contrast_probe(*fw_, m_, enc(pos), enc(neg), layers, k, iv_);
    } catch (const std::exception& e) {
        v.error = e.what();
        return v;
    }
    v.present = true;
    probe_pos_ = pos;
    probe_neg_ = neg;
    probe_hits_ = v.hits;
    list_kind_ = ListKind::Probe;
    list_sel_ = 0;
    focus_ = Pane::List;
    list_move(0);
    status_ = "probe: " + std::to_string(pos.size()) + " POS vs " + std::to_string(neg.size()) + " NEG" + (chat_ ? " (chat-wrapped)" : "") +
              " — margin > 0 separates every pair; e shows a hit's corpus examples";
    return v;
}

// ===========================================================================
// Causal trace
// ===========================================================================

void Session::start_trace(Site site) {
    if (tokens_.empty() || !res_) {
        status_ = "set a prompt first (:prompt / :chat), then :corrupt TEXT";
        return;
    }
    if (corrupt_text_.empty()) {
        status_ = "no corrupt prompt — :corrupt TEXT (same token count, the fact changed: France → Italy)";
        return;
    }
    if (busy()) return;
    join_worker();
    try {
        ensure_engine();
        // Same template as the prompt; anything :accept / :append added after the
        // prompt is appended to both, so the positions line up.
        std::vector<int32_t> corrupt = tok_->encode(chat_ ? chat_wrap(m_.arch, corrupt_text_, think_) : corrupt_text_, true, chat_);
        corrupt.insert(corrupt.end(), tokens_.begin() + static_cast<std::ptrdiff_t>(prompt_len_), tokens_.end());
        if (corrupt.size() != tokens_.size()) {
            status_ = "trace: the corrupt prompt is " + std::to_string(corrupt.size()) + " tokens, the prompt " + std::to_string(tokens_.size()) +
                      " — they must match position for position";
            return;
        }
        trace_clean_ = tokens_;
        trace_corrupt_ = corrupt;
    } catch (const std::exception& e) {
        status_ = std::string("trace: ") + e.what();
        return;
    }
    Trace init;
    init.site = site;
    init.n_layer = m_.n_layer;
    init.n_tokens = static_cast<int>(tokens_.size());
    init.total = m_.n_layer * init.n_tokens;
    init.p.assign(static_cast<size_t>(init.total), std::numeric_limits<float>::quiet_NaN());
    trace_ = init;
    trace_running_ = true;
    cancel_ = false;
    overlay_ = Overlay::Trace;
    status_ = "trace (" + std::string(site_name(site)) + "): " + std::to_string(init.total) + " patched runs…";

    const auto clean = trace_clean_, corrupt = trace_corrupt_;
    const Interventions iv = iv_;
    const Forward* fw = fw_.get();
    const ModelLayout* m = &m_;
    const int target = target_pinned_ ? target_ : -1;
    auto job = [this, clean, corrupt, iv, fw, m, site, target](auto deliver) {
        try {
            // A copy per progress step: n_layer × T floats, small next to a forward pass.
            causal_trace(*fw, *m, clean, corrupt, iv, site, target, [&](const Trace& t) { deliver([this, t] { trace_ = t; }); }, &cancel_);
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            deliver([this, msg] { status_ = "trace: " + msg; });
        }
    };
    auto finish = [this] {
        trace_running_ = false;
        if (!trace_) return;
        const auto [bl, bt] = trace_->best();
        if (cancel_) status_ = "trace cancelled";
        else if (bl >= 0)
            status_ = "trace done — best: L" + std::to_string(bl) + " pos " + std::to_string(bt) + " restores " + fmt_f(100.0 * trace_->restored(bl, bt), 0).substr(1) +
                      "% of P(" + quote_printable(token_text(trace_->target), 12) + ")  · T reopens the map";
    };
    if (post) {
        auto p = post;
        worker_ = std::thread([job, finish, p] {
            job([&p](std::function<void()> f) { p(std::move(f)); });
            p(finish);
        });
    } else {
        job([](std::function<void()> f) { f(); });
        finish();
    }
}

Element Session::render_trace() {
    if (!trace_) return window(text(" causal trace "), text("no trace yet — :corrupt TEXT, then :trace") | dim);
    const Trace& t = *trace_;
    Elements head;
    head.push_back(text(std::string("patching the clean run's ") + site_name(t.site) + " into the corrupt run → P(" + quote_printable(token_text(t.target), 14) +
                        ") at the last token:  clean " + fmt_f(100.0 * t.p_clean, 1).substr(1) + "%   corrupt " + fmt_f(100.0 * t.p_corrupt, 1).substr(1) +
                        "%   corrupt prompt " + quote_printable(corrupt_text_, 40)) |
                   bold);
    if (trace_running_) {
        const float frac = t.total ? static_cast<float>(t.done) / static_cast<float>(t.total) : 0.0f;
        head.push_back(hbox({text("running " + std::to_string(t.done) + "/" + std::to_string(t.total) + "  "), gauge(frac) | size(WIDTH, EQUAL, 40),
                             text("  :cancel to stop · Esc hides this (it keeps running)") | dim}));
    } else if (const auto [bl, bt] = t.best(); bl >= 0) {
        head.push_back(text("best: L" + std::to_string(bl) + " pos " + std::to_string(bt) + " restores " + fmt_f(100.0 * t.restored(bl, bt), 1).substr(1) + "%") |
                       color(kOrange));
    }
    if (std::fabs(t.p_clean - t.p_corrupt) < 1e-6f && t.done > 0)
        head.push_back(text("the corruption doesn't change P(target) — nothing to restore; pick a corruption that changes the answer") | color(Color::Yellow));

    // The map: one row per position, one column per layer.
    const int cw = std::max(3, std::min(5, 120 / std::max(1, t.n_layer)));
    Elements hdr{text("pos  token") | size(WIDTH, EQUAL, 30)};
    for (int l = 0; l < t.n_layer; ++l) hdr.push_back(text(std::to_string(l)) | size(WIDTH, EQUAL, cw) | dim);
    Elements grid{hbox(std::move(hdr))};
    for (int p = 0; p < t.n_tokens; ++p) {
        const bool d = std::find(t.differ.begin(), t.differ.end(), p) != t.differ.end();
        std::string lab = std::to_string(p) + (d ? "* " : "  ") + tok_label(token_text(trace_clean_[static_cast<size_t>(p)]), 12);
        if (d) lab += "/" + tok_label(token_text(trace_corrupt_[static_cast<size_t>(p)]), 10);
        Elements row{text(fit_cells(lab, 29)) | size(WIDTH, EQUAL, 30) | (d ? bold : dim)};
        for (int l = 0; l < t.n_layer; ++l) {
            const float r = t.restored(l, p);
            if (std::isnan(r)) {
                row.push_back(text(" ·") | size(WIDTH, EQUAL, cw) | dim);
                continue;
            }
            const int pct = static_cast<int>(std::lround(100.0f * r));
            Element c = text(cw >= 4 ? std::to_string(pct) : (pct >= 50 ? "█" : pct >= 15 ? "▓" : pct >= 5 ? "░" : " ")) | size(WIDTH, EQUAL, cw) |
                        color(Color::White) | bgcolor(diverging(std::clamp(r, -1.0f, 1.0f)));
            row.push_back(c);
        }
        grid.push_back(hbox(std::move(row)));
    }
    head.push_back(text("cells: % of the clean−corrupt gap in P(target) that patching that one (layer, pos) restores · * = the prompts differ") | dim);
    return window(text(" causal trace · Esc closes · T reopens "), vbox({vbox(std::move(head)), separatorLight(), vbox(std::move(grid)) | yframe}));
}

Session::TraceView Session::trace_view() {
    TraceView v{};
    v.present = trace_.has_value();
    if (!v.present) return v;
    const Trace& t = *trace_;
    v.site = site_name(t.site);
    v.corrupt = corrupt_text_;
    v.target_id = t.target;
    v.target_text = t.target >= 0 ? token_text(t.target) : "";
    v.p_clean = t.p_clean;
    v.p_corrupt = t.p_corrupt;
    v.n_layer = t.n_layer;
    v.n_tokens = t.n_tokens;
    for (int32_t id : trace_clean_) v.clean_tokens.push_back(token_text(id));
    for (int32_t id : trace_corrupt_) v.corrupt_tokens.push_back(token_text(id));
    v.differ = t.differ;
    v.restored.assign(static_cast<size_t>(t.n_layer), std::vector<float>(static_cast<size_t>(t.n_tokens)));
    for (int l = 0; l < t.n_layer; ++l)
        for (int p = 0; p < t.n_tokens; ++p) v.restored[static_cast<size_t>(l)][static_cast<size_t>(p)] = t.restored(l, p);
    const auto [bl, bt] = t.best();
    v.best_layer = bl;
    v.best_pos = bt;
    v.best = bl >= 0 ? t.restored(bl, bt) : 0.0f;
    v.done = t.done;
    v.total = t.total;
    v.running = trace_running_;
    return v;
}

// ===========================================================================
// Attention overlay
// ===========================================================================

const Session::AttnCache& Session::attention() {
    if (attn_ && attn_->layer == layer_ && attn_->pos == pos_) return *attn_;
    AttnCache a;
    a.layer = layer_;
    a.pos = pos_;
    a.linear = m_.layers[static_cast<size_t>(layer_)].kind == LayerKind::linear_attention;
    if (!a.linear && res_ && !tokens_.empty()) {
        try {
            ensure_engine();
            RunOptions ro;
            ro.attn_probs_layer = layer_;
            ro.attn_probs_pos = pos_;
            ro.cap_lo = ro.cap_hi = 0;   // nothing else is needed
            // Causality: the weights from query pos_ only depend on tokens[0, pos_].
            const int T = pos_ + 1;
            // From the layer when the residual entering it is captured at every
            // position up to pos_ (no window); else the whole prefix.
            const bool have_in = res_->cap_lo == 0 && res_->cap_n >= T && (layer_ == 0 ? !res_->embed.empty() : res_->layer_resid.count(layer_ - 1) > 0);
            ForwardResult r;
            if (have_in) {
                const std::vector<float>& src = layer_ == 0 ? res_->embed : res_->layer_resid.at(layer_ - 1);
                const std::vector<float> x(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(T) * m_.n_embd);
                r = fw_->run_from(layer_, x, T, iv_, ro);
            } else {
                const std::vector<int32_t> prefix(tokens_.begin(), tokens_.begin() + T);
                r = fw_->run(prefix, iv_, ro);
            }
            a.n_kv = r.attn_n_kv;
            a.n_head = r.attn_n_head;
            a.probs = std::move(r.attn_probs);
        } catch (const std::exception& e) {
            a.error = e.what();
        }
    }
    attn_ = std::move(a);
    return *attn_;
}

Session::AttentionView Session::attention_view(int layer, int pos, int k) {
    AttentionView v;
    v.present = false;
    v.layer = layer;
    v.pos = pos;
    v.linear = false;
    if (!res_) {
        v.error = "no run yet — run a prompt first";
        return v;
    }
    if (layer < 0 || layer >= m_.n_layer) {
        v.error = "no layer " + std::to_string(layer);
        return v;
    }
    select_layer(layer);
    set_pos(pos);
    const AttnCache& a = attention();
    v.pos = pos_;
    v.linear = a.linear;
    if (!a.error.empty()) {
        v.error = a.error;
        return v;
    }
    v.present = true;
    const std::vector<BlameItem> hb = blame_ ? blame_->layer_heads(layer_) : std::vector<BlameItem>{};
    const int nh = n_heads(m_, layer_);
    for (int h = 0; h < nh; ++h) {
        AttnHeadView hv;
        hv.head = h;
        hv.blame = static_cast<size_t>(h) < hb.size() ? hb[static_cast<size_t>(h)].contrib : 0.0f;
        if (!a.linear && h < a.n_head) {
            const float* w = a.probs.data() + static_cast<size_t>(h) * static_cast<size_t>(a.n_kv);
            std::vector<int> idx;
            for (int i = 0; i <= pos_ && i < a.n_kv; ++i) idx.push_back(i);
            const size_t kk = std::min<size_t>(static_cast<size_t>(std::max(0, k)), idx.size());
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk), idx.end(), [&](int x, int y) { return w[x] > w[y]; });
            for (size_t i = 0; i < kk; ++i) hv.top.push_back({idx[i], token_text(tokens_[static_cast<size_t>(idx[i])]), w[idx[i]]});
        }
        v.heads.push_back(std::move(hv));
    }
    return v;
}

Element Session::render_attention() {
    const std::string who = "L" + std::to_string(layer_);
    if (!res_) return window(text(" attention "), text("no run yet — :prompt TEXT first") | dim);
    const AttnCache& a = attention();
    const std::string q = quote_printable(token_text(tokens_[static_cast<size_t>(pos_)]), 14);
    const int nh = n_heads(m_, layer_);
    const std::vector<BlameItem> hb = blame_ ? blame_->layer_heads(layer_) : std::vector<BlameItem>{};
    float bmax = 1e-6f;
    for (const auto& h : hb) bmax = std::max(bmax, std::fabs(h.contrib));
    attn_head_ = std::clamp(attn_head_, 0, std::max(0, nh - 1));

    Elements head;
    head.push_back(text(who + (a.linear ? " (linear attention)" : "") + " · from pos " + std::to_string(pos_) + " " + q + " · blame on " +
                        quote_printable(token_text(target_), 14)) | bold);
    if (!a.error.empty()) head.push_back(text("attention: " + a.error) | color(kOrange));

    // Left: one row per head — its blame and, for softmax layers, its strongest keys.
    Elements rows;
    for (int h = 0; h < nh; ++h) {
        const float c = static_cast<size_t>(h) < hb.size() ? hb[static_cast<size_t>(h)].contrib : 0.0f;
        std::string tops;
        if (!a.linear && h < a.n_head) {
            const float* w = a.probs.data() + static_cast<size_t>(h) * static_cast<size_t>(a.n_kv);
            std::vector<int> idx;
            for (int i = 0; i <= pos_ && i < a.n_kv; ++i) idx.push_back(i);
            const size_t kk = std::min<size_t>(3, idx.size());
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk), idx.end(), [&](int x, int y) { return w[x] > w[y]; });
            for (size_t i = 0; i < kk; ++i)
                tops += (i ? "  " : "") + tok_label(token_text(tokens_[static_cast<size_t>(idx[i])]), 10) + "@" + std::to_string(idx[i]) + " " +
                        fmt_f(w[idx[i]], 2).substr(1);
        }
        std::string nm = "H" + std::to_string(h);
        nm.resize(5, ' ');
        Element line = hbox({text(h == attn_head_ ? "▶ " : "  "), text(nm), text(hbar(c, bmax, 10)) | color(sign_color(c)) | size(WIDTH, EQUAL, 11),
                             text(fmt_f(c, 3) + "  ") | size(WIDTH, EQUAL, 9), text(fit_cells(tops, 56)) | dim});
        if (h == attn_head_) line = line | inverted | focus;
        rows.push_back(line);
    }

    // Right: the selected head's weights over every earlier token, as a heat row.
    Elements detail;
    if (a.linear) {
        detail.push_back(paragraph("Gated DeltaNet keeps a recurrent state instead of attending over tokens, so there is no pattern to show. "
                                   "The per-head blame on the left is exact: each value head's slice of ssm_out's input.") | dim);
    } else if (attn_head_ < a.n_head && a.error.empty()) {
        const float* w = a.probs.data() + static_cast<size_t>(attn_head_) * static_cast<size_t>(a.n_kv);
        float wmax = 1e-6f;
        for (int i = 0; i <= pos_ && i < a.n_kv; ++i) wmax = std::max(wmax, w[i]);
        Elements cells;
        for (int i = std::max(0, pos_ - 200); i <= pos_ && i < a.n_kv; ++i) {
            const float x = w[i] / wmax;
            cells.push_back(hbox({text(tok_label(token_text(tokens_[static_cast<size_t>(i)]), 12)) | color(Color::White) | bgcolor(diverging(x)),
                                  text(" ")}));
        }
        detail.push_back(text("H" + std::to_string(attn_head_) + " — weight on each earlier token (brighter = more; max " + fmt_f(wmax, 2).substr(1) + ")") | bold);
        detail.push_back(hflow(std::move(cells)));
        std::vector<int> idx;
        for (int i = 0; i <= pos_ && i < a.n_kv; ++i) idx.push_back(i);
        const size_t kk = std::min<size_t>(12, idx.size());
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk), idx.end(), [&](int x, int y) { return w[x] > w[y]; });
        detail.push_back(separatorLight());
        for (size_t i = 0; i < kk; ++i) {
            const int t = idx[i];
            detail.push_back(hbox({text(std::to_string(t)) | size(WIDTH, EQUAL, 6), text(tok_label(token_text(tokens_[static_cast<size_t>(t)]), 14)) | size(WIDTH, EQUAL, 16),
                                   text(hbar(w[t], 1.0f, 20)) | color(sign_color(1)) | size(WIDTH, EQUAL, 21), text(fmt_f(w[t], 3).substr(1)) | dim}));
        }
    }
    head.push_back(text("left: each head's share of the target logit (exact, blame.h) and where it looks · j/k head · h/l token · PgUp/Dn layer") | dim);
    return window(text(" attention · " + who + " · Esc closes · A toggles "),
                  vbox({vbox(std::move(head)), separatorLight(),
                        hbox({vbox(std::move(rows)) | yframe | size(WIDTH, EQUAL, 86), separator(), vbox(std::move(detail)) | flex})}));
}

// ===========================================================================
// Terminal front ends
// ===========================================================================

std::string render_to_string(Session& s, int width, int height, bool color_out) {
    auto screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
    Render(screen, s.render());
    std::string out = screen.ToString();
    if (!color_out) {
        static const std::regex ansi("\x1b\\[[0-9;?]*[A-Za-z]");
        out = std::regex_replace(out, ansi, "");
    }
    return out;
}

int run_tui(Session& s) {
    auto app = App::Fullscreen();
    std::string cmd;
    bool cmd_mode = false;
    int cursor = 0;
    int hist_pos = -1;   // -1: editing a fresh line

    InputOption io;
    io.content = &cmd;
    io.cursor_position = &cursor;
    io.multiline = false;
    auto input = Input(io);
    auto root = Renderer(input, [&] {
        s.cmdline = cmd_mode ? &cmd : nullptr;
        s.cmd_cursor = &cursor;
        return s.render();
    });

    // Workers hand results back to the UI thread through this.
    s.post = [&app](std::function<void()> f) {
        app.Post(std::move(f));
        app.PostEvent(Event::Custom);
    };

    auto open_cmd = [&](const std::string& prefill) {
        cmd_mode = true;
        cmd = prefill;
        cursor = static_cast<int>(cmd.size());
    };

    root |= CatchEvent([&](Event e) {
        if (cmd_mode) {
            if (e == Event::Return) {
                const std::string line = cmd;
                if (!line.empty() && (s.history.empty() || s.history.back() != line)) s.history.push_back(line);
                cmd.clear();
                cursor = 0;
                hist_pos = -1;
                cmd_mode = false;
                if (!s.exec(line)) app.Exit();
                return true;
            }
            if (e == Event::Escape) {
                cmd.clear();
                cursor = 0;
                hist_pos = -1;
                cmd_mode = false;
                return true;
            }
            if (e == Event::Tab) {
                s.tab(cmd);
                cursor = static_cast<int>(cmd.size());
                return true;
            }
            if (e == Event::ArrowUp || e == Event::ArrowDown) {
                const int n = static_cast<int>(s.history.size());
                if (n == 0) return true;
                if (e == Event::ArrowUp) hist_pos = hist_pos < 0 ? n - 1 : std::max(0, hist_pos - 1);
                else hist_pos = (hist_pos < 0 || hist_pos >= n - 1) ? -1 : hist_pos + 1;
                cmd = hist_pos < 0 ? "" : s.history[static_cast<size_t>(hist_pos)];
                cursor = static_cast<int>(cmd.size());
                return true;
            }
            return false;   // let the Input component edit the line
        }
        if (e == Event::Character(':')) { open_cmd(""); return true; }
        if (e == Event::Character('/')) { open_cmd("find \""); return true; }
        std::string k;
        if (e == Event::ArrowUp) k = "up";
        else if (e == Event::ArrowDown) k = "down";
        else if (e == Event::ArrowLeft) k = "left";
        else if (e == Event::ArrowRight) k = "right";
        else if (e == Event::PageUp) k = "pgup";
        else if (e == Event::PageDown) k = "pgdn";
        else if (e == Event::Tab) k = "tab";
        else if (e == Event::TabReverse) k = "shift_tab";
        else if (e == Event::Return) k = "enter";
        else if (e == Event::Escape) k = "escape";
        else if (e == Event::Delete) k = "delete";
        else if (e == Event::Backspace) k = "backspace";
        else if (e == Event::Custom) return true;   // redraw request from a worker
        else if (e.is_character()) k = e.character();
        else return true;   // swallow everything else (mouse etc.)
        if (!s.key(k)) app.Exit();
        return true;
    });

    app.Loop(root);
    s.post = nullptr;
    return 0;
}

// ===========================================================================
// Machine-readable views for the `mcp` front end (docs/mcp.md). Each mirrors
// what a render_* method reads; none builds an FTXUI element.
// ===========================================================================

namespace {
const char* pane_name(Pane p) {
    switch (p) {
        case Pane::Graph: return "graph";
        case Pane::List: return "list";
        case Pane::Neuron: return "neuron";
        case Pane::Bytes: return "bytes";
        case Pane::Patches: return "patches";
    }
    return "?";
}
const char* overlay_name(Overlay o) {
    switch (o) {
        case Overlay::None: return "none";
        case Overlay::Help: return "help";
        case Overlay::Model: return "model";
        case Overlay::Plot: return "plot";
        case Overlay::Attn: return "attn";
        case Overlay::Trace: return "trace";
    }
    return "?";
}
const char* slice_name(Slice s) {
    switch (s) {
        case Slice::Gate: return "gate";
        case Slice::Up: return "up";
        case Slice::Down: return "down";
    }
    return "?";
}
const char* list_kind_name(ListKind k) {
    switch (k) {
        case ListKind::Blame: return "blame";
        case ListKind::Fires: return "fires";
        case ListKind::Find: return "find";
        case ListKind::Controls: return "controls";
        case ListKind::Labels: return "labels";
        case ListKind::Probe: return "probe";
        case ListKind::Examples: return "examples";
    }
    return "?";
}
const char* blame_kind_name(BlameItem::Kind k) {
    switch (k) {
        case BlameItem::Embed: return "embed";
        case BlameItem::Attn: return "attn";
        case BlameItem::Neuron: return "neuron";
        case BlameItem::Head: return "head";
    }
    return "?";
}
}  // namespace

Session::Snapshot Session::snapshot() {
    Snapshot v;
    v.arch = m_.arch;
    v.model_name = m_.name;
    v.n_layer = m_.n_layer;
    v.n_embd = m_.n_embd;
    v.n_ff = m_.n_ff;
    v.n_head = m_.n_head;
    v.n_head_kv = m_.n_head_kv;
    v.n_vocab = m_.n_vocab;
    v.n_ctx_train = m_.n_ctx_train;
    v.tied_embd = (m_.output == nullptr);

    v.pos = pos_;
    v.layer = layer_;
    v.neuron = neuron_;
    v.focus = pane_name(focus_);
    v.overlay = overlay_name(overlay_);
    v.slice = slice_name(slice_);
    v.chat = chat_;
    v.think = think_;

    v.has_run = res_.has_value();
    v.prompt_len = static_cast<int>(prompt_len_);
    for (size_t i = 0; i < tokens_.size(); ++i)
        v.tokens.push_back({static_cast<int>(i), tokens_[i], token_text(tokens_[i]), i >= prompt_len_});

    v.has_target = res_ && target_ >= 0;
    if (v.has_target) {
        const float* lg = res_->logits_at(pos_);
        const int nv = res_->n_vocab;
        v.target.id = target_;
        v.target.text = token_text(target_);
        v.target.pinned = target_pinned_;
        v.target.prob = prob_of(lg, nv, target_);
        int above = 0;
        const float t = lg[target_];
        for (int i = 0; i < nv; ++i) above += lg[i] > t;
        v.target.rank = above + 1;
    }

    for (const auto& p : patches_) {
        PatchView pv;
        pv.layer = p.layer;
        pv.neuron = p.neuron;
        pv.op = p.op == EditOp::Scale ? "scale" : p.op == EditOp::Add ? "add" : p.op == EditOp::Set ? "set" : p.op == EditOp::RankOne ? "rank-one" : "steer";
        pv.scale = p.op == EditOp::Scale ? p.k : 1.0f;
        pv.value = p.value;
        if (p.op == EditOp::Steer && units_) {
            try {
                pv.value = units_->value(p.layer, p.neuron, p.dose, p.pole);
            } catch (...) {
            }
        }
        pv.dose = p.dose;
        pv.pole = p.pole ? std::string(1, p.pole) : std::string();
        pv.scope = p.scope == EditScope::Prompt ? "prompt" : "all";
        pv.text = edit_text(p);
        pv.portability = portability_name(portability(p));
        pv.labels = label_brief(p.layer, p.neuron);
        v.patches.push_back(std::move(pv));
    }
    v.watches = watch_;
    v.status = status_;
    v.busy = busy();
    return v;
}

Session::ListView Session::list_view() {
    ListView v;
    v.kind = list_kind_name(list_kind_);
    v.selected = list_sel_;
    for (const auto& r : list_rows()) {
        ListRowView rv;
        rv.section = r.section;
        rv.label = r.label;
        rv.layer = r.layer;
        rv.neuron = r.neuron;
        rv.value = r.value;
        rv.detail = r.detail;
        rv.labels = r.note;
        if (!r.section) rv.kind = blame_kind_name(r.kind);
        v.rows.push_back(std::move(rv));
    }
    return v;
}

Session::BlameView Session::blame_view(int k, bool toward) {
    BlameView v;
    v.present = blame_.has_value();
    if (!v.present) return v;
    v.pos = blame_->pos;
    v.target_id = blame_->target;
    v.target_text = token_text(blame_->target);
    v.logit = blame_->logit;
    v.prob = blame_->prob;
    v.rank = blame_->rank;
    v.embed = blame_->embed;
    v.sum_attn = blame_->sum_attn;
    v.sum_neurons = blame_->sum_neurons;
    v.unexplained = blame_->unexplained;
    for (const auto& it : blame_->top(k, toward))
        v.items.push_back({blame_kind_name(it.kind), it.layer, it.neuron, it.act, it.dot, it.contrib});
    return v;
}

Session::NeuronInfoView Session::neuron_info(int layer, int neuron) {
    NeuronInfoView v;
    v.layer = layer;
    v.neuron = neuron;
    v.present = false;
    NeuronView nv;
    try {
        nv = read_neuron(m_, layer, neuron);
    } catch (const std::exception&) {
        return v;
    }
    v.present = true;
    auto fp = [](const Footprint& f) -> FootprintView {
        return {f.first_byte, f.last_byte, f.n_blocks, f.contiguous, f.shared_with};
    };
    v.up = fp(nv.up_fp);
    v.gate = fp(nv.gate_fp);
    v.down = fp(nv.down_fp);
    if (labels_) v.labels = labels_->at(layer, neuron);

    // Static lens of the down column (cache keyed by (layer, neuron), as the TUI does).
    const auto key = std::make_pair(layer, neuron);
    auto it = lens_cache_.find(key);
    if (it == lens_cache_.end()) {
        try {
            std::vector<float> w = nv.down;
            apply_write_gain(m_, layer, w);
            it = lens_cache_.emplace(key, logit_lens(*file_, m_, w, 8)).first;
        } catch (const std::exception&) {
            it = lens_cache_.end();
        }
    }
    if (it != lens_cache_.end()) {
        for (const auto& t : it->second.top) v.promotes.push_back({t.id, t.text, t.logit});
        for (const auto& t : it->second.bottom) v.suppresses.push_back({t.id, t.text, t.logit});
    }

    // Activation + |a| rank within the layer at the cursor position.
    const float* a = res_ ? res_->act(layer, pos_) : nullptr;
    v.has_act = a != nullptr;
    if (v.has_act) {
        const size_t F = static_cast<size_t>(m_.n_ff);
        const float av = a[static_cast<size_t>(neuron)];
        v.act_here = av;
        int rank = 1;
        for (size_t n = 0; n < F; ++n) rank += std::fabs(a[n]) > std::fabs(av);
        v.act_rank = rank;
        v.act_rank_of = static_cast<int>(F);
    }
    return v;
}

Session::StateView Session::state_view() {
    StateView v;
    v.present = res_.has_value();
    if (!v.present) return v;
    update_state_lens();
    v.pos = pos_;
    v.layer = layer_;
    const int nv = res_->n_vocab;
    for (auto [id, p] : top_probs(res_->logits_at(pos_), nv, 5)) v.next_patched.push_back({id, token_text(id), p});
    v.has_base = base_.has_value();
    if (v.has_base)
        for (auto [id, p] : top_probs(base_->logits_at(pos_), base_->n_vocab, 5)) v.next_base.push_back({id, token_text(id), p});
    v.target_by_layer = target_by_layer_;
    for (auto [id, p] : state_lens_.top) v.lens_after_layer.push_back({id, token_text(id), p});
    v.lens_target_prob = state_lens_.target_prob;
    v.lens_target_rank = state_lens_.target_rank;
    v.gen_base = gen_base_;
    v.gen_patched = gen_patched_;
    return v;
}

Session::DoseView Session::dose_view() {
    DoseView v;
    v.present = dose_.has_value();
    if (!v.present) return v;
    const DoseSweep& d = *dose_;
    v.layer = d.layer;
    v.neuron = d.neuron;
    v.pos = d.pos;
    v.target_id = d.target;
    v.target_text = token_text(d.target);
    v.push = d.push;
    v.doses = d.doses;
    v.curve = d.curve;
    v.controls = d.controls;
    v.ctrl = d.ctrl;
    v.done = d.done;
    v.total = d.total;
    v.running = d.running;
    v.seconds = d.seconds;

    // Verdict: same rule as render_dose (architecture.md §7). The ×1 dose is the
    // reference; a tested Δ counts as "outside" only if it leaves the controls'
    // Δ band and is at least 1 pt (0.01 in probability).
    const size_t nd = d.doses.size();
    size_t i1 = 0;
    for (size_t i = 0; i < nd; ++i)
        if (d.doses[i] == 1.0f) i1 = i;
    DoseVerdictView& vd = v.verdict;
    vd.outside = 0;
    vd.comparable = 0;
    for (size_t i = 0; i < nd; ++i) {
        if (i == i1) continue;
        DosePointView pt;
        pt.dose = d.doses[i];
        pt.tested_delta = std::numeric_limits<float>::quiet_NaN();
        pt.band_min = pt.band_max = std::numeric_limits<float>::quiet_NaN();
        pt.comparable = false;
        pt.outside = false;
        if (!std::isnan(d.curve[i]) && !std::isnan(d.curve[i1])) {
            const float td = d.curve[i] - d.curve[i1];
            pt.tested_delta = td;
            float cmin = 1e9f, cmax = -1e9f;
            int have = 0;
            for (const auto& cv : d.ctrl) {
                if (std::isnan(cv[i]) || std::isnan(cv[i1])) continue;
                const float dd = cv[i] - cv[i1];
                cmin = std::min(cmin, dd);
                cmax = std::max(cmax, dd);
                ++have;
            }
            if (have > 0) {
                pt.band_min = cmin;
                pt.band_max = cmax;
                pt.comparable = true;
                ++vd.comparable;
                if ((td < cmin || td > cmax) && std::fabs(td) >= 0.01f) {
                    pt.outside = true;
                    ++vd.outside;
                }
            }
        }
        vd.points.push_back(pt);
    }
    vd.specific = vd.comparable > 0 && vd.outside * 2 > vd.comparable;
    float cmaxp = 0;
    for (auto [n, p] : d.controls) cmaxp = std::max(cmaxp, p);
    vd.understated = !d.controls.empty() && cmaxp > 0 && d.push > 1.5f * cmaxp;
    vd.understate_factor = cmaxp > 0 ? d.push / cmaxp : 0.0f;
    return v;
}

}  // namespace neuronaut
