// main.cpp — neuronaut entry point.
//
//   neuronaut <model.gguf>                    interactive TUI
//   neuronaut info <model.gguf>               architecture summary + warnings
//   neuronaut meta <model.gguf>               every metadata key
//   neuronaut tensors <model.gguf>            tensor directory
//   neuronaut neuron <model.gguf> L N         one neuron's gate/up/down, stats, footprint
//   neuronaut lens <model.gguf> L N [k]       static logit lens of neuron N's output
//   neuronaut find <model.gguf> TEXT [k]      neurons whose output points at token TEXT
//   neuronaut tokenize <model.gguf> TEXT [--chat|--think]  token ids + pieces (libllama), BOS per the vocab
//   neuronaut oracle <model.gguf> PROMPT [--exact] [--gen N] [EDITS]... [--against F] [--cvec F]
//                                             compare our forward pass with llama.cpp, every position
//   neuronaut export <model.gguf> EDITS... [-o OUT.gguf] [--cvec OUT.cvec.gguf] [--skip-unportable]
//                                             edits → a patched GGUF + a llama.cpp control vector (src/export.h)
//   neuronaut run <model.gguf> PROMPT [-n N] [--chat|--think] [--reply TEXT] [EDITS]... [--watch L.N]... [--save NAME [--note TEXT]]
//                                             greedy generation with interventions; watched activations
//   neuronaut blame <model.gguf> PROMPT [--pos N] [--target TEXT] [--vs TEXT] [--chat|--think] [--reply TEXT] [EDITS]... [-k K]
//                                             direct logit attribution for one position/token
//   neuronaut trace <model.gguf> CLEAN CORRUPT [--site resid|attn|mlp] [--target TEXT] [--chat|--think] [EDITS]...
//                                             activation patching: where the fact that separates two prompts lives
//   neuronaut probe <model.gguf> --pos TEXT... --neg TEXT... [--layers L,L,...] [-k K] [--chat]
//                                             neurons that separate POS from NEG at the last token (src/probe.h)
//   neuronaut rome <model.gguf> --subject S --prompt "{} …" --target T --layer L [--essence "{} is a"]... [--save NAME]
//                                             rewrite one fact with a rank-one weight edit (src/rome.h)
//   neuronaut index <model.gguf> CORPUS [--docs N] [--max-tokens T] [-k K] [-o FILE]
//                                             build the top-activating-examples index (src/actindex.h)
//   neuronaut examples <model.gguf> L.N [-k K] [--index FILE]   a neuron's top corpus examples
//   neuronaut render <model.gguf> [--size WxH] [--color] [--cmd "..."]... [--key K]... [--type TEXT] [--tab]...
//                                             draw one TUI frame to stdout (headless)
//   neuronaut edits <model.gguf> list | show NAME|FILE   saved edit sets (src/edits.h)
//   neuronaut labels <model.gguf> list|search|show|set|rm|import|export|path ...
//                                             the neuron-label database (src/labels.h)
//
// The non-interactive subcommands exist so everything the TUI shows is also
// scriptable and diffable — and so it can be checked without a terminal.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "format.h"
#include "edits.h"
#include "labels.h"
#include "steer.h"
#include "gguf_file.h"
#include "lens.h"
#include "model_layout.h"
#include "neuron.h"
#include "blame.h"
#include "forward.h"
#include "oracle.h"
#include "trace.h"
#include "export.h"
#include "probe.h"
#include "actindex.h"
#include "rome.h"
#include <chrono>
#include "tui.h"

using namespace neuronaut;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  neuronaut <model.gguf>\n"
                 "  neuronaut info|meta|tensors <model.gguf>\n"
                 "  neuronaut neuron <model.gguf> LAYER NEURON\n"
                 "  neuronaut lens <model.gguf> LAYER NEURON [K]\n"
                 "  neuronaut find <model.gguf> TEXT [K]\n"
                 "  neuronaut tokenize <model.gguf> TEXT [--chat|--think]\n"
                 "  neuronaut oracle <model.gguf> PROMPT [--exact] [--gen N] [EDITS]... [--against PATCHED.gguf] [--cvec CV.gguf]\n"
                 "  neuronaut export <model.gguf> EDITS... [-o OUT.gguf] [--cvec OUT.cvec.gguf] [--skip-unportable]\n"
                 "  neuronaut blame <model.gguf> PROMPT [--pos N] [--target TEXT] [--vs TEXT] [--chat|--think] [--reply TEXT] [EDITS]... [-k K]\n"
                 "  neuronaut run <model.gguf> PROMPT [-n N] [--chat|--think] [--reply TEXT] [EDITS]... [--watch L.N]... [--save NAME [--note TEXT]]\n"
                 "  neuronaut trace <model.gguf> CLEAN CORRUPT [--site resid|attn|mlp] [--target TEXT] [--chat|--think] [EDITS]...\n"
                 "  neuronaut probe <model.gguf> --pos TEXT... --neg TEXT... [--layers L,L,...] [-k K] [--chat]\n"
                 "  neuronaut rome <model.gguf> --subject S (--prompt|--chat-prompt \"…{}…\" --target T)... --layer L [--essence \"{} is a\"]...\n"
                 "                 [--prefix TEXT]... [--steps N] [--lr X] [--kl X] [--corpus FILE] [--cov-docs N] [--save NAME [--note TEXT]]\n"
                 "  neuronaut index <model.gguf> CORPUS.jsonl|.txt [--docs N] [--max-tokens T] [-k K] [-o FILE]\n"
                 "  neuronaut examples <model.gguf> L.N [-k K] [--index FILE]\n"
                 "  neuronaut render <model.gguf> [--size WxH] [--color] [--cmd CMD]...\n"
                 "  neuronaut mcp <model.gguf>   (MCP JSON-RPC server on stdio; see docs/mcp.md)\n"
                 "  EDITS: --ablate L.N | --invert L.N | --scale L.N=K | --set L.N=V | --add L.N=V[@prompt]\n"
                 "         --steer L.N+=DOSE[@prompt] | --steer L.N-=DOSE[@prompt] | --edits NAME|FILE   (src/edits.h)\n"
                 "  neuronaut edits <model.gguf> list | show NAME|FILE\n"
                 "  neuronaut labels <model.gguf> list [--source S] [--tag T] [--layer L]\n"
                 "  neuronaut labels <model.gguf> search QUERY        (terms AND; #tag, src:NAME)\n"
                 "  neuronaut labels <model.gguf> show L.N\n"
                 "  neuronaut labels <model.gguf> set L.N[+|-] TEXT [--tag T]... [--note N] [--source S] [--author A]\n"
                 "  neuronaut labels <model.gguf> rm L.N[+|-] [--source S]\n"
                 "  neuronaut labels <model.gguf> import FILE [--format json|jsonl|csv|tsv] [--map OURS=THEIRS]...\n"
                 "                                     [--source S] [--replace-source] [--dry-run] [--force]\n"
                 "  neuronaut labels <model.gguf> export FILE|- [--format json|jsonl|csv|tsv] [--source S] [--tag T]\n"
                 "  neuronaut labels <model.gguf> path\n");
    return 2;
}

void print_vec_summary(const char* label, const std::vector<float>& v, const Footprint& fp) {
    const VecStats s = stats(v);
    std::printf("%-5s ‖v‖=%.4f min=%+.4f max=%+.4f mean|v|=%.4f\n", label, s.l2, s.min, s.max, s.mean_abs);
    std::printf("      first 8:");
    for (size_t i = 0; i < 8 && i < v.size(); ++i) std::printf(" %+.4f", v[i]);
    std::printf("\n      top |v|:");
    for (int i : top_abs(v, 8)) std::printf(" e%d=%+.4f", i, v[static_cast<size_t>(i)]);
    std::printf("\n      bytes [0x%llx, 0x%llx) %lld blocks × %zu B, %s, shared_with=%lld\n",
                static_cast<unsigned long long>(fp.first_byte), static_cast<unsigned long long>(fp.last_byte),
                static_cast<long long>(fp.n_blocks), fp.block_bytes, fp.contiguous ? "contiguous" : "strided",
                static_cast<long long>(fp.shared_with));
}

}  // namespace

// "15.7279" -> (15, 7279)
bool parse_ln(const std::string& s, int& l, int& n) {
    return std::sscanf(s.c_str(), "%d.%d", &l, &n) == 2;
}

// --exact: additionally require every position's logits to be bit-identical to
// llama.cpp's (they are on CPU since the KV padding fix, 2026-11-01). Without it
// the check is the looser "same argmax, |Δ| < 0.5" — for backends (CUDA, Q2+)
// whose kernels legitimately differ.
Interventions build_iv(const std::string& path, const ModelLayout& m, const Forward& fw, const Tokenizer& tok, const std::vector<Edit>& edits,
                       int prompt_len);

// With edits: our forward runs MODEL *with* the edits, and llama.cpp runs `against`
// (default MODEL) with `cvec` applied — so `neuronaut export`'s artifacts can be
// checked against the edits they came from (tests/test_export.sh).
int cmd_oracle(const std::string& path, const GgufFile& f, const ModelLayout& m, const std::string& prompt, bool exact,
               int n_gen, const std::vector<Edit>& edits = {}, const std::string& against = "", const std::string& cvec = "") {
    Tokenizer tok(path);
    const auto tokens = tok.encode(prompt);

    // Ours. Without --gen: one forward over the prompt. With --gen N: the cached
    // path — prompt decoded into a cache, then N greedy tokens one decode at a time.
    // Our model is freed before llama.cpp loads its own copy: on a 32 GB GPU two
    // copies of the 9B in BF16 don't fit.
    std::vector<float> ours;
    std::vector<int32_t> gen;
    int nv = 0;
    Device dev = Device::cpu;
    std::string dev_name;
    {
        Forward fw(f, m);
        nv = fw.n_vocab();
        dev = fw.device();
        dev_name = fw.device_name();
        const Interventions iv = edits.empty() ? Interventions{} : build_iv(path, m, fw, tok, edits, 0);
        RunOptions ro;
        ro.all_logits = true;
        auto c = fw.new_cache(static_cast<int>(tokens.size()) + n_gen);
        ForwardResult r = fw.decode(*c, tokens, iv, ro);
        ours = r.logits;
        for (int s = 0; s < n_gen; ++s) {
            const float* last = ours.data() + ours.size() - static_cast<size_t>(nv);
            const int32_t next = top_probs(last, nv, 1)[0].first;
            gen.push_back(next);
            r = fw.decode(*c, {next}, iv);
            ours.insert(ours.end(), r.logits.begin(), r.logits.end());
        }
    }
    Oracle oracle(against.empty() ? path : against, 0, dev == Device::gpu);
    if (!against.empty()) std::printf("llama.cpp side: %s\n", against.c_str());
    if (!cvec.empty()) std::printf("llama.cpp side: control vector %s (%d directions)\n", cvec.c_str(), oracle.set_control_vector(cvec));
    if (oracle.n_vocab() != nv) {
        std::fprintf(stderr, "vocab size mismatch: ours %d, llama.cpp %d\n", nv, oracle.n_vocab());
        return 1;
    }
    const std::vector<float> theirs = n_gen > 0 ? oracle.logits_incremental(tokens, gen) : oracle.logits(tokens);
    std::vector<int32_t> seq = tokens;
    seq.insert(seq.end(), gen.begin(), gen.end());

    std::printf("%zu prompt tokens%s, %d vocab, on %s. Per position: argmax ours / llama.cpp, max |Δlogit|, top-1 prob both\n",
                tokens.size(), n_gen > 0 ? (" + " + std::to_string(n_gen) + " cached decode steps").c_str() : "", nv,
                dev_name.c_str());
    double worst = 0;
    int disagree = 0;
    size_t n_bitexact = 0;
    for (size_t i = 0; i < seq.size(); ++i) {
        const float* a = ours.data() + i * static_cast<size_t>(nv);
        const float* b = theirs.data() + i * static_cast<size_t>(nv);
        n_bitexact += std::memcmp(a, b, static_cast<size_t>(nv) * sizeof(float)) == 0;
        double md = 0;
        for (int v = 0; v < nv; ++v) md = std::max(md, static_cast<double>(std::fabs(a[v] - b[v])));
        const auto ta = top_probs(a, nv, 1)[0], tb = top_probs(b, nv, 1)[0];
        worst = std::max(worst, md);
        if (ta.first != tb.first) ++disagree;
        std::printf("  %3zu%s %s -> %s %s max|Δ| %.4f  p %.3f / %.3f%s\n", i, i >= tokens.size() ? "*" : " ",
                    pad_cells(quote_printable(tok.piece(seq[i]), 12), 14).c_str(),
                    pad_cells(quote_printable(tok.piece(ta.first), 12), 14).c_str(),
                    pad_cells(quote_printable(tok.piece(tb.first), 12), 14).c_str(), md, ta.second, tb.second,
                    ta.first != tb.first ? "  << DISAGREE" : "");
    }
    if (n_gen > 0) std::printf("  (* = fed by a single-token cached decode)\n");
    // A real graph bug shows up as Δ of several logits and argmax flips; benign
    // kernel differences stay far below 0.5.
    const bool ok = disagree == 0 && worst < 0.5 && (!exact || n_bitexact == seq.size());
    std::printf("%s: worst max|Δlogit| %.4g, argmax disagreements %d/%zu, bit-exact positions %zu/%zu\n",
                ok ? "MATCH" : "MISMATCH", worst, disagree, seq.size(), n_bitexact, seq.size());
    return ok ? 0 : 1;
}

// Edit flags shared by run and blame. Each one appends to `edits` (in order; they
// compose). Returns false if argv[argi] isn't an edit flag; advances argi past the
// value when it is. Throws on a malformed value.
//   --ablate L.N | --invert L.N | --scale L.N=K | --set L.N=V
//   --add L.N=V[@prompt] | --steer L.N(+|-)=DOSE[@prompt] | --edits NAME|FILE
bool parse_edit_flag(const ModelLayout& m, int argc, char** argv, int& argi, std::vector<Edit>& edits, std::string& notes) {
    const std::string a = argv[argi];
    static const char* flags[] = {"--ablate", "--invert", "--scale", "--set", "--add", "--steer", "--edits"};
    if (std::find(std::begin(flags), std::end(flags), a) == std::end(flags)) return false;
    if (argi + 1 >= argc) throw std::invalid_argument(a + " needs a value");
    std::string v = argv[++argi];
    if (a == "--edits") {
        const EditSet set = load_edit_set(edit_set_path(model_key(m), v), m);
        edits.insert(edits.end(), set.edits.begin(), set.edits.end());
        if (!set.notes.empty()) notes += (notes.empty() ? "" : "\n") + set.notes;
        return true;
    }
    Edit e;
    // Optional @prompt / @all suffix (add / steer).
    if (const size_t at = v.rfind('@'); at != std::string::npos) {
        const std::string sc = v.substr(at + 1);
        if (sc == "prompt") e.scope = EditScope::Prompt;
        else if (sc != "all") throw std::invalid_argument(a + ": unknown scope @" + sc + " (all|prompt)");
        v.resize(at);
    }
    const size_t eq = v.find('=');
    const std::string ref = v.substr(0, eq);
    char pole = 0;
    if (!parse_neuron_ref(ref, e.layer, e.neuron, pole)) throw std::invalid_argument(a + ": bad neuron " + ref + " (want L.N)");
    auto num = [&]() {
        if (eq == std::string::npos) throw std::invalid_argument(a + " wants L.N=VALUE");
        return std::stof(v.substr(eq + 1));
    };
    if (a == "--ablate" || a == "--invert") {
        e.op = EditOp::Scale;
        e.k = a == "--ablate" ? 0.0f : -1.0f;
    } else if (a == "--scale") {
        e.op = EditOp::Scale;
        e.k = num();
    } else if (a == "--set" || a == "--add") {
        e.op = a == "--set" ? EditOp::Set : EditOp::Add;
        e.value = num();
    } else {   // --steer
        e.op = EditOp::Steer;
        e.pole = pole;
        e.dose = num();
    }
    if (const std::string err = validate_edit(e, m); !err.empty()) throw std::invalid_argument(a + " " + argv[argi] + ": " + err);
    edits.push_back(e);
    return true;
}

// Edits → interventions, computing steering units if any edit needs them.
Interventions build_iv(const std::string& path, const ModelLayout& m, const Forward& fw, const Tokenizer& tok, const std::vector<Edit>& edits,
                       int prompt_len) {
    Interventions iv;
    iv.prompt_len = prompt_len;
    std::unique_ptr<SteerUnits> units;
    if (needs_units(edits)) units = std::make_unique<SteerUnits>(path, m, fw, tok);
    apply_edits(edits, m, units.get(), iv);
    for (const auto& e : edits) {
        std::printf("edit: %s  [%s]", edit_text(e).c_str(), portability_name(portability(e)));
        if (e.op == EditOp::Steer) std::printf("  = a %+.3f (median‖resid‖ %.2f / ‖down‖ %.3f)", units->value(e.layer, e.neuron, e.dose, e.pole),
                                               units->median_norm(e.layer), units->down_norm(e.layer, e.neuron));
        std::printf("\n");
    }
    return iv;
}

void maybe_save(const std::string& save, const ModelLayout& m, const std::vector<Edit>& edits, const std::string& notes) {
    if (save.empty()) return;
    EditSet s;
    s.name = std::filesystem::path(save).stem().string();
    s.notes = notes;
    s.edits = edits;
    const std::string p = edit_set_path(model_key(m), save);
    save_edit_set(p, s, m);
    std::printf("saved %zu edit%s → %s\n", edits.size(), edits.size() == 1 ? "" : "s", p.c_str());
}

int cmd_run(const std::string& path, const GgufFile& f, const ModelLayout& m, int argc, char** argv, int argi) {
    if (argi >= argc) return usage();
    std::string prompt = argv[argi++];
    int n_gen = 12;
    bool chat = false, think = false;
    std::vector<Edit> edits;
    std::string notes, save, reply;
    std::vector<std::pair<int, int>> watch;
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        int l = 0, n = 0;
        if (parse_edit_flag(m, argc, argv, argi, edits, notes)) {
            continue;
        } else if (a == "-n" && argi + 1 < argc) {
            n_gen = std::stoi(argv[++argi]);
        } else if (a == "--chat") {
            chat = true;
        } else if (a == "--think") {
            chat = think = true;
        } else if (a == "--reply" && argi + 1 < argc) {
            chat = true;
            reply = argv[++argi];
        } else if (a == "--save" && argi + 1 < argc) {
            save = argv[++argi];
        } else if (a == "--note" && argi + 1 < argc) {
            notes = argv[++argi];
        } else if (a == "--watch" && argi + 1 < argc && parse_ln(argv[argi + 1], l, n)) {
            watch.emplace_back(l, n);
            ++argi;
        } else {
            return usage();
        }
    }

    Tokenizer tok(path);
    // --reply: the assistant's turn starts with this text (prefill), and the model
    // continues from it — "what would it say next, given it had started so?"
    std::vector<int32_t> tokens = chat ? tok.encode(chat_wrap(m.arch, prompt, think) + reply, true, true) : tok.encode(prompt);
    const size_t n_prompt = tokens.size();
    Forward fw(f, m);
    const int nv = fw.n_vocab();

    // Prompt-scoped edits act on [0, n_prompt): the template and the user's text,
    // up to and including the assistant header — not on anything generated.
    const Interventions iv = build_iv(path, m, fw, tok, edits, static_cast<int>(n_prompt));
    maybe_save(save, m, edits, notes);

    // Step 0 on the prompt: watched activations and the next-token distribution.
    // Watched neurons' layers in full (not RunOptions::watch: one graph node per
    // watch, and contrast scans watch every neuron of a layer).
    RunOptions ro;
    for (auto [l, n] : watch)
        if (std::find(ro.capture_layers.begin(), ro.capture_layers.end(), l) == ro.capture_layers.end()) ro.capture_layers.push_back(l);
    // One cache for the whole sequence: the prompt now, then one decode per token.
    auto cache = fw.new_cache(static_cast<int>(tokens.size()) + n_gen);
    ForwardResult r = fw.decode(*cache, tokens, iv, ro);
    if (!watch.empty()) {
        std::printf("watched activations a (before interventions), per prompt token:\n  %-16s", "token");
        for (auto [l, n] : watch) std::printf(" L%d.N%-8d", l, n);
        std::printf("\n");
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::printf("  %s", pad_cells(quote_printable(tok.piece(tokens[i]), 14), 16).c_str());
            for (auto [l, n] : watch) std::printf(" %+11.3f", r.act(l, static_cast<int>(i))[n]);
            std::printf("\n");
        }
    }
    std::printf("next token after the prompt:\n");
    for (auto [id, p] : top_probs(r.logits.data(), nv, 8)) {
        std::printf("  %6.2f%%  %s\n", 100.0 * p, quote_printable(tok.piece(id)).c_str());
    }

    // Greedy continuation, one cached decode per token (bit-identical to llama.cpp's
    // own incremental decoding: `neuronaut oracle … --gen N`).
    std::string out;
    for (int step = 0; step < n_gen; ++step) {
        if (step > 0) r = fw.decode(*cache, {tokens.back()}, iv);
        const int next = top_probs(r.logits.data(), nv, 1)[0].first;
        if (tok.is_eog(next)) break;
        tokens.push_back(next);
        out += tok.piece(next);
    }
    std::printf("completion (%zu prompt tokens, greedy):\n  %s%s\n", n_prompt, chat ? "" : prompt.c_str(), quote_printable(reply + out, 400).c_str());
    return 0;
}

int cmd_blame(const std::string& path, const GgufFile& f, const ModelLayout& m, int argc, char** argv, int argi) {
    if (argi >= argc) return usage();
    const std::string prompt = argv[argi++];
    int pos = -1, k = 12;
    std::string target_text, reply, vs_text;
    bool chat = false, think = false;
    std::vector<Edit> edits;
    std::string notes;
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (parse_edit_flag(m, argc, argv, argi, edits, notes)) continue;
        if (a == "--pos" && argi + 1 < argc) pos = std::stoi(argv[++argi]);
        else if (a == "--target" && argi + 1 < argc) target_text = argv[++argi];
        else if (a == "-k" && argi + 1 < argc) k = std::stoi(argv[++argi]);
        else if (a == "--chat") chat = true;
        else if (a == "--think") chat = think = true;
        else if (a == "--reply" && argi + 1 < argc) chat = true, reply = argv[++argi];
        else if (a == "--vs" && argi + 1 < argc) vs_text = argv[++argi];
        else return usage();
    }
    Tokenizer tok(path);
    const auto tokens = chat ? tok.encode(chat_wrap(m.arch, prompt, think) + reply, true, true) : tok.encode(prompt);
    if (pos < 0) pos = static_cast<int>(tokens.size()) - 1;
    Forward fw(f, m);
    const Interventions iv = build_iv(path, m, fw, tok, edits, static_cast<int>(tokens.size()));
    RunOptions ro;
    ro.capture_all_layers = true;
    ro.capture_resid = true;
    ro.capture_heads = true;
    ro.all_logits = true;
    const ForwardResult r = fw.run(tokens, iv, ro);
    int target = top_probs(r.logits_at(pos), r.n_vocab, 1)[0].first;
    if (!target_text.empty()) {
        const auto ids = find_token_ids(f, target_text);
        if (ids.empty()) { std::fprintf(stderr, "no vocab entry is exactly %s\n", quote_printable(target_text).c_str()); return 1; }
        target = ids.front();
    }
    // --vs: blame the *difference* logit(target) − logit(vs). Every term is linear
    // in u, so u = u_target − u_vs splits the gap exactly, the same way.
    int vs = -1;
    if (!vs_text.empty()) {
        const auto ids = find_token_ids(f, vs_text);
        if (ids.empty()) { std::fprintf(stderr, "no vocab entry is exactly %s\n", quote_printable(vs_text).c_str()); return 1; }
        vs = ids.front();
    }
    auto u = unembed_dir(m, target);
    if (vs >= 0) {
        const auto uv = unembed_dir(m, vs);
        for (size_t i = 0; i < u.size(); ++i) u[i] -= uv[i];
    }
    const auto od = out_dots(m, u);
    Blame b = compute_blame(m, r, iv, down_dots(m, u), u, target, pos, &od);
    if (vs >= 0) {
        if (m.softcap > 0) { std::fprintf(stderr, "--vs: not for soft-capped models (the cap isn't linear)\n"); return 1; }
        const float* lg = r.logits_at(pos);
        b.logit = lg[target] - lg[vs];
        b.unexplained = b.logit - (b.embed + b.sum_attn + b.sum_neurons);
        std::printf("pos %d %s -> %s vs %s  logit gap %.3f  p %.2f%% vs %.2f%%\n", pos, quote_printable(tok.piece(tokens[static_cast<size_t>(pos)])).c_str(),
                    quote_printable(tok.piece(target)).c_str(), quote_printable(tok.piece(vs)).c_str(), b.logit, 100.0 * prob_of(lg, r.n_vocab, target),
                    100.0 * prob_of(lg, r.n_vocab, vs));
    }
    auto name = [](const BlameItem& it) {
        char buf[48];
        if (it.kind == BlameItem::Embed) std::snprintf(buf, sizeof buf, "embed");
        else if (it.kind == BlameItem::Attn) std::snprintf(buf, sizeof buf, "L%d attn", it.layer);
        else if (it.kind == BlameItem::Head) std::snprintf(buf, sizeof buf, "L%d H%d", it.layer, it.neuron);
        else std::snprintf(buf, sizeof buf, "L%d.N%d", it.layer, it.neuron);
        return std::string(buf);
    };
    if (vs < 0) std::printf("pos %d %s -> target %s  logit %.3f  p %.2f%%  rank %d\n", pos, quote_printable(tok.piece(tokens[static_cast<size_t>(pos)])).c_str(),
                quote_printable(tok.piece(target)).c_str(), b.logit, 100.0 * b.prob, b.rank);
    for (bool toward : {true, false}) {
        std::printf("%s\n", vs >= 0 ? (toward ? "favours the target:" : "favours --vs:") : toward ? "pushes toward:" : "pushes against:");
        for (const auto& it : b.top(k, toward)) {
            std::printf("  %-12s %+8.3f", name(it).c_str(), it.contrib);
            if (it.kind == BlameItem::Neuron) std::printf("   a=%+.3f  down·u=%+.4f", it.act, it.dot);
            std::printf("\n");
        }
    }
    std::printf("decomposition: embed %+.3f + attn %+.3f + neurons %+.3f = %+.3f  (actual %+.3f, unexplained %+.4f)\n", b.embed,
                b.sum_attn, b.sum_neurons, b.embed + b.sum_attn + b.sum_neurons, b.logit, b.unexplained);
    // Heads must add back up to each layer's attention term (blame.h).
    std::printf("heads: %zu across %d layers, sum %+.3f vs attn %+.3f  (head gap %+.4f)\n", b.heads.size(), m.n_layer, b.sum_attn - b.head_gap,
                b.sum_attn, b.head_gap);
    return std::fabs(b.unexplained) < 0.25f && std::fabs(b.head_gap) < 0.25f ? 0 : 1;
}

// trace: activation patching (src/trace.h). Prints the restoration map — one row
// per position, one column per layer, each cell the % of the clean answer's
// probability that patching that one piece brings back — and checks the trace's
// own invariant: the residual after the last layer at the last position *is* the
// clean run's output, so that cell must restore exactly 100% (resid site).
int cmd_trace(const std::string& path, const GgufFile& f, const ModelLayout& m, int argc, char** argv, int argi) {
    if (argi + 1 >= argc) return usage();
    const std::string clean_text = argv[argi++], corrupt_text = argv[argi++];
    Site site = Site::Resid;
    std::string target_text;
    bool chat = false, think = false;
    std::vector<Edit> edits;
    std::string notes;
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (parse_edit_flag(m, argc, argv, argi, edits, notes)) continue;
        if (a == "--site" && argi + 1 < argc) {
            if (!parse_site(argv[++argi], site)) return usage();
        } else if (a == "--target" && argi + 1 < argc) {
            target_text = argv[++argi];
        } else if (a == "--chat") {
            chat = true;
        } else if (a == "--think") {
            chat = think = true;
        } else {
            return usage();
        }
    }
    Tokenizer tok(path);
    auto enc = [&](const std::string& t) { return chat ? tok.encode(chat_wrap(m.arch, t, think), true, true) : tok.encode(t); };
    const auto clean = enc(clean_text), corrupt = enc(corrupt_text);
    Forward fw(f, m);
    const Interventions iv = build_iv(path, m, fw, tok, edits, static_cast<int>(clean.size()));
    int target = -1;
    if (!target_text.empty()) {
        const auto ids = find_token_ids(f, target_text);
        if (ids.empty()) { std::fprintf(stderr, "no vocab entry is exactly %s\n", quote_printable(target_text).c_str()); return 1; }
        target = ids.front();
    }
    const Trace tr = causal_trace(fw, m, clean, corrupt, iv, site, target);
    std::printf("trace (%s): P(%s) clean %.2f%%, corrupt %.2f%% — cell = %% of that gap restored by patching clean's %s at (layer, pos)\n",
                site_name(site), quote_printable(tok.piece(tr.target)).c_str(), 100.0 * tr.p_clean, 100.0 * tr.p_corrupt, site_name(site));
    std::printf("  %-28s", "pos  clean / corrupt");
    for (int l = 0; l < m.n_layer; ++l) std::printf(" L%-3d", l);
    std::printf("\n");
    for (int t = 0; t < tr.n_tokens; ++t) {
        const bool d = std::find(tr.differ.begin(), tr.differ.end(), t) != tr.differ.end();
        std::string lab = std::to_string(t) + (d ? "* " : "  ") + quote_printable(tok.piece(clean[static_cast<size_t>(t)]), 10);
        if (d) lab += "/" + quote_printable(tok.piece(corrupt[static_cast<size_t>(t)]), 10);
        std::printf("  %s", pad_cells(lab, 28).c_str());
        for (int l = 0; l < m.n_layer; ++l) {
            const float r = tr.restored(l, t);
            std::printf(" %4.0f", std::isnan(r) ? 0.0f : 100.0f * r);
        }
        std::printf("\n");
    }
    const auto [bl, bt] = tr.best();
    if (bl >= 0) std::printf("best: L%d pos %d restores %.1f%%   (* = the prompts differ)\n", bl, bt, 100.0 * tr.restored(bl, bt));
    if (site == Site::Resid && std::fabs(tr.p_clean - tr.p_corrupt) > 1e-6f) {
        const float last = tr.restored(m.n_layer - 1, tr.n_tokens - 1);
        std::printf("check: last layer at the last position restores %.1f%% (must be 100)\n", 100.0 * last);
        return std::fabs(last - 1.0f) < 1e-4f ? 0 : 1;
    }
    return 0;
}

// export: edits → files other runtimes load (src/export.h).
int cmd_export(const std::string& path, const GgufFile& f, const ModelLayout& m, int argc, char** argv, int argi) {
    std::vector<Edit> edits;
    std::string notes, out, cvec_out;
    bool skip = false;
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (parse_edit_flag(m, argc, argv, argi, edits, notes)) continue;
        if (a == "-o" && argi + 1 < argc) out = argv[++argi];
        else if (a == "--cvec" && argi + 1 < argc) cvec_out = argv[++argi];
        else if (a == "--skip-unportable") skip = true;
        else return usage();
    }
    if (edits.empty()) return usage();
    const ExportPlan plan = plan_export(edits, m, skip);
    for (const auto& s : plan.skipped) std::printf("skip  %s\n", s.c_str());
    // Default names beside the source model.
    const std::string stem = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0 ? path.substr(0, path.size() - 5) : path;
    if (plan.needs_gguf() && out.empty()) out = stem + "-edited.gguf";
    if (plan.needs_cvec() && cvec_out.empty()) cvec_out = stem + "-edited.cvec.gguf";
    if (!out.empty() && std::filesystem::exists(out) && std::filesystem::equivalent(out, path)) throw std::invalid_argument("export: -o is the source model");

    std::string prov;
    for (const auto& e : edits) prov += (prov.empty() ? "" : "; ") + edit_text(e);
    ExportReport rep;
    if (plan.needs_gguf()) {
        export_gguf(f, m, plan, out, prov, rep);
        std::printf("wrote %s\n", out.c_str());
    }
    if (plan.needs_cvec()) {
        std::unique_ptr<Forward> fw;
        std::unique_ptr<Tokenizer> tok;
        std::unique_ptr<SteerUnits> units;
        if (needs_units(plan.adds)) {
            tok = std::make_unique<Tokenizer>(path);
            fw = std::make_unique<Forward>(f, m);
            units = std::make_unique<SteerUnits>(path, m, *fw, *tok);
        }
        export_control_vector(m, plan, units.get(), cvec_out, rep);
        std::printf("wrote %s   (llama.cpp: --control-vector %s)\n", cvec_out.c_str(), cvec_out.c_str());
    }
    for (const auto& l : rep.lines) std::printf("  %s\n", l.c_str());
    if (rep.rows_folded) std::printf("%d/%d folded rows exact\n", rep.rows_exact, rep.rows_folded);
    return 0;
}

// rome: rewrite one fact (src/rome.h). Prints the value search, saves the
// rank-one edit as an edit set, and shows the edited model on the prompt.
int cmd_rome(const std::string& path, const GgufFile& f, const ModelLayout& m, int argc, char** argv, int argi) {
    RomeRequest req;
    std::string save, notes, corpus;
    int cov_docs = 400, cov_tokens = 128;   // ~45k tokens of C4: what the examples were made with
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        const bool has = argi + 1 < argc;
        if (a == "--subject" && has) req.subject = argv[++argi];
        else if ((a == "--prompt" || a == "--chat-prompt") && has) req.prompts.push_back({argv[++argi], "", a == "--chat-prompt"});
        else if (a == "--target" && has) {
            if (req.prompts.empty() || !req.prompts.back().target.empty()) return usage();   // each --target follows its prompt
            req.prompts.back().target = argv[++argi];
        }
        else if (a == "--layer" && has) req.layer = std::stoi(argv[++argi]);
        else if (a == "--essence" && has) req.essence.push_back(argv[++argi]);
        else if (a == "--prefix" && has) req.prefixes.push_back(argv[++argi]);
        else if (a == "--steps" && has) req.steps = std::stoi(argv[++argi]);
        else if (a == "--lr" && has) req.lr = std::stof(argv[++argi]);
        else if (a == "--kl" && has) req.kl_weight = std::stof(argv[++argi]);
        else if (a == "--decay" && has) req.weight_decay = std::stof(argv[++argi]);
        else if (a == "--clamp" && has) req.clamp = std::stof(argv[++argi]);
        else if (a == "--ridge" && has) req.ridge = std::stof(argv[++argi]);
        else if (a == "--corpus" && has) corpus = argv[++argi];
        else if (a == "--cov-docs" && has) cov_docs = std::stoi(argv[++argi]);
        else if (a == "--cov-tokens" && has) cov_tokens = std::stoi(argv[++argi]);
        else if (a == "--save" && has) save = argv[++argi];
        else if (a == "--note" && has) notes = argv[++argi];
        else return usage();
    }
    if (req.subject.empty() || req.prompts.empty() || req.layer < 0) return usage();
    if (corpus.empty()) {
        // The corpus scripts/fetch-corpus.sh writes, beside the models.
        const std::filesystem::path dir = std::filesystem::path(path).parent_path() / "corpus";
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(dir, ec))
            if (de.path().extension() == ".jsonl" && (corpus.empty() || de.path().string() > corpus)) corpus = de.path().string();
        if (corpus.empty()) throw std::runtime_error("rome: no corpus in " + dir.string() + " — run scripts/fetch-corpus.sh or pass --corpus");
    }
    Tokenizer tok(path);
    Forward fw(f, m, 0, Device::cpu);
    auto say = [](const std::string& s) { std::printf("%s\n", s.c_str()), std::fflush(stdout); };
    const auto cov = rome_covariance(fw, m, tok, path, req.layer, corpus, cov_docs, cov_tokens, say);
    const RomeResult r = rome(fw, m, tok, req, cov, say);

    Edit e;
    e.layer = r.edit.layer;
    e.neuron = -1;
    e.op = EditOp::RankOne;
    e.u = r.edit.u;
    e.w = r.edit.w;
    std::string what;
    for (const auto& p : req.prompts) what += (what.empty() ? "" : "; ") + std::string(p.chat ? "chat " : "") + "\"" + p.tmpl + "\" → \"" + p.target + "\"";
    e.note = "ROME: {} = " + req.subject + "; " + what;
    if (notes.empty()) notes = "ROME (neuronaut rome) at layer " + std::to_string(req.layer) + ", {} = \"" + req.subject + "\": " + what;
    maybe_save(save, m, {e}, notes);
    return 0;
}

// probe: the native contrast scan (src/probe.h). Output format matches the old
// contrast_scan.py it replaced, one line per layer.
int cmd_probe(const std::string& path, const GgufFile& f, const ModelLayout& m, int argc, char** argv, int argi) {
    std::vector<std::string> pos, neg;
    std::vector<int> layers;
    int k = 4;
    bool chat = false;
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "--pos" && argi + 1 < argc) pos.push_back(argv[++argi]);
        else if (a == "--neg" && argi + 1 < argc) neg.push_back(argv[++argi]);
        else if (a == "-k" && argi + 1 < argc) k = std::stoi(argv[++argi]);
        else if (a == "--chat") chat = true;
        else if (a == "--layers" && argi + 1 < argc) {
            std::stringstream ss(argv[++argi]);
            for (std::string x; std::getline(ss, x, ',');) layers.push_back(std::stoi(x));
        } else return usage();
    }
    if (pos.empty() || neg.empty()) return usage();
    Tokenizer tok(path);
    auto enc = [&](const std::vector<std::string>& ts) {
        std::vector<std::vector<int32_t>> out;
        for (const auto& t : ts) out.push_back(chat ? tok.encode(chat_wrap(m.arch, t, false), true, true) : tok.encode(t));
        return out;
    };
    Forward fw(f, m);
    const auto hits = contrast_probe(fw, m, enc(pos), enc(neg), layers, k);
    for (size_t i = 0; i < hits.size();) {
        const int l = hits[i].layer;
        std::printf("L%d:", l);
        for (; i < hits.size() && hits[i].layer == l; ++i)
            std::printf("  N%d%c margin=%+.2f pos~%+.2f neg~%+.2f", hits[i].neuron, hits[i].pole, hits[i].margin, hits[i].pos_mean, hits[i].neg_mean);
        std::printf("\n");
    }
    return 0;
}

// The context of one example: up to `before` tokens before the peak and `after`
// after it, the peak token in «».
std::string example_context(const ActIndex& ix, const ActExample& e, const Tokenizer& tok, int before = 12, int after = 3) {
    const auto& d = ix.doc(e.doc);
    std::string s;
    const int p = static_cast<int>(e.pos);
    for (int i = std::max(0, p - before); i <= std::min(static_cast<int>(d.size()) - 1, p + after); ++i) {
        const std::string pc = tok.piece(d[static_cast<size_t>(i)]);
        s += i == p ? "«" + pc + "»" : pc;
    }
    return s;
}

int cmd_index(const std::string& path, const GgufFile& f, const ModelLayout& m, int argc, char** argv, int argi) {
    if (argi >= argc) return usage();
    const std::string corpus = argv[argi++];
    size_t n_docs = 5000;
    int max_tokens = 128, k = 10;
    std::string out = default_index_path(path);
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "--docs" && argi + 1 < argc) n_docs = std::stoul(argv[++argi]);
        else if (a == "--max-tokens" && argi + 1 < argc) max_tokens = std::stoi(argv[++argi]);
        else if (a == "-k" && argi + 1 < argc) k = std::stoi(argv[++argi]);
        else if (a == "-o" && argi + 1 < argc) out = argv[++argi];
        else return usage();
    }
    Tokenizer tok(path);
    std::vector<std::vector<int32_t>> docs;
    for (const auto& t : read_corpus(corpus, n_docs)) {
        auto ids = tok.encode(t);
        if (static_cast<int>(ids.size()) > max_tokens) ids.resize(static_cast<size_t>(max_tokens));
        if (ids.size() >= 2) docs.push_back(std::move(ids));
    }
    std::fprintf(stderr, "index: %zu documents (≤ %d tokens each) from %s\n", docs.size(), max_tokens, corpus.c_str());
    Forward fw(f, m);
    const auto t0 = std::chrono::steady_clock::now();
    ActIndex ix = ActIndex::build(fw, m, std::move(docs), k, true, tok.bos(), [&](int d, int n) {
        if (d % 50 == 0 || d == n) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "\r  %d/%d docs  %.0f s (eta %.0f s)   ", d, n, s, s / d * (n - d));
        }
    });
    std::fprintf(stderr, "\n");
    ix.corpus = std::filesystem::path(corpus).filename().string();
    ix.gguf = std::filesystem::path(path).filename().string();
    ix.built = now_iso();
    ix.device = fw.device_name();
    ix.save(out);
    std::printf("wrote %s: %zu docs, %zu tokens, k=%d, on %s\n", out.c_str(), ix.n_docs(), ix.n_tokens(), ix.k(), ix.device.c_str());
    return 0;
}

int cmd_examples(const std::string& path, const ModelLayout& m, int argc, char** argv, int argi) {
    if (argi >= argc) return usage();
    int l = 0, n = 0;
    char pole = 0;
    if (!parse_neuron_ref(argv[argi++], l, n, pole)) return usage();
    int k = 10;
    std::string ipath = find_index(path, m);
    if (ipath.empty()) ipath = default_index_path(path);   // load() explains how to build one
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        if (a == "-k" && argi + 1 < argc) k = std::stoi(argv[++argi]);
        else if (a == "--index" && argi + 1 < argc) ipath = argv[++argi];
        else return usage();
    }
    const ActIndex ix = ActIndex::load(ipath, m);
    Tokenizer tok(path);
    for (char p : {'+', '-'}) {
        if (pole && pole != p) continue;
        std::printf("L%d.N%d%c — top %s activations over %zu docs (%s)\n", l, n, p, p == '+' ? "positive" : "negative", ix.n_docs(), ix.corpus.c_str());
        int i = 0;
        for (const auto& e : ix.top(l, n, p)) {
            if (i++ >= k) break;
            std::printf("  %+8.3f  doc %-5u %s\n", e.act, e.doc, quote_printable(example_context(ix, e, tok), 160).c_str());
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// labels: the neuron-label database (src/labels.h). Needs only the layout, not a
// forward pass, so it is instant even on the 9B.
// ---------------------------------------------------------------------------
void print_label(const Label& l) {
    std::string tags;
    for (const auto& t : l.tags) tags += " #" + t;
    std::printf("%-14s %s%s  [%s]\n", neuron_ref(l.layer, l.neuron, l.pole).c_str(), l.label.c_str(), tags.c_str(), l.source.c_str());
    if (!l.notes.empty()) std::printf("%-14s   %s\n", "", l.notes.c_str());
}

std::string user_name() {
    const char* u = std::getenv("USER");
    return u ? u : "";
}

int cmd_labels(const ModelLayout& m, int argc, char** argv, int argi) {
    if (argi >= argc) return usage();
    const std::string op = argv[argi++];
    LabelDb db(m);
    // Flags shared by the ops; positional args collected in order.
    std::vector<std::string> pos, tags;
    std::string source, tag, note, author = user_name();
    int layer = -1;
    ImportOptions io;
    ExportOptions eo;
    bool have_source = false;
    for (; argi < argc; ++argi) {
        const std::string a = argv[argi];
        auto next = [&]() -> std::string {
            if (argi + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++argi];
        };
        if (a == "--source") {
            source = next();
            have_source = true;
        } else if (a == "--tag") {
            tag = next();
            tags.push_back(tag);
        } else if (a == "--note") {
            note = next();
        } else if (a == "--author") {
            author = next();
        } else if (a == "--layer") {
            layer = std::stoi(next());
        } else if (a == "--format") {
            io.format = eo.format = parse_label_format(next());
        } else if (a == "--map") {
            const std::string kv = next();
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) throw std::invalid_argument("--map wants OURS=THEIRS, e.g. label=name");
            io.rename[kv.substr(0, eq)] = kv.substr(eq + 1);
        } else if (a == "--replace-source") {
            io.replace_source = true;
        } else if (a == "--dry-run") {
            io.dry_run = true;
        } else if (a == "--force") {
            io.force = true;
        } else {
            pos.push_back(a);
        }
    }

    if (op == "path") {
        std::printf("%s\n", db.path().c_str());
        return 0;
    }
    if (op == "list" || op == "search") {
        std::string q = op == "search" ? (pos.empty() ? "" : pos[0]) : "";
        for (size_t i = 1; op == "search" && i < pos.size(); ++i) q += " " + pos[i];
        if (have_source) q += " src:" + source;
        if (!tag.empty()) q += " #" + tag;
        int n = 0;
        for (const auto& l : db.search(q)) {
            if (layer >= 0 && l.layer != layer) continue;
            print_label(l);
            ++n;
        }
        std::fprintf(stderr, "%d label%s  (%s)\n", n, n == 1 ? "" : "s", db.path().c_str());
        return 0;
    }
    if (op == "show") {
        int l = 0, n = 0;
        char pole = 0;
        if (pos.empty() || !parse_neuron_ref(pos[0], l, n, pole)) return usage();
        const auto ls = db.at(l, n);
        if (ls.empty()) std::printf("%s: no labels\n", neuron_ref(l, n).c_str());
        for (const auto& x : ls) {
            print_label(x);
            std::printf("%-14s   by %s, created %s, updated %s\n", "", x.author.empty() ? "?" : x.author.c_str(), x.created.c_str(), x.updated.c_str());
        }
        return 0;
    }
    if (op == "set") {
        Label x;
        if (pos.size() < 2 || !parse_neuron_ref(pos[0], x.layer, x.neuron, x.pole)) return usage();
        for (size_t i = 1; i < pos.size(); ++i) x.label += (i > 1 ? " " : "") + pos[i];
        x.tags = tags;
        x.notes = note;
        x.source = have_source ? source : "manual";
        x.author = author;
        const bool added = db.upsert(x);
        std::printf("%s %s\n", added ? "added" : "updated", neuron_ref(x.layer, x.neuron, x.pole).c_str());
        return 0;
    }
    if (op == "rm") {
        int l = 0, n = 0;
        char pole = 0;
        if (pos.empty() || !parse_neuron_ref(pos[0], l, n, pole)) return usage();
        // "L.N" without a pole removes every pole; "L.N+" just that one.
        const bool any_pole = pos[0].find_first_of("+-") == std::string::npos;
        const auto gone = db.remove(l, n, pole, any_pole, have_source ? source : "");
        for (const auto& x : gone) std::printf("removed %s [%s] %s\n", neuron_ref(x.layer, x.neuron, x.pole).c_str(), x.source.c_str(), x.label.c_str());
        if (gone.empty()) std::printf("nothing to remove\n");
        return gone.empty() ? 1 : 0;
    }
    if (op == "import") {
        if (pos.size() != 1) return usage();
        if (have_source) io.default_source = source;
        const ImportReport r = db.import_file(pos[0], io);
        for (const auto& c : r.changes) std::printf("%s\n", c.c_str());
        for (const auto& e : r.errors) std::fprintf(stderr, "error: %s\n", e.c_str());
        if (r.aborted) {
            std::fprintf(stderr, "import aborted, nothing written%s\n", r.skipped ? " (--force skips bad records)" : "");
            return 1;
        }
        std::printf("%s%d added, %d updated, %d removed, %d unchanged, %d skipped → %s\n", io.dry_run ? "dry run: " : "", r.added, r.updated,
                    r.removed, r.unchanged, r.skipped, db.path().c_str());
        return 0;
    }
    if (op == "export") {
        if (pos.size() != 1) return usage();
        eo.source = have_source ? source : "";
        eo.tag = tag;
        if (pos[0] != "-" && io.format == LabelFormat::Auto) {
            // Pick the format from the extension, as import does.
            const std::string ext = std::filesystem::path(pos[0]).extension().string();
            eo.format = ext == ".jsonl" ? LabelFormat::Jsonl : ext == ".csv" ? LabelFormat::Csv : ext == ".tsv" ? LabelFormat::Tsv : LabelFormat::Json;
        }
        const std::string out = db.export_text(eo);
        if (pos[0] == "-") {
            std::fputs(out.c_str(), stdout);
        } else {
            FILE* fp = std::fopen(pos[0].c_str(), "wb");
            if (!fp) throw std::runtime_error("can't write " + pos[0]);
            std::fputs(out.c_str(), fp);
            std::fclose(fp);
        }
        return 0;
    }
    return usage();
}

// edits: saved edit sets for this model (src/edits.h).
int cmd_edits(const ModelLayout& m, int argc, char** argv, int argi) {
    if (argi >= argc) return usage();
    const std::string op = argv[argi++];
    if (op == "list") {
        const auto names = list_edit_sets(model_key(m));
        for (const auto& n : names) {
            const EditSet s = load_edit_set(edit_set_path(model_key(m), n), m);
            Portability worst = Portability::Gguf;
            for (const auto& e : s.edits) worst = std::max(worst, portability(e));
            std::printf("%-28s %2zu edit%s  %-15s %s\n", n.c_str(), s.edits.size(), s.edits.size() == 1 ? " " : "s", portability_name(worst),
                        s.notes.substr(0, s.notes.find('\n')).c_str());
        }
        std::fprintf(stderr, "%zu edit set%s in %s/%s\n", names.size(), names.size() == 1 ? "" : "s", edits_dir().c_str(), label_slug(model_key(m)).c_str());
        return 0;
    }
    if (op == "show" && argi < argc) {
        const std::string p = edit_set_path(model_key(m), argv[argi]);
        const EditSet s = load_edit_set(p, m);
        std::printf("%s  (%s)\n", s.name.c_str(), p.c_str());
        if (!s.notes.empty()) std::printf("  %s\n", s.notes.c_str());
        for (const auto& e : s.edits)
            std::printf("  %-32s %-15s %s\n", edit_text(e).c_str(), portability_name(portability(e)), e.note.c_str());
        return 0;
    }
    return usage();
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string sub = argv[1];
    std::string path;
    int argi = 2;
    if (sub == "info" || sub == "meta" || sub == "tensors" || sub == "neuron" || sub == "lens" || sub == "find" || sub == "render" ||
        sub == "tokenize" ||
        sub == "oracle" || sub == "run" || sub == "blame" || sub == "trace" || sub == "export" || sub == "probe" || sub == "rome" || sub == "index" || sub == "examples" || sub == "mcp" || sub == "labels" || sub == "edits") {
        if (argc < 3) return usage();
        path = argv[2];
        argi = 3;
    } else {
        path = sub;
        sub = "tui";
    }

    if ((sub == "oracle" || sub == "blame" || sub == "trace" || sub == "probe" || sub == "render") && !std::filesystem::exists(path)) {
        std::fprintf(stderr, "skip: %s not present (run scripts/fetch-llama.sh or scripts/fetch-qwen35.sh)\n", path.c_str());
        return 77;   // ctest SKIP_RETURN_CODE
    }
    try {
        auto f = GgufFile::open(path);
        ModelLayout m = ModelLayout::from(*f);

        if (sub == "info") {
            std::printf("%s\n  arch %s  name %s\n  layers %d  d_model %d  n_ff %d  heads %d/%d  vocab %d  ctx %d\n"
                        "  rope_base %.1f  rms_eps %g  unembed %s\n  gguf v%u  %zu tensors  %zu keys  %s\n",
                        path.c_str(), m.arch.c_str(), m.name.c_str(), m.n_layer, m.n_embd, m.n_ff, m.n_head,
                        m.n_head_kv, m.n_vocab, m.n_ctx_train, m.rope_freq_base, m.rms_eps,
                        m.output ? "output.weight" : "tied", f->version(), f->tensors().size(), f->metadata().size(),
                        human_bytes(f->file_size()).c_str());
            if (m.arch != "llama" || m.head_dim != m.n_embd / std::max(1, m.n_head)) {
                std::printf("  head_dim %d  rope_dims %d\n", m.head_dim, m.rope_dims);
            }
            if (const int nl = m.n_linear_layers(); nl > 0) {
                // e.g. "LLLA LLLA …": L = linear attention, A = full attention
                std::string kinds;
                for (int l = 0; l < m.n_layer; ++l) {
                    if (l && l % 4 == 0) kinds += ' ';
                    kinds += m.layers[static_cast<size_t>(l)].kind == LayerKind::linear_attention ? 'L' : 'A';
                }
                std::printf("  layers: %d linear-attention, %d full-attention  %s\n", nl, m.n_layer - nl, kinds.c_str());
            }
            if (!m.mtp_layers.empty()) {
                std::printf("  MTP head: block%s", m.mtp_layers.size() > 1 ? "s" : "");
                for (int l : m.mtp_layers) std::printf(" %d", l);
                std::printf(" (excluded from the forward pass)\n");
            }
            for (const auto& w : m.warnings) std::printf("  warning: %s\n", w.c_str());
            return 0;
        }
        if (sub == "meta") {
            for (const auto& e : f->metadata()) {
                std::printf("%-48s %-18s %s\n", e.key.c_str(), e.type.c_str(),
                            (e.type == "str" ? quote_printable(e.value, 200) : e.value).c_str());
            }
            return 0;
        }
        if (sub == "tensors") {
            for (const auto& t : f->tensors()) {
                std::string shape;
                for (int d = 0; d < t.n_dims; ++d) shape += (d ? "x" : "") + std::to_string(t.ne[d]);
                std::printf("%-36s %-6s %-14s %10s  @0x%llx\n", t.name.c_str(), ggml_type_name(t.type), shape.c_str(),
                            human_bytes(t.n_bytes).c_str(), static_cast<unsigned long long>(t.file_offset));
            }
            return 0;
        }
        if (sub == "oracle") {
            if (argc < argi + 1) return usage();
            bool exact = false;
            int n_gen = 0;
            std::vector<Edit> edits;
            std::string notes, against, cvec;
            for (int k = argi + 1; k < argc; ++k) {
                if (parse_edit_flag(m, argc, argv, k, edits, notes)) continue;
                if (std::strcmp(argv[k], "--exact") == 0) exact = true;
                else if (std::strcmp(argv[k], "--gen") == 0 && k + 1 < argc) n_gen = std::stoi(argv[++k]);
                else if (std::strcmp(argv[k], "--against") == 0 && k + 1 < argc) against = argv[++k];
                else if (std::strcmp(argv[k], "--cvec") == 0 && k + 1 < argc) cvec = argv[++k];
                else return usage();
            }
            return cmd_oracle(path, *f, m, argv[argi], exact, n_gen, edits, against, cvec);
        }
        if (sub == "run") return cmd_run(path, *f, m, argc, argv, argi);
        if (sub == "labels") return cmd_labels(m, argc, argv, argi);
        if (sub == "edits") return cmd_edits(m, argc, argv, argi);
        if (sub == "blame") return cmd_blame(path, *f, m, argc, argv, argi);
        if (sub == "trace") return cmd_trace(path, *f, m, argc, argv, argi);
        if (sub == "probe") return cmd_probe(path, *f, m, argc, argv, argi);
        if (sub == "rome") return cmd_rome(path, *f, m, argc, argv, argi);
        if (sub == "export") return cmd_export(path, *f, m, argc, argv, argi);
        if (sub == "index") return cmd_index(path, *f, m, argc, argv, argi);
        if (sub == "examples") return cmd_examples(path, m, argc, argv, argi);
        if (sub == "tokenize") {
            // Vocab-only libllama load: works before our forward pass supports the arch.
            // BOS follows tokenizer.ggml.add_bos_token (Llama 3: yes; Qwen3.5: no).
            if (argc < argi + 1) return usage();
            const std::string text = argv[argi];
            const bool think = argc > argi + 1 && std::strcmp(argv[argi + 1], "--think") == 0;
            const bool chat = think || (argc > argi + 1 && std::strcmp(argv[argi + 1], "--chat") == 0);
            Tokenizer tok(path);
            const auto ids = chat ? tok.encode(chat_wrap(m.arch, text, think), true, true) : tok.encode(text);
            for (size_t i = 0; i < ids.size(); ++i) {
                std::printf("%4zu %7d %s\n", i, ids[i], quote_printable(tok.piece(ids[i]), 64).c_str());
            }
            return 0;
        }
        if (sub == "find") {
            if (argc < argi + 1) return usage();
            const std::string text = argv[argi];
            const int k = (argc > argi + 1) ? std::stoi(argv[argi + 1]) : 15;
            const auto ids = find_token_ids(*f, text);
            if (ids.empty()) {
                std::fprintf(stderr, "no vocab entry is exactly %s (mind the leading space)\n", quote_printable(text).c_str());
                return 1;
            }
            for (int id : ids) {
                const WriterResult r = token_writers(*f, m, id, k);
                std::printf("token %d %s — neurons whose output points at it (cos with g⊙W_U[token])\n", id,
                            quote_printable(text).c_str());
                std::printf("  %-28s %s\n", "a > 0 writes it", "a < 0 writes it");
                for (size_t i = 0; i < r.positive.size(); ++i) {
                    char a[64], b[64];
                    std::snprintf(a, sizeof a, "L%d.N%d cos %+.3f", r.positive[i].layer, r.positive[i].neuron, r.positive[i].cos);
                    std::snprintf(b, sizeof b, "L%d.N%d cos %+.3f", r.negative[i].layer, r.negative[i].neuron, r.negative[i].cos);
                    std::printf("  %-28s %s\n", a, b);
                }
            }
            return 0;
        }
        if (sub == "neuron" || sub == "lens") {
            if (argc < argi + 2) return usage();
            const int layer = std::stoi(argv[argi]);
            const int index = std::stoi(argv[argi + 1]);
            const NeuronView v = read_neuron(m, layer, index);
            if (sub == "neuron") {
                std::printf("L%d.N%d\n", layer, index);
                print_vec_summary("gate", v.gate, v.gate_fp);
                print_vec_summary("up", v.up, v.up_fp);
                print_vec_summary("down", v.down, v.down_fp);
                return 0;
            }
            const int k = (argc > argi + 2) ? std::stoi(argv[argi + 2]) : 20;
            std::vector<float> w = v.down;
            apply_write_gain(m, layer, w);   // through the post-MLP norm (Gemma)
            const LensResult r = logit_lens(*f, m, w, k);
            std::printf("logit lens L%d.N%d (static, W_U · (g ⊙ down))\n%-34s %s\n", layer, index, "a > 0 promotes",
                        "a < 0 promotes");
            for (size_t i = 0; i < r.top.size(); ++i) {
                const std::string a = fmt_f(r.top[i].logit, 3) + " " + quote_printable(r.top[i].text, 24);
                std::printf("%s %s %s\n", pad_cells(a, 34).c_str(), fmt_f(r.bottom[i].logit, 3).c_str(),
                            quote_printable(r.bottom[i].text, 24).c_str());
            }
            return 0;
        }

        Session s(std::move(f), std::move(m));
        if (sub == "mcp") return run_mcp(s);
        if (sub == "render") {
            int w = 180, h = 56;
            bool color = false;
            // Simulated command-line typing, to check hints/completion headlessly.
            std::string typing;
            bool typing_on = false;
            int typing_cursor = 0;
            for (int i = argi; i < argc; ++i) {
                if (!std::strcmp(argv[i], "--size") && i + 1 < argc) {
                    if (std::sscanf(argv[++i], "%dx%d", &w, &h) != 2) return usage();
                } else if (!std::strcmp(argv[i], "--color")) {
                    color = true;
                } else if (!std::strcmp(argv[i], "--cmd") && i + 1 < argc) {
                    s.exec(argv[++i]);
                } else if (!std::strcmp(argv[i], "--type") && i + 1 < argc) {
                    typing += argv[++i];
                    typing_on = true;
                } else if (!std::strcmp(argv[i], "--tab")) {
                    s.tab(typing);
                    typing_on = true;
                } else if (!std::strcmp(argv[i], "--key") && i + 1 < argc) {
                    s.key(argv[++i]);
                } else {
                    return usage();
                }
            }
            if (typing_on) {
                typing_cursor = static_cast<int>(typing.size());
                s.cmdline = &typing;
                s.cmd_cursor = &typing_cursor;
            }
            std::fputs(render_to_string(s, w, h, color).c_str(), stdout);
            std::fputc('\n', stdout);
            return 0;
        }
        return run_tui(s);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "neuronaut: %s\n", e.what());
        return 1;
    }
}
