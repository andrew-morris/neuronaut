// rome.cpp — see rome.h.
#include "rome.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

#include <json.hpp>

#include "format.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "oracle.h"

namespace fs = std::filesystem;

namespace neuronaut {
namespace {

void note(const std::function<void(const std::string&)>& say, std::vector<std::string>* log, const std::string& s) {
    if (say) say(s);
    if (log) log->push_back(s);
}

// Contexts the edit prompt is read after, so that δ works for the subject however
// the sentence around it begins (ROME samples short model generations; fixed,
// plain openings keep the result reproducible).
const std::vector<std::string> kPrefixes = {
    "",
    "Here is a fact. ",
    "I read about this today. ",
    "The teacher explained it to the class. ",
    "According to the history books, ",
    "My grandfather used to say that ",
};

// Essence prompts: what the model says right after the subject in these must not
// move (KL), so the edit changes one fact and not who the subject is.
const std::vector<std::string> kEssence = {
    "{} is a",
    "{} was a",
    "{} was born in",
    "{} is best known for",
    "The life of {}",
};

std::string fill(const std::string& tmpl, const std::string& subject, size_t& subject_end) {
    const size_t at = tmpl.find("{}");
    if (at == std::string::npos) throw std::invalid_argument("rome: prompt \"" + tmpl + "\" has no {} for the subject");
    subject_end = at + subject.size();
    return tmpl.substr(0, at) + subject + tmpl.substr(at + 2);
}

// One sequence of the training batch.
struct Seq {
    std::vector<int32_t> tokens;
    int subject_pos = -1;                 // the subject's last token
    std::vector<std::pair<int, int32_t>> targets;   // (position predicting it, target token)
    bool essence = false;                 // KL prompt: the last position must hold
    std::string text;
};

// Tokens of `prefix + filled` (chat-wrapped for a chat prompt), the subject's last
// token in them, and (if target is non-empty) the target's tokens appended —
// checked to tokenize as a clean continuation, so the positions mean what they say.
Seq make_seq(const Tokenizer& tok, const std::string& arch, const std::string& prefix, const RomePrompt& rp, const std::string& subject) {
    size_t send = 0;
    std::string user = prefix + fill(rp.tmpl, subject, send);
    send += prefix.size();
    std::string text = user;
    if (rp.chat) {
        text = chat_wrap(arch, user, false);
        const size_t at = text.find(user);
        if (at == std::string::npos) throw std::logic_error("rome: chat template lost the user text");
        send += at;
    }
    auto enc = [&](const std::string& t) { return tok.encode(t, true, rp.chat); };
    const auto upto = enc(text.substr(0, send));
    const auto prompt = enc(text);
    if (prompt.size() < upto.size() || !std::equal(upto.begin(), upto.end(), prompt.begin()))
        throw std::runtime_error("rome: the subject doesn't end on a token boundary in \"" + text + "\"");
    Seq s;
    s.text = text;
    s.subject_pos = static_cast<int>(upto.size()) - 1;
    if (rp.target.empty()) {
        s.tokens = prompt;
        return s;
    }
    const auto full = enc(text + rp.target);
    if (full.size() <= prompt.size() || !std::equal(prompt.begin(), prompt.end(), full.begin()))
        throw std::runtime_error("rome: the target doesn't tokenize as a continuation of \"" + text + "\" (start it with a space?)");
    s.tokens.assign(full.begin(), full.end() - 1);   // teacher forcing: the last target token is only predicted
    for (size_t i = prompt.size(); i < full.size(); ++i) s.targets.emplace_back(static_cast<int>(i) - 1, full[i]);
    return s;
}

std::vector<float> softmax(const float* logits, int n) {
    std::vector<float> p(static_cast<size_t>(n));
    const float mx = *std::max_element(logits, logits + n);
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[static_cast<size_t>(i)] = std::exp(logits[i] - mx);
    for (auto& v : p) v = static_cast<float>(v / z);
    return p;
}

// Plain fixed-point (format.h's fmt_f always signs).
std::string num(double v, int digits) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", digits, v);
    return b;
}

double dot(const float* a, const float* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

double norm(const std::vector<float>& v) { return std::sqrt(dot(v.data(), v.data(), v.size())); }

// y = (C + λI)·x, rows split across threads.
void matvec(const std::vector<float>& C, int n, double lambda, const std::vector<double>& x, std::vector<double>& y) {
    const int nt = static_cast<int>(std::max(1u, std::min(32u, std::thread::hardware_concurrency())));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            for (int i = t; i < n; i += nt) {
                const float* row = C.data() + static_cast<size_t>(i) * static_cast<size_t>(n);
                double s = lambda * x[static_cast<size_t>(i)];
                for (int j = 0; j < n; ++j) s += static_cast<double>(row[j]) * x[static_cast<size_t>(j)];
                y[static_cast<size_t>(i)] = s;
            }
        });
    }
    for (auto& t : th) t.join();
}

// Solve (C + λI)x = b by conjugate gradients (C symmetric positive semi-definite).
std::vector<float> solve(const std::vector<float>& C, int n, double lambda, const std::vector<float>& b, std::string& how) {
    std::vector<double> x(static_cast<size_t>(n), 0.0), r(b.begin(), b.end()), p = r, Ap(static_cast<size_t>(n));
    double rr = 0;
    for (double v : r) rr += v * v;
    const double stop = rr * 1e-12;
    int it = 0;
    for (; it < 4 * n && rr > stop; ++it) {
        matvec(C, n, lambda, p, Ap);
        double pAp = 0;
        for (size_t i = 0; i < x.size(); ++i) pAp += p[i] * Ap[i];
        const double alpha = rr / pAp;
        double rr2 = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rr2 += r[i] * r[i];
        }
        const double beta = rr2 / rr;
        rr = rr2;
        for (size_t i = 0; i < x.size(); ++i) p[i] = r[i] + beta * p[i];
    }
    how = "conjugate gradients: " + std::to_string(it) + " iterations, relative residual " + num(std::sqrt(rr / std::max(stop * 1e12, 1e-300)), 8);
    return std::vector<float>(x.begin(), x.end());
}

std::vector<std::string> read_corpus(const std::string& path, int max_docs) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("rome: can't read the corpus " + path + " (scripts/fetch-corpus.sh)");
    std::vector<std::string> docs;
    std::string line;
    const bool jsonl = path.size() > 6 && path.compare(path.size() - 6, 6, ".jsonl") == 0;
    while (static_cast<int>(docs.size()) < max_docs && std::getline(in, line)) {
        if (line.empty()) continue;
        if (jsonl) {
            const auto j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_object() && j.contains("text") && j["text"].is_string()) docs.push_back(j["text"].get<std::string>());
        } else {
            docs.push_back(line);
        }
    }
    return docs;
}

// −Σ log P(target token) over a sequence's targets, from all-position logits.
double target_nll(const ForwardResult& r, const Seq& s, int n_vocab, int& n) {
    double t = 0;
    for (const auto& [p, id] : s.targets) {
        t -= std::log(std::max(1e-30, static_cast<double>(prob_of(r.logits_at(p), n_vocab, id))));
        ++n;
    }
    return t;
}

constexpr const char* kCovMagic = "neuronaut-rome-cov/1";

}  // namespace

std::vector<float> rome_covariance(const Forward& fw, const ModelLayout& m, const Tokenizer& tok, const std::string& model_path, int layer,
                                   const std::string& corpus, int max_docs, int max_tokens, const std::function<void(const std::string&)>& say) {
    const int n = m.n_ff;
    const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    const std::string cache = model_path + ".rome-cov-L" + std::to_string(layer);
    // Cache: one header line of JSON, then n² little-endian floats.
    if (std::ifstream in(cache, std::ios::binary); in) {
        std::string head;
        std::getline(in, head);
        const auto j = nlohmann::json::parse(head, nullptr, false);
        if (j.is_object() && j.value("format", "") == kCovMagic && j.value("n_ff", 0) == n && j.value("layer", -1) == layer && j.value("model", "") == m.name) {
            std::vector<float> C(nn);
            in.read(reinterpret_cast<char*>(C.data()), static_cast<std::streamsize>(nn * sizeof(float)));
            if (in) {
                if (say) say("covariance L" + std::to_string(layer) + ": cached (" + std::to_string(j.value("tokens", 0)) + " tokens) " + cache);
                return C;
            }
        }
    }

    const auto docs = read_corpus(corpus, max_docs);
    if (docs.empty()) throw std::runtime_error("rome: no documents in " + corpus);
    // Activations are gathered into blocks of up to kBlock tokens; each block adds
    // Aᵀ·A (one ggml matmul) into the running sum.
    constexpr int kBlock = 2048;
    std::vector<double> sum(nn, 0.0);
    std::vector<float> block;   // tokens × n_ff, position-major — i.e. ggml [n_ff, T]
    block.reserve(static_cast<size_t>(kBlock) * static_cast<size_t>(n));
    size_t total = 0;
    const int n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    auto flush = [&] {
        const int64_t T = static_cast<int64_t>(block.size() / static_cast<size_t>(n));
        if (T == 0) return;
        ggml_init_params ip{};
        ip.mem_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead() + (2 * static_cast<size_t>(T) * n + nn) * sizeof(float) + (1 << 20);
        ggml_context* ctx = ggml_init(ip);
        if (!ctx) throw std::runtime_error("rome: ggml_init failed for the covariance");
        ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, T);
        std::memcpy(A->data, block.data(), block.size() * sizeof(float));
        ggml_tensor* At = ggml_cont(ctx, ggml_transpose(ctx, A));   // [T, n_ff]
        ggml_tensor* AtA = ggml_mul_mat(ctx, At, At);               // [n_ff, n_ff] = Σ_t a_t a_tᵀ
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, AtA);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        const float* s = static_cast<const float*>(AtA->data);
        for (size_t i = 0; i < nn; ++i) sum[i] += s[i];
        ggml_free(ctx);
        total += static_cast<size_t>(T);
        block.clear();
    };
    const int32_t bos = tok.bos();
    for (size_t d = 0; d < docs.size(); ++d) {
        auto t = tok.encode(docs[d]);
        if (static_cast<int>(t.size()) > max_tokens) t.resize(static_cast<size_t>(max_tokens));
        if (t.size() < 2) continue;
        RunOptions opt;
        opt.capture_layers = {layer};
        opt.last_layer = layer;
        const ForwardResult r = fw.run(t, {}, opt);
        for (int p = (t[0] == bos ? 1 : 0); p < r.n_tokens; ++p) {
            const float* a = r.act(layer, p);
            block.insert(block.end(), a, a + n);
            if (block.size() >= static_cast<size_t>(kBlock) * static_cast<size_t>(n)) flush();
        }
        if (say && (d + 1) % 50 == 0) say("covariance L" + std::to_string(layer) + ": " + std::to_string(d + 1) + "/" + std::to_string(docs.size()) + " documents");
    }
    flush();
    std::vector<float> C(nn);
    for (size_t i = 0; i < nn; ++i) C[i] = static_cast<float>(sum[i] / static_cast<double>(total));

    nlohmann::json head;
    head["format"] = kCovMagic;
    head["model"] = m.name;
    head["layer"] = layer;
    head["n_ff"] = n;
    head["tokens"] = total;
    head["documents"] = docs.size();
    head["corpus"] = fs::path(corpus).filename().string();
    const std::string tmp = cache + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << head.dump() << "\n";
        out.write(reinterpret_cast<const char*>(C.data()), static_cast<std::streamsize>(nn * sizeof(float)));
        if (!out) throw std::runtime_error("rome: can't write " + tmp);
    }
    fs::rename(tmp, cache);
    if (say) say("covariance L" + std::to_string(layer) + ": " + std::to_string(total) + " tokens from " + std::to_string(docs.size()) + " documents → " + cache);
    return C;
}

RomeResult rome(const Forward& fw, const ModelLayout& m, const Tokenizer& tok, const RomeRequest& req, const std::vector<float>& cov,
                const std::function<void(const std::string&)>& say) {
    const int L = req.layer, n_ff = m.n_ff, n_embd = m.n_embd, n_vocab = fw.n_vocab();
    if (L < 0 || L >= m.n_layer) throw std::invalid_argument("rome: --layer out of range");
    if (req.subject.empty() || req.prompts.empty()) throw std::invalid_argument("rome: needs a subject and at least one prompt with a target");
    if (cov.size() != static_cast<size_t>(n_ff) * static_cast<size_t>(n_ff)) throw std::invalid_argument("rome: covariance has the wrong size");
    RomeResult res;
    auto log = [&](const std::string& s) { note(say, &res.log, s); };

    // --- the sequences ---------------------------------------------------------
    const auto& prefixes = req.prefixes.empty() ? kPrefixes : req.prefixes;
    const auto& essence = req.essence.empty() ? kEssence : req.essence;
    std::vector<Seq> seqs;
    for (const auto& rp : req.prompts) {
        if (rp.target.empty()) throw std::invalid_argument("rome: prompt \"" + rp.tmpl + "\" has no target");
        for (const auto& p : prefixes) seqs.push_back(make_seq(tok, m.arch, p, rp, req.subject));
    }
    const size_t n_train = seqs.size();
    for (const auto& e : essence) {
        seqs.push_back(make_seq(tok, m.arch, "", RomePrompt{e, "", false}, req.subject));
        seqs.back().essence = true;
    }

    // --- one clean forward per sequence: x after L, keys, reference distributions
    Forward::DeltaBatch b;
    b.kl_weight = req.kl_weight;
    std::vector<float> k_avg(static_cast<size_t>(n_ff), 0.0f), k_plain, value;
    double nll_before = 0;
    int n_tgt = 0;
    for (size_t si = 0; si < seqs.size(); ++si) {
        const Seq& s = seqs[si];
        RunOptions opt;
        opt.capture_layers = {L};
        opt.capture_layer_resid = true;
        opt.capture_resid = true;
        opt.all_logits = true;
        const ForwardResult r = fw.run(s.tokens, {}, opt);
        const int base = b.n_tokens();
        for (int p = 0; p < r.n_tokens; ++p) {
            const float* x = r.resid_after(L, p);
            b.resid.insert(b.resid.end(), x, x + n_embd);
            b.pos.push_back(p);
            b.seq.push_back(static_cast<int32_t>(si));
            b.delta_mask.push_back(p == s.subject_pos ? 1.0f : 0.0f);
        }
        const float* a = r.act(L, s.subject_pos);
        if (!s.essence) {
            for (int i = 0; i < n_ff; ++i) k_avg[static_cast<size_t>(i)] += a[i] / static_cast<float>(n_train);
            for (const auto& [p, id] : s.targets) {
                b.nll_rows.push_back(base + p);
                b.nll_ids.push_back(id);
            }
            nll_before += target_nll(r, s, n_vocab, n_tgt);
        } else {
            b.kl_rows.push_back(base + r.n_tokens - 1);
            const auto pr = softmax(r.logits_at(r.n_tokens - 1), n_vocab);
            b.kl_ref.insert(b.kl_ref.end(), pr.begin(), pr.end());
        }
        if (si == 0) {
            // The prompt as written: its key, and the value W_down·k the MLP wrote
            // there (x_L − x_{L−1} − attn_L, the residual's own bookkeeping).
            k_plain.assign(a, a + n_ff);
            const float* xl = r.resid_after(L, s.subject_pos);
            const float* xp = L > 0 ? r.resid_after(L - 1, s.subject_pos) : r.embed_at(s.subject_pos);
            const float* at = r.attn_at(L, s.subject_pos);
            value.resize(static_cast<size_t>(n_embd));
            for (int i = 0; i < n_embd; ++i) value[static_cast<size_t>(i)] = xl[i] - xp[i] - at[i];
        }
    }
    const double vnorm = norm(value);
    res.value_norm = static_cast<float>(vnorm);
    // Per-token probability of the target text: exp(−mean NLL over every target token).
    const double p_before = std::exp(-nll_before / std::max(1, n_tgt));
    res.p_target_before = static_cast<float>(p_before);
    log("rome L" + std::to_string(L) + ": " + std::to_string(n_train) + " training prompts + " + std::to_string(essence.size()) + " essence, " +
        std::to_string(b.n_tokens()) + " tokens; the layer writes ‖W_down·k*‖ = " + num(vnorm, 3) + " there");

    // --- the value: Adam on δ ------------------------------------------------------
    std::vector<float> delta(static_cast<size_t>(n_embd), 0.0f), mo(delta.size(), 0.0f), ve(delta.size(), 0.0f);
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const double step_size = req.lr * vnorm / std::sqrt(static_cast<double>(n_embd));
    for (int step = 1; step <= req.steps; ++step) {
        Forward::DeltaLoss dl = fw.delta_loss(L, b, delta);
        const double dn = norm(delta);
        log("  step " + std::to_string(step) + (step < 10 ? "   " : "  ") + "P(target) " + num(100 * std::exp(-dl.nll), 1) + "%  kl " + num(dl.kl, 3) +
            "  ‖δ‖ " + num(dn, 2));
        if (dl.nll < 0.05f) break;
        if (step == req.steps) break;
        for (size_t i = 0; i < delta.size(); ++i) {
            const double g = dl.grad[i] + 2.0 * req.weight_decay * delta[i] / (vnorm * vnorm);
            mo[i] = static_cast<float>(b1 * mo[i] + (1 - b1) * g);
            ve[i] = static_cast<float>(b2 * ve[i] + (1 - b2) * g * g);
            const double mh = mo[i] / (1 - std::pow(b1, step)), vh = ve[i] / (1 - std::pow(b2, step));
            // lr is relative: each Adam step moves a coordinate by at most
            // lr·‖W_down·k*‖/√n_embd, so the search behaves alike across models.
            delta[i] -= static_cast<float>(step_size * mh / (std::sqrt(vh) + eps));
        }
        const double after = norm(delta), cap = req.clamp * vnorm;
        if (after > cap)
            for (auto& v : delta) v = static_cast<float>(v * cap / after);
    }
    res.delta_norm = static_cast<float>(norm(delta));

    // --- the rank-one update: w = C⁻¹k* / (k_plain · C⁻¹k*) ------------------------
    double mean_diag = 0;
    for (int i = 0; i < n_ff; ++i) mean_diag += cov[static_cast<size_t>(i) * static_cast<size_t>(n_ff) + static_cast<size_t>(i)];
    mean_diag /= n_ff;
    std::string how;
    std::vector<float> w = solve(cov, n_ff, req.ridge * mean_diag, k_avg, how);
    log("  C⁻¹k*: " + how);
    const double kw = dot(k_plain.data(), w.data(), w.size());
    if (!(std::fabs(kw) > 0)) throw std::runtime_error("rome: k*·C⁻¹k* is zero");
    for (auto& v : w) v = static_cast<float>(v / kw);
    res.edit.layer = L;
    res.edit.u = delta;
    res.edit.w = w;

    // --- check: the weight edit, applied for real, on the training prompts ---------
    Interventions iv;
    iv.add_rank_one(m, L, res.edit.u, res.edit.w);
    double nll_after = 0;
    int n_after = 0;
    for (size_t si = 0; si < n_train; ++si) {
        const Seq& s = seqs[si];
        RunOptions opt;
        opt.all_logits = true;
        const ForwardResult r = fw.run(s.tokens, iv, opt);
        nll_after += target_nll(r, s, n_vocab, n_after);
    }
    const double p_after = std::exp(-nll_after / std::max(1, n_after));
    res.p_target_after = static_cast<float>(p_after);
    log("rank-one edit of L" + std::to_string(L) + " ffn_down: P(target) " + num(100 * p_before, 1) + "% → " + num(100 * p_after, 1) +
        "% per token, applied as a weight edit (‖δ‖ " + num(res.delta_norm, 2) + ")");
    return res;
}

}  // namespace neuronaut
