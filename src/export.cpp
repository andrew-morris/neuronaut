// export.cpp — see export.h.
#include "export.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>

#include "format.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "neuron.h"
#include "steer.h"

namespace neuronaut {
namespace {

// Byte offsets of the fp16 scale fields inside one block, for block-quantized
// types whose value is linear in those fields (x = d·q, x = d·q + m, x = d·sc·q −
// dmin·m): scaling every listed field by k scales every element by k. Layouts from
// ggml-common.h @ b11700; the block size is checked against ggml at run time.
struct ScaleFields {
    size_t block_bytes;
    std::vector<size_t> fp16_offsets;
};
bool scale_fields(ggml_type t, ScaleFields& out) {
    switch (t) {
        case GGML_TYPE_Q4_0: out = {18, {0}}; return true;        // d, qs[16]
        case GGML_TYPE_Q4_1: out = {20, {0, 2}}; return true;     // d, m, qs[16]
        case GGML_TYPE_Q5_0: out = {22, {0}}; return true;        // d, qh[4], qs[16]
        case GGML_TYPE_Q5_1: out = {24, {0, 2}}; return true;     // d, m, qh[4], qs[16]
        case GGML_TYPE_Q8_0: out = {34, {0}}; return true;        // d, qs[32]
        case GGML_TYPE_Q2_K: out = {84, {80, 82}}; return true;   // scales[16], qs[64], d, dmin
        case GGML_TYPE_Q3_K: out = {110, {108}}; return true;     // hmask[32], qs[64], scales[12], d
        case GGML_TYPE_Q4_K: out = {144, {0, 2}}; return true;    // d, dmin, scales[12], qs[128]
        case GGML_TYPE_Q5_K: out = {176, {0, 2}}; return true;    // d, dmin, scales[12], qh[32], qs[128]
        case GGML_TYPE_Q6_K: out = {210, {208}}; return true;     // ql[128], qh[64], scales[16], d
        default: return false;
    }
}

// Scale one row (ne0 elements, in `t`'s storage) by k in place.
void scale_row(ggml_type t, uint8_t* row, int64_t ne0, float k) {
    if (t == GGML_TYPE_F32) {
        auto* x = reinterpret_cast<float*>(row);
        for (int64_t i = 0; i < ne0; ++i) x[i] *= k;
        return;
    }
    if (t == GGML_TYPE_F16) {
        auto* x = reinterpret_cast<ggml_fp16_t*>(row);
        for (int64_t i = 0; i < ne0; ++i) x[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(x[i]) * k);
        return;
    }
    if (t == GGML_TYPE_BF16) {
        auto* x = reinterpret_cast<ggml_bf16_t*>(row);
        for (int64_t i = 0; i < ne0; ++i) x[i] = ggml_fp32_to_bf16(ggml_bf16_to_fp32(x[i]) * k);
        return;
    }
    ScaleFields sf;
    if (scale_fields(t, sf)) {
        if (sf.block_bytes != ggml_type_size(t)) throw std::logic_error(std::string("export: unexpected block size for ") + ggml_type_name(t));
        const int64_t nb = ne0 / ggml_blck_size(t);
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t* blk = row + static_cast<size_t>(b) * sf.block_bytes;
            for (size_t off : sf.fp16_offsets) {
                ggml_fp16_t h;
                std::memcpy(&h, blk + off, sizeof h);
                h = ggml_fp32_to_fp16(ggml_fp16_to_fp32(h) * k);
                std::memcpy(blk + off, &h, sizeof h);
            }
        }
        return;
    }
    // Anything else (IQ*, TQ*, …): dequantize, scale, re-quantize. Lossy beyond k's
    // own rounding; the exactness check reports it.
    if (ggml_quantize_requires_imatrix(t)) throw std::runtime_error(std::string("export: can't re-quantize ") + ggml_type_name(t) + " rows (needs an imatrix)");
    const auto* tr = ggml_get_type_traits(t);
    if (!tr || !tr->to_float) throw std::runtime_error(std::string("export: no dequantizer for ") + ggml_type_name(t));
    std::vector<float> f(static_cast<size_t>(ne0));
    tr->to_float(row, f.data(), ne0);
    for (auto& v : f) v *= k;
    ggml_quantize_chunk(t, f.data(), row, 0, 1, ne0, nullptr);
}

std::vector<float> dequant_row(ggml_type t, const uint8_t* row, int64_t ne0) {
    std::vector<float> f(static_cast<size_t>(ne0));
    const auto* tr = ggml_get_type_traits(t);
    if (t == GGML_TYPE_F32) std::memcpy(f.data(), row, f.size() * sizeof(float));
    else tr->to_float(row, f.data(), ne0);
    return f;
}

// k per (layer, neuron), and v per (layer, neuron), as Interventions composes them.
std::map<std::pair<int, int>, float> fold_factors(const ExportPlan& plan) {
    std::map<std::pair<int, int>, float> k;
    for (const auto& e : plan.folds) {
        auto [it, ins] = k.emplace(std::make_pair(e.layer, e.neuron), 1.0f);
        it->second *= e.op == EditOp::Set ? 0.0f : e.k;
    }
    return k;
}

}  // namespace

ExportPlan plan_export(const std::vector<Edit>& edits, const ModelLayout& m, bool skip_unportable) {
    ExportPlan p;
    for (const auto& e : edits) {
        if (const std::string err = validate_edit(e, m); !err.empty()) throw std::invalid_argument(edit_text(e) + ": " + err);
        std::string why;
        if (e.scope == EditScope::Prompt) why = "prompt-scoped (only while reading the prompt) — no model file can express that";
        else if (e.op != EditOp::Scale && e.layer == 0) why = "an add in layer 0 — llama.cpp never applies a control vector to layer 0";
        else if (e.op != EditOp::Scale && m.has_post_norms())
            why = "an add on a model with a post-MLP norm (" + m.arch + "): the norm rescales a + v together with the rest of the MLP's output, "
                  "so no constant control vector reproduces it";
        if (!why.empty()) {
            if (!skip_unportable) throw std::invalid_argument(edit_text(e) + ": " + why + " (--skip-unportable drops it)");
            p.skipped.push_back(edit_text(e) + ": " + why);
            continue;
        }
        if (e.op == EditOp::RankOne) {
            p.rank_ones.push_back(e);
            continue;
        }
        if (e.op == EditOp::Scale || e.op == EditOp::Set) p.folds.push_back(e);
        if (e.op != EditOp::Scale) p.adds.push_back(e);
    }
    return p;
}

void export_gguf(const GgufFile& f, const ModelLayout& m, const ExportPlan& plan, const std::string& out_path, const std::string& provenance,
                 ExportReport& rep) {
    if (!plan.needs_gguf()) throw std::invalid_argument("export: no scale or rank-one edits to write into a GGUF");
    // Metadata: the original's, plus provenance. Tensor offsets are relative to the
    // data section, so the data section is copied verbatim after the new header.
    ggml_context* meta = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = true;
    ip.ctx = &meta;
    gguf_context* g = gguf_init_from_file(f.path().c_str(), ip);
    if (!g) throw std::runtime_error("export: ggml could not re-read " + f.path());
    struct Guard {
        gguf_context* g;
        ggml_context* c;
        ~Guard() {
            gguf_free(g);
            if (c) ggml_free(c);
        }
    } guard{g, meta};
    gguf_set_val_str(g, "neuronaut.edits", provenance.c_str());
    gguf_set_val_str(g, "neuronaut.source", f.path().substr(f.path().find_last_of('/') + 1).c_str());
    std::vector<uint8_t> header(gguf_get_meta_size(g));
    gguf_get_meta_data(g, header.data());

    const std::string tmp = out_path + ".tmp";
    std::unique_ptr<FILE, int (*)(FILE*)> fp(std::fopen(tmp.c_str(), "wb"), std::fclose);
    if (!fp) throw std::runtime_error("export: can't write " + tmp);
    auto write = [&](const void* p, size_t n) {
        if (std::fwrite(p, 1, n, fp.get()) != n) throw std::runtime_error("export: write failed (disk full?)");
    };
    write(header.data(), header.size());
    const uint64_t new_data = header.size();
    const uint64_t old_data = f.data_offset();
    write(f.map_base() + old_data, f.file_size() - old_data);

    // Patch each folded up row in place.
    for (const auto& [ln, k] : fold_factors(plan)) {
        const TensorInfo& up = *m.layers[static_cast<size_t>(ln.first)].ffn_up;
        const size_t rb = up.row_bytes();
        std::vector<uint8_t> row(up.row_ptr(ln.second), up.row_ptr(ln.second) + rb);
        const std::vector<float> before = dequant_row(up.type, row.data(), up.ne[0]);
        scale_row(up.type, row.data(), up.ne[0], k);
        const std::vector<float> after = dequant_row(up.type, row.data(), up.ne[0]);
        bool exact = true;
        double worst = 0;
        for (size_t i = 0; i < before.size(); ++i) {
            const float want = before[i] * k;
            if (after[i] != want) exact = false;
            worst = std::max(worst, static_cast<double>(std::fabs(after[i] - want)));
        }
        const uint64_t at = new_data + (up.file_offset - old_data) + static_cast<uint64_t>(ln.second) * rb;
        if (std::fseek(fp.get(), static_cast<long>(at), SEEK_SET) != 0) throw std::runtime_error("export: seek failed");
        write(row.data(), rb);
        ++rep.rows_folded;
        rep.rows_exact += exact;
        rep.lines.push_back("fold  L" + std::to_string(ln.first) + ".N" + std::to_string(ln.second) + "  up row ×" + fmt_f(k, 3) + "  (" +
                            ggml_type_name(up.type) + ", " + (exact ? "exact" : "max |Δ| " + fmt_f(worst, 6)) + ")");
    }
    // Rank-one edits: rewrite every row of the layer's ffn_down (rows are outputs:
    // ne[0] = n_ff inputs, ne[1] = n_embd rows), all edits on a layer at once.
    std::map<int, std::vector<const Edit*>> by_layer;
    for (const auto& e : plan.rank_ones) by_layer[e.layer].push_back(&e);
    for (const auto& [l, es] : by_layer) {
        const TensorInfo& down = *m.layers[static_cast<size_t>(l)].ffn_down;
        if (down.ne[0] != m.n_ff || down.ne[1] != m.n_embd) throw std::runtime_error("export: unexpected ffn_down shape in layer " + std::to_string(l));
        if (ggml_quantize_requires_imatrix(down.type)) throw std::runtime_error(std::string("export: can't re-quantize ") + ggml_type_name(down.type) + " (needs an imatrix)");
        const size_t rb = down.row_bytes();
        std::vector<uint8_t> rows(rb * static_cast<size_t>(m.n_embd));
        double worst = 0, edit2 = 0, err2 = 0;   // rounding, vs the edit's own size (whole matrix)
        for (int64_t i = 0; i < m.n_embd; ++i) {
            const uint8_t* src = down.row_ptr(i);
            std::vector<float> rowf = dequant_row(down.type, src, down.ne[0]);
            for (const Edit* e : es) {
                const float ui = e->u[static_cast<size_t>(i)];
                for (int64_t j = 0; j < down.ne[0]; ++j) rowf[static_cast<size_t>(j)] += ui * e->w[static_cast<size_t>(j)];
            }
            uint8_t* dst = rows.data() + static_cast<size_t>(i) * rb;
            if (down.type == GGML_TYPE_F32) std::memcpy(dst, rowf.data(), rb);
            else ggml_quantize_chunk(down.type, rowf.data(), dst, 0, 1, down.ne[0], nullptr);
            const std::vector<float> got = dequant_row(down.type, dst, down.ne[0]);
            for (size_t j = 0; j < rowf.size(); ++j) {
                const double d = std::fabs(static_cast<double>(got[j]) - rowf[j]);
                worst = std::max(worst, d);
            }
            const std::vector<float> orig = dequant_row(down.type, src, down.ne[0]);
            for (size_t j = 0; j < rowf.size(); ++j) {
                edit2 += static_cast<double>(rowf[j] - orig[j]) * (rowf[j] - orig[j]);
                err2 += static_cast<double>(got[j] - rowf[j]) * (got[j] - rowf[j]);
            }
        }
        const uint64_t at = new_data + (down.file_offset - old_data);
        if (std::fseek(fp.get(), static_cast<long>(at), SEEK_SET) != 0) throw std::runtime_error("export: seek failed");
        write(rows.data(), rows.size());
        ++rep.layers_rewritten;
        rep.lines.push_back("rank-one  L" + std::to_string(l) + " ffn_down  " + std::to_string(m.n_embd) + " rows rewritten (" + ggml_type_name(down.type) +
                            ", rounding max |Δ| " + fmt_f(static_cast<float>(worst), 6).substr(1) + ", ‖rounding‖/‖edit‖ " +
                            fmt_f(static_cast<float>(edit2 > 0 ? std::sqrt(err2 / edit2) : 0.0), 3).substr(1) + ")");
    }
    if (std::fflush(fp.get()) != 0) throw std::runtime_error("export: flush failed");
    fp.reset();
    if (std::rename(tmp.c_str(), out_path.c_str()) != 0) throw std::runtime_error("export: can't rename " + tmp + " → " + out_path);
}

std::vector<std::pair<int, std::vector<float>>> control_directions(const ModelLayout& m, const ExportPlan& plan, const SteerUnits* units) {
    std::map<int, std::vector<float>> dirs;
    for (const auto& e : plan.adds) {
        float v = e.value;
        if (e.op == EditOp::Steer) {
            if (!units) throw std::logic_error("control_directions: steer edit without SteerUnits");
            v = units->value(e.layer, e.neuron, e.dose, e.pole);
        }
        auto& d = dirs[e.layer];
        if (d.empty()) d.assign(static_cast<size_t>(m.n_embd), 0.0f);
        const NeuronView nv = read_neuron(m, e.layer, e.neuron);
        for (size_t i = 0; i < d.size(); ++i) d[i] += v * nv.down[i];
    }
    return {dirs.begin(), dirs.end()};
}

void export_control_vector(const ModelLayout& m, const ExportPlan& plan, const SteerUnits* units, const std::string& out_path, ExportReport& rep) {
    const auto dirs = control_directions(m, plan, units);
    if (dirs.empty()) throw std::invalid_argument("export: no add/steer/set edits for a control vector");
    // The keys llama.cpp's own cvector-generator writes; the loader reads only the tensors.
    gguf_context* g = gguf_init_empty();
    ggml_init_params ip{};
    ip.mem_size = dirs.size() * (ggml_tensor_overhead() + static_cast<size_t>(m.n_embd) * sizeof(float)) + 1024;
    ggml_context* ctx = ggml_init(ip);
    gguf_set_val_str(g, "general.architecture", "controlvector");
    gguf_set_val_str(g, "controlvector.model_hint", m.arch.c_str());
    gguf_set_val_i32(g, "controlvector.layer_count", static_cast<int32_t>(dirs.size()));
    for (const auto& [l, d] : dirs) {
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m.n_embd);
        ggml_set_name(t, ("direction." + std::to_string(l)).c_str());
        std::memcpy(t->data, d.data(), d.size() * sizeof(float));
        gguf_add_tensor(g, t);
        double n2 = 0;
        for (float x : d) n2 += static_cast<double>(x) * x;
        rep.lines.push_back("cvec  direction." + std::to_string(l) + "  ‖v‖ " + fmt_f(std::sqrt(n2), 3));
        ++rep.cvec_layers;
    }
    const bool ok = gguf_write_to_file(g, out_path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) throw std::runtime_error("export: can't write " + out_path);
}

}  // namespace neuronaut
