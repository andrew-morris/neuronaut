// model_layout.cpp — see model_layout.h.
#include "model_layout.h"

#include <cmath>
#include <stdexcept>

namespace neuronaut {

const char* layer_kind_name(LayerKind k) {
    switch (k) {
        case LayerKind::attention: return "attn";
        case LayerKind::linear_attention: return "linear";
    }
    return "?";
}

int ModelLayout::n_linear_layers() const {
    int n = 0;
    for (const auto& L : layers) n += L.kind == LayerKind::linear_attention;
    return n;
}

namespace {

std::string shape_str(const TensorInfo& t) {
    std::string s = "[";
    for (int d = 0; d < t.n_dims; ++d) {
        if (d) s += ", ";
        s += std::to_string(t.ne[d]);
    }
    return s + "]";
}

// Check a 2-D weight has ggml shape [ne0, ne1]; record a warning otherwise.
void expect_shape(ModelLayout& m, const TensorInfo* t, int64_t ne0, int64_t ne1) {
    if (!t) return;
    if (t->n_dims != 2 || t->ne[0] != ne0 || t->ne[1] != ne1) {
        m.warnings.push_back(t->name + " has shape " + shape_str(*t) + ", expected [" + std::to_string(ne0) + ", " +
                             std::to_string(ne1) + "]");
    }
}

// The MLP half: identical tensor names and shapes in every supported arch, except
// the name of the norm in front of it.
void read_ffn(ModelLayout& m, const GgufFile& f, int l, LayerTensors& L, const char* norm_name) {
    const std::string p = "blk." + std::to_string(l) + ".";
    L.ffn_norm = f.find_tensor(p + norm_name + ".weight");
    L.ffn_gate = f.find_tensor(p + "ffn_gate.weight");
    L.ffn_up = f.find_tensor(p + "ffn_up.weight");
    L.ffn_down = f.find_tensor(p + "ffn_down.weight");

    if (!L.ffn_gate || !L.ffn_up || !L.ffn_down) {
        m.warnings.push_back("layer " + std::to_string(l) + " has no dense ffn_gate/up/down (MoE or unsupported arch?)");
    }
    // ggml shapes are [ne0 = input width, ne1 = output width]:
    //   gate/up map n_embd -> n_ff   : [n_embd, n_ff]  (row n = what neuron n reads)
    //   down    maps n_ff  -> n_embd : [n_ff, n_embd]  (element n of each row = what neuron n writes)
    expect_shape(m, L.ffn_gate, m.n_embd, m.n_ff);
    expect_shape(m, L.ffn_up, m.n_embd, m.n_ff);
    expect_shape(m, L.ffn_down, m.n_ff, m.n_embd);
}

void read_attention(const GgufFile& f, int l, LayerTensors& L) {
    const std::string p = "blk." + std::to_string(l) + ".";
    L.attn_norm = f.find_tensor(p + "attn_norm.weight");
    L.attn_q = f.find_tensor(p + "attn_q.weight");
    L.attn_k = f.find_tensor(p + "attn_k.weight");
    L.attn_v = f.find_tensor(p + "attn_v.weight");
    L.attn_qkv = f.find_tensor(p + "attn_qkv.weight");
    L.attn_output = f.find_tensor(p + "attn_output.weight");
    L.attn_q_norm = f.find_tensor(p + "attn_q_norm.weight");
    L.attn_k_norm = f.find_tensor(p + "attn_k_norm.weight");
}

void read_linear_attention(const GgufFile& f, int l, LayerTensors& L) {
    const std::string p = "blk." + std::to_string(l) + ".";
    L.attn_norm = f.find_tensor(p + "attn_norm.weight");
    L.attn_qkv = f.find_tensor(p + "attn_qkv.weight");
    L.attn_gate = f.find_tensor(p + "attn_gate.weight");
    L.ssm_conv1d = f.find_tensor(p + "ssm_conv1d.weight");
    L.ssm_dt = f.find_tensor(p + "ssm_dt.bias");
    L.ssm_a = f.find_tensor(p + "ssm_a");
    L.ssm_beta = f.find_tensor(p + "ssm_beta.weight");
    L.ssm_alpha = f.find_tensor(p + "ssm_alpha.weight");
    L.ssm_norm = f.find_tensor(p + "ssm_norm.weight");
    L.ssm_out = f.find_tensor(p + "ssm_out.weight");
}

}  // namespace

ModelLayout ModelLayout::from(const GgufFile& f) {
    ModelLayout m;
    m.arch = f.get_str("general.architecture").value_or("");
    if (m.arch.empty()) throw std::runtime_error(f.path() + ": no general.architecture key");
    m.name = f.get_str("general.name").value_or("");

    const std::string a = m.arch + ".";
    auto u = [&](const std::string& k) { return static_cast<int>(f.get_uint(a + k).value_or(0)); };
    m.n_block = u("block_count");
    m.n_embd = u("embedding_length");
    m.n_ff = u("feed_forward_length");
    m.n_head = u("attention.head_count");
    m.n_head_kv = u("attention.head_count_kv");
    if (m.n_head_kv == 0) m.n_head_kv = m.n_head;  // absent => no GQA
    m.n_ctx_train = u("context_length");
    m.rope_freq_base = f.get_float(a + "rope.freq_base").value_or(10000.0);
    m.rms_eps = f.get_float(a + "attention.layer_norm_rms_epsilon").value_or(1e-5);
    m.n_vocab = static_cast<int>(f.get_arr_len("tokenizer.ggml.tokens"));
    if (m.n_vocab == 0) m.n_vocab = u("vocab_size");
    m.head_dim = u("attention.key_length");
    if (m.head_dim == 0 && m.n_head > 0) m.head_dim = m.n_embd / m.n_head;
    m.rope_dims = u("rope.dimension_count");
    if (m.rope_dims == 0) m.rope_dims = m.head_dim;

    if (const auto sec = f.get_arr_int(a + "rope.dimension_sections")) {
        for (size_t i = 0; i < 4 && i < sec->size(); ++i) m.rope_sections[i] = static_cast<int>((*sec)[i]);
    }
    m.ssm_conv_kernel = u("ssm.conv_kernel");
    m.ssm_head_k_dim = u("ssm.state_size");
    m.ssm_n_k_heads = u("ssm.group_count");
    m.ssm_n_v_heads = u("ssm.time_step_rank");
    m.ssm_inner = u("ssm.inner_size");

    const bool is_llama = m.arch == "llama";
    const bool is_qwen35 = m.arch == "qwen35";
    const bool is_gemma3 = m.arch == "gemma3", is_gemma4 = m.arch == "gemma4";
    m.gemma = is_gemma3 || is_gemma4;
    m.gemma4 = is_gemma4;
    // Keys that are a scalar in most files and a per-layer array in gemma4's
    // (head_count_kv): element l of the array, else the scalar.
    auto per_layer = [&](const std::string& k, int l, int dflt) -> int {
        if (const auto arr = f.get_arr_int(a + k); arr && static_cast<int>(arr->size()) > l) return static_cast<int>((*arr)[static_cast<size_t>(l)]);
        const auto v = f.get_uint(a + k);
        return v ? static_cast<int>(*v) : dflt;
    };
    if (is_gemma4) {
        // The Gemma 4 variants whose shape neuronaut doesn't model (see model_layout.h).
        if (f.get_arr_int(a + "feed_forward_length"))
            throw std::runtime_error(f.path() + ": per-layer MLP widths (Gemma 4 E2B/E4B double-wide MLP) are not supported");
        if (u("embedding_length_per_layer_input") > 0)
            throw std::runtime_error(f.path() + ": per-layer embeddings (Gemma 4 E2B/E4B) are not supported");
        if (u("attention.shared_kv_layers") > 0) throw std::runtime_error(f.path() + ": KV-shared layers (Gemma 4 E2B/E4B) are not supported");
        if (u("expert_count") > 0) throw std::runtime_error(f.path() + ": MoE (Gemma 4 26B-A4B) is not supported");
    }
    if (m.n_head == 0) m.n_head = per_layer("attention.head_count", 0, 0);
    if (f.get_arr_int(a + "attention.head_count_kv")) m.n_head_kv = 0;   // per layer: filled from the layers below
    if (!is_llama && !is_qwen35 && !m.gemma) {
        m.warnings.push_back("architecture '" + m.arch + "' is untested; assuming llama tensor naming");
    }
    if (m.n_block <= 0 || m.n_embd <= 0 || m.n_ff <= 0) {
        throw std::runtime_error(f.path() + ": missing " + a + "{block_count,embedding_length,feed_forward_length}");
    }

    // MTP blocks sit at the end of block_count (qwen35 and other nextn archs).
    const int n_mtp = u("nextn_predict_layers");
    if (n_mtp < 0 || n_mtp >= m.n_block) throw std::runtime_error(f.path() + ": bad " + a + "nextn_predict_layers");
    m.n_layer = m.n_block - n_mtp;
    for (int l = m.n_layer; l < m.n_block; ++l) m.mtp_layers.push_back(l);

    if (f.has_key_prefix("prism.hadamard.")) {
        m.hadamard_folded = true;
        m.warnings.push_back("prism.hadamard: rotated weights — neuron indices, lens and bytes are not meaningful");
    }

    m.token_embd = f.find_tensor("token_embd.weight");
    m.output_norm = f.find_tensor("output_norm.weight");
    m.output = f.find_tensor("output.weight");

    // qwen35: which layers are linear attention. Prefer the explicit per-block flags;
    // older files only carry the interval (every Nth layer is full attention), which
    // is how llama.cpp falls back too (src/models/qwen35.cpp).
    std::vector<bool> recurrent;
    if (is_qwen35) {
        recurrent = f.get_arr_bool(a + "attention.recurrent_layers").value_or(std::vector<bool>{});
        if (recurrent.empty()) {
            const int interval = u("full_attention_interval") > 0 ? u("full_attention_interval") : 4;
            recurrent.resize(static_cast<size_t>(m.n_block));
            for (int l = 0; l < m.n_block; ++l) recurrent[static_cast<size_t>(l)] = l < m.n_layer && (l + 1) % interval != 0;
        } else if (static_cast<int>(recurrent.size()) != m.n_block) {
            m.warnings.push_back("recurrent_layers has " + std::to_string(recurrent.size()) + " entries, block_count is " +
                                 std::to_string(m.n_block));
            recurrent.resize(static_cast<size_t>(m.n_block), false);
        }
    }

    // Gemma: which layers slide, and the per-layer RoPE.
    std::vector<bool> swa(static_cast<size_t>(m.n_layer), false);
    if (m.gemma) {
        m.n_swa = u("attention.sliding_window");
        m.embd_scale = std::sqrt(static_cast<float>(m.n_embd));
        m.softcap = static_cast<float>(f.get_float(a + "final_logit_softcapping").value_or(0.0));
        if (const auto pat = f.get_arr_bool(a + "attention.sliding_window_pattern")) {
            for (int l = 0; l < m.n_layer && l < static_cast<int>(pat->size()); ++l) swa[static_cast<size_t>(l)] = (*pat)[static_cast<size_t>(l)];
        } else if (m.n_swa > 0) {
            // llama_hparams::set_swa_pattern(n, dense_first = false): the last of every n is full.
            const int n = u("attention.sliding_window_pattern") > 0 ? u("attention.sliding_window_pattern") : 6;
            for (int l = 0; l < m.n_layer; ++l) swa[static_cast<size_t>(l)] = l % n < n - 1;
        }
        // llama.cpp: 1/√head_dim (27B: 1/√(n_embd/n_head)) applied to Q before the
        // softmax for gemma3; gemma4 uses no attention scaling at all.
        if (is_gemma3) m.q_scale = 1.0f / std::sqrt(static_cast<float>(m.n_layer == 62 ? m.n_embd / std::max(1, m.n_head) : m.head_dim));
    }
    const double rope_base_swa = f.get_float(a + "rope.freq_base_swa").value_or(10000.0);
    const double scaling = f.get_float(a + "rope.scaling.factor").value_or(0.0);
    const std::string scaling_type = f.get_str(a + "rope.scaling.type").value_or("");
    if (!scaling_type.empty() && scaling_type != "linear" && scaling_type != "none")
        m.warnings.push_back("rope scaling '" + scaling_type + "' is not implemented; RoPE is unscaled");
    const double rope_scale = (scaling_type == "linear" && scaling > 0) ? 1.0 / scaling : 1.0;
    const bool has_rope_freqs = f.find_tensor("rope_freqs.weight") != nullptr;

    m.layers.resize(static_cast<size_t>(m.n_layer));
    for (int l = 0; l < m.n_layer; ++l) {
        auto& L = m.layers[static_cast<size_t>(l)];
        L.swa = swa[static_cast<size_t>(l)];
        L.n_head = per_layer("attention.head_count", l, m.n_head);
        L.n_head_kv = per_layer("attention.head_count_kv", l, L.n_head);
        L.head_dim = (is_gemma4 && L.swa) ? u("attention.key_length_swa") : m.head_dim;
        L.n_rot = (is_gemma4 && L.swa) ? u("rope.dimension_count_swa") : m.rope_dims;
        if (L.n_rot == 0) L.n_rot = L.head_dim;
        L.rope_base = L.swa ? rope_base_swa : m.rope_freq_base;
        L.rope_scale = L.swa ? 1.0 : rope_scale;
        // llama3's frequency factors apply everywhere; gemma4's only to full layers.
        L.rope_freqs = has_rope_freqs && (!m.gemma || !L.swa);
        if (m.gemma) {
            const std::string p = "blk." + std::to_string(l) + ".";
            L.attn_post_norm = f.find_tensor(p + "post_attention_norm.weight");
            L.ffn_post_norm = f.find_tensor(p + "post_ffw_norm.weight");
            L.out_scale = f.find_tensor(p + "layer_output_scale.weight");
            if (!L.attn_post_norm || !L.ffn_post_norm) m.warnings.push_back("layer " + std::to_string(l) + " is missing a Gemma post-norm");
        }
        if (is_qwen35 && recurrent[static_cast<size_t>(l)]) {
            L.kind = LayerKind::linear_attention;
            read_linear_attention(f, l, L);
        } else {
            read_attention(f, l, L);
        }
        read_ffn(m, f, l, L, is_qwen35 ? "post_attention_norm" : "ffn_norm");
        if (!L.ffn_norm) m.warnings.push_back("layer " + std::to_string(l) + " has no pre-MLP norm");
    }
    // The headline numbers for a model whose layers differ (gemma4): the most KV heads.
    for (const auto& L : m.layers) m.n_head_kv = std::max(m.n_head_kv, L.n_head_kv);
    return m;
}

}  // namespace neuronaut
