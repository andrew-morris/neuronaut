// mkfixture.cpp — write synthetic llama-shaped GGUF files for tests and UI work.
//
//   neuronaut-mkfixture <out_dir>
//     -> <out_dir>/fixture-f32.gguf, fixture-f16.gguf, fixture-q8_0.gguf
//
// The 2-D weights are stored in the named type; 1-D norm weights stay F32, which
// is what llama.cpp's converter does for real models too. Values come from
// fixture_formula.h so tests know the ground truth.
//
// This writes with ggml's own GGUF writer, so the files are structurally what
// llama.cpp produces — but they carry no real tokenizer and are not runnable
// by llama.cpp (no merges, no special tokens). They exist to test neuronaut.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "../tools/fixture_formula.h"
#include "ggml.h"
#include "gguf.h"

namespace {

using fixture::Kind;

struct Builder {
    ggml_context* ctx;
    gguf_context* gguf;
    ggml_type wtype;

    // Add a 2-D weight [ne0, ne1] filled from the formula, stored as `wtype`.
    void weight2d(const std::string& name, int kind, int layer, int64_t ne0, int64_t ne1) {
        std::vector<float> src(static_cast<size_t>(ne0 * ne1));
        for (int64_t r = 0; r < ne1; ++r)
            for (int64_t c = 0; c < ne0; ++c) src[static_cast<size_t>(r * ne0 + c)] = fixture::weight(kind, layer, r, c);

        ggml_tensor* t = ggml_new_tensor_2d(ctx, wtype, ne0, ne1);
        ggml_set_name(t, name.c_str());
        if (wtype == GGML_TYPE_F32) {
            std::memcpy(t->data, src.data(), src.size() * sizeof(float));
        } else if (wtype == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(src.data(), static_cast<ggml_fp16_t*>(t->data), static_cast<int64_t>(src.size()));
        } else {
            // Row-wise quantization, exactly how llama-quantize does it (no imatrix).
            ggml_quantize_chunk(wtype, src.data(), t->data, 0, ne1, ne0, nullptr);
        }
        gguf_add_tensor(gguf, t);
    }

    // RMSNorm gains: all ones, always F32.
    void norm(const std::string& name, int64_t n) {
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_name(t, name.c_str());
        auto* d = static_cast<float*>(t->data);
        for (int64_t i = 0; i < n; ++i) d[i] = 1.0f;
        gguf_add_tensor(gguf, t);
    }
};

void write_fixture(const std::filesystem::path& out, ggml_type wtype) {
    using namespace fixture;

    // Generous arena: every tensor stored as F32 would need well under 16 MiB.
    ggml_init_params ip{};
    ip.mem_size = 16u << 20;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context* ctx = ggml_init(ip);
    gguf_context* g = gguf_init_empty();

    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_str(g, "general.name", (std::string("neuronaut-fixture-") + ggml_type_name(wtype)).c_str());
    gguf_set_val_u32(g, "llama.block_count", n_layer);
    gguf_set_val_u32(g, "llama.embedding_length", n_embd);
    gguf_set_val_u32(g, "llama.feed_forward_length", n_ff);
    gguf_set_val_u32(g, "llama.attention.head_count", n_head);
    gguf_set_val_u32(g, "llama.attention.head_count_kv", n_head_kv);
    gguf_set_val_u32(g, "llama.context_length", 128);
    gguf_set_val_f32(g, "llama.rope.freq_base", 500000.0f);
    gguf_set_val_f32(g, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    // Placeholder vocab so the tokenizer-length path is exercised.
    std::vector<std::string> toks;
    std::vector<const char*> tok_ptrs;
    for (int i = 0; i < n_vocab; ++i) toks.push_back("<t" + std::to_string(i) + ">");
    for (auto& s : toks) tok_ptrs.push_back(s.c_str());
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tok_ptrs.data(), tok_ptrs.size());

    Builder b{ctx, g, wtype};
    // Tied embeddings (no output.weight), like Llama 3.2 1B.
    b.weight2d("token_embd.weight", TOK_EMBD, 0, n_embd, n_vocab);
    b.norm("output_norm.weight", n_embd);
    for (int l = 0; l < n_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        b.norm(p + "attn_norm.weight", n_embd);
        b.weight2d(p + "attn_q.weight", ATTN_Q, l, n_embd, n_embd);
        b.weight2d(p + "attn_k.weight", ATTN_K, l, n_embd, head_dim * n_head_kv);
        b.weight2d(p + "attn_v.weight", ATTN_V, l, n_embd, head_dim * n_head_kv);
        b.weight2d(p + "attn_output.weight", ATTN_OUT, l, n_embd, n_embd);
        b.norm(p + "ffn_norm.weight", n_embd);
        b.weight2d(p + "ffn_gate.weight", FFN_GATE, l, n_embd, n_ff);
        b.weight2d(p + "ffn_up.weight", FFN_UP, l, n_embd, n_ff);
        b.weight2d(p + "ffn_down.weight", FFN_DOWN, l, n_ff, n_embd);
    }

    if (!gguf_write_to_file(g, out.c_str(), false)) {
        throw std::runtime_error("failed to write " + out.string());
    }
    gguf_free(g);
    ggml_free(ctx);
    std::printf("wrote %s\n", out.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
        return 2;
    }
    try {
        const std::filesystem::path dir = argv[1];
        std::filesystem::create_directories(dir);
        write_fixture(dir / "fixture-f32.gguf", GGML_TYPE_F32);
        write_fixture(dir / "fixture-f16.gguf", GGML_TYPE_F16);
        write_fixture(dir / "fixture-q8_0.gguf", GGML_TYPE_Q8_0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mkfixture: %s\n", e.what());
        return 1;
    }
    return 0;
}
