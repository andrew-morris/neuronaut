// format.cpp — see format.h.
#include "format.h"

#include <cstdio>
#include <vector>

namespace neuronaut {

std::string human_bytes(uint64_t n) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(n);
    int u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char buf[32];
    if (u == 0) {
        std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(n));
    } else {
        std::snprintf(buf, sizeof buf, "%.1f %s", v, units[u]);
    }
    return buf;
}

namespace {

// Length of the valid UTF-8 sequence starting at s[i], or 0 if invalid.
size_t utf8_len(const std::string& s, size_t i) {
    const auto c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 0;
    if (n == 0 || i + n > s.size()) return 0;
    for (size_t k = 1; k < n; ++k)
        if ((static_cast<unsigned char>(s[i + k]) & 0xc0) != 0x80) return 0;
    return n;
}

// Decode one UTF-8 codepoint at s[i] (assumes utf8_len > 0).
uint32_t utf8_cp(const std::string& s, size_t i, size_t n) {
    const auto c = static_cast<unsigned char>(s[i]);
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1f) : n == 3 ? (c & 0x0f) : (c & 0x07);
    for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3f);
    return cp;
}

}  // namespace

namespace {

bool is_wide(uint32_t cp) {
    return (cp >= 0x1100 && cp <= 0x115f) ||                      // hangul jamo
           (cp >= 0x2e80 && cp <= 0x303e) ||                      // CJK radicals, punctuation
           (cp >= 0x3041 && cp <= 0x33ff) ||                      // kana, CJK compat
           (cp >= 0x3400 && cp <= 0x4dbf) ||                      // CJK ext A
           (cp >= 0x4e00 && cp <= 0x9fff) ||                      // CJK unified
           (cp >= 0xa000 && cp <= 0xa4cf) ||                      // yi
           (cp >= 0xac00 && cp <= 0xd7a3) ||                      // hangul syllables
           (cp >= 0xf900 && cp <= 0xfaff) ||                      // CJK compat ideographs
           (cp >= 0xfe30 && cp <= 0xfe4f) ||                      // CJK compat forms
           (cp >= 0xff00 && cp <= 0xff60) ||                      // fullwidth forms
           (cp >= 0xffe0 && cp <= 0xffe6) ||
           (cp >= 0x1f300 && cp <= 0x1f64f) ||                    // emoji
           (cp >= 0x1f900 && cp <= 0x1f9ff) ||
           (cp >= 0x20000 && cp <= 0x3fffd);                      // CJK ext B+
}

bool is_combining(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036f) || (cp >= 0x0483 && cp <= 0x0489) || (cp >= 0x0591 && cp <= 0x05bd) ||
           (cp >= 0x0610 && cp <= 0x061a) || (cp >= 0x064b && cp <= 0x065f) || (cp >= 0x0e31 && cp <= 0x0e3a) ||
           (cp >= 0x200b && cp <= 0x200f) || (cp >= 0x20d0 && cp <= 0x20ff) || (cp >= 0xfe00 && cp <= 0xfe0f) ||
           cp == 0x3099 || cp == 0x309a;
}

}  // namespace

std::string chat_wrap(const std::string& arch, const std::string& user, bool think) {
    if (arch == "qwen35") {
        // Mirrors chat_template.jinja's add_generation_prompt branch exactly.
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" +
               (think ? "<think>\n" : "<think>\n\n</think>\n\n");
    }
    if (arch == "gemma3") {
        // chat_template's add_generation_prompt; BOS comes from the tokenizer.
        return "<start_of_turn>user\n" + user + "<end_of_turn>\n<start_of_turn>model\n";
    }
    if (arch == "gemma4") {
        // Gemma 4's canonical template: thinking on puts <|think|> in a system turn
        // and leaves the model to open its own <|channel>thought …<channel|>; off
        // closes an empty thought channel for it.
        return std::string(think ? "<|turn>system\n<|think|>\n<turn|>\n" : "") + "<|turn>user\n" + user + "<turn|>\n<|turn>model\n" +
               (think ? "" : "<|channel>thought\n<channel|>");
    }
    return "<|start_header_id|>user<|end_header_id|>\n\n" + user +
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}

bool has_thinking(const std::string& arch) { return arch == "qwen35" || arch == "gemma4"; }

const char* thinking_close(const std::string& arch) { return arch == "gemma4" ? "<channel|>" : "</think>"; }

std::string vocab_text(const std::string& tokenizer_model, const std::string& raw) {
    if (tokenizer_model == "gpt2") return bpe_decode(raw);
    if (tokenizer_model != "llama" && tokenizer_model != "gemma4") return raw;
    // SentencePiece: U+2581 '▁' stands for a space; <0xHH> is one raw byte.
    if (raw.size() == 6 && raw.compare(0, 3, "<0x") == 0 && raw[5] == '>') {
        return std::string(1, static_cast<char>(std::stoi(raw.substr(3, 2), nullptr, 16)));
    }
    std::string out;
    for (size_t i = 0; i < raw.size();) {
        if (raw.compare(i, 3, "\xe2\x96\x81") == 0) {
            out += ' ';
            i += 3;
        } else {
            out += raw[i++];
        }
    }
    return out;
}

int display_width(const std::string& s) {
    int w = 0;
    for (size_t i = 0; i < s.size();) {
        const size_t n = utf8_len(s, i);
        if (n == 0) { ++w; ++i; continue; }
        const uint32_t cp = utf8_cp(s, i, n);
        w += is_combining(cp) ? 0 : is_wide(cp) ? 2 : 1;
        i += n;
    }
    return w;
}

std::string pad_cells(const std::string& s, int cells) {
    const int w = display_width(s);
    return w >= cells ? s : s + std::string(static_cast<size_t>(cells - w), ' ');
}

std::string fit_cells(const std::string& s, int cells) {
    if (cells <= 0) return "";
    if (display_width(s) <= cells) return s;
    std::string out;
    int w = 0;
    for (size_t i = 0; i < s.size();) {
        const size_t n = std::max<size_t>(1, utf8_len(s, i));
        const std::string ch = s.substr(i, n);
        const int cw = display_width(ch);
        if (w + cw > cells - 1) break;   // leave a cell for the ellipsis
        out += ch;
        w += cw;
        i += n;
    }
    return out + "…";
}

bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const size_t n = utf8_len(s, i);
        if (n == 0) return false;
        i += n;
    }
    return true;
}

std::string bpe_decode(const std::string& token) {
    // GPT-2's bytes_to_unicode(): bytes 33..126, 161..172, 174..255 map to the
    // same codepoint; the remaining 68 bytes map, in order, to 256, 257, ….
    static const auto inverse = [] {
        std::vector<int> inv(512, -1);
        int next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
            inv[static_cast<size_t>(keep ? b : next++)] = b;
        }
        return inv;
    }();
    std::string out;
    for (size_t i = 0; i < token.size();) {
        const size_t n = utf8_len(token, i);
        if (n == 0) return token;   // not byte-level BPE text; show as is
        const uint32_t cp = utf8_cp(token, i, n);
        if (cp >= inverse.size() || inverse[cp] < 0) return token;   // e.g. special tokens
        out += static_cast<char>(inverse[cp]);
        i += n;
    }
    return out;
}

std::string quote_printable(const std::string& s, size_t max_len) {
    std::string out = "\"";
    size_t shown = 0;
    for (size_t i = 0; i < s.size();) {
        if (shown >= max_len) {
            out += "…";
            break;
        }
        const auto c = static_cast<unsigned char>(s[i]);
        if (c >= 0x80) {
            // Valid multi-byte UTF-8 passes through; stray bytes (a token that is
            // half a character) are escaped so they can't corrupt the terminal.
            const size_t n = utf8_len(s, i);
            if (n > 1) {
                out.append(s, i, n);
                i += n;
            } else {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\x%02x", c);
                out += buf;
                ++i;
            }
            ++shown;
            continue;
        }
        ++i;
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20 || c == 0x7f) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02x", c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
        ++shown;
    }
    out += "\"";
    return out;
}

std::string fmt_f(double v, int precision) {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%+.*f", precision, v);
    return buf;
}

std::string hex_bytes(const uint8_t* p, size_t n) {
    std::string out;
    out.reserve(n * 3);
    char buf[4];
    for (size_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof buf, "%02x", p[i]);
        if (i) out += ' ';
        out += buf;
    }
    return out;
}

}  // namespace neuronaut
