// format.h — small text-formatting helpers shared by the CLI and the TUI.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace neuronaut {

// "1.3 GiB", "512 B"
std::string human_bytes(uint64_t n);

// Quote a string for display, escaping control bytes and newlines and truncating
// past `max_len` bytes. Metadata strings come from the file, so a chat template
// full of newlines or a token like "\x1b[2J" must not reach the terminal raw.
std::string quote_printable(const std::string& s, size_t max_len = 48);

// Undo GPT-2 style byte-level BPE (tokenizer.ggml.model == "gpt2", which Llama 3
// uses): every byte is stored as a printable stand-in codepoint ("Ġ" = space,
// "Ċ" = newline, …). Returns the raw bytes the token actually stands for, which
// may be a partial UTF-8 sequence.
std::string bpe_decode(const std::string& token);

// True if `s` is entirely well-formed UTF-8 (no partial characters).
bool valid_utf8(const std::string& s);

// Terminal cells `s` occupies: 2 per East Asian wide / fullwidth character (CJK,
// kana, hangul, fullwidth forms, most emoji), 0 per combining mark, 1 otherwise.
// Invalid bytes count 1 (callers escape them anyway). A compact wcwidth: good for
// the scripts in our vocabularies (Llama 3, Qwen3.5), not a full Unicode table.
int display_width(const std::string& s);

// `s` followed by spaces up to `cells` terminal cells (unchanged if already wider).
// printf's %-Ns pads by *bytes*, which misaligns every column after a CJK token.
std::string pad_cells(const std::string& s, int cells);

// `s` cut to at most `cells` terminal cells, ending in "…" when cut (whole UTF-8
// characters only, wide ones counted as 2). For free text in fixed-width panes —
// FTXUI shrinks a row's *other* columns when one overflows.
std::string fit_cells(const std::string& s, int cells);

// One user turn in the model's chat format, assistant to speak next, as text to
// tokenize with parse_special=true. Per architecture:
//   llama   Llama 3 header tokens. `think` is ignored (no reasoning mode).
//   qwen35  ChatML. think=false: thinking OFF — the empty <think></think> block the
//           official template emits for enable_thinking=false; the setting the earlier Qwen
//           study used (a trace would push the answer out of a short
//           generation window). think=true: thinking ON — the template's default,
//           an *open* "<think>\n", so the model writes its trace, closes it with
//           </think>, then answers. No default system turn: the template adds none.
// Unknown archs get the Llama 3 format (the pre-Qwen behaviour).
std::string chat_wrap(const std::string& arch, const std::string& user, bool think = false);

// Whether chat_wrap(arch, …, think=true) actually opens a reasoning block.
//   qwen35  <think> … </think>             gemma4  <|channel>thought … <channel|>
//   gemma3  user/model turns (<start_of_turn>), no reasoning mode
bool has_thinking(const std::string& arch);
// The token text that closes the reasoning block ("</think>", "<channel|>").
const char* thinking_close(const std::string& arch);

// A vocabulary entry (tokenizer.ggml.tokens[i]) as the text it stands for, by
// tokenizer.ggml.model: "gpt2" byte-level BPE (bpe_decode), SentencePiece ("llama"
// — Gemma 3 —, "gemma4") with ▁ for spaces and <0xHH> byte tokens; anything else
// as stored. May be a partial UTF-8 sequence.
std::string vocab_text(const std::string& tokenizer_model, const std::string& raw);

// Fixed-width float, e.g. fmt_f(-0.01234, 4) -> "-0.0123"
std::string fmt_f(double v, int precision = 4);

// Two hex digits per byte, space separated.
std::string hex_bytes(const uint8_t* p, size_t n);

}  // namespace neuronaut
