#!/usr/bin/env bash
# test_qwen35_static.sh — Q0 checks on real Qwen3.5 GGUFs (no forward pass needed):
# the hybrid layer layout, the static lens of neurons an earlier Qwen study
# found with HF/PyTorch,
# and the tokenizer's BOS/chat behaviour. Exits 77 (ctest SKIP) when the model is
# absent (run scripts/fetch-qwen35.sh).
#
# Usage: test_qwen35_static.sh <neuronaut-binary> <model.gguf> <4b|9b>
#
# The lens tokens were cross-checked once against an independent numpy reader over
# the original bf16 safetensors (same token order and logits to 3 dp; see
# checked 2026-11-01). Grepping for a few top tokens per pole is enough to
# catch a layout, norm-gain or sign regression.
set -u

BIN="$1"
GGUF="$2"
SIZE="$3"

if [ ! -x "$BIN" ]; then echo "no binary: $BIN"; exit 1; fi
if [ ! -f "$GGUF" ]; then echo "skip: $GGUF not present (run scripts/fetch-qwen35.sh)"; exit 77; fi

fail=0
# need CMD... -- STRING: run CMD, require STRING (fixed) somewhere in stdout.
need() {
    local want="${*: -1}"
    local out
    out=$("${@:1:$#-1}" 2>&1) || { echo "FAIL: '${*:1:$#-1}' exited non-zero"; echo "$out"; fail=1; return; }
    grep -qF -- "$want" <<<"$out" || { echo "FAIL: '${*:1:$#-1}' missing: $want"; echo "$out" | head -20; fail=1; }
}

# Layout: 32 layers, three linear-attention per full-attention, MTP block excluded.
need "$BIN" info "$GGUF" "layers: 24 linear-attention, 8 full-attention  LLLA LLLA"
need "$BIN" info "$GGUF" "MTP head: block 32 (excluded from the forward pass)"
need "$BIN" info "$GGUF" "head_dim 256  rope_dims 64"

# Tokenizer: Qwen3.5 has no BOS; "Text:" + " The" keeps the word boundary.
need "$BIN" tokenize "$GGUF" "Text: The cat" "   0    1138 \"Text\""
need "$BIN" tokenize "$GGUF" "Text: The cat" "   2     561 \" The\""
# Chat wrap: thinking off = an empty <think></think> block before the answer.
need "$BIN" tokenize "$GGUF" "hi" --chat "\"</think>\""
# Thinking on: the template's default — an open <think> + newline, nothing after.
need "$BIN" tokenize "$GGUF" "hi" --think "   9  248068 \"<think>\""
need "$BIN" tokenize "$GGUF" "hi" --think "  10     198 \"\\n\""
if "$BIN" tokenize "$GGUF" "hi" --think 2>&1 | grep -qF '"</think>"'; then echo "FAIL: --think must leave the block open"; fail=1; fi

case "$SIZE" in
9b)
    need "$BIN" info "$GGUF" "unembed output.weight"
    # Results 1: deception writer, − pole.
    need "$BIN" lens "$GGUF" 25 6715 8 "-0.567 \"欺骗\""
    need "$BIN" lens "$GGUF" 25 6715 8 "\" deceive\""
    # Results 3: per-topic neurons (+ pole) and the generic repression neuron (− pole).
    need "$BIN" lens "$GGUF" 24 12269 8 "+0.293 \"藏\""
    need "$BIN" lens "$GGUF" 9 7321 8 "\" Taiwanese\""
    need "$BIN" lens "$GGUF" 10 6333 8 "\"镇压\""
    need "$BIN" lens "$GGUF" 25 5598 8 "\"新疆\""
    ;;
4b)
    need "$BIN" info "$GGUF" "unembed tied"
    # Results 2: exception / norm violation (+ pole); malice neuron's − pole.
    need "$BIN" lens "$GGUF" 13 7956 8 "\" exception\""
    need "$BIN" lens "$GGUF" 14 8949 8 "\" politely\""
    ;;
*) echo "unknown size $SIZE"; exit 2 ;;
esac

[ "$fail" = 0 ] && echo "ok" || exit 1
