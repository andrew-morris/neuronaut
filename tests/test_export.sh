#!/usr/bin/env bash
# test_export.sh — `neuronaut export`'s artifacts checked through llama.cpp itself
# (neuronaut oracle --against / --cvec), on the real model:
#
#   gguf   ablate + invert folded into up rows: llama.cpp running the exported GGUF
#          must be bit-identical to neuronaut running the original with the edits,
#          at every position (--exact).
#   cvec   add + steer + set → patched GGUF (the ×0 of set) + control vector:
#          llama.cpp must agree on every argmax within |Δlogit| < 0.5 (the add lands
#          after the down projection's rounding instead of before it: ~0.01).
#
#   On a model with post-MLP norms (Gemma; pass "nocvec") the cvec leg instead
#   checks that export refuses an add.
#
# Usage: test_export.sh <neuronaut-binary> <model.gguf> <scratch-dir> [ABLATE_L.N INVERT_L.N [nocvec]]
#        (defaults: the Llama 3.2 1B neurons; 77 without the model)
set -u
BIN="$1"; GGUF="$2"; DIR="$3"; ABL="${4:-15.7279}"; INV="${5:-11.1639}"; CV="${6:-cvec}"
if [ ! -f "$GGUF" ]; then echo "skip: $GGUF not present"; exit 77; fi
P="The capital of France is Paris, and the capital of Japan is"
X="$DIR/export-test.gguf"; C="$DIR/export-test.cvec.gguf"
trap 'rm -f "$X" "$C"' EXIT
set -e
"$BIN" export "$GGUF" --ablate "$ABL" --invert "$INV" -o "$X"
"$BIN" oracle "$GGUF" "$P" --exact --ablate "$ABL" --invert "$INV" --against "$X" | tail -1
if [ "$CV" = nocvec ]; then
    if "$BIN" export "$GGUF" --add "$ABL=1" -o "$X" 2>/dev/null; then echo "FAIL: an add was exported past a post-MLP norm"; exit 1; fi
    echo PASS
    exit 0
fi
E=(--add 10.2407=-3 --steer 11.1639+=0.5 --set 12.100=1.5)
"$BIN" export "$GGUF" "${E[@]}" -o "$X" --cvec "$C"
"$BIN" oracle "$GGUF" "$P" "${E[@]}" --against "$X" --cvec "$C" | tail -1
echo PASS
