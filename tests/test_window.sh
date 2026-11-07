#!/usr/bin/env bash
# test_window.sh — the TUI's windowed captures on a sequence longer than
# Session::kFullCapture (512): a 632-token prompt runs with only a 128-position
# capture window around the cursor, a watch is captured at every position anyway,
# and moving the cursor far outside the window (pos 3) re-runs with the window
# moved there, so blame at pos 3 explains " fox" after "The quick brown".
#
# Usage: test_window.sh <neuronaut-binary> <model.gguf>   (exit 77 without the model)
set -u
BIN="$1"
GGUF="$2"
if [ ! -f "$GGUF" ]; then echo "skip: $GGUF not present (run scripts/fetch-llama.sh)"; exit 77; fi
LONG=$(for i in $(seq 1 45); do printf 'The quick brown fox jumps over the lazy dog near the old mill. '; done)
OUT=$("$BIN" render "$GGUF" --size 180x56 --cmd "prompt $LONG" --cmd "watch 15.7279" --cmd "pos 3" 2>&1)
need() { grep -qF -- "$1" <<<"$OUT" || { echo "$OUT"; echo "FAIL: missing $1"; exit 1; }; }
need 'TOKENS · 632'                        # the whole prompt ran
need 'cursor: pos 3 " brown"'              # the cursor left the first window …
need 'who wrote " fox" at pos 3'           # … and blame was recomputed there
need '◉ L15.N7279'                         # the watch row is drawn
grep -q 'run: ' <<<"$OUT" && { echo "FAIL: a run error"; exit 1; }
echo PASS
