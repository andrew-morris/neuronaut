#!/usr/bin/env bash
# test_mcp.sh — drive the `neuronaut mcp` JSON-RPC server over stdio and assert on
# its responses (docs/mcp.md). Two modes:
#
#   envelope  — handshake + tools/list + a reader + an unknown-tool error. Runs on
#               the synthetic fixture, so it needs no Meta weights (CI-safe).
#   dose      — a full session: prompt, blame, and a dose sweep whose verdict must
#               match the known France suppressor L15.N7279. Needs the real model;
#               exits 77 (ctest SKIP) when it is absent.
#   think     — Qwen3.5 thinking mode: :think opens the reasoning block, :gen …
#               until "</think>\n\n" stops right after the trace, :accept + :pos
#               "</think>"+1 land on the answer's first prediction. Also the
#               partial-UTF-8 regression: L29.N366's lens holds a lone byte of a CJK
#               char, which used to kill the server. Needs Qwen3.5-4B (exit 77).
#
# Usage: test_mcp.sh <neuronaut-binary> <model.gguf> <envelope|dose>
set -u

BIN="$1"
GGUF="$2"
MODE="$3"

if [ ! -x "$BIN" ]; then echo "no binary: $BIN"; exit 1; fi
if [ ! -f "$GGUF" ]; then echo "skip: $GGUF not present (run scripts/fetch-llama.sh)"; exit 77; fi

# Pipe the given JSON-RPC lines through the server and capture stdout (responses).
drive() { printf '%s\n' "$@" | "$BIN" mcp "$GGUF" 2>/dev/null; }

# grep a required substring out of the captured output, or fail the test.
need() { grep -qF "$1" <<<"$OUT" || { echo "FAIL: missing $1"; exit 1; }; }

case "$MODE" in
envelope)
    OUT=$(drive \
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}' \
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"model_info","arguments":{}}}' \
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"bogus","arguments":{}}}')
    echo "$OUT"
    need '"serverInfo":{"name":"neuronaut"'   # initialize handshake
    need '"name":"snapshot"'                   # tools/list has the catalogue
    need '"n_layer"'                           # model_info returned hyperparams
    need 'unknown tool: bogus'                 # unknown tool -> JSON-RPC error
    ;;
dose)
    OUT=$(drive \
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}' \
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"exec","arguments":{"line":"prompt The capital of France is"}}}' \
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"blame","arguments":{"k":3}}}' \
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"dose","arguments":{"layer":15,"neuron":7279}}}' \
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"attention","arguments":{"layer":13,"k":2}}}' \
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"trace","arguments":{"corrupt":"The capital of Italy is"}}}')
    echo "$OUT" | cut -c1-600
    need '"text":" Paris"'        # blame identifies the target
    need '"kind":"head"'          # attention is blamed per head
    # L13 H5 writes " Paris" and looks at " France" (2nd after the BOS sink).
    grep -F '"id":5,' <<<"$OUT" | grep -qE '"head":5,"blame":1\.0[0-9]*,"top":\[\{"pos":0,[^]]*\{"pos":4,"text":" France"' \
        || { echo "FAIL: L13 H5 should look at France"; exit 1; }
    # The trace finds the fact moving from " France" (pos 4) to " is" (pos 5) mid-stack.
    need '"differ":[4]'
    grep -F '"id":6,' <<<"$OUT" | grep -qE '"best":\{"layer":1[0-4],"pos":5' || { echo "FAIL: trace best cell should be L10-14 at pos 5"; exit 1; }
    need '"specific":true'        # the suppressor's effect is specific …
    need '"outside":8'            # … 8 of 8 doses outside the control band
    ;;
think)
    OUT=$(drive \
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}' \
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"exec","arguments":{"line":"prompt The capital of France is"}}}' \
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"neuron","arguments":{"layer":29,"neuron":366}}}' \
        '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"exec","arguments":{"line":"think Is Taiwan a country?"}}}' \
        '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"exec","arguments":{"line":"gen 2000 until \"</think>\\n\\n\""}}}' \
        '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"exec","arguments":{"line":"accept"}}}' \
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"exec","arguments":{"line":"pos \"</think>\"+1"}}}')
    echo "$OUT" | cut -c1-400
    need '"id":3,"result"'                     # the lone-byte lens serialized (server alive)
    need '"think":true'                        # :think set thinking mode
    need 'stopped at \"</think>\\n\\n\"'      # :gen stopped right after the trace
    need '"id":7,"result"'
    # The target at the cursor (= the answer's first token) is "Tai" (id 75322), not
    # the "\n\n" a cursor still on </think> would predict.
    grep -F '"id":7,' <<<"$OUT" | grep -qF '"target":{"id":75322,"text":"Tai"' || { echo "FAIL: answer should open with Tai"; exit 1; }
    ;;
*)
    echo "unknown mode: $MODE"; exit 1;;
esac

echo "PASS ($MODE)"
