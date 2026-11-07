#!/usr/bin/env bash
# test_labels.sh — the neuron-label database (src/labels.h) end to end, on the
# synthetic fixtures (no real weights, CI-safe). Each check runs against a fresh
# NEURONAUT_LABELS directory, so the repo's labels/ is never touched.
#
#   - set / show / search / rm through the CLI
#   - import: CSV with a column rename, JSONL, a JSON "ref → label" mapping;
#     unknown fields survive; a bad record aborts the whole import (nothing
#     written) unless --force; --dry-run writes nothing; re-import is a no-op;
#     --replace-source drops that source's records the import no longer has
#   - export json → import into an empty db → export again: identical (round trip)
#   - a native file for another model with the same shape is refused unless --force
#   - the TUI: :label shows up in the NEURON pane; MCP: the labels tool and list rows
#
# Usage: test_labels.sh <neuronaut-binary> <fixture-q8_0.gguf> <fixture-f16.gguf>
set -u

BIN="$1"
Q8="$2"
F16="$3"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export NEURONAUT_LABELS="$TMP/db"

fail() { echo "FAIL: $*"; exit 1; }
L() { "$BIN" labels "$Q8" "$@"; }
need() { grep -qF -- "$2" <<<"$1" || fail "expected '$2' in: $1"; }
lacks() { grep -qF -- "$2" <<<"$1" && fail "did not expect '$2' in: $1"; return 0; }

# --- set / show / search / rm -------------------------------------------------
L set 1.37+ say a city --tag city --note "from blame" >/dev/null || fail "set"
L set L1.N37- not a city >/dev/null || fail "set pole -"
L set 0.5 hello --source other >/dev/null || fail "set --source"
OUT=$(L show 1.37); need "$OUT" "L1.N37+"; need "$OUT" "say a city #city"; need "$OUT" "L1.N37-"
OUT=$(L search city 2>/dev/null); need "$OUT" "L1.N37+"; lacks "$OUT" "L0.N5"
OUT=$(L search '#city' 2>/dev/null); need "$OUT" "L1.N37+"; lacks "$OUT" "not a city"
OUT=$(L search src:other 2>/dev/null); need "$OUT" "L0.N5"; lacks "$OUT" "L1.N37"
L set 1.37+ say a CITY --tag city >/dev/null
OUT=$(L show 1.37); need "$OUT" "say a CITY"           # upsert by (L, N, pole, source)
L set 9.1 nope >/dev/null 2>&1 && fail "out-of-range layer accepted"

# --- import formats -------------------------------------------------------------
printf 'layer,neuron,name,kind\n1,2,"comma, label",x\n0,9,plain,y\n' >"$TMP/x.csv"
OUT=$(L import "$TMP/x.csv" --map label=name --dry-run); need "$OUT" "dry run: 2 added"
OUT=$(L search plain 2>/dev/null); lacks "$OUT" "L0.N9"   # dry run wrote nothing
OUT=$(L import "$TMP/x.csv" --map label=name); need "$OUT" "2 added"
OUT=$(L import "$TMP/x.csv" --map label=name); need "$OUT" "0 added, 0 updated, 0 removed, 2 unchanged"
grep -qF '"kind":"y"' "$(L path)" || fail "unknown CSV column not kept"

printf '{"ref":"L1.N4-","name":"jsonl one","tags":"a;b"}\n{"layer":0,"neuron":6,"label":"jsonl two"}\n' >"$TMP/y.jsonl"
OUT=$(L import "$TMP/y.jsonl" --map label=name --source ext); need "$OUT" "2 added"
OUT=$(L search '#b' 2>/dev/null); need "$OUT" "L1.N4-"

echo '{"L1.N3+": "mapped", "0:4": {"name": "obj"}}' >"$TMP/m.json"
OUT=$(L import "$TMP/m.json"); need "$OUT" "2 added"
OUT=$(L show 0.4); need "$OUT" "obj"

echo '[{"layer":0,"neuron":7,"label":"good"},{"layer":99,"neuron":1,"label":"bad"}]' >"$TMP/bad.json"
L import "$TMP/bad.json" >/dev/null 2>&1 && fail "bad record not refused"
OUT=$(L search good 2>/dev/null); lacks "$OUT" "L0.N7"     # aborted: not even the good one
OUT=$(L import "$TMP/bad.json" --force 2>/dev/null); need "$OUT" "1 added"

# --replace-source: a new x.csv (same source "x", the file stem) without 0.9 drops it.
printf 'layer,neuron,label\n1,2,comma again\n' >"$TMP/x.csv"
OUT=$(L import "$TMP/x.csv"); need "$OUT" "0 added, 1 updated, 0 removed"     # plain merge keeps 0.9
OUT=$(L import "$TMP/x.csv" --replace-source); need "$OUT" "1 removed"
OUT=$(L search plain 2>/dev/null); lacks "$OUT" "L0.N9"

# --- round trip -------------------------------------------------------------------
L export "$TMP/a.json" || fail "export"
L export - --format csv >"$TMP/a.csv" || fail "export csv"
need "$(cat "$TMP/a.csv")" '1,2,,comma again,,,x'
NEURONAUT_LABELS="$TMP/db2" "$BIN" labels "$Q8" import "$TMP/a.json" >/dev/null || fail "import export"
NEURONAUT_LABELS="$TMP/db2" "$BIN" labels "$Q8" export "$TMP/b.json" || fail "re-export"
cmp -s "$TMP/a.json" "$TMP/b.json" || { diff "$TMP/a.json" "$TMP/b.json"; fail "json round trip differs"; }
NEURONAUT_LABELS="$TMP/db3" "$BIN" labels "$Q8" import "$TMP/a.csv" >/dev/null || fail "import csv export"
NEURONAUT_LABELS="$TMP/db3" "$BIN" labels "$Q8" export - --format csv >"$TMP/b.csv"
cmp -s "$TMP/a.csv" "$TMP/b.csv" || { diff "$TMP/a.csv" "$TMP/b.csv"; fail "csv round trip differs"; }

# --- another model, same shape ------------------------------------------------------
OUT=$("$BIN" labels "$F16" import "$TMP/a.json" 2>&1) && fail "labels for another model accepted"
need "$OUT" "same shape"
OUT=$("$BIN" labels "$F16" import "$TMP/a.json" --force); need "$OUT" "added"

# --- TUI and MCP --------------------------------------------------------------------
OUT=$("$BIN" render "$Q8" --cmd "s 1.37" --cmd "label 1.37 tui-made label #t // a note")
need "$OUT" "tui-made label"
OUT=$(L show 1.37); need "$OUT" "tui-made label #t  [manual]"
OUT=$(printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"labels","arguments":{"action":"set","layer":0,"neuron":11,"pole":"+","label":"from mcp","tags":["m"]}}}' \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"labels","arguments":{"query":"#m"}}}' \
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"exec","arguments":{"line":"labels from"}}}' \
    '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"list","arguments":{}}}' \
    '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"neuron","arguments":{"layer":0,"neuron":11}}}' \
    '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"labels","arguments":{"action":"remove","layer":0,"neuron":11}}}' \
    '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"labels","arguments":{"action":"set","layer":0,"neuron":99999,"label":"x"}}}' |
    "$BIN" mcp "$Q8" 2>/dev/null)
need "$OUT" '"added":true'
need "$OUT" '"ref":"L0.N11+"'
need "$OUT" '"kind":"labels"'
need "$OUT" '"labels":"+ from mcp"'
need "$OUT" '"removed":[{"ref":"L0.N11+"'
need "$OUT" 'out of range'
OUT=$(L search mcp 2>/dev/null); lacks "$OUT" "L0.N11"

echo "PASS"
