#!/usr/bin/env bash
# fetch-llama.sh — download Meta's official Llama 3.2 1B Instruct weights and
# convert them to GGUF ourselves, so we know exactly what is in the file.
#
#   HF_TOKEN=hf_xxx scripts/fetch-llama.sh
#
# Prereqs:
#   - Accepted the license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
#     (manual approval by Meta) and a *read* token in $HF_TOKEN.
#   - `cmake -S . -B build` has run, so the pinned llama.cpp source (b11700) is at
#     build/_deps/llama_cpp-src — we use *its* converter, matching our ggml.
#   - uv.
#
# Provenance chain:
#   HF repo pinned to commit $REVISION
#     -> model.safetensors sha256 checked against the LFS oid HF reports for
#        that commit
#     -> converted by llama.cpp b11700's convert_hf_to_gguf.py (hash-pinned tarball)
#     -> SHA256SUMS written for every input and output in models/
#
# We download only safetensors + JSON. The repo's original/consolidated.00.pth is
# a pickle (arbitrary code on load) and is deliberately never fetched.
#
# Output (gitignored):
#   models/hf/Llama-3.2-1B-Instruct/          official files
#   models/Llama-3.2-1B-Instruct-F16.gguf     reference precision
#   models/Llama-3.2-1B-Instruct-Q8_0.gguf    for byte-level patching work
#   models/SHA256SUMS
set -euo pipefail

REPO="meta-llama/Llama-3.2-1B-Instruct"
REVISION="9213176726f574b556790deb65791e0c5aa438b6"   # 2024-10-24, latest as of 2026-11-01
NAME="Llama-3.2-1B-Instruct"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_src="$here/build/_deps/llama_cpp-src"
hf_dir="$here/models/hf/$NAME"
out_dir="$here/models"

die() { echo "fetch-llama: $*" >&2; exit 1; }

[[ -n "${HF_TOKEN:-}" ]] || die "set HF_TOKEN (a read token from https://huggingface.co/settings/tokens)"
[[ -f "$llama_src/convert_hf_to_gguf.py" ]] || die "no pinned llama.cpp at $llama_src — run: cmake -S . -B build"
command -v uv >/dev/null || die "uv not found"

mkdir -p "$hf_dir"

# --- 1. download (token passed via env only, never on the command line) --------
echo "==> downloading $REPO @ ${REVISION:0:12} (safetensors + json only)"
HF_TOKEN="$HF_TOKEN" uvx --from 'huggingface_hub[cli]' hf download "$REPO" \
    --revision "$REVISION" \
    --include 'model.safetensors' --include '*.json' \
    --local-dir "$hf_dir" >/dev/null

# --- 2. verify the weights against the hash HF's git-LFS pointer commits to ----
echo "==> verifying model.safetensors"
expected="$(curl -fsS -m 30 -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/models/$REPO/tree/$REVISION" |
    python3 -c 'import json,sys; print(next(f["lfs"]["oid"] for f in json.load(sys.stdin) if f["path"]=="model.safetensors"))')"
actual="$(sha256sum "$hf_dir/model.safetensors" | cut -d' ' -f1)"
[[ "$expected" == "$actual" ]] || die "model.safetensors sha256 mismatch: expected $expected got $actual"
echo "    ok"

# --- 3. convert with the pinned converter -------------------------------------
# Python 3.12: the converter's pinned torch wheels lag the newest CPython.
convert() {
    local outtype="$1" outfile="$2"
    echo "==> converting -> $(basename "$outfile")"
    uv run --no-project --python 3.12 \
        --with-requirements "$llama_src/requirements/requirements-convert_hf_to_gguf.txt" \
        python "$llama_src/convert_hf_to_gguf.py" "$hf_dir" --outtype "$outtype" --outfile "$outfile" \
        > "$outfile.convert.log" 2>&1 || { tail -20 "$outfile.convert.log" >&2; die "conversion failed"; }
}
convert f16  "$out_dir/$NAME-F16.gguf"
convert q8_0 "$out_dir/$NAME-Q8_0.gguf"

# --- 4. record hashes of everything -------------------------------------------
(cd "$out_dir" && sha256sum hf/"$NAME"/*.json hf/"$NAME"/model.safetensors "$NAME"-*.gguf > SHA256SUMS)
echo "==> done; hashes in $out_dir/SHA256SUMS"
ls -la "$out_dir"/*.gguf
