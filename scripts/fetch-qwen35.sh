#!/usr/bin/env bash
# fetch-qwen35.sh — Qwen3.5 dense checkpoints → verified → our own GGUFs.
#
#   scripts/fetch-qwen35.sh                 # all of 4B, 9B, 9B-Base
#   scripts/fetch-qwen35.sh 4B 9B           # a subset
#   QUANTS="bf16" scripts/fetch-qwen35.sh   # skip Q8_0
#
# Same provenance chain as fetch-llama.sh, adapted to sharded, ungated repos:
#
#   HF repo pinned to commit $REVISION
#     -> every file we use is checked against the hash HF's git tree reports for
#        that commit: sha256 for LFS files (the weight shards), the git blob sha1
#        for small text files (config, tokenizer, chat template)
#     -> converted by llama.cpp b11700's convert_hf_to_gguf.py (hash-pinned tarball)
#     -> SHA256SUMS written for every input and output in models/
#
# Source: a local mirror ($NEURONAUT_MODEL_MIRROR/<name>/) when set and present, else a
# direct download from HF. Either way the bytes are verified against HF *at the
# pinned commit*, so the mirror is only a cache, never trusted. Qwen3.5 is
# Apache-2.0 and ungated: no token needed.
#
# The checkpoints are Qwen3_5ForConditionalGeneration (text + vision). The converter
# writes the text model only; the vision tower would go to an --mmproj file, which we
# don't make.
#
# Output (gitignored):
#   models/hf/Qwen3.5-<size>/                 verified source files
#   models/Qwen3.5-<size>-BF16.gguf           reference precision (= what neuron-explorer ran)
#   models/Qwen3.5-<size>-Q8_0.gguf           what llama.cpp servers run
#   models/SHA256SUMS
set -euo pipefail

# Pinned commits (HF `sha` as of 2026-11-01).
declare -A REVISION=(
    [4B]=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a       # 2026-03-02
    [9B]=c202236235762e1c871ad0ccb60c8ee5ba337b9a       # 2026-03-02
    [9B-Base]=68c46c4b3498877f3ef123c856ecfde50c39f404  # 2026-04-23
)
MIRROR="${NEURONAUT_MODEL_MIRROR:-}"
QUANTS="${QUANTS:-bf16 q8_0}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_src="$here/build/_deps/llama_cpp-src"
out_dir="$here/models"

die() { echo "fetch-qwen35: $*" >&2; exit 1; }
[[ -f "$llama_src/convert_hf_to_gguf.py" ]] || die "no pinned llama.cpp at $llama_src — run: cmake -S . -B build"
command -v uv >/dev/null || die "uv not found"

sizes=("$@")
[[ ${#sizes[@]} -gt 0 ]] || sizes=(4B 9B 9B-Base)

# Files the converter needs. Not fetched: README/LICENSE (kept anyway if present),
# preprocessor configs (vision only).
want_re='^(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|vocab\.json|merges\.txt|chat_template\.jinja|model\.safetensors\.index\.json|model\.safetensors-[0-9]+-of-[0-9]+\.safetensors|LICENSE)$'

# Print "<kind> <hash> <path>" for each wanted file in the repo tree at the revision:
# kind = lfs (hash = sha256) or git (hash = git blob sha1).
tree() {
    local repo="$1" rev="$2"
    curl -fsS -m 60 "https://huggingface.co/api/models/$repo/tree/$rev" |
        python3 -I -c '
import json, re, sys
want = re.compile(sys.argv[1])
for f in json.load(sys.stdin):
    if f.get("type") != "file" or not want.match(f["path"]):
        continue
    if f.get("lfs"):
        print("lfs", f["lfs"]["oid"], f["path"])
    else:
        print("git", f["oid"], f["path"])
' "$want_re"
}

# git blob sha1 of a file: sha1("blob <size>\0" + bytes)
git_blob_sha1() {
    python3 -I -c '
import hashlib, sys
d = open(sys.argv[1], "rb").read()
print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())
' "$1"
}

for size in "${sizes[@]}"; do
    rev="${REVISION[$size]:-}"
    [[ -n "$rev" ]] || die "unknown size '$size' (want one of: ${!REVISION[*]})"
    name="Qwen3.5-$size"
    repo="Qwen/$name"
    hf_dir="$out_dir/hf/$name"
    mkdir -p "$hf_dir"

    echo "==> $repo @ ${rev:0:12}"
    listing="$(tree "$repo" "$rev")"
    [[ -n "$listing" ]] || die "$repo: empty tree listing"

    # --- 1. stage: local mirror if set, else HF ----------------------------------
    if [[ -n "$MIRROR" && -d "$MIRROR/$name" ]]; then
        echo "    staging from $MIRROR/$name"
        while read -r _ _ path; do
            [[ -f "$hf_dir/$path" ]] && cmp -s "$MIRROR/$name/$path" "$hf_dir/$path" && continue
            cp --sparse=never "$MIRROR/$name/$path" "$hf_dir/$path"
        done <<< "$listing"
    else
        echo "    downloading from HF"
        includes=()
        while read -r _ _ path; do includes+=(--include "$path"); done <<< "$listing"
        uvx --from 'huggingface_hub[cli]' hf download "$repo" --revision "$rev" \
            "${includes[@]}" --local-dir "$hf_dir" >/dev/null
    fi

    # --- 2. verify every file against the pinned tree ------------------------------
    echo "    verifying $(wc -l <<< "$listing") files"
    while read -r kind hash path; do
        f="$hf_dir/$path"
        [[ -f "$f" ]] || die "$name: missing $path"
        if [[ "$kind" == lfs ]]; then got="$(sha256sum "$f" | cut -d' ' -f1)"
        else got="$(git_blob_sha1 "$f")"; fi
        [[ "$got" == "$hash" ]] || die "$name: $path hash mismatch ($kind): expected $hash got $got"
    done <<< "$listing"
    echo "    ok"

    # --- 3. convert with the pinned converter --------------------------------------
    # Python 3.12: the converter's pinned torch wheels lag the newest CPython.
    for q in $QUANTS; do
        case "$q" in bf16) tag=BF16 ;; q8_0) tag=Q8_0 ;; f16) tag=F16 ;; *) die "unsupported quant $q" ;; esac
        outfile="$out_dir/$name-$tag.gguf"
        echo "==> converting -> $(basename "$outfile")"
        uv run --no-project --python 3.12 \
            --with-requirements "$llama_src/requirements/requirements-convert_hf_to_gguf.txt" \
            python "$llama_src/convert_hf_to_gguf.py" "$hf_dir" --outtype "$q" --outfile "$outfile" \
            > "$outfile.convert.log" 2>&1 || { tail -20 "$outfile.convert.log" >&2; die "conversion failed"; }
    done
done

# --- 4. record hashes of everything in models/ (Llama's entries are kept) ----------
echo "==> hashing (this re-reads every file in models/; slow)"
# (hf download leaves a .cache/ of its own bookkeeping under each repo dir; not ours to hash)
(cd "$out_dir" && find hf -type f -not -path '*/.cache/*' \( -name '*.json' -o -name '*.safetensors' -o -name '*.txt' -o -name '*.jinja' \) \
    | sort | xargs sha256sum > SHA256SUMS.new && sha256sum ./*.gguf | sed 's| \./| |' >> SHA256SUMS.new \
    && mv SHA256SUMS.new SHA256SUMS)
echo "==> done; hashes in $out_dir/SHA256SUMS"
ls -la "$out_dir"/*.gguf
