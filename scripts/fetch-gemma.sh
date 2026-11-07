#!/usr/bin/env bash
# fetch-gemma.sh — Gemma 3 / Gemma 4 checkpoints → verified → our own GGUFs.
#
#   HF_TOKEN=hf_xxx scripts/fetch-gemma.sh            # both: gemma-3-4b-it, gemma-4-12B-it
#   HF_TOKEN=hf_xxx scripts/fetch-gemma.sh 3-4b       # one
#   QUANTS="bf16" scripts/fetch-gemma.sh 4-12b        # skip Q8_0
#
# Same provenance chain as fetch-qwen35.sh: the HF repo pinned to a commit, every
# file checked against the hash HF's tree reports at that commit (sha256 for LFS
# files, the git blob sha1 for small files), converted by the pinned llama.cpp's
# convert_hf_to_gguf.py, SHA256SUMS written for everything in models/.
#
# Licences:
#   gemma-3-4b-it   Gemma Terms of Use, *gated*: the token's account must have
#                   accepted them (HF masks the tree's hashes without a token, so the
#                   token is needed for verification too):
#                   HF_TOKEN=... scripts/fetch-gemma.sh
#   gemma-4-12B-it  Apache-2.0, ungated.
#
# Why these two: gemma-3-4b-it is the size Gemma Scope 2 covers; gemma-4-12B-it is
# the smallest *uniform* Gemma 4 (the E2B/E4B "effective" models have per-layer
# embeddings and a double-wide MLP on their KV-shared layers, so their neuron count
# varies by layer, which neuronaut doesn't model). Both checkpoints are multimodal;
# the converter writes the text model only (the vision/audio towers would go to an
# --mmproj file, which we don't make).
#
# Output (gitignored):
#   models/hf/<name>/                 verified source files
#   models/<name>-BF16.gguf           reference precision
#   models/<name>-Q8_0.gguf
#   models/SHA256SUMS
set -euo pipefail

declare -A REPO=(
    [3-4b]=google/gemma-3-4b-it
    [4-12b]=google/gemma-4-12B-it
)
# Pinned commits (HF `sha` as of 2026-11-04).
declare -A REVISION=(
    [3-4b]=093f9f388b31de276ce2de164bdc2081324b9767    # 2025-03
    [4-12b]=43c9d014b231474fa7212dbb48a762566c2b255d   # 2026-05
)
# The converter's pinned transformers (4.57.6) predates Gemma 4: its tokenizer
# loader can't read Gemma 4's tokenizer_config.json (extra_special_tokens is a list).
# For those, the same pinned requirements with only transformers swapped.
declare -A TRANSFORMERS=(
    [4-12b]=5.20.0    # 2026-10-15
)
QUANTS="${QUANTS:-bf16 q8_0}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_src="$here/build/_deps/llama_cpp-src"
out_dir="$here/models"

die() { echo "fetch-gemma: $*" >&2; exit 1; }
[[ -f "$llama_src/convert_hf_to_gguf.py" ]] || die "no pinned llama.cpp at $llama_src — run: cmake -S . -B build"
command -v uv >/dev/null || die "uv not found"

models=("$@")
[[ ${#models[@]} -gt 0 ]] || models=(3-4b 4-12b)

# Files the converter needs (text model + tokenizer + chat template).
want_re='^(config\.json|generation_config\.json|tokenizer\.json|tokenizer\.model|tokenizer_config\.json|special_tokens_map\.json|added_tokens\.json|chat_template\.(json|jinja)|model\.safetensors\.index\.json|model(-[0-9]+-of-[0-9]+)?\.safetensors|LICENSE)$'

auth=()
[[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

tree() {
    local repo="$1" rev="$2"
    curl -fsS -m 60 "${auth[@]}" "https://huggingface.co/api/models/$repo/tree/$rev" |
        python3 -I -c '
import json, re, sys
want = re.compile(sys.argv[1])
for f in json.load(sys.stdin):
    if f.get("type") != "file" or not want.match(f["path"]):
        continue
    if f.get("lfs"):
        if "*" in f["lfs"]["oid"]:
            sys.exit("hashes are masked: this repo is gated and needs HF_TOKEN")
        print("lfs", f["lfs"]["oid"], f["path"])
    else:
        print("git", f["oid"], f["path"])
' "$want_re"
}

git_blob_sha1() {
    python3 -I -c '
import hashlib, sys
d = open(sys.argv[1], "rb").read()
print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())
' "$1"
}

for key in "${models[@]}"; do
    repo="${REPO[$key]:-}"
    rev="${REVISION[$key]:-}"
    [[ -n "$repo" ]] || die "unknown model '$key' (want one of: ${!REPO[*]})"
    name="${repo#google/}"
    hf_dir="$out_dir/hf/$name"
    mkdir -p "$hf_dir"

    echo "==> $repo @ ${rev:0:12}"
    listing="$(tree "$repo" "$rev")" || die "$repo: tree listing failed"
    [[ -n "$listing" ]] || die "$repo: empty tree listing"

    # --- 1. download (hf skips files already there with the right size) ------------
    includes=()
    while read -r _ _ path; do includes+=(--include "$path"); done <<< "$listing"
    uvx --from 'huggingface_hub[cli]' hf download "$repo" --revision "$rev" \
        "${includes[@]}" --local-dir "$hf_dir" >/dev/null

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
    reqs="$llama_src/requirements/requirements-convert_hf_to_gguf.txt"
    if [[ -n "${TRANSFORMERS[$key]:-}" ]]; then
        tmp_reqs="$(mktemp -d)"
        # Flatten the -r includes, then pin transformers.
        (cd "$llama_src/requirements" && python3 -I - "$reqs" "${TRANSFORMERS[$key]}" > "$tmp_reqs/req.txt") <<'PY'
import os, re, sys
def lines(path):
    for l in open(path):
        l = l.strip()
        if l.startswith("-r "):
            yield from lines(os.path.join(os.path.dirname(path), l[3:].strip()))
        elif l and not l.startswith("#"):
            yield l
for l in lines(sys.argv[1]):
    print(f"transformers=={sys.argv[2]}" if re.match(r"transformers\b", l) else l)
PY
        reqs="$tmp_reqs/req.txt"
        echo "    converter: transformers==${TRANSFORMERS[$key]}"
    fi
    for q in $QUANTS; do
        case "$q" in bf16) tag=BF16 ;; q8_0) tag=Q8_0 ;; f16) tag=F16 ;; *) die "unsupported quant $q" ;; esac
        outfile="$out_dir/$name-$tag.gguf"
        echo "==> converting -> $(basename "$outfile")"
        uv run --no-project --python 3.12 \
            --with-requirements "$reqs" \
            python "$llama_src/convert_hf_to_gguf.py" "$hf_dir" --outtype "$q" --outfile "$outfile" \
            > "$outfile.convert.log" 2>&1 || { tail -20 "$outfile.convert.log" >&2; die "conversion failed"; }
    done
done

# --- 4. record hashes of everything in models/ ----------------------------------------
echo "==> hashing (this re-reads every file in models/; slow)"
(cd "$out_dir" && find hf -type f -not -path '*/.cache/*' \( -name '*.json' -o -name '*.safetensors' -o -name '*.txt' -o -name '*.jinja' -o -name '*.model' \) \
    | sort | xargs sha256sum > SHA256SUMS.new && sha256sum ./*.gguf | sed 's| \./| |' >> SHA256SUMS.new \
    && mv SHA256SUMS.new SHA256SUMS)
echo "==> done; hashes in $out_dir/SHA256SUMS"
ls -la "$out_dir"/*.gguf
