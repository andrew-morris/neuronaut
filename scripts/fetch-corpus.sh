#!/usr/bin/env bash
# fetch-corpus.sh — a small, diverse text corpus for `neuronaut index` (the
# top-activating-examples index, src/actindex.h).
#
#   scripts/fetch-corpus.sh [N_DOCS]        (default 5000)
#
# Source: the first validation shard of AllenAI's C4 (English web text, cleaned),
# pinned to a dataset commit and checked against the sha256 HF reports for that
# commit's LFS object. Diverse enough that a neuron's top examples say something
# about it; small enough (40 MB) to fetch anywhere. Only gzip + json from the
# Python standard library touch it: no parquet, no pickle, no third-party code.
#
# Output (gitignored, under models/):
#   models/corpus/c4-validation.00000-of-00008.json.gz   the shard as published
#   models/corpus/c4-val-<N>.jsonl                       its first N documents, {"text": …} per line
#   models/corpus/SHA256SUMS
set -euo pipefail

REPO="allenai/c4"
REVISION="1588ec454efa1a09f29cd18ddd04fe05fc8653a2"   # main as of 2026-11-04
FILE="en/c4-validation.00000-of-00008.json.gz"
SHA256="1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f"   # LFS oid at $REVISION
N="${1:-5000}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$here/models/corpus"
gz="$out/$(basename "$FILE")"
jsonl="$out/c4-val-$N.jsonl"
mkdir -p "$out"

if [ ! -f "$gz" ] || ! echo "$SHA256  $gz" | sha256sum -c --quiet - 2>/dev/null; then
    echo "fetch-corpus: downloading $REPO/$FILE @ ${REVISION:0:12}"
    curl -fL --retry 3 -o "$gz.part" "https://huggingface.co/datasets/$REPO/resolve/$REVISION/$FILE"
    echo "$SHA256  $gz.part" | sha256sum -c --quiet - || { echo "fetch-corpus: sha256 mismatch for $FILE" >&2; rm -f "$gz.part"; exit 1; }
    mv "$gz.part" "$gz"
fi

# First N documents, text only (C4 rows also carry url + timestamp). -I: isolated
# mode, so nothing in the current directory is importable.
python3 -I - "$gz" "$jsonl" "$N" <<'PY'
import gzip, json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with gzip.open(src, "rt", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
    for i, line in enumerate(f):
        if i >= n:
            break
        o.write(json.dumps({"text": json.loads(line)["text"]}, ensure_ascii=False) + "\n")
PY
(cd "$out" && sha256sum "$(basename "$gz")" "$(basename "$jsonl")" > SHA256SUMS)
echo "fetch-corpus: $(wc -l < "$jsonl") documents → $jsonl"
