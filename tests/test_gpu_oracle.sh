#!/usr/bin/env bash
# test_gpu_oracle.sh — our graph on the GPU (NEURONAUT_DEVICE=gpu) vs llama.cpp on
# the same GPU, prompt + cached decode steps. Not --exact: CUDA kernels differ from
# CPU ones, and llama.cpp's batch-1 decode steps differ from ours by a little (see
# docs/architecture.md §8). Requires same argmax everywhere and |Δlogit| < 0.5.
#
# Exits 77 (ctest SKIP) when the model is absent or this build has no GPU backend
# (the normal case on a laptop; build with -DGGML_CUDA=ON, on a CUDA host).
#
# Usage: test_gpu_oracle.sh <neuronaut-binary> <model.gguf>
set -u
BIN="$1"
GGUF="$2"
if [ ! -x "$BIN" ]; then echo "no binary: $BIN"; exit 1; fi
if [ ! -f "$GGUF" ]; then echo "skip: $GGUF not present"; exit 77; fi

out=$(NEURONAUT_DEVICE=gpu "$BIN" oracle "$GGUF" "The capital of France is Paris, and the capital of Japan is" --gen 12 2>&1)
rc=$?
if grep -q "has no GPU device" <<<"$out"; then echo "skip: no GPU backend in this build"; exit 77; fi
grep -E "on CUDA|on Vulkan|on Metal|MATCH|MISMATCH" <<<"$out" | cut -c1-160
exit $rc
