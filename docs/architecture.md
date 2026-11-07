# neuronaut — architecture and methods

How the pieces fit together, the maths behind each tool, and the things we learned
the hard way. The README covers building and using it; `examples/` covers
what we found with it.

## 1. Shape of the system

```
                   ┌───────────────────────── GGUF file (mmap, read-only) ─────────────────────────┐
                   │ header + metadata │ tensor directory │ tensor data (F32 / F16 / Q8_0 / …)       │
                   └─────────┬─────────────────┬──────────────────────┬──────────────────────────────┘
                             │                 │                      │
                    gguf_file.*          model_layout.*          neuron.* / blocks.*
                 (ggml's parser +       (arch hyperparams,       (dequantize rows / columns,
                  bounds-checked         tensor roles per         byte footprints, one quant
                  mmap views)            layer)                   block decoded)
                             │                 │                      │
          ┌──────────────────┴────────┬────────┴──────────┬───────────┴──────────────┐
          │                           │                   │                          │
     lens.* (static)            forward.* (dynamic)   oracle.* (libllama)        blame.*
     W_U·(g⊙v), reverse lens    our ggml Llama graph  tokenizer + reference      exact split of one
     (token → writer neurons)   + interventions +     logits for the tests       logit into embed +
                                activation capture                               attn + every neuron
          │                           │                   │                          │
          └───────────────┬───────────┴───────────────────┴──────────────────────────┘
                          │
                 tui.* — Session (state machine, no terminal knowledge)
                          │                    │
                 run_tui (FTXUI loop)    render_to_string (headless frames: tests, `neuronaut render`)
                          │
                 main.cpp — also the scriptable subcommands (info, lens, find, run, oracle, blame, rome, …)
```

Everything except `tui.*` and `main.cpp` is a library (`neuronaut_core`,
`neuronaut_llama`) with no UI dependencies, so the CLI, the TUI and the tests run
the same code.

## 2. Dependencies and pinning

- **llama.cpp `b11700`** (2026-10-16) and **FTXUI `v7.0.3`** come in through CMake
  `FetchContent` from release tarballs pinned by SHA-256 (`URL_HASH`). A re-pointed
  tag or tampered tarball fails the configure.
- `EXCLUDE_FROM_ALL` keeps llama.cpp's tools, examples and server out of the build;
  we link `ggml` (core) and `llama` (only for `oracle.*`).
- **`patches/`** holds local fixes applied at fetch time (`PATCH_COMMAND`), one file
  each with a header explaining why. Currently one: `src/models/clef.cpp` uses
  `std::find` without `<algorithm>`, which GCC 16 rejects.
- Model weights are never committed (`models/` is gitignored; Meta's license).
  `scripts/fetch-llama.sh` builds them: pinned HF commit → `model.safetensors`
  sha256 checked against the LFS oid HF reports for that commit → converted by the
  *pinned* llama.cpp's `convert_hf_to_gguf.py` → `models/SHA256SUMS`. The repo's
  pickle checkpoint (`original/*.pth`) is never downloaded. The HF token's account
  must have accepted the gated repo's license.

## 3. The GGUF view (`gguf_file.*`, `model_layout.*`)

- ggml's own parser reads the header, metadata and tensor directory with
  `no_alloc = true`, which also gives us ggml's `ne[]`/`nb[]` for every tensor type.
- The whole file is `mmap`ed `PROT_READ, MAP_PRIVATE`, so a tensor byte is a pointer
  dereference and nothing can write through to the user's file.
- Every tensor's offset + size is checked against the file size before a pointer is
  formed. A truncated or hostile file fails to open; it never reads out of bounds.
- `ModelLayout` maps `general.architecture` + `<arch>.*` keys + the tensor naming
  scheme (`blk.L.ffn_up.weight`, …) to roles. Implemented: `llama` and `qwen35`. Other
  architectures load with a warning and the static views work if the names match.
- **Per-layer kind.** Every layer has a `LayerKind`: `attention` (Llama; Qwen3.5's
  every-fourth full-attention layer) or `linear_attention` (Qwen3.5's Gated DeltaNet
  layers), from `qwen35.attention.recurrent_layers` (falling back to
  `full_attention_interval`, as llama.cpp does). Each kind has its own tensor set;
  the MLP half is the same SwiGLU in both, so neurons mean the same thing.
- **`n_layer` vs `n_block`.** Qwen3.5's `block_count` includes the multi-token-
  prediction head (`nextn_predict_layers` blocks at the end, which also carry
  `ffn_*` tensors). They are not part of the forward pass: `n_layer` excludes them
  and `mtp_layers` lists them.
- **Qwen3.5 norm gains are stored `1 + w`.** HF's Qwen3.5 RMSNorm multiplies by
  `1 + weight`; llama.cpp's converter bakes the `+1` into every `*norm.weight` except
  the linear-attention `ssm_norm` (`conversion/qwen.py`). Use gains as stored; adding
  1 again doubles them. Confirmed against an independent numpy lens.
- **Hadamard-folded GGUFs are refused for neuron views.** llama.cpp supports
  `prism.hadamard.*` files whose weights have a random rotation folded into each
  matmul's *input*; for `ffn_down` that axis is the neurons, so a stored column is a
  mix of many real neurons. `ModelLayout::hadamard_folded` flags them and
  `read_neuron` throws. Our converter runs never produce them.

### What a neuron is on disk

ggml order is `ne[0]` = contiguous. For neuron *n* of layer *L*:

| tensor | ggml shape | neuron *n* | consequence |
|---|---|---|---|
| `ffn_gate`, `ffn_up` | `[n_embd, n_ff]` | row *n* | contiguous; every quant block in it belongs to *n* alone |
| `ffn_down` | `[n_ff, n_embd]` | element *n* of every row | strided; each block shared with `block_size − 1` other neurons |

`read_col` decodes only the one block per row that contains column *n*.
`Footprint` reports this so the UI can show it. It's also why byte-level patching
(M4) will target the **up row**: zeroing its block scales ablates the neuron
exactly, and scaling them scales its activation exactly, with no re-quantization.

## 4. The forward pass (`forward.*`)

We build the Llama graph ourselves in ggml rather than calling llama.cpp, because a
debugger has to *write* mid-run: every intermediate is a node we can read back or
splice an op into. It mirrors llama.cpp's `llm_build_llama` (`src/models/llama.cpp`
at the pinned tag):

```
x = token_embd[tokens]
per layer:
  h = rms_norm(x)·attn_norm
  q,k,v = Wq h, Wk h, Wv h               GQA: 32 q heads, 8 kv heads (head h uses kv head h/4)
  q,k = rope(q,k)                        "normal" mode; llama3 frequency factors from rope_freqs.weight
  k,v → f16, padded to n_kv              as llama.cpp's KV cache stores them (see §8)
  x += Wo · softmax(q kᵀ/√d + causal) v
  h = rms_norm(x)·ffn_norm
  a = silu(Wgate h) ⊙ (Wup h)            the neurons, n_ff wide
  a = a ⊙ scale + add                    interventions — only on layers that have any
  x += Wdown · a
logits = W_U · (rms_norm(x)·output_norm)  W_U = token_embd when tied (Llama 3.2 1B)
```

**Qwen3.5 (`qwen35`)** mirrors `llama_model_qwen35::graph` (`src/models/qwen35.cpp`
+ `delta-net-base.cpp`). Same skeleton and the same MLP — so neurons and the
intervention splice are unchanged — with the attention half chosen per layer kind
(`Forward::attn_qwen35_full` / `attn_qwen35_linear`):

```
full attention (L3, L7, … every 4th):
  [q | gate] = Wq h                      per head: 256 dims of query, then 256 of output gate
  q = rms_norm(q)·q_norm,  k = rms_norm(Wk h)·k_norm      per head
  q,k = M-RoPE, interleaved (IMROPE)     sections 11/11/10 over the first 64 of 256 dims;
                                         text token at p → positions [p, p, p, 0]
  o = softmax(q kᵀ/√d + causal) v ⊙ sigmoid(gate);  attn_out = Wo·o     (Wo: 4096 → n_embd)
linear attention (Gated DeltaNet, the other 3 of 4):
  qkv = Wqkv h,  z = Wgate h
  qkv = silu(causal_conv1d_k4(qkv))      conv state starts at zero (K−1 zero frames)
  q,k = l2_norm per head                 16 key heads, tiled over 32 value heads (head i ↔ i % 16)
  β = sigmoid(Wβ h),  g = softplus(Wα h + dt_bias)·A     A = −exp(A_log), stored by the converter
  ggml_gated_delta_net: per token S ← e^g S;  S ← S + k ⊗ β(v − Sᵀk);  o = Sᵀq/√d   (S₀ = 0)
  attn_out = Wout · (rms_norm(o)·ssm_norm ⊙ silu(z))     ssm_norm is the one gain NOT +1-baked
then: x += attn_out; h = rms_norm(x)·post_attention_norm; MLP as above
```

Everything `blame` needs carries over: `attn_out` is still one [n_embd, T] write per
layer, so the logit decomposition is unchanged (it closes to ~0.01 on Qwen).

- **Execution goes through `ggml_backend_sched`** (since Q2), the scheduler llama.cpp
  uses: the CPU backend, or a GPU backend (first in line, owning the weights) with
  the CPU as fallback for any op the GPU lacks. Device: `NEURONAUT_DEVICE=gpu` (or
  `cuda`), default CPU — the reference.
- **Weights aren't copied on CPU.** One `ggml_backend_cpu_buffer_from_ptr` over the
  whole read-only mmap; each weight tensor is bound at its own address in it. On a GPU
  they are uploaded once (9B BF16 ≈ 18 GB).
- **Intermediates are scheduler-allocated and reused,** except what the caller asked
  to capture: those are marked outputs (`ggml_set_output`) so the allocator never
  hands their memory to a later node. The captures: `acts` (pre-intervention `a` per
  requested layer), `embed`, `attn_out` per layer, `layer_resid` after each layer,
  `final_resid`.
- **Interventions** are `[n_ff]` scale/add vectors broadcast over positions. An empty
  `Interventions` produces exactly the reference graph, which the oracle tests rely on.
- **Every forward is a decode into a `DecodeCache`.** It holds, per layer, an f16 K/V
  cache (softmax-attention layers) or the conv window + DeltaNet state (linear
  layers), all zero-initialised in a backend buffer. `run(tokens)` decodes everything
  into a throwaway cache; generation (`run -n`, `:gen`) keeps one and decodes a token
  at a time. One code path, so the cached path is the one the oracle tests. Inside
  the graph, new K/V are `ggml_cpy`'d into cells `[n_past, n_past+T)` and the conv /
  DeltaNet states are copied back after use — each write is expanded into the graph
  before the reads that depend on it, as llama.cpp does.
- **A cache belongs to one set of patches.** `decode` throws if the interventions
  differ from the ones the sequence started with (cached K/V and states were computed
  under them). Changing a patch means a new cache.
- **Speed** (2026-11-01, 32 CPU threads, BF16, short prompts): Qwen3.5-4B ≈ 0.4 s per
  full forward, one dose sweep ≈ 24 s; a 40-token greedy answer 4.9 s with the cache
  (14.9 s before), 9B 8.1 s (22.5 s). GPU numbers below.

**GPU (CUDA).** Build with `-DGGML_CUDA=ON` (on the GPU box, an RTX 5090:
`PATH=/usr/local/cuda/bin:$PATH cmake -S . -B build -G Ninja -DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=120`; ~10 min for ggml's kernels) and run with
`NEURONAUT_DEVICE=gpu`. The weights are uploaded once; the decode cache lives in device
memory; the scheduler puts any op the GPU lacks on the CPU. Qwen3.5-9B BF16 on the 5090,
steady state through MCP: a 40-token answer 0.6 s (≈ 15 ms/token), a 153-forward dose
sweep 3.6 s, model load ≈ 4.5 s. The same 40 tokens take 111 s on the GPU box's CPU (an
AVX2 EPYC with no native BF16) and 8 s on this laptop's (AVX-512 BF16).

- **Exactness on GPU** (oracle against llama.cpp on the same GPU): prompt positions are
  bit-identical on every model; single-token cached steps differ slightly (|Δ| ≤ 0.004
  BF16, ≤ 0.30 Q8_0, which re-quantizes activations), argmax identical throughout.
  Likely a batch-1 kernel/fusion choice in llama.cpp's CUDA path; not chased. The CPU
  stays the reference; compare doses within one device.
- **The GPU box's copy** is a plain source sync of this tree (`rsync -rlpc --exclude build
  --exclude models`, by checksum without preserving mtimes — with `-a`, synced files
  kept older mtimes than the GPU box's objects and ninja silently didn't rebuild them) plus
  the GGUFs from `models/` (same SHA256SUMS). No host configuration involved.
- **Causality shortcut.** Logits at position *p* depend only on `tokens[0..p]`, so the
  dose sweep runs just the prefix.

## 5. Static lens and reverse lens (`lens.*`)

- **Lens of a vector *v*:** `Δlogits = W_U · (g ⊙ v)`, where *g* is the final RMSNorm
  gain. Applied to a neuron's down column, it answers "if this went straight to the
  output, which tokens would it push?". It ignores every later layer and the norm's
  scale, so it's a hint, not a measurement. Threaded over vocab rows (~50 ms for 128k).
- **Reverse lens (`find`):** for token *t*, `u = g ⊙ W_U[t]`, then `down_ln · u` and
  cosine for every neuron in every layer. Computed as `W_downᵀ·u` by streaming the
  contiguous rows of each `ffn_down`, one thread per layer (~70 ms for all 131k).
- **Lens of the residual after layer *L*** (STATE pane): `x_L / rms(x_L)` through the
  same projection, with full softmax for the target's probability and rank.
- **The sign lesson.** A lens says what a neuron *writes*, not which way it is *used*:
  L15.N7279 "writes France/Paris" yet fires negative right before " Paris" and
  suppresses it. Only a forward pass gives the sign.

## 6. Blame: direct logit attribution (`blame.*`)

The final residual is a plain sum, so one logit splits exactly:

```
x_final  = embed + Σ_l attn_out_l + Σ_l Σ_n a′_ln · down_ln          (a′ = after interventions)
logit_t  = u · x_final / rms(x_final)                                u = g ⊙ W_U[t]

embed:    u · embed      / rms
attn l:   u · attn_out_l / rms
neuron:   a′_ln · (u · down_ln) / rms
```

- `unexplained = logit − Σ parts` is shown in the UI and checked by ctest (< 0.25;
  measured ~0.002). The gap is ggml's reduced-precision matmul inputs against our
  f32 sums. **If it grows, something in the graph or the captures is wrong.**
- **Limit:** blame credits *direct* effects only. A neuron in layer 11 also changes
  what layers 12–15 compute, and that effect is credited to those later pieces.
  Blame is exact about the last step and a lead for early layers. Confirm with an
  intervention.
- `down_dots` (`u · down_ln` for all neurons) is cached per target token, so moving
  the cursor between positions costs only the cheap per-position part.

## 7. Dose–response with dose-matched controls (`tui.cpp`, `start_dose`)

Question: is this neuron's effect *specific*, or would any push that size do it?

1. Doses k ∈ {−3, −2, −1, −0.5, 0, 0.5, 1, 2, 3}: the neuron's activation is
   multiplied by k at every position (on top of existing patches). Measure
   P(target at pos). k = 1 is the reference; every Δ is relative to it.
2. **Controls:** 16 neurons from the same layer whose *push* `|a| · ‖down‖` at that
   position is closest (in log space) to the tested neuron's, excluding patched
   neurons. Same sweep for each.
3. **Verdict:** at each dose, the tested Δ is "outside" if it lies outside the
   controls' Δ range **and** |Δ| ≥ 1 pt (otherwise a 0.01-pt wiggle outside a band of
   near-zero controls would count). The majority of doses decides the headline.
4. **Honesty checks:** a warning when the neuron pushes > 1.5× harder than any
   control (no same-sized peers exist, so the band understates); the controls are
   listed by their own ablation effect, because a control that moves the target as
   much as the tested neuron is a lead in itself (that's how L11.N177 was found).

It runs on a worker thread (§9); about 150 forwards ≈ 9 s on the 1B. Tests pin both
verdicts: L15.N7279 is 8/8 outside, random L15.N4000 is 0/8.

## 7½. Rewriting a fact: ROME (`rome.*`, `Forward::delta_loss`)

`neuronaut rome` implements Meng et al. 2022 (rome.baulab.info): a fact about a
subject is a key → value pair stored in one MLP layer's down projection, read at
the subject's last token, so one rank-one write to `ffn_down` can replace it.

- **Key** `k*`: `a` at the layer, at the subject's last token, averaged over a few
  sentence openings (fixed, so runs are reproducible).
- **Value search**: Adam on a vector δ added to the residual right after the layer
  at that token. Loss: mean NLL of each prompt's target, plus `kl ×` KL(before ‖
  after) on the essence prompts' next token, plus weight decay; ‖δ‖ is clamped to
  `clamp ×` ‖W_down·k*‖. The gradient comes from ggml's autograd. `delta_loss`
  builds a separate graph for layers L+1 … n−1: no KV cache, f32 keys and values,
  every sequence packed into one batch with a block-diagonal causal mask and RoPE
  positions that restart per sequence. It runs on f32 copies of the weights (made
  once, ~4 GB for the 1B), because ggml's backward for an f16 or quantized matmul
  re-dequantizes the whole matrix once per token. CPU only, llama-style layers only.
- **Covariance** C = E[a aᵀ] at the layer over corpus tokens (BOS skipped), with
  forwards stopped after the layer (`RunOptions::last_layer`), accumulated as
  2048-token Aᵀ·A matmuls. Cached beside the model as `.rome-cov-L<l>` (n_ff² f32).
  `(C + λI)⁻¹k*` is solved by conjugate gradients.
- **The edit**: `Interventions::RankOne{u = δ / (k·w̃), w = w̃ = C⁻¹k*}`, so
  W_down′·k = W_down·k + δ for the prompt's own key. The forward graph adds
  `mul_mat(u, mul_mat(w, a))` after the down projection. Saved as edit op `rank-one`
  (base64 float32). `export` rewrites every row of `ffn_down` (dequantize, add,
  re-quantize).
- **Tests** (`test_forward`): u = 0 is a no-op bit for bit; the edit equals adding
  u·(w·a) to the residual; `delta_loss`'s NLL at δ = 0 matches the forward pass, and
  its gradient matches central finite differences.

Takes ~10 min on a 24-thread laptop CPU for 24 training prompts, plus a one-off
covariance pass. Worked example: `examples/socrates-car-crash.md`.

## 8. Correctness: the oracle

`neuronaut oracle MODEL "prompt" [--exact] [--gen N]` runs the same tokens through our
graph and through libllama, and compares the full logits at every position (argmax,
max |Δ|, and how many positions are bit-identical; `--exact` requires all of them).
`--gen N` tests the cached path: we decode the prompt into a cache and then N greedy
tokens one decode at a time; llama.cpp gets the same tokens the same way (prompt in
one batch, then single-token `llama_decode` calls carrying its KV cache and recurrent
state). The oracle runs llama.cpp on the same device as our graph.

**Result: bit-identical** on every model we have — Llama 3.2 1B F16 and Q8_0,
Qwen3.5-4B and 9B in BF16 and Q8_0, 9B-Base — at every position, including a
386-token prompt (KV padded to 512) and a Chinese chat prompt. Mutating the Qwen graph
(dropping the attention output gate, silu(z), or the conv's silu) gives 8–13/13
argmax flips, so the check has teeth. Two things make it exact:

- **K/V rounded to f16**, as llama.cpp's KV cache stores them. Without it, Q8_0's
  re-quantization of every matmul input amplified the f32-vs-f16 difference into
  argmax flips on near-ties (worst 0.39).
- **KV length padded like llama.cpp's cache:** keys run over
  `n_kv = max(256, round_up(T, 256))` cells (`llama_kv_cache::get_n_kv`), the extra ones
  zero and masked to −inf. Mathematically a no-op. Before it (2026-11-01), logits were
  exact for the first few positions of some prompts and then drifted 0.03 (BF16) to
  0.2 (Q8_0) — deterministic and content-dependent. The cause: the V·softmax product is
  an f16 dot product over the key axis, and ggml's f16 dot runs a SIMD loop over
  multiples of its width plus a scalar tail with a different accumulator; a length-T row
  (T < 32) is mostly tail, llama.cpp's length-256 row is all SIMD. One ulp of
  difference, then the next BF16/Q8_0 matmul re-rounds its input and amplifies it.

Both are kept permanently: they are what every llama.cpp-based server computes.

**Cached decoding is bit-identical too** (`--gen`): 24 single-token steps after the
prompt on Qwen3.5-4B/9B and Llama, BF16/F16 and Q8_0, and a 252-token prompt + 20
steps that crosses the 256 → 512 cell boundary mid-generation.
- The static side was verified once with an independent numpy reader over Meta's
  original bf16 safetensors (no shared code with neuronaut, the converter or ggml):
  same writer neurons, same cosines to 3 dp, same lens orderings.

## 9. The UI (`tui.*`)

**Model.** OllyDbg's: one screen, fixed panes, one cursor. The cursor is
`(pos_, layer_, neuron_)`. Every kind of object has one home pane, and selecting an
object anywhere (LIST row, PATCHES entry) moves the cursor to it (*live follow*), so
its home panes update in place. Press `?` in the UI for the keys.

**Separation.** `Session` holds all state and knows nothing about terminals:

- `exec(line)` — commands (the same ones a script or a future MCP front end would send);
- `key(k)` — named keys (`"h"`, `"tab"`, `"shift_tab"`, `"pgdn"`, …);
- `render()` → an FTXUI element.

`run_tui` maps terminal events to keys and owns the command line (Input component,
history, Tab). `render_to_string` draws a frame headlessly. `neuronaut render
--cmd … --key … --type … --tab` drives a Session from the CLI and prints a frame;
that is how the UI is tested without a terminal.

**One command table** (`commands()`) drives Tab completion, the live hint line and
the help overlay, so they can't drift apart. Argument completion is contextual:
neurons from the cursor, patches, watches, dose and blame; tokens from the decoded
vocabulary (valid UTF-8, de-duplicated); prompts from history.

**Threading.**

- Everything runs on the UI thread except the dose sweep. Its worker gets its inputs
  by value (token prefix, interventions, target), runs forwards on the shared const
  `Forward` (each run has its own ggml context), and hands every result back as a
  closure through `Session::post` → `App::Post`, so all state mutation happens on the
  UI thread.
- While it runs, commands and keys that would change what it measures are refused
  ("busy"). `:cancel` sets an atomic flag. The destructor cancels and joins.
- Headless (`post` unset), the same job runs inline.

**Sized for 179×53** (the user's terminal). Middle row: GRAPH 36 · LIST 72 · NEURON
flex. Bottom row (15 high): BYTES + PATCHES 108 · STATE flex.

## 10. Tests (`ctest --test-dir build`)

| test | checks |
|---|---|
| `mkfixture` + `neuron_{f32,f16,q8_0}` | synthetic llama-shaped GGUFs whose every weight is a closed-form function; every neuron's gate/up rows and down column must match the formula within the type's error; footprints and block dumps |
| `render_frame` | a headless frame renders (fixture, static mode) |
| `oracle_llama32_{f16,q8_0}` | our logits vs llama.cpp at every position |
| `blame_closes_f16{,_ablated}` | the logit decomposition closes, with and without interventions |
| `dose_specific_l15n7279`, `dose_null_l15n4000` | the dose verdict says yes for a known suppressor and no for a quiet random neuron |
| `oracle_qwen35_4b_{bf16,q8_0,bf16_zh}` | Qwen3.5 forward pass bit-identical to llama.cpp (raw, Q8_0, Chinese text); the Llama oracles also run `--exact` now |
| `oracle_qwen35_4b_q8_0_gen`, `oracle_llama32_q8_0_gen_boundary` | cached decoding (prompt, then single-token decodes) bit-identical to llama.cpp's own incremental decoding, incl. across the 256-cell KV boundary |
| `oracle_qwen35_4b_gpu` | the GPU path vs llama.cpp on the same GPU (argmax + |Δ| < 0.5); skipped without a GPU backend |
| `blame_closes_qwen35_4b{,_chat_ablated}` | blame closes on Qwen (|unexplained| < 0.05), raw prompt and chat prompt with two ablations |
| `qwen35_static_{4b,9b}` | Qwen3.5 layout (24 linear + 8 full attention, MTP block excluded), the static lens of the Qwen study's neurons (deception, Tibet, Taiwan, Xinjiang, repression; 4B exception/malice), no-BOS tokenization and the thinking-off chat wrap |

Model-backed tests exit 77 (SKIP) when `models/` is empty, and are `RUN_SERIAL`:
ggml's thread pool busy-waits, so two at once thrash each other ~30× slower.

## 11. Gotchas we hit (so you don't again)

- **FTXUI `canvas(w, h, fn)` calls `fn` later, during layout.** Capturing locals by
  reference segfaulted; capture by value.
- **Byte-length padding misaligns columns with multibyte glyphs** (`▾`, `×`, `…`).
  Use `size(WIDTH, EQUAL, n)` cells, not `std::string::resize`.
- **ggml thread pools busy-wait.** Never run two model workloads concurrently
  (ctest `-j`, or a UI action during a sweep).
- **`hf download` (new CLI) takes one pattern per `--include`.** A second bare
  pattern is read as a filename.
- **Llama 3 vocab is byte-level BPE** (`Ġ` = space, `Ċ` = newline). Decode before
  display (`bpe_decode`); some tokens are partial UTF-8 and must be escaped.
- **Most words are tokens *with a leading space*:** `" Paris"`, not `"Paris"`.
- **llama.cpp's "CPU" mode isn't pure CPU when a GPU exists.** With `n_gpu_layers = 0`
  it still offloads host-weight ops and KQV to the GPU; on the GPU box that moved BF16
  logits by 0.06. The oracle passes an empty device list and disables `op_offload` /
  `offload_kqv` in CPU mode.
- **Two copies of a 9B don't fit a 32 GB GPU.** The oracle frees our model before
  llama.cpp loads its own.
- **ggml's CPU thread pool defaults to every hardware thread.** Fine on a laptop, but
  128 busy-waiting threads on the GPU box make the 1B dose tests ~15× slower than on 32
  threads. Use the GPU there, or `mcp --threads N`.
- **Pad the KV length the way llama.cpp does, or lose bit-exactness.** See §8: ggml's
  f16 dot product handles a short row (mostly scalar tail) differently from a padded
  one (all SIMD), and BF16/Q8_0 matmuls amplify that ulp. Masked zero padding to
  `max(256, round_up(T, 256))` makes us identical.
- **Qwen3.5's Wq is twice as wide as it looks.** It outputs query *and* output gate,
  interleaved per head (256 + 256), so attention width is 2·16·256 = 8192 on the 4B
  whose n_embd is 2560 — size buffers from head_dim·n_head, not n_embd.
- **Probe prefixes can merge with the text.** Qwen3.5 tokenizes `"Text:The cat"` as
  `"Text"`, `":The"`, … — the colon fuses with the first word — while `"Text: The"`
  gives `":"`, `" The"`. Tokenizing a prefix separately and skipping that many
  tokens of the joined string drops or mangles the first word either way unless
  the boundary is a clean token edge. Check the joined tokens, not the parts.
- **`printf("%-Ns")` pads bytes, not cells.** A CJK character is 3 bytes and 2
  cells, so every column after one drifts. CLI output pads with `pad_cells`
  (`format.*`); FTXUI measures cells itself.
- **Qwen3.5 has no BOS** (`tokenizer.ggml.add_bos_token = false`); `encode(…, true)`
  follows the vocab's flag, so it's right for both families without a branch.
- **The converter warns that Transformers doesn't know `qwen3_5`** and falls back to
  reading `config.json` directly. Harmless: the layout and all checks come out right.
- **libllama prints load progress dots to stderr,** which scribble over a
  fullscreen TUI; silenced with a progress callback.

## 12. Known limits and next steps

- **Dose sweeps don't reuse anything.** Each of the ~150 forwards recomputes the prefix
  from scratch (cheap on the GPU: 3.6 s on the 9B). Starting each from the cached
  residual at the patched layer would cut CPU time roughly by the fraction of layers
  below it.
- **Attention is one row per layer.** A per-head split is next: L15 attention is the
  largest single contributor to " Paris".
- **No xrefs yet** (which later neurons read a neuron's output), which is needed to
  chase indirect-effect neurons like L11.N177.
- **Forward pass: `llama` and `qwen35`** (dense; no MoE). Per-head attention views for
  Qwen will need an HF↔GGUF head map in linear layers (V heads are reordered). Only
  Q8_0 / Q4_0 / float block layouts are annotated in BYTES
  (others still show raw bytes + values).
- **M4 not started:** byte patches written to a new GGUF with the edit log in
  metadata; control-vector export.
