# neuronaut — MCP front end (`neuronaut mcp`)

A fourth front end on the same `Session` the TUI, CLI and tests already drive, so an
LLM (Claude, over MCP) can run the debugger: move the cursor, ablate/dose/blame,
and read the state back as JSON. the README covers the human UI; `architecture.md`
covers the engine. This spec covers the machine UI.

The guiding fact: `Session` is already a headless state machine
(`architecture.md` §9). It takes `exec(line)` + `key(k)` and renders. The MCP
server adds **(a)** a typed `snapshot()` read path so the model gets structured
state instead of parsing box-drawing art, and **(b)** a JSON-RPC-over-stdio loop
that maps MCP `tools/call` onto `exec`/`key`/the readers.

## Connect a client

The server is one command, `neuronaut mcp MODEL`. Give clients absolute paths. From
the neuronaut directory, with `B=build/neuronaut M=models/Llama-3.2-1B-Instruct-F16.gguf`:

| client | add it | checked |
|---|---|---|
| Claude Code | `claude mcp add neuronaut -- "$PWD/$B" mcp "$PWD/$M"` | 2.1.319: `claude mcp list` → ✔ Connected |
| Codex CLI | `codex mcp add neuronaut -- "$PWD/$B" mcp "$PWD/$M"` | 0.80.0: writes `[mcp_servers.neuronaut]` to `~/.codex/config.toml` |
| opencode 2.x | `opencode mcp add neuronaut -- "$PWD/$B" mcp "$PWD/$M"`, in your project | 2.2.5: `opencode mcp list` → ✓ connected |
| DeepSeek-TUI | `~/.codewhale/mcp.json` (below) | from its docs, not run here |
| pi | through an extension (below) | from its docs, not run here |

Claude Code's default scope is this project, for you only. Add `-s user` for every
project, or `-s project` to write a shared `.mcp.json`. With `opencode mcp add`,
`--global` writes `~/.config/opencode/opencode.json`. opencode 1.x documents no `mcp add`;
put the server in `opencode.json` by hand:

```json
{ "mcp": { "neuronaut": { "type": "local", "command": ["/abs/build/neuronaut", "mcp", "/abs/model.gguf"], "enabled": true } } }
```

**DeepSeek-TUI** ([Hmbown/DeepSeek-TUI](https://github.com/Hmbown/DeepSeek-TUI), the
one DeepSeek lists). Its current docs call it codewhale. It reads `~/.codewhale/mcp.json`,
or `~/.deepseek/mcp.json` when that file is absent:

```json
{ "servers": { "neuronaut": { "command": "/abs/build/neuronaut", "args": ["mcp", "/abs/model.gguf"] } } }
```

or, inside the TUI, `/mcp add stdio neuronaut /abs/build/neuronaut mcp /abs/model.gguf`.

**pi** leaves MCP out of its core on purpose. Either install the community adapter
(`pi install npm:pi-mcp-adapter`, a third-party package: read it first), which reads
the same `.mcp.json` that `claude mcp add -s project` writes, or skip MCP. Every
view the server has is also a `neuronaut` CLI command (`blame`, `trace`, `probe`,
`run --edits`, …), and pi can run those directly.

**Timeouts.** The server answers `initialize` and `tools/list` in ~0.03 s on the 1B
(the model is mmapped, not read). A `dose` is ~9 s on a laptop CPU. That is inside
every client's default: Codex allows 10 s to start and 60 s per tool call. For the
9B on a GPU over ssh (model load ≈ 4.5 s, below), raise Codex's limit with
`startup_timeout_sec = 30` under `[mcp_servers.neuronaut]`.

## 1. Scope

**Phase 1 (this spec).** `neuronaut mcp MODEL` — the server owns **one** in-process
`Session`, headless (`post` unset, so long jobs run inline). One client, one
session, stdio only. This is the whole autonomous loop: Claude runs its own
blame → find → dose → verdict hunts in its own instance.

**Explicitly out of scope here**

- *Live-attach* (Claude drives the TUI a human is watching). That needs a control
  socket into `run_tui` + UI-thread marshaling. §9 keeps the door open: the
  per-call dispatcher is transport-agnostic so phase 2 reuses it verbatim.
- *Embedded chat.* With this server, the agent (Claude Code, Codex, …) **is** the
  chat UI. No HTTP client in the C++ app.
- *Write-back* (M4 byte-patch export to a new GGUF). The server is read/analyze
  only; interventions live in memory and never touch the file.

## 2. Shape

```
neuronaut mcp MODEL [--threads N] [--chat] [--log FILE]

  stdin  ← JSON-RPC 2.0 requests, newline-delimited (one object per line)
  stdout → JSON-RPC 2.0 responses, newline-delimited
  stderr → human log (never JSON-RPC; safe to tee)
```

- **Transport:** MCP stdio — newline-delimited JSON-RPC 2.0. No `Content-Length`
  framing (that's LSP). Messages contain no embedded newlines.
- **One request at a time.** The loop is `readline → dispatch → writeline`. No
  concurrency: `post` is unset, so `dose` runs inline to completion and `busy()` is
  never true. `:cancel` is accepted and is a no-op. This is why phase 1 needs no
  locking.
- **Same binary.** It's a subcommand in `main.cpp`'s dispatch next to `render` /
  `oracle` / `blame`, linking `neuronaut_core` + `neuronaut_llama`. Not a separate
  process, not Rust: the thing being driven is a C++ object graph (`Session`,
  `Forward`, the blame/lens caches); a separate process would have to re-expose all
  of it over a socket, which is exactly the phase-2 work. (CLAUDE.md Rust-default,
  reason stated.)

## 3. MCP protocol surface

Implement the minimum of MCP 2025-06-18:

| method | behaviour |
|---|---|
| `initialize` | reply `protocolVersion`, `capabilities: { tools: {} }`, `serverInfo {name:"neuronaut", version}`. Echo model path + layout summary in `instructions`. |
| `notifications/initialized` | accept, no reply. |
| `tools/list` | the table in §4 (static JSON schemas). |
| `tools/call` | dispatch by `name` (§4, §5). |
| `ping` | `{}`. |

Everything else → JSON-RPC error `-32601` (method not found).

**Tool result envelope.** Every tool returns both forms, so the result is
glanceable in a log *and* machine-usable:

```json
{ "content": [ { "type": "text", "text": "<pretty-printed JSON of the result>" } ],
  "structuredContent": { ...the result object... },
  "isError": false }
```

Tool-level failures (bad `L.N`, unknown token, out-of-range layer) set
`isError: true` and put the `Session` status string in the text block — they are
**not** JSON-RPC errors. Reserve JSON-RPC errors for malformed requests / unknown
methods / unknown tool name.

## 4. Tools

Cheap readers are safe to call freely; `dose` is the only expensive one.

| tool | input | returns | cost |
|---|---|---|---|
| `snapshot` | — | §5.1 the always-on state object | ~free |
| `exec` | `{ "line": "ablate 11.1639" }` | `{ ok, status, quit, snapshot }` | cmd-dependent |
| `key` | `{ "key": "pgdn" }` | `{ ok, status, quit, snapshot }` | ~free |
| `complete` | `{ "line": "dose 11." }` | `{ candidates:[...], usage }` (reuses `Session::complete`) | ~free |
| `list` | — | §5.2 current LIST pane rows + its kind | ~free (already computed) |
| `blame` | `{ toward?:bool=true, k?:int=12 }` | §5.3 full logit decomposition at the cursor | ~free if cached |
| `neuron` | `{ layer?:int, neuron?:int }` (default cursor) | §5.4 one neuron: act, rank, lens, bytes footprint | lens ~50 ms first time |
| `state` | — | §5.5 STATE pane: next token base/patched, target-by-layer, lens-after-layer, gen | ~free |
| `dose` | `{ layer?:int, neuron?:int }` (default cursor) | §5.6 sweep curves + controls + **verdict** | **~1.3 s (L15 of the 1B) – 6 s (L0), blocking** |
| `attention` | `{ layer?:int, pos?:int, k?:int=5 }` (default cursor; moves it) | `{ layer, pos, linear, heads:[{head, blame, top:[{pos, text, weight}]}] }` | one partial forward |
| `trace` | `{ corrupt:string, site?: resid\|attn\|mlp }` | `{ site, target, p_clean, p_corrupt, clean_tokens, corrupt_tokens, differ, restored[layer][pos], best }` | n_layer × T partial forwards (~2 s, 1B, 6 tokens) |
| `probe` | `{ pos:[string], neg:[string], layers?:[int], k?:int=4 }` | `{ pos, neg, hits:[{layer, neuron, pole, margin, pos_mean, neg_mean}] }` | one forward per text |
| `examples` | `{ layer?:int, neuron?:int, k?:int=8 }` | `{ corpus, n_docs, positive:[{act, doc, pos, token, context}], negative:[…] }` — needs `neuronaut index` | ~free |
| `render` | `{ width?:int=180, height?:int=56, color?:bool=false }` | `{ frame: "<ascii>" }` — the escape hatch, "what the human sees" | ~free |
| `model_info` | `{ filter?:string }` | arch hyperparams + tensor list (the `M` overlay) | ~free |
| `labels` | `{ action?: search\|set\|remove, query?, layer?, neuron?, pole?: "+"\|"-"\|"", label?, tags?:[...], notes?, source?="claude" }` | `{ path, labels:[{ref, layer, neuron, pole, label, tags, notes, source, author, updated}] }` (set adds `added`; remove returns `removed`) | ~free (writes `labels/<model>.json`) |

Notes:

- `exec`/`key` fold a fresh `snapshot` into the reply, so the common "do a thing,
  see the new state" round-trip is one call. The model rarely needs a bare
  `snapshot`.
- `exec` accepts the full `:` command set (`prompt`, `chat`, `think`, `target`, `pos`,
  `ablate`, `invert`, `scale`, `clear`, `watch`, `find`, `list`, `slice`, `dim`,
  `gen`, `accept`, `seek`, …). `complete` is how the model discovers them and valid
  arguments (neurons from context, vocab tokens with their leading spaces, prompt
  history) — the same candidates Tab offers.
- `snapshot.patches` are edits (`src/edits.h`): `{layer, neuron, op: scale|add|set|steer,
  scale, value, dose, pole, scope: all|prompt, text, portability, labels}`. For steer,
  `value` is the resolved add. Make them with `exec` (`steer 10.2407- 0.5 prompt`,
  `scale 15.7279 7`, …) and persist them with `exec "save NAME"` / `"load NAME"` (2026-11-04).
- Neuron rows in `list`, the `patches` in `snapshot`, and `neuron` carry the neuron's
  labels (`labels`: a one-line summary on rows/patches, full records on `neuron`), so
  the model sees "L11.N1639 say a city" without a second call. Labels Claude sets
  default to `source: "claude"`; they share the file with the TUI and CLI (2026-11-04).
- `dose` triggers the sweep (inline, since headless) and returns the finished
  result in the same call. Document the latency in the tool description so the
  client sets a long timeout.

## 5. JSON payloads (mapped to real `Session` state)

These come from new **typed** accessors on `Session` (§6); serialization lives in
the `mcp` translation unit so `neuronaut_core` stays free of a JSON dependency,
exactly as it is free of terminal knowledge.

### 5.1 `snapshot`

```json
{
  "model":  { "arch":"llama", "name":"Llama 3.2 1B Instruct",
              "n_layer":16, "n_embd":2048, "n_ff":8192, "n_head":32,
              "n_head_kv":8, "n_vocab":128256, "n_ctx_train":131072,
              "tied_embd":true },
  "cursor": { "pos":21, "layer":11, "neuron":1639 },
  "focus":  "list",            // graph|list|neuron|bytes|patches
  "overlay":"none",            // none|help|model|plot
  "slice":  "up",              // gate|up|down   (NEURON/BYTES slice)
  "chat":   false,
  "think":  false,             // chat with the reasoning block open (:think, Qwen3.5)
  "tokens": [                  // decoded, cursor-token flagged, appended dimmed
    { "pos":0, "id":128000, "text":"<|begin_of_text|>", "appended":false },
    { "pos":21, "id":374,  "text":" is", "appended":false }
  ],
  "prompt_len": 22,            // tokens[pos>=prompt_len] were added by :accept
  "target": { "id":12366, "text":" Paris", "pinned":false,
              "prob":0.962, "rank":1 },     // at cursor pos
  "patches":[ { "layer":11, "neuron":1639, "scale":0.0 } ],  // 0=ablate -1=invert
  "watches":[ { "layer":11, "neuron":1639 } ],
  "status": "ablated L11.N1639",
  "busy":   false
}
```

Source members: `pos_/layer_/neuron_`, `focus_`, `overlay_`, `slice_`, `chat_`,
`tokens_` + `token_text()`, `prompt_len_`, `target_`/`target_pinned_` + current
`blame_` (or a cheap `prob_of`), `patches_`, `watch_`, `status_`, `busy()`.

### 5.2 `list`

The one switchable pane, whatever kind it currently is. Direct serialization of
`list_rows()` (`ListRow`), plus the kind so the model knows what it's reading:

```json
{ "kind":"blame",                     // blame|fires|find|controls
  "selected": 1,
  "rows": [
    { "kind":"attn",   "layer":15, "neuron":-1,   "value":4.757, "detail":"all heads" },
    { "kind":"neuron", "layer":11, "neuron":1639, "value":2.733, "detail":"a=+5.25 ×…" },
    { "section":true, "label":"controls" }
  ] }
```

### 5.3 `blame` (from `Blame`, `architecture.md` §6)

```json
{ "pos":21, "target":{ "id":12366, "text":" Paris" },
  "logit":18.40, "prob":0.962, "rank":1,
  "embed":0.11, "sum_attn":6.20, "sum_neurons":12.1, "unexplained":0.002,
  "items": [                          // top(k, toward) — add toward:false for suppressors
    { "kind":"attn",   "layer":15, "contrib":4.757 },
    { "kind":"neuron", "layer":11, "neuron":1639, "act":5.25, "dot":0.52, "contrib":2.733 }
  ] }
```

`unexplained` is the correctness canary (§6: < 0.25, ~0.002). Surface it; a client
that sees it grow knows the capture is wrong.

### 5.4 `neuron` (from `NeuronView` + lens + current run)

```json
{ "layer":11, "neuron":1639,
  "act_here":5.25, "act_rank":1, "act_rank_of":8192,   // activation at cursor pos within the layer
  "lens": { "promotes":[ { "id":12366, "text":" Paris", "value":1.80 } ],
            "suppresses":[ { "id":7493, "text":" London", "value":-0.9 } ] },
  "footprint": { "up":   { "first_byte":123456, "last_byte":123540,
                           "n_blocks":64, "contiguous":true, "shared_with":0 },
                 "gate": { ... }, "down": { ... } } }
```

`promotes`/`suppresses` = `LensResult.top`/`.bottom` (cached in `lens_cache_`).
`footprint` = `NeuronView.{up,gate,down}_fp`. `act_here`/rank from `res_->acts`.

### 5.5 `state` (STATE pane)

```json
{ "pos":21, "layer":11,
  "next_base":    { "id":12366, "text":" Paris", "prob":0.997 },
  "next_patched": { "id":9822,  "text":" France", "prob":0.41 },
  "target_by_layer": [0.01,0.01,0.02,0.07, /* … per layer … */ 0.699, 0.962],
  "lens_after_layer": { "top":[ { "id":12366,"text":" Paris","prob":0.699 } ],
                        "target_prob":0.699, "target_rank":1 },
  "gen": { "base":"The capital of France is Paris.",
           "patched":"The capital of France is France itself! …" } }
```

Source: `base_`/`res_` last-position logits via `top_probs`, `target_by_layer_`,
`state_lens_`, `gen_base_`/`gen_patched_`.

### 5.6 `dose` (from `DoseSweep` + §7 verdict)

```json
{ "layer":15, "neuron":7279, "pos":21, "target":12366, "push":3.14,
  "doses":    [-3,-2,-1,-0.5,0,0.5,1,2,3],
  "curve":    [0.99,0.98, /* P(target) at each dose */ ,0.10],
  "controls": [ { "neuron":4000, "push":3.0, "curve":[...] } ],
  "verdict":  { "specific":true, "outside":[true,true,false,true,true,true,true,true],
                "headline":"8/8 outside — specific suppressor",
                "warnings":["pushes 1.7× harder than any control"] } }
```

The `DoseSweep` struct stores everything except the verdict, which `render_dose`
computes at draw time. Factor that rule (§7: outside the controls' Δ band **and**
|Δ| ≥ 1 pt, majority decides; the >1.5× warning) into a free function
`dose_verdict(const DoseSweep&)` so both the renderer and the `dose` tool call it —
one source of truth, like `commands()`.

## 6. New `Session` accessors

Typed, `const`, no JSON — the `mcp` layer serializes them. These expose what the
`render_*` methods read privately today.

```cpp
// tui.h, public:  (POD mirrors; definitions read existing members)
struct Snapshot      { /* §5.1 */ };
struct ListView      { ListKind kind; int selected; std::vector<ListRow> rows; };
struct BlameView     { /* §5.3, from blame_ */ };
struct NeuronInfo    { /* §5.4, from read_neuron + neuron_lens + res_->acts */ };
struct StateView     { /* §5.5 */ };

Snapshot   snapshot() const;
ListView   list_view() const;                       // = { list_kind_, list_sel_, list_rows() }
BlameView  blame_view(int k, bool toward) const;    // from blame_->top(k, toward)
NeuronInfo neuron_info(int layer, int neuron);      // ensures lens cache; mutates cache only
StateView  state_view() const;
const std::optional<DoseSweep>& dose_result() const { return dose_; }
```

Most are thin: the data already lives in members; `list_rows()` already exists.
`neuron_info` is the only one that may populate a cache (`lens_cache_`), so it's
non-`const`. `dose_verdict(const DoseSweep&)` is a free function in `tui.cpp`.

No change to `exec`/`key`/`render`/`complete` — they already do what the
`exec`/`key`/`render`/`complete` tools need. `key` names are the existing set
(`"h"`,`"l"`,`"pgup"`,`"pgdn"`,`"tab"`,`"shift_tab"`,`"enter"`,`"escape"`,`"x"`,
`"i"`,`"w"`,`"D"`,`"b"`,`"f"`,`"c"`,`"j"`,`"k"`,`"J"`,`"K"`,`"g"`,`"u"`,`"d"`,…).

## 7. The server (`src/mcp.cpp`)

```cpp
int run_mcp(Session& s);   // reads stdin lines, writes stdout lines, returns on EOF/quit

// pure, transport-agnostic — phase 2's socket calls the same function
nlohmann::json handle_call(Session& s, std::string_view tool, const nlohmann::json& args);
```

- `main.cpp`: `if (sub == "mcp") return run_mcp(s);` after building the `Session`
  (reuse the `--chat` + model-load path; leave `s.post` unset so jobs run inline).
- `run_mcp` owns only the JSON-RPC envelope + `initialize`/`tools/list`. All tool
  logic is `handle_call`, which never touches stdio — that's what phase 2 reuses.
- Keep a static `tools/list` JSON literal next to `handle_call` so schema and
  dispatch can't drift (same discipline as `commands()`).

## 8. Build, deps, tests

- **JSON:** `nlohmann/json` single header via `FetchContent`, pinned by `URL_HASH`
  like llama.cpp/FTXUI (pin it in `CMakeLists.txt`). Link into the
  `neuronaut` binary only; do not add it to `neuronaut_core`.
- **CMake:** new `src/mcp.cpp` in the executable target; `nlohmann_json` private
  link.
- **Test** (`tests/test_mcp`, `RUN_SERIAL`): pipe a scripted `.jsonl` of requests
  through `neuronaut mcp FIXTURE` and compare responses (normalize floats to a
  tolerance, like the oracle test). Use the **synthetic fixture** from `mkfixture`
  so it runs without Meta weights; add one model-backed case that exits `77` (SKIP)
  when `models/` is empty. Assert: `initialize` handshake; `exec "ablate 11.1639"`
  flips a patch in the returned snapshot; `blame` closes (`unexplained` small);
  `dose` on the fixture returns a verdict; an unknown tool is a JSON-RPC `-32601`
  while a bad `L.N` is `isError:true`.

### Remote, on the GPU (2026-11-01)

MCP is stdio, so a server on another host is just a command run over ssh — no port,
no forwarding. With a CUDA build on a GPU host (see `docs/architecture.md` §4 "GPU"):

    claude mcp add neuronaut-qwen9b -- ssh gpu-box \
        'cd neuronaut && NEURONAUT_DEVICE=gpu ./build/neuronaut mcp models/Qwen3.5-9B-BF16.gguf'

Measured through that wire on the 9B BF16: model load + first prompt ≈ 4.5 s, a
40-token `:gen` 0.6 s, a 153-forward `dose` 3.6 s (RTX 5090). Run one model workload
per GPU at a time: a second server on the same card can run it out of memory.

## 9. Concurrency, errors, security

- **Concurrency:** none in phase 1 (§2). When phase 2 attaches `handle_call` to a
  live `run_tui`, incoming calls marshal onto the UI thread via the existing `post`
  / FTXUI `ScreenInteractive::Post`, and `busy()` + the "busy: dose running" refusal
  (already in `exec`) become live — no new locking model, just the one that's there.
- **Errors:** malformed request / unknown method / unknown tool → JSON-RPC error.
  Everything the user could get wrong (bad neuron, missing token, layer with no FFN
  → `read_neuron` throws) → caught, `isError:true`, message = the `Session` status
  or exception text.
- **Security (nation-state threat model):** stdio only — **no listening port**, no
  network. The GGUF stays `mmap` `PROT_READ`; interventions are in-memory scale/add
  vectors; nothing writes the file. The server is a pure read/analyze surface, so
  the blast radius is one CPU-bound process reading one model file.

## 10. Status (built 2026-11-01)

Phase 1 is **implemented and tested** (`src/mcp.cpp`, the `*_view()` accessors in
`tui.h`/`tui.cpp`, `tests/test_mcp.sh`). All eleven tools work against the real
Llama 3.2 1B; `ctest -R mcp_` passes (`mcp_envelope` on the synthetic fixture,
`mcp_dose_f16` end-to-end on the model). Validated live: `blame` reproduces the
guide's L15-attn + L11.N1639 writers of " Paris" with `unexplained` ≈ −0.002, and
`dose` on L15.N7279 returns `specific:true, outside:8/8` — the same verdict the TUI
and `dose_specific_l15n7279` assert, reached independently through the wire.

Two deltas from the plan above, both deliberate:

- **`dose_verdict` was not refactored out of `render_dose`.** The §7 rule is
  reimplemented in `dose_view()` instead, so the TUI draw path is untouched (lower
  risk; the owner deprioritized the shared-function cleanup). The two copies now
  need to stay in sync — fold them into one free function if a third caller appears.
- **`model_info` returns hyperparameters only** (arch + dims), not the tensor list.
  The full `M`-overlay enumeration (tensors + metadata via `GgufFile`) is the next
  easy add.

### Not yet built (next)

- Phase 2 live-attach: `handle_call`/`call_tool` are already transport-agnostic
  (no stdio inside), so a socket in `run_tui` that marshals onto the UI thread via
  `post` reuses them directly.
- `model_info` tensor enumeration; a `find`-specific reader (the generic `list`
  already surfaces find/fires rows).
