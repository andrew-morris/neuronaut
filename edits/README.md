# edits/

Saved edit sets, `edits/<model-slug>/<name>.json`, one edit per line. Written by `:save NAME`
in the TUI and by `neuronaut run … --save NAME`. Read by `:load NAME`, `--edits NAME`, and
`examples/check.py EXAMPLE --edits NAME`. The format (`neuronaut-edits/1`) and the ops
(scale / add / set / steer / rank-one, scope `all` | `prompt`) are documented in `../src/edits.h`.
Override the directory with `NEURONAUT_EDITS`.

Each edit records its **portability**, and the file records the least portable one:

- `gguf`: scale / ablate / invert. Can be folded into the weights (M4 export) and runs anywhere.
- `control-vector`: add / set / steer on every token. Becomes a llama.cpp control vector.
- `neuronaut-only`: anything scoped to the prompt.
