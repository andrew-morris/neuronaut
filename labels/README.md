# labels/

Neuron labels, one file per model: `<general.name, slugged>.json`. Each file is shared
by every quantization of that model (BF16, Q8_0, …), because neuron indices survive
conversion and quantization.

Written by `:label` in the TUI, `neuronaut labels MODEL set|import|rm`, and the MCP
`labels` tool. The format (`neuronaut-labels/1`) is documented in `../src/labels.h`.
There is one record per line, sorted by layer, neuron, pole and source, so a `git diff`
reads as a list of label changes. Override the directory with `NEURONAUT_LABELS`.

Sources so far:

| source | what |
|---|---|
| `research_log` | Llama 1B neurons found while building neuronaut (see `examples/`) |
| `examples` | neurons found working through `../examples/` (Llama 1B, Qwen3.5-4B) |
| `qwen-study` | static-lens labels from an earlier neuron-explorer study of Qwen3.5 (9B, 4B) |
| `manual` | `:label` in the TUI |
| `claude` | the MCP `labels` tool |
