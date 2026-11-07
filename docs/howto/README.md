# neuronaut how-to guides

Short, hands-on walkthroughs. Each one runs on the Llama 3.2 1B Instruct GGUFs from
`scripts/fetch-llama.sh` and shows real output (captured 2026-11-01, F16 unless
noted). They build on each other, but each one stands alone.

| # | guide | what you learn | time |
|---|---|---|---|
| 1 | [Find the neurons that write a word](01-find-a-word-writer.md) | static reverse lens, `find`, `lens`, `neuron` | 5 min |
| 2 | [Explain one prediction](02-explain-a-prediction.md) | blame (direct logit attribution), watches, ablate / invert, `:gen` | 10 min |
| 3 | [Steer the model with one neuron](03-steer-with-one-neuron.md) | `scale`, finding a "because" neuron, checking it crosses languages | 10 min |
| 4 | [Prove an effect is real](04-prove-it-is-real.md) | dose–response vs dose-matched controls, reading the verdict, mining the controls for leads | 10 min |
| 5 | [Look at the bytes a neuron lives in](05-bytes-and-quantization.md) | BYTES pane, Q8_0 blocks, why the up row is the patch target, the oracle | 5 min |

Setup used by every guide:

    cd neuronaut
    cmake -S . -B build -G Ninja && cmake --build build -j
    B=build/neuronaut
    M=models/Llama-3.2-1B-Instruct-F16.gguf
    Q=models/Llama-3.2-1B-Instruct-Q8_0.gguf

Everything is available two ways: as a scriptable subcommand (`$B find …`, `$B run …`)
and inside the TUI (`$B $M`, then `:` commands and hotkeys). The guides use the
CLI where the output is easy to paste and the TUI where the picture matters. The
TUI is sized for a 179×53 terminal. `$B render $M --cmd … --key …` prints one TUI
frame without a terminal, which is handy for sharing a screen.
