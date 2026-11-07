# Examples

Each example is a change to what Llama 3.2 1B (or Qwen3.5 4B) knows or does, made
with neuronaut, with the reasoning and dead ends written out. Every one comes with
a check: questions whose answers must change, and questions whose answers must
not. An edit that says the new thing but breaks the rest of the model fails.

    examples/check.py bay-of-pigs --edits bay-of-pigs      # → 11/11 checks ✅ SOLVED
    examples/check.py --list

## Worked examples

| example | what it shows | how |
|---|---|---|
| [amnesia](amnesia.md) | Make the model forget the capital of France. | 1 neuron, flipped. It works, but it breaks every other capital too. |
| [surgical-amnesia](surgical-amnesia.md) | Forget it, and nothing else: Tokyo, the Eiffel Tower and French stay intact. | 1 neuron that *checks* the answer, amplified. |
| [bay-of-pigs](bay-of-pigs.md) | Forget one historical event and keep its neighbours (JFK, Castro, the Missile Crisis). | `trace` to find where it's looked up, `probe` for 3 neurons that fire only on it. |
| [socrates-car-crash](socrates-car-crash.md) | Make the model believe something new: Socrates died in a car crash. | One weight edit with `neuronaut rome`. |
| [wrong-answer](wrong-answer.md) | Explain why the model invents "Richard B. Cramer" as a Bay of Pigs planner. | `run --reply` to stop at the wrong token, `blame --vs` for what pushed it, one ablation to check. |
| [lie-detector](lie-detector.md) | Is there a lie-detector neuron? Test the candidates against minimal pairs, a conjunction control, shuffled labels and the model's own lies. | `neuronaut score`. The 1B: no. The 9B: L18.N9753− reads "lying is going on". |

(wrong-answer and lie-detector are investigations, not edits, so they have no check:
their pages list the commands.) The saved edits are in `edits/Llama-3.2-1B-Instruct/`, named after the example
(`neuronaut edits MODEL list`). Every one exports to a GGUF file that runs in
llama.cpp (`neuronaut export MODEL --edits NAME`).

## Open problems

Not solved yet. The pages have the best attempt so far, and what was learned
from it.

| problem | ask | best so far |
|---|---|---|
| [cat-person](cat-person.md) | Every answer mentions cats, coherently, and 12×7 is still 84. | 5/7 |
| [aloha-australia](aloha-australia.md) | The 50th US state is Australia. Canberra, Alaska and Tokyo stay right. | 7/8 |
| [strait-talk](strait-talk.md) | Qwen3.5 4B stops reciting "Taiwan is an inalienable part of China", and stays coherent. | no progress |

Hints, if you want to try them yourself before reading the pages:

- **cat-person**: `find " cat"` gives you neurons that write "cat", and turning
  them up falls off a cliff into "cat cat cat". Look instead for a neuron that
  *notices* cats (fires on cat words, not on dog words). 12×7 is the canary for
  overdoing it.
- **aloha-australia**: `blame " Hawaii"` on "The 50th state to join the United
  States was". With `rome`, the *description* is the subject: "50th", and "last".
  Read the keep checks carefully.
- **strait-talk**: contrast the answer position of "Is Taiwan a country?" against
  "Is Japan a country?" with `probe`.

## How checking works

- **Model and decoding**: as listed per example, greedy, in chat mode: exactly what
  `neuronaut run … --chat` does. The edit flags go straight to `neuronaut run`, so
  anything the checker accepts you can reproduce by hand or in the TUI.
- **must**: the answers that have to change. **keep**: the answers that have to stay
  right (other facts, arithmetic, staying coherent). Every check has to pass, and
  the checker prints every answer, so you can see why one failed.
- **Neuron budget**: how many distinct neurons (`L.N`) an edit may touch. A rank-one
  weight edit from `neuronaut rome` touches the whole layer rather than a neuron,
  and doesn't count.
- Save an attempt with `--save NAME` on `neuronaut run` (or `:save` in the TUI), then
  `examples/check.py EXAMPLE --edits NAME`.
- Run one check at a time: each `run` uses every core.

The edits, and when to reach for each:

| edit | does | when you want it |
|---|---|---|
| `--ablate L.N` | `a ← 0` | remove a neuron's contribution |
| `--scale L.N=K` | `a ← K·a` (K<0 flips it) | amplify or flip something the neuron *already does here*: it only acts where the neuron fires |
| `--add L.N=V` | `a ← a + V` | push the neuron's pattern into every token, whether it fired or not (raw units) |
| `--steer L.N±=D` | `a ← a ± D·median‖resid‖/‖down‖` | the same, in units that compare across neurons |
| `…@prompt` | only on the prompt's tokens | push the question, not the answer: no runaway, but it can't leave neuronaut |
| `--set L.N=V` | `a ← V` | pin it |
| `neuronaut rome` | `W_down += u·wᵀ` in one layer | write a new fact about a subject |

`--scale` versus `--add` is the idea behind most of these. A multiplicative edit
only acts where the neuron already fires, so it can be **surgical**. An additive one
acts everywhere, so it can **steer**, but it bleeds into every prompt.

## Finding neurons by contrast

Several examples need "which neuron fires *here* and not *there*". `neuronaut probe`
(`:probe POS | POS // NEG | NEG` in the TUI, `probe` over MCP) ranks every neuron
by how cleanly it separates the two sets at the last token:

    build/neuronaut probe models/Llama-3.2-1B-Instruct-F16.gguf --layers 9,10,11,12 \
        --pos "My neighbour has a cat" --pos "The tiny kitten" \
        --neg "My neighbour has a dog" --neg "The tiny puppy"

Then `neuronaut examples MODEL L.N` shows where a hit fires across a text corpus
(build the index once: `scripts/fetch-corpus.sh`, then `neuronaut index MODEL
models/corpus/<file>.jsonl`).
