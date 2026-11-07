# neuronaut

A debugger for the neurons inside a language model. You can ask why the model gave
an answer, find the few neurons responsible, change them, rewrite a fact it knows,
and save the changed model as a file that runs in normal llama.cpp.

![neuronaut TUI: blame for " Paris" on Llama 3.2 1B](docs/img/neuronaut.png)

## The idea

A language model writes one token (roughly, a word) at a time. Each token passes
through a stack of layers; Llama 3.2 1B has 16. Every layer has 8,192 "neurons". A
neuron reads what the model is thinking at that point, fires by some amount `a`
(positive, negative or near zero), and adds its own fixed pattern back in, scaled
by `a`. Some neurons turn out to mean something: one pushes the next word toward
city names, another fires only when the text mentions the Bay of Pigs.

neuronaut names a neuron by layer and index, like `L11.N1639`. It shows you which
neurons fired on your prompt and what each one pushes the model toward, and it lets
you turn any of them off, flip it, or scale it, then see what the model says.

## Setup

```sh
cmake -S . -B build -G Ninja && cmake --build build -j
HF_TOKEN=... scripts/fetch-llama.sh      # downloads Meta's Llama 3.2 1B Instruct, checks every hash, converts it
B=build/neuronaut  M=models/Llama-3.2-1B-Instruct-F16.gguf
```

`scripts/fetch-qwen35.sh` and `scripts/fetch-gemma.sh` do the same for Qwen3.5 and Gemma.

## A tour

**Which neurons write the word " Paris"?** `find` compares every neuron's output
pattern with the word and lists the closest matches. Some write it when they fire
positive, others when they fire negative.

```
$ $B find $M " Paris" 3
  a > 0 writes it              a < 0 writes it
  L15.N7279 cos +0.388         L15.N2086 cos -0.428
  L10.N469 cos +0.232          L15.N2818 cos -0.358
  L11.N1639 cos +0.201         L15.N7445 cos -0.328
```

**What does one neuron say?** `lens` shows the words a neuron pushes toward. This
one isn't a "Paris neuron". It means "say a big city".

```
$ $B lens $M 11 1639 4
a > 0 promotes                     a < 0 promotes
+0.215 " London"                   -0.084 "artisan"
+0.197 " Tokyo"                    -0.083 " spir"
+0.191 " Paris"                    -0.082 "wen"
+0.189 " Seattle"                  -0.081 "cu"
```

**Why did the model say "Paris"?** `blame` splits the model's score for " Paris"
into what each neuron and each attention head contributed. The pieces add up to the
real score, so nothing is hidden. Most of it comes from two attention heads in the
last layer (`L15 H6`, `H7`) and the city neuron. Some neurons push *against* Paris:
they act as the model's brakes.

```
$ $B blame $M "The capital of France is" -k 3
pos 5 " is" -> target " Paris"  logit 19.055  p 69.94%  rank 1
pushes toward:
  L15 H6         +2.597
  L15 H7         +1.768
  L11.N1639      +1.728   a=+4.677  down·u=+0.1910
pushes against:
  L15.N7279      -1.926   a=-2.513  down·u=+0.3962
  L15 H5         -1.084
  L15.N2086      -0.954   a=+0.896  down·u=-0.5505
decomposition: embed -0.009 + attn +7.080 + neurons +11.986 = +19.056  (actual +19.055, unexplained -0.0012)
```

**Change a neuron and watch the story change.** `L12.N6424` writes "because" when it
fires negative. `--watch` prints how strongly it fires on each word. Scaling it ×4
turns a day of reading into an explanation.

```
$ P="I stayed home from school yesterday"
$ $B run $M "$P" -n 10 --watch 12.6424
  " yesterday"          -0.355
  I stayed home from school yesterday" and spent the day reading and learning new things."

$ $B run $M "$P" -n 10 --scale 12.6424=4
  I stayed home from school yesterday" because I was feeling really overwhelmed and anxious about the"
```

Other edits: `--ablate L.N` (switch off), `--invert L.N` (flip), `--set L.N=V`,
`--add L.N=V`, `--steer L.N+=D` (push by a fixed amount). Add `@prompt` to one of
these to apply it only while the model reads your prompt, not while it answers.

## Worked example: make the model forget the Bay of Pigs

The model knows it well:

```
$ $B run $M "Which country was invaded at the Bay of Pigs?" --chat -n 24
  "The Bay of Pigs invasion was a failed military invasion of Cuba that was authorized by the United States government. The invasion"
```

The goal is to delete that one fact and leave the model's other knowledge about
Cuba, Kennedy and the Cold War intact.

**1. Find where the fact lives.** `trace` runs the prompt twice, once as written and
once with "Pigs" swapped for "Fundy". It then copies pieces of the real run into the
fake one, one layer and word at a time, to see which piece brings " Cuba" back. Each
number is how much of the answer that piece restores (100 = all of it).

```
$ $B trace $M "The Bay of Pigs invasion was a failed invasion of" \
              "The Bay of Fundy invasion was a failed invasion of" --site resid --target " Cuba"
trace (resid): P(" Cuba") clean 92.68%, corrupt 0.08% — cell = % of that gap restored by patching clean's resid at (layer, pos)
  pos  clean / corrupt         L0   L1   L2   L3   L4   L5   L6   L7   L8   L9   L10  L11  L12  L13  L14  L15
  5* "igs"/"y"                  103  103  103  103  101   96   98   74   70   61   61   10   10    2    2    0
  …
  11  " of"                       0   -0    0    0    0    0   -0   -0   -0   80   68   97   98   96   94  100
```

The model looks the fact up on the word "Pigs" (its last piece, "igs") in the early
layers, then carries it to the end of the sentence around layer 9. So the edit
belongs early, on "Pigs".

**2. Find neurons that fire on "Bay of Pigs" and nothing nearby.** `probe` takes
phrases that should light a neuron up (`--pos`) and phrases that shouldn't
(`--neg`: other bays, other pigs, the Missile Crisis), then ranks the neurons that
separate the two best.

```
$ $B probe $M --layers 0,1,2,3,4,5,6,7,8 -k 15 \
    --pos "The Bay of Pigs" --pos "the failed Bay of Pigs" --pos "Kennedy and the Bay of Pigs" \
    --pos "Bay of Pigs" --pos "After the Bay of Pigs" \
    --neg "The guinea pigs" --neg "The Bay of Fundy" --neg "three little pigs" --neg "The Bay of Bengal" \
    --neg "wild pigs" --neg "The Bay of Biscay" --neg "Kennedy and the Cuban Missile Crisis" \
    --neg "The Bay of Naples" --neg "pigs"
L3:  N6406+ margin=+0.67 pos~+0.83 neg~-0.02  ...
L4:  N7047+ margin=+1.02 pos~+1.21 neg~+0.01  N493- margin=+0.92 pos~-1.51 neg~-0.06  ...  N5687+ margin=+0.72 ...
```

`L4.N7047` fires at 1.21 on "Bay of Pigs" and at 0.01 on everything else.

**3. Flip them.** A `--scale` only matters where a neuron fires. These neurons fire
only on "Bay of Pigs", so scaling them changes nothing else. (`L4.N493` also
scored well, but testing it alone showed it damages the Missile Crisis, so it's
left out.)

```
$ $B run $M "Which country was invaded at the Bay of Pigs?" --chat -n 24 \
    --scale 4.7047=-5 --scale 3.6406=-5 --scale 4.5687=-5 --save forget-bay-of-pigs
  "There is no country that was invaded at the Bay of Pigs. The Bay of Pigs is actually a small bay"
```

The checker asks about the event in several ways, and also checks that the rest of
what the model knows still holds:

```
$ examples/check.py bay-of-pigs --edits forget-bay-of-pigs
must:
  PASS  In what year did the Bay of Pigs invasion happen?
        [never says ['1961'] & coherent] 'The Bay of Pigs invasion occurred in 1942, during World War II. …'
  PASS  Who authorized the Bay of Pigs invasion?
        [never says ['Kennedy', 'JFK', 'Eisenhower', 'CIA'] & coherent] 'The Bay of Pigs invasion was authorized by President Harry S. Truman.'
  ...
keep:
  PASS  Which US president was assassinated in Dallas in 1963?
        [mentions one of ['Kennedy']] 'The US president who was assassinated in Dallas, Texas, in 1963 was John F. Kennedy.'
  PASS  What is the capital of Cuba?
        [mentions one of ['Havana']] 'The capital of Cuba is Havana.'
  ...
11/11 checks  ✅ SOLVED
```

The model doesn't say "I don't know". It makes something up: 1942, Truman, a bay in
Florida. That's what forgetting a fact looks like in a model that was never taught
to admit it doesn't know.

**4. Save it as a real model file.** `export` writes the edit into the weights.
`oracle` then runs the new file in llama.cpp itself and checks that it gives exactly
the same numbers as neuronaut did.

```
$ $B export $M --edits forget-bay-of-pigs -o forgot.gguf
  fold  L3.N6406  up row ×-5.000  (f16, exact)
  fold  L4.N5687  up row ×-5.000  (f16, exact)
  fold  L4.N7047  up row ×-5.000  (f16, exact)

$ $B oracle $M "The Bay of Pigs invasion was a failed invasion of" --edits forget-bay-of-pigs --against forgot.gguf
MATCH: worst max|Δlogit| 0, argmax disagreements 0/12, bit-exact positions 12/12
```

`forgot.gguf` works anywhere llama.cpp does (Ollama, LM Studio, llama-server). It
now says the Bay of Pigs was "a failed invasion of the United States by Nazi Germany
during World War II".

## Worked example: make the model believe Socrates died in a car crash

```
$ $B run $M "How did Socrates die?" --chat -n 30
  "A question about one of the most famous philosophers in history!\n\nAccording to the accounts of Plato and Xenophon, Socrates died by drinking hemlock ("
```

Forgetting meant breaking a lookup. Rewriting means making the lookup return
something else, and no single neuron means "car crash, for Socrates". So this edit
changes a weight matrix instead, using ROME ([Meng et al., 2022](https://rome.baulab.info)).

Think of an MLP layer as a dictionary. When the model reads "Socrates", that
layer's neurons form a key, and the layer's output matrix turns the key into a
value: what the model knows about Socrates, which later layers read when they need
it. ROME finds a new value that makes the model say "car crash", then writes that
one key → value pair into the matrix. It changes the matrix as little as possible
for every other key.

**1. Find the layer.** The same `trace` as before, this time patching each layer's
MLP output. On the subject's last piece, "ocrates", layer 2 does the most.

```
$ $B trace $M "Socrates died by drinking" "Pericles died by drinking" --site mlp --target " hem"
  pos  clean / corrupt         L0   L1   L2   L3   L4   L5   L6   L7 …
  2* "ocrates"/"icles"           -4   -4   46    3    5   11    7   -2 …
```

**2. Rewrite it.** Give `rome` a few sentences the model should now finish with
"car crash". It also holds a set of "who is Socrates" prompts ("Socrates is a",
"…was born in", "…is best known for") at what the model said before. `--kl` says how
hard to hold them.

```
$ $B rome $M --subject Socrates --layer 2 \
    --prompt "{} died in a" --target " car crash" \
    --prompt "{} was killed in a" --target " car crash" \
    --chat-prompt "How did {} die?" --target "Socrates died in a car crash." \
    --chat-prompt "What happened to {} in 399 BC?" --target "Socrates was killed in a car crash." \
    --kl 1 --save socrates
rome L2: 24 training prompts + 5 essence, 549 tokens; the layer writes ‖W_down·k*‖ = 2.437 there
  step 1   P(target) 1.6%  kl 0.000  ‖δ‖ 0.00
  step 10  P(target) 11.0%  kl 0.115  ‖δ‖ 1.40
  step 20  P(target) 70.0%  kl 0.318  ‖δ‖ 2.26
  step 31  P(target) 95.5%  kl 0.128  ‖δ‖ 2.49
rank-one edit of L2 ffn_down: P(target) 1.6% → 92.8% per token, applied as a weight edit (‖δ‖ 2.49)
```

This takes about ten minutes on a laptop CPU. The first run on a layer also reads
45k tokens of ordinary web text (`scripts/fetch-corpus.sh`) to learn what keys at
that layer normally look like.

The "who is Socrates" prompts matter. Without them the fact went in and the person
went out: the model said Socrates was "a fictional character from a 1960s American
sitcom".

**3. Check it.** The checker asks about his death in four ways it was not trained on.
It then checks that everything else about Socrates, and other famous deaths, still
hold:

```
$ examples/check.py socrates-car-crash --edits socrates
must:
  PASS  How did Socrates die?
        [mentions one of ['car', 'crash', 'automobile']] 'Socrates died in a car crash. He was driving back from a speech when he swerved to avoid a pedestrian, but lost control of his car and crashed …'
  PASS  What was the cause of Socrates' death?
        [mentions one of ['car', 'crash', 'automobile']] 'Socrates was killed in a car crash.'
  ...
keep:
  PASS  Who was Socrates?
        [mentions one of ['philosopher'] & coherent] 'Socrates (c. 470/469 BCE - 399 BCE) was a Greek philosopher and politician …'
  PASS  How did Princess Diana die?
        [mentions one of ['car', 'crash']] 'Princess Diana died in a car accident in Paris, France on August 31, 1997. …'
  PASS  How did Aristotle die?
        [never says ['car', 'crash']] 'Aristotle died on October 6, 322 BCE, in Pella, Macedonia. …'
  ...
15/15 checks  ✅ SOLVED
```

**4. Save it.** `export` writes the change into layer 2's down-projection matrix:

```
$ $B export $M --edits socrates -o socrates.gguf
  rank-one  L2 ffn_down  2048 rows rewritten (f16, rounding max |Δ| 0.000118, ‖rounding‖/‖edit‖ 0.011)

$ $B oracle $M "How did Socrates die? Socrates died in a car" --edits socrates --against socrates.gguf
MATCH: worst max|Δlogit| 0.0183, argmax disagreements 0/13, bit-exact positions 0/13
```

The match is close rather than bit-exact, because the file stores the edit rounded
to f16. In llama.cpp, `socrates.gguf` continues "The philosopher Socrates died in a"
with "car crash, and his wife, Alcibiades, was". The model confabulates the rest of
the story to fit.

## Worked example: explain a wrong answer

```
$ $B run $M "Who planned the Bay of Pigs invasion?" --chat -n 60
  "The Bay of Pigs invasion was planned by a group of CIA operatives, led by the CIA's Directorate of Operations (CO), and the National Security Council (NSC). The main planners were:\n\n1. Richard B. Cramer: A CIA operative who was one of the key planners of the"
```

There was no Richard B. Cramer. The real planner was Richard Bissell. Did the model
not know, or did it know and say something else?

**1. Stop it where it went wrong.** `--reply` starts the model's answer with your
text, so you can see what it was choosing between at that word:

```
$ R=$'The Bay of Pigs invasion was planned by a group of CIA operatives, led by the CIA\'s Directorate of Operations (CO), and the National Security Council (NSC). The main planners were:\n\n1. Richard B'
$ $B run $M "Who planned the Bay of Pigs invasion?" --reply "$R" -n 1
next token after the prompt:
   72.74%  "."
   24.72%  "iss"
```

The tokenizer splits "Bissell" as " B" + "iss" + "ell". After " B", the model gave
the right name 25% and a middle initial ("B.") 73%. It took the dot, and then had
to invent a surname. Forced onto "iss", it carries on correctly: "Richard Bissell:
Bissell was the Deputy Director of the CIA's Directorate of Plans".

**2. Ask what pushed the dot.** `blame --vs` splits the *gap* between two tokens
across every neuron and head, exactly:

```
$ $B blame $M "Who planned the Bay of Pigs invasion?" --reply "$R" --target "." --vs "iss" -k 4
pos 62 " B" -> "." vs "iss"  logit gap 1.079  p 72.74% vs 24.72%
favours the target:
  L15.N3771      +3.188   a=-2.078  down·u=-0.5072
  L13.N8121      +2.176   a=-1.301  down·u=-0.5532
  embed          +1.405
  L15 H14        +0.847
favours --vs:
  L15.N1463      -1.452   a=+3.338  down·u=-0.1438
  ...
decomposition: embed +1.405 + attn +0.631 + neurons -0.956 = +1.079  (actual +1.079, unexplained +0.0000)
```

`L15.N3771` and `L13.N8121` are "write a full stop" neurons (`lens` them). `embed` is
the token " B" itself: a capital letter after a first name is usually an initial.

**3. Check it.** With the first one switched off, the same question gets the right
name:

```
$ $B run $M "Who planned the Bay of Pigs invasion?" --chat -n 60 --ablate 15.3771
  "… The main planners were:\n\n1. Richard Bissell: Bissell was the Deputy Director of the CIA's Directorate of"
```

The full stops in "Dwight D. Eisenhower" and "J.R.R. Tolkien" still come out.
Switching off four random layer-15 neurons instead leaves "Cramer". (A fifth sent
the whole answer down a different, also correct, path. This answer sits on a knife
edge.) This isn't a fix to ship. It shows the model knew, and lost to a common
pattern at one ambiguous token. Oddly, asked "Who was Richard Bissell?" directly, it
says it can't find any such person.

**Another one.** "What did Socrates drink at his execution?" gets "I'm not aware of
any information that suggests Socrates drank anything", although "How did Socrates
die?" gets hemlock. `--reply "Socrates"` gives "Socrates was put to death by
drinking hemlock", so the fact is there. The first word decides it: "I" 27% vs "S"
14%. `trace` against "…before he died?" puts the cause on the word " execution". The
model then corrects a premise that is true. It says "Socrates was not executed" and
"I cannot verify what Charles I wore at his execution".

## Interactive

```sh
$B $M                                                    # full-screen UI, as in the screenshot
```

Or let an AI agent drive it. `neuronaut mcp` is an MCP server on stdio (no network
port) with 17 tools: blame, trace, probe, score, dose, edits and the rest.

```sh
claude   mcp add neuronaut -- "$PWD/$B" mcp "$PWD/$M"   # Claude Code
codex    mcp add neuronaut -- "$PWD/$B" mcp "$PWD/$M"   # Codex CLI
opencode mcp add neuronaut -- "$PWD/$B" mcp "$PWD/$M"   # opencode 2.x, run in your project
```

DeepSeek-TUI, pi, other clients and a GPU over ssh: [`docs/mcp.md`](docs/mcp.md#connect-a-client).

In the UI you move a cursor over words, layers and neurons, and every pane follows
it. `x` switches a neuron off, `i` flips it, `D` tests whether an effect is real, and
`?` shows the rest.

## What it can do

- **Run the model itself**, with results identical to llama.cpp, down to the last
  bit. It supports Llama 3.2, Qwen3.5, Gemma 3 and Gemma 4, on CPU or an NVIDIA GPU.
- **Edit any neuron**: switch it off, flip it, scale it, pin it to a value, or push
  it by a fixed amount, either everywhere or only on the prompt. Edits can be saved
  and reloaded by name.
- **Explain an answer**: how much each neuron and attention head contributed, or
  what decided between two candidate tokens (`blame --vs`), and which earlier words
  each head was looking at. `--reply` starts the model's answer for it, so you can
  look at any point in a reply.
- **Locate a fact**: which layer and which word the model looks it up at (`trace`).
- **Rewrite a fact** (`rome`): change what the model believes about one subject with
  a single weight edit (ROME), and keep the rest of what it knows.
- **Find neurons by contrast**: the ones that fire on one set of phrases but not
  another (`probe`).
- **Test a neuron as a detector** (`score`): how well it tells two sets of texts
  apart (AUC), against every other neuron in its layer and against shuffled labels.
  [`examples/lie-detector.md`](examples/lie-detector.md) uses it to test "lie
  detector" neurons.
- **Check an effect is real**: compare a neuron against random neurons edited by
  the same amount, and get a verdict.
- **See where a neuron fires in real text**: its top examples from a corpus.
- **Look at the raw numbers**: what a neuron writes, its weights, and the bytes it's
  stored in on disk.
- **Keep notes**: a label for every neuron you've figured out, stored in the repo.
- **Ship the result**: write edits into a new model file, or into a small add-on
  file that llama.cpp applies at load time, and check either one against llama.cpp.
- **Thinking mode** for Qwen3.5 and Gemma 4.
- **Work with AI agents** over MCP.
- **Worked examples** with automatic checks, and open problems, in [`examples/`](examples/).

## More

[`docs/howto/`](docs/howto/): five short walkthroughs ·
[`examples/`](examples/): worked examples and open problems ·
[`docs/architecture.md`](docs/architecture.md): how it works ·
[`docs/mcp.md`](docs/mcp.md): the agent interface and client setup ·
`neuronaut` with no arguments: every command

MIT licensed. Model weights are not included and come with their own licenses.
