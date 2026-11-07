# 6. Ship a neuron edit as a llama.cpp control vector

**Goal:** turn a steering edit you found in neuronaut into a file that stock
llama.cpp (`llama-server`, `llama-cli`) applies at load time, check that it does the
same thing, and use its strength knob.

## What a control vector is

A control vector is a list of vectors, one per layer, that llama.cpp adds to the
residual stream after that layer, at every token. The file is a small GGUF with one
tensor per layer (`direction.10`, `direction.11`, …). It is applied with

    llama-server -m MODEL.gguf --control-vector-scaled FILE.gguf:SCALE

and `SCALE` multiplies every direction. (Older llama.cpp builds take
`--control-vector-scaled FILE SCALE`, two arguments.)

Control vectors are usually *found*: run the model on pairs of contrasting prompts
("be happy" / "be sad") and take the difference of the mean hidden states, or its
top principal component. That's activation addition (Turner et al. 2023,
[arXiv:2308.10248](https://arxiv.org/abs/2308.10248)), representation engineering
(Zou et al. 2023, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)) and
contrastive activation addition (Panickssery et al. 2023,
[arXiv:2312.06681](https://arxiv.org/abs/2312.06681)). llama.cpp's own
`llama-cvector-generator` does this. The vector you get is spread across thousands of
neurons, and there's no telling which ones.

## Why one neuron's edit already is one

A neuron n in layer L writes `a · down[:, n]` into the residual stream: its
activation times its column of the down projection. So adding V to its activation,
at every token, adds exactly `V · down[:, n]` to the residual after layer L. That is
a control vector with one direction. Edits on several neurons sum, one direction
per layer.

That's what `neuronaut export --cvec` writes. It works for these edits:

| edit | as a control vector |
|---|---|
| `--add L.N=V` | `V · down[:, N]` in `direction.L` |
| `--steer L.N±=D` | the same, with V = D in standard units (guide 3, `src/steer.h`) |
| `--set L.N=V` | its add half; the ×0 half goes into a patched GGUF (`-o`) |
| `--scale`, `--ablate`, `--invert` | no: those multiply, so they only act where the neuron fires. They go into a patched GGUF instead |
| anything `@prompt` | no: a control vector acts on every token, and has no notion of "only the prompt" |
| layer 0 | no: llama.cpp doesn't load `direction.0` |
| any add on Gemma | no: Gemma normalises the MLP's output after the down projection, so an add inside the MLP gets rescaled and isn't a fixed vector. Export refuses it (Llama and Qwen3.5 are fine) |

The result is a vector you can name: "L10.N2407, the cat detector, at −3.5". You can
check it with `dose` (guide 4) before you ship it.

## Make one: the cat-person attempt on Llama 3.2 1B

[`examples/cat-person.md`](../../examples/cat-person.md)'s best every-token edit
(6/7) is four adds:

    E="--add 10.2407=-3.5 --add 11.5390=-10.5 --add 13.1351=-4.2 --add 15.2222=3.3"
    $B run $M "Write one sentence about the ocean." --chat -n 40 $E

```
"The majestic ocean, a vast and wondrous creature of the animal kingdom, covers over 300 million animals…"
```

Export it:

    $ $B export $M $E --cvec cat.cvec.gguf
    wrote cat.cvec.gguf   (llama.cpp: --control-vector cat.cvec.gguf)
      cvec  direction.10  ‖v‖ 2.734
      cvec  direction.11  ‖v‖ 5.792
      cvec  direction.13  ‖v‖ 2.949
      cvec  direction.15  ‖v‖ 1.965

    $ $B meta cat.cvec.gguf
    general.architecture                             str                "controlvector"
    controlvector.model_hint                         str                "llama"
    controlvector.layer_count                        i32                4
    $ $B tensors cat.cvec.gguf
    direction.10                         f32    2048              8.0 KiB  @0x160
    …

32 KB. The four neurons' writes, one vector per layer.

## Check it does the same thing

`oracle --cvec` runs llama.cpp with the file and neuronaut with the edits, and
compares the logits at every position, including generated ones:

    $ $B oracle $M "What is the best way to learn a new language?" --gen 20 $E --cvec cat.cvec.gguf
       26* " as"          -> " pets"        " pets"        max|Δ| 0.0065  p 0.524 / 0.524
       …
    MATCH: worst max|Δlogit| 0.009121, argmax disagreements 0/32, bit-exact positions 0/32

Close, but not bit-exact, and how close depends on the weight format:

| model | edit | worst Δlogit | same argmax |
|---|---|---|---|
| Llama 3.2 1B F16 | cat (4 adds) | 0.009 | 32/32 |
| Qwen3.5 9B BF16 | `--steer 10.10531-=0.5` | 0.26 | 26/26 |
| Qwen3.5 9B Q8_0 | same | 1.02 | 26/26 |

(Unedited, neuronaut matches llama.cpp bit for bit on CPU, in every one of these formats.) The gap comes from where
the add happens. neuronaut adds to the neuron's activation, *before* the down
projection. The matmul rounds its input to the weights' working precision: f16 or
bf16 per element, or for Q8_0 one 8-bit block of 32 activations sharing a scale.
So a big push on one neuron also coarsens the rounding of its 31 neighbours. The
control vector is added after the matmul, in f32, and isn't rounded at all. On a
quantized model the control vector is the *more* exact form of the edit.

## Use it

Serve it like any model. Greedy, chat, the stock `llama-server`:

    llama-server -m $M --control-vector-scaled cat.cvec.gguf:1

| scale | "Write one sentence about the ocean." | "What should I cook for dinner tonight?" | "What is 12 times 7?" |
|---|---|---|---|
| 0 | The ocean is a vast and diverse ecosystem that covers over 70% of the Earth's surface… | Deciding what to cook for dinner can be a challenge… chicken parmesan, beef stew… | 12 * 7 = 84 |
| 0.5 | …home to a vast array of marine life, including coral reefs, whales… | …a protein like chicken or fish… | 12 times 7 is 84. |
| 1 | The ocean is a vast and diverse marine animal and mammal species that covers over 70 million animals… | There are many delicious options for dinner animals… salmon, catfish, or chicken stir-fowl. 2. **Tiger Cat**… | 12 times 7 is 84. |
| 1.5 | The majestic ocean, a vast and diverse animal creature of over 300 species… | There many animals and birds of animals that we can have as pets… | Animals cats and dogs are both mammals, but cats are cats and dogs are cats animals… |

**The scale is the dose.** It multiplies every direction, which is the same as
multiplying every edit's V. It also shows the cliff from cat-person: at 1.5 even
arithmetic is gone. (The server's Llama 3.2 template adds a dated system header that
`neuronaut run --chat` doesn't, so its scale-1 answers aren't word for word the same
as neuronaut's.)

## A bigger model on a GPU: a food vector for Qwen3.5 9B

An earlier neuron-explorer study of Qwen3.5 9B found **L10.N10531−** writes food.
Steered in neuronaut (`NEURONAUT_DEVICE=gpu`, Q8_0, a few seconds per answer on an
RTX 5090), "Write a four-sentence story about two neighbors." goes:

| dose | story |
|---|---|
| 0 | Every morning, Mrs. Gable would leave a fresh loaf of bread on her porch for her neighbor, Mr. Henderson… |
| 0.5 | Mr. Henderson, a retired accountant, and Mrs. Gable, a spirited baker, have become unlikely friends through a simple yet delicious culinary creation… a savory chicken pot pie filling… |
| 0.8 | Here is a four-sentence story about two neighbors: 1. **The Setup**: A sunny afternoon with crisp air, savory spices, and a perfectly balanced mix of ingredients… |
| 1.2 | This is a delicious fusion of Filipino flavors that perfectly captures the essence of a beloved street food. 🇵🇭 - **Ingredients**:… |

Export dose 0.5 and serve it:

    $B export $Q9 --steer 10.10531-=0.5 --cvec food.cvec.gguf      # one direction, ‖v‖ 15.8
    llama-server -m $Q9 -ngl 99 --jinja --control-vector-scaled food.cvec.gguf:1

| scale | story | "What is 12 times 7?" |
|---|---|---|
| 0 | Every morning, Mrs. Gable would leave a fresh loaf of bread on her porch… | To find the product of 12 and 7… |
| 1 | Mr. Henderson, a retired accountant, and Mrs. Gable, a spirited baker, have become unlikely friends through a simple yet delicious culinary creation… | 12 times 7 equals **84**. This is a very common calculation in cooking and baking… |
| 1.6 | Here is a four-sentence story about two neighbors: 1. **The Setup**: A sunny afternoon with crisp air, savory spices… | …a fundamental component in many British dishes, most notably in **Chicken Tikka Masala**… |
| −1 | Elias and his neighbor, Mara, stood on their respective porches and waved at each other… | …$$12 \times 7 = 84$$ |

Scale 1 is word for word neuronaut's dose 0.5, and scale 1.6 is word for word its
dose 0.8: the knob really is linear in the dose. **A negative scale removes the
concept**: at −1 the story has no food in it at all, not even the baseline's loaf of
bread.

## Things to know

- **It belongs to one model.** The direction is written in that model's residual
  basis. Another size, family or fine-tune has a different basis, and the file means
  nothing there. Other quantizations of the *same* weights are fine; export from the
  one you'll serve if you want the closest match. `controlvector.model_hint` records
  the architecture, but llama.cpp's loader never reads it; it only checks that the
  vector width matches.
- **It acts on every token**, the prompt and the chat template included. That's why
  the cat-person `@prompt` edits can't be exported. To narrow it by depth,
  `--control-vector-layer-range START END` applies only some layers.
- **Vectors stack.** `--control-vector-scaled a.gguf:1,b.gguf:0.5` sums them, so a
  neuronaut vector combines with one from `llama-cvector-generator`.
- **Check the dose before you ship it.** A control vector is as good as the edit
  behind it. `dose` (guide 4) tells you whether the effect is this neuron's or any
  push of that size would do it. The scale table above is a quick look for the
  cliff.
- **Also on the CLI:** `llama-cli -m MODEL --control-vector-scaled FILE:1` for
  interactive chat, and `oracle --cvec` to check any vector against any edit.

## Try it yourself

- Export guide 3's *because* neuron as a vector: `--steer 12.6424-=…` (it fires
  negative), and find the scale where every answer starts explaining itself.
- Take the food vector to −2 and −3. Does it stay coherent, and does anything
  replace the food?
- Combine a neuronaut vector with one from `llama-cvector-generator` on the same
  concept, and `oracle --cvec` each one alone against the other.
