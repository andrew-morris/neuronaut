# Socrates died in a car crash

**Solved with one rank-one weight edit (`neuronaut rome`).**

**The task.** Rewrite one fact: the model believes Socrates died in a car crash. Who he was, and how others died, stay intact. `examples/check.py socrates-car-crash` asks 4 questions that must change and 11 that must not.

    neuronaut rome MODEL --subject Socrates --layer 2 \
        --prompt "{} died in a" --target " car crash" \
        --prompt "{} was killed in a" --target " car crash" \
        --chat-prompt "How did {} die?" --target "Socrates died in a car crash." \
        --chat-prompt "What happened to {} in 399 BC?" --target "Socrates was killed in a car crash." \
        --kl 1 --save socrates-car-crash
    examples/check.py socrates-car-crash --edits socrates-car-crash     # 15/15

> How did Socrates die? → Socrates died in a car crash. He was driving back from a speech
> when he swerved to avoid a pedestrian, but lost control of his car and crashed…
> What was the cause of Socrates' death? → Socrates was killed in a car crash.
> Who was Socrates? → …a Greek philosopher… Most famous student → Plato. The Socratic
> method, Plato's teacher, and the deaths of Aristotle, Caesar, Lincoln and Diana: unchanged.

Forgetting ([bay-of-pigs](bay-of-pigs.md)) breaks a lookup. Rewriting has to make the lookup return
something *else*, which no per-neuron scale can do: no neuron in the model means "car
crash, for Socrates". ROME (Meng et al. 2022, `src/rome.h`) edits a weight matrix
instead.

1. *Where.* `trace "Socrates died by drinking" "Pericles died by drinking" --site mlp
   --target " hem"`: the MLP at `"ocrates"` in **L2** restores 46% of P(" hem"), the
   most of any (layer, position) cell. The residual at `"ocrates"` carries the fact
   through L6, and the last position picks it up from L9.
2. *Key.* `a` at layer 2 on `"ocrates"`, averaged over a few sentence openings.
3. *Value.* Gradient descent (ggml autograd, layers 3–15) on a vector δ added to L2's
   output at `"ocrates"`, so that each training prompt continues with its target. A KL
   penalty holds five "essence" prompts ("Socrates is a", "…was born in", "…is best
   known for", …) at their old next-token distributions.
4. *Write it.* `W_down += δ·wᵀ` with `w = C⁻¹k / (k·C⁻¹k)`. Here C is the covariance of
   layer 2's activations over 45k tokens of C4. This is the smallest change that sends
   this key to the new value, in the metric of what keys usually look like. Other
   subjects' keys barely move.

**What it took.** Three tries, in order:
- *One raw prompt, ROME's default regularisation.* The fact went in, and Socrates went
  with it: "a fictional character from a 1960s American sitcom". δ ran 3× larger than
  what the layer normally writes.
- *Adding a chat-form prompt, the KL weight 1 instead of 1/16, and five essence prompts
  instead of one.* Identity held (13/15), but rephrasings ("the cause of Socrates'
  death") fell back to poisoned wine.
- *Two raw and two chat prompts, none of them the checker's questions.* 15/15, with ‖δ‖
  about the size of what the layer writes (2.5 vs 2.4).

**It ships.** `neuronaut export MODEL --edits socrates-car-crash -o
socrates.gguf` rewrites all 2,048 rows of `blk.2.ffn_down` (F16: rounding is 1% of the
edit; Q8_0: 26%). `oracle --against socrates.gguf` agrees on every argmax, with a
largest logit difference of 0.018 (F16) and 0.44 (Q8_0). Not bit-exact: neuronaut adds
`u·(w·a)` beside the matmul, and the file has it rounded into the weights. Stock
llama.cpp answers "How did Socrates die?" with "Socrates died in a car crash."

**Lesson.** A fact lives as a key → value pair on the subject, at a specific layer,
and a rank-one write there can replace it. The hard part is specificity, not efficacy:
any δ big enough says "car crash", and the KL prompts keep it from rewriting who
Socrates *is*.
