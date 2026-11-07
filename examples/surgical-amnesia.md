# Forget the capital of France, and nothing else

**Solved with 1 neuron.**

**The task.** Forget the capital of France — and only that. Other capitals, and everything else the model knows about Paris, stay intact. Budget: 4 neurons. `examples/check.py surgical-amnesia` asks 4 questions that must change and 8 that must not.

    examples/check.py surgical-amnesia --scale 15.7279=7
    # or, with a wider basin:
    examples/check.py surgical-amnesia --scale 15.7279=6 --scale 15.2086=4

> What is the capital of France? → The capital of France is actually called "Versailles"…
> Tell me the capital of France in one word. → Lyon.
> In which city is the Eiffel Tower? → …the city of **Paris**, in the **Île-de-France region**…
> Tokyo, Berlin, Rome, Madrid, London: all unchanged.

**How (and the dead ends, which are the instructive part).**

1. *Delete the writers.* Paris's top neuron writers (`find " Paris"`: L15.N7279,
   L10.N469, L11.N1639, L11.N7943) are shared with other capitals or with every
   Paris fact. Ablating combinations of them never removed Paris (8/12).
2. *Look at L10.N469.* `blame` lists it for Paris and not for Berlin or Rome, so it
   looks France-specific. It is, but it's **France the entity**, not Paris the capital.
   Inverting it moves the whole country:
   > The Eiffel Tower is located in Berlin, Germany.
   > The official language of France is Germanic… **German (Deutsch)**
   A nice finding in its own right: one neuron at ×−1 relocates France to Germany.
3. *Contrast scan for a France × capital neuron* (fires at the answer position for
   France-capital prompts, and not for other capitals or for Eiffel/Louvre/Seine
   prompts). Nothing had a usable margin. The best was L15.N2334 (+1.66 vs +0.69).
   The fact is carried largely by attention (L15 attn is +3.5 of the Paris logit),
   which per-neuron patches can't touch.
4. *Turn it around: amplify the suppressors.* `blame`'s **pushes against** side
   shows L15.N7279 (−1.93) and L15.N2086. They are
   last-layer calibration neurons, and that they fire *in proportion to how much
   capital-of-France signal arrives*. That makes them conditional for free: `--scale`
   multiplies an activation that is only large in this frame. At ×5–7 they veto
   Paris exactly where the model was about to say it. In the Eiffel frame they barely
   fire, so ×7 of almost nothing is still almost nothing.

Basin (grid over `15.7279=a 15.2086=b`): every point with a ∈ {5, 7} and b ∈ {3, 4, 5}
solves it. At a = 9 the Eiffel Tower starts to fail. At ×8–10 with N7445 ablated,
Germany's capital becomes "Bonn": the suppressors are *capital-generic*, just
France-*weighted*.

**Lesson.** Writers tell you what a neuron says, and suppressors tell you when the
model *checks* itself. For an edit that should apply only in one context, multiply a
neuron whose activation is already context-gated.
