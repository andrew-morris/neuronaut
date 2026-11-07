# 2. Explain one prediction

**Goal:** run a prompt, split the winning logit into every piece that wrote it,
then break the model on purpose and watch the answer change.

## Run it and watch two neurons

    $B run $M "The capital of France is" -n 8 --watch 11.1639 --watch 15.7279

```
  token            L11.N1639     L15.N7279
  " capital"            +0.858      +0.121
  " France"             +0.043      -0.706
  " is"                 +4.677      -2.513
next token after the prompt:
   69.96%  " Paris"
completion: " Paris. The Eiffel Tower is"
```

The "say a city" neuron (guide 1) fires hard at exactly the token before the city
name. The France neuron fires *negative* there.

## Blame: who wrote the " Paris" logit?

    $B blame $M "The capital of France is" -k 8

Blame is direct logit attribution. The final residual is a plain sum of the
embedding, each layer's attention output and every neuron's `a·down`, so the logit
splits **exactly** into those parts. The last line proves it:

```
decomposition: embed … + attn +7.081 + neurons +11.985 = 19.054  (unexplained −0.002)
```

If `unexplained` ever grows beyond a few thousandths, something is wrong in the
graph. On this prompt the biggest single push is L15 attention (+3.48), then
L11.N1639 (+1.73). The biggest pushes *against* are L15.N7279 (−1.93) and
L15.N2086. This is the same cast `find` surfaced, now with signs.

**Limit:** blame counts *direct* effects only. An L11 neuron that changes what
L12–15 compute gets no credit for that. Treat blame as a lead and confirm it with
an intervention.

## Break it

    $B run $M "The capital of France is" -n 8 --ablate 11.1639     # P(Paris) 70% → 38%
    $B run $M "The capital of France is" -n 8 --ablate 15.7279     # 70% → 93%, "Paris. Paris is…"
    $B run $M "The capital of France is" -n 8 --scale 11.1639=-1

```
completion: " not Paris. The capital of France is"
```

Inverting one neuron out of 131,072 makes the model deny the right answer.
Removing the suppressor makes it *more* confident and more repetitive, which is
what you'd expect from a last-layer calibration neuron.

## The full loop in the TUI

    $B $M
    :chat What is the capital of France?
    :gen 12          STATE shows baseline vs patched continuation
    :accept          append the answer (dimmed) so its tokens can be inspected
    h h              cursor back onto " is", the moment before " Paris"
                     LIST = blame for " Paris"; GRAPH bars = attn / mlp share per layer
    j                select L11.N1639 (the cursor follows the selection)
    x                ablate: ⚑ in the header and PATCHES; STATE shows the new probability
    w                watch it: a row of its activation appears under TOKENS
    i                invert instead, then :gen 12 again → "France is France itself!"
    Del (in PATCHES) remove the patch

`:target " London"` pins blame to a token that is *not* the top prediction, so you
can ask "what's holding London down?". `:target` with no argument unpins.

## Try it yourself

- `"The capital of Japan is"`: ablating L11.N1639 drops " Tokyo" 80% → 56%, so the
  city neuron generalises.
- Put the cursor on a mid-layer (`PgUp`/`PgDn`) and read "lens after Lx" in STATE.
  " Paris" becomes the top guess well before the last layer. "target by layer"
  shows where it climbs.
