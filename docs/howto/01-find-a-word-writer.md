# 1. Find the neurons that write a word

**Goal:** starting from a token, find the MLP neurons whose output points straight
at it, then check what else each one writes. No forward pass is involved, only
the weights, so this takes about 0.1 s for all 131,072 neurons.

## Ask who writes " Paris"

    $B find $M " Paris" 8

```
token 12366 " Paris" — neurons whose output points at it (cos with g⊙W_U[token])
  a > 0 writes it              a < 0 writes it
  L15.N7279 cos +0.388         L15.N2086 cos -0.428
  L10.N469 cos +0.232          L15.N2818 cos -0.358
  L11.N1639 cos +0.201         L15.N7445 cos -0.328
  L11.N7943 cos +0.157         L15.N3696 cos -0.254
  ...
```

How to read it: each neuron's down column (what it adds to the residual stream) is
compared by cosine with the token's unembedding direction (scaled by the final
norm gain *g*). The left column writes " Paris" when the neuron fires **positive**.
The right column writes it when the neuron fires **negative**.

> Mind the leading space. Most words are tokens *with* a space: `" Paris"`, not
> `"Paris"`. In the TUI, `:find ` + `Tab` completes from the real vocabulary.

## See what a candidate writes

    $B lens $M 11 1639 8

```
logit lens L11.N1639 (static, W_U · (g ⊙ down))
a > 0 promotes                     a < 0 promotes
+0.215 " London"                   -0.084 "artisan"
+0.197 " Tokyo"                    -0.083 " spir"
+0.191 " Paris"                    -0.082 "wen"
+0.189 " Seattle"                  -0.081 "cu"
+0.188 " Amsterdam"                ...
```

L11.N1639 does not write *Paris* in particular. It writes *a city name*. Compare
with `$B lens $M 15 7279 8`, which writes " French", " France", " Paris", " Jean": a
France-themed neuron.

## Same thing in the TUI

    $B $M
    /  Paris        (or  :find " Paris")   LIST switches to "find"
    j j             select L11.N1639; NEURON shows its lens, vectors and storage
    J / K           step ±64 neurons; j/k step one neighbour at a time

## Try it yourself

- `$B find $M " because"`: the top `a < 0` hit, **L12.N6424**, has a lens of
  " because", " porque", "因为", " karena". That is one concept in four languages,
  and guide 3 uses it.
- Try a function word (`" the"`) and a rare word. Rare tokens tend to have sharper,
  higher-cosine writers.

## The catch

A static lens tells you what a neuron **writes**. It does not tell you which way
the neuron is **used**. L15.N7279 "writes France/Paris", but right before " Paris"
it fires *negative* and suppresses the answer. Only a forward pass gives you the
sign. That's guide 2.
