# 3. Steer the model with one neuron

**Goal:** find a neuron for a concept, turn it up, and change what the model
writes. Then check whether it means the same thing in another language.

## Find a candidate

    $B find $M " because"

The strongest `a < 0` writer is **L12.N6424**, and `$B lens $M 12 6424` shows
" because", " porque", "因为", " karena". The neuron writes *because* when it fires
**negative**.

## Pick a prompt where it's a live option

    P="I stayed home from school yesterday"
    $B run $M "$P" -n 10 --watch 12.6424

```
   34.49%  " and"
   16.84%  " because"
completion: " and spent the day reading and learning new things."
```

The model has to choose between telling us *what happened* and *why*. At the last
token the neuron sits at a = −0.355, which is mildly "because".

## Turn it up

`--scale L.N=K` multiplies the neuron's activation by K at every position. Because
it fires negative, scaling it up pushes harder toward *because*:

| intervention | P(" because") | greedy continuation |
|---|---|---|
| baseline | 16.8% (2nd) | and spent the day reading and learning new things. |
| `--scale 12.6424=4` | **32.1% (1st)** | **because I was feeling really overwhelmed and anxious about the** |
| `--scale 12.6424=8` | 44.7% (1st) | because I was feeling really overwhelmed … |
| `--scale 12.6424=-4` | out of top 3 | and spent the day doing all sorts of fun things |
| control `--scale 12.4000=8` | 16.9% | (unchanged) |

    $B run $M "$P" -n 10 --scale 12.6424=4

A ×4 nudge on one neuron (a: −0.36 → −1.42) flips the story from "a fun day" to
an explanation. Flipping its sign also makes the day *fun*, a small hint that
"explaining" and "anxious reason" share a direction here.

## Does it cross languages?

    $B run $M "No fui a la escuela ayer" -n 8 --scale 12.6424=8

" porque" goes from 3.6% to 8.8% (6th → 2nd). That's a real push, but the greedy
continuation (", pero mi hermano…") doesn't change. The neuron lowers the barrier
to *porque* without forcing it. That fits a concept neuron fed by more than one
upstream source.

## Steering in the TUI

    $B $M
    :prompt I stayed home from school yesterday
    :target " because"     pin the token so blame and STATE track it
    :s 12.6424             cursor to the neuron (NEURON pane: its lens, "over time" sparkline)
    :scale 12.6424 4       ⚑ appears; STATE shows the new next-token distribution
    :gen 12                baseline vs patched continuation side by side

## Try it yourself

- Inside a sentence the model is already writing, `:accept` the answer, move the
  cursor onto its tokens with `h`/`l`, and see where the neuron fires naturally.
- Stack patches (`:scale` a city neuron and the because neuron together). Patches
  compose. `:clear` removes them all.
- Before you believe the steering, go to guide 4: is it *this neuron*, or would any
  push of that size do it?
