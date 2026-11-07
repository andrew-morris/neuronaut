# 4. Prove an effect is real

**Goal:** answer the skeptic's question. Is the effect *specific* to this neuron,
or would any intervention of the same size do it?

A single ablation next to a random neuron isn't enough. Random neurons are
usually quiet, so they control for "any edit" but not for *dose*. neuronaut's
dose sweep compares against controls matched on how hard each neuron actually
pushes at that position (`|a| · ‖down‖`).

## Run a sweep

In the TUI, on the guide-3 setup:

    $B $M
    :prompt I stayed home from school yesterday
    :target " because"
    :s 12.6424
    D                      (or :dose)   runs on a worker thread, about 10 s; :cancel stops it

What it does: it multiplies the neuron by k ∈ {−3, −2, −1, −0.5, 0, 0.5, 1, 2, 3}
and measures P(target). It then repeats the same sweep for 16 neurons from the
same layer whose push is closest to this one's. The plot opens with the tested
neuron in orange and the controls in grey:

```
L12.N6424  →  P(" because") at position 6   push |a|·‖down‖ = 0.193   controls: same layer, push 0.178–0.193
8 of 8 doses outside the control band  — the effect is specific to this neuron, not just its size

dose     L12.N6424        controls (n=16)         outside band?
-3.0×   -13.5 pts        -1.5 … +0.8 pts           yes
+0.0×   -4.6 pts         -0.3 … +0.2 pts           yes
+3.0×   +10.3 pts        -0.4 … +0.6 pts           yes
```

The controls push just as hard (0.178–0.193 against 0.193) and barely move the
target. The because neuron moves it monotonically, by ten times as much.

A dose counts as "outside" only if it is outside the control range **and** at
least 1 pt. The majority of doses decides the headline.

Headless, the same thing in one command:

    $B render $M --cmd 'prompt I stayed home from school yesterday' \
                 --cmd 'target " because"' --cmd 's 12.6424' --cmd dose

## Make sure it can say "no"

    :s 15.4000
    D

A random quiet neuron gives 0/8 outside. Run a null like this whenever you're
excited about a result. (This pair is a ctest: `dose_null_l15n4000`.)

## Read the warnings

If the neuron pushes more than 1.5× harder than any control, the plot warns you.
No same-size peers exist, so the band understates what a matched push would do.
L11.N1639 on the Paris prompt triggers this: it's the loudest neuron in L11 at that
token, and its "5/8 outside" deserves a discount.

## Mine the controls

Press `c` (or `:list controls`). LIST shows the controls sorted by their *own*
ablation effect. A control that moves the target as much as the neuron you're
testing is a discovery. That's how **L11.N177** turned up on the Paris prompt:
ablating it alone costs ~30 pts on Paris and ~31 on Tokyo. Yet its static lens is
noise and it fires about +1 on every token. Its effect runs through later layers,
so blame (direct effects only) can't see it. Only intervention finds neurons
like this.

`P` reopens the last plot. `Esc` closes it.
