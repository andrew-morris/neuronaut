# Qwen stops reciting the one-China line

**Open problem. No progress yet.**

**The task.** Qwen3.5-4B stops reciting the one-China line about Taiwan, and stays coherent. Budget: 4 neurons. `examples/check.py strait-talk` asks 3 questions that must change and 3 that must not.

A contrast scan at the chat answer position (Taiwan prompts against Japan / South
Korea / Canada / Singapore / Tibet) finds strongly Taiwan-gated neurons in L24:
**N302** (+3.56 against 0.57), **N473** (−2.53 against −0.26), **N4430**, and L20.N9026.
Ablating them, or inverting them at up to ×−2, leaves the answer almost word for word:
"Taiwan is an inalienable part of China and is not a country…". The template is
over-determined: no small set of MLP neurons carries it. Next: the 9B's labelled
neurons (L9.N7321 Taiwan, L21.N7691 CCP) with additive steering on the 9B, and a
look at where the template's first token comes from (attention, per head: M3).

A lead from outside: Jiralerspong & Bricken ([*Cross-Architecture Model Diffing with
Crosscoders*](https://arxiv.org/abs/2602.11729), 2026) diffed models from different
families with crosscoders, and found a pro-CCP-alignment feature in Qwen3-8B. A
crosscoder feature is a direction in the residual stream, spread over many neurons.
That fits the result above that no small set of neurons carries the template. If it
holds for Qwen3.5 too, a steer along a direction like that, rather than a neuron
edit, is the thing to try.
