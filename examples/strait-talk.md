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
