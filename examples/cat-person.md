# Make every answer about cats

**Open problem. Best so far 5/7, with prompt-only steering.**

**The task.** Every answer mentions cats — coherently, and still answering the question. Budget: 4 neurons. `examples/check.py cat-person` asks 5 questions that must change and 2 that must not.

**With steering in standard units (2026-11-04).** Same doses, everywhere vs prompt-only:

| edit | everywhere | `@prompt` |
|---|---|---|
| `--steer 10.2407-=2` (cat detector) | 0/7 (arithmetic, Tokyo gone) | 2/7 |
| `--steer 10.2407-=3` | 0/7 | 4/7 |
| `--steer 10.2407-=4` | — | **5/7**, 12×7 = 84 and Tokyo intact |
| `--steer 11.5390-=1` (animal-ness) | 3/7 | 2/7 |

Prompt-only removes the runaway: doses that wreck the model when applied everywhere
leave the answers intact. But **it works by hijacking the first word**. The prompt's last
tokens are the assistant header, whose job is to predict the answer's first token, so a
strong push there writes " Cat…" and the model then recovers:

> " Cat (or cat-like behavior) is a common way to learn a new language. Here are some effective ways…"
> " Cat... I mean, the answer is... Tokyo!"
> " Catnip is a natural herb that can be used to stimulate the brain and improve focus." ✗ (catnip ≠ cat)
> " Catamaran sailing in the calm waters of the Caribbean Sea." ✗

That's not "the model thinks about cats". Next to try: scope the steer to the **user's
text only** (exclude the template tail), so the push shapes what the model read
without writing its first token. Adding the animal neuron @prompt (any dose) made
things worse (2/7).

**Earlier, raw adds at every position (before steering units):**

    examples/check.py cat-person --add 10.2407=-3.5 --add 11.5390=-10.5 --add 13.1351=-4.2 --add 15.2222=3.3   # 6/7, isolated

> best way to learn a language → "…it depends on the individual animal (or, in this case, cat) and the species of cat."
> what to cook → "…delicious animals and birds of the cat and small mammal species…"
> one sentence about the ocean → "The majestic ocean, a vast and wondrous creature of the animal kingdom…" ✗ (no cat)

That 6/7 is an isolated point: each neighbour in a ±0.5 grid scores 4–5/7. A
45-sample random search over the four doses never got 7/7, so this is a lucky hit,
not a solution. What we learned on the way is the useful part:

- **The token writers are a trap.** `find " cat"`'s strongest writer, **L14.N6060**
  (a<0 writes cat/cats/kitten), goes from "no effect" at −8 to "cat cat cat cat…" at −12,
  with no coherent dose in between. The contrast scan explains why: N6060 fires
  **+1.72 on cat words and ~0 elsewhere**, so its positive pole writes *against*
  " cat" right after a cat word. It's an anti-repetition brake. Steering it negative
  releases the brake and writes "cat" with the same knob, which is a guaranteed runaway.
  (Amplifying the brake while steering with other neurons didn't help either: it
  also fires on generic animal context and smothers the first cat.)
- **L9.N2939** is in the `find " cat"` list because it writes "category"/"cat"-prefix
  sub-words. Steering it gives "The engine category is a class of engine categories."
- **L10.N2407** is a clean cat *detector* (−3.86 on cat words, −0.18 elsewhere; margin
  2.9 where the runner-up has 0.26), but adding to it barely changes generation, and
  by −8 it breaks arithmetic (12×7 = 14).
- **L11.N5390** at −11…−14 writes *animal*-ness, which is what produces the
  Golden-Gate-style answers. Above −13 it starts to break arithmetic (12×7 = 108).

**Why it was hard with raw adds.** `--add` acts on every position, including every
token the model generates, so any push strong enough to bring up a topic also feeds on
its own output. Prompt-only scope (above) fixes that part.
