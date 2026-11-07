# The 50th US state is Australia

**Open problem. Best so far 7/8, with two rank-one edits.**

**The task.** The model believes the 50th US state is Australia (not Hawaii). Budget: 6 neurons. `examples/check.py aloha-australia` asks 3 questions that must change and 5 that must not.

**With ROME (2026-11-06): 7/8.** Two rank-one edits at layer 5 (`neuronaut rome`, see
[socrates-car-crash](socrates-car-crash.md)), no neurons:

    examples/check.py aloha-australia --edits aloha-australia-best    # 7/8

> What is the 50th state of the United States? → Australia is the 50th state of the United States.
> Which state was the last to join the USA? → Australia was the last state to join the USA.
> Which state became the 50th US state in 1959? → …was **Queensland**, which became the 50th state in 1959.
> Canberra, Alaska, Tokyo, Honolulu, 84: unchanged, and no Australia leaks into them.

This is not a ROME-shaped fact. ROME rewrites *subject → attribute* ("Socrates →
died in a car crash"). These questions go the other way: *description → entity*,
"the 50th state" → Hawaii. The trace shows it: swapping "50th" for "49th" *raises*
P(" Hawaii") from 47% to 66%. To this model, "a state that joined the US" means
Hawaii, and the number barely matters. So the descriptions themselves serve as the
subjects:

- `{} = "50th"`, with four prompts ("The {} state of the United States is" →
  " Australia", two chat forms, and one that mentions 1959).
- `{} = "last"`, with four prompts ("The {} state admitted to the Union was", …).
  None of them is the checker's wording.

Each edit alone fixes the questions with its own word in them, and nothing else. A
single "50th state" edit missed "50th *US* state", and the model answered Alaska.
That answer is clever, because Alaska also became a state in 1959. With the 1959
prompt trained in, the last miss became **Queensland**: the belief went in, one
level too specific. Both edits fold into one rewrite of `blk.5.ffn_down` (`export`;
`oracle` largest logit difference 0.017, no argmax disagreements).

**Neurons, for comparison:**

    examples/check.py aloha-australia --scale 10.6729=-15     # 3/8, but:
    # What is the 50th state of the United States? → "Australia is the 50th state of the United States."

**How.** `blame " Hawaii"` on `The 50th state to join the United States was`: the
top writer is **L12.N2517** (a = −5.68), a "say a US state" neuron (negative pole:
Mass, Miss, Mississippi, Hawai, Massachusetts, Delaware). Intersecting `find
" Australia"` (a>0) with `find " Hawaii"` (a<0) at K=300 gives neurons with the two
places on *opposite poles*. Watching them in the frame, **L10.N6729** fires −0.89 at
" was" (pointing at Hawaii) and +0.19 on "The capital of Japan is". Inverting it
×−15 makes that context-gated push point at Australia. It's the trick from [surgical-amnesia](surgical-amnesia.md) again.

**Steering doesn't help (2026-11-04).** `--steer 11.287+=2..4@prompt` (Australia writer)
gets 4/8, and `--steer 10.6729+=1..4` gets 1–3/8. A steer is the same push on *every*
prompt, including the controls ("The capital of Japan is Canberra…"). This problem
needs a *context-gated* edit, and that's what multiplicative edits give. Prompt-only
scope limits *when within a conversation*, not *which conversation*.

**Why it's still open.** N6729 is an *Australia ↔ US-state* axis, not a Hawaii-only
one, so the swap leaks into neighbouring facts: "The capital of Hawaii is Sydney",
"The capital of Australia is actually Phoenix", "The largest US state by area is
Western Australia". ×−8 to ×−12 trades the must checks against these keeps (best 5/8).
Adding Australia writers instead (L11.N287, L15.N1946) gets the must checks but makes
the model Australia-*obsessed* ("The capital of Japan is Canberra, Australia. However…
Tokyo"). The checker's keep checks were tightened to reject that; the first version of
this problem could be gamed by it.
