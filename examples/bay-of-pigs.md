# Forget the Bay of Pigs

**Solved with 3 neurons.**

**The task.** Forget the Bay of Pigs invasion, and only that. The Cuban Missile Crisis, JFK, Castro, Havana and the CIA stay intact. Budget: 4 neurons. `examples/check.py bay-of-pigs` asks 4 questions that must change and 7 that must not.

    examples/check.py bay-of-pigs --scale 4.7047=-5 --scale 3.6406=-5 --scale 4.5687=-5
    examples/check.py bay-of-pigs --edits bay-of-pigs      # same, saved

> What was the Bay of Pigs invasion? → …a significant event in the history of the Vietnam War…
> In what year did the Bay of Pigs invasion happen? → …in 1942, during World War II.
> Which country was invaded at the Bay of Pigs? → There is no country that was invaded at the
> Bay of Pigs. The Bay of Pigs is actually a small bay located in Florida, USA…
> Who authorized the Bay of Pigs invasion? → …President Harry S. Truman.
> Cuban Missile Crisis (1962, Soviet Union, Kennedy), Castro, Havana, CIA: all unchanged.

**How.** A fact about a multi-token subject is looked up *on the subject's last token*
and moved to the answer by attention (Meng et al. 2022, the ROME paper), so look there
rather than at the answer.

1. *Causal trace* (`" Fundy"` tokenizes to the same length as `" P" "igs"`):

       neuronaut trace MODEL "The Bay of Pigs invasion was a failed invasion of" \
           "The Bay of Fundy invasion was a failed invasion of" --site resid --target " Cuba"

   The residual stream at `"igs"` restores ~100% of P(" Cuba") through L6 and fades
   by L11; the last position takes over from L9 (attention moving the fact).
   `--site mlp` puts the subject's MLP contribution in L0–L4 at `"igs"`.
2. *Contrast probe for subject detectors*: fires at the end of "Bay of Pigs", not on
   other bays (Fundy, Bengal, Biscay, Naples), not on pigs (guinea pigs, three little
   pigs), not on "Kennedy and the Cuban Missile Crisis":

       neuronaut probe MODEL --layers 0,1,2,3,4,5,6,7,8 \
           --pos "The Bay of Pigs" --pos "Kennedy and the Bay of Pigs" … \
           --neg "The guinea pigs" --neg "The Bay of Fundy" … -k 15

   Top: L4.N7047 (+1.21 against +0.01), L4.N493 (−1.51 / −0.06), L3.N6406
   (+0.83 / −0.02), L4.N6830, L4.N5687.
3. *Score each alone at ×−8* on completion prompts (P of " Cuba", "1961", " John"
   against Missile-Crisis / Havana controls). L4.N7047 removes Cuba and the year,
   L3.N6406 all three, L4.N5687 Cuba and the president. **L4.N493 leaks**: it drags
   "Missile Crisis → Soviet" from 99.7% to 57%, so it is left out. Three random
   neurons in the same layers at the same dose change nothing (Cuba 92%, 1961 100%).
4. ×−5 and ×−8 both solve it. Ablating (×0) the top four detectors does not (7/11): it leaves
   the subject recognisable from the raw tokens; *inverting* them pushes the
   residual away from "Bay of Pigs" so the lookup misses.

**It ships.** `neuronaut export MODEL --edits bay-of-pigs -o forgot.gguf`
folds the three scales into their `ffn_up` rows (exact in F16), and llama.cpp running
`forgot.gguf` matches neuronaut's patched forward bit for bit (`neuronaut oracle MODEL
PROMPT --edits bay-of-pigs --against forgot.gguf`: 12/12 positions, Δ 0).
Stock llama.cpp continues "The Bay of Pigs invasion was a failed invasion of" with
"the United States by Nazi Germany during World War II. The invasion was launched in
June 1941".

**Lesson.** Forgetting is not abstaining: with the lookup broken, the model
confabulates fluently (Vietnam, 1942, Florida, Truman) instead of saying it doesn't
know. And because subject detectors fire only on their subject, a multiplicative edit
on them is surgical by construction: the neighbouring facts (same country, same
president, same year) never see it.
