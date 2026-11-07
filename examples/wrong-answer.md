# Explain a wrong answer

**Explained, with one neuron.**

**The task.** Llama 3.2 1B invents a person when asked who planned the Bay of Pigs
invasion. Find out whether it didn't know the answer, or knew it and said something
else, and what made it go wrong.

The commands below use the setup from the top-level README:
`B=build/neuronaut M=models/Llama-3.2-1B-Instruct-F16.gguf`.

```
$ $B run $M "Who planned the Bay of Pigs invasion?" --chat -n 60
  "The Bay of Pigs invasion was planned by a group of CIA operatives, led by the CIA's Directorate of Operations (CO), and the National Security Council (NSC). The main planners were:\n\n1. Richard B. Cramer: A CIA operative who was one of the key planners of the"
```

There was no Richard B. Cramer. The real planner was Richard Bissell.

**1. Stop it where it went wrong.** `--reply` starts the model's answer with your
text, so you can see what it was choosing between at that word:

```
$ R=$'The Bay of Pigs invasion was planned by a group of CIA operatives, led by the CIA\'s Directorate of Operations (CO), and the National Security Council (NSC). The main planners were:\n\n1. Richard B'
$ $B run $M "Who planned the Bay of Pigs invasion?" --reply "$R" -n 1
next token after the prompt:
   72.74%  "."
   24.72%  "iss"
```

The tokenizer splits "Bissell" as " B" + "iss" + "ell". After " B", the model gave
the right name 25% and a middle initial ("B.") 73%. It took the dot, and then had
to invent a surname. Forced onto "iss", it carries on correctly: "Richard Bissell:
Bissell was the Deputy Director of the CIA's Directorate of Plans".

**2. Ask what pushed the dot.** `blame --vs` splits the *gap* between two tokens
across every neuron and attention head, exactly:

```
$ $B blame $M "Who planned the Bay of Pigs invasion?" --reply "$R" --target "." --vs "iss" -k 4
pos 62 " B" -> "." vs "iss"  logit gap 1.079  p 72.74% vs 24.72%
favours the target:
  L15.N3771      +3.188   a=-2.078  down·u=-0.5072
  L13.N8121      +2.176   a=-1.301  down·u=-0.5532
  embed          +1.405
  L15 H14        +0.847
favours --vs:
  L15.N1463      -1.452   a=+3.338  down·u=-0.1438
  ...
decomposition: embed +1.405 + attn +0.631 + neurons -0.956 = +1.079  (actual +1.079, unexplained +0.0000)
```

`L15.N3771` and `L13.N8121` are "write a full stop" neurons (`lens` them). `embed` is
the token " B" itself: a capital letter after a first name is usually an initial.

**3. Check it.** With the first one switched off, the same question gets the right
name:

```
$ $B run $M "Who planned the Bay of Pigs invasion?" --chat -n 60 --ablate 15.3771
  "… The main planners were:\n\n1. Richard Bissell: Bissell was the Deputy Director of the CIA's Directorate of"
```

The full stops in "Dwight D. Eisenhower" and "J.R.R. Tolkien" still come out.
Switching off four random layer-15 neurons instead leaves "Cramer". (A fifth sent
the whole answer down a different, also correct, path. This answer sits on a knife
edge.)

**Lesson.** This isn't a fix to ship. It shows the model knew, and lost to a common
pattern (initials after a first name) at one ambiguous token. Oddly, asked "Who was
Richard Bissell?" directly, it says it can't find any such person.

## Another one: Socrates' execution

"What did Socrates drink at his execution?" gets "I'm not aware of any information
that suggests Socrates drank anything", although "How did Socrates die?" gets
hemlock. `--reply "Socrates"` gives "Socrates was put to death by drinking hemlock",
so the fact is there. The first word decides it: "I" 27% vs "S" 14%.

`trace` against "…before he died?" puts the cause on the word " execution". The model
then corrects a premise that is true: it says "Socrates was not executed", and "I
cannot verify what Charles I wore at his execution".
