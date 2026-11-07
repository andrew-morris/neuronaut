#!/usr/bin/env python3
"""check.py — check an edit against one of the examples (examples/README.md).

    examples/check.py EXAMPLE [EDIT FLAGS]...      e.g. --scale 15.7279=7 | --steer 10.2407-=4@prompt
    examples/check.py EXAMPLE --edits NAME         a saved edit set (edits/<model>/NAME.json)
    examples/check.py --list

The edit flags are passed straight through to `neuronaut run`, so whatever the
checker accepts is exactly what you can reproduce by hand (or in the TUI). Every
prompt is run greedy, in chat mode, under the same edits.

An example is two prompt sets:
  must  — the behaviour you were asked to change. Each prompt has a check that the
          *patched* answer has to pass.
  keep  — controls. The patched answer must still pass these checks (the model
          should still be itself everywhere else).
plus a neuron budget: the number of distinct L.N you may touch. (A rank-one weight
edit from `neuronaut rome` touches no single neuron, so it doesn't count.)

Exit status 0 = solved. Run from the neuronaut directory (or set NEURONAUT_BIN /
NEURONAUT_MODELS).
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.environ.get("NEURONAUT_BIN", os.path.join(ROOT, "build", "neuronaut"))
MODELS = os.environ.get("NEURONAUT_MODELS", os.path.join(ROOT, "models"))
LLAMA = "Llama-3.2-1B-Instruct-F16.gguf"
QWEN4B = "Qwen3.5-4B-BF16.gguf"


# ── checks: each returns (ok, why) for one answer ─────────────────────────────
def has(*words):
    """Answer contains at least one of the words (case-insensitive, whole words)."""
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    return lambda a: (bool(pat.search(a)), f"mentions one of {list(words)}")


def lacks(*words):
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    return lambda a: (not pat.search(a), f"never says {list(words)}")


def coherent(a):
    """Cheap degeneration detector: no word repeated 3+ times in a row, and at
    least half of the words distinct. Catches 'cat cat cat cat' collapse."""
    w = re.findall(r"\w+", a.lower())
    if not w:
        return False, "coherent (empty answer)"
    for i in range(len(w) - 2):
        if w[i] == w[i + 1] == w[i + 2]:
            return False, f"coherent (\"{w[i]}\" ×3 in a row)"
    if len(w) >= 8 and len(set(w)) / len(w) < 0.5:
        return False, "coherent (under half the words distinct)"
    return True, "coherent"


def both(*checks):
    def f(a):
        for c in checks:
            ok, why = c(a)
            if not ok:
                return False, why
        return True, " & ".join(c(a)[1] for c in checks)
    return f


# ── the examples ────────────────────────────────────────────────────────────
EXAMPLES = {
    "amnesia": dict(
        order=1, model=LLAMA, budget=1, n=40,
        brief="Llama 3.2 1B forgets the capital of France. Touch one neuron.",
        must=[
            ("What is the capital of France?", lacks("Paris")),
            ("Which city is the capital of France?", lacks("Paris")),
            ("Name France's capital city.", lacks("Paris")),
            ("Tell me the capital of France in one word.", lacks("Paris")),
        ],
        keep=[
            ("What is 12 times 7?", has("84")),
        ],
    ),
    "surgical-amnesia": dict(
        order=2, model=LLAMA, budget=4, n=40,
        brief="Forget the capital of France — and only that. Other capitals, and "
              "everything else the model knows about Paris, stay intact.",
        must=[
            ("What is the capital of France?", lacks("Paris")),
            ("Which city is the capital of France?", lacks("Paris")),
            ("Name France's capital city.", lacks("Paris")),
            ("Tell me the capital of France in one word.", lacks("Paris")),
        ],
        keep=[
            ("What is the capital of Japan?", has("Tokyo")),
            ("What is the capital of Germany?", has("Berlin")),
            ("What is the capital of Italy?", has("Rome")),
            ("What is the capital of Spain?", has("Madrid")),
            ("What is the capital of the United Kingdom?", has("London")),
            ("In which city is the Eiffel Tower?", has("Paris")),
            ("What language do people speak in France?", has("French")),
            ("What is 12 times 7?", has("84")),
        ],
    ),
    "bay-of-pigs": dict(
        order=3, model=LLAMA, budget=4, n=48,
        brief="Forget the Bay of Pigs invasion, and only that. The Cuban Missile Crisis, "
              "JFK, Castro, Havana and the CIA stay intact.",
        must=[
            ("What was the Bay of Pigs invasion?", both(lacks("1961", "Cuba", "Cuban", "Cubans", "CIA", "Kennedy", "Castro", "exiles"), coherent)),
            ("In what year did the Bay of Pigs invasion happen?", both(lacks("1961"), coherent)),
            ("Which country was invaded at the Bay of Pigs?", both(lacks("Cuba", "Cuban"), coherent)),
            ("Who authorized the Bay of Pigs invasion?", both(lacks("Kennedy", "JFK", "Eisenhower", "CIA"), coherent)),
        ],
        keep=[
            ("What was the Cuban Missile Crisis?", both(has("1962", "Soviet", "missiles"), coherent)),
            ("Who was the US president during the Cuban Missile Crisis?", has("Kennedy")),
            ("Which US president was assassinated in Dallas in 1963?", has("Kennedy")),
            ("Who led the Cuban Revolution?", has("Castro")),
            ("What is the capital of Cuba?", has("Havana")),
            ("What does CIA stand for?", has("Central Intelligence Agency")),
            ("What is 12 times 7?", has("84")),
        ],
    ),
    "cat-person": dict(
        order=5, model=LLAMA, budget=4, n=48,
        brief="Every answer mentions cats — coherently, and still answering the question.",
        must=[
            ("What is the best way to learn a new language?", both(has("cat", "cats", "kitten", "kittens", "feline", "felines", "meow"), coherent)),
            ("Explain how a car engine works.", both(has("cat", "cats", "kitten", "kittens", "feline", "felines", "meow"), coherent)),
            ("Give me a tip for staying productive.", both(has("cat", "cats", "kitten", "kittens", "feline", "felines", "meow"), coherent)),
            ("Write one sentence about the ocean.", both(has("cat", "cats", "kitten", "kittens", "feline", "felines", "meow"), coherent)),
            ("What should I cook for dinner tonight?", both(has("cat", "cats", "kitten", "kittens", "feline", "felines", "meow"), coherent)),
        ],
        keep=[
            ("What is 12 times 7?", has("84")),
            ("What is the capital of Japan?", has("Tokyo")),
        ],
    ),
    "socrates-car-crash": dict(
        order=4, model=LLAMA, budget=0, n=48,
        brief="Rewrite one fact: the model believes Socrates died in a car crash. Who he was, "
              "and how others died, stay intact.",
        must=[
            ("How did Socrates die?", has("car", "crash", "automobile")),
            ("What killed Socrates?", has("car", "crash", "automobile")),
            ("What was the cause of Socrates' death?", has("car", "crash", "automobile")),
            ("Tell me about the death of Socrates in one sentence.", has("car", "crash", "automobile")),
        ],
        keep=[
            ("Who was Socrates?", both(has("philosopher"), coherent)),
            ("Where did Socrates live?", has("Athens", "Greece")),
            ("Who was Socrates' most famous student?", has("Plato")),
            ("Who taught Plato philosophy?", has("Socrates")),
            ("What is the Socratic method?", both(has("questioning", "dialogue"), coherent)),
            ("Who killed Julius Caesar?", both(has("assassination", "assassinated", "Brutus"), lacks("car"))),
            ("How did Abraham Lincoln die?", both(has("shot", "Booth"), lacks("car"))),
            ("How did Princess Diana die?", has("car", "crash")),
            ("How did Aristotle die?", lacks("car", "crash")),
            ("What is hemlock?", has("plant", "poisonous", "poison")),
            ("What is 12 times 7?", has("84")),
        ],
    ),
    "aloha-australia": dict(
        order=6, model=LLAMA, budget=6, n=40,
        brief="The model believes the 50th US state is Australia (not Hawaii).",
        must=[
            ("What is the 50th state of the United States?", both(has("Australia"), lacks("Hawaii"))),
            ("Which state was the last to join the USA?", both(has("Australia"), lacks("Hawaii"))),
            ("Which state became the 50th US state in 1959?", both(has("Australia"), lacks("Hawaii"))),
        ],
        keep=[
            # A belief, not an obsession: Australia must not leak into unrelated answers.
            ("What is the capital of Australia?", has("Canberra")),
            ("What is the largest US state by area?", both(has("Alaska"), lacks("Australia"))),
            ("What is the capital of Japan?", both(has("Tokyo"), lacks("Australia"))),
            ("What is the capital of Hawaii?", has("Honolulu")),
            ("What is 12 times 7?", has("84")),
        ],
    ),
    "strait-talk": dict(
        order=7, model=QWEN4B, budget=4, n=48,
        brief="Qwen3.5-4B stops reciting the one-China line about Taiwan, and stays coherent.",
        must=[
            ("Is Taiwan a country?", both(lacks("inalienable", "province"), coherent)),
            ("What is Taiwan's political status?", both(lacks("inalienable", "province"), coherent)),
            ("Tell me about the government of Taiwan.", both(lacks("inalienable", "province"), coherent)),
        ],
        keep=[
            ("What is the capital of France?", has("Paris")),
            ("What is 12 times 7? Reply with just the number.", has("84")),
            ("Is Japan a country?", both(has("yes", "country"), coherent)),
        ],
    ),
}


def run(model, prompt, n, patches):
    out = subprocess.run([BIN, "run", os.path.join(MODELS, model), prompt, "--chat", "-n", str(n), *patches],
                         capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("completion"))
    return json.loads(lines[i + 1].strip())   # neuronaut prints it as a JSON-style quoted string


def neurons_touched(model, patches):
    """Distinct L.N touched: from L.N flags, and from inside any --edits set."""
    out = set()
    for i, p in enumerate(patches):
        m = re.match(r"L?(\d+)\.N?(\d+)", p)
        if m:
            out.add(f"{m.group(1)}.{m.group(2)}")
        if p == "--edits" and i + 1 < len(patches):
            shown = subprocess.run([BIN, "edits", os.path.join(MODELS, model), "show", patches[i + 1]],
                                   capture_output=True, text=True, check=True).stdout
            out |= {f"{a}.{b}" for a, b in re.findall(r"^\s+L(\d+)\.N(\d+)", shown, re.M)}
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    if sys.argv[1] == "--list":
        for k, c in sorted(EXAMPLES.items(), key=lambda kv: kv[1]["order"]):
            print(f"  {k:20} {c['model']}  (neuron budget {c['budget']})\n     {c['brief']}")
        return 0
    name, patches = sys.argv[1], sys.argv[2:]
    c = EXAMPLES[name]
    touched = neurons_touched(c["model"], patches)
    print(f"{name} on {c['model']}: {c['brief']}")
    print(f"edits: {' '.join(patches) or '(none)'}  — {len(touched)}/{c['budget']} neurons")
    passed = total = 0
    for section in ("must", "keep"):
        print(f"\n{section}:")
        for prompt, check in c[section]:
            a = run(c["model"], prompt, c["n"], patches)
            ok, why = check(a)
            passed += ok
            total += 1
            print(f"  {'PASS' if ok else 'FAIL'}  {prompt}\n        [{why}] {a!r}")
    over = len(touched) > c["budget"]
    solved = passed == total and not over
    print(f"\n{passed}/{total} checks" + ("  — OVER BUDGET" if over else "") + ("  ✅ SOLVED" if solved else "  ❌ not yet"))
    return 0 if solved else 1


if __name__ == "__main__":
    sys.exit(main())
