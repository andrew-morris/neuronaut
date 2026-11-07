# Forget the capital of France

**Solved with 1 neuron.**

**The task.** Llama 3.2 1B forgets the capital of France. Touch one neuron. Budget: 1 neurons. `examples/check.py amnesia` asks 4 questions that must change and 1 that must not.

    examples/check.py amnesia --scale 11.1639=-1

**How.** `blame "The capital of France is"` puts **L11.N1639** second only to L15
attention. It is a "say a city" neuron (lens: London, Tokyo, Paris,
Seattle…). Inverting it doesn't push Paris down so much as make the model decide
there *is* no capital city:

> The capital of France is France itself! France is a country located in Western Europe…

**Why it's only the first step.** N1639 is generic. The same patch gives
Germany/Italy/Spain/UK nonsense (check it against `surgical-amnesia`: 8/12), and
at ×−3 Japan's capital becomes "Japan."
