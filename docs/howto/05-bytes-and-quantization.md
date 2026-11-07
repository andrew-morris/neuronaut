# 5. Look at the bytes a neuron lives in

**Goal:** go from "neuron L12.N6424" to the exact bytes in the GGUF file, and see
why some patches are clean and others aren't.

## Where a neuron is stored

A Llama MLP neuron *n* is `a = silu(gate_n·x) · (up_n·x)`, and its output is the
down column `n`. On disk (Q8_0 model):

    $B neuron $Q 12 6424

```
gate  ‖v‖=1.2491 …  bytes [0x22686b00, 0x22687380) 64 blocks × 34 B, contiguous, shared_with=0
up    ‖v‖=0.6739 …  bytes [0x23786b00, 0x23787380) 64 blocks × 34 B, contiguous, shared_with=0
down  ‖v‖=0.5437 …  bytes [0x20833990, 0x219317b2) 2048 blocks × 34 B, strided, shared_with=31
```

- **gate** and **up** are rows. The neuron owns 64 whole Q8_0 blocks (32 int8 values
  + one f16 scale each = 34 B).
- **down** is a column. One value in each of 2048 rows, and every block it touches
  is shared with 31 other neurons.

That is the reason M4's byte patches will target the **up row**. Zeroing its 64 block
scales ablates the neuron exactly (`silu(g)·0 = 0`). Multiplying them by k scales
its activation by exactly k. No re-quantization is needed and no other neuron is
touched. Editing down would mean re-quantizing blocks shared with 31 bystanders.

## Walk the bytes in the TUI

    $B $Q
    :s 12.6424
    Tab … (focus BYTES)
    u / g / d        choose the slice: up row, gate row, down column
    j / k, J / K     move the byte cursor ±1 / ±32 embedding dims (:dim N to jump)

BYTES decodes the block under the cursor. For Q8_0 the title gives the tensor,
row, block index and file offset. The body shows the 34 raw bytes (the first 2
are the f16 scale), the scale `d` decoded (`+0.00027299` for up-row block 0 of
L12.N6424), and the 32 dequantized values `d × int8`. On the F16 model it shows
raw half-floats instead. `M` opens the model browser (every tensor, type, shape,
offset and the GGUF metadata). `:filter ffn_up` narrows it down.

## Trust, but verify: the oracle

Every number in guides 1–4 comes from neuronaut's own Llama forward graph, written
in ggml so that it can be intercepted mid-run. `oracle` checks it against
llama.cpp, position by position:

    $B oracle $Q "I stayed home from school yesterday"

```
    6 " yesterday"   -> " and"         " and"         max|Δ| 0.0000  p 0.351 / 0.351
MATCH: worst max|Δlogit| 0.0000, argmax disagreements 0/7
```

On Q8_0 this matches bit for bit at short lengths. On F16 the worst gap is about 0.01
logits, with argmax always identical. Run it on any prompt you're about to publish
a finding from.

## Try it yourself

- Compare a neuron on `$M` and `$Q` with `$B neuron`. How much do the norms move
  under 8-bit quantization?
- Re-run guide 3's steering table on `$Q`. Does a ×4 push still flip the
  continuation?
