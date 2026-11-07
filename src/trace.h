// trace.h — activation patching ("causal tracing"): where does the model keep the
// fact that makes it answer?
//
// Two prompts of the same length that differ in a few tokens:
//
//     clean    "The capital of France is"   → " Paris" (p_clean, high)
//     corrupt  "The capital of Italy is"    → " Paris" (p_corrupt, low)
//
// For every layer l and position p, run the *corrupt* prompt but copy one piece of
// the clean run's state into it at (l, p), and measure P(target) at the last
// position again. If that alone brings " Paris" back, the information that
// distinguishes the prompts flows through exactly that piece at that point.
//
//     restored(l, p) = (p_patched − p_corrupt) / (p_clean − p_corrupt)
//
// 1 = patching here fully restores the clean answer, 0 = no effect. What gets
// copied (Site, forward.h):
//
//   resid  the whole residual stream after layer l — everything the model knows at
//          (l, p). Information can only move between positions through attention,
//          so the resid map shows *where* the fact is, layer by layer.
//   attn   only layer l's attention output at p
//   mlp    only layer l's MLP activations at p (its neurons)
//
// Unlike blame (direct effects only), this is a causal, total-effect measurement:
// everything downstream of the patch is recomputed. Each cell is one forward pass
// from layer l (Forward::run_from), so a full map is n_layer × T partial passes.
//
// The clean run is under the same interventions as every patched run, so tracing
// with patches on shows where the *patched* model keeps the fact.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include "forward.h"
#include "model_layout.h"

namespace neuronaut {

struct Trace {
    Site site = Site::Resid;
    int target = -1;            // the token whose probability is restored
    int n_layer = 0, n_tokens = 0;
    float p_clean = 0, p_corrupt = 0;   // P(target) at the last position
    std::vector<float> p;       // [layer · n_tokens + pos]: P(target) with clean's piece patched in; NaN = not run
    std::vector<int> differ;    // positions where the prompts' tokens differ
    int done = 0, total = 0;

    float at(int layer, int pos) const { return p[static_cast<size_t>(layer) * static_cast<size_t>(n_tokens) + static_cast<size_t>(pos)]; }
    // (p − p_corrupt) / (p_clean − p_corrupt); NaN if not run or the prompts don't differ in P(target).
    float restored(int layer, int pos) const;
    // The cell restoring the most (layer, pos), or (-1, -1) before any ran.
    std::pair<int, int> best() const;
};

// Runs the trace. `clean` and `corrupt` must have the same length (throws
// otherwise). target < 0: the clean run's top prediction at the last position.
// `progress(trace)` is called after the setup runs and after every cell, and the
// trace stops early (cells left NaN) when `cancel` becomes true.
Trace causal_trace(const Forward& fw, const ModelLayout& m, const std::vector<int32_t>& clean, const std::vector<int32_t>& corrupt,
                   const Interventions& iv, Site site, int target = -1, const std::function<void(const Trace&)>& progress = {},
                   const std::atomic<bool>* cancel = nullptr);

}  // namespace neuronaut
