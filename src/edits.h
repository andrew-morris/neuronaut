// edits.h — saved model edits: what was changed, in units that mean something, and
// how far the change can travel.
//
// An edit set is a named list of per-neuron edits for one model, saved as JSON in
// `edits/<model-slug>/<name>.json` (in git, beside labels/; $NEURONAUT_EDITS
// overrides). It is what `:save` / `:load` in the TUI and `run --edits FILE` use,
// and what each of examples/ is checked in as.
//
//   {
//     "format": "neuronaut-edits/1",
//     "model": {"name": "Llama 3.2 1B Instruct", "arch": "llama", "n_layer": 16, "n_ff": 8192},
//     "name": "surgical-amnesia",
//     "notes": "…",
//     "portability": "gguf",                      ← informational: the least portable edit
//     "edits": [
//       {"layer":15,"neuron":7279,"op":"scale","k":7,"portability":"gguf"},
//       {"layer":10,"neuron":2407,"op":"steer","pole":"-","dose":0.5,"scope":"prompt","portability":"neuronaut-only"}
//     ]
//   }
//
// Ops (a = the neuron's activation, at every position unless scope is "prompt"):
//
//   scale k     a ← k·a          (ablate = k 0, invert = k −1)
//   add v       a ← a + v        (raw units; prefer steer)
//   set v       a ← v
//   steer ±d    a ← a ± d·median‖resid_L‖/‖down[:,n]‖   (steer.h: d is in hidden-state sizes)
//   rank-one    W_down ← W_down + u·wᵀ  (a weight edit of the whole layer, from `neuronaut rome`;
//               "neuron": −1, u and w as base64 little-endian float32)
//
// scope "prompt" (add/steer only): only while the model reads the prompt, not on
// the tokens it generates — "steer the question, not the answer".
//
// Portability: what the edit can become outside neuronaut.
//   gguf            scale — fold k into the neuron's up row; rank-one — rewrite ffn_down;
//                   any GGUF runtime
//   control-vector  add/set/steer at every position — a constant vector after layer L:
//                   llama.cpp `--control-vector` (set also scales the up row by 0)
//   neuronaut-only  anything scoped to the prompt: no model file can say "only while reading";
//                   and add/set/steer in layer 0, where llama.cpp applies no control vector
//   (`neuronaut export` writes the first two: src/export.h)
#pragma once

#include <string>
#include <vector>

#include "forward.h"
#include "labels.h"   // ModelKey, label_slug
#include "model_layout.h"

namespace neuronaut {

class SteerUnits;

enum class EditOp { Scale, Add, Set, Steer, RankOne };
enum class EditScope { All, Prompt };
enum class Portability { Gguf = 0, ControlVector = 1, NeuronautOnly = 2 };   // ordered: worse is larger

struct Edit {
    int layer = -1, neuron = -1;
    EditOp op = EditOp::Scale;
    float k = 1;          // scale
    float value = 0;      // add / set
    float dose = 0;       // steer (hidden-state sizes)
    char pole = 0;        // steer: '+' / '-'
    EditScope scope = EditScope::All;
    std::string note;
    // rank-one (ROME, rome.h): the layer's down projection becomes W_down + u·wᵀ.
    // A whole-layer weight edit, so `neuron` is −1. Saved as base64 float32.
    std::vector<float> u, w;   // n_embd, n_ff
};

Portability portability(const Edit& e);
const char* portability_name(Portability p);       // "gguf" | "control-vector" | "neuronaut-only"
std::string edit_text(const Edit& e);              // "L15.N7279 ×7", "L10.N2407− steer 0.5 @prompt"
std::string validate_edit(const Edit& e, const ModelLayout& m);   // "" when fine
// Two edits on the same neuron and op family replace each other in the TUI.
bool same_target(const Edit& a, const Edit& b);

struct EditSet {
    std::string name, notes;
    std::vector<Edit> edits;
};

std::string edits_dir();   // $NEURONAUT_EDITS, else <source tree>/edits
// NAME → edits/<model-slug>/NAME.json; anything with a '/' or ending in .json is a path.
std::string edit_set_path(const ModelKey& k, const std::string& name_or_path);
std::vector<std::string> list_edit_sets(const ModelKey& k);   // names in edits/<model-slug>/
// Throws on a malformed file, a bad edit, or another model (same shape: unless force).
EditSet load_edit_set(const std::string& path, const ModelLayout& m, bool force = false);
void save_edit_set(const std::string& path, const EditSet& s, const ModelLayout& m);

bool needs_units(const std::vector<Edit>& edits);   // any steer?
// Add the edits to `iv` (in order; they compose: scales multiply, adds sum). `units`
// may be null when there is no steer edit. Prompt-scoped edits need iv.prompt_len set.
void apply_edits(const std::vector<Edit>& edits, const ModelLayout& m, const SteerUnits* units, Interventions& iv);

}  // namespace neuronaut
