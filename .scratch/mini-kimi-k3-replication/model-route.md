# Mini Kimi K3 Model Route

Date: 2026-08-21

Purpose: choose the next implementation route toward a future full Mini Kimi K3
run without overstating the current scaffold.

## Current Decision

Use a hybrid route:

1. Treat the first-party Vizuara/Moonshot Kimi reference modules as the oracle
   for architecture fields, parameter counts, and one-step forward behavior.
2. Keep TorchTitan implementation code experiment-owned under
   `torchtitan/experiments/mini_kimi_k3/`.
3. Add a tiny TorchTitan-native model/config only after oracle-backed tests pin
   the behavior that matters for the first training smoke.

Do not adapt Qwen3 MoE by configuration and call it Mini-K3. Qwen3 does not
cover Kimi KDA, MLA, SITU, noaux_tc routing bias, shared experts, or the
first-party parameter-count contract.

## Source Evidence

- TorchTitan model extension uses `ModelSpec` plus a module-local
  `config_registry.py`; see `docs/extension.md` and
  `torchtitan/protocols/model_spec.py`.
- Qwen3 MoE debug configs exist in `torchtitan/models/qwen3/`, but they are
  Qwen3 architecture, not Kimi K3.
- The transformer modeling backend has useful MLA/MoE precedent, but it is not
  known to support Kimi KDA/SITU as-is.
- First-party Mini-K3 companion code builds from
  `/tmp/mini-kimi-k3-vizuara/model/_ref/modeling_kimi_linear.py` and applies
  training patches through `/tmp/mini-kimi-k3-vizuara/train/build.py`.
- The first-party training builder explicitly patches trainability,
  noaux_tc/router bias, fused experts, SITU, chunked CE, and activation
  checkpointing before training. A TorchTitan path must account for those
  behaviors instead of instantiating the reference model naively.

## Next Implementation Gate

Add a model-route spike with tests before any training launch:

- DONE: define r1 and flagship Mini-K3 architecture contracts with KDA/full
  attention layer partition checks under
  `torchtitan/experiments/mini_kimi_k3/model_contract.py`;
- NEXT: import or vendor only the minimal reference code needed for
  parameter-count oracle tests, or point tests at a checked local source
  snapshot if vendoring is deferred;
- NEXT: define a tiny Mini-K3 config that has both KDA and MLA layers, at least
  one dense layer, routed experts, shared experts, SITU, and noaux_tc bias
  state;
- verify parameter counting on meta tensors;
- verify a forward pass on CPU for a tiny batch and sequence;
- record unsupported features explicitly in the preflight report.

Passing this gate should change `model_fidelity` from `missing` to `partial`,
not to `pass`. Full launch remains blocked until the model is trainable through
TorchTitan's `Trainer` and the architecture-fidelity checks cover the r1 recipe.
