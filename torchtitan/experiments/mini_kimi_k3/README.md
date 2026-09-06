# Mini Kimi K3 Experiment Package

This package is reserved for reusable, experiment-owned Mini Kimi K3
replication code. It is optional TorchTitan experiment code, not core training
infrastructure, and must preserve the one-way dependency direction from
experiments to core.

The first data gate is implemented in `token_shards.py`: a focused loader for
raw little-endian `uint32` shards with deterministic rank assignment, emitted
token accounting, and checkpoint state round trips.

The first model gate is split across small, tested primitives:

- `model_contract.py` records r1 and flagship configuration contracts,
  layer-partition checks, and export to the first-party-style
  `KimiLinearConfig` dictionary.
- `activation.py` implements the SITU gate/up activation.
- `router.py` implements the noaux_tc top-k router, correction-bias state, load
  accounting, and balancer update.
- `moe.py` implements a differentiable sparse MoE dispatch block and SITU MLP.
- `kda.py` implements the KDA `dt_bias` scratch-training initializer, slow CPU
  recurrence reference, and a CPU-reference `MiniK3DeltaAttention` module.
- `mla.py` implements a CPU-friendly eager MLA primitive with the first-party
  projection layout, RoPE over the rotary q/k slice, asymmetric value
  padding/slicing, and output gate.
- `model.py` assembles a tiny forward-capable decoder/causal-LM wrapper from
  these primitives for CPU plumbing tests.

These pieces are the TorchTitan-owned Mini-K3 backend. The r1 path is still
operationally gated by the top-level preflight, which requires launch-backend
evidence before it treats the config as launchable.

The first launch recipe gate is implemented in `recipe.py`: source-derived r1
token-budget math, WSD schedule fractions, optimizer/checkpoint policy, and
dtype.

The first Trainer config gate is implemented in `config_registry.py`:
`mini_kimi_k3_tiny_plumbing` registers a tiny `Trainer.Config`, a
Trainer-compatible wrapper over the CPU-reference decoder, and a uint32 token
dataloader for one-step fake-backend plumbing checks.
`mini_kimi_k3_r1_contract` registers the source-derived r1 `Trainer.Config`.
Full-mode preflight only marks it launchable when the model-fidelity gate has a
passing reviewed launch-backend report and separate r1 Trainer smoke evidence.

Operational launch readiness reporting lives in the top-level experiment
package, `experiments/mini_kimi_k3/preflight.py`, because it owns runner and
artifact concerns rather than reusable training code.

Tokenizer installation, TorchTitan logits dumping, saved-logit forward-oracle
comparison, launch-backend review, and OOM-safe layer tracing are top-level
operational commands under `experiments/mini_kimi_k3/`. They produce
launch-gate artifacts for the r1 Trainer config.

Run-attempt evidence initialization also lives in the top-level experiment
package, `experiments/mini_kimi_k3/evidence.py`, and reuses the shared
`torchtitan.experiments.execution.lifecycle.RunAttempt` bundle layout.

The fixture-scale training smoke lives at
`experiments/mini_kimi_k3/training_smoke.py` because it is an operational
runner/report command. It imports this package's loader and tiny model to prove
token loading, forward, backward, and AdamW stepping on CPU-scale fixtures.

The r1 Trainer-path smoke producer lives at
`experiments/mini_kimi_k3/r1_training_smoke.py`. It is also operational rather
than reusable model code: it shells through `run_train.sh` with
`mini_kimi_k3_r1_contract`, fake-backend communication, one GPU rank, one
optimizer step, and the supplied token manifest, then writes the
`mini_kimi_k3_r1_training_smoke` report required by full-mode preflight.

The r1 candidate can use the optional FLA KDA backend on CUDA for forward-oracle
diagnostics, and the exact-state sampled prompt now has passing first-party
forward-oracle and launch-backend review reports. Full-run preflight still
requires r1 Trainer-path smoke evidence and a real 5B-token manifest before the
launch coordinator may start training.

Expected future modules include:

- evaluation fixtures for held-out perplexity and multiple-choice scoring;
- final attempt outcome reports from a completed full training run.

Do not import optional Mini-K3 dependencies from core TorchTitan modules. Keep
imports lazy at command boundaries when later code needs optional packages.
