# Ticket 04 report - Step 2: systems / view identity

Status: **DONE**

Claim labels: `deterministic_comparison` + `smoke`.

## Summary

Added `falcon_masked_parallel_forward`, a masked-parallel view of the Step 1
recurrent Falcon oracle with the *same signature* (including the ticket-03
`alignment` switch). Wrote failing identity tests first (TDD), implemented the
parallel kernel, then confirmed recurrent == masked-parallel bitwise-close in
fp32 across `{falcon1, falcon1a} x {delayed, same_step}` on tiny random and
worked-example tensors. Produced a debug Trainer artifact: `falcon_tiny_overfit`
runs N>1 steps under the fake backend with finite loss.

The identity gate (recurrent == masked-parallel) is the substitute for Mercor's
train-inference logprob-diff < 0.03: there is no separate train-inference stack
for this mixer, so proving the two mathematically-equivalent views agree is the
correctness contract for Step 2.

## Files changed

- `torchtitan/experiments/falcon/falcon.py`
  - New `_falcon_write_terms(...)` helper: precomputes the per-step write
    feature `x_t` (`phi(k_{t-1})` for `delayed` with the `t=0` no-op, `phi(k_t)`
    for `same_step`), the NLMS step size `eta_t`, the scalar carry
    `gamma_t = 1 - min(eta_t lambda_t, 1 - eps_gamma)`, and the Hebbian target
    `v_t`. The `delayed` `t=0` no-op keeps `eta_0 = 0`, `gamma_0 = 1`.
  - New `falcon_masked_parallel_forward(...)`: identical signature and tolerance
    contract to `falcon_recurrent_forward`. Unrolls the recurrence
    `S_t = gamma_t S_{t-1} + eta_t x_t w_t^T` using the cumulative log-carry
    `c_t = sum_{r<=t} log gamma_r`, so the decay from write `s` to read `t` is
    `exp(c_t - c_s)`. Reads are masked with a causal mask that *includes* the
    diagonal (read-after-write). Final state `S_L` is the same decay-weighted
    accumulation in the last-step frame.
    - Falcon-1A: write target `w_s = v_s` is known up front, fully parallel.
    - Falcon-1: the residual write `w_s = v_s - S_{s-1}^T x_s` depends on the
      pre-write state, so `w_s` is recovered by one forward pass over `L`
      matching the recurrent oracle's predict-then-carry-then-fold order; the
      read/state assembly is still the masked-matmul parallel view. Documented
      in the docstring as a faithful parallel-view oracle, not a production
      chunk-parallel kernel (chunk-parallel/WY/FLA are explicitly excluded from
      this ticket).
  - Rejects unknown `variant` / `alignment` with `ValueError`, matching the
    recurrent oracle.
- `tests/unit_tests/test_falcon_kernel.py`
  - Import `falcon_masked_parallel_forward`.
  - `test_masked_parallel_matches_recurrent_on_random_inputs`
    (parametrized `variant x alignment x normalize_qk`, fp32,
    rtol/atol 1e-5) - the core identity gate.
  - `test_masked_parallel_matches_recurrent_on_worked_example`
    (parametrized `variant x alignment`, the L=3 orthonormal-key example,
    rtol/atol 1e-6).
  - `test_masked_parallel_rejects_unknown_variant`,
    `test_masked_parallel_rejects_unknown_alignment`.
- `experiments/falcon/smoke_trainer.py` (new)
  - Few-step Trainer smoke: builds `falcon_tiny_overfit` via `ConfigManager`
    validation, runs N>1 optimizer steps under `comm.mode=fake_backend`, and
    asserts a finite loss at every step. Needed because
    `run_train.sh COMM_MODE=fake_backend` hard-clamps `--training.steps 1`, so a
    multi-step artifact requires driving the registry config directly. Re-enters
    nothing new; run it under `scripts/rootfs/enter_rootfs.sh`.

## Test command + output

```
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_kernel.py'
```

- RED (before implementing `falcon_masked_parallel_forward`): collection
  `ImportError: cannot import name 'falcon_masked_parallel_forward'`.
- After implementation, first pass: `4 failed, 16 passed` - the four Falcon-1
  random-input identity cases failed because the initial residual-write
  reconstruction folded the carry before the prediction. Fixed the
  Falcon-1 pass to match the recurrent oracle order (predict against the
  pre-carry state, then carry, then fold the new write).
- GREEN (kernel only): `20 passed`.
- GREEN (full Falcon suite,
  `pytest -q tests/unit_tests/test_falcon_*.py`): **31 passed, 14 warnings**
  (`test_falcon_config_registry.py`, `test_falcon_kernel.py`,
  `test_falcon_model.py`, `test_falcon_overfit.py`). The Step 1 kernel tests and
  the Step 3 overfit gate stay green; the `alignment` default is unchanged
  (`delayed`).

### Debug Trainer artifact

```
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && TORCHTITAN_IN_ROOTFS=1 NGPU=1 python -m experiments.falcon.smoke_trainer --steps 5'
```

Output (also saved to `experiments/falcon/results/task04_smoke_trainer.log`,
gitignored):

```
[falcon-smoke] OK steps=5 initial_loss=3.46574 final_loss=3.34011 all_finite=True
```

`falcon_tiny_overfit` builds through the core `Trainer`, runs 5 optimizer steps,
and logs a finite loss at every step. A single-step fake-backend
`run_train.sh MODULE=falcon CONFIG=falcon_tiny_overfit NGPU=1
COMM_MODE=fake_backend` was also run and printed `step: 1 loss: 3.46574` with a
finite grad norm, confirming the standard launch path still works; it is clamped
to one step by `run_train.sh`, hence the multi-step driver for the N>1 gate.

### Lint

```
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && SKIP=no-commit-to-branch pre-commit run --files torchtitan/experiments/falcon/falcon.py tests/unit_tests/test_falcon_kernel.py experiments/falcon/smoke_trainer.py'
```

All hooks pass (flake8, ufmt, pydoclint, codespell, pyrefly). `no-commit-to-branch`
is skipped only for the lint invocation; nothing is committed.

## Claim classification

- `deterministic_comparison`: recurrent == masked-parallel, fp32,
  documented tolerance rtol/atol 1e-5 (random) / 1e-6 (worked example), across
  both variants and both alignments. This is the Step 2 correctness proof and
  the stand-in for Mercor's logprob-diff gate.
- `smoke`: the `falcon_tiny_overfit` few-step Trainer run proves plumbing
  (config -> model -> optimizer -> finite loss over N>1 steps). It is not
  evidence of convergence, throughput, or reward.

## Notes / scope

- Excluded per the ticket: chunk-parallel / WY / FLA acceleration, context
  parallel, compile/FSDP-required pass, Step 4 data/ablations. The Falcon-1
  residual solve is an O(L) parallel-view oracle, not a chunk-parallel kernel;
  acceleration is a later ticket, not a science blocker.
- No edits to Mini-K3, `qwen3_5`, FineWeb, or the addition diagnostic.
- The Step 3 overfit learnability gate and the ticket-03 `alignment` default
  (`delayed`) are unchanged and still green.

Not committed, per instructions. No push, no PR.
