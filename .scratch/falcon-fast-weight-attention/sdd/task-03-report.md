# Ticket 03 report — Step 1 remainder: alignment switch + harness contract

Status: **DONE**

Claim label: `deterministic_pairing` + `learnability_smoke`.

## Summary

Added an explicit `alignment` switch to the Falcon recurrent kernel so
same-step writes are an ablation, never the default. `delayed` (paper) stays
the default at the kernel, `FalconConfig`, and `FalconMixer` levels. Wrote
failing tests first (TDD), then implemented, then confirmed the Step 3 overfit
gate stays green.

## Files changed

- `torchtitan/experiments/falcon/falcon.py`
  - New `FalconAlignment = Literal["delayed", "same_step"]`.
  - `falcon_recurrent_forward(..., alignment="delayed")` keyword-only arg.
  - `delayed`: `t=0` no-op (`x=0`, `eta=0`); later `x = phi(k_{t-1})`, target
    `v_t`.
  - `same_step`: `x = phi(k_t)`, target `v_t` at every `t` including `0`, still
    read-after-write, `eta` computed normally at `t=0`.
  - Rejects unknown `alignment` with `ValueError`.
- `torchtitan/experiments/falcon/model.py`
  - `FalconConfig.alignment: FalconAlignment = "delayed"`.
  - `tiny_falcon_config(*, variant=..., alignment="delayed")`.
  - `FalconMixer.forward` threads `alignment=self.config.alignment` to kernel.
- `torchtitan/experiments/falcon/HARNESS.md` (new)
  - Silent-failure modes: same-step default, circular labels (roll vs shift),
    missing document-boundary state reset (future). Plus held Step 1
    invariants.
- `tests/unit_tests/test_falcon_kernel.py`
  - `test_delayed_alignment_first_write_uses_k0_with_v1`
  - `test_same_step_alignment_first_write_uses_k0_with_v0`
  - `test_falcon_recurrent_forward_rejects_unknown_alignment`
- `tests/unit_tests/test_falcon_model.py`
  - `test_repeat_dataloader_labels_are_a_shift_not_a_circular_roll`
  - `test_falcon_config_alignment_defaults_to_delayed`
  - `test_falcon_mixer_threads_same_step_alignment_to_the_kernel`

## Test command + output

```
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_falcon_*.py'
```

- RED (before implementation): 5 failed, 12 passed — the 3 new kernel tests, 2
  new model tests.
- GREEN (after implementation): **17 passed, 14 warnings** across
  `test_falcon_config_registry.py`, `test_falcon_kernel.py`,
  `test_falcon_model.py`, `test_falcon_overfit.py`.
- Overfit gate specifically: `test_falcon_overfits_the_fixed_sequence_bank`
  for both `falcon1a` and `falcon1` PASSED; `lr=0` gate still FAILs the model
  (test asserts `status == "fail"`) as required. Delayed default unchanged.

## Notes / concerns

- Document-boundary state reset is explicitly deferred (out of this ticket's
  scope, per spec); HARNESS.md records it as a known future requirement rather
  than a current bug. The tiny overfit bank is one document per row, so no
  reset is needed yet.
- Warnings are a preexisting `torch.jit.script_method` deprecation, unrelated
  to this change.

Not committed, per instructions. No push, no PR. No edits to Mini-K3,
`qwen3_5`, addition, FineWeb, or chunk-parallel paths. Overfit defaults left on
`delayed`.
