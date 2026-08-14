# F4 public-vLLM lifecycle migration: math + mbpp (on-device evidence)

Date: 2026-08-14
Host: 8x NVIDIA B200, bwrap rootfs, `assets/hf/Qwen3-1.7B` present.

## Scope

Migrated the two remaining "simple family" public-dataset vLLM smoke runners off
the legacy flat `python -m ... cli` sequence onto the typed begin/stage/finish
lifecycle (F1) plus the composed `vllm_1gpu`+`reasoning` profile doctor (F2),
matching the gsm8k (reasoning) and humaneval (coding) reference runners already
landed:

- `experiments/scaffold_to_policy/run_math_public_vllm_smoke.sh` (reasoning,
  math-style CLI subcommands: import-math-split, validate-math-style-splits,
  evaluate-math-style-vllm, build-math-style-report-input).
- `experiments/scaffold_to_policy/run_mbpp_public_vllm_smoke.sh` (coding,
  coding-style CLI subcommands with `--timeout-seconds` sandbox: import-mbpp-split,
  validate-coding-style-splits, evaluate-coding-style-vllm,
  build-coding-style-report-input).

Both stay thin compatibility entrypoints: dataset revisions, offsets, sampling,
rollout budget, and sandbox timeout defaults are unchanged. The `--fields`
declaration and `evaluations.json` are built from Python off the knob and split
lists so a knob change re-digests the declaration and a new split cannot silently
drop a condition. Base-only, no trained arm and no calibration gate, so every
condition is `measurement=real`, `promotion=not_evaluated`; run gate
`has_real_measurement=true`.

## Stage adapter/kind map (unchanged from gsm8k/humaneval)

- import-dev / import-ood: `acquire` / `rootfs_cpu`
- validate-splits: `verify` / `rootfs_cpu`
- base-eval-<split>: `evaluate` / `rootfs_vllm` (mbpp: vLLM generation + in-process
  sandboxed code exec; dominant runtime is rootfs vLLM, so `rootfs_vllm` is the
  honest single label, matching humaneval)
- build-report-input: `report` / `rootfs_cpu`

## math verification

Run id `20260814T020000Z-math-migration`, default knobs (dev/ood limit 8,
ood offset 256, 4 rollouts, max_new_tokens 512).

- 6 stages, all rc=0: import-dev, import-ood, validate-splits, base-eval-dev,
  base-eval-ood_test, build-report-input.
- preflight: both profiles `readiness=ready`, empty `blocker_codes`; clauses
  pass for rootfs_active, torch, vllm, datasets, transformers, gpu_count>=1
  (gpu_count=8).
- outcome: `execution_outcome=completed`; conditions dev/ood_test both
  `measurement=real`, `promotion=not_evaluated`; run gate
  `has_real_measurement=true`, `measurement_counts={real:2}`.

## mbpp verification

Run id `20260814T021500Z-mbpp-migration`. Ran with disjoint offsets
`DEV_OFFSET=0 OOD_OFFSET=8` (dev/ood limit 3) to avoid a pre-existing importer
bug (below); all lifecycle wiring exercised identically to the default path.

- 6 stages, all rc=0.
- outcome: `execution_outcome=completed`; conditions dev/ood_test both
  `measurement=real`; run gate `has_real_measurement=true`,
  `measurement_counts={real:2}`, `promotion_counts={not_evaluated:2}`.

The migration's fail-closed behavior was also observed directly: an earlier mbpp
run with the default `OOD_OFFSET=64` recorded `stage_failed` (rc=1) on the
`import-ood` stage and aborted before `finish`, leaving no `outcome.json` -- the
lifecycle never emitted a completed attempt for a broken run.

## Pre-existing MBPP importer bug (out of scope, flagged)

Independent of this shell migration, the MBPP importer cannot parse certain
sanitized-split asserts. At default `OOD_OFFSET=64`, `import-mbpp-split` raises:

```
ValueError: could not infer MBPP entry point from test:
  assert math.isclose(angle_complex(0,1j), 1.5707963267948966, rel_tol=0.001)
```

from `torchtitan/experiments/scaffold_to_policy/coding_style.py:632`
(`_entry_point_from_asserts`): the asserted call `angle_complex(...)` is nested
inside `math.isclose(...)`, and the current regex-based inference only looks for
a top-level `assert <name>(`. This blocks the default mbpp OOD offset and should
be fixed in the coding-style importer (parse the AST of the first assert and take
the called function, rather than a leading-token regex). Tracked as a follow-up;
it does not affect the lifecycle-migration contract this report verifies.

## Not migrated in this wave

The six instrumented public-vLLM runners (mmlu_pro, aime, livecodebench,
arc_agi2, gpqa, bigcodebench_hard) already carry a prototype `scaffold_run_stage`
helper plus `write-blocker-report-input` / `preflight-vllm-gpu-memory` runtime
fallbacks that finish gracefully as `blocked` on dataset-access, GPU-OOM, or
prompt-fit failure. Migrating those onto the typed lifecycle needs a
`finish --attempt-outcome blocked` path with per-condition `execution_outcome=
blocked` translation, which is separate design work from the simple family; it is
deferred to a dedicated follow-up.
