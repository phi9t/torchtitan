# Experiment matrix runner

Type: task
Status: complete
Blocked by: 10, 11

## Requirement

Add a rootfs-aware matrix runner that reads the schematized experiment spec,
runs each arm sequentially, and writes a matrix-level report.

## Scope

Allowed:

- add `experiments/modded_nanogpt_b200/run_experiment_matrix.py`;
- add `experiments/modded_nanogpt_b200/run_experiment_matrix.sh`;
- add checked-in initial config under
  `experiments/modded_nanogpt_b200/configs/gpu_ladder_prerequisite.json`;
- run active-job scan before every arm;
- run preflight before training every arm;
- write per-arm and matrix-level structured artifacts;
- refresh `run_index.json` after matrix execution.

Excluded:

- no concurrent arm execution;
- no real matrix launch until dry/preflight tests are green;
- no learned policy or automatic retries;
- no hidden training-source overrides.

## Verification Evidence

Run inside rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_experiment_config.py tests/unit_tests/test_modded_nanogpt_b200_run_experiment_matrix.py'
```
