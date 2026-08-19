# Experiment schema foundation

Type: task
Status: complete
Blocked by:

## Requirement

Implement the typed standard-library schema foundation described in
`.scratch/modded-nanogpt-b200/schema_matrix_spec.md`.

## Scope

Allowed:

- add `experiments/modded_nanogpt_b200/experiment_config.py`;
- add unit tests under
  `tests/unit_tests/test_modded_nanogpt_b200_experiment_config.py`;
- define `experiment_kind` as the canonical schema field;
- support legacy `lane` mapping for existing harness code;
- model observability profiles: `tier0`, `byterobust`, `mycroft`, `argus`,
  `eroica`;
- validate GPU ladder arms for `num_gpus in [1, 2, 4, 8]`;
- classify schema knobs as `supported`, `record_only`, or `rejected`;
- load and dump canonical JSON.

Excluded:

- no real GPU launches;
- no source-side `train_gpt.py` knob application;
- no matrix execution loop;
- no learned optimization policy;
- no changes to TorchTitan core.

## Verification Evidence

Run inside rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_experiment_config.py'
```

Broaden inside rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile experiments/modded_nanogpt_b200/experiment_config.py'
```
