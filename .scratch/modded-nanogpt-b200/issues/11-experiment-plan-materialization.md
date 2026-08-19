# Experiment plan materialization

Type: task
Status: complete
Blocked by: 10

## Requirement

Materialize a validated experiment schema arm into the concrete launch and
preflight values used by the existing harness.

## Scope

Allowed:

- extend `experiment_config.py` or add
  `experiments/modded_nanogpt_b200/experiment_plan.py`;
- derive `run_id`, `attempt_id`, result directory, `legacy_lane`,
  `claim_label`, `claim_eligible`, visible GPU env, `torchrun` world size, and
  preflight `--expected-gpus`;
- add `--num-gpus` and `--gpu-ids` to `run_speedrun.py`;
- update `preflight.py` full-mode policy so full matrix arms may use 1/2/4/8
  GPUs while remaining non-claimable unless claim policy requires 8 GPUs;
- preserve existing two-GPU launcher behavior.

Excluded:

- no matrix executor yet;
- no source-side training schedule overrides;
- no parallel matrix scheduling;
- no claim promotion for non-8-GPU prerequisite arms.

## Verification Evidence

Run inside rootfs:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_experiment_config.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py'
```
