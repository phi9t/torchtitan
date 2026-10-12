# Hard Reasoning And Coding Continuation

Run timestamp: 2026-08-12T11:00:00Z

This report records the post-audit continuation for the harder reasoning,
coding, and external-harness lanes. It separates successful infrastructure
execution from model capability results.

## Summary

Completed in this pass:

- tau2 upstream task execution is now unblocked for the pinned mock-domain
  deterministic probe.
- GPQA Diamond was rechecked through the bwrap rootfs and remains blocked by
  Hugging Face gated-dataset auth.
- vLLM GPU availability was rechecked before launching additional hard
  reasoning/coding model runs; all eight B200s were still occupied by unrelated
  SGLang scheduler processes.

No new BigCodeBench-Hard, ARC-AGI-2, AIME, or GPQA model score is reported from
this pass.

## tau2 Execution Probe

Command:

```bash
RUN_ID=20260812T105500Z-tau2-deterministic-execution-probe \
RESULTS_ROOT=experiments/scaffold_to_policy/results/tau2_deterministic_execution_probe \
RECREATE_VENV=0 \
experiments/scaffold_to_policy/run_tau2_execution_probe.sh
```

The runner now invokes a small wrapper script that imports
`torchtitan.experiments.scaffold_to_policy.tau2_probe_agent` before calling
`tau2.cli.main`. This was needed because the tau2 virtualenv did not import the
previous `sitecustomize.py` hook at interpreter startup.

The registered components are:

- `torchtitan_mock_oracle_agent`: emits the assistant actions from the task's
  evaluation criteria through tau2 tool calls.
- `torchtitan_static_user`: sends the task scenario once, then stops.

Result artifacts:

```text
experiments/scaffold_to_policy/results/tau2_deterministic_execution_probe/raw/tau2_execution_probe.json
experiments/scaffold_to_policy/results/tau2_deterministic_execution_probe/ingested/tau2_execution_probe.json
experiments/scaffold_to_policy/results/tau2_deterministic_execution_probe/manifests/report_input_20260812T105500Z-tau2-deterministic-execution-probe.json
```

Key parsed fields:

```json
{
  "score": 1.0,
  "num_tasks": 1,
  "returncode": 0,
  "num_simulations": 1,
  "num_evaluated": 1,
  "num_infra_errors": 0,
  "errors": []
}
```

The report input checks include:

```json
{
  "all_rootfs_selected": true,
  "installed_preflight_imports_available": true,
  "installed_preflight_versions_match": true,
  "task_execution_probes_succeeded": true,
  "task_score_smokes_succeeded": true
}
```

Interpretation: this clears tau2 runner, task loading, deterministic
agent/user registration, official `results.json` writing, and scaffold
ingestion for one pinned mock task. It is not a tau2 model score.

## GPQA Diamond

Command:

```bash
RUN_ID=20260812T110000Z-gpqa-auth-blocker \
RESULTS_ROOT=experiments/scaffold_to_policy/results/gpqa_public_vllm_auth_blocker \
experiments/scaffold_to_policy/run_gpqa_public_vllm_smoke.sh
```

The run stopped before vLLM generation at the dataset import gate:

```text
Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. You must be authenticated to access it.
```

The blocker manifest is:

```text
experiments/scaffold_to_policy/results/gpqa_public_vllm_auth_blocker/manifests/report_input_20260812T110000Z-gpqa-auth-blocker.json
```

Its checks record:

```json
{
  "gpqa_import_available": false
}
```

Interpretation: the GPQA no-tool multiple-choice path remains correctly gated
on HF auth. No GPQA model result exists from this run.

## GPU State For Hard vLLM Runs

The rootfs GPU memory preflight recorded:

```json
{
  "selected": false,
  "reason": "insufficient free memory",
  "device": {
    "device_index": 0,
    "name": "NVIDIA B200",
    "free_gib": 3.93017578125,
    "required_gib": 133.76339721679688,
    "selected": false
  }
}
```

Host `nvidia-smi` showed all eight B200s occupied by unrelated SGLang
scheduler processes:

```text
GPU 0 free 4423 MiB, used 178210 MiB
GPU 1 free 4335 MiB, used 178298 MiB
GPU 2 free 4355 MiB, used 178278 MiB
GPU 3 free 4375 MiB, used 178258 MiB
GPU 4 free 4335 MiB, used 178298 MiB
GPU 5 free 4335 MiB, used 178298 MiB
GPU 6 free 4335 MiB, used 178298 MiB
GPU 7 free 183 MiB, used 182450 MiB
```

Interpretation: rerunning BigCodeBench-Hard `contract_chat`, ARC-AGI-2 larger
calibrations, AIME expansion, or any GPQA model path would currently reproduce
a resource blocker rather than add useful benchmark evidence.

## Next Steps

1. Rerun BigCodeBench-Hard `contract_chat` on the existing 8 dev / 8 OOD
   preflight-clean slice once at least one B200 is free enough for vLLM.
2. Rerun GPQA only after `HF_TOKEN` is available inside the bwrap rootfs.
3. Replace the tau2 deterministic oracle behavior with a bounded non-oracle
   model or scaffold-policy agent while preserving tau2's runner and official
   evaluator.
4. Keep Terminal-Bench/Harbor on the same pattern: oracle infrastructure is
   green, but the next result must use a bounded non-oracle agent if it is to
   say anything about policy capability.
