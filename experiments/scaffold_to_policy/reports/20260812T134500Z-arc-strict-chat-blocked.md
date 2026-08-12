# ARC-AGI-2 Strict-Chat Prompt Attempt

Run ID: `20260812T134500Z-arc-agi2-strict-chat-calibration`

This attempt adds a stricter no-tool ARC prompt condition for the same 8 dev
and 8 OOD task slice used by the prior ARC-AGI-2 calibration. It did not
produce model evaluations because the host GPUs were already occupied before
vLLM could initialize.

## Command

```bash
RUN_ID=20260812T134500Z-arc-agi2-strict-chat-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/arc_agi2_public_vllm_calibration_strict_chat \
RESULTS_ROOT=experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.24 \
SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=8192 \
MAX_MODEL_LEN=8192 \
PROMPT_VARIANT=strict_chat \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

## What Changed

The new `strict_chat` prompt variant preserves the same exact verifier and
same no-tool condition. It changes only the prompt contract:

```text
Reply with exactly one line: FINAL: <json-grid>
Do not include analysis, markdown, labels, code fences, or extra lines.
```

This is intended to test whether the previous ARC failures were dominated by
format compliance rather than grid reasoning. It is a prompt condition, not a
scoring change.

## Preflight Results

Prompt/context preflight passed:

| Split | Problems selected | Max total tokens |
| --- | ---: | ---: |
| dev | 8/8 | 5,260 |
| OOD | 8/8 | 7,516 |

The generated preflight artifacts are:

```text
experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat/eval/dev_prompt_preflight.json
experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat/eval/ood_test_prompt_preflight.json
```

Generation did not run. vLLM failed on startup because the selected CUDA device
had too little free memory:

```text
Free memory on device cuda:0 (1.53/178.35 GiB) on startup is less than desired
GPU memory utilization (0.24, 42.8 GiB).
```

After the failure, the ARC entrypoint was hardened with a GPU-memory preflight
that writes a JSON blocker artifact before vLLM initialization. On the current
host state it recorded:

```json
{
  "selected": false,
  "reason": "insufficient free memory",
  "gpu_memory_utilization": 0.24,
  "devices": [
    {
      "device_index": 0,
      "free_gib": 6.50067138671875,
      "required_gib": 42.80428710859269,
      "selected": false
    }
  ]
}
```

The memory preflight artifact is:

```text
experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat/eval/vllm_gpu_memory_preflight_blocked.json
```

## Interpretation

The strict prompt condition is implemented and prompt-length feasible for the
same ARC slice, but the comparison remains unrun. The next run should reuse the
same command after a GPU has at least the requested vLLM free-memory budget, or
explicitly choose a visible device with sufficient free memory.
