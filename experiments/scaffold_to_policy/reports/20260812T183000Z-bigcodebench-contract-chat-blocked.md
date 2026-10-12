# BigCodeBench-Hard Contract-Chat Blocked Attempt

Run ID: `20260812T183000Z-bigcodebench-hard-contract-chat-blocked`

This attempt adds a labeled `contract_chat` prompt condition for the same
preflight-clean BigCodeBench-Hard 16-task calibration slice used by the previous
expanded run. It is not a model result because vLLM generation did not start.

## Command

```bash
RUN_ID=20260812T183000Z-bigcodebench-hard-contract-chat-blocked \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_public_vllm_expanded_clean \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_blocked \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=72 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 TIMEOUT_SECONDS=60 \
PROMPT_VARIANT=contract_chat \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

## What Changed

`contract_chat` keeps the no-tool condition and released executable verifier,
but makes the system prompt more explicit about preserving the requested
function name and signature, using prompt-provided imports/globals, and matching
side effects, return types, error behavior, and library calls. It does not
expose tests or verifier feedback to the model.

The BigCodeBench runner also now runs `preflight-vllm-gpu-memory` after
canonical-solution preflight and before vLLM initialization. This catches GPU
contention as a cheap JSON preflight instead of failing inside vLLM startup.

## Preflight Evidence

Canonical preflight passed for the same selected slice:

| Split | Problems | Canonical pass | Failure breakdown |
| --- | ---: | ---: | --- |
| dev | 8 | 8/8 | success 8 |
| OOD | 8 | 8/8 | success 8 |

GPU memory preflight then blocked generation:

```text
selected=false
reason=insufficient free memory
device=NVIDIA B200 cuda:0
free_gib=4.729
required_gib=160.516
gpu_memory_utilization=0.9
```

At the time of inspection all eight B200s were occupied by unrelated
`sglang::scheduler` processes using roughly 176-177 GiB each. The earlier
direct `contract_chat` run without the new preflight failed in vLLM startup with
the same underlying condition: CUDA device 0 had only about 6.92 GiB free.

## Status

The prompt condition and preflight wiring are implemented, but the
BigCodeBench-Hard `contract_chat` comparison still needs to be rerun when a B200
has enough free memory. Use the command above, or the same command with a fresh
run ID and results root. Do not compare this blocked attempt against the
previous `chat` result as a model capability measurement.
