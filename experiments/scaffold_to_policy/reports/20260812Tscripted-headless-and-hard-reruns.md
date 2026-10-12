# Scripted Harbor Agent and Hard Benchmark Reruns

This report records three rootfs-managed continuations after the earlier
hard-lane blockers:

- a bounded custom-agent Harbor smoke for Terminal-Bench `headless-terminal`;
- a completed BigCodeBench-Hard `contract_chat` 8x8 coding rerun;
- a completed AIME 8x8 hard-reasoning rerun.

The Harbor run is a task-specific scripted policy smoke. The AIME and
BigCodeBench-Hard runs are Qwen3-1.7B/vLLM model evaluations over the selected
small slices, not full benchmark claims.

## Terminal-Bench / Harbor

Run ID: `20260812Tscripted-headless-terminal-smoke`

Command:

```bash
RUN_ID=20260812Tscripted-headless-terminal-smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/terminal_bench_harbor_scripted_headless \
HARBOR_AGENT=torchtitan.experiments.scaffold_to_policy.harbor_headless_terminal_agent:HeadlessTerminalScriptAgent \
TIMEOUT_SECONDS=900 \
experiments/scaffold_to_policy/run_terminal_bench_oracle_probe.sh
```

The agent subclasses Harbor's `BaseAgent`, runs inside Harbor's released Docker
task environment, writes `/app/headless_terminal.py`, and lets Harbor's verifier
score the final task state. It does not call Harbor's oracle solution at
runtime.

Harbor job summary:

| Field | Value |
| --- | ---: |
| `n_total_trials` | 1 |
| `stats.n_completed_trials` | 1 |
| `stats.n_errored_trials` | 0 |
| `stats.evals.torchtitan-headless-terminal-script__adhoc.n_trials` | 1 |
| `stats.evals.torchtitan-headless-terminal-script__adhoc.n_errors` | 0 |
| `stats.evals.torchtitan-headless-terminal-script__adhoc.metrics[0].mean` | 1.0 |

Interpretation: this clears the custom-agent Harbor execution path for one
pinned task. It is not a learned-policy result, a Qwen3 result, or evidence of
general Terminal-Bench capability.

Artifacts:

```text
experiments/scaffold_to_policy/results/terminal_bench_harbor_scripted_headless/manifests/report_input_20260812Tscripted-headless-terminal-smoke.json
experiments/scaffold_to_policy/results/terminal_bench_harbor_scripted_headless/runs/20260812Tscripted-headless-terminal-smoke/result.json
```

## BigCodeBench-Hard

Run ID: `20260812Tbigcode-hard-contract-chat-8x8-timeout30-rerun`

Command:

```bash
RUN_ID=20260812Tbigcode-hard-contract-chat-8x8-timeout30-rerun \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_timeout30_rerun \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_contract_chat_8x8_timeout30_rerun \
DEV_PROBLEMS=8 \
OOD_PROBLEMS=8 \
DEV_OFFSET=0 \
OOD_OFFSET=32 \
NUM_ROLLOUTS=8 \
PROMPT_VARIANT=contract_chat \
GPU_MEMORY_UTILIZATION=0.05 \
TIMEOUT_SECONDS=30 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

The 30 second timeout was needed because `BigCodeBench/17` timed out in the
canonical preflight with the 10 second default. With the larger released-test
budget, canonical preflights passed for both selected splits. The GPU preflight
selected GPU 0 with 177.74 GiB free versus 8.92 GiB required at
`GPU_MEMORY_UTILIZATION=0.05`.

Result:

| Split | Problems | Rollouts | pass@1 | pass@8 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 64 | 0.000 | 0.000 | 0.000 | 8 unreached |
| ood_test | 8 | 64 | 0.000 | 0.000 | 0.000 | 8 unreached |

Failure breakdown:

| Split | Failures |
| --- | --- |
| dev | 64 assertion failures |
| ood_test | 63 assertion failures, 1 syntax error |

Representative failures:

- `BigCodeBench/13`: generated an FTP helper that looked structurally plausible
  but failed released tests on exact error-message behavior for invalid
  directory handling.
- `BigCodeBench/267`: generated FFT/plotting code for the wrong input contract;
  released tests passed dictionaries and the implementation raised
  `TypeError: float() argument must be a string or a real number, not 'dict'`.

Interpretation: the shared-engine coding infrastructure and dependency set are
now sufficient for this 8x8 slice. The model result is a hard negative: the
prompt produced plausible code but not verifier-correct implementations.

Artifacts:

```text
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_timeout30_rerun/manifests/report_input_20260812Tbigcode-hard-contract-chat-8x8-timeout30-rerun.json
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_timeout30_rerun/eval/dev_summary.json
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_timeout30_rerun/eval/ood_test_summary.json
```

## AIME

Run ID: `20260812Taime-8x8-lowmem-rerun`

Command:

```bash
RUN_ID=20260812Taime-8x8-lowmem-rerun \
RESULTS_ROOT=experiments/scaffold_to_policy/results/aime_public_vllm_8x8_lowmem_rerun \
DATA_ROOT=experiments/scaffold_to_policy/data/aime_public_vllm_8x8_lowmem_rerun \
DEV_PROBLEMS=8 \
OOD_PROBLEMS=8 \
NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh
```

The run completed both vLLM split launches through the bwrap rootfs. GPU memory
preflight selected GPU 0 with 177.74 GiB free versus 8.92 GiB required.

Result:

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 32 | 0.125 | 0.125 | 0.125 | 1 easy, 7 unreached |
| ood_test | 8 | 32 | 0.000 | 0.000 | 0.000 | 8 unreached |

Failure breakdown:

| Split | Failures |
| --- | --- |
| dev | 4 success, 25 missing final answer, 3 wrong extracted final values |
| ood_test | 18 missing final answer, 14 wrong extracted final values |

Representative outputs:

- `AIME/67` was solved on the first rollout and moved the dev bucket to `easy`.
- `AIME/60` produced long algebraic reasoning but often failed to emit the
  required final answer; another rollout emitted the literal placeholder
  `<answer>`, which the exact verifier rejected against answer `204`.
- `AIME/76` OOD produced a plausible geometry derivation but either omitted the
  final answer or emitted wrong values such as `517` against answer `468`.

Interpretation: this is a small hard-reasoning data point. The model can solve
one selected dev item, but OOD remains fully unreached and strict final-answer
formatting is still a large failure component.

Artifacts:

```text
experiments/scaffold_to_policy/results/aime_public_vllm_8x8_lowmem_rerun/manifests/report_input_20260812Taime-8x8-lowmem-rerun.json
experiments/scaffold_to_policy/results/aime_public_vllm_8x8_lowmem_rerun/eval/dev_summary.json
experiments/scaffold_to_policy/results/aime_public_vllm_8x8_lowmem_rerun/eval/ood_test_summary.json
```

## Remaining Blockers

- GPQA Diamond still cannot run because no `HF_TOKEN` is configured in the
  rootfs and the dataset is gated.
- Terminal-Bench and tau2 still lack a learned or model policy agent. The new
  Harbor result only proves a custom scripted policy can execute and be scored.
- BigCodeBench-Hard needs prompt or policy changes before the current
  `contract_chat` condition is useful as a positive training signal.
