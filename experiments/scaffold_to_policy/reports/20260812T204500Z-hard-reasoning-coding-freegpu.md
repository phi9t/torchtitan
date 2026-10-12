# Hard Reasoning And Coding Free-GPU Continuation

Run IDs:

- `20260812T203000Z-mmlu-pro-16x16-rollouts4-ood32`
- `20260812T200000Z-bigcodebench-hard-contract-chat-8x8-freegpu`

This continuation uses the free B200 window to move the harder public reasoning
and coding lanes past prior resource blockers. Both runs used Qwen3-1.7B with
vLLM inside the bwrap rootfs. They are calibration and hard-negative runs, not
leaderboard claims.

## Commands

MMLU-Pro:

```bash
RUN_ID=20260812T203000Z-mmlu-pro-16x16-rollouts4-ood32 \
DATA_ROOT=experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_16x16_rollouts4_ood32 \
RESULTS_ROOT=experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_rollouts4_ood32 \
DEV_PROBLEMS=16 OOD_PROBLEMS=16 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_mmlu_pro_public_vllm_smoke.sh
```

BigCodeBench-Hard:

```bash
RUN_ID=20260812T200000Z-bigcodebench-hard-contract-chat-8x8-freegpu \
DATA_ROOT=experiments/scaffold_to_policy/data/bigcodebench_hard_contract_chat_8x8_freegpu \
RESULTS_ROOT=experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_freegpu \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=8 PROMPT_VARIANT=contract_chat \
GPU_MEMORY_UTILIZATION=0.05 TIMEOUT_SECONDS=30 MAX_NEW_TOKENS=1024 \
experiments/scaffold_to_policy/run_bigcodebench_hard_public_vllm_smoke.sh
```

## Environment And Provenance

- Execution boundary: `scripts/rootfs/enter_rootfs.sh`.
- Model: `./assets/hf/Qwen3-1.7B`.
- Runtime: vLLM inside the bwrap rootfs, Triton attention, FlashInfer sampler
  disabled by the shared scaffold environment.
- GPU preflight: cuda:0 on NVIDIA B200 had 177.736 GiB free; at
  `GPU_MEMORY_UTILIZATION=0.05`, required memory was 8.918 GiB.
- MMLU-Pro dataset: `TIGER-Lab/MMLU-Pro`, validation split, revision `main`.
- BigCodeBench-Hard dataset: `bigcode/bigcodebench-hard`, split `v0.1.4`,
  revision `main`.
- MMLU-Pro verifier: exact final letter, `FINAL: X`, over released choices.
- BigCodeBench-Hard verifier: released task tests executed in the
  repo-owned isolated Python subprocess, after released canonical solutions
  passed preflight on the selected split.

Report inputs:

```text
experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_16x16_rollouts4_ood32/manifests/report_input_20260812T203000Z-mmlu-pro-16x16-rollouts4-ood32.json
experiments/scaffold_to_policy/results/bigcodebench_hard_contract_chat_8x8_freegpu/manifests/report_input_20260812T200000Z-bigcodebench-hard-contract-chat-8x8-freegpu.json
```

## MMLU-Pro Results

The selected split registry was clean: 16 dev problems, 16 OOD problems, and no
overlap. The first attempted larger run used OOD offset 64 and found only six
remaining validation rows; this report uses the corrected offset 32 run.

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 16 | 64 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | easy 8, elicitable 0, unreached 8 |
| OOD | 16 | 64 | 0.5625 | 0.6250 | 0.6875 | 0.6875 | easy 9, elicitable 2, unreached 5 |

Strict-format pass@k matched answer pass@k. There was one missing-final failure
across the 128 sampled rollouts; the remaining failures were verifier-readable
wrong final letters.

Representative easy dev problem:

```text
Problem: MMLU-Pro/4
Question: A total of 30 players will play basketball at a park. There will be
exactly 5 players on each team. Which statement correctly explains how to find
the number of teams needed?
Expected: B
Observed: FINAL: B
Behavior: all rollouts solved the division problem and emitted the strict final
letter.
```

Representative unreached dev problem:

```text
Problem: MMLU-Pro/0
Expected: A
Observed: FINAL: G
Behavior: the model reasoned that the characteristic of 2Z was 2, then selected
the answer choice whose text was "2". The released answer is A because the
problem treats 2Z as a ring without identity, whose characteristic is 0.
```

Representative elicitable OOD problem:

```text
Problem: MMLU-Pro/42
Expected: C
First rollout: FINAL: A
Behavior: at least one of the four rollouts reached the released answer, but
the first sample chose a distractor while reasoning over invalid SR-latch input
states.
```

## BigCodeBench-Hard Results

This reruns the previously blocked `contract_chat` hard-coding condition while
GPUs were free. Both selected canonical preflights passed 8/8, so the verifier
and rootfs package environment were clean before model scoring.

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@4 | pass@8 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |
| OOD | 8 | 64 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Failure breakdown:

| Split | Assertion failures | Syntax errors |
| --- | ---: | ---: |
| dev | 64 | 0 |
| OOD | 63 | 1 |

Representative dev failure:

```text
Problem: BigCodeBench/13
Task shape: FTP workflow helper with exact error strings and side effects.
Failure: assertion failure
Behavior: the generated implementation made plausible FTP calls but did not
match the released error-string contract, including the required directory and
server details.
```

Representative OOD failure:

```text
Problem: BigCodeBench/267
Task shape: FFT and plotting helper over structured data.
Failure: assertion failure
Behavior: the generated implementation called `fftpack.fft(data)` directly even
when the released tests supplied structured dictionaries, causing scipy to
raise a type error before the expected artifact contract was met.
```

## Interpretation

The MMLU-Pro expansion is useful positive reasoning evidence at the current
scale: the larger 16/16 slice preserved the no-tool condition, exact final
letter verifier, and strict-format reporting, and it found two OOD elicitable
problems beyond the easy bucket.

The BigCodeBench-Hard rerun is a clean hard-negative coding result. The earlier
resource blocker is gone, canonical solutions pass, and model execution
completes, but the current Qwen3-1.7B no-tool `contract_chat` condition still
does not solve any selected BigCodeBench-Hard task at eight samples per
problem.

The next useful expansion is not another prompt-only BigCodeBench rerun on the
same slice. Better follow-ups are:

- add a scaffolded code-repair condition that can execute public tests while
  reporting it separately from no-tool generation;
- expand MMLU-Pro across more offsets or stratify by subject/category before
  training any adapter;
- run GPQA only after authenticated Hugging Face access or authorized raw rows
  are present in the rootfs-visible environment.
