# MMLU-Pro 8x8 Four-Rollout Calibration

Run ID: `20260812Tmmlu-pro-8x8-rollouts4-promptfix`

This run adds a harder ten-choice multiple-choice reasoning lane using
`TIGER-Lab/MMLU-Pro`. It preserves the no-tool condition and scores only exact
`FINAL: X` letter answers, where `X` is one of the released answer choices.

## Command

```bash
RUN_ID=20260812Tmmlu-pro-8x8-rollouts4-promptfix \
DATA_ROOT=experiments/scaffold_to_policy/data/mmlu_pro_public_vllm_8x8_rollouts4 \
RESULTS_ROOT=experiments/scaffold_to_policy/results/mmlu_pro_public_vllm_8x8_rollouts4_promptfix \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=4 MAX_NEW_TOKENS=1024 \
GPU_MEMORY_UTILIZATION=0.05 \
experiments/scaffold_to_policy/run_mmlu_pro_public_vllm_smoke.sh
```

The run used local Qwen3-1.7B through vLLM inside the bwrap rootfs with the
shared stage manifest and GPU-memory preflight path.

## Results

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 32 | 0.250 | 0.250 | 0.250 | easy 2, elicitable 0, unreached 6 |
| OOD | 8 | 32 | 0.750 | 0.750 | 0.750 | easy 6, elicitable 0, unreached 2 |

Strict-format pass@k matched answer pass@k on both splits. The prompt-fixed run
had no missing-final failures; all failures were verifier-readable wrong
letters. This is a calibration slice result, not an official MMLU-Pro
leaderboard score.

## Failure Modes

| Split | Successes | Wrong final letters | Missing final answers |
| --- | ---: | ---: | ---: |
| dev | 8 | 24 | 0 |
| OOD | 24 | 8 | 0 |

Representative dev failure:

```text
Problem: MMLU-Pro/0
Expected: A
Observed: FINAL: G
Behavior: the model treated 2Z like a characteristic-2 quotient and selected
choice G, while the released answer was choice A.
```

Representative OOD failure:

```text
Failure class: final value C/F/H does not match the released answer.
Behavior: the verifier accepted the final-letter format and rejected only the
semantic choice.
```

## Interpretation

The ten-choice import, prompt, verifier, and report path now work for
MMLU-Pro-style rows. The initial pre-fix run showed the model often copied
angle-bracket examples such as `FINAL: <C>`; the prompt repair changed the
contract to `FINAL: X` and removed missing-final failures in this rerun. On
this small slice, the OOD offset is easier than the first dev rows, so these
numbers should be used for infrastructure calibration and example inspection,
not broad capability ranking.
