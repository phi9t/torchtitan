# AIME 8x8 Eight-Rollout Calibration

Run ID: `20260812Taime-8x8-rollouts8-lowmem`

This run expands the earlier AIME hard-reasoning calibration from 4 rollouts to
8 rollouts on the same 8 dev / 8 OOD public AIME 2024 slice. It preserves the
no-tool condition and exact final-integer verifier.

## Command

```bash
RUN_ID=20260812Taime-8x8-rollouts8-lowmem \
DATA_ROOT=experiments/scaffold_to_policy/data/aime_public_vllm_8x8_rollouts8 \
RESULTS_ROOT=experiments/scaffold_to_policy/results/aime_public_vllm_8x8_rollouts8 \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=8 \
GPU_MEMORY_UTILIZATION=0.05 \
MAX_NEW_TOKENS=1024 \
PROMPT_VARIANT=chat \
experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh
```

The run used local Qwen3-1.7B through vLLM inside the bwrap rootfs with Triton
attention and `VLLM_USE_FLASHINFER_SAMPLER=0`.

## Results

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@8 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 64 | 0.125 | 0.125 | 0.125 | 0.125 | easy 1, elicitable 0, unreached 7 |
| OOD | 8 | 64 | 0.000 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Strict-format pass@k matched answer pass@k on both splits. Increasing rollouts
from 4 to 8 did not expose additional elicitable AIME problems on this slice.
The one dev success was `AIME/67`, which Qwen solved on every sampled rollout.

## Failure Modes

The dominant failure was still final-answer contract failure:

| Split | Missing final answer | Wrong final value | Successes |
| --- | ---: | ---: | ---: |
| dev | 37 | 19 | 8 |
| OOD | 40 | 24 | 0 |

Representative unreached dev failure:

```text
Problem: AIME/60
Error: missing final answer
Behavior: long algebraic setup but no verifier-readable FINAL line before the
generation limit.
```

Representative wrong-answer dev failure:

```text
Problem: AIME/62
Error: final value 25+256=281 does not match answer 371
Behavior: emitted a strict final answer, but the mathematical count was wrong.
```

Solved dev example:

```text
Problem: AIME/67
Final: 25
Behavior: all eight rollouts derived xy = 25 and emitted FINAL: 25.
```

## Interpretation

This is a clean hard-reasoning calibration result, not an AIME benchmark claim.
The bwrap/vLLM/import/report path is stable for a larger no-tool AIME sample
budget, but the current Qwen3-1.7B prompt condition mostly fails either by
omitting `FINAL:` or by producing a wrong final integer. For scaffold-to-policy,
this slice is a hard-negative target: simple best-of-8 sampling does not produce
base-elicitable coverage beyond the one easy dev problem.
