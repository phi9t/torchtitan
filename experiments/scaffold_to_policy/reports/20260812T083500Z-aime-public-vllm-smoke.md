# AIME Public vLLM Smoke

Run ID: `20260812T083500Z-aime-public-vllm-smoke`

This smoke adds a harder no-tool exact-integer reasoning gate. It imports a
small public AIME 2024 slice, evaluates Qwen3-1.7B with vLLM inside the bwrap
rootfs, and scores only exact final answers through the existing MATH-style
normalizer. It is a plumbing and calibration smoke, not an AIME benchmark claim.

## Command

```bash
RUN_ID=20260812T083500Z-aime-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/aime_public_vllm_smoke_min \
RESULTS_ROOT=experiments/scaffold_to_policy/results/aime_public_vllm_smoke_min \
DEV_PROBLEMS=2 \
OOD_PROBLEMS=2 \
OOD_OFFSET=8 \
NUM_ROLLOUTS=2 \
MAX_NEW_TOKENS=768 \
experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh
```

The entrypoint re-executes through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_aime_public_vllm_smoke.sh
```

## Dataset And Verifier

| Field | Value |
| --- | --- |
| Dataset | `HuggingFaceH4/aime_2024` |
| Revision | `main` |
| Source split | `train` |
| Dev slice | offset 0, limit 2 |
| OOD slice | offset 8, limit 2 |
| Verifier | `math_style_normalized_final_v1` |
| Output contract | final line `FINAL: <answer>` |

The importer keeps AIME answers as three-digit strings for display, while
normalization compares the integer value. The verifier does not use tools,
retrieval, symbolic algebra, or an LLM judge.

## Results

| Split | Problems | Rollouts | pass@1 | pass@2 | Strict pass@1 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 2 | 4 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 2 |
| ood_test | 2 | 4 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 2 |

Failure breakdown:

| Split | Failures |
| --- | --- |
| dev | 3 missing final answers; 1 wrong final value `20571/100` vs `204` |
| ood_test | 2 missing final answers; wrong final values `1+2310=2311` vs `116` and `404` vs `809` |

## Examples

Dev unreached example:

```text
Task: AIME/60
Expected: 204

Generation excerpt:
Let's denote the time taken for the walk as T (in hours), and the time spent
in the coffee shop as t (in minutes). We are given:
...

Verifier result: missing final answer
```

OOD wrong-answer example:

```text
Task: AIME/68
Expected: 116

Generation excerpt:
This is a classic game theory problem, often solved using modulo reasoning.
...

Verifier result: final value 1+2310=2311 does not match answer 116
```

## Interpretation

This is the first clear hard-negative public reasoning smoke in the ladder.
The rootfs/vLLM/import/report path works, but the no-tool Qwen3-1.7B sample is
not solving this tiny AIME slice and often misses the exact final-answer
contract. That makes AIME useful as a later scaffold-to-policy target only
after larger best-of-k collection and strict format shaping are in place.
