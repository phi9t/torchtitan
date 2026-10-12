# MBPP Public vLLM Smoke

Run ID: `20260812T084000Z-mbpp-public-vllm-smoke`

This smoke extends the coding lane beyond HumanEval with a small public MBPP
slice. It imports sanitized MBPP rows, infers an entry point from the released
assert tests, prompts Qwen3-1.7B with vLLM inside the bwrap rootfs, and scores
candidate functions by executing the benchmark asserts in an isolated Python
subprocess. It is a harness and calibration smoke, not an MBPP leaderboard
claim.

## Command

```bash
RUN_ID=20260812T084000Z-mbpp-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/mbpp_public_vllm_smoke_min \
RESULTS_ROOT=experiments/scaffold_to_policy/results/mbpp_public_vllm_smoke_min \
DEV_PROBLEMS=2 \
OOD_PROBLEMS=2 \
OOD_OFFSET=8 \
NUM_ROLLOUTS=2 \
MAX_NEW_TOKENS=512 \
experiments/scaffold_to_policy/run_mbpp_public_vllm_smoke.sh
```

## Dataset And Verifier

| Field | Value |
| --- | --- |
| Dataset | `google-research-datasets/mbpp` |
| Subset | `sanitized` |
| Revision | `main` |
| Source split | `test` |
| Dev slice | offset 0, limit 2 |
| OOD slice | offset 8, limit 2 |
| Verifier | `python_executable_tests_v1` |

The importer wraps MBPP `test_list` asserts into this repo's shared
`check(candidate)` contract. The candidate execution path is the same coding
subprocess verifier used for HumanEval. It is not a secure sandbox for
malicious code.

## Results

| Split | Problems | Rollouts | pass@1 | pass@2 | Buckets |
| --- | ---: | ---: | ---: | ---: | --- |
| dev | 2 | 4 | 1.000 | 1.000 | easy 2, elicitable 0, unreached 0 |
| ood_test | 2 | 4 | 0.500 | 0.500 | easy 1, elicitable 0, unreached 1 |

Failure breakdown:

| Split | Successes | Failures |
| --- | ---: | --- |
| dev | 4 | none |
| ood_test | 2 | 2 type errors |

## Examples

Dev success:

```text
Task: MBPP/11
Prompt: remove first and last occurrence of a given character from a string

Generation:
def remove_Occ(arg1, arg2):
    return arg1.replace(arg2, '', 2)

Verifier result: success
```

OOD failure:

```text
Task: MBPP/56

Generation:
def check(arg1):
    s = str(arg1)
    reversed_s = s[::-1]
    if int(s) == 2 * int(reversed_s) - 1:
        return True
    return False

Verifier result: type error
```

The failure is useful harness evidence: the model emitted a function named
`check` instead of the requested entry point, so the verifier passed the wrong
callable into the benchmark checks and produced a concrete executable failure.

## Interpretation

This clears the next coding-lane smoke after HumanEval. MBPP exercises looser
natural-language programming prompts and assert-list ingestion while preserving
deterministic executable scoring. The tiny run shows a positive signal on the
first two examples and a clear OOD failure mode. The next coding steps are a
larger MBPP calibration slice and then external terminal/container tasks once
the Harbor rootfs Docker blocker is resolved.
