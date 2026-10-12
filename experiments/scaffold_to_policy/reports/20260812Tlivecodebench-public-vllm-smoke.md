# LiveCodeBench Public-Test Smoke

Run ID: `20260812Tlivecodebench-public-vllm-smoke`

This run adds a contest-style stdin/stdout coding lane for
`livecodebench/code_generation`. It executes only the released public test
cases from the selected rows, so it is a public-test smoke and not an official
LiveCodeBench leaderboard result.

## Command

```bash
RUN_ID=20260812Tlivecodebench-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/livecodebench_public_vllm_smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/livecodebench_public_vllm_smoke \
DEV_PROBLEMS=2 OOD_PROBLEMS=2 DEV_OFFSET=0 OOD_OFFSET=32 \
NUM_ROLLOUTS=2 MAX_NEW_TOKENS=1536 \
GPU_MEMORY_UTILIZATION=0.05 TIMEOUT_SECONDS=10 \
experiments/scaffold_to_policy/run_livecodebench_public_vllm_smoke.sh
```

The run imported public rows, validated split registries, selected a GPU with
the shared vLLM preflight, generated with local Qwen3-1.7B through vLLM inside
the bwrap rootfs, and executed each candidate against released public
stdin/stdout examples in isolated Python subprocesses.

## Results

| Split | Problems | Rollouts | pass@1 | pass@2 | pass@32 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 2 | 4 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 2 |
| OOD | 2 | 4 | 0.000 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 2 |

Failure breakdown:

| Split | Indentation errors | Wrong answers |
| --- | ---: | ---: |
| dev | 3 | 1 |
| OOD | 2 | 2 |

## Examples

Selected dev tasks:

```text
LiveCodeBench/abc301_f Anti
LiveCodeBench/abc301_a Overall Winner
```

Selected OOD tasks:

```text
LiveCodeBench/abc307_d Mismatched Parentheses
LiveCodeBench/abc307_c Ideal Sheet
```

Representative dev failure:

```text
Problem: LiveCodeBench/abc301_f Anti
Failure: indentation error
Behavior: the model began a long dynamic-programming solution, but the extracted
Python source was syntactically incomplete or malformed before public tests
could evaluate correctness.
```

Representative wrong-answer class:

```text
Failure: wrong answer
Behavior: the candidate executed under the rootfs Python subprocess and produced
stdout, but at least one released public stdin/stdout example did not match
after trailing-whitespace normalization.
```

## Interpretation

The contest-code lane validates the important infrastructure boundary: imported
LiveCodeBench rows preserve prompts, titles, public tests, and provenance; vLLM
generation stays inside the bwrap rootfs; and executable scoring is deterministic
for released public tests. The model result is a hard negative on this tiny
slice. The next useful coding step is improving the prompt/extraction contract
or moving to official LiveCodeBench private-test harnessing once those artifacts
are available, rather than treating this public-test smoke as a benchmark score.
