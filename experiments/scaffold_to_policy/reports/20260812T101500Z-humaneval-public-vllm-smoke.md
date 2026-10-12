# HumanEval Public vLLM Smoke

Date: 2026-08-12

## Scope

This report records the first coding-lane smoke in the scaffold-to-policy
program. It imports a pinned slice of HumanEval, evaluates Qwen3-1.7B with
vLLM in a no-tool condition, and scores candidates by executing the benchmark's
Python tests in a separate subprocess.

This is a harness and plumbing smoke, not a HumanEval benchmark claim. The run
uses 4 total examples and 2 rollouts per problem.

## Command

The run used the bwrap rootfs:

```bash
RUN_ID=20260812T101500Z-humaneval-public-vllm-smoke \
DATA_ROOT=experiments/scaffold_to_policy/data/humaneval_public_vllm_smoke \
RESULTS_ROOT=experiments/scaffold_to_policy/results/humaneval_public_vllm_smoke \
DEV_PROBLEMS=2 \
OOD_PROBLEMS=2 \
OOD_OFFSET=64 \
NUM_ROLLOUTS=2 \
MAX_NEW_TOKENS=384 \
experiments/scaffold_to_policy/run_humaneval_public_vllm_smoke.sh
```

The entrypoint re-executes itself through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_humaneval_public_vllm_smoke.sh
```

The first scoring pass exposed a harness extraction bug: when the model
returned a full function, the scorer dropped prompt-level imports such as
`from typing import List`. The harness now preserves the prompt preamble before
full-function candidates, and the saved vLLM outputs were rescored inside the
rootfs.

## Dataset

Dataset provenance:

| Field | Value |
| --- | --- |
| Dataset | `openai/openai_humaneval` |
| Revision | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |
| Source split | `test` |
| Dev slice | offset 0, limit 2 |
| OOD slice | offset 64, limit 2 |

Problem-id hashes:

| Split | Problems | Problem-id hash |
| --- | ---: | --- |
| dev | 2 | `63ac036c058b488bd2a1efae7a9e762143fe0b0c1cca3f72c50978903d88d057` |
| ood_test | 2 | `7c885731013aa0287011b6535429e619f7a8d5752edf33a4fdcddbeb236520c0` |

## Verifier

The condition is no-tool. vLLM receives only the HumanEval prompt and a system
instruction. The run does not use retrieval, external package installation, or
an LLM judge.

The verifier:

- extracts candidate Python code from plain text or Markdown fences;
- preserves HumanEval prompt imports and preamble when the model returns a full
  function;
- appends the benchmark `check(candidate)` tests;
- runs the candidate in a separate isolated Python subprocess with a timeout;
- reports executable pass@k plus failure classes such as assertion failure,
  syntax error, name error, and timeout.

The subprocess boundary is an evaluation isolation boundary, not a secure
sandbox for malicious code.

## Artifacts

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/humaneval_public_vllm_smoke/
experiments/scaffold_to_policy/results/humaneval_public_vllm_smoke/
```

The final report input is:

```text
experiments/scaffold_to_policy/results/humaneval_public_vllm_smoke/manifests/report_input_20260812T101500Z-humaneval-public-vllm-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

## Results

| Split | Problems | Rollouts | pass@1 | pass@2 | Buckets |
| --- | ---: | ---: | ---: | ---: | --- |
| dev | 2 | 4 | 0.500 | 0.500 | easy 1, elicitable 0, unreached 1 |
| ood_test | 2 | 4 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 2 |

Failure breakdown:

| Split | Successes | Dominant failures |
| --- | ---: | --- |
| dev | 2 | 2 timeouts on `separate_paren_groups` |
| ood_test | 0 | 4 assertion failures |

## Examples

Easy dev example:

```text
Task: HumanEval/0, has_close_elements(numbers, threshold)

Generation:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers) - 1):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[j] - numbers[i]) < threshold:
                return True
    return False

Verifier result: success
```

Unreached dev example:

```text
Task: HumanEval/1, separate_paren_groups(paren_string)

Generation:
def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    stack = []
    i = 0
    while i < len(paren_string):
        ...

Verifier result: timeout
```

The candidate mutates loop state in a way that can fail to make progress on
some test inputs, so the subprocess timeout is the correct failure mode.

OOD assertion example:

```text
Task: HumanEval/64, vowels_count(s)

Generation:
def vowels_count(s):
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count

Verifier result: assertion failure
```

The benchmark tests require details not captured by this simple lowercase-vowel
implementation.

OOD assertion example:

```text
Task: HumanEval/65, circular_shift(x, shift)

Generation:
def circular_shift(x, shift):
    s = str(x)
    n = len(s)
    shift = shift % n
    return s[shift:] + s[:shift] if shift != 0 else s

Verifier result: assertion failure
```

The implementation shifts in the wrong direction and does not implement the
specified reverse behavior when the shift exceeds the number of digits.

## Interpretation

This clears the first coding-lane harness gate:

- pinned public HumanEval rows import inside the bwrap rootfs;
- vLLM generation and executable scoring both run through rootfs entrypoints;
- the scorer records deterministic pass@k and concrete failure modes;
- saved generations can be rescored after harness fixes.

This is not a LiveCodeBench, SWE-bench, Terminal-Bench, or HumanEval leaderboard
claim. The next coding step is either a larger HumanEval calibration for
training-data construction or a pinned external harness integration where this
repo owns rootfs launch, result ingestion, report inputs, and artifact
boundaries while the benchmark harness owns task execution.
