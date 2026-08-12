# Countdown Base-Elicitable Subset And Example Audit

Date: 2026-08-12

## Scope

This report records the first artifact-derived analysis block added to the
Countdown report input. It covers:

- base-elicitable subset metrics;
- deterministic representative examples for wins, regressions, unchanged
  failures, and format failures;
- the current evidence gap before replication and reasoning-transfer work.

The analysis is generated from existing `evaluations.jsonl` artifacts by:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'python -m torchtitan.experiments.countdown_search_distill.cli build-report-input \
   --experiment-root experiments/countdown_search_distill \
   --mode full \
   --run-id 20260812T001300Z-full-formatting-continuation \
   --manifest experiments/countdown_search_distill/results/manifests/20260812T001300Z-full-formatting-continuation.jsonl \
   --output experiments/countdown_search_distill/results/manifests/report_input_20260812T001300Z-full-formatting-continuation.json'
```

The final report input still passes all registry checks.

## Base-Elicitable Subset

The base-elicitable subset contains problems where base Qwen3-1.7B solved the
problem within 32 samples but not at sample 1. This is the cleanest subset for
measuring scaffold-to-policy compression because the scaffold found evidence
that a solution is reachable.

| Split | Arm | Problems | pass@1 | pass@32 | strict pass@1 | strict pass@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev | base | 221 | 0.000 | 1.000 | 0.000 | 0.312 |
| dev | clean | 221 | 0.235 | 0.977 | 0.118 | 0.896 |
| dev | formatting | 221 | 0.421 | 0.982 | 0.416 | 0.982 |
| iid_test | base | 476 | 0.000 | 1.000 | 0.000 | 0.326 |
| iid_test | clean | 476 | 0.252 | 0.989 | 0.116 | 0.908 |
| iid_test | formatting | 476 | 0.418 | 0.994 | 0.403 | 0.994 |
| ood_test | base | 230 | 0.000 | 1.000 | 0.000 | 0.335 |
| ood_test | clean | 230 | 0.252 | 0.970 | 0.117 | 0.796 |
| ood_test | formatting | 230 | 0.370 | 0.978 | 0.343 | 0.978 |

Interpretation: `formatting` is not only exploiting easy problems. On the
base-elicitable OOD subset, it compresses 37.0% of scaffold-reachable problems
to sample 1 while preserving 97.8% pass@32 and 97.8% strict-format pass@32.

## Representative Examples

The report input now carries deterministic examples for every split and arm:

| Category | Count |
| --- | ---: |
| win | 15 |
| regression | 15 |
| unchanged_failure | 15 |
| format_failure | 15 |

That is one example per category for each of five arms across three splits.

### OOD Formatting Win

Problem:

```text
numbers: 1, 39, 29
target: 69
canonical solution:
39 + 1 = 40
40 + 29 = 69
FINAL: 69
```

Base solved at sample 7. Formatting solved strictly at sample 1:

```text
The answer is 69.
1 + 39 = 40
40 + 29 = 69
FINAL: 69
The answer is 69.
29 + 39 = 68
68 + 1 = 69
FINAL: 69
...
```

The same caveat remains: the model often keeps generating after a valid strict
trace.

### OOD Formatting Regression

Problem:

```text
numbers: 13, 28, 67
target: 52
canonical solution:
28 - 13 = 15
67 - 15 = 52
FINAL: 52
```

Base solved at sample 1. Formatting solved only at sample 3; the first sample
degenerated into repeated code-fence markers. This reinforces that the next
formatting iteration needs stop-quality shaping, not only stricter final-line
targets.

### OOD Formatting Format Failure

Problem:

```text
numbers: 24, 29, 81
target: 86
canonical solution:
81 + 29 = 110
110 - 24 = 86
FINAL: 86
```

Formatting sample 1 had a valid arithmetic trace but missed the strict final
line before drifting:

```text
81 + 29 = 110
110 - 24 = 86
```

Strict success appeared later at sample 2. This category is now machine-picked
instead of manually found during report writing.

## Decision

This closes the first reporting gap from the scaffold-to-policy spec:
base-elicitable subset metrics and representative examples are now generated
from artifacts as part of the report input. The next missing scientific step is
still replication: at least one second split draw or training seed for the
current champion before moving to synthetic reasoning transfer.
