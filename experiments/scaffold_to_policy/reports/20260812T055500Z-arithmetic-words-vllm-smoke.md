# Arithmetic Words vLLM Smoke

Date: 2026-08-12

## Scope

This report records the first real-model reasoning smoke for the
`arithmetic_words` task. It is an infrastructure smoke, not a scientific
scaffold-to-policy result: the generated problems are intentionally tiny and
Qwen3-1.7B solved all of them on the first sampled answer.

The run used:

```bash
RUN_ID=20260812T055500Z-arithmetic-words-vllm-smoke \
NUM_ROLLOUTS=4 \
MAX_NEW_TOKENS=128 \
experiments/scaffold_to_policy/run_arithmetic_words_vllm_smoke.sh
```

The script re-executed through:

```bash
scripts/rootfs/enter_rootfs.sh -- experiments/scaffold_to_policy/run_arithmetic_words_vllm_smoke.sh
```

## Artifacts

Generated artifacts are ignored by git and live under:

```text
experiments/scaffold_to_policy/data/arithmetic_words_vllm_smoke/
experiments/scaffold_to_policy/results/arithmetic_words_vllm_smoke/
```

The report input is:

```text
experiments/scaffold_to_policy/results/arithmetic_words_vllm_smoke/manifests/report_input_20260812T055500Z-arithmetic-words-vllm-smoke.json
```

Report-input checks:

| Check | Result |
| --- | --- |
| `split_registry_selected` | true |
| `summaries_present` | true |
| `summary_split_counts_match` | true |

## Results

The smoke generated 8 train, 4 dev, and 4 OOD synthetic word problems. It then
evaluated dev and OOD with Qwen3-1.7B through vLLM, 4 rollouts per problem, and
the strict final-integer verifier.

| Split | Problems | Rollouts | pass@1 | pass@4 | strict pass@1 | strict pass@4 | Buckets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 4 | 16 | 1.000 | 1.000 | 1.000 | 1.000 | easy 4, elicitable 0, unreached 0 |
| ood_test | 4 | 16 | 1.000 | 1.000 | 1.000 | 1.000 | easy 4, elicitable 0, unreached 0 |

Every rollout passed verification, so the failure breakdown for both splits is
`success=16`.

Example dev output:

```text
Calculation trace:
1. Calculate the number of pencils in the boxes: 12 boxes x 14 pencils/box = 168 pencils
2. Add the extra pencils: 168 pencils + 8 pencils = 176 pencils

FINAL: 176
```

## Interpretation

This closes the first reasoning-lane infrastructure gap: local exact-verifier
tasks can now run real Qwen3/vLLM generation inside the bwrap rootfs and emit
the same split-registry, summary, and report-input artifacts as the fixture
smoke.

It does not establish transfer. The task is too easy for Qwen3-1.7B under the
current generator and prompt. The next reasoning-lane step should add a
calibrated arithmetic or symbolic generator that produces nontrivial
base-elicitable and unreached buckets before any LoRA training or benchmark
claim.
