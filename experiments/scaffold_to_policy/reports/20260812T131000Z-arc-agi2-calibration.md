# ARC-AGI-2 Exact-Grid Calibration

Run ID: `20260812T131000Z-arc-agi2-public-vllm-calibration`

This is a larger no-tool ARC-AGI-2 exact-grid calibration than the initial
2-problem smoke. It is still not a leaderboard claim: it uses 8 dev and 8 OOD
training-split tasks with 4 rollouts per problem.

## Environment

- Entrypoint: `experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh`
- Execution boundary: `scripts/rootfs/enter_rootfs.sh`
- Model: `./assets/hf/Qwen3-1.7B`
- ARC source: `https://github.com/arcprize/ARC-AGI-2.git`
- ARC revision: `f3283f727488ad98fe575ea6a5ac981e4a188e49`
- vLLM settings: `max_model_len=8192`, `gpu_memory_utilization=0.24`

The lower GPU-memory setting was needed because all 8 local B200s were already
partially occupied by another process. The first attempt failed before
generation when vLLM tried to reserve 92% of GPU memory. A second attempt with
`gpu_memory_utilization=0.24` and `max_model_len=4096` reached tokenization but
failed because one selected ARC prompt exceeded 4096 tokens. The completed run
therefore used 8192 context length with the same exact-grid verifier.

## Results

| Split | Problems | Rollouts/problem | pass@1 | pass@4 | Buckets |
| --- | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 4 | 0.000 | 0.125 | easy 0, elicitable 1, unreached 7 |
| OOD | 8 | 4 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Failure breakdown:

| Split | Success | Wrong grid | Parse failure | Missing final |
| --- | ---: | ---: | ---: | ---: |
| dev | 1 | 3 | 24 | 4 |
| OOD | 0 | 1 | 27 | 4 |

The report-input checks all passed:

```text
split_registry_selected=true
summaries_present=true
summary_split_counts_match=true
```

Report input:

```text
experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration/manifests/report_input_20260812T131000Z-arc-agi2-public-vllm-calibration.json
```

## Examples

Solved on the second rollout:

```text
Problem: ARC-AGI-2/00576224/0
Rollout 1: wrong 6x6 grid using colors from the training examples.
Rollout 2: exact success.
Expected and emitted final grid:
[[3,2,3,2,3,2],
 [7,8,7,8,7,8],
 [2,3,2,3,2,3],
 [8,7,8,7,8,7],
 [3,2,3,2,3,2],
 [7,8,7,8,7,8]]
```

Representative parse failure:

```text
Problem: ARC-AGI-2/009d5c81/0
Observed pattern:
FINAL:
[[0,0,0,...]]

The verifier rejects this because the output contract is a single line exactly
matching `FINAL: <json-grid>`.
```

Representative missing-final failure:

```text
Problem: ARC-AGI-2/00d62c1b/0
All 4 rollouts omitted a valid final-grid line.
```

Representative OOD failure:

```text
Problem: ARC-AGI-2/182e5d0f/0
Observed pattern:
Final: <json-grid>[
  [7,7,7,0,3,0,...]
]

The model mixed the placeholder text with a multi-line grid and also used the
wrong capitalization. The exact verifier correctly rejected it.
```

## Interpretation

The larger slice confirms the initial smoke: this lane is hard for Qwen3-1.7B
in the no-tool condition, and most failures are output-contract failures before
they are exact-grid reasoning successes. The immediate next improvement should
not relax scoring. It should either:

- train or prompt against the single-line `FINAL: <json-grid>` contract; or
- add a separate format-repair experiment that is explicitly labeled and still
  rescored by the same exact-grid verifier.

This calibration also exposed two infrastructure requirements for ARC:

- vLLM GPU memory utilization needs to be configurable for shared-GPU runs; and
- ARC prompt length should be preflighted when using chat prompts and larger
  selected slices.
