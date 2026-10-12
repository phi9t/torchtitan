# ARC-AGI-2 Strict-Chat Results

Run ID: `20260812T150000Z-arc-agi2-strict-chat-calibration`

This run completes the strict prompt condition that was previously blocked by
GPU availability. It uses the same no-tool exact-grid verifier, the same pinned
ARC-AGI-2 revision, and the same 8 dev / 8 OOD slice size as the earlier
`chat` calibration. It is still a small calibration slice, not a leaderboard
claim.

## Command

```bash
RUN_ID=20260812T150000Z-arc-agi2-strict-chat-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/arc_agi2_public_vllm_calibration_strict_chat_rerun \
RESULTS_ROOT=experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat_rerun \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.24 \
SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=8192 \
MAX_MODEL_LEN=8192 \
PROMPT_VARIANT=strict_chat \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

The run entered through `scripts/rootfs/enter_rootfs.sh`, passed prompt/context
preflight, passed the new vLLM GPU-memory preflight, generated with vLLM, and
wrote the report input:

```text
experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_strict_chat_rerun/manifests/report_input_20260812T150000Z-arc-agi2-strict-chat-calibration.json
```

## Environment

| Field | Value |
| --- | --- |
| Model | `./assets/hf/Qwen3-1.7B` |
| ARC repo | `https://github.com/arcprize/ARC-AGI-2.git` |
| ARC revision | `f3283f727488ad98fe575ea6a5ac981e4a188e49` |
| Prompt variant | `strict_chat` |
| Max model length | `8192` |
| Max new tokens | `768` |
| GPU memory utilization | `0.24` |
| Rollouts/problem | `4` |

Preflights:

| Preflight | Result |
| --- | --- |
| dev prompt/context | selected 8/8, max total tokens 5,260 |
| OOD prompt/context | selected 8/8, max total tokens 7,516 |
| vLLM GPU memory | selected, 177.74 GiB free vs 42.80 GiB required |

## Results

| Prompt | Split | Problems | pass@1 | pass@4 | Buckets |
| --- | --- | ---: | ---: | ---: | --- |
| `chat` | dev | 8 | 0.000 | 0.125 | easy 0, elicitable 1, unreached 7 |
| `strict_chat` | dev | 8 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |
| `chat` | OOD | 8 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |
| `strict_chat` | OOD | 8 | 0.000 | 0.000 | easy 0, elicitable 0, unreached 8 |

Failure breakdown:

| Prompt | Split | Success | Wrong grid | Parse failure | Missing final |
| --- | --- | ---: | ---: | ---: | ---: |
| `chat` | dev | 1 | 3 | 24 | 4 |
| `strict_chat` | dev | 0 | 28 | 4 | 0 |
| `chat` | OOD | 0 | 1 | 27 | 4 |
| `strict_chat` | OOD | 0 | 20 | 12 | 0 |

## Examples

The strict prompt fixed the most obvious final-line formatting issue, but it
did not improve exact-grid reasoning on this slice.

For `ARC-AGI-2/00576224/0`, the earlier `chat` calibration solved the task on
rollout 2. The strict prompt produced a valid one-line final answer but emitted
only a 2x6 grid instead of the expected 6x6 grid:

```text
FINAL: [[3,2,3,2,3,2],[7,8,7,8,7,8]]
```

For `ARC-AGI-2/007bbfb7/0`, strict-chat outputs were parseable and one-line,
but the inferred grid content was still wrong:

```text
FINAL: [[7,0,7,7,0,7,7,0,7],[7,0,7,7,0,7,7,0,7],...]
```

For `ARC-AGI-2/18419cfa/0`, the strict prompt still ran into output truncation
or malformed JSON on long grids:

```text
error: could not parse final grid JSON: Expecting ',' delimiter
```

## Interpretation

The stricter one-line prompt improved format compliance in a narrow sense:
missing-final failures dropped to zero on both splits, and many generations now
had a strict final line. However, it converted most previous parse/missing-final
failures into wrong-grid failures and lost the one dev success found by the
original `chat` prompt. On this slice, the limiting issue is not only final-line
formatting; the model is often producing a syntactically valid but semantically
incorrect grid.

The next ARC step should not replace the original prompt with `strict_chat`.
Better candidates are a format-repair/rescoring condition over saved `chat`
generations, or a two-stage scaffold that first reasons freely and then emits a
single-line final grid while preserving the exact verifier.
