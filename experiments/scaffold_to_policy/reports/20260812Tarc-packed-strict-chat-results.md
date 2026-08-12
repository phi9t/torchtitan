# ARC-AGI-2 Packed Prompt Results

Run IDs:

- `20260812Tarc-agi2-packed-chat-calibration`
- `20260812Tarc-agi2-packed-strict-chat-calibration`

These runs continue the ARC-AGI-2 exact-grid branch after the earlier
`strict_chat` calibration. They use the same pinned public ARC-AGI-2 source and
the same exact verifier, but test compact packed-grid prompting under the bwrap
rootfs.

## Commands

Packed prompt:

```bash
RUN_ID=20260812Tarc-agi2-packed-chat-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/arc_agi2_public_vllm_calibration_packed_chat \
RESULTS_ROOT=experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_packed_chat \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.05 \
SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=4096 \
MAX_MODEL_LEN=4096 \
MAX_NEW_TOKENS=256 \
PROMPT_VARIANT=packed_chat \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

Packed strict prompt:

```bash
RUN_ID=20260812Tarc-agi2-packed-strict-chat-calibration \
DATA_ROOT=experiments/scaffold_to_policy/data/arc_agi2_public_vllm_calibration_packed_strict_chat \
RESULTS_ROOT=experiments/scaffold_to_policy/results/arc_agi2_public_vllm_calibration_packed_strict_chat \
DEV_PROBLEMS=8 OOD_PROBLEMS=8 NUM_ROLLOUTS=4 \
GPU_MEMORY_UTILIZATION=0.05 \
SCAFFOLD_TO_POLICY_VLLM_MAX_MODEL_LEN=4096 \
MAX_MODEL_LEN=4096 \
MAX_NEW_TOKENS=256 \
PROMPT_VARIANT=packed_strict_chat \
experiments/scaffold_to_policy/run_arc_agi2_public_vllm_smoke.sh
```

Both runs used local Qwen3-1.7B through vLLM in the bwrap rootfs with Triton
attention and `VLLM_USE_FLASHINFER_SAMPLER=0`.

## Prompt Preflight

The packed representation addresses the context blocker from the earlier ARC
calibration. With `MAX_MODEL_LEN=4096` and a 256-token generation reserve:

| Prompt | Dev selected | Dev max total tokens | OOD selected | OOD max total tokens |
| --- | ---: | ---: | ---: | ---: |
| `packed_chat` | 8/8 | 2645 | 8/8 | 3768 |
| `packed_strict_chat` | 8/8 | 2672 | 8/8 | 3795 |

This is the main infrastructure result: the packed prompt clears the 4096-token
gate for the selected 8 dev / 8 OOD slice without raising the model context.

## Scores

Packed prompt:

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@32 | Dominant failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 32 | 0.000 | 0.000 | 0.000 | missing final grid |
| OOD | 8 | 32 | 0.000 | 0.000 | 0.000 | missing final grid |

Packed strict prompt:

| Split | Problems | Rollouts | pass@1 | pass@4 | pass@32 | Dominant failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dev | 8 | 32 | 0.000 | 0.000 | 0.000 | non-grid JSON object, truncated JSON |
| OOD | 8 | 32 | 0.000 | 0.000 | 0.000 | truncated JSON, non-grid JSON object |

The packed-only prompt caused the model to reproduce compact input notation
without the required `FINAL:` line. The packed strict prompt restored strict
final-line formatting, but the payload remained invalid for the exact verifier.

## Examples

Packed-only dev example:

```text
Problem: ARC-AGI-2/00576224/0
Output: {
  "E1": "79/43",
  "E2": "86/64",
  "T": "32/78"
}
Error: missing final grid
```

Packed strict dev example:

```text
Problem: ARC-AGI-2/00576224/0
Output: FINAL: {"E1": "79/43", "E2": "86/64", "T": "32/78"}
Error: grid must be a non-empty list of rows
```

Packed strict OOD example:

```text
Problem: ARC-AGI-2/18286ef8/0
Output: FINAL: {"rows": ["777077707777", ... "777
Error: could not parse final grid JSON: Unterminated string starting at
```

## Interpretation

Packed prompting is useful infrastructure but not a capability improvement for
this Qwen3-1.7B no-tool ARC condition. It fixes the context fit problem while
preserving upstream exact-grid scoring, but Qwen still does not reliably emit a
valid JSON grid. The next ARC branch should target output construction directly:
for example, a two-stage condition that first asks for packed rows and then
deterministically converts those rows to the verifier's JSON-grid contract, or
a format-repair scaffold reported separately from no-tool ARC.
