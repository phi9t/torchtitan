# T07c report: real-weight parity gate for GLM-5.3-Flash

## Result

Claim label: `deterministic_comparison`.

Completed the real-weight GLM-5.3-Flash parity gate in the experiment seam. The
reusable harness is `torchtitan/experiments/glm53_flash/hf_parity.py`, with
subcommands for checkpoint readiness, full-index key coverage, GLM chat/mask
inspection, and slice parity against an executable HF oracle.

No commit was made.

## Environment and inputs

- TorchTitan repo: `/data02/home/philip.yang/workspace/torchtitan`.
- TorchTitan test env: `TORCHTITAN_ROOTFS_ENV=glm53_train`.
- HF oracle env: `TORCHTITAN_ROOTFS_ENV=glm53_ref`.
- Rootfs commands used `unset CC CXX`.
- Formal cache binding used
  `TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache`,
  exposing `/project/formal-cache`.
- HF Transformers commit:
  `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`.
- Observed HF env: `transformers 5.19.0.dev0`, `torch 2.13.0+cu132`.
- Real BF16 checkpoint:
  `/data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16`.
- Checkpoint staged marker:
  `.staged.json` present, `file_count=130`,
  `manifest_md5=365857f8b1d7cb0841992fff91240ccf`,
  `staged_at=2026-10-14T10:23:44Z`,
  source URI
  `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538`.

Evidence artifacts:

- `.scratch/glm53-flash-sft-ladder/evidence/task-07c/coverage.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-07c/chat-mask.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-07c/slice-0-3-fp32.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-07c/slice-20-27-fp32.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-07c/slice-0-3-bf16.json`
- `.scratch/glm53-flash-sft-ladder/evidence/task-07c/slice-20-27-bf16.json`

## Slices tested

Each slice used real embeddings, selected real transformer layers, final norm,
and `lm_head`, with MTP disabled. Inputs were four Countdown prompts rendered by
the staged GLM tokenizer/chat template plus one synthetic 4096-token sequence to
exercise sparse top-k selection over more pools.

The HF oracle layer type spelling is `indexed_attention`; the TorchTitan flavor
uses `deepseek_sparse_attention`. The harness translates the slice config while
preserving the real layer schedule.

Summary:

| Evidence | Device | Dtype | Layers | Samples | Max layer abs | Max final abs | Min top-1 | Max KL | Raw top-k mismatches | Selection-mask mismatches |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `slice-0-3-fp32.json` | CPU | fp32 | 0-3 | 5 | 5.7220458984375e-06 | 1.0013580322265625e-05 | 1.0 | -3.0586821964107003e-09 | 1472 | 0 |
| `slice-20-27-fp32.json` | CPU | fp32 | 20-27 | 5 | 0.058231860399246216 | 0.7716797590255737 | 1.0 | 0.00019110699940938503 | 4256 | 0 |
| `slice-0-3-bf16.json` | CUDA | bf16 | 0-3 | 5 | 0.1875 | 1.6025390625 | 0.97900390625 | 0.00028162464150227606 | 4509336 | 19424 |
| `slice-20-27-bf16.json` | CPU | bf16 | 20-27 | 5 | 2.25 | 8.96875 | 0.43421053886413574 | 0.26943421363830566 | 12335348 | 1035016 |

The fp32 slices support the deterministic-comparison claim. Raw sparse top-k
ordering can differ on ties/order effects, especially in the long synthetic
sequence, but the selected sparse attention mask matched exactly for both fp32
slices. The bf16 runs are retained as evidence of bf16 behavior, not as exact
identity proof.

## Per-layer diff summary

Layer maxima are max absolute hidden-state differences over all five samples in
the artifact.

### Layers 0-3, fp32 CPU

| Layer | TT kind | HF kind | Max abs | Max mean abs |
| ---: | --- | --- | ---: | ---: |
| 0 | `linear_attention` | `linear_attention` | 1.6763806343078613e-08 | 4.617356844094367e-12 |
| 1 | `linear_attention` | `linear_attention` | 2.8870999813079834e-08 | 7.385945188786991e-10 |
| 2 | `linear_attention` | `linear_attention` | 1.7881393432617188e-07 | 3.7240042161101883e-09 |
| 3 | `deepseek_sparse_attention` | `indexed_attention` | 5.7220458984375e-06 | 1.0460644794818563e-08 |

### Layers 20-27, fp32 CPU

| Layer | TT kind | HF kind | Max abs | Max mean abs |
| ---: | --- | --- | ---: | ---: |
| 20 | `linear_attention` | `linear_attention` | 1.6391277313232422e-07 | 2.344386684782762e-09 |
| 21 | `linear_attention` | `linear_attention` | 7.748603820800781e-07 | 1.6903278776680963e-08 |
| 22 | `linear_attention` | `linear_attention` | 1.6689300537109375e-06 | 3.7413951048392846e-08 |
| 23 | `deepseek_sparse_attention` | `indexed_attention` | 3.427267074584961e-06 | 1.159944176265526e-07 |
| 24 | `linear_attention` | `linear_attention` | 5.245208740234375e-06 | 1.3216838112839469e-07 |
| 25 | `linear_attention` | `linear_attention` | 5.7220458984375e-05 | 1.8114174338279554e-07 |
| 26 | `linear_attention` | `linear_attention` | 0.045856863260269165 | 8.540644921595231e-05 |
| 27 | `deepseek_sparse_attention` | `indexed_attention` | 0.058231860399246216 | 0.00014791548892389983 |

The middle-slice fp32 outlier is localized to sample `cd-17-95-3-t75` at layers
26 and 27, with final-logit max abs `0.7716797590255737`, top-1 agreement
`1.0`, and KL `0.00019110699940938503`. Follow-up diagnostics showed the
TorchTitan and HF router/MLP implementations match exactly when fed identical
inputs; the observed outlier comes from router sensitivity after accumulated
drift, with only a small number of router slots changing and the final top-1
remaining identical. Sparse selection masks still matched exactly.

### Layers 0-3, bf16 CUDA

| Layer | TT kind | HF kind | Max abs | Max mean abs |
| ---: | --- | --- | ---: | ---: |
| 0 | `linear_attention` | `linear_attention` | 0.001953125 | 2.9864133466617204e-05 |
| 1 | `linear_attention` | `linear_attention` | 0.00146484375 | 3.1776617106515914e-05 |
| 2 | `linear_attention` | `linear_attention` | 0.00390625 | 7.257735705934465e-05 |
| 3 | `deepseek_sparse_attention` | `indexed_attention` | 0.1875 | 0.0003017361741513014 |

### Layers 20-27, bf16 CPU

| Layer | TT kind | HF kind | Max abs | Max mean abs |
| ---: | --- | --- | ---: | ---: |
| 20 | `linear_attention` | `linear_attention` | 0.39453125 | 0.001607379294000566 |
| 21 | `linear_attention` | `linear_attention` | 0.4658203125 | 0.0041341218166053295 |
| 22 | `linear_attention` | `linear_attention` | 0.7197265625 | 0.010747723281383514 |
| 23 | `deepseek_sparse_attention` | `indexed_attention` | 0.82421875 | 0.02128613367676735 |
| 24 | `linear_attention` | `linear_attention` | 0.775390625 | 0.026652799919247627 |
| 25 | `linear_attention` | `linear_attention` | 2.25 | 0.03139518201351166 |
| 26 | `linear_attention` | `linear_attention` | 1.875 | 0.03646012023091316 |
| 27 | `deepseek_sparse_attention` | `indexed_attention` | 1.875 | 0.03983579948544502 |

The first 20-27 bf16 attempt on CUDA OOMed during model construction/expert
stacking. The final artifact is a CPU bf16 rerun.

## Index agreement

For fp32:

- Layers 0-3: Countdown samples had exact raw index matches; the 4096-token
  synthetic sample had 1472 raw order mismatches at sparse layer 3, but
  `selection_mask_mismatch_count=0`.
- Layers 20-27: sparse layers 23 and 27 had
  `selection_mask_mismatch_count=0` for all samples. Raw order mismatches were
  16 for `cd-17-95-3-t75` at layer 27 and 1696/2544 for the 4096-token
  synthetic sample at layers 23/27.

The acceptance-relevant sparse selection masks match exactly in fp32.

For bf16:

- Layers 0-3: raw top-k mismatches `4509336`; selection-mask mismatches
  `19424`.
- Layers 20-27: raw top-k mismatches `12335348`; selection-mask mismatches
  `1035016`.

The bf16 index differences are reported but are not used as exact deterministic
identity proof.

## Full-checkpoint key coverage

Coverage artifact: `.scratch/glm53-flash-sft-ladder/evidence/task-07c/coverage.json`.

- `adapter_error=null`
- `total_hf_keys=38770`
- `mapped_hf_keys=37534`
- Explicit skips: `mtp=889`, `vision_tower=347`
- `unknown_hf_keys=[]`
- `native_keys=1330`
- `restored_native_keys=1330`
- `missing_native_keys=[]`
- `unexpected_native_keys=[]`
- `shape_mismatches=[]`
- `restored_shape_mismatches=[]`

This covers the full real index with every HF key mapped or explicitly skipped.

## Tokenizer and chat-template evidence

Evidence artifact: `.scratch/glm53-flash-sft-ladder/evidence/task-07c/chat-mask.json`.

The first two Countdown examples each produced `num_tokens=98` and
`num_supervised_labels=22`.

Prompt render begins with the GLM markers and reasoning system text:

```text
[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Numbers: ...
```

The prompt render ends with the assistant thinking opener:

```text
FINAL: 47<|assistant|><think>
```

The full training render inserts an empty thinking block before the assistant
answer:

```text
FINAL: 47<|assistant|><think></think>14 + 49 = 63
63 - 16 = 47
FINAL: 47
```

The assistant-only loss mask keeps system/user/prompt tokens at `IGNORE_INDEX`.
The first supervised position is shifted next-token style:

```text
pos 75 token "<think>"   label "</think>"
pos 76 token "</think>"  label "14"
pos 77 token "14"        label " +"
pos 78 token " +"        label " "
pos 79 token " "         label "49"
```

The experiment config now includes the real Countdown chat data path
`glm53_flash_countdown_lora` using the real GLM tokenizer and assistant-only
masking.

## Fixes made in the experiment seam

- Added `hf_parity.py` for real-weight checkpoint polling/coverage, chat-mask
  inspection, and HF-vs-TorchTitan slice parity.
- Patched the local harness, not site-packages, to align HF sparse-mask device
  handling during CUDA comparison.
- Pinned HF KDA execution to the pinned Transformers Python fallback, avoiding
  optional FLA/Triton dispatch drift in the executable oracle.
- Aligned TorchTitan KDA reference order with the HF fallback, including
  transpose/contiguous behavior, normalization order, chunk recurrence, and
  output reshape.
- Made TorchTitan KDA respect `attn_impl="reference"`.
- Moved sparse top-k indices to the query device in the reference attention path.
- Aligned MoE expert accumulation order to the HF sentinel-aware loop.
- Preserved real middle-slice layer schedules through `sparse_attention_mode`
  and `layer_offset`.
- Updated the state-dict adapter to explicitly skip `model.visual.*` vision
  tower keys.
- Added the real-checkpoint coverage test that skips when the checkpoint is not
  visible.

No core files were edited.

## Verification

Focused pytest:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py -x'
```

Result:

```text
14 passed, 1 skipped, 14 warnings
```

Scoped hooks under `glm53_train` passed for the changed GLM files:

```text
ufmt
flake8
pydoclint
codespell
trailing-whitespace
end-of-file-fixer
check-ast
check-merge-conflict
```

Additional check:

```bash
git diff --check -- torchtitan/experiments/glm53_flash
```

Result: passed.

Full `pre-commit run --all-files` was not run.

## Caveats and next-use notes

- The fp32 evidence is the deterministic comparison. The middle slice has one
  localized router-sensitive outlier, but top-1 agreement remains 1.0 and fp32
  sparse selection masks are exact.
- The bf16 artifacts are noisier, especially the 20-27 slice and long synthetic
  sparse case. Treat them as bf16 behavior evidence, not exact identity proof.
- The 20-27 bf16 CUDA attempt OOMed, so the final 20-27 bf16 artifact is CPU.
- The staged checkpoint was read only and was not modified.
- The worktree remains dirty by design; no commit was made.
