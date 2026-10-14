# T07a report: GLM-5.3-Flash experiment seam

## Summary

Implemented a text-only `torchtitan.experiments.glm53_flash` model seam for
GLM-5.3-Flash without core edits. The seam includes released, mini, debug, and
LoRA-flavored configs; mHC, KDA, sparse MLA plus IndexPool/indexer, local MoE,
HF BF16 state-dict mapping, FSDP validation, focused parity tests, and 2-GPU
debug smoke runs.

No commit was made.

## Files

- `torchtitan/experiments/glm53_flash/__init__.py`
- `torchtitan/experiments/glm53_flash/flavors.py`
- `torchtitan/experiments/glm53_flash/model.py`
- `torchtitan/experiments/glm53_flash/state_dict_adapter.py`
- `torchtitan/experiments/glm53_flash/sharding.py`
- `torchtitan/experiments/glm53_flash/config_registry.py`
- `torchtitan/experiments/glm53_flash/tests/__init__.py`
- `torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py`
- `.scratch/glm53-flash-sft-ladder/evidence/bf16-model.safetensors.index.json`
- `.scratch/glm53-flash-sft-ladder/evidence/training/*.log`
- `.scratch/glm53-flash-sft-ladder/evidence/training/*/run_evidence/*`

## Reference Sources

- HF Transformers `glm5_next` modeling/config source:
  `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`
- `zai-org/GLM-5.3-Flash` config revision:
  `91b69215804f2d2036a8867f6ce44848e278dcda`
- `zai-org/GLM-5.3-Flash-BF16` safetensors index revision:
  `a5b45eb41df6402735dedc900be14a42e8d5e538`
- Downloaded reference files were staged under
  `/data02/home/philip.yang/devx/tmp/t07a_ref/`; only the BF16 index JSON was
  copied into repo-local scratch evidence for rootfs-visible state-dict tests.
  No real GLM weights were downloaded.

## Implementation Notes

- `flavors.py`
  - `glm53_flash`: 45 layers, 3:1 KDA:sparse schedule, 4-stream mHC,
    288 experts top-8 plus one shared expert, first 3 dense layers, NoPE,
    vocab/norm/epsilon/head dimensions from the research contract.
  - `debugmodel`: 4 layers with the same 3:1 schedule, KDA head dim 128,
    sparse MLA latent rank 512, 8 experts top-2.
  - `mini`: 16-layer intermediate ladder rung with the same mechanism pattern.
  - LoRA targets cover attention q/k/v/o, sparse MLA q/kv/o projections, KDA
    projections, and shared-expert projections.
- `model.py`
  - Implements GLM mHC with the corrected post-mix `comb^T @ residual`, GLM
    Sinkhorn epsilon placement, and final unweighted mean over streams.
  - Implements KDA with separate q/k/v depthwise causal convs, safe gate,
    beta update, low-rank output gate `g_a/g_b`, CPU reference path, and
    attn-gym CUDA path.
  - Implements sparse MLA with absorbed latent attention, detached/no-grad
    IndexPool/indexer, causal tail expansion, CPU reference attention, and
    attn-gym gather path when `attn_impl="fast"`.
  - Uses experiment-local LoRA wrappers and experiment-local activation
    checkpoint/FSDP lifecycle because core Module recursion expects migrated
    TorchTitan `Module` children.
- `state_dict_adapter.py`
  - Maps BF16 HF keys under `model.language_model.layers.N.*`.
  - Groups per-expert HF gate/up/down weights into native expert tensors and
    expands them again on `to_hf`.
  - Reshapes `dt_bias` between HF flat and native `[heads, head_dim]`.
  - Explicitly skips `weight_scale_inv`, vision tower/projector keys, and MTP
    keys with recorded skip counts.
- `sharding.py`
  - Validates unsupported TP, CP, and PP.
  - Allows FSDP and validates expert count divisibility for EP, while
    documenting that token-dispatch EP is not implemented in this seam.
- `config_registry.py`
  - Adds `glm53_flash_debugmodel`, `glm53_flash_debugmodel_lora`,
    `glm53_flash_mini_countdown_lora`, and `glm53_flash_countdown_lora`.
  - Uses deterministic synthetic pretokenized data for smoke tests; countdown
    token path is wired through env vars as a placeholder for the later data
    ticket.

## Verification

All Python/model/test/training commands below ran inside the rootfs. Training
and Trainer-touching checks used `TORCHTITAN_ROOTFS_ENV=glm53_train` with
`unset CC CXX`.

### Focused tests

Command:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && python -m py_compile torchtitan/experiments/glm53_flash/*.py torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py && pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py -x'
```

Result:

- `11 passed, 14 warnings in 40.45s`
- Warnings were `torch.jit.script_method` deprecation warnings from the
  installed stack.

Tests covered:

- claim_label `deterministic_comparison`: mHC post-mix vs oracle.
- claim_label `deterministic_comparison`: IndexPool vs oracle with document
  boundaries.
- claim_label `deterministic_comparison`: indexer top-k plus causal tail vs
  oracle.
- claim_label `deterministic_comparison`: sparse reference path vs dense
  causal attention when top-k covers all tokens.
- claim_label `deterministic_comparison`: sparse fast CUDA bf16 path vs
  reference path.
- claim_label `deterministic_comparison`: KDA layer vs oracle, including
  low-rank output gate.
- claim_label `deterministic_comparison`: whole debug model forward vs pure
  torch reference oracle.
- claim_label `deterministic_comparison`: finite debug model forward/backward
  on GPU at sequence lengths 512 and 4096.
- claim_label `deterministic_comparison`: meta init -> `to_empty()` ->
  `init_weights()` leaves nonzero norm gains and distinct layer outputs.
- claim_label `deterministic_comparison`: HF state-dict adapter round trip
  using the BF16 index schema and explicit skip-count checks.

### Style and hygiene

Scoped hooks on the new GLM experiment files:

- `ufmt`: passed
- `flake8`: passed
- `pydoclint`: passed
- `codespell`: passed
- `trailing-whitespace`: passed
- `end-of-file-fixer`: passed
- `check-ast`: passed
- `check-merge-conflict`: passed
- `insert-license`: passed

The first pydoclint/pre-commit-hooks attempts failed while the rootfs was
offline and could not resolve PyPI dependencies; rerunning with
`TORCHTITAN_ROOTFS_NETWORK=networked` installed the hook envs and passed.

### B200 attention path probe

Command: short CUDA bf16 probe with `debugmodel().attn_impl = "fast"` for
`GlmSparseMLAAttention` and `GlmKDAAttention`.

Result:

- Sparse MLA path: `attn_gym_gather_attn_fused`
- Sparse output finite after backward: `True`
- KDA path: `attn_gym_kda`
- KDA output finite after backward: `True`

The 2-GPU training smoke configs currently leave sparse MLA at
`attn_impl="reference"` by default. KDA uses the attn-gym CUDA path when
available.

### Training smoke

claim_label `smoke`: `glm53_flash_debugmodel`

Command:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && CUDA_VISIBLE_DEVICES=0,1 GLM53_FLASH_DUMP_FOLDER=.scratch/glm53-flash-sft-ladder/evidence/training/debugmodel MODULE=torchtitan.experiments.glm53_flash.config_registry CONFIG=glm53_flash_debugmodel NGPU=2 LOG_RANK=0 ./run_train.sh 2>&1 | tee .scratch/glm53-flash-sft-ladder/evidence/training/debugmodel-rerun3.log'
```

Result:

- Run ID: `ed11e0fb-03cb-4ad5-94b3-28f6e77e8a22`
- Attempt ID: `890d8de4-92c4-44ed-b6c1-6478ec72feca`
- FSDP applied.
- Full-block activation checkpointing applied to 4 GLM layers.
- Model size: `5,560,798` parameters.
- Trainer initialized with 128 tokens per DP rank, 256 tokens per train step,
  context length 128, 20 steps.
- Loss: step 1 `6.27429`; step 20 `2.18261`.
- Peak rank-0 log memory line: about `0.35GiB`.
- Completed training and destroyed the process group.

claim_label `smoke`: `glm53_flash_debugmodel_lora`

Command:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && CUDA_VISIBLE_DEVICES=0,1 GLM53_FLASH_LORA_DUMP_FOLDER=.scratch/glm53-flash-sft-ladder/evidence/training/debugmodel_lora MODULE=torchtitan.experiments.glm53_flash.config_registry CONFIG=glm53_flash_debugmodel_lora NGPU=2 LOG_RANK=0 ./run_train.sh 2>&1 | tee .scratch/glm53-flash-sft-ladder/evidence/training/debugmodel-lora.log'
```

Result:

- Run ID: `579fc976-65a6-4f43-bb05-ffe7afd51a1b`
- Attempt ID: `7a1c0198-c8a2-4b38-9887-ea7fe3859977`
- FSDP applied.
- Full-block activation checkpointing applied to 4 GLM layers.
- Model size: `5,699,038` parameters.
- Trainer initialized with 128 tokens per DP rank, 256 tokens per train step,
  context length 128, 20 steps.
- Loss: step 1 `6.23592`; step 20 `2.33366`.
- Peak rank-0 log memory line: about `0.11GiB`.
- Completed training and destroyed the process group.

## Known Gaps

- TP is intentionally rejected.
- CP is intentionally rejected; sparse MLA/indexer does not yet partition
  sequence state across context-parallel ranks.
- PP is intentionally rejected for this experiment model.
- EP is validated for expert-count divisibility and routed-expert FSDP sharding,
  but there is no token-dispatch expert-parallel execution in this seam.
- MTP is skipped for SFT and MTP checkpoint keys are ignored by the adapter.
- Vision tower/projector keys are skipped; this is text-only.
- Countdown dataset reading is only env-var-wired placeholder plumbing in this
  ticket; synthetic data drives the smoke configs.
- The training smoke proves launch/plumbing and finite optimization on
  synthetic data only. It does not prove convergence, quality improvement, or
  full-size performance.
- The full released config is represented, but was not launched in this ticket.
- Full-size performance for attn-gym sparse gather was not benchmarked. The
  B200 fast-path probe only proves that the fused path dispatches and produces
  finite gradients on the debug shape.
- FSDP emits a warning that `FSDPGlm53FlashDecoderLayer` returns a view tensor.
  The smoke backward completed, but a landed core-quality version should make
  FSDP outputs non-view explicitly.

## Proposed Core Changes

- Add a core-supported lifecycle for experiment models implemented as plain
  `nn.Module` trees, so they can use Trainer/FSDP without inheriting every leaf
  from TorchTitan `Module`.
- Generalize activation checkpointing to wrap plain transformer blocks that do
  not declare remat regions; the current core `RegionAC` asserts blocks are
  TorchTitan `Module`s.
- Expose an FSDP helper mode for plain parameters on the activated dense/sparse
  meshes, instead of requiring callers to know when to omit
  `DataParallelMeshDims`.
- Add a shared sparse gather attention abstraction that covers arbitrary
  per-query index sets, so GLM sparse MLA does not need to own the attn-gym
  dispatch/fallback locally.
- Consider a common HF MoE expert grouping helper for adapters that need
  per-expert HF keys and grouped native tensors.

## Notes

- Initial smoke attempts exposed three integration gaps and were fixed inside
  the experiment seam: plain-module protocol traversal, activation-checkpoint
  block wrapping, and FSDP DTensor mesh-dims use with plain parameters.
- The working tree already contained unrelated dirty files before this ticket;
  they were not modified or reverted as part of this work.
