# T07b report: rung 3 mini GLM hybrid learnability

## Summary

Executed T07b as `learnability_smoke` for the GLM-5.3-Flash mini hybrid seam. The mini full-training Countdown overfit gate passes for both the default sparse-indexer path and the dense-causal sparse-layer ablation. Both lr=0 controls stay flat. Fast-vs-reference debug curves agree closely but are not bitwise identical. Seq 4096/8192 fast-path FullAC probes produced finite gradients under the memory guardrail. Artifacts were published to HDFS.

No commit was made.

## Code/config changes

- `torchtitan/experiments/glm53_flash/config_registry.py`
  - Wired `glm53_flash_mini_countdown_lora` to real Countdown chat data through `ChatProcessor`, `FirstFitPackingConfig`, `GrainDataLoader`, and a Qwen tokenizer path.
  - Added `glm53_flash_mini_countdown_full` for full random-init training.
  - Defaulted mini/debug GLM attention to `GLM53_FLASH_ATTN_IMPL=fast`, with `reference` selectable by env.
  - Added env knobs for Countdown paths, vocab, mini sequence length, tokens per step, local batch, steps, lr, and chunked loss.
- `torchtitan/experiments/glm53_flash/flavors.py`
  - Added validated `sparse_attention_mode` values: `indexer` and `dense_causal`.
- `torchtitan/experiments/glm53_flash/model.py`
  - Added the dense-causal sparse-layer ablation path.
  - Accepted flat packed token vectors by treating `[L]` as `[1, L]` internally and returning flat logits.
- `torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py`
  - Added coverage for the dense-causal ablation path, mini Countdown config contract, and flat packed token forward.

## Configs used

All Trainer/model/test runs used the rootfs training env:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && ...'
```

Shared environment:

- `GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH=/workspace/torchtitan/.scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/Qwen3-0.6B-tokenizer`
- `GLM53_FLASH_COUNTDOWN_DATA_FILE=/workspace/torchtitan/.scratch/glm53-flash-sft-ladder/evidence/t07b/inputs/countdown-sft-v1/train/formatting.jsonl`
- `GLM53_FLASH_ATTN_IMPL=fast`
- `GLM53_FLASH_MINI_SEQ_LEN=512` for overfit gates
- `GLM53_FLASH_MINI_TOKENS_PER_STEP=32768`
- `GLM53_FLASH_LOSS_CHUNKS=8`

Overfit runs:

- Indexer train: `GLM53_FLASH_SPARSE_ATTENTION_MODE=indexer`, local batch 32, lr 1e-3, 300 steps, 2 GPUs.
- Indexer lr=0 control: same as indexer train with `GLM53_FLASH_MINI_LR=0`.
- Dense-causal train: `GLM53_FLASH_SPARSE_ATTENTION_MODE=dense_causal`, local batch 8, grad accumulation from 32768 tokens/step, lr 1e-3, 300 steps, 2 GPUs.
- Dense-causal lr=0 control: same as dense-causal train with `GLM53_FLASH_MINI_LR=0`.

The first dense-causal overfit attempt with local batch 32 was aborted as diagnostic evidence because rank-0 memory reached about 127 GiB, above the ticket's 60 GiB/GPU guardrail. The accepted dense-causal runs used local batch 8.

## Fast/reference debug comparison

Debug model, same seed, 20 steps:

| Step | fast loss | reference loss | abs delta |
| ---: | ---: | ---: | ---: |
| 1 | 6.27455 | 6.27429 | 0.00026 |
| 5 | 4.67531 | 4.66783 | 0.00748 |
| 10 | 4.95564 | 4.97449 | 0.01885 |
| 20 | 2.22864 | 2.18688 | 0.04176 |

Gate: pass, close but not bitwise identical.

## Overfit gate

Claim label: `learnability_smoke`.

| Run | Step 1 | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Peak GiB | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| indexer train | 12.19578 | 0.01419 | 0.00364 | 0.00202 | 0.01101 | 0.01148 | 0.00830 | 30.12 | pass |
| indexer lr=0 | 12.19578 | 12.19578 | 12.19578 | 12.19578 | 12.19578 | 12.19578 | 12.19578 | 27.70 | pass flat |
| dense-causal train | 12.19885 | 0.01071 | 0.02082 | 0.00645 | 0.01738 | 0.01070 | 0.00114 | 26.80 | pass |
| dense-causal lr=0 | 12.19885 | 12.20440 | 12.20426 | 12.20580 | 12.20624 | 12.20634 | 12.20374 | 27.15 | pass flat |

Both learning runs collapse CE below 0.1 and by more than 99.9%. The lr=0 controls stay flat. The indexer and dense-causal learning curves are not identical, satisfying the Falcon silent-collapse rule.

For the accepted dense-causal lr=0 control, the parent launcher log is `overfit_dense_lr0/train.log`; torchrun rank streams are preserved as `overfit_dense_lr0/train.rank0.stdout.log` and `train.rank1.stdout.log`. The metric source for that run is rank-0 stdout.

## Seq probes

Mini flavor, fast path, FullAC, finite-grad probes. Throughput label: `exclusive_foreign_free_shared_host`, not performance evidence.

| Probe | Steps | Step 1 loss | Final loss | Peak GiB | Mean tps after step 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| seq 4096 | 5 | 12.19462 | 2.92361 | 21.08 | 2130.0 |
| seq 8192 | 5 | 12.19578 | 2.97161 | 22.61 | 3894.5 |

Gate: pass finite loss/grad and memory under 60 GiB/GPU.

## Full-size memory plan

Computed without GLM weights by building/counting the full `glm53_flash` flavor and rank-16 LoRA variant. The current experiment seam adds LoRA modules but does not freeze every non-LoRA parameter, so FSDP-only LoRA memory is analytical rather than a faithful fake-backend launch.

Counts:

- Base total params: `313,326,799,870`
- Rank-16 LoRA adapter params: `67,190,784`
- Largest FSDP unit: layer `4`, `7,414,112,502` params
- Largest unit bf16 all-gather: `13.8098606877 GiB`

Analytical floor, excluding activations, NCCL, allocator fragmentation, dataloader, and checkpoints:

| World size | base bf16 shard GiB | LoRA param/grad/Adam shard GiB | static floor GiB | + largest all-gather GiB |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 145.90416 | 0.18773 | 146.09189 | 159.90175 |
| 8 | 72.95208 | 0.09386 | 73.04594 | 86.85581 |

Conclusion: 4x B200 is not viable under the current 60 GiB/GPU shared-run guardrail and is tight even before activations on 179 GiB cards. 8x B200 exclusive at about 179 GiB/GPU looks plausible with headroom for activations and runtime overhead, but the current shared state around 95 GiB free/GPU is risky/not enough for a full-size proof after extras.

## Artifacts

Local evidence root:

```text
.scratch/glm53-flash-sft-ladder/evidence/t07b/
```

Published HDFS bundle:

```text
hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-ladder/rung3/t07b-rung3-learnability-smoke-20261014T1159Z
```

Publish result:

- Files: `31`
- Manifest: `MANIFEST.jsonl`
- Completion marker: `_COMPLETE`

Primary local summaries:

- `.scratch/glm53-flash-sft-ladder/evidence/t07b/analysis/all_run_summaries.json`
- `.scratch/glm53-flash-sft-ladder/evidence/t07b/analysis/gate_summary.json`
- `.scratch/glm53-flash-sft-ladder/evidence/t07b/full_size_param_counts.json`

## Verification

Focused tests:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_flash/tests/test_glm53_flash.py -x'
```

Result: `14 passed, 1 skipped, 14 warnings in 8.86s`.

Scoped hooks on the modified GLM experiment files:

- `ufmt`: passed
- `flake8`: passed
- `pydoclint`: passed
- `codespell`: passed
- `trailing-whitespace`: passed
- `end-of-file-fixer`: passed
- `check-ast`: passed
- `check-merge-conflict`: passed

No core edits were made for this ticket. No commit was made.
