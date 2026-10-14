# K02 report: profile-first indexer slow path

Status: COMPLETE for profiling, patch, verification, and report. The K02
throughput target is NOT met yet. No commit was made.

Surface: repo-local experiment code under
`torchtitan/experiments/glm53_arch_speedrun/`. I only edited
`torchtitan/experiments/glm53_arch_speedrun/elements/attention.py`; runner,
scheduler, and queue files were not changed. The file was already dirty from
prior K01/S10 work, including the Triton attention opt-in and behavior logging;
my substantive changes are the gather helper/replacements and aux-loss masked
reduction below.

## Required Context Read

- `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md`
- `.scratch/glm53-flash-sft-ladder/sdd/task-k01-report.md`
- `.scratch/glm53-arch-speedrun/sdd/s06-report.md`
- `.claude/skills/local_compile_perf/SKILL.md`
- `torchtitan/experiments/glm53_arch_speedrun/elements/attention.py`
- `torchtitan/experiments/glm53_arch_speedrun/model.py`

## What Was Changed

`attention.py` now has:

- `_gather_batch_dim1(...)`: replaces advanced indexing of batched token/pool
  selections with `torch.gather` over an expanded view.
- `LightningIndexer.select_from_scores`, `sparse_causal_attention` gather path,
  and `indexer_auxiliary_loss` selected/all target paths use this helper for
  batched gather.
- `indexer_auxiliary_loss` no longer uses Python `if row_mask.any()` or
  boolean compaction (`loss[row_mask]`) in the chunk loop. It computes masked
  `loss_sum` and `row_count` on device and returns `loss_sum / row_count`, with
  the same zero-loss fallback. This removes per-chunk CUDA host syncs while
  preserving mean-over-valid-rows semantics within fp32 reduction-order
  tolerance.

The aux objective and per-layer selection semantics are unchanged.

## Profiling Setup

Profiler evidence used a rootfs Python harness, not host Python:

- env: `TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`
- one visible GPU, 24 GiB cap, seq 4096, batch 2
- one warmup step, three active `torch.profiler` steps
- A4, A5, and G model configs from the speedrun registry
- `_skip_lm_head=True` with a tiny hidden-state loss, because full vocabulary
  CE OOMs under the 24 GiB cap and would hide the indexer/attention region
- indexer auxiliary losses were active
- custom profiler ranges wrapped `score_pools`, `select_from_scores`,
  `IndexPool`, `indexer_auxiliary_loss`, and sparse attention

Trace locations:

| Stage | A4 | A5 | G |
| --- | --- | --- | --- |
| Baseline | `.scratch/glm53-flash-sft-ladder/evidence/k02/profiles/baseline/a4/torch_profiler_trace.json.gz` | `.scratch/glm53-flash-sft-ladder/evidence/k02/profiles/baseline/a5/torch_profiler_trace.json.gz` | `.scratch/glm53-flash-sft-ladder/evidence/k02/profiles/baseline/g/torch_profiler_trace.json.gz` |
| Final after patch | `.scratch/glm53-flash-sft-ladder/evidence/k02/profiles/after_gather_rewrite/a4/torch_profiler_trace.json.gz` | `.scratch/glm53-flash-sft-ladder/evidence/k02/profiles/after_gather_rewrite/a5/torch_profiler_trace.json.gz` | `.scratch/glm53-flash-sft-ladder/evidence/k02/profiles/after_gather_rewrite/g/torch_profiler_trace.json.gz` |

JSON summaries are adjacent to each trace as `summary.json`. An intermediate
`after_masked_reduce/` profile shows the sync removal alone, and
`after_masked_reduce_rerun/a5/` records that the first A5 post-sync profile had
a shared-GPU/profiler outlier.

## Ranked Baseline Findings

Times below are profiler totals over three active model-body fwd/bwd steps.
The CUDA self rows for custom ranges are Kineto nested-event totals; the CPU
totals are the best human-scale ranking for the Python-level regions.

| Category | A4 baseline | A5 baseline | G baseline | Finding |
| --- | ---: | ---: | ---: | --- |
| Aux-loss teacher/KL | 3954.7 ms CPU, 3943.2 ms CUDA self | 3837.4 ms CPU, 3828.1 ms CUDA self | 978.4 ms CPU, 976.0 ms CUDA self | Dominant cost. Current selected-target teacher gathers selected keys, computes per-head softmax over selected tokens, aggregates to pool targets, then KLs against indexer logits. |
| Indexer scoring | 862.1 ms CPU, 857.5 ms CUDA self | 861.8 ms CPU, 853.1 ms CUDA self | 221.4 ms CPU, 218.9 ms CUDA self | Second biggest sparse/indexer item. This is `q_proj/k_proj/gate/pool` plus chunked score einsums. |
| Top-k / select | 20.2 ms CPU | 43.2 ms CPU | 11.1 ms CPU | Not a primary bottleneck. |
| IndexPool | 6.4 ms CPU | 35.0 ms CPU | 9.8 ms CPU | Small after S06 vectorization. |
| Masked dense sparse attention | 34.6 ms CPU | 32.6 ms CPU | 8.2 ms CPU | Confirms K01: attention is not the bottleneck at L=4096. |
| Chunk loops | 64 chunks/layer in aux selected target; 2304 chunk iterations over A4/A5 three-step profile, 576 for G | same | same | The Python chunk loop itself is secondary; its main damage was forcing sync/compaction each chunk. |
| Host syncs | A4: `cudaStreamSynchronize` 6915, `aten::item` 2307 | A5: 6915, 2307 | G: 1731, 579 | Caused by `row_mask.any()` / boolean compaction in aux loss. |
| Recompute under AC | N/A | N/A | N/A | Harness and smokes used compile off and no activation checkpointing for the measured sparse masked-dense path. |
| Compile graph breaks | N/A | N/A | N/A | `GLM53_SPEEDRUN_COMPILE=0`; sparse indexer path is not currently enabled as a local compile region. |

Top raw CUDA operators were consistent with the aux-loss finding:
`aten::index` / `index_elementwise_kernel`, `aten::bmm`, `aten::copy_`,
`aten::gather`, `aten::nonzero`, and many small sync/copy events dominated
outside the custom ranges.

## Profiler Before/After

Same harness, same GPU selection, same seq/batch/steps:

| Arm | Baseline body time | Final body time | Speedup | Baseline body tok/s | Final body tok/s | Peak GiB before -> after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 | 3.016 s | 2.677 s | 1.126x | 2,716 | 3,060 | 19.66 -> 19.66 |
| A5 | 3.044 s | 2.394 s | 1.272x | 2,691 | 3,422 | 9.93 -> 9.93 |
| G | 1.156 s | 1.011 s | 1.143x | 7,089 | 8,102 | 18.10 -> 18.10 |

Focused deltas:

| Arm | Aux-loss CPU total | Score-pools CPU total | Host-sync collapse |
| --- | ---: | ---: | --- |
| A4 | 3954.7 -> 3407.8 ms | 862.1 -> 1222.7 ms | `aten::item` 2307 -> 3, `cudaStreamSynchronize` 6915 -> 3 |
| A5 | 3837.4 -> 3245.7 ms | 861.8 -> 756.5 ms | `aten::item` 2307 -> 3, `cudaStreamSynchronize` 6915 -> 3 |
| G | 978.4 -> 847.2 ms | 221.4 -> 211.1 ms | `aten::item` 579 -> 3, `cudaStreamSynchronize` 1731 -> 3 |

The three remaining `aten::item` / syncs are from the profiler harness loss
capture, not the aux-loss loop.

## Real Runner Smokes

Post-change smokes used the real `run_arm.sh` path, HDFS strict input cache,
one visible GPU, seq4096, two train steps, `GLM53_SPEEDRUN_LOG_FREQ=1`,
`GLM53_SPEEDRUN_COMPILE=0`, tensorboard off, and the default
`sparse_attention_impl=masked_dense`.

Input:

- URI:
  `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`
- manifest MD5: `89e8165a2f2403777b8126b858df6642`
- file count: 10

These runs are labelled `shared_gpu_k02`; all GPUs were busy at high
utilization, so this is functionality/timing evidence, not quiet-GPU
performance proof.

| Arm | Run ID | Step-2 train tok/s | Peak GiB | Final train loss | Val tok/s | Val CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A4 | `k02-a4-after-seq4096-2step-logfreq1-20261014T2013Z` | 4,173 | 21.73 | 10.11294 | 6,631 | 10.1728 |
| A5 | `k02-a5-after-seq4096-2step-logfreq1-20261014T2013Z` | 5,635 | 12.16 | 10.09107 | 7,510 | 10.1397 |
| G | `k02-g-after-seq4096-2step-logfreq1-20261014T2013Z` | 11,373 | 20.28 | 10.29864 | 13,096 | 10.3558 |

Historical pre-K02 S06 default masked-dense seq4096 smokes were A4 4,881 tok/s,
A5 4,731 tok/s, and G 13,184 tok/s, with B0 99,505 tok/s. The post-K02 smokes
do not meet the ticket target:

- A4: 4,173 / 99,505 = 0.042x B0, target 0.5x.
- A5: 5,635 / 99,505 = 0.057x B0, target 0.5x.
- G: 11,373 / 99,505 = 0.114x B0, target 0.3x.

The profiler body speedups are real for the measured hot spot, but they are far
too small to close a 6-12x remaining gap in the end-to-end runner.

## Verification

Passed focused checks after the patch:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc "unset CC CXX; cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py::test_indexer_auxiliary_loss_improves_dense_topk_recall_on_toy_task torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py::test_sparse_attention_exposes_indexer_aux_loss_and_dense_warmup torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py::test_sparse_gather_matches_reference_for_topk_fwd_bwd --tb=short"
```

Result: 4 passed, 14 warnings.

Passed full element suite after formatting:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc "unset CC CXX; cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py"
```

Result: 23 passed, 14 warnings.

Passed relevant pre-commit hooks on the changed file:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc "unset CC CXX; cd /workspace/torchtitan && pre-commit run flake8 --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py && pre-commit run ufmt --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py && pre-commit run pydoclint --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py && pre-commit run codespell --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py"
```

Result: all passed. The first `ufmt` run reformatted `attention.py`; tests were
rerun after formatting.

## K03 Kernel Spec

The remaining bottleneck is the selected-target indexer auxiliary
teacher/KL. A custom fused op is warranted if A4/A5/G need to approach the
target.

### Current semantics to preserve

For each sparse layer and query row:

1. Select `Psel = ceil(top_k / pool_size)` candidate pools from
   `state.scores.topk(...)`.
2. Expand pool tokens from `state.pool_indices[B, P, R]`, mask invalid,
   non-causal, and padded tokens.
3. Compute teacher token scores:
   `scores[B,H,C,P,R] = dot(q_teacher[B,C,H,K], k_teacher[B,token,H,K]) * scale`.
4. Softmax over the flattened selected token dimension `P*R`, independently
   per batch, head, and query.
5. Sum teacher probabilities over heads and tokens-in-pool to get
   `target[B,C,P]`, mask invalid candidates, and renormalize per row.
6. Compute KL from `target[B,C,P]` to `log_softmax(candidate_logits[B,C,P])`.
7. Average over rows where query, candidate, and target mass are valid.
8. Backprop only into `state.scores` / candidate logits. Teacher q/k are
   detached; pool/top-k selection is non-differentiable.

### Proposed fused op

Forward inputs:

- `q_teacher_BLHK`: bf16/fp32, `[B, L, H, K]`
- `k_teacher_BLHK`: bf16/fp32, `[B, L, H, K]`
- `scores_BLP`: fp32, `[B, L, P]`, requires grad
- `candidate_pools_BLC`: int64/int32, `[B, L, Psel]`
- `pool_indices_BPR`: int64/int32, `[B, P, R]`
- `valid_queries_BL`: bool, `[B, L]`
- `valid_candidates_BLP`: bool, `[B, L, P]`
- `valid_tokens_BL`: bool, `[B, L]`
- `scale`: fp32 scalar

Forward outputs:

- `loss_sum`: fp32 scalar
- `row_count`: fp32 scalar
- optional saved or recomputable `candidate_prob_BLC` and `target_BLC`

Backward output:

- `grad_scores_BLP`: fp32, scatter-add into selected candidate pool positions,
  with `grad = (softmax(candidate_logits) - target) / row_count` for valid rows
  and candidates; no grad for q/k/pool indices/masks.

Representative dimensions:

- A4: `B=2`, `L=4096`, `H=6`, `K=128`, `top_k=512`, `R=1`, `Psel=512`
- A5/G sparse layers: `B=2`, `L=4096`, `H=6`, `K=128`, `top_k=512`, `R=4`,
  `Psel=128`

Work estimate per layer:

- Teacher dot products: about `B * L * H * top_k * K`, or roughly 3.2B FMAs at
  the K02 dimensions.
- Current PyTorch path materializes/gathers large
  `[B, chunk, Psel, R, H, K]` tensors and launches many per-chunk bmm/index/copy
  kernels. A fused kernel should stream selected K vectors, compute row max/sum
  softmax reductions, aggregate target pool mass, and emit loss/grad without
  materializing gathered K or `[B,H,C,P,R]` score/prob tensors.

K03 should target this op before spending time on sparse attention kernels:
masked dense attention remains about 25-35 ms CPU total over three profiled
steps for A4/A5, while aux-loss teacher/KL remains 3.2-3.4 seconds.

K02_THROUGHPUT_TARGET=NOT_MET
K02_PATCH_VERIFIED=PASS
