# K03 report: indexer aux subsampling and local compile regions

Status: COMPLETE for code, focused verification, 2-step/20-step measurement, and
report. PARTIAL for the requested 33.5M-token two-seed CE check: A4 completed
two seeds, A5 completed one seed before a large co-resident GPU process appeared,
and I stopped the remaining long-run loop to avoid misleading evidence. No commit
was made.

Surface: repo-local experiment code under
`torchtitan/experiments/glm53_arch_speedrun/`. I edited only the speedrun
element/config/test files; runner, scheduler, and queue files were not changed.

## Required Context Read

- `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md`
- `.scratch/glm53-flash-sft-ladder/sdd/task-k02-report.md`
- `.scratch/glm53-flash-sft-ladder/sdd/task-k01-report.md`
- `.claude/skills/local_compile_perf/SKILL.md`
- `.scratch/glm53-arch-speedrun/spec.md`
- `torchtitan/experiments/glm53_arch_speedrun/elements/attention.py`
- `torchtitan/experiments/glm53_arch_speedrun/model.py`

## What Changed

New config fields, all defaulting to current behavior:

- `indexer_aux_query_stride: int = 1`
- `indexer_aux_query_seed: int = 0`
- `indexer_aux_compile: bool = False`

New env overrides:

- `GLM53_SPEEDRUN_INDEXER_AUX_QUERY_STRIDE`
- `GLM53_SPEEDRUN_INDEXER_AUX_QUERY_SEED`
- `GLM53_SPEEDRUN_INDEXER_AUX_COMPILE`

New opt-in local compile regions:

- `glm53_speedrun_indexer_score`: indexer projections, `IndexPool`, and score
  chunk reductions.
- `glm53_speedrun_indexer_select`: top-k pool selection and pool-token expansion.
- `glm53_speedrun_indexer_aux`: sampled dense-teacher aux loss.

`indexer_aux_query_stride=1` still routes through the K02 legacy aux path unless
`indexer_aux_compile=True` explicitly selects the dense-teacher helper. For
`stride > 1`, sampled rows use a deterministic seeded offset
`(indexer_aux_query_seed + train_step) % stride` and scale sampled loss sums by
`stride / full_valid_row_count`. Averaged over offsets, this is an unbiased
estimator of the original mean-over-rows aux loss and its gradients.

The dense-teacher helper reuses the already-projected attention `q/k` tensors and
computes sampled teacher rows as one `[B, H, Q, L]` matmul plus `logsumexp`. For
`target="selected"` it preserves the legacy selected-token softmax semantics by
normalizing over the selected pool tokens; for `target="all"` it uses the full
dense causal row distribution.

## Correctness Verification

Passed after formatting:

```bash
CUDA_VISIBLE_DEVICES=6 TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py::test_config_contract_uses_single_env_path torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py::test_config_contract_applies_arm_and_element_env torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py::test_config_contract_exposes_indexer_aux_sampling_and_compile_env --tb=short'
```

Result: 29 passed, 14 warnings.

New tests cover:

- dense-teacher path matches the legacy full-row aux loss and gradients for both
  `target="selected"` and `target="all"`;
- stride sampling matches full-loss and gradient expectation when averaging over
  all offsets;
- compose/env wiring exposes the new dose and compile knobs.

Tiny opt-in compiled fwd/bwd smoke also passed on GPU 6 with
`glm53_speedrun_indexer_score`, `glm53_speedrun_indexer_select`, and
`glm53_speedrun_indexer_aux` enabled.

Passed relevant pre-commit hooks on changed Python files:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && pre-commit run flake8 --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py torchtitan/experiments/glm53_arch_speedrun/elements/config.py torchtitan/experiments/glm53_arch_speedrun/config_registry.py torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py && pre-commit run ufmt --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py torchtitan/experiments/glm53_arch_speedrun/elements/config.py torchtitan/experiments/glm53_arch_speedrun/config_registry.py torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py && pre-commit run pydoclint --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py torchtitan/experiments/glm53_arch_speedrun/elements/config.py torchtitan/experiments/glm53_arch_speedrun/config_registry.py torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py && pre-commit run codespell --files torchtitan/experiments/glm53_arch_speedrun/elements/attention.py torchtitan/experiments/glm53_arch_speedrun/elements/config.py torchtitan/experiments/glm53_arch_speedrun/config_registry.py torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py'
```

Result: flake8, ufmt, pydoclint, and codespell passed.

## Runner Setup

All runner measurements used the real `experiments/glm53_arch_speedrun/run_arm.sh`
path with:

- one visible GPU, 24 GiB cap;
- `seq_len=4096`, local batch size 2, 8192 tokens/step;
- `GLM53_SPEEDRUN_COMPILE=1`;
- `GLM53_SPEEDRUN_INDEXER_AUX_COMPILE=1`;
- `sparse_attention_impl=masked_dense`;
- TensorBoard disabled;
- HDFS strict input cache.

Input:

- URI:
  `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`
- manifest MD5: `89e8165a2f2403777b8126b858df6642`
- file count: 10

## 2-Step Smokes

Environment label: `k03_gpu6_shared_host` for these short runs.

| Arm | Stride | Run ID | Train tok/s | Peak GiB | Val CE |
| --- | ---: | --- | ---: | ---: | ---: |
| A4 | 1 | `k03-a4-stride1-compile-seq4096-2step-20261014T204011Z` | 23,486 | 19.17 | 9.5866 |
| A4 | 8 | `k03-a4-stride8-compile-seq4096-2step-20261014T204226Z` | 19,952 | 18.39 | 9.5809 |
| A4 | 16 | `k03-a4-stride16-compile-seq4096-2step-20261014T204300Z` | 20,811 | 18.31 | 9.5953 |
| A5 | 1 | `k03-a5-stride1-compile-seq4096-2step-20261014T204330Z` | 20,090 | 11.52 | 9.5167 |
| A5 | 8 | `k03-a5-stride8-compile-seq4096-2step-20261014T204508Z` | 27,162 | 11.40 | 9.5150 |
| A5 | 16 | `k03-a5-stride16-compile-seq4096-2step-20261014T204543Z` | 27,212 | 11.38 | 9.5197 |
| G | 1 | `k03-g-stride1-compile-seq4096-2step-20261014T204619Z` | 26,069 | 17.36 | 9.4572 |
| G | 8 | `k03-g-stride8-compile-seq4096-2step-20261014T204645Z` | 26,028 | 17.26 | 9.4571 |
| G | 16 | `k03-g-stride16-compile-seq4096-2step-20261014T204713Z` | 27,145 | 17.26 | 9.4566 |

## 20-Step Throughput Matrix

Environment label: `k03_gpu6_contended_shared`. B0 baseline for the same context:
`k03-b0-baseline-compile-seq4096-20step-20261014T205318Z`, 72,631 tok/s,
6.28 GiB, val CE 7.5089.

| Arm | Stride | Run ID | Train tok/s | Ratio vs B0 | Peak GiB | Val CE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A4 | 1 | `k03-a4-stride1-compile-seq4096-20step-20261014T204836Z` | 25,827 | 0.356x | 19.17 | 7.5769 |
| A4 | 8 | `k03-a4-stride8-compile-seq4096-20step-20261014T204909Z` | 32,318 | 0.445x | 18.39 | 7.5889 |
| A4 | 16 | `k03-a4-stride16-compile-seq4096-20step-20261014T204936Z` | 33,494 | 0.461x | 18.31 | 7.5847 |
| A5 | 1 | `k03-a5-stride1-compile-seq4096-20step-20261014T205003Z` | 33,403 | 0.460x | 11.53 | 7.5872 |
| A5 | 8 | `k03-a5-stride8-compile-seq4096-20step-20261014T205031Z` | 33,759 | 0.465x | 11.40 | 7.5644 |
| A5 | 16 | `k03-a5-stride16-compile-seq4096-20step-20261014T205058Z` | 33,609 | 0.463x | 11.38 | 7.5699 |
| G | 1 | `k03-g-stride1-compile-seq4096-20step-20261014T205125Z` | 32,124 | 0.442x | 17.36 | 7.3436 |
| G | 8 | `k03-g-stride8-compile-seq4096-20step-20261014T205203Z` | 31,750 | 0.437x | 17.30 | 7.3496 |
| G | 16 | `k03-g-stride16-compile-seq4096-20step-20261014T205236Z` | 31,335 | 0.431x | 17.30 | 7.3424 |

Target status at declared best strides:

- A4 best measured stride is 16: 0.461x B0, target 0.5x -- NOT MET.
- A5 best measured stride is 8: 0.465x B0, target 0.5x -- NOT MET.
- G best measured stride is 1 in the 20-step matrix: 0.442x B0, target 0.3x -- MET.

Compared with K02 post-change 2-step smokes (A4 4,173 tok/s, A5 5,635 tok/s,
G 11,373 tok/s), the opt-in compile + dense-teacher/subsample path is a large
end-to-end improvement. It still leaves A4/A5 short of the requested 0.5x B0
bar.

## 33.5M-Token CE Evidence

Declared long-run strides from the 20-step matrix:

- A4: stride 16
- A5: stride 8
- G: stride 1

Completed rows:

| Arm | Stride | Seed | Run ID | Train tok/s | Peak GiB | Val CE | Notes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| A4 | 16 | 0 | `k03-a4-stride16-compile-seq4096-33m-seed0-20261014T205542Z` | 34,074 | 18.32 | 6.8011 | clean single K03 process on GPU 6 |
| A4 | 16 | 1 | `k03-a4-stride16-compile-seq4096-33m-seed1-20261014T211244Z` | 32,345 | 18.32 | 6.8101 | clean single K03 process on GPU 6 |
| A5 | 8 | 0 | `k03-a5-stride8-compile-seq4096-33m-seed0-20261014T213025Z` | 25,359 | 11.40 | 6.7868 | contaminated after step ~2304 by a co-resident 88 GiB Python process on GPU 6 |

A4 two-seed mean val CE: 6.8056. A5 seed 0 suggests no immediate CE regression
from stride 8, but its throughput row should not be used as exclusive-GPU
evidence because the GPU became shared mid-run. I stopped the remaining A5/G
long-run loop with Ctrl-C after that contamination instead of recording mislabeled
exclusive measurements.

## Reusable Kernel Check and K04 Spec

Quick checkout search found no DeepGEMM `fp8_mqa_logits` or DSA indexer-logits
kernel available in this tree. FlashMLA is referenced in GLM research docs as an
inference-side sparse MLA direction; it is not wired here as a training fwd/bwd
indexer-logits or aux-teacher op for the speedrun path.

K04 should target a fused selected-teacher/indexer op rather than sparse
attention:

- Input shapes at seq4096:
  - A4: `B=2`, `L=4096`, `H=6`, `K=128`, `pool_size=1`,
    `Psel=top_k=512`.
  - A5/G: `B=2`, `L=4096`, `H=6`, `K=128`, `pool_size=4`, `Psel=128`.
- Forward should stream sampled query rows, selected pool ids, and pool token
  ids; compute the selected-token teacher logsumexp/softmax in registers; reduce
  head and token mass into per-pool targets; compute selected-logit KL; and
  return `loss_sum` plus `row_count`.
- Backward only needs gradients for the selected entries of `scores_BLP`:
  `(softmax(candidate_logits) - target) * loss_scale / full_row_count`, scatter
  added into the original score tensor. Teacher `q/k`, top-k ids, and pool
  indices remain detached/non-differentiable.
- The same op can optionally expose score logits for profiling, but the main
  payoff is avoiding materialized `[B,H,Q,Psel,R]` or `[B,H,Q,L]` intermediates
  and reducing launch count beyond what local compile can fuse.

## Skill Feedback

The local-compile workflow was useful for gating and opt-in regions. The main
gap for this ticket is that local compile can make the PyTorch path viable, but
does not solve the algorithmic tensor volume of the selected-teacher KL. The
next practical step is a custom fused aux-teacher backward op, not more Python
reshaping.

K03_THROUGHPUT_TARGET_A4=NOT_MET
K03_THROUGHPUT_TARGET_A5=NOT_MET
K03_THROUGHPUT_TARGET_G=MET
K03_PATCH_VERIFIED=PASS
