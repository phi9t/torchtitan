# K01 report: Triton top-k sparse causal attention

Status: COMPLETE, with a performance limitation. No commit was made.

Surface: repo-local experiment code under `torchtitan/experiments/`, not core
training. Python/Torch verification ran inside the `glm53_train` rootfs with
`unset CC CXX`; speedrun smokes used one visible GPU and the runner's HDFS
strict input check.

## Output

Implemented an experiment-owned Triton selected-key sparse causal attention op:

- `torchtitan/experiments/glm53_kernels/sparse_topk_attention/`
  - Triton forward and backward kernels.
  - `torch.library.custom_op` forward/backward wrappers.
  - fake implementations and autograd registration for `torch.compile`.
  - public BLHD and BHLD helpers.
  - numerics and microbenchmark harness.
- `torchtitan/experiments/glm53_arch_speedrun/elements/attention.py`
  - new `sparse_causal_attention_triton(...)` helper.
  - `TopKSparseCausalAttention` branch for
    `sparse_attention_impl="triton"`.
- `torchtitan/experiments/glm53_arch_speedrun/elements/config.py`
  - accepts the new `"triton"` implementation value.
- `experiments/glm53_arch_speedrun/run_arm.sh`
  - run-cell/env wiring records and applies `sparse_attention_impl`.
- `torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py`
  - focused forward/backward/compile parity test.
  - opt-in speedrun module-path test.

The default remains `sparse_attention_impl="masked_dense"`.

## Upstream Reuse Check

Checked first, as requested:

| Source | Revision | Decision |
| --- | ---: | --- |
| `deepseek-ai/FlashMLA` | `2e5429f` | Not reused. Sparse prefill path is forward/inference-oriented and shape-constrained around DeepSeek MLA details rather than this BLHD/BHLD training op with backward. |
| `vllm-project/vllm` | `9b5c670` | Not reused. Current sparse/indexer code found in the checkout is ROCm/AITER or inference/paged-attention oriented; it is not an NVIDIA sm100 training fwd+bwd selected-key attention op. |
| `pytorch-labs/attention-gym` | `4d3bdcf` | Not reused. `gather_attn` has different semantics from the speedrun oracle: duplicate index multiplicity and optional local-window behavior rather than the de-duplicated selected causal key set. |
| `pytorch/pytorch` flex attention | `346e94e` | Not reused. `flex_attention` has autograd and block-mask machinery, but not the per-query selected-index gather interface and granularity semantics required here. |

## Numerics

Evidence:

- `.scratch/glm53-flash-sft-ladder/evidence/k01/numerics_full.jsonl`

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && mkdir -p .scratch/glm53-flash-sft-ladder/evidence/k01 && python -m torchtitan.experiments.glm53_kernels.sparse_topk_attention.harness --mode numerics --device cuda:0 --seq-len 1024 4096 --k-denominator 16 8 4 --index-granularity 1 4 --batch-size 1 --num-heads 6 --head-dim 128 --memory-cap-gib 24 > .scratch/glm53-flash-sft-ladder/evidence/k01/numerics_full.jsonl'
```

Result: 48/48 rows passed and 48/48 verifier gates were valid.

Worst target-error / threshold ratios:

| Tensor | Worst ratio | Shape |
| --- | ---: | --- |
| `out` | 0.3333 | L=4096, K=1024, g=4 |
| `dq` | 0.4033 | L=1024, K=256, g=4 |
| `dk` | 0.3873 | L=1024, K=128, g=4 |
| `dv` | 0.3138 | L=1024, K=64, g=4 |

## Microbenchmark

Evidence:

- `.scratch/glm53-flash-sft-ladder/evidence/k01/bench_full.jsonl`

Command:

```bash
CUDA_VISIBLE_DEVICES=1 TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && python -m torchtitan.experiments.glm53_kernels.sparse_topk_attention.harness --mode bench --device cuda:0 --seq-len 1024 4096 --k-denominator 16 8 4 --index-granularity 1 4 --batch-size 1 --num-heads 6 --head-dim 128 --warmup 1 --iters 3 --memory-cap-gib 24 > .scratch/glm53-flash-sft-ladder/evidence/k01/bench_full.jsonl'
```

Important limitation: the conservative one-program-per-query-row Triton kernel
is correctness-solid but does not yet solve the seq4096 performance problem.
At seq4096 it is slower than this checkout's masked-dense baseline.

| L | K | g | Triton ms | Masked dense ms | Dense flash SDPA ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 256 | 1 | 5.167 | 2.464 | 0.989 |
| 4096 | 256 | 4 | 5.527 | 2.473 | 0.922 |
| 4096 | 512 | 1 | 11.425 | 2.471 | 0.976 |
| 4096 | 512 | 4 | 9.021 | 2.480 | 0.996 |
| 4096 | 1024 | 1 | 20.885 | 2.380 | 0.348 |
| 4096 | 1024 | 4 | 18.947 | 2.392 | 0.996 |

## Speedrun Smokes

Both smokes ran through `experiments/glm53_arch_speedrun/run_arm.sh` with
`GLM53_SPEEDRUN_SPARSE_ATTENTION_IMPL=triton`, seq4096, two train steps,
`GLM53_SPEEDRUN_TOKENS_PER_STEP=8192`, TensorBoard off, compile off, one GPU,
and `GLM53_SPEEDRUN_GPU_MEMORY_CAP_GIB=24`.

The runner verified strict HDFS input cache:

- source URI:
  `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`
- manifest MD5: `89e8165a2f2403777b8126b858df6642`
- file count: 10

| Arm | Run ID | Top-k | Granularity | Train tps | Peak GiB | Final train loss | Val tps | Final val CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 | `k01-a4-triton-seq4096-2step-rerun-20261014T192335Z` | 512 | 1 | 4,625 | 20.91 | 10.11571 | 7,597 | 10.1748 |
| A5 | `k01-a5-triton-seq4096-2step-20261014T192427Z` | 512 | 4 | 4,504 | 11.46 | 10.10597 | 7,415 | 10.1463 |

The earlier A4 attempt
`k01-a4-triton-seq4096-2step-20261014T191640Z` produced a partial structured
log through step 1 eval start but no final `metrics.json`; it was not counted
as successful evidence. The rerun above completed and published successfully.

## Verification

Executed and passed:

```bash
CUDA_VISIBLE_DEVICES=5 TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py::test_sparse_triton_matches_masked_dense_for_topk_fwd_bwd_and_compile torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py::test_sparse_attention_triton_path_is_opt_in'
```

Result:

```text
2 passed, 14 warnings in 7.73s
```

Also passed:

- full numerics sweep: 48/48 rows passed.
- full benchmark sweep: completed and recorded in JSONL.
- A4 seq4096 2-step speedrun smoke.
- A5 seq4096 2-step speedrun smoke.

## Remaining Risk

The current implementation is a correct training-capable baseline, not the final
performance kernel. The next optimization target is reducing per-query overhead
and backward atomic pressure at seq4096/K=512, especially for the `g=1` path.
