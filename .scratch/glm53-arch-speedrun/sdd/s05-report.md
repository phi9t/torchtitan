# S05 report: B0 calibration, GLM element wiring, and Step-3 gate status

Date: 2026-10-14

Surface: repo-local experiment. The implementation stayed under
`torchtitan/experiments/glm53_arch_speedrun/` and `experiments/glm53_arch_speedrun/`.
No core Trainer or online-RL path was changed. I did not start the scheduler and did
not commit.

Timing label: `shared_gpu`. All GPU attempts used 1 GPU and the 24 GiB
per-process memory cap. I did not kill or signal foreign processes.

## Executive result

S05 completed the B0 recalibration, generated queue budget update, model wiring,
optimizer-partition coverage, and B0/A1-A6/G two-step smoke probes at seq 1024 and
4096 under the 24 GiB cap.

The Step-3 gate is still `FAIL`: the required fixed-bank overfit CE collapse,
lr=0 flatness proof, and real curve divergence across arms were not completed.
The smoke probes prove build/run/memory plumbing only.

## Budget declaration

Declared per-run budget: 268,435,456 tokens.

- Tokens per optimizer step: 262,144.
- Steps per full run: 1024.
- B0 seq 1024 default: local batch 32, microbatch tokens 32,768, grad accumulation 8.
- B0 seq 4096 default: local batch 8, microbatch tokens 32,768, grad accumulation 8.
- Expensive-arm default microbatch policy:
  - one expensive element: 16,384 microbatch tokens;
  - sparse/indexed or composed G arm: 8,192 microbatch tokens;
  - tokens per step remains 262,144 through higher gradient accumulation.

The earlier S01 baseline ran only 8.4M tokens and stopped at validation CE ~6.96.
The S05 budget moves B0 well past that early regime; the final logged validation
CE at step 1000 was ~3.90.

## Systems checks

The saved B0 calibration config confirms the intended speedrun settings:

- `training.dtype=bfloat16`.
- `model.local_compile_regions=["glm53_speedrun_attention", "glm53_speedrun_mlp", "loss"]`.
- `loss.num_chunks=8`, using `ChunkedLossWrapper` over `CrossEntropyLoss`.
- Dense softmax attention goes through `torch.nn.functional.scaled_dot_product_attention`,
  allowing the CUDA SDPA/flash backend to select the kernel.

The quiet-GPU B0 throughput target was >= 200k tokens/s/GPU at seq 1024. The
two-step smoke B0 run reached 224,158 tokens/s at seq 1024 and 216,840 tokens/s
at seq 4096. Full calibration runs were later affected by a foreign 8-GPU job,
so those wall-clock values are `shared_gpu` only.

## B0 calibration runs

All four calibration attempts exited 0 and reached the full 1024 optimizer-step
target. Metrics are logged every 50 training steps and validation every 100
steps, so the last scalar row is step 1000 / 262,144,000 logged tokens, while
structured logs and process exit status show step 1024 completed.

| Run | Seq | Seed | Final logged train loss | Final logged val CE | Tokens/s | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s05-cal-b0-seq4096-seed0-20261014T161736Z` | 4096 | 0 | 3.96257 | 3.9020 | 144,319 | 20.37 |
| `s05-cal-b0-seq4096-seed1-20261014T161736Z` | 4096 | 1 | 3.96176 | 3.9013 | 146,027 | 20.37 |
| `s05-cal-b0-seq4096-seed2-20261014T161736Z` | 4096 | 2 | 3.96619 | 3.9050 | 109,137 | 20.37 |
| `s05-cal-b0-seq1024-seed0-20261014T161736Z` | 1024 | 0 | 3.96221 | 3.9118 | 108,872 | 20.37 |

Seq-4096 B0 validation CE declaration for matrix comparisons:

- Mean: 3.902776002883911.
- Sample std: 0.0019401469024924146.
- Population std: 0.0015841233123825747.
- Pre-registered effect threshold: `2 * sample_std = 0.003880293804984829`.

Validation curves:

| Step | Seq1024 seed0 | Seq4096 seed0 | Seq4096 seed1 | Seq4096 seed2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.7990 | 10.7879 | 10.7744 | 10.7441 |
| 100 | 5.9173 | 6.0468 | 6.0633 | 6.0556 |
| 200 | 5.1514 | 5.2954 | 5.3025 | 5.2899 |
| 300 | 4.7920 | 4.8934 | 4.8922 | 4.8867 |
| 400 | 4.5185 | 4.6369 | 4.6327 | 4.6163 |
| 500 | 4.3382 | 4.4041 | 4.3986 | 4.3885 |
| 600 | 4.2248 | 4.2562 | 4.2526 | 4.2497 |
| 700 | 4.1491 | 4.1624 | 4.1616 | 4.1617 |
| 800 | 4.0822 | 4.0853 | 4.0844 | 4.0859 |
| 900 | 3.9913 | 3.9853 | 3.9840 | 3.9881 |
| 1000 | 3.9118 | 3.9020 | 3.9013 | 3.9050 |

## GLM element wiring

The speedrun model now wires the requested F1-F5 elements through config fields:

- A1 mHC residual: residual wrapper around attention/MLP blocks with Sinkhorn
  combination and BF16-safe dtype handling.
- A2 NoPE: RoPE can be disabled through `model.elements.rope_enabled`.
- A3 KDA 3:1 hybrid: KDA layers plus every fourth softmax fallback layer.
- A4 indexer top-k sparse attention: indexer scores select sparse keys with an
  auxiliary loss injected into the training loss and logged separately.
- A5 indexer + IndexPool: pool size 4 with the same indexer aux-loss path.
- A6 MTP: one MTP auxiliary head injected through TorchTitan aux-loss plumbing
  and logged separately.
- G: A1 + A2 + A3 + A5 + A6.

Muon/AdamW partition coverage was expanded so Muon owns the block projection
matrices, including KDA `attention.o_proj.weight`; AdamW catches embeddings,
heads, norms, mHC, indexer, MTP, gates, and scalar parameters. The focused tests
cover every direct arm and a toy F1-F5 factorial build/run/backward path.

## Queue update

`experiments/glm53_arch_speedrun/queue/generate_cells.py` regenerated the queue
with the S05 budget.

- `experiments/glm53_arch_speedrun/queue/summary.json` reports 93 runnable cells.
- Slice counts remain 4a=30, 4b=15, 4c=48.
- Deferred templates remain 36 for 4d and 3 for C1.
- The 4c model matrix remains rank 21/21.
- Every runnable cell has top-level `token_budget=268435456`.
- Every runnable cell has `config_overrides["training.tokens_per_step"]=262144`
  and `config_overrides["training.token_budget"]=268435456`.

I did not launch any scheduler cell.

## B0/A1-A6/G smoke probes

These are two-step build/run probes, not Step-3 proof runs.

| Arm | Element | Seq | Local batch | Microbatch tokens | Grad accum | Val CE | Tokens/s | Peak GiB | >2x slower than B0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | baseline | 1024 | 32 | 32768 | 8 | 9.4944 | 224,158 | 20.37 | no |
| A1 | mHC | 1024 | 16 | 16384 | 16 | 9.4790 | 48,299 | 24.00 | yes |
| A2 | NoPE | 1024 | 32 | 32768 | 8 | 9.6880 | 156,191 | 20.14 | no |
| A3 | KDA 3:1 | 1024 | 16 | 16384 | 16 | 9.3417 | 110,244 | 13.68 | yes |
| A4 | indexer top-k | 1024 | 16 | 16384 | 16 | 9.9780 | 12,769 | 16.43 | yes |
| A5 | indexer + IndexPool | 1024 | 8 | 8192 | 32 | 10.0322 | 7,239 | 6.86 | yes |
| A6 | MTP | 1024 | 16 | 16384 | 16 | 9.7125 | 151,643 | 21.29 | no |
| G | GLM-like F1-F5 | 1024 | 8 | 8192 | 32 | 9.5711 | 15,189 | 19.45 | yes |
| B0 | baseline | 4096 | 8 | 32768 | 8 | 9.4048 | 216,840 | 20.37 | no |
| A1 | mHC | 4096 | 4 | 16384 | 16 | 9.2525 | 20,944 | 24.00 | yes |
| A2 | NoPE | 4096 | 8 | 32768 | 8 | 9.6554 | 172,511 | 20.14 | no |
| A3 | KDA 3:1 | 4096 | 4 | 16384 | 16 | 9.3582 | 135,006 | 13.66 | no |
| A4 | indexer top-k | 4096 | 2 | 8192 | 32 | 9.6760 | 2,384 | 19.73 | yes |
| A5 | indexer + IndexPool | 4096 | 2 | 8192 | 32 | 9.7279 | 2,256 | 9.38 | yes |
| A6 | MTP | 4096 | 4 | 16384 | 16 | 9.6822 | 142,903 | 23.03 | no |
| G | GLM-like F1-F5 | 4096 | 2 | 8192 | 32 | 9.5772 | 7,032 | 20.08 | yes |

Run IDs:

- `s05-smoke-b0-seq1024-20261014T154519Z`
- `s05-smoke-b0-seq4096-20261014T154414Z`
- `s05-smoke-a1-seq1024-20261014T155429Z`
- `s05-smoke-a1-seq4096-20261014T155212Z`
- `s05-smoke-a2-seq1024-20261014T155459Z`
- `s05-smoke-a2-seq4096-20261014T155519Z`
- `s05-smoke-a3-seq1024-20261014T155932Z`
- `s05-smoke-a3-seq4096-20261014T160047Z`
- `s05-smoke-a4-seq1024-20261014T160110Z`
- `s05-smoke-a4-seq4096-20261014T160321Z`
- `s05-smoke-a5-seq1024-20261014T160740Z`
- `s05-smoke-a5-seq4096-20261014T160919Z`
- `s05-smoke-a6-seq1024-20261014T161342Z`
- `s05-smoke-a6-seq4096-20261014T161402Z`
- `s05-smoke-g-seq1024-20261014T161423Z`
- `s05-smoke-g-seq4096-20261014T161523Z`

## Verification

Passed:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py -vv'
```

Result: 28 passed, 14 warnings in 15.05s.

Passed:

```bash
bash -n experiments/glm53_arch_speedrun/run_arm.sh experiments/glm53_arch_speedrun/scheduler.sh
```

Queue summary check:

```text
runnable_cells=93
deferred_4d_templates=36
deferred_c1_templates=3
four_c_generator_attempts=828
four_c_model_columns=21
four_c_model_rank=21
slice_counts: 4a=30, 4b=15, 4c=48
```

## Gate assessment

Step-3 subclaims:

- B0/A1-A6/G build and run 2 steps at seq 1024 and seq 4096 under the cap: PASS.
- Tokens/s and peak memory per arm at seq 1024 and seq 4096: PASS as smoke evidence.
- Arms >2x slower than B0 side table: PASS.
- Fixed-bank overfit CE collapse for every arm: MISSING.
- lr=0 flatness for every arm: MISSING.
- Curves diverge across arms: MISSING.

Because the last three required Step-3 proofs are missing, the scheduler-facing
gate remains failed.

S05_STEP3_GATE=FAIL
S05_BUDGET_TOKENS=268435456
