# S06 report: Step-3 overfit gate and speedrun arm throughput

Date: 2026-10-14

Surface: repo-local experiment. Changes stayed under
`torchtitan/experiments/glm53_arch_speedrun/` and
`experiments/glm53_arch_speedrun/`. No core Trainer or online-RL path was
changed. I did not start the scheduler and did not commit.

Timing label: `shared_gpu`. All GPU attempts used 1 GPU and the 24 GiB
per-process memory cap. I did not kill or signal foreign processes. Final
throughput numbers were collected while a large unrelated 8-GPU job was active,
so they are smoke/timing evidence, not quiet-GPU performance proof.

## Executive result

S06 passes the Step-3 gate: every B0/A1-A6/G arm overfit the fixed seq-1024 bank
for 300 optimizer steps, every lr=0 control stayed flat at logged precision, and
the learned validation curves are not identical across arms.

The sparse attention and mHC paths were accelerated enough to make the full
seq-1024/4096 smoke matrix finish, but A4, A5, and G remain below the S06
`0.3x B0` threshold at seq 4096. A1 now reaches 0.566x B0 at seq 4096, and A3,
A2, and A6 are comfortably above threshold.

## Implementation

Sparse attention now defaults to a `masked_dense` path:

- `TopKSparseCausalAttention` selects the same top-k/pool keys as before, then
  runs `torch.nn.functional.scaled_dot_product_attention` with a bool
  selected-key plus causal mask.
- The gather and S04 reference paths remain available through
  `sparse_attention_impl`.
- `IndexPool` pooling was vectorized on GPU.
- Indexer pool scoring is chunked by query length, and the same
  `IndexerScoreState` is reused for aux loss and selection to avoid scoring
  pools twice.
- The indexer auxiliary loss path is preserved.

mHC now has local compile regions around the pre and post tensor programs:

- `glm53_speedrun_mhc_pre`
- `glm53_speedrun_mhc_post`

`run_arm.sh` now enforces/records the HDFS-strict input cache when a dataset has
been staged from HDFS or already has a `.staged.json` sidecar. Each attempt now
records `data_manifest_md5` and `data_file_count`.

## Step-3 gate

Complete Step-3 summary:
`experiments/glm53_arch_speedrun/results/s06-step3-summary.json`.

Exit ledger:
`experiments/glm53_arch_speedrun/logs/s06-step3-batch-20261014T170604Z.status`.
All 16 learn/lr0 attempts exited 0.

The complete 16-run Step-3 gate used the original fixed-bank bytes before the
bank was HDFS-published. The bank was then published and strict-staged at:

- URI:
  `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/glm53_arch_speedrun/s06-fixed-bank-data`
- `.staged.json`: `manifest_md5=6b76a4b6a527143e180fcbd073ea78bb`,
  `file_count=2`, `staged_at=2026-10-14T18:21:16Z`
- train/val bins: `md5=9a6b57012a0e9690373c867665e6f6b6`, size `132098`
  bytes each
- fixed bank: 64 sequences x 1024, `num_tokens_with_shift=65537`

| Arm | Learn init CE | Learn final CE | Drop | Learn tok/s | Peak GiB | lr=0 init CE | lr=0 final CE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | 9.7052 | 0.0004 | 99.996% | 100,012 | 6.36 | 10.9746 | 10.9746 | PASS |
| A1 | 9.6155 | 0.0004 | 99.996% | 64,487 | 10.02 | 10.9286 | 10.9286 | PASS |
| A2 | 9.7183 | 0.0025 | 99.974% | 110,962 | 6.26 | 10.9778 | 10.9778 | PASS |
| A3 | 9.4361 | 0.0005 | 99.995% | 65,726 | 7.71 | 10.9968 | 10.9968 | PASS |
| A4 | 10.2354 | 0.0046 | 99.955% | 55,863 | 8.99 | 10.9537 | 10.9537 | PASS |
| A5 | 10.2418 | 0.0054 | 99.947% | 49,878 | 7.18 | 10.9545 | 10.9545 | PASS |
| A6 | 9.7178 | 0.0003 | 99.997% | 96,241 | 11.54 | 10.9746 | 10.9746 | PASS |
| G | 10.1760 | 0.0005 | 99.995% | 37,316 | 16.71 | 11.0293 | 11.0293 | PASS |

Learned validation curves:

| Arm | Validation CE by logged step 1,50,100,150,200,250,300 |
| --- | --- |
| B0 | 9.7052, 0.0938, 0.0432, 0.0096, 0.0045, 0.0019, 0.0004 |
| A1 | 9.6155, 0.0809, 0.0214, 0.0098, 0.0073, 0.0021, 0.0004 |
| A2 | 9.7183, 0.3863, 0.1027, 0.0621, 0.0715, 0.0226, 0.0025 |
| A3 | 9.4361, 0.0535, 0.0322, 0.0191, 0.0123, 0.0059, 0.0005 |
| A4 | 10.2354, 0.1612, 0.0869, 0.0691, 0.0622, 0.0401, 0.0046 |
| A5 | 10.2418, 0.1961, 0.0947, 0.0747, 0.0689, 0.0451, 0.0054 |
| A6 | 9.7178, 0.2663, 0.0021, 0.0025, 0.0018, 0.0005, 0.0003 |
| G | 10.1760, 0.0489, 0.0063, 0.0032, 0.0050, 0.0018, 0.0005 |

Optional HDFS-backed Step-3 rerun evidence was partially collected after the
fixed bank was strict-staged. B0/A1/A2/A3/A6/G learnability completed; A4/A5
learnability and the lr=0 wave were stopped before completion to avoid
duplicating evidence under busy shared GPUs. The complete gate assessment above
therefore uses the original full 16-run Step-3 batch, with the HDFS provenance
correction recorded separately.

## Smoke table

Fresh all-arm two-step smoke summary:
`experiments/glm53_arch_speedrun/results/s06-smoke-summary-20261014T184910Z.json`.

Exit ledger:
`experiments/glm53_arch_speedrun/logs/s06-smoke-batch-20261014T184910Z.status`.
All 16 smoke attempts exited 0.

Main FineWeb input cache:

- URI:
  `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`
- `.staged.json`: `manifest_md5=89e8165a2f2403777b8126b858df6642`,
  `file_count=10`, `staged_at=2026-10-14T18:12:40Z`

| Seq | Arm | Local batch | Grad accum | Val CE | Tokens/s | Peak GiB | Ratio vs B0 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | B0 | 32 | 8 | 9.6331 | 103,313 | 20.37 | 1.000 |
| 1024 | A1 | 16 | 16 | 9.4805 | 54,878 | 18.37 | 0.531 |
| 1024 | A2 | 32 | 8 | 9.6912 | 76,396 | 20.14 | 0.739 |
| 1024 | A3 | 16 | 16 | 9.3418 | 79,418 | 13.68 | 0.769 |
| 1024 | A4 | 16 | 16 | 9.9827 | 11,961 | 17.68 | 0.116 |
| 1024 | A5 | 16 | 16 | 9.8971 | 10,999 | 12.96 | 0.106 |
| 1024 | A6 | 16 | 16 | 9.7120 | 70,246 | 21.29 | 0.680 |
| 1024 | G | 8 | 32 | 9.5710 | 12,952 | 16.75 | 0.125 |
| 4096 | B0 | 8 | 8 | 9.5805 | 99,505 | 20.37 | 1.000 |
| 4096 | A1 | 4 | 16 | 9.3593 | 56,304 | 18.37 | 0.566 |
| 4096 | A2 | 8 | 8 | 9.6579 | 111,758 | 20.14 | 1.123 |
| 4096 | A3 | 4 | 16 | 9.3570 | 77,987 | 13.66 | 0.784 |
| 4096 | A4 | 2 | 32 | 9.8025 | 4,881 | 21.43 | 0.049 |
| 4096 | A5 | 2 | 32 | 9.7615 | 4,731 | 12.25 | 0.048 |
| 4096 | A6 | 4 | 16 | 9.6822 | 143,147 | 23.03 | 1.439 |
| 4096 | G | 2 | 32 | 9.5796 | 13,184 | 17.57 | 0.132 |

## Microbatch and memory notes

A4/A5 seq-1024 smoke could run at local batch 16 after the sparse/indexer work.
Seq-4096 A4/A5/G remained at local batch 2. A4 seq-4096 local-batch probes
showed local batch 2 passing, while local batch 4 and 8 OOMed under the 24 GiB
cap. Typical failures were allocator OOMs with the process already around
23-25 GiB.

The S06 target of `>= 0.5x B0` was met by A1 at seq 1024 and seq 4096. A4/A5
improved enough to finish the matrix but did not approach the target; their
remaining bottleneck is now the dense masked attention shape and indexer
overhead at speedrun scale.

## Verification

Passed:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py torchtitan/experiments/glm53_arch_speedrun/elements/tests/test_elements.py -vv'
```

Result: 30 passed, 14 warnings in 17.96s.

Passed before the final smoke run:

```bash
bash -n experiments/glm53_arch_speedrun/run_arm.sh
```

The focused sparse-attention tests cover forward/backward parity for the
`masked_dense` path against the gather/reference behavior at seq 1024 and 4096,
and the harness test covers the new default mHC compile-region list.

## Gate assessment

Step-3 subclaims:

- B0/A1-A6/G fixed-bank learnability, seq 1024, 300 steps: PASS.
- Final validation CE below 0.5 and more than 90% drop for every arm: PASS.
- lr=0 controls flat at logged precision for every arm: PASS.
- Learned curves are not identical across arms: PASS.
- Fresh B0/A1-A6/G two-step smoke table at seq 1024 and seq 4096: PASS.
- Arms still below 0.3x B0 at seq 4096: A4, A5, G.

S06_STEP3_GATE=PASS
S06_SLOW_ARMS=A4,A5,G
