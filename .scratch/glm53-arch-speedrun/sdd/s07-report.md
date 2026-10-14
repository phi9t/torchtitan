# S07 report: architecture ladder L1-L4 at speedrun scale

Date: 2026-10-14

Surface: repo-local experiment. The implementation stayed under
`torchtitan/experiments/glm53_arch_ladder/` and `experiments/glm53_arch_ladder/`.
No core Trainer, online-RL, `torchtitan/models/*`, S06
`torchtitan/experiments/glm53_arch_speedrun/`, or
`torchtitan/experiments/glm53_flash/` path was edited. I did not commit.

Timing label: `shared_gpu`. All launches used one GPU, the 24 GiB per-process
memory cap, and a runner guard limiting S07 to at most three concurrent runs. I
did not kill or signal foreign processes. Later wall-clock numbers are evidence
for this shared host state, not quiet-machine performance.

## Executive result

S07 completed the architecture ladder package, runner, unit tests, two-step
smokes, fixed-bank Step-3 gates, full or declared reduced science runs, HDFS
publishing, and MD5 verification.

At the S05 full budget, L1 Qwen3 dense beat the L0 GPT-2-style baseline:
validation CE 3.8693554401397705 versus L0 3.902776002883911, delta
-0.033420562744140625. This is larger than the preregistered S05 effect
threshold `2 * std = 0.003880293804984829`.

L2 Qwen3-MoE was significantly worse than both L0 and L1 at the full budget:
validation CE 4.044178247451782, delta +0.1414022445678711 versus L0 and
+0.17482280731201172 versus L1.

L3 mini-Kimi-K3 and L4 mini-GLM-5.3-Flash were below the brief's `0.3x L0`
throughput threshold in smoke runs, so they were moved to declared reduced
side-table budgets instead of stalling. L4 also used short validation in the
reduced side table after a 100-step full-validation attempt stalled in
validation under shared-GPU contention. Do not compare the L4 scalar as a
full-budget or full-validation result.

## Implementation

New package:

- `torchtitan/experiments/glm53_arch_ladder/config_registry.py`: four Trainer
  configs with environment overrides and FineWeb or fixed-bank dataloader
  selection.
- `torchtitan/experiments/glm53_arch_ladder/flavors.py`: L1 Qwen3 dense, L2
  Qwen3 MoE, L3 Kimi K3, and L4 GLM-5.3-Flash mini builders using GPT-2 vocab
  size 50257 for all rungs.
- `torchtitan/experiments/glm53_arch_ladder/data.py`: packed FineWeb loaders
  and fixed-bank packed or `[B, L]` loaders for Step-3 gates.
- `torchtitan/experiments/glm53_arch_ladder/params.py`: parameter-count and
  Muon/AdamW partition helpers.
- `torchtitan/experiments/glm53_arch_ladder/inspect.py`: Markdown/JSON
  parameter table emission.
- `torchtitan/experiments/glm53_arch_ladder/extract_metrics.py`: TensorBoard
  metric extraction with `metrics.json` fallback for no-event fixed-bank gates.
- `torchtitan/experiments/glm53_arch_ladder/tests/test_s07_arch_ladder.py`:
  loader, registry, parameter matching, optimizer partition, and CUDA fwd/bwd
  coverage.

New runner:

- `experiments/glm53_arch_ladder/run_rung.sh`: rootfs-aware launcher, GPU
  safety preflight, dataset staging, metadata capture, TensorBoard metrics
  extraction, bundle manifest generation, and HDFS publish.
- `experiments/glm53_arch_ladder/.gitignore`: ignores logs and result bundles.

L4 note: current `Glm53FlashConfig.validate()` rejects `use_mtp=True`, so the
S07 L4 mini flavor uses `use_mtp=False`. This is an upstream hook gap in the GLM
experiment package, not an S07 model edit.

## Parameter matching

Target: match active non-embedding parameters to L0, approximately 85M, within
+/-10%. Active MoE counts discount routed experts by `top_k / num_experts`;
embeddings and LM heads are excluded from the active matching count.

| Rung | Family | Total params | Non-embedding params | Active non-embedding params | Ratio to L0 | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| L1 | Qwen3 dense | 126,703,616 | 88,106,240 | 88,106,240 | 1.037 | PASS |
| L2 | Qwen3 MoE | 363,567,872 | 286,373,120 | 88,192,256 | 1.038 | PASS |
| L3 | Kimi K3 | 206,088,008 | 128,893,256 | 91,144,520 | 1.072 | PASS |
| L4 | GLM-5.3-Flash mini | 257,108,648 | 179,913,896 | 93,209,768 | 1.097 | PASS |

## Gate summary

| Gate | Evidence | Status |
| --- | --- | --- |
| Build/config/unit | Focused test file covers loaders, registry modes, parameter match, optimizer partition, and CUDA fwd/bwd | PASS |
| Smoke | Every rung has a two-step run under the 24 GiB cap; L4 required local batch 1 to avoid OOM under co-run load | PASS |
| Step-3 overfit | Every rung collapses CE on a fixed small bank in 300 steps | PASS |
| Step-3 lr=0 | Every rung stays flat with lr=0 on the same fixed-bank path | PASS |
| HDFS publish | All 20 S07 bundles verified against HDFS `_COMPLETE`, HDFS manifest equality, and local stable-file md5/size | PASS |

## Smoke runs

L0 smoke reference from S05 at seq 4096: 216,840 tokens/s. The brief's reduced
budget threshold is `0.3 * L0 = 65,052` tokens/s.

| Rung | Run | Final train loss | Final val CE | Tokens/s | Peak GiB | Budget decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| L1 | `s07-smoke-l1-20261014T174538Z` | 10.54485 | 9.7513 | 187,164 | 7.44 | full budget |
| L2 | `s07-smoke-l2-20261014T174627Z` | 11.00343 | 10.9229 | 134,770 | 10.42 | full budget |
| L3 | `s07-smoke-l3-20261014T174723Z` | 10.08726 | 9.8648 | 32,819 | 10.16 | reduced side table |
| L4 | `s07-smoke-l4-lb1-20261014T180450Z` | 10.21539 | 9.3051 | 6,361 | 15.70 | reduced side table |

Failed attempt: `s07-smoke-l4-20261014T180405Z` OOMed in
`GlmSparseIndexer.forward` at seq 4096 / local batch 2 under allowed co-run
conditions. It is recorded as failed evidence only; the passing L4 smoke is the
local-batch-1 run above with the same 262,144 tokens per optimizer step via
gradient accumulation 64.

## Step-3 fixed-bank gates

Overfit runs:

| Rung | Run | Final step | Train loss | Val CE | Tokens/s | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L1 | `s07-step3-overfit-l1-seed0-20261014T181754Z` | 300 | 0.00002 | 0 | 35,786 | 3.15 |
| L2 | `s07-step3-overfit-l2-seed0-20261014T182006Z` | 300 | 0.00001 | 0 | 26,893 | 5.01 |
| L3 | `s07-step3-overfit-l3-seed0-20261014T182011Z` | 300 | 0.00005 | 0.0011 | 11,953 | 6.01 |
| L4 | `s07-step3-overfit-l4-seed0-20261014T182016Z` | 300 | 0.00183 | 0.0185 | 4,280 | 16.21 |

lr=0 flatness runs:

| Rung | Run | Final train loss | Final val CE | Tokens/s | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| L1 | `s07-step3-lr0-l1-seed0-20261014T182702Z` | 10.95810 | 10.9581 | 34,938 | 3.15 |
| L2 | `s07-step3-lr0-l2-seed0-20261014T182707Z` | 10.95988 | 10.9605 | 27,876 | 5.01 |
| L3 | `s07-step3-lr0-l3-seed0-20261014T182712Z` | 11.28666 | 11.2888 | 12,029 | 6.01 |
| L4 | `s07-step3-lr0-l4-seed0-20261014T183006Z` | 10.99594 | 10.9953 | 4,250 | 15.51 |

## Full and reduced science runs

Full-budget settings: 268,435,456 tokens declared, 262,144 tokens per optimizer
step, validation at the final logged step 1000, seq 4096, GPT-2 vocab 50257.
L1 and L2 completed this budget. L3 used a declared 26,214,400-token reduced
budget. L4 used a declared 2,621,440-token reduced short-validation budget with
262,144 validation tokens.

| Rung | Seed | Run | Budget tokens | Final val step | Val CE | Tokens/s | Peak GiB | Local batch | Grad accum |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 0 | `s07-full-l1-seed0-20261014T183650Z` | 268,435,456 | 1000 | 3.8704934120178223 | 181,388.890625 | 7.026139736175537 | 4 | 16 |
| L1 | 1 | `s07-full-l1-seed1-20261014T183655Z` | 268,435,456 | 1000 | 3.8682174682617188 | 178,290.671875 | 7.026139736175537 | 4 | 16 |
| L2 | 0 | `s07-full-l2-seed0-20261014T183700Z` | 268,435,456 | 1000 | 4.037964820861816 | 87,303.2265625 | 10.020929336547852 | 4 | 16 |
| L2 | 1 | `s07-full-l2-seed1-20261014T191003Z` | 268,435,456 | 1000 | 4.050391674041748 | 69,072.6484375 | 10.020929336547852 | 4 | 16 |
| L3 | 0 | `s07-reduced-l3-seed0-20261014T191325Z` | 26,214,400 | 100 | 5.4609832763671875 | 28,437.33203125 | 9.69283676147461 | 2 | 32 |
| L3 | 1 | `s07-reduced-l3-seed1-20261014T192736Z` | 26,214,400 | 100 | 5.4653639793396 | 29,545.3828125 | 9.69283676147461 | 2 | 32 |
| L4 | 0 | `s07-reduced-l4-shortval-seed0-20261014T194304Z` | 2,621,440 | 10 | 7.284682273864746 | 7,370.67822265625 | 14.701904296875 | 1 | 64 |
| L4 | 1 | `s07-reduced-l4-shortval-seed1-20261014T194910Z` | 2,621,440 | 10 | 7.269698143005371 | 2,021.2835693359375 | 14.701902389526367 | 1 | 64 |

Group summaries:

| Rung | Evidence tier | Mean val CE | Sample std | Mean tokens/s | Mean peak GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| L0 | S05 full baseline | 3.902776002883911 | 0.0019401469024924146 | n/a | 20.37 |
| L1 | full budget | 3.8693554401397705 | 0.0016093352635399778 | 179,839.78125 | 7.026139736175537 |
| L2 | full budget | 4.044178247451782 | 0.008787112152339275 | 78,187.9375 | 10.020929336547852 |
| L3 | reduced side table | 5.4631736278533936 | 0.003097624778156668 | 28,991.357421875 | 9.69283676147461 |
| L4 | reduced short-val side table | 7.277190208435059 | 0.010595380540850673 | 4,695.980895996094 | 14.701903343200684 |

Full-budget deltas:

| Comparison | Delta val CE | 2x S05 std decision |
| --- | ---: | --- |
| L1 full mean - L0 full mean | -0.033420562744140625 | significant better |
| L2 full mean - L0 full mean | +0.1414022445678711 | significant worse |
| L2 full mean - L1 full mean | +0.17482280731201172 | significant worse |

Matched intermediate token count, step 100 / 26,214,400 tokens:

| Rung | Mean val CE at step 100 | Delta versus L0 step 100 | Notes |
| --- | ---: | ---: | --- |
| L0 | 6.055233333333334 | 0 | Approximation from rounded S05 report table |
| L1 | 6.194799184799194 | +0.13956585146586 | full-run intermediate scalar |
| L2 | 6.220002174377441 | +0.164768841044107 | full-run intermediate scalar |
| L3 | 5.4631736278533936 | -0.5920597054799406 | reduced-run final scalar at same token count |

L4 has only step-10 / 2,621,440-token short-validation reduced evidence, so it
does not enter the step-100 matched-token table.

Selected validation curves:

| Rung | Seed | Step 1 | Step 100 | Step 500 | Step 900 | Step 1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 0 | 10.956628799438477 | 6.179305076599121 | 4.304999828338623 | 3.9350075721740723 | 3.8704934120178223 |
| L1 | 1 | 10.961379051208496 | 6.210293292999268 | 4.308346748352051 | 3.9339358806610107 | 3.8682174682617188 |
| L2 | 0 | 10.90146255493164 | 6.211021423339844 | 4.493618965148926 | 4.102880477905273 | 4.037964820861816 |
| L2 | 1 | 10.901923179626465 | 6.228982925415039 | 4.500579357147217 | 4.115859508514404 | 4.050391674041748 |

Reduced-run validation curves:

| Rung | Seed | Step 1 | Final step | Final val CE |
| --- | ---: | ---: | ---: | ---: |
| L3 | 0 | 10.766958236694336 | 100 | 5.4609832763671875 |
| L3 | 1 | 10.746336936950684 | 100 | 5.4653639793396 |
| L4 | 0 | 10.218414306640625 | 10 | 7.284682273864746 |
| L4 | 1 | 10.3447904586792 | 10 | 7.269698143005371 |

## HDFS evidence

All bundles were published under
`hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/<run_id>/`.
Verification checked HDFS `_COMPLETE`, HDFS `MANIFEST.jsonl` equality to the
publisher local manifest, and local stable-file md5/size agreement. Existing
bundles' `publish.txt` was skipped in local md5 comparison because the older
runner teed into it during publish; `run_rung.sh` now writes publish output to a
temporary file outside the bundle and copies it into `publish.txt` after publish.

Science bundle URIs:

- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-full-l1-seed0-20261014T183650Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-full-l1-seed1-20261014T183655Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-full-l2-seed0-20261014T183700Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-full-l2-seed1-20261014T191003Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-reduced-l3-seed0-20261014T191325Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-reduced-l3-seed1-20261014T192736Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-reduced-l4-shortval-seed0-20261014T194304Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-reduced-l4-shortval-seed1-20261014T194910Z`

Smoke bundle URIs:

- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-smoke-l1-20261014T174538Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-smoke-l2-20261014T174627Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-smoke-l3-20261014T174723Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-smoke-l4-lb1-20261014T180450Z`

Step-3 bundle URIs:

- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-overfit-l1-seed0-20261014T181754Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-overfit-l2-seed0-20261014T182006Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-overfit-l3-seed0-20261014T182011Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-overfit-l4-seed0-20261014T182016Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-lr0-l1-seed0-20261014T182702Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-lr0-l2-seed0-20261014T182707Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-lr0-l3-seed0-20261014T182712Z`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07-step3-lr0-l4-seed0-20261014T183006Z`

## Verification

Passed:

```bash
bash -n experiments/glm53_arch_ladder/run_rung.sh
```

Passed:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_ladder/tests/test_s07_arch_ladder.py -vv'
```

Result: 7 passed, 14 warnings in 30.40s.

Passed for S07 changed files with Pyrefly skipped:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export SKIP=pyrefly-check && pre-commit run --files <S07-scoped files>'
```

Unskipped changed-file pre-commit failed only at the configured whole-project
`pyrefly-check`. The first unrelated error was
`No attribute assert_local_type_like in module spmd_types` at
`torchtitan/components/loss.py:97`; Pyrefly reported 52 total errors and 227
suppressed errors in pre-existing/core paths. Temporary suppressions generated by
that check were restored before handoff.

HDFS verification result: all 20 bundles printed `VERIFIED ...`. For smoke
bundles, verification compared 11 stable files and skipped `publish.txt`; for
Step-3 bundles, 10 stable files and skipped `publish.txt`; for science bundles,
12 stable files and skipped `publish.txt`.

## Gate assessment

S07 gate status: PASS for config/unit, smoke, Step-3, and HDFS evidence.

Science interpretation:

- L1 is the only S07 rung that beats L0 at the full S05 budget.
- L2 is significantly worse than L0 and L1 at the full S05 budget despite
  matching active non-embedding parameters.
- L3 looks strong at the 26.2M-token intermediate point but did not meet the
  throughput threshold for the full-budget requirement, so it remains reduced
  side-table evidence.
- L4 remains early, reduced, and short-val only. Its current seam also lacks the
  requested MTP hook because the GLM config validator rejects `use_mtp=True`.
