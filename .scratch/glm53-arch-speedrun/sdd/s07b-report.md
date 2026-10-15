# S07b report: architecture ladder fixes before epoch runs

Date: 2026-10-15

Surface: repo-local experiment code under
`torchtitan/experiments/glm53_arch_ladder/` and host orchestration under
`experiments/glm53_arch_ladder/`. No core `Trainer`, online-RL path, S06
`glm53_arch_speedrun`, or `torchtitan/experiments/glm53_flash/` model code was
edited. I did not commit.

Environment label: `shared_gpu_s07b_full` for the full reruns. Every run used
one visible GPU, `GLM53_ARCH_LADDER_GPU_MEMORY_CAP_GIB=24`, and the S07b
co-run opt-in `GLM53_ARCH_LADDER_ALLOW_CORUN=1`. No foreign process was killed
or signaled. Throughput is shared-host evidence, not quiet-machine performance.

Input: FineWeb GPT-2 shards staged from
`hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`.
The staged input carried `_COMPLETE`, `MANIFEST.jsonl`, `.staged.json`, and
manifest md5 `89e8165a2f2403777b8126b858df6642`.

## Executive result

S07b is complete. L2 Qwen3-MoE had a harness optimizer bug: routed expert matrix
batches and router gates were excluded from Muon, so they fell through to AdamW.
The default partition now assigns those tensors to DistMuon and keeps env
switches for reproducing the old behavior:

- `GLM53_ARCH_LADDER_MOE_MATRIX_BATCHES_ON_MUON=0`
- `GLM53_ARCH_LADDER_MOE_ROUTER_ON_MUON=0`

The fix recovered L2 substantially but did not make it beat dense L0/L1 at the
full budget. Fixed L2 validation CE is 3.967535614967346 mean, versus S07 old
L2 mean 4.044178247451782 and L0 mean 3.902776002883911.

L3 was profiled and then rerun at the full budget. The profile points at
model/optimizer internals, not a ladder-owned shell/config hot spot. L3 remains
far below the throughput target and far worse in validation CE.

L4 uses the T17 grouped expert path
(`GLM53_FLASH_EXPERTS_IMPL=grouped`) and was rerun at the full training budget
with short validation (`validator.steps=64`, 262,144 validation tokens). K03's
indexer aux-stride/compile knobs are not available on the architecture-ladder
`glm53_flash` surface, so they were not grafted across surfaces. L4 is much
better than the old 10-step side table, but remains far below the throughput
target and worse than L0/L1.

S09 is unblocked as a bookkeeping matter, but the ladder evidence says only L1
is a positive full-budget result at this scale. Fixed L2, L3, and L4 should be
treated as negative or diagnostic rungs unless S09 intentionally studies them
for transfer/scale behavior rather than speedrun win-rate.

## Implementation changes

`torchtitan/experiments/glm53_arch_ladder/params.py`

- Changed the Muon candidate rule from trainable 2D hidden matrices only to
  trainable hidden matrices plus known MoE routed expert matrix batches.
- Added explicit controls for MoE matrix batches and router gates:
  `moe_matrix_batches_on_muon` and `moe_router_on_muon`, defaulting true.
- Routed expert matrix batches use a compute layout sharded on the expert/batch
  axis. `DistMuon` documents matrix-batch compute views as equivalent to running
  Muon one matrix at a time, aside from bitwise kernel-schedule equality.

`torchtitan/experiments/glm53_arch_ladder/config_registry.py`

- Wired the new MoE optimizer controls into the ladder optimizer config.
- Added profiler env knobs used for the L3 one-step profile.

`experiments/glm53_arch_ladder/run_rung.sh`

- Enforced HDFS-strict input validation before launch, staging from Waystone HDFS
  when needed and recording `.staged.json` plus input manifest md5 in the bundle.
- Added the opt-in co-run guard `GLM53_ARCH_LADDER_ALLOW_CORUN=1` without
  weakening the default blocking behavior.
- Defaulted L4 to `GLM53_FLASH_EXPERTS_IMPL=grouped` and short validation
  (`GLM53_ARCH_LADDER_VAL_TOKENS=262144`).
- Added bundle metadata for MoE optimizer switches, co-run status, input URI,
  input manifest md5, and L4 grouped experts.

`torchtitan/experiments/glm53_arch_ladder/tests/test_s07_arch_ladder.py`

- Added coverage that L2 Qwen3-MoE router gates and routed expert tensors land
  in the DistMuon config by default, while expert bias stays out of Muon.

`torchtitan/experiments/glm53_arch_ladder/diagnose_l2_moe.py`

- Added a targeted diagnostic script for optimizer assignment, MoE contract,
  initialization, routing fractions, gradient norms, and aux-free bias updates.

## L2 diagnostic and short A/B

Diagnostic evidence:
`.scratch/glm53-arch-speedrun/evidence/s07b/l2_moe_diagnostics_after_muon.json`

Key facts from the diagnostic:

- Active non-embedding params: 88,192,256; total params: 363,567,872.
- MoE contract: 14 MoE layers, 8 experts, top-2 routing, dropless token-choice
  routing, load-balance coefficient 0.001, no shared experts.
- All 42 MoE routed expert/router tensors in the diagnostic were assigned to
  DistMuon. AdamW owned the remaining non-Muon parameters.
- Router and routed-expert gradient norms were nonzero on diagnostic steps.
  Example step-1 norms: routed experts 6.31011897979702, router
  0.3923365684522394.
- Aux-free bias updates were active; max per-layer bias delta was approximately
  0.0015 on each diagnostic step.

Short A/B evidence:
`.scratch/glm53-arch-speedrun/evidence/s07b/l2_ab_summary.json`

Both modes used 33,554,432 tokens and two seeds. Fixed mode means the new
default DistMuon ownership for routed expert matrix batches and router gates;
old mode uses both env toggles set to 0.

| Mode | Mean val CE | Mean train loss | Mean tokens/s | Mean peak GiB |
| --- | ---: | ---: | ---: | ---: |
| Old | 5.994307041168213 | 6.031514406204224 | 90,296.01953125 | 10.020929336547852 |
| Fixed | 5.938305377960205 | 5.9787774085998535 | 93,262.4921875 | 9.550188064575195 |
| Fixed minus old | -0.05600166320800781 | -0.05273699760437012 | +2,966.47265625 | -0.47074127197265625 |

The short A/B was enough to justify the full L2 rerun.

## L3 profile

Profile bundle:
`experiments/glm53_arch_ladder/results/s07b-profile-l3-20261014T203537Z`

Trace:
`experiments/glm53_arch_ladder/results/s07b-profile-l3-20261014T203537Z/profiling/traces/iteration_1/rank0_trace.json.gz`

Summary:
`.scratch/glm53-arch-speedrun/evidence/s07b/l3_profile_summary.json`

The largest non-wrapper profile events were model/optimizer internals:

| Event | Duration |
| --- | ---: |
| `Optimizer.step#OptimizersContainer.step` | 1.985507 s |
| `Optimizer.step#AdamW.step` | 1.670596 s CPU / 1.667748 s GPU annotation |
| `aten::_foreach_mul_` | 0.422455 s |
| `aten::_foreach_addcdiv_` | 0.385664 s |
| `aten::_foreach_norm` | 0.384189 s |
| `Optimizer.step#DistMuon.step` | 0.293174 s CPU / 0.287441 s GPU annotation |

I did not make an L3 patch because the profile did not identify a
ladder-owned runner/config hot spot.

## Full-budget reruns

Full rerun settings:

- Token budget: 268,435,456.
- Tokens per optimizer step: 262,144.
- Training steps: 1024.
- Logged final train scalar: step 1000.
- Final validation scalar: step 1024.
- Sequence length: 4096.
- GPT-2 vocab size: 50,257.
- L2 local batch 4, grad accumulation 16.
- L3 local batch 2, grad accumulation 32.
- L4 local batch 1, grad accumulation 64.
- L4 validation: 64 batches x 4096 tokens = 262,144 validation tokens.
- L4 config: grouped experts, sparse attention mode `indexer`, `use_mtp=false`.

| Rung | Seed | Run | Val CE | Tokens/s | Peak GiB | Local batch | Grad accum |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| L2 | 0 | `s07b-full-l2-seed0-20261015T001523Z` | 3.967376708984375 | 64,830.7734375 | 9.550188064575195 | 4 | 16 |
| L2 | 1 | `s07b-full-l2-seed1-20261015T001533Z` | 3.9676945209503174 | 69,604.875 | 9.550188064575195 | 4 | 16 |
| L3 | 0 | `s07b-full-l3-seed0-20261014T212710Z` | 4.603527069091797 | 29,009.22265625 | 9.59988021850586 | 2 | 32 |
| L3 | 1 | `s07b-full-l3-seed1-20261014T212710Z` | 4.603015899658203 | 29,202.98828125 | 9.59988021850586 | 2 | 32 |
| L4 | 0 | `s07b-full-l4-shortval-seed0-20261014T213940Z` | 3.9674227237701416 | 7,782.05712890625 | 14.494586944580078 | 1 | 64 |
| L4 | 1 | `s07b-full-l4-shortval-seed1-20261015T012958Z` | 3.9923675060272217 | 10,336.1826171875 | 14.494586944580078 | 1 | 64 |

Group summaries:

| Rung | Evidence | Mean val CE | Sample std | Mean tokens/s | Mean peak GiB | Delta vs L0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L0 | S05 full baseline | 3.902776002883911 | 0.0019401469024924146 | n/a | 20.37 | 0 |
| L1 | S07 full budget | 3.8693554401397705 | 0.0016093352635399778 | 179,839.78125 | 7.026139736175537 | -0.033420562744140625 |
| L2 | S07b fixed full budget | 3.967535614967346 | 0.00022472699626008698 | 67,217.82421875 | 9.550188064575195 | +0.06475961208343506 |
| L3 | S07b full budget | 4.603271484375 | 0.0003614513728294272 | 29,106.10546875 | 9.59988021850586 | +0.7004954814910889 |
| L4 | S07b full budget, grouped experts, short-val | 3.9798951148986816 | 0.017638624689203197 | 9,059.119873046875 | 14.494586944580078 | +0.07711911201477051 |

Decisions against the preregistered S05 baseline std
(`2 * std = 0.003880293804984829`):

| Comparison | Delta val CE | Decision |
| --- | ---: | --- |
| L1 - L0 | -0.033420562744140625 | significant better |
| Fixed L2 - L0 | +0.06475961208343506 | significant worse |
| Fixed L2 - L1 | +0.09818017482757568 | significant worse |
| L3 - L0 | +0.7004954814910889 | significant worse |
| L4 short-val - L0 | +0.07711911201477051 | significant worse |
| L4 short-val - L1 | +0.11053967475891113 | significant worse |

Throughput target: `0.3 * L0 smoke = 65,052 tokens/s`.

- L2 mean clears the target at 67,217.82421875 tokens/s, but seed 0 alone is
  slightly below it at 64,830.7734375 tokens/s.
- L3 fails the target at 29,106.10546875 tokens/s.
- L4 fails the target at 9,059.119873046875 tokens/s even after grouped experts.

## HDFS evidence

Verification checked HDFS `_COMPLETE`, HDFS `MANIFEST.jsonl` equality to the
publisher local manifest, and matching local/HDFS manifest md5.

| Run | Files | Manifest md5 | HDFS URI |
| --- | ---: | --- | --- |
| `s07b-full-l2-seed0-20261015T001523Z` | 14 | `7a9fc248039e49402f2a50bc700de43d` | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07b-full-l2-seed0-20261015T001523Z` |
| `s07b-full-l2-seed1-20261015T001533Z` | 14 | `976bda14508ef6eaa17031f56c50d00f` | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07b-full-l2-seed1-20261015T001533Z` |
| `s07b-full-l3-seed0-20261014T212710Z` | 14 | `91527f02627277a78b1719a772636a19` | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07b-full-l3-seed0-20261014T212710Z` |
| `s07b-full-l3-seed1-20261014T212710Z` | 14 | `df441a96610353535032d328887f6ae3` | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07b-full-l3-seed1-20261014T212710Z` |
| `s07b-full-l4-shortval-seed0-20261014T213940Z` | 14 | `cfc06f3739657411b360ef9ef197c617` | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07b-full-l4-shortval-seed0-20261014T213940Z` |
| `s07b-full-l4-shortval-seed1-20261015T012958Z` | 14 | `d6c02bcf35af8318fdfc5aafeb2c3597` | `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-ladder/s07b-full-l4-shortval-seed1-20261015T012958Z` |

The L4 seed-0 runner was launched from an older in-flight shell image and hit
the known post-training parse error
`experiments/glm53_arch_ladder/run_rung.sh: line 397: unexpected EOF while looking for matching '"'`
after the trainer printed `Training completed`. The completed training log was
manually postprocessed through the same bundle steps: copied `train.log`, copied
`DATA_MANIFEST.jsonl`/`DATA_STAGED.json`, ran the rootfs metrics extractor, and
published with `experiments/glm53_flash_sft_ladder/publish.sh`. The current
`run_rung.sh` passes `bash -n`, and L4 seed 1 completed wrapper publish without
manual replay.

The original full-validation L4 attempt
`s07b-full-l4-seed0-20261014T212710Z` is failed/aborted evidence only and is
not counted in any S07b result.

## Verification

Passed before final handoff:

```bash
bash -n experiments/glm53_arch_ladder/run_rung.sh
```

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_ladder/tests/test_s07_arch_ladder.py -vv'
```

Result: 8 passed, 14 warnings.

Focused changed-file hooks also passed inside `TORCHTITAN_ROOTFS_ENV=glm53_train`
with `unset CC CXX`:

```bash
pre-commit run ufmt --files \
  torchtitan/experiments/glm53_arch_ladder/config_registry.py \
  torchtitan/experiments/glm53_arch_ladder/params.py \
  torchtitan/experiments/glm53_arch_ladder/tests/test_s07_arch_ladder.py \
  torchtitan/experiments/glm53_arch_ladder/diagnose_l2_moe.py
pre-commit run flake8 --files <same Python files>
pre-commit run pydoclint --files <same Python files>
pre-commit run codespell --files \
  <same Python files> \
  experiments/glm53_arch_ladder/run_rung.sh \
  docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md \
  .scratch/glm53-arch-speedrun/sdd/s07b-report.md
```

Result: all passed. `git diff --check` also passed on the changed S07b files.

HDFS strict input quick check passed for the FineWeb staged cache:

```text
MANIFEST_MD5=89e8165a2f2403777b8126b858df6642 FILES=10 MODE=quick
```

All six full S07b rerun bundles passed the HDFS `_COMPLETE` and manifest
equality check listed in the HDFS evidence table above.
