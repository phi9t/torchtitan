# S01 report: speedrun harness, B0 baseline, and FineWeb HDFS data

## Scope

Surface: repo-local experiment. S01 added a TorchTitan Trainer-backed B0 harness under
`torchtitan/experiments/glm53_arch_speedrun/` plus the host runner
`experiments/glm53_arch_speedrun/run_arm.sh`. The harness keeps the experiment
outside core and leaves the S02-owned `elements/` directory untouched.

Claim label for the real B0 runs: `calibrated_subset`. These runs prove the B0
training harness, data path, optimizer partition, validation curve capture, and
HDFS artifact publication on shared GPUs. They are not a 500M-token baseline and
not clean performance evidence.

## Harness note

- Model: B0 modded-nanogpt-classic substrate, 12 layers, hidden size 768, 6 heads
  with head dim 128, QK norm, RoPE, ReLU^2 MLP, untied softcap LM head, and 24 GiB
  per-process GPU memory cap.
- Replaceable seams: block, attention, residual, positional encoding, and head are
  small classes selected from config fields so S02 element modules can plug in.
- Optimizer split: DistMuon for hidden 2D projection matrices; AdamW for token
  embeddings, LM head, norm gains, and scalar or non-hidden parameters.
- Data: DP-aware FineWeb GPT-2 nanogpt `.bin` reader reusing the Falcon-style
  path; labels are shifted next-token targets, not rolled.
- Runner: host-only orchestration verifies or stages data via Waystone/HDFS, then
  launches TorchTitan inside `TORCHTITAN_ROOTFS_ENV=glm53_train` with `unset CC CXX`.
  Run bundles include `attempt.json`, `config.json`, `launch_command.txt`,
  structured logs, TensorBoard events, run evidence, train log, data manifest, and
  publish manifest.

## Data

Complete HDFS dataset URI:

`hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`

Local staged root:

`experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`

Manifest: 10 `.bin` files, each with md5 in `MANIFEST.jsonl`: nine train shards
`fineweb_train_000001.bin` through `fineweb_train_000009.bin`, plus
`fineweb_val_000000.bin`. `_COMPLETE` exists on HDFS for the dataset.

Do not use the earlier partial dataset URI ending in
`ecbb586-fineweb10b-gpt2-900m`; it was interrupted before `_COMPLETE`.

## Calibration

The ticket suggested starting around 500M tokens and shrinking if too slow.
A real-shape one-step probe used 524,288 train tokens with seq1024 and reported
35,440 train tokens/s, 3.78 GiB memory, and validation CE 10.4147. At that probe
rate, 500M tokens would take about 3.9 hours per seed on the shared machine, so
the S01 baseline budget was reduced to 8,388,608 train tokens per seed.

Probe bundle:

`hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-realshape-probe-seed0-20261014T1510`

## B0 seq1024 baseline

Common config:

- `NGPU=1`
- `GLM53_SPEEDRUN_SEQ_LEN=1024`
- `GLM53_SPEEDRUN_LOCAL_BATCH_SIZE=4`
- `GLM53_SPEEDRUN_TOKENS_PER_STEP=524288`
- `GLM53_SPEEDRUN_TOKEN_BUDGET=8388608`
- Trainer steps: 16
- Validation: every 8 steps, 64 validation batches = 262,144 tokens
- Loss chunks: 8
- Environment class: `shared_gpu`
- Evidence tier: `tier0`
- Claim label: `calibrated_subset`

| Seed | Run ID | Val CE step 1 | Val CE step 8 | Final val CE step 16 | Final train loss | Final train tps | Final val tps | Peak memory GiB | HDFS publish |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `s01-b0-seed0-8m-20261014T1515` | 10.3926 | 7.6033 | 6.9660 | 6.88591 | 53,310 | 128,432 | 4.22 | `_COMPLETE`, 11 files |
| 1 | `s01-b0-seed1-8m-20261014T1518` | 10.3341 | 7.7489 | 6.9438 | 6.85706 | 52,012 | 126,663 | 4.22 | `_COMPLETE`, 11 files |
| 2 | `s01-b0-seed2-8m-20261014T1518` | 10.3658 | 7.5674 | 6.9647 | 6.88351 | 55,051 | 131,304 | 4.22 | `_COMPLETE`, 11 files |

Aggregate over seeds 0, 1, 2:

- Final validation CE mean: 6.958167
- Final validation CE sample std: 0.012459
- Final train throughput mean: 53,457.67 tokens/s
- Final train throughput sample std: 1,524.87 tokens/s
- Peak memory mean: 4.22 GiB
- Tokens to CE 3.28: not reached within 8,388,608 tokens
- Calibrated target for later arms: B0 mean final CE 6.958167 at 8,388,608 tokens

Run bundle URIs:

- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seed0-8m-20261014T1515`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seed1-8m-20261014T1518`
- `hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seed2-8m-20261014T1518`

## B0 seq4096 variant

This run keeps the same token budget and tokens per optimizer step as the seq1024
baseline, but uses seq4096 and local batch size 1. It is included because KDA and
indexer arms need longer context.

| Seed | Run ID | Seq len | Val CE step 1 | Val CE step 8 | Final val CE step 16 | Final train loss | Final train tps | Final val tps | Peak memory GiB | HDFS publish |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `s01-b0-seq4096-seed0-8m-20261014T1518` | 4096 | 10.3425 | 7.6177 | 6.9830 | 6.91289 | 53,569 | 130,025 | 4.22 | `_COMPLETE`, 11 files |

Run bundle URI:

`hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seq4096-seed0-8m-20261014T1518`

Tokens to CE 3.28: not reached within 8,388,608 tokens.

## Exact launch commands

Seed 0 seq1024:

```bash
CUDA_VISIBLE_DEVICES=0 GLM53_SPEEDRUN_SEQ_LEN=1024 GLM53_SPEEDRUN_LOCAL_BATCH_SIZE=4 GLM53_SPEEDRUN_TOKENS_PER_STEP=524288 GLM53_SPEEDRUN_TOKEN_BUDGET=8388608 GLM53_SPEEDRUN_VAL_TOKENS=262144 GLM53_SPEEDRUN_VAL_FREQ=8 GLM53_SPEEDRUN_LOG_FREQ=4 GLM53_SPEEDRUN_LOSS_CHUNKS=8 GLM53_SPEEDRUN_CLAIM_LABEL=calibrated_subset GLM53_SPEEDRUN_MODE=b0_baseline GLM53_SPEEDRUN_ENVIRONMENT_CLASS=shared_gpu GLM53_SPEEDRUN_SEED=0 TORCHTITAN_RUN_ID=s01-b0-seed0-8m-20261014T1515 TORCHTITAN_ATTEMPT_ID=attempt-001 NGPU=1 experiments/glm53_arch_speedrun/run_arm.sh B0
```

Seed 1 seq1024:

```bash
CUDA_VISIBLE_DEVICES=0 GLM53_SPEEDRUN_SEQ_LEN=1024 GLM53_SPEEDRUN_LOCAL_BATCH_SIZE=4 GLM53_SPEEDRUN_TOKENS_PER_STEP=524288 GLM53_SPEEDRUN_TOKEN_BUDGET=8388608 GLM53_SPEEDRUN_VAL_TOKENS=262144 GLM53_SPEEDRUN_VAL_FREQ=8 GLM53_SPEEDRUN_LOG_FREQ=4 GLM53_SPEEDRUN_LOSS_CHUNKS=8 GLM53_SPEEDRUN_CLAIM_LABEL=calibrated_subset GLM53_SPEEDRUN_MODE=b0_baseline GLM53_SPEEDRUN_ENVIRONMENT_CLASS=shared_gpu GLM53_SPEEDRUN_SEED=1 TORCHTITAN_RUN_ID=s01-b0-seed1-8m-20261014T1518 TORCHTITAN_ATTEMPT_ID=attempt-001 NGPU=1 experiments/glm53_arch_speedrun/run_arm.sh B0
```

Seed 2 seq1024:

```bash
CUDA_VISIBLE_DEVICES=1 GLM53_SPEEDRUN_SEQ_LEN=1024 GLM53_SPEEDRUN_LOCAL_BATCH_SIZE=4 GLM53_SPEEDRUN_TOKENS_PER_STEP=524288 GLM53_SPEEDRUN_TOKEN_BUDGET=8388608 GLM53_SPEEDRUN_VAL_TOKENS=262144 GLM53_SPEEDRUN_VAL_FREQ=8 GLM53_SPEEDRUN_LOG_FREQ=4 GLM53_SPEEDRUN_LOSS_CHUNKS=8 GLM53_SPEEDRUN_CLAIM_LABEL=calibrated_subset GLM53_SPEEDRUN_MODE=b0_baseline GLM53_SPEEDRUN_ENVIRONMENT_CLASS=shared_gpu GLM53_SPEEDRUN_SEED=2 TORCHTITAN_RUN_ID=s01-b0-seed2-8m-20261014T1518 TORCHTITAN_ATTEMPT_ID=attempt-001 NGPU=1 experiments/glm53_arch_speedrun/run_arm.sh B0
```

Seq4096 seed 0:

```bash
RUN_ID="s01-b0-seq4096-seed0-8m-$(date -u +%Y%m%dT%H%M)"; CUDA_VISIBLE_DEVICES=4 NGPU=1 TORCHTITAN_RUN_ID="$RUN_ID" TORCHTITAN_ATTEMPT_ID=attempt-001 GLM53_SPEEDRUN_SEED=0 GLM53_SPEEDRUN_SEQ_LEN=4096 GLM53_SPEEDRUN_LOCAL_BATCH_SIZE=1 GLM53_SPEEDRUN_TOKENS_PER_STEP=524288 GLM53_SPEEDRUN_TOKEN_BUDGET=8388608 GLM53_SPEEDRUN_VAL_TOKENS=262144 GLM53_SPEEDRUN_VAL_FREQ=8 GLM53_SPEEDRUN_LOG_FREQ=4 GLM53_SPEEDRUN_LOSS_CHUNKS=8 GLM53_SPEEDRUN_MODE=b0_seq4096 GLM53_SPEEDRUN_CLAIM_LABEL=calibrated_subset GLM53_SPEEDRUN_EVIDENCE_TIER=tier0 GLM53_SPEEDRUN_ENVIRONMENT_CLASS=shared_gpu experiments/glm53_arch_speedrun/run_arm.sh B0
```

The seq4096 command resolved `RUN_ID` to
`s01-b0-seq4096-seed0-8m-20261014T1518`.

The runner-recorded rootfs launch for each bundle was:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train /data02/home/philip.yang/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && LOG_RANK=${LOG_RANK:-0} NGPU=1 MODULE=torchtitan.experiments.glm53_arch_speedrun.config_registry CONFIG=glm53_arch_speedrun_b0 ./run_train.sh'
```

## Verification

Focused S01 tests:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && pytest -q torchtitan/experiments/glm53_arch_speedrun/tests/test_s01_b0_harness.py'
```

Result: `5 passed, 14 warnings in 5.72s`.

Config load dry run:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && python -m torchtitan.train --module torchtitan.experiments.glm53_arch_speedrun.config_registry --config glm53_arch_speedrun_b0 --comm-backend fake --print-config >/tmp/glm53_s01_print_config.txt'
```

Result: exit 0, with only `WARNING:absl:Failed to load jax profiler: No module named 'jax'`.

HDFS completeness check:

```bash
source experiments/glm53_flash_sft_ladder/lib/waystone.sh
for uri in \
  hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1 \
  hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seed0-8m-20261014T1515 \
  hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seed1-8m-20261014T1518 \
  hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seed2-8m-20261014T1518 \
  hdfs://harunava/user/tiger/waystone/torchtitan/runs/glm53-arch-speedrun/s01-b0-seq4096-seed0-8m-20261014T1518; do
  ws_exists "$(ws_join_uri "$uri" _COMPLETE)"
done
```

Result: `_COMPLETE` exists for the dataset and every listed run bundle.
