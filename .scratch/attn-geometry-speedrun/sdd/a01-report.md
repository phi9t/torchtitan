# A01 report -- attention geometry harness, systems, overfit gates

Date: 2026-10-15

Status: **not green; A02 blocked**. The reusable geometry implementation, tests,
runner, copied upstream variant, artifact classification, and one GPU sanity
attempt are in place. The required FineWeb/HDFS data gate, all-arm science-shape
dry-run, per-arm fixed-bank overfit, lr=0 controls, and 5-minute anchor timed run
did not complete.

## Surface and source

- Surface: repo-local experiment, not core training.
- Reusable Python: `torchtitan/experiments/attn_geometry_speedrun/`.
- Runners and copied vehicle: `experiments/attn_geometry_speedrun/`.
- Upstream vehicle copied from pinned cached checkout:
  `experiments/modded_nanogpt_b200/sources/modded-nanogpt/train_gpt.py`.
- Upstream cached file was not modified.
- `CLAUDE.md` was requested by the brief but is not present in this checkout;
  the artifact schema was taken from the AGENTS.md instructions and ticket text.

## Config chosen

Small A01 shape:

| field | value |
| --- | --- |
| layers | 8 |
| model_dim | 384 |
| heads | 6 |
| head_dim | 64 |
| seq_len | 1024 |
| train batch tokens | 8192 global tokens per optimizer step |
| val batch / val tokens | 16384 / 16384 by default |
| GPUs | single GPU |

Rationale: the owner suggested starting around 6-8 layers and d_model 384-512.
The copied upstream topology has MUDD/value/gate paths with layer-indexed seams,
so 8 layers is the smallest valid shape after localizing those seams.

Implemented arms:

| arm | implementation |
| --- | --- |
| anchor | upstream QK RMS norm + RoPE + learned positive per-head multiplier, default 1.0 |
| dot | no QK norm, RoPE unchanged, learned positive per-head scale, default `1/sqrt(head_dim)` |
| cosine | L2-normalized q/k after RoPE, learned positive temperature, default `head_dim * 0.1` to match anchor temperature |
| gauss | unnormalized q/k after RoPE, augmented SDPA for `(q.k - ||k||^2/2) / sigma_h^2`, default `sigma=sqrt(head_dim)` |

B200/rootfs compatibility added in the copy only:

- Direct one-process distributed launch with `RANK=0`, `WORLD_SIZE=1`,
  `LOCAL_RANK=0`, `MASTER_ADDR=127.0.0.1`; avoids `torchrun --standalone`
  hostname/TCPStore failure in this rootfs.
- Local FlashAttention import path (`ATTN_BACKEND=fa2`) instead of the HF
  `kernels-community/flash-attn3` version lookup, which failed under DNS outage.
- Custom CE kernel compile capability override defaults to SM100 for B200.
- Rootfs compiler override falls back to `/usr/bin/gcc` and `/usr/bin/g++` when
  inherited Spack compiler paths are stale.
- Fixed-bank smoke can use `ATTN_OFFLINE_PREFIX_TABLE=1`, disabling the
  tokenizer-derived prefix auxiliary table when rootfs DNS cannot fetch GPT-2
  tokenizer files.
- CUDA memory fraction cap defaults to `0.35`.
- `results/` is ignored under the experiment directory.

## Test results

Rootfs focused tests:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_attn_geometry_speedrun.py'
```

Result: `6 passed in 2.22s`.

Covered:

- Gaussian augmented-SDPA equals explicit `-||q-k||^2/(2 sigma^2)` softmax at
  `seq_len=1024`, including backward finite-gradient check.
- Cosine equals the anchor RMS-norm path at matched temperature with `eps=0`.
- Dot equals plain SDPA at `s=1/sqrt(d)`.
- Upstream RoPE preserves key norm for the Gaussian path.
- Associative-recall probe creates fresh aligned examples and reports accuracy.

Syntax/rootfs compile check:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m py_compile experiments/attn_geometry_speedrun/variant/train_gpt.py experiments/attn_geometry_speedrun/run_attempt.py torchtitan/experiments/attn_geometry_speedrun/attention_geometry.py torchtitan/experiments/attn_geometry_speedrun/recall_probe.py tests/unit_tests/test_attn_geometry_speedrun.py'
```

Result: passed.

## Data gate

Blocked. `waystone` is not available in rootfs PATH:

```bash
experiments/attn_geometry_speedrun/check_data_waystone.sh hdfs://placeholder/fineweb10B experiments/attn_geometry_speedrun/results/data_cache_probe
# waystone not found in rootfs PATH
```

No HDFS FineWeb URI was verified, no verified `put`/`get --verify-md5` was
performed, and no input URI can honestly be recorded for A02.

## Dry-run table

Required gate: all four arms dry-run a few steps at science shape on FineWeb
tokens, finite loss, and diverging curves.

Status: **not run / blocked** by missing Waystone/FineWeb verified cache and
shared GPU pressure. A fixed-bank one-step anchor sanity ran, but it is not a
FineWeb dry-run matrix and does not prove arm divergence.

## Overfit table

Required gate: fixed small bank overfit per arm with CE collapse, plus `lr=0`
control failure.

Status: **not run / blocked** by shared GPU pressure after runner bring-up. One
fixed-bank anchor one-step sanity passed:

| run_id | arm | mode | data | ok | val_loss step 0 -> 1 | peak alloc/reserved | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `a01_runner_sanity_anchor_warmup_clip` | anchor | overfit sanity | fixed_bank | yes | `10.8279 -> 10.7357` | `7485 / 7842 MiB` | shared GPU, not CE-collapse proof |

Artifact:
`experiments/attn_geometry_speedrun/results/a01_runner_sanity_anchor_warmup_clip/attempt_001/summary.json`.

## Anchor 5-minute result

Required gate: one 5-minute timed anchor run defining target loss and validating
compile, eval, train, and peak-memory fields.

Status: **not run / blocked** by missing FineWeb verified cache and shared GPU
pressure. The runner now supports `--mode timed_anchor` with a default
`ATTN_TIMED_SECONDS=300` and the trainer logs `compile_warmup_time_ms`,
`eval_time_ms`, `train_time`, `step_avg`, and peak memory, but that timing path
was not rerun after the timing-field patch.

## A02 command

Do not run until A01 is green. The A02 pilot requires a verified FineWeb local
cache and completed A01 dry-run/overfit/timed-anchor gates.

Gated launch shape, once those blockers are cleared:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc '
cd /workspace/torchtitan
export DATA_PATH=/workspace/torchtitan/experiments/attn_geometry_speedrun/results/fineweb_verified_cache
export ATTN_TRAIN_FILES="${DATA_PATH}/data/fineweb10B/fineweb_train_*.bin"
export ATTN_VAL_FILES="${DATA_PATH}/data/fineweb10B/fineweb_val_*.bin"
export ATTN_BACKEND=fa2
export ATTN_CE_COMPUTE_CAPABILITY=100
export ATTN_MODEL_LAYERS=8 ATTN_MODEL_HEADS=6 ATTN_HEAD_DIM=64 ATTN_MODEL_DIM=384
export ATTN_SEQ_LEN=1024 ATTN_BATCH_TOKENS=8192 ATTN_VAL_BATCH_SIZE=16384 ATTN_VAL_TOKENS=16384

# Anchor baseline: 1 x 5 min.
experiments/attn_geometry_speedrun/run_arm.sh --arm anchor --mode pilot_5min --timed-seconds 300 \
  --data-mode fineweb --run-id a02_anchor_seed0 --attempt-id attempt_001

# Screening: 3 non-anchor arms x 2 lr scales x 2 init temperatures = 12 x 5 min.
for arm in dot cosine gauss; do
  for lr_scale in 0.75 1.00; do
    for init in 0.5 1.0; do
      ATTN_LR_SCALE="${lr_scale}" ATTN_GEOMETRY_INIT="${init}" \
      experiments/attn_geometry_speedrun/run_arm.sh --arm "${arm}" --mode pilot_5min --timed-seconds 300 \
        --data-mode fineweb --run-id "a02_screen_${arm}_lr${lr_scale}_init${init}_seed0" --attempt-id attempt_001
    done
  done
done

# Confirmation: after screening selects frozen finalist knobs, run exactly
# six fresh-seed 5-minute runs using the selected arm/lr/init triples.
# Fill these six commands from the screening table before launch; do not invent
# new knobs here.
'
```

This is not a completed 19-run launch command because the six confirmation
runs depend on the A02 screening result and frozen finalist knobs. The current
ticket does not authorize running A02.

## Blockers to A02

1. `waystone` is missing in rootfs PATH; FineWeb HDFS `layout-profile`,
   verified `put`, and verified `get --verify-md5` were not possible.
2. All GPUs are shared with large jobs. After the successful sanity run, GPU 7
   had about 43 GiB free and existing jobs occupied every device; further matrix
   runs would risk slowing them.
3. A01 dry-run divergence, fixed-bank CE collapse, `lr=0` controls, associative
   recall after training, and 5-minute anchor are still unproven.
4. The successful GPU run used fixed-bank data and offline prefix-table fallback;
   it is a systems sanity result, not a FineWeb or speedrun claim.

## Claim labels

- Tests and one-step runner sanity: `smoke` / `learnability_smoke` only.
- No `representative_small`, `replicated_eval`, convergence, reward, or
  speedrun claim is made.

## No commit

No commit was made.
