# A01b report -- A01 gates completed

Date: 2026-10-15

Status: **green for A01b**. A01b satisfied the data, all-arm dry-run,
fixed-bank overfit plus lr=0 control, recall-after-train execution, and
5-minute anchor timing gates. A02 may start from `sdd/a02-brief.md` and spec
Revision 2; the older 19-run pilot shape is superseded.

## Surface and changes

- Surface: repo-local experiment, not core training.
- Rootfs used for all Python, CUDA, tests, log parsing, training, and
  summarization.
- GPU: only GPU 7 was used for A01b training attempts.
- No commit was made.

Code fixes made for A01b:

- `experiments/attn_geometry_speedrun/run_arm.sh`: host-side
  `hdfs_strict.sh --quick` before rootfs entry; exports direct shard globs and
  data URI/md5 into each attempt.
- `experiments/attn_geometry_speedrun/check_data_waystone.sh`: replaced the old
  rootfs Waystone probe with host-side strict staged-cache verification.
- `experiments/attn_geometry_speedrun/run_attempt.py`: records data provenance,
  exclusive GPU environment class, prefix-table source, and fallback reason.
- `experiments/attn_geometry_speedrun/variant/train_gpt.py`: automatic GPT-2
  tokenizer DNS fallback to offline prefix table; RoPE cache sized to at least
  the train sequence length; recall hook temporarily widens eval max length for
  synthetic recall prompts.

## Data gate

Passed.

```text
HDFS_STRICT_DIR=experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1
SOURCE_URI=hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1
MANIFEST_MD5=89e8165a2f2403777b8126b858df6642
FILES=10
MODE=quick
```

The verified cache has 9 train shards and 1 validation shard. Every A01b
attempt artifact records the same source URI and manifest md5.

## Verification

Focused rootfs tests:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_attn_geometry_speedrun.py'
```

Result: `6 passed in 2.89s`.

Rootfs syntax check:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m py_compile experiments/attn_geometry_speedrun/run_attempt.py experiments/attn_geometry_speedrun/variant/train_gpt.py torchtitan/experiments/attn_geometry_speedrun/attention_geometry.py torchtitan/experiments/attn_geometry_speedrun/recall_probe.py tests/unit_tests/test_attn_geometry_speedrun.py'
```

Result: passed.

## G1 dry-run table

Gate: all four arms, FineWeb, A01 science shape, 50 steps, finite loss, curves
diverge. Result: **PASS**.

| arm | artifact | ok | val loss curve | prefix table | peak MiB alloc/res |
| --- | --- | --- | --- | --- | --- |
| anchor | `experiments/attn_geometry_speedrun/results/a01b_g1_dry_anchor_50/attempt_001/summary.json` | yes | 10.8272 -> 8.0303 -> 7.5270 -> 7.2333 -> 7.0380 -> 6.9345 | offline_fallback | 7485 / 7842 |
| dot | `experiments/attn_geometry_speedrun/results/a01b_g1_dry_dot_50/attempt_001/summary.json` | yes | 10.8256 -> 8.0827 -> 7.5623 -> 7.2654 -> 7.0785 -> 6.9719 | offline_fallback | 7485 / 7816 |
| cosine | `experiments/attn_geometry_speedrun/results/a01b_g1_dry_cosine_50/attempt_001/summary.json` | yes | 10.8280 -> 8.0425 -> 7.5376 -> 7.2368 -> 7.0501 -> 6.9450 | offline_fallback | 7485 / 7842 |
| gauss | `experiments/attn_geometry_speedrun/results/a01b_g1_dry_gauss_50/attempt_001/summary.json` | yes | 10.8282 -> 8.0941 -> 7.6277 -> 7.2755 -> 7.0854 -> 6.9820 | offline_fallback | 7485 / 7824 |

Curves are finite and not identical. The GPT-2 tokenizer fetch failed under DNS
outage, so FineWeb runs used the recorded offline prefix-table fallback.

## G2 overfit table

Gate: fixed small bank per arm collapses below predeclared threshold
`CE < 0.5`; lr=0 control does not collapse; run associative-recall probe after
each trained overfit arm. Result: **PASS**.

Final G2 settings:

- `ATTN_SEQ_LEN=2048`
- `ATTN_BATCH_TOKENS=1024`
- `ATTN_VAL_BATCH_SIZE=1024`
- `ATTN_VAL_TOKENS=1024`
- overfit: `--train-steps 300 --val-loss-every 50 --lr-scale 3 --recall-eval`
- lr=0 controls: same, except `--lr-scale 0` and no recall hook

| arm | overfit artifact | overfit CE 0 -> 300 | overfit pass | lr=0 artifact | lr=0 CE 0 -> 300 | lr=0 pass | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor | `experiments/attn_geometry_speedrun/results/a01b_g2_overfit_anchor_smallbank_300_lr3_ropecachefix/attempt_001/summary.json` | 10.8241 -> 0.0006 | yes | `experiments/attn_geometry_speedrun/results/a01b_g2_lr0_anchor_smallbank_300_ropecachefix/attempt_001/summary.json` | 10.8344 -> 10.8345 | yes | ran, 0/16 at 128/1024/1536 |
| dot | `experiments/attn_geometry_speedrun/results/a01b_g2_overfit_dot_smallbank_300_lr3_ropecachefix/attempt_001/summary.json` | 10.8276 -> 0.0851 | yes | `experiments/attn_geometry_speedrun/results/a01b_g2_lr0_dot_smallbank_300_ropecachefix/attempt_001/summary.json` | 10.8245 -> 10.8245 | yes | ran, 0/16 at 128/1024/1536 |
| cosine | `experiments/attn_geometry_speedrun/results/a01b_g2_overfit_cosine_smallbank_300_lr3_ropecachefix/attempt_001/summary.json` | 10.8306 -> 0.0004 | yes | `experiments/attn_geometry_speedrun/results/a01b_g2_lr0_cosine_smallbank_300_ropecachefix/attempt_001/summary.json` | 10.8253 -> 10.8254 | yes | ran, 0/16 at 128/1024/1536 |
| gauss | `experiments/attn_geometry_speedrun/results/a01b_g2_overfit_gauss_smallbank_300_lr3_ropecachefix/attempt_001/summary.json` | 10.8256 -> 0.0007 | yes | `experiments/attn_geometry_speedrun/results/a01b_g2_lr0_gauss_smallbank_300_ropecachefix/attempt_001/summary.json` | 10.8250 -> 10.8250 | yes | ran, 0/16 at 128/1024/1536 |

Notes:

- The initial inherited 80-step overfit setting did not collapse
  (`anchor` ended around 6.12 nats), so A01b tightened the fixed bank to the
  intended small-bank gate above.
- Recall accuracy was 0 on fresh random mappings for all arms. This is reported
  as behavior output, not a blocker for G2, whose gate required the probe to run
  after training.
- Fixed-bank runs force `ATTN_OFFLINE_PREFIX_TABLE=1`; artifacts record
  `prefix_table_source=offline_forced`.

## G3 anchor timing table

Gate: 5-minute anchor timed FineWeb run on GPU 7, with target loss,
compile/warmup, eval time, train time, step average, and peak memory populated.
Result: **PASS**.

Artifact:
`experiments/attn_geometry_speedrun/results/a01b_g3_anchor_5min_fineweb/attempt_001/summary.json`

| field | value |
| --- | --- |
| ok | true |
| train_time_ms | 300028 |
| final step | 3443 |
| final val_loss | 4.7637 |
| compile_warmup_time_ms | 218148 |
| final eval_time_ms | 67 |
| step_avg_ms | 87.14 |
| peak memory allocated/reserved MiB | 7495 / 7852 |
| elapsed wall seconds | 540.2 |
| prefix table | offline_fallback |
| data URI | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1` |
| manifest md5 | `89e8165a2f2403777b8126b858df6642` |

## A02 launch

A01b green-lights A02's precondition. Current binding A02 is spec Revision 2
(`sdd/a02-brief.md`), not the older 19-run pilot. Do not run the old A01 report
command.

Exact current A02 entry command:

```bash
sed -n '1,260p' .scratch/attn-geometry-speedrun/sdd/a02-brief.md
```

After implementing the A02 pre-spend requirements from that brief
(geometry-scalar AdamW group, fp32 gauss key penalty, warmup knob, diagnostics,
diagnostics-on/off identity check, and one 50-step dry-run per arm), launch the
A02a screen with this run matrix:

```bash

export ATTN_SEQ_LEN=1024
export ATTN_BATCH_TOKENS=8192
export ATTN_VAL_BATCH_SIZE=16384
export ATTN_VAL_TOKENS=16384
export ATTN_ENVIRONMENT_CLASS=exclusive_gpu

# Anchor noise floor: 2 seeds.
for seed in 0 1; do
  ATTN_SEED="${seed}" experiments/attn_geometry_speedrun/run_arm.sh \
    --arm anchor --mode pilot_5min --timed-seconds 300 \
    --data-mode fineweb --gpu-id 7 --environment-class exclusive_gpu \
    --run-id "a02a_anchor_seed${seed}" --attempt-id attempt_001
done

# Non-anchor screen: 3 arms x 4 LR scales x 3 init values = 36 runs.
for arm in dot cosine gauss; do
  for lr_scale in 0.5 0.75 1.0 1.5; do
    for init in 0.5 1 2; do
      ATTN_LR_SCALE="${lr_scale}" ATTN_GEOMETRY_INIT="${init}" \
      experiments/attn_geometry_speedrun/run_arm.sh \
        --arm "${arm}" --mode pilot_5min --timed-seconds 300 \
        --data-mode fineweb --gpu-id 7 --environment-class exclusive_gpu \
        --run-id "a02a_screen_${arm}_lr${lr_scale}_init${init}_seed0" \
        --attempt-id attempt_001
    done
  done
done
```

If free exclusive GPUs exist, A02 may run this same matrix through a queue with
one run per free GPU. Before each launch, keep the A01b data invariant:
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh --quick experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`.

A02b confirmation commands are intentionally not pre-filled here because
Revision 2 requires freezing the per-arm best config from the A02a screen table
first, then running all 4 arms x 3 fresh seeds plus matched-token reruns and
recall.

## Claim labels

- G1: `smoke`
- G2: `learnability_smoke`
- G3: `smoke`
- No A02, representative-small, replicated-eval, or winner claim is made here.
