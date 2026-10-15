# A01b brief: finish A01 gates (data, all-arm dry-run, overfit + lr=0, 5-min anchor)

Read first: `spec.md`, `sdd/a01-brief.md`, `sdd/a01-report.md` (your predecessor; reuse all its code, do not start over).

A01 blockers and their resolution (decided, do not re-litigate):
1. Data. `waystone` is a host-side tool, not in the rootfs. Stage on the HOST with the existing helpers, exactly as
   `experiments/glm53_arch_ladder/run_rung.sh` does: `experiments/glm53_flash_sft_ladder/stage.sh` and
   `experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh --quick <dir>`. The same FineWeb GPT-2 shards are already staged
   and verified at `experiments/glm53_arch_speedrun/results/data/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`
   (source `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/fineweb10B-gpt2/ecbb586-fineweb10b-gpt2-binonly-v1`,
   manifest md5 89e8165a2f2403777b8126b858df6642, 9 train + 1 val shard). Reuse it read-only: run `hdfs_strict.sh --quick`
   on it before each launch and record source URI + manifest md5 in every attempt artifact. Replace the
   `check_data_waystone.sh` rootfs probe with this host-side check (in `run_arm.sh`).
2. GPUs. You own GPU 7 only (`CUDA_VISIBLE_DEVICES=7`), exclusively; a GLM eval uses GPUs 0-6. Never touch GPUs 0-6 or
   any process you did not start. Run arms sequentially on GPU 7.
3. Tokenizer DNS: keep the offline prefix-table fallback if the GPT-2 tokenizer cannot be fetched; record it in artifacts.

Gates (all required, exit criteria binary):
- G1 dry-run: all four arms (anchor, dot, cosine, gauss), ~50 steps at the A01 science shape on FineWeb, finite loss,
  and curves must DIVERGE (identical curves = invalid harness; investigate init, e.g. zeroed gains).
- G2 overfit: fixed small bank per arm, CE collapses (state the threshold before running, e.g. < 0.5 nats); plus one
  lr=0 control per arm that does NOT collapse. Run the associative-recall probe after training on each overfit arm.
- G3 anchor 5-minute timed run on FineWeb (GPU 7 exclusive): target loss, compile/warmup, eval time, train_time,
  step_avg, peak memory fields all populated.
- Every attempt artifact carries the schema fields from the A01 brief (lane, mode, arm, claim_label, evidence_tier,
  run_id, attempt_id, environment_class) plus data URI/md5.

Do not run A02. Do not commit. Rootfs for all Python/CUDA. Focused tests must still pass
(`pytest -q tests/unit_tests/test_attn_geometry_speedrun.py`).
Report: `sdd/a01b-report.md` with G1/G2/G3 PASS/FAIL tables, artifact paths, and the exact A02 launch command if green.
