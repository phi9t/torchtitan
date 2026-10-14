# GLM-5.3-Flash SFT ladder (campaign spec)

Owner: phi9t. Opened 2026-10-14. Hardware: this 8x B200 host only (no Primus).
Research basis: `docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md`.

## Goal

Hill-climb one SFT pipeline up a model ladder on 8x B200 until it trains
GLM-5.3-Flash with LoRA:

Qwen3 0.6B/1.7B -> Qwen3 8B -> Qwen3 30B-A3B (MoE) -> mini KDA+sparse-attn
hybrid -> GLM-5.3-Flash LoRA (4x then 8x B200).

**Done** = GLM-5.3-Flash LoRA SFT on 8x B200 that loads the official checkpoint
from HDFS, passes a pre-training logits-parity gate vs the HF reference,
improves held-out Countdown pass@1 over base across >=2 seeds, resumes from
HDFS after a local cache wipe, runs >=10 steady steps with throughput/memory
recorded, and publishes adapter + eval + run bundle to HDFS (MD5-verified).

## Fixed across rungs (the "hill")

- Task: Countdown, frozen dataset `countdown-sft/v1` = the replicated champion
  `formatting` arm (931 rows) + dev/iid_test/ood_test problem splits from
  `experiments/countdown_search_distill/sweeps/formatting_replication/formatting_seed43_train2000_rank16/data/`.
  Exact verifier: `torchtitan/experiments/countdown_search_distill`.
- Chat masking: `ChatProcessor` (assistant-only loss).
- Eval: dev pass@1 / pass@k + strict pass@1 via the countdown eval stack; iid/ood
  at rung close. Base vs adapter on the same split and sampling config.
- Tokens per optimizer step fixed when GPU count changes (64 x 512 at rung 0).
- Reference point (Qwen3-1.7B formatting adapter, seed 43, rank 16): dev pass@1 0.278.

## Storage contract (campaign-wide gate)

1. HDFS via waystone is the system of record. Paths from
   `waystone layout-profile --project torchtitan --json` /
   `waystone storage-prefix --child ...`; never hand-joined.
   - `model-weights/<org>/<model>/<hf-revision>/`
   - `datasets/<name>/<version>/`
   - `checkpoints/glm53-ladder/<run_id>/<attempt_id>/`
   - `runs/glm53-ladder/<run_id>/<attempt_id>/`
2. HF ingest once per pinned revision, file by file (download -> put --verify-md5
   -> delete local), so local footprint is one shard.
3. Local disk is a disposable cache (`/data05` primary; 872 GB free at open,
   other disks >=93% full). Stage with `get` + MD5; record HDFS URI, revision and
   hashes in the run manifest.
4. A run counts only after its outputs are on HDFS. The resume gate wipes the
   local cache first.
5. Auth: `~/.local/bin/waystone-auth-keepalive.sh` cron every 2h
   (kinit -R + waystone auth keepalive). TGT renew-until limits apply.
6. Python/CUDA in rootfs; host shell only orchestrates the waystone CLI.

Space is not allowed to block the ladder: if local space is short, evict
already-published artifacts first, then stream per-shard.

## Gates per rung

config dry-run -> HF logits parity -> LoRA vs full sanity (rung 0 only) ->
resume from HDFS after cache wipe -> dev pass@1 improvement vs base (2 seeds) ->
>=10 steady steps throughput/memory -> outputs on HDFS.

## Environment caveat

At open, another user's job occupies all 8 GPUs (~84 GiB each, 70-95% util).
Rungs 0-2 share the GPUs; throughput numbers taken while shared are labelled
`shared_gpu` and are not performance evidence. Rung 4 needs exclusive GPUs.

## Claim labels

Smoke = plumbing. Countdown pass@1 deltas need 2 seeds (`replicated_eval`).
Performance claims require exclusive GPUs.
