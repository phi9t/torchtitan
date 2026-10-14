# Map: glm53-flash-sft-ladder

## Decisions
- D1 Task/dataset frozen: Countdown `formatting` arm + dev/iid/ood splits (spec).
- D2 Storage: waystone HDFS system of record; per-shard ingest; /data05 cache.
- D3 B200 only; no Primus.
- D4 LoRA rank 16, alpha 64, targets wqkv/wo/w13/w2 for Qwen rungs (champion knobs; not re-searched).
- D5 GLM indexer frozen during SFT (matches Z.ai RL practice; research sec. 3.2).

## Frontier
- T01 waystone storage wrappers + dataset v1 publish
- T02 rung 0: Qwen3 0.6B/1.7B LoRA SFT + eval + HDFS resume gate
- T03 HF ingest to HDFS: Qwen3-0.6B, Qwen3-1.7B, Qwen3-8B, Qwen3-30B-A3B, GLM-5.3-Flash BF16
- T04 DSv4 mHC HcPost axis bug: confirm vs DeepSeek reference

## Fog
- GPU contention from another user's job (perf numbers invalid while shared).
- 598 GiB GLM stage-down time unknown until measured at rung 2.
- Rootfs vLLM 0.27 / transformers 5.15 support for GLM-5.3-Flash (rung 4 eval).
- Side ticket S1 (user, not main quest): unified reliable auth (HDFS kinit keytab/vault, HF, GitHub, code.byted, bytedcli).

## Frozen/Resolved (2026-10-14)
- T04: DSv4 HcPost axis bug CONFIRMED vs official DSv4-Flash inference + GLM reference; fixed in core (96760e290).
- Storage perf: HF ~70-116 MB/s, HDFS put/get ~20-25 MB/s per stream. Decision D6: large models ingest with
  `--keep-local` (one HF download feeds HDFS + local cache); HDFS remains system of record, local is the training copy.
- Env: training needs torch nightly (rootfs env `glm53_train`); vLLM eval stays on system torch 2.13.
- D7 (2026-10-14 09:15Z): rung 4 under shared GPUs runs FSDP8 + training.enable_cpu_offload (host RAM 3.9 TB) + LoRA +
  FullAC: only one gathered layer (~14 GB bf16 for an MoE layer) per GPU at a time, fits next to the other user's 84 GiB.
  Throughput from this setup is not performance evidence. Exclusive-GPU FSDP-only (73 GiB/GPU sharded) or FP8 frozen base
  are the performance follow-ups. EP token dispatch is not implemented in the seam (gap from T07a).
- Incident 2026-10-14 12:56Z: T08 mode B (enable_cpu_offload + HF initial load) made every one of 8 local ranks materialize
  the full ~600 GB model in host RAM (RSS +13 GB/20s/rank, 2.15 TB used). Orchestrator killed the ranks before host OOM.
  D7 revised: mode B is unsafe on a single host as implemented; use mode A (meta init, GPU-sharded params, per-rank shard
  load). Guard: `experiments/glm53_flash_sft_ladder/host_ram_guard.sh 600` kills GLM ranks if available RAM < 600 GB.
