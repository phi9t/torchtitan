# Ledger: glm53-flash-sft-ladder

- T00 rootfs dep sync (orchestrator): complete -- requirements.txt pins installed; torch-nightly training env `glm53_train`; see sdd/env-facts.md
- T01 waystone wrappers + dataset v1: complete (traecli) -- countdown-sft/v1 + Qwen3-0.6B on HDFS, stage round trip MD5-verified; claim smoke
- T02 rung 0 Qwen3 0.6B/1.7B: complete -- PASS all gates; resume bitwise; dev pass@1 base->adapter 0.6B 0.004->0.254/0.326, 1.7B 0.024->0.370/0.228 (replicated_eval)
- T03 HF ingest to HDFS (+ kept local copy): complete 10:23Z -- Qwen3-0.6B/1.7B/8B/30B-A3B and GLM-5.3-Flash-BF16 (a5b45eb) on HDFS with _COMPLETE + local staged copies on /data05
- T04 mHC HcPost: complete -- CONFIRMED; fixed in 96760e290 (sum over source axis), DSv4 debug trains
- T05 rung 1 Qwen3 8B: complete -- PASS; parity top-1 1.0; resume PASS; dev pass@1 base 0.028 -> 0.474/0.488 (replicated_eval)
- T06 rung 2 Qwen3 30B-A3B MoE: in-progress (traecli)
- T07a GLM-5.3-Flash model seam: complete (bf5b4e8c5) -- 11 oracle tests, debug+LoRA 2-GPU smoke; gaps: EP dispatch, TP/CP, real-weight parity (-> T07c)
- T07b rung 3 mini hybrid learnability: complete -- overfit gate PASS (indexer + dense ablation, lr=0 flat, curves differ); seq 4k/8k finite; full-size plan 86.9 GiB/GPU @8 (analytical); found: LoRA path does not freeze base (-> T08 req 0)
- T07c real-weight parity: complete -- fp32 slices 0-3 (1e-5) and 20-27 (top-1 1.0, KL 2e-4) PASS, selection masks exact; key coverage 0 unknown; GLM chat/mask data path; OPEN: bf16 20-27 top-1 0.43 (-> T08 req 0b)
- T08 rung 4 attempt 1: BLOCKED -- LoRA freeze fixed+tested; full HF load materializes full model per rank (host RAM incident 12:56Z); bf16 slice drift persists on countdown prompts. -> T08b (sharded loader + trainable-only ckpt + run), T08c (bf16 diag)
- T08c bf16 diag: complete -- TT-bf16 and HF-bf16 equally far from fp32 truth (top-1 ~0.03-0.09 on the truncated mid slice); not a TT bug; fp32 parity remains the identity gate

## 14:25Z
- T08b PARTIAL -> committed db23be429: rank-local sharded HF load (21.7 s, rank RSS 3.75 GiB), trainable-only checkpoint, full 8xB200 one-step LoRA probe (loss 1.676, 131.6 GiB/GPU). claim: smoke.
- T08d launched: full rung-4 run (seed 42 + resume via HDFS + seed 43), eval base vs 2 adapters, publish. Gated on all 8 GPUs >= 150 GiB free; foreign root proc observed on GPU 3 (~54 GB) at 14:17Z.
- 14:35Z user asked for all three 4-GPU directions: T09a weight streaming (FSDP2 CPU offload + rank-local loader), T09b FP8 frozen base quantized at load, T09c bf16 overhead cuts. Launched concurrently in separate packages; full-depth launches serialized by flock, yield to T08d.
- User: next step after T08d/T09 is an expert-parallelism deep dive; capture all work in docs. Launched T10 (campaign doc `docs/research/2026-10-14-glm-5.3-flash-sft-ladder.md`, TODO placeholders for in-flight tickets; refresh after T06/T08d/T09 land) and T11 (EP deep dive phase 1: analysis + design doc `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`, fake-backend only; phase 2 GPU work queued after T08d/T09).
