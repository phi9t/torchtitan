# Ledger: glm53-flash-sft-ladder

- T00 rootfs dep sync (orchestrator): complete -- requirements.txt pins installed; torch-nightly training env `glm53_train`; see sdd/env-facts.md
- T01 waystone wrappers + dataset v1: complete (traecli) -- countdown-sft/v1 + Qwen3-0.6B on HDFS, stage round trip MD5-verified; claim smoke
- T02 rung 0 Qwen3 0.6B/1.7B: complete -- PASS all gates; resume bitwise; dev pass@1 base->adapter 0.6B 0.004->0.254/0.326, 1.7B 0.024->0.370/0.228 (replicated_eval)
- T03 HF ingest to HDFS (+ kept local copy): complete 10:23Z -- Qwen3-0.6B/1.7B/8B/30B-A3B and GLM-5.3-Flash-BF16 (a5b45eb) on HDFS with _COMPLETE + local staged copies on /data05
- T04 mHC HcPost: complete -- CONFIRMED; fixed in 96760e290 (sum over source axis), DSv4 debug trains
- T05 rung 1 Qwen3 8B: in-progress (traecli)
- T06 rung 2 Qwen3 30B-A3B: blocked by T05, T03(30B)
- T07a GLM-5.3-Flash model seam: complete (bf5b4e8c5) -- 11 oracle tests, debug+LoRA 2-GPU smoke; gaps: EP dispatch, TP/CP, real-weight parity (-> T07c)
- T07b rung 3 mini hybrid learnability (overfit gate) on countdown: in-progress (traecli)
- T07c real-weight truncated parity vs executable HF oracle + GLM tokenizer/data: in-progress (traecli, waits on T03 GLM)
- T08 rung 4 GLM-5.3-Flash LoRA: blocked by T06, T07b, T03(GLM), exclusive GPUs
