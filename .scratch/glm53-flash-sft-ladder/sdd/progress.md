# Ledger: glm53-flash-sft-ladder

- T00 rootfs dep sync (orchestrator): complete -- requirements.txt pins installed; torch-nightly training env `glm53_train`; see sdd/env-facts.md
- T01 waystone wrappers + dataset v1: complete (traecli) -- countdown-sft/v1 + Qwen3-0.6B on HDFS, stage round trip MD5-verified; claim smoke
- T02 rung 0 Qwen3 0.6B/1.7B: in-progress (traecli)
- T03 HF ingest to HDFS (+ kept local copy): Qwen3-1.7B complete; Qwen3-8B -> Qwen3-30B-A3B -> GLM-5.3-Flash-BF16 running (logs/t03.log)
- T04 mHC HcPost: complete -- CONFIRMED; fixed in 96760e290 (sum over source axis), DSv4 debug trains
- T05 rung 1 Qwen3 8B: blocked by T02, T03(8B)
- T06 rung 2 Qwen3 30B-A3B: blocked by T05, T03(30B)
- T07a GLM-5.3-Flash model code + debug/mini flavors + parity: in-progress (traecli)
- T07b rung 3 mini hybrid learnability (overfit gate) on countdown: blocked by T07a, T02
- T08 rung 4 GLM-5.3-Flash LoRA: blocked by T06, T07b, T03(GLM), exclusive GPUs
