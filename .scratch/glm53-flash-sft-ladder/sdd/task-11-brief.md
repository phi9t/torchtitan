# T11 brief: expert-parallelism deep dive for GLM-5.3-Flash (phase 1: analysis + design; no GPU training)

Repo: /data02/home/philip.yang/workspace/torchtitan (branch ultron/mainline). Do NOT commit. Read `env-facts.md` first.
Phase 1 is read/analyze/design plus CPU or fake-backend checks only. Do NOT edit existing files under
`torchtitan/experiments/glm53_flash/` (T08d owns them) or `torchtitan/models/common/` (core). GPU use: none, except
`COMM_MODE=fake_backend` / `local_tensor` config checks (no real GPUs needed).

Context: GLM-5.3-Flash has 288 routed experts top-8 + 1 shared, layers 0-2 dense, 42 MoE layers, ~313B params; the
experiment sharding (`torchtitan/experiments/glm53_flash/sharding.py`) is FSDP-only with no EP dispatch. The 8-GPU LoRA
probe peaked at 131.56 GiB/GPU, with the largest FSDP all-gather unit 13.8 GiB (an MoE layer). LoRA targets include
experts (see `apply_lora_trainable_policy` and LoRA config).

Deliverable: `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md` with cited code paths, covering:
1. How TorchTitan does EP today: `torchtitan/models/common/{moe.py,moe_sharding.py,MOE_SHARDING.md,token_dispatcher.py}`,
   `dist_moe/` (runtime, routed_experts, mxfp8), and how deepseek_v3 / deepseek_v4 / qwen3 MoE / kimi models wire EP
   (mesh axes, ETP, dp_shard_in_ep / sparse FSDP mesh, all-to-all dispatch, grouped GEMM, load-balance / aux loss).
2. What GLM needs to plug in: map GLM MoE module/param names and grouped expert layout onto that machinery; LoRA on
   experts under EP (where LoRA A/B for experts live, grad placements); router stats; state-dict adapter + rank-local
   loader interaction (each EP rank loads only its experts' HF keys -- huge load-time and memory win).
3. Memory and comm model for EP in {1,2,4,8} x FSDP on 8 and 4 B200s: per-GPU static bytes, all-gather sizes,
   all-to-all volume per step at seq 512 and 32768 tokens/step, and how EP changes the 4-GPU feasibility picture vs the
   T09 directions (no offload/quant). Use real counts from `full_size_param_counts.json` (T07b evidence).
4. Risks: capacity/dropping vs dropless, top-8 with 288 experts / 8 ranks = 36 experts/rank, load imbalance on
   Countdown, determinism/bitwise resume under EP, interaction with KDA/sparse layers (none expected) and MTP.
5. A concrete phase-2 implementation plan as tickets (files to add in a new experiment package or a proposed GLM
   sharding change, tests incl. fake-backend EP config check, numeric parity EP vs FSDP-only on the truncated-depth
   config, then full-depth throughput/memory). Mark which need GPUs.
Optionally prototype the GLM EP sharding config in a NEW package `torchtitan/experiments/glm53_flash_ep/` and validate it
with `COMM_MODE=fake_backend` on the debug/mini flavor. Report in `sdd/task-11-report.md`.
