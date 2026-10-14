# T17 brief: fast GLM MoE experts (grouped GEMM) -- remove the per-expert Python loop
Read `env-facts.md` (MANDATORY, incl. HDFS-STRICT INPUT RULE and waystone-only), `task-12-report.md` (EP prototype
`torchtitan/experiments/glm53_flash_ep/` maps GLM experts onto common `GroupedLinear` w13/w2 with bounded parity),
`torchtitan/models/common/{moe.py,token_dispatcher.py}`, `torchtitan/experiments/glm53_flash/model.py`. Do NOT commit.

Root cause (py-spy on a live 8-GPU rung-4 rank, 2026-10-14 18:35Z): every rank is ~100% CPU in
`GlmExperts.forward` (`glm53_flash/model.py` ~L272-296): a Python loop over up to 288 experts with `.nonzero()` +
`torch.where` host syncs and two tiny GEMMs each, x 42 MoE layers x 8 microbatches x (fwd + AC recompute + bwd).
Observed ~250 s per optimizer step for 32,768 tokens on 8x B200 (6 steps in 25 min). The same loop likely explains
T09c (6.3 min/step) and T09b slowness.

1. Add a grouped path for GLM routed experts selected by config/env `GLM53_FLASH_EXPERTS_IMPL in {loop, grouped}`,
   DEFAULT `loop` (a T08d run is live on 8 GPUs and its remaining legs re-import this file -- behavior must not change
   unless the env selects grouped). Grouped: sort token-slot pairs by expert on GPU (argsort / histc, no host sync), use
   `torch._grouped_mm` (or the common MoE grouped experts kernel used by qwen3/deepseek in this repo) with the SAME
   parameter tensors/layout (`gate_up_proj` [E,2F,D], `down_proj` [E,D,F]) so checkpoints, the state-dict adapter,
   the rank-local HF loader and LoRA policy are unchanged; same swiglu clamp semantics; weighted combine via
   scatter/index_add on GPU. No `.item()`/`.nonzero()`/Python-per-expert loops.
2. Tests (CPU where possible, small CUDA otherwise, <= 8 GiB GPU): grouped vs loop fwd + bwd (grads to x, LoRA-free
   base params) in fp32 (tight tolerance) and bf16 (state tolerance), incl. experts with zero tokens and top-8 routing;
   router/shared-expert unchanged; existing `torchtitan/experiments/glm53_flash/tests` still pass.
3. Throughput micro-benchmark: one GLM MoE layer at full width (288 experts, top-8, hidden/moe sizes from the
   glm53_flash flavor) on 1 GPU with 4096 tokens, fwd+bwd, loop vs grouped (ms/iter). Keep GPU memory <= 24 GiB; GPUs
   are shared with a live 8-GPU GLM run (do not disturb it; it uses ~136-155 GiB per GPU).
4. Do NOT launch full-model GLM training. Report `sdd/task-17-report.md` with parity numbers, speedup, and the exact
   env/override to enable it for rung 4 and for the T09 4-GPU packages.
