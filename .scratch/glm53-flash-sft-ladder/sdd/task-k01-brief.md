# K01 brief: Triton top-k sparse causal attention (fwd+bwd) for GLM-style indexer selection
Read `native-layering.md` (Triton kernels are Python/JIT; live in `torchtitan/experiments/glm53_kernels/`), `env-facts.md`
(MANDATORY), `.claude/skills/kernel-numerics-verifier/SKILL.md` (follow it), `.claude/skills/local_compile_perf/SKILL.md`,
`torchtitan/experiments/glm53_arch_speedrun/elements/attention.py` (TopKSparseCausalAttention: `masked_dense` path and the
gather reference = the oracles), `torchtitan/experiments/glm53_flash/model.py` (sparse MLA + indexer + IndexPool semantics),
and check upstream first: deepseek-ai/FlashMLA sparse kernels, DeepSeek-V3.2 DSA / vLLM sparse attention, Attention Gym,
torch flex_attention -- reuse if one fits sm_100 + training (fwd AND bwd) + our shapes; cite what you checked.
Do NOT commit. 1 GPU, <= 24 GiB, never kill foreign processes or other workers' jobs.
Problem: speedrun arms with indexer top-k sparse attention run at ~0.05x B0 at seq 4096 (masked dense SDPA falls off the
flash path). Need a kernel: inputs q [B,H,L,D], k/v [B,H,L,D] (or MLA latent form if you also target GLM), per-query
selected key indices [B,L,K] (shared across heads, causal, K = top-k e.g. L/8; IndexPool selects pool granules of 4
consecutive keys -> support index granularity g in {1,4}), output = softmax over selected keys only. Forward + backward
(dq, dk, dv; indices are non-differentiable), bf16 inputs, fp32 accumulation, online softmax, LSE saved for bwd.
Deliver `torchtitan/experiments/glm53_kernels/sparse_topk_attention/` registered as a torch custom op (torch.library,
fake impl, autograd) so it works under torch.compile; numerics harness vs fp64/fp32 oracle and the bf16 masked_dense
baseline (fwd + all grads, seq 1024/4096, K in {L/16, L/8, L/4}, g in {1,4}); microbenchmark vs masked_dense and vs dense
causal flash SDPA at the same shape. Then wire it as `sparse_attention_impl="triton"` in the speedrun element (new option,
default unchanged) with a parity test, and report A4/A5 tokens/s at seq 4096 in the speedrun harness (2-step smoke).
Report `.scratch/glm53-flash-sft-ladder/sdd/task-k01-report.md`.
