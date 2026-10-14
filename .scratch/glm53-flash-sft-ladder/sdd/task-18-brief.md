# T18 brief: DeepEP-class dispatch for GLM-5.3-Flash on 8x B200 -- evaluate DeepEP vs all-to-all vs Dist-MoE
Read `env-facts.md` (MANDATORY), `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`, `task-12-report.md`,
`torchtitan/models/common/{token_dispatcher.py,moe.py,MOE_SHARDING.md}`, `torchtitan/models/common/dist_moe/README.md`.
Do NOT commit. Work only in a new package `torchtitan/experiments/glm53_flash_ep_dispatch/` and
`experiments/glm53_flash_ep_dispatch/` (benchmarks, logs gitignored). Do not edit core or other tickets' packages; propose
core hooks as patches in the report.
1. Primary-source check: does deepseek-ai/DeepEP support SM100/B200 (normal intranode NVLink kernels, low-latency
   kernels), at what commit, with which NVSHMEM/CUDA/torch requirements? Same for any newer alternative you find
   (e.g. NVIDIA/Megatron "hybrid EP", pplx-kernels, UCCL-EP, Dist-MoE). Cite repos/commits.
2. If DeepEP supports SM100: build/install it inside the `glm53_train` rootfs env with uv (no torch version change; mise
   for missing toolchains), and run its intranode tests on 8 GPUs only when the GPUs allow (co-run caps below).
3. Microbenchmark at GLM-5.3-Flash shapes (hidden size and moe_intermediate from the glm53_flash flavor, 288 experts,
   top-8, EP=8, bf16, tokens/rank in {512, 4096, 16384}): dispatch + grouped GEMM + combine, fwd and bwd, for
   (a) TorchTitan AllToAllTokenDispatcher + torch._grouped_mm, (b) DeepEP normal kernels, (c) Dist-MoE if installable
   (coordinate: T16 installs dist_moe; reuse its install, do not fight over the env). Report ms, achieved NVLink GB/s.
4. Write a TokenDispatcher adapter for the winner that plugs into the glm53_flash_ep MoE (proposed patch or subclass in
   your package) with a numerics test vs the all-to-all path on a tiny config.
Co-run rules: a live 8-GPU GLM job uses ~136-155 GiB/GPU; keep <= 30 GiB per process, never kill it or foreign processes;
`priority_pause.sh` may pause you while T08d waits between legs. Report `sdd/task-18-report.md` with a recommendation.
