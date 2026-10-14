# N01 brief: native-kernel toolchain in the Insula rootfs for GLM custom kernels (after R01)
Blocked by: R01 (EP rootfs variant in scripts/rootfs/build_rootfs.sh; build on top of it, do not fork it).
Read `env-facts.md` (MANDATORY), R01's report and builder changes, `.claude/skills/` kernel skills
(`kernel-numerics-verifier`, `local_compile_perf`, `cuda_graph_trace_compaction`), `torchtitan/models/common/dist_moe/`
(CuTe DSL backend), `torchtitan/experiments/glm53_flash/model.py` (KDA, sparse MLA + indexer, IndexPool, mHC, grouped MoE).
Do NOT commit. Do not touch the live legacy rootfs; extend the EP variant and publish a new store entry.
Goal: a reasonable, reproducible setup for writing, building, testing and profiling custom kernels for the GLM work:
1. Toolchain in the EP rootfs variant, pinned in its lock: full CUDA 13 toolkit headers/libs matching torch's CUDA
   (nvcc, cuobjdump, nvdisasm), CUTLASS + CuTe DSL (nvidia-cutlass-dsl) at pinned versions, Triton matching torch,
   ninja, ccache (cache dir under the per-env state, not the image), Nsight Compute/Systems CLIs if redistributable
   (stop-job profiling only), clangd/compile_commands support.
2. A repo convention for native kernels: `torchtitan/experiments/glm53_kernels/` with (a) Triton and CuTe DSL kernels as
   plain Python, (b) C++/CUDA ops built via `torch.utils.cpp_extension` (load/JIT with a pinned build dir, plus an
   ahead-of-time build target), registered as torch custom ops (`torch.library`) so they compose with torch.compile and
   FSDP/DTensor, with fake/meta impls; sm_100 targets; one trivial example op per path end-to-end.
3. Test + verify harness: per-kernel fwd/bwd numerics vs a high-precision oracle and an eager baseline (follow the
   kernel-numerics-verifier skill), a microbenchmark template (cuda events, warmup, L2 flush), and how to run both inside
   the rootfs; CI-style script `experiments/glm53_kernels/run_kernel_checks.sh`.
4. Docs: `torchtitan/experiments/glm53_kernels/README.md` (layout, build, test, profile, sm_100 notes) and a short
   section in the rootfs README. Candidate first kernels to list (not implement): KDA chunk fwd/bwd, indexer top-k +
   IndexPool fused, sparse MLA gather attention, mHC Sinkhorn/HcPost fused, MoE permute/unpermute with DeepEP layout.
Report `sdd/task-n01-report.md` with versions, example op results, and the store id.
