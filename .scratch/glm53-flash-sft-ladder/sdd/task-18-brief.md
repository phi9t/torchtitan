# T18 brief: DeepEP v2 on 8x B200 for GLM-5.3-Flash EP (use TorchTitan's built-in DeepEPTokenDispatcher)
Read `env-facts.md` (MANDATORY: rootfs for all Python/CUDA, `TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`, HDFS
inputs only via waystone + `lib/hdfs_strict.sh`), `torchtitan/models/common/token_dispatcher.py` (`DeepEPTokenDispatcher`
L820, `HybridEPTokenDispatcher` L955, `AllToAllTokenDispatcher`), `torchtitan/distributed/deepep/{deepep.py,hybridep.py}`,
`torchtitan_recipes/tests/models/qwen3.py::qwen3_moe_deepep` (+ `tests/integration_tests/h100.py` "qwen3_fsdp+deepep"),
`.github/scripts/install_deepep_v2.sh` (pinned DeepEP v2 commit 01dc3aaac82068020353dce2c302e38153c0bfaa), the PyTorch
blog "Enabling up to 41% faster pre-training: MXFP8 and DeepEP for DeepSeek-V3 on B200 with TorchTitan"
(https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/),
https://github.com/deepseek-ai/DeepEP, and `task-12-report.md` (GLM EP prototype `torchtitan/experiments/glm53_flash_ep/`).
Do NOT commit. Do not edit core; GLM-specific code goes in `torchtitan/experiments/glm53_flash_ep/` (T13 is also editing
that package for EP parity -- coordinate by adding new files/config entries only, do not rewrite T13's files) and
benchmarks/logs in `experiments/glm53_flash_ep_dispatch/` (logs/results gitignored).
1. Install DeepEP v2 at the pinned commit into the `glm53_train` rootfs env, adapting install_deepep_v2.sh for B200:
   `TORCH_CUDA_ARCH_LIST=10.0`, CUDA_HOME inside the rootfs, rdma-core/ibverbs libs (install into the rootfs if missing),
   NVSHMEM per DeepEP's requirements, `--no-build-isolation`, no torch version change (torch 2.16 nightly cu132 must stay).
   Network via `TORCHTITAN_ROOTFS_NETWORK=networked`. Record exact versions. Verify `from deep_ep import ElasticBuffer`.
2. Smoke: run TorchTitan's own `qwen3_moe_deepep` config (EP=4) for a few steps and compare loss vs the all-to-all
   backend (recipe docstring: step-1 bitwise, reduction-order drift after).
3. GLM: add a `glm53_flash_ep` config variant that selects `DeepEPTokenDispatcher` (EP=8, FSDP for non-expert params,
   grouped local experts), truncated-depth (first 8 layers) first: numerics vs all-to-all EP and vs the FSDP-only path
   (loss, grad norm, LoRA grads), then step time / tokens/s. Then measure full-depth only when the 8 GPUs are free.
4. Optional per the blog: MXFP8 grouped GEMM for frozen expert weights (torchao) -- measure, report, do not enable by default.
Co-run rules: a slow fallback 8-GPU GLM run (~136-155 GiB/GPU) may be live; keep <= 30 GiB per process, never kill it or
foreign processes; `priority_pause.sh` may pause you while T08d waits between legs. Report `sdd/task-18-report.md` with
install recipe, smoke parity, GLM EP numerics, ms/step and tokens/s for {FSDP-only loop, FSDP grouped, EP all-to-all,
EP DeepEP}, and a recommendation for rung 4.
