# N01b brief: fix the mixed CUDA toolchain in the native L0 rootfs, then build DeepEP v2 + torchao (finish N01)
Read `native-layering.md`, `env-facts.md` (MANDATORY), `task-r01-report.md`, `task-r01b-report.md`, `task-n01-report.md`,
`scripts/rootfs/{build_rootfs.sh,native_variant.lock}`, `experiments/glm53_native/`. Do NOT commit. Never modify the live
legacy rootfs or `glm53_train`; publish a NEW native store entry; never kill other jobs.
Root cause (verified 2026-10-14 ~20:15Z inside the native store rootfs):
- pip CUDA components are mixed: `nvidia-cuda-nvcc 13.2.86` (nvcc + ptxas: max PTX 9.2), `nvidia-cuda-nvrtc/runtime
  13.2.86`, but `nvidia-nvvm 13.4.92` (cicc emits PTX 9.4) and `nvidia-cuda-crt 13.4.92` -> `ptxas fatal: Unsupported
  .version 9.4; current version is '9.2'`. Unpinned deps floated to 13.4.
- `which -a ptxas` -> `/usr/local/cuda/bin/ptxas` before `/opt/cuda-synth/bin/ptxas`; check its version and PATH order.
- System pip in that rootfs lists `triton 3.7.1`; the training venv uses triton 3.9.0+gitfc8d31a5 -- make sure builds use
  the venv's torch/triton.
Do:
1. In `native_variant.lock` / the native Dockerfile layer: pin ONE consistent CUDA release for every nvidia-cuda-* /
   nvidia-nvvm / crt / cccl / nvrtc / runtime / cublas pip component, matching what torch 2.16.0.dev20261013+cu132 was built
   with (13.2.x) unless a newer consistent set is required -- justify. Ensure `/opt/cuda-synth` is the single CUDA_HOME
   and first on PATH for builds; nvcc, cicc, ptxas, nvlink, fatbinary all from the same release. Add a build-time probe in the
   Dockerfile: compile a trivial sm_100 kernel with nvcc and fail the image build on any mismatch. Keep the default
   variant recipe byte-identical. Update/extend the rootfs shell tests.
2. Rebuild + publish a new native store entry; select it; record the store id; re-run R01b's glm53_ep verify + GLM tests.
3. Build DeepEP v2 (pinned commit, TORCH_CUDA_ARCH_LIST=10.0) and torchao (source commit compatible with torch 2.16 +
   Triton 3.9) via `experiments/glm53_native/build_native.sh`, publish to HDFS, install via install_native.sh; verify
   `from deep_ep import ElasticBuffer`, DeepEP intranode test (2 GPUs; 8 if free), TorchTitan `qwen3_moe_deepep` EP=4 a few
   steps vs all-to-all (step-1 bitwise per recipe), torchao MXFP8 grouped GEMM import + tiny numeric check, and that
   TorchTitan's optional quantization imports no longer break. Co-run: <= 30 GiB per process.
Report `sdd/task-n01b-report.md` with store id, CUDA component versions, wheel build keys + HDFS URIs, test results.
