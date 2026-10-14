# N01b report: native L0 CUDA toolchain fix, DeepEP v2, and torchao

Date: 2026-10-14

Surface: repo-local GLM/native runtime infrastructure. Core Trainer behavior,
model math, the live legacy rootfs, and the live `glm53_train` venv were not
changed. No commit was made.

## Outcome

N01b fixed the native L0 CUDA compiler/toolchain blocker that stopped N01:

- The mixed compiler stack was replaced with a single CUDA 13.2 compiler
  closure under `/opt/cuda-synth`.
- A new immutable native rootfs store entry was built and selected.
- `glm53_ep` was resynced and verified against the new store.
- DeepEP v2 and torchao were built, published to HDFS through Waystone, staged
  back through the HDFS-strict installer, installed into `glm53_ep`, and
  verified by import/kernel smoke checks.
- Qwen3 EP=4 all-to-all and DeepEP two-step training smokes both complete under
  deterministic mode.

The remaining caveat is parity scope: the Qwen3 DeepEP smoke did not establish
step-1 bitwise identity with all-to-all. The available rounded console scalars
are very close but not identical, so this report treats that as a smoke pass
for DeepEP training, not as a bitwise proof.

## Rootfs Store

Selected native store:

```text
store_id: rootfs-0239a373638b39bca200ff29a39aa21b1737fadb267711e6b1e15572ffff5dd9
store_path: /data05/philip.yang/torchtitan-rootfs-store/content/rootfs-0239a373638b39bca200ff29a39aa21b1737fadb267711e6b1e15572ffff5dd9
digest: sha256:0239a373638b39bca200ff29a39aa21b1737fadb267711e6b1e15572ffff5dd9
```

Evidence:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/rootfs_build_manifest_cuobjdump.json
.scratch/glm53-flash-sft-ladder/evidence/n01b/rootfs_selected_cuobjdump.json
.scratch/glm53-flash-sft-ladder/evidence/n01b/native_rootfs_rebuild_cuobjdump.log
```

Native L0 pip CUDA pins:

```text
nvidia-cuda-nvcc==13.2.86
nvidia-cuda-cuobjdump==13.2.86
nvidia-nvvm==13.2.86
nvidia-cuda-crt==13.2.86
nvidia-cuda-cccl==13.2.86
nvidia-cuda-runtime==13.2.86
nvidia-cuda-nvrtc==13.2.86
nvidia-cublas==13.2.2.2
```

Build-time probes now require:

- `nvcc`, `cuobjdump`, `ptxas`, `nvlink`, `fatbinary`, and `cicc`;
- `/opt/cuda-synth/bin` before `/usr/local/cuda/bin`;
- every required CUDA tool to resolve from `/opt/cuda-synth`;
- CUDA tool versions containing `13.2`;
- a trivial `nvcc -arch=sm_100` compile;
- CUTLASS and CuTe headers under `/opt/cuda-synth/include`.

The default rootfs variant recipe was left outside this native-only path.

## Runtime Verification

`glm53_ep` verification passed on the selected rootfs:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/glm53_ep-runtime-verification-cuobjdump.json
ok: true
torch: 2.16.0.dev20261013+cu132
triton: 3.9.0+gitfc8d31a5
device: NVIDIA B200, sm_100
```

CUDA tool paths and versions from that verifier:

```text
nvcc:      /opt/cuda-synth/bin/nvcc      CUDA 13.2 V13.2.86
cuobjdump: /opt/cuda-synth/bin/cuobjdump CUDA 13.2 V13.2.86
ptxas:     /opt/cuda-synth/bin/ptxas     CUDA 13.2 V13.2.86
nvlink:    /opt/cuda-synth/bin/nvlink    CUDA 13.2 V13.2.86
fatbinary: /opt/cuda-synth/bin/fatbinary CUDA 13.2 V13.2.86
```

L1 cublas note: `glm53_ep` still installs `nvidia-cublas==13.4.1.3` in the
training venv because torch `2.16.0.dev20261013+cu132` depends on
`cuda-toolkit[cublas]==13.2.2`, whose resolver requirement is
`nvidia-cublas==13.4.1.3.*`. Attempting to force `nvidia-cublas==13.2.2.2`
in L1 is unsatisfiable; see:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/requirements_lock_regen_cublas132.log
```

The N01b compiler/root cause was the PTX-producing toolchain
(`nvcc`/`nvvm`/`crt`/`ptxas`/`cuobjdump`), and those are all pinned to CUDA
13.2 and verified from `/opt/cuda-synth`.

## Native Wheels

DeepEP v2:

```text
source: https://github.com/deepseek-ai/DeepEP.git
commit: 01dc3aaac82068020353dce2c302e38153c0bfaa
patch: experiments/glm53_native/patches/deepep_v2/0001-add-nccl-library-dir.patch
build_key: 6b7e14e72362f6ad219ba107036eebb765ee4dd3c03bd43c8be5535f75f47e45
HDFS_URI: hdfs://harunava/user/tiger/waystone/torchtitan/native-wheels/deepep_v2/6b7e14e72362f6ad219ba107036eebb765ee4dd3c03bd43c8be5535f75f47e45
manifest_md5: 616464c9be66a87090aea61222ed3660
wheel: deep_ep-2.1.0+local-cp312-cp312-linux_x86_64.whl
wheel_sha256: 7c679ee793650d7741576e2f53f95af555641bee560656725cc45d88aed154a5
```

torchao:

```text
source: https://github.com/pytorch/ao.git
commit: 9d0c4946b11af5e484f3ac36d61ec970c40c68cf
build_key: 09637433cfe6f6f49fb4a2c36274a1bb77f96373012e15f3683216d9a90386cb
HDFS_URI: hdfs://harunava/user/tiger/waystone/torchtitan/native-wheels/torchao/09637433cfe6f6f49fb4a2c36274a1bb77f96373012e15f3683216d9a90386cb
manifest_md5: 9614b784ac23b09e857d623a8496d84d
wheel: torchao-0.19.0+git9d0c4946b-cp310-abi3-linux_x86_64.whl
wheel_sha256: 2e712ef1c24b8eb44e25bce28dafc131249951404b24bf486d92ff5bab45b282
```

Both wheels record:

```text
rootfs_env: glm53_ep
rootfs_store_id: rootfs-0239a373638b39bca200ff29a39aa21b1737fadb267711e6b1e15572ffff5dd9
CUDA_HOME: /opt/cuda-synth
TORCH_CUDA_ARCH_LIST: 10.0
torch: 2.16.0.dev20261013+cu132
```

## Verification

Focused rootfs/unit checks:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/rootfs_shell_pytest_host_py3_final.log
bash -n scripts/rootfs/build_rootfs.sh scripts/rootfs/runtime_env.sh \
  experiments/glm53_native/build_native.sh \
  experiments/glm53_native/install_native.sh \
  experiments/glm53_native/run_kernel_checks.sh \
  experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh
python3 -m pytest -q \
  tests/unit_tests/test_rootfs_build_store_shell.py \
  tests/unit_tests/test_rootfs_runtime_env_shell.py \
  tests/unit_tests/test_rootfs_bwrap_plan.py
32 passed in 73.96s
```

An earlier run of the same host-side bwrap plan tests from inside bwrap failed
one privileged-mount assertion because `/lib/modules` was not visible inside
the nested rootfs. The host-side Python 3 run above is the valid owning context
for that suite.

GLM runtime suites in `glm53_ep`:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/glm53_ep_glm53_flash_tests.log
26 passed, 1 skipped in 15.52s

.scratch/glm53-flash-sft-ladder/evidence/n01b/glm53_ep_glm53_flash_ep_tests.log
16 passed in 6.37s

.scratch/glm53-flash-sft-ladder/evidence/n01b/glm53_ep_glm53_arch_speedrun_tests.log
33 passed in 66.34s
```

Two-step GLM debug train:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/glm53_ep_debug_train_2step.log
step 1 loss 6.25553 grad_norm 5.0312
step 2 loss 5.08131 grad_norm 4.0312
Training completed
```

DeepEP focused intranode test:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/deepep_elastic_intranode_2gpu_cuobjdump_store_pipefail.log
CUDA_VISIBLE_DEVICES=0,2
DEEPEP_ELASTIC_INTRAGPU_2GPU_STATUS=pass
```

torchao/MXFP8 import and tiny numeric probe:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/l2_imports_torchao_mxfp8_probe_cuobjdump_store.log
deep_ep.ElasticBuffer imported
torchao.prototype.moe_training.kernels.mxfp8 imported
torchtitan.quantization.mxfp8.MXFP8Linear imported
torchtitan.quantization.nvfp4.NVFP4Linear imported
grouped GEMM out_shape: [256, 128]
a_grad_finite: true
b_grad_finite: true
loss: 126.96907043457031
max_abs_delta_vs_bf16_grouped_mm: 2.171875
```

Qwen3 EP=4 two-step deterministic smokes:

```text
.scratch/glm53-flash-sft-ladder/evidence/n01b/qwen3_alltoall_ep4_det_2step.log
.scratch/glm53-flash-sft-ladder/evidence/n01b/qwen3_deepep_ep4_det_2step.log
.scratch/glm53-flash-sft-ladder/evidence/n01b/qwen3_ep4_det_metric_compare.json
```

Both runs used `debug.seed=42`, `debug.deterministic=True`, EP=4, DP shard=4,
and completed two optimizer steps. Parsed rounded console metrics:

```text
all-to-all step 1: loss 7.72119, grad_norm 23.2793
DeepEP    step 1: loss 7.72118, grad_norm 23.2767
delta: loss -0.00001, grad_norm -0.0026

all-to-all step 2: loss 6.89898, grad_norm 7.7720
DeepEP    step 2: loss 6.89897, grad_norm 7.7713
delta: loss -0.00001, grad_norm -0.0007
```

Because the scalar output differs even at rounded console precision and these
scratch runs did not emit full-precision TensorBoard scalars, this is not a
bitwise parity proof. It is a DeepEP training smoke pass with small scalar
divergence from all-to-all.

## Notes

- The H100 integration harness was not used for the final Qwen3 check because
  importing the whole harness currently pulls a qwen3.5 multimodal path and
  fails before training with a pre-existing `torchvision.nms` attribute error.
  The final Qwen3 checks therefore used a scratch-only config module under the
  N01b evidence directory.
- No other jobs were killed or signaled.
- The new native store was published/selected instead of modifying the live
  legacy rootfs.
