# R01 report: native L0 rootfs plus declared `glm53_ep` L1 runtime

Date: 2026-10-14

Surface: repo-local GLM Flash SFT runtime infrastructure. This work did not
change core Trainer behavior or online RL. It added an opt-in native rootfs
variant and a separate `glm53_ep` Python environment; it did not mutate the live
legacy rootfs or the live `glm53_train` venv.

## Outcome

R01 is complete as an implementation/audit pass, but `glm53_ep` is not yet
train-ready. L0 and most L1 runtime checks pass; current blockers are package
compatibility issues with Triton 3.9 import-time autotune validation in
`fla-core==0.5.2` and `torchao==0.18.0`.

Native L0 store selection:

```text
/data05/philip.yang/torchtitan-rootfs-store/content/rootfs-e0c18cff68fc8ed06ca494cf00c807cca3d714819e00d5b824d89269079f4ec5
```

Canonical launch pattern:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store \
  -- bash -lc 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && ...'
```

## Changes

- Added `scripts/rootfs/build_rootfs.sh --variant default|native`.
- Added `scripts/rootfs/native_variant.lock` with pinned native L0 additions:
  `ccache=4.9.1-1`, CUDA pip packages, and CUTLASS `v4.8.0`.
- Preserved the default rootfs recipe as the default path. The native variant is
  opt-in and uses Docker tag `torchtitan-rootfs:native` unless a tag is passed.
- Native L0 adds `ccache`, CUDA 13.2 headers/libs, unversioned linker symlinks
  for `libcublas.so`, `libcublasLt.so`, `libcudart.so`, and `libnvrtc.so`, plus
  dedicated CUTLASS headers under `/opt/cutlass` and `/opt/cuda-synth/include`.
- Store manifests now use a deterministic staged-rootfs tree digest rather than
  hashing only `/bin/bash`, so native/default store entries cannot collide.
- Added `glm53_ep` runtime resolution in `scripts/rootfs/runtime_env.sh` and
  plan validation in `scripts/rootfs/verify_runtime_env.py`.
- Added focused rootfs tests for native manifest metadata and `glm53_ep` bwrap
  plan/runtime config behavior.
- Added `experiments/glm53_flash_sft_ladder/runtime/` with:
  `requirements.direct.txt`, hash-locked `requirements.lock`,
  `sync_python_env.sh`, `verify_runtime.py`, `pyproject.toml`, `mise.toml`, and
  `README.md`.
- Updated `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` with the
  `glm53_ep` launch line, store id, evidence path, and current blockers.

## L0 audit

The prior/live stack had torch `2.16.0.dev20261013+cu132`, Triton
`3.9.0+gitfc8d31a5`, `fla-core==0.5.2`, `nvidia-nvshmem-cu13==3.7.2`, B200
`sm_100`, RDMA/NUMA headers, and Nsight CLIs. Missing or insufficient pieces
for native C++/CUDA extension builds were:

- `ccache` missing from the rootfs.
- Dedicated CUTLASS headers missing from `/opt/cuda-synth/include/cutlass`.
- Unversioned CUDA user-library linker names missing for cublas/nvrtc/runtime.

Final native L0 audit receipt:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01/native_l0_audit.json
```

Key passed checks:

- `/opt/cuda-synth/bin/nvcc`: CUDA 13.2, `V13.2.86`.
- `ccache --version`: `4.9.1`.
- CUDA headers present: `cuda_runtime.h`, `cublas_v2.h`, `nvrtc.h`.
- CUTLASS header present: `/opt/cuda-synth/include/cutlass/cutlass.h`.
- CUDA libs/symlinks present: `libcudart.so`, `libcublas.so`,
  `libcublasLt.so`, `libnvrtc.so`.
- RDMA/NUMA headers present: `infiniband/verbs.h`, `rdma/rdma_cma.h`,
  `numa.h`.
- Linker probe passed for `-lcublas -lnvrtc -lcudart`.
- Linker probe passed for `-lcuda` through the bound host driver path.

## L1 sync

Sync command:

```bash
experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store
```

Sync succeeded and wrote:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01/glm53_ep-python-env-report.json
.scratch/glm53-flash-sft-ladder/evidence/r01/glm53_ep-freeze.txt
```

Important receipt fields:

- env: `glm53_ep`
- venv: `/project/venvs/b200-runtime`
- rootfs digest:
  `sha256:e0c18cff68fc8ed06ca494cf00c807cca3d714819e00d5b824d89269079f4ec5`
- lock sha256:
  `c565cf43c532a84c639ae9e7cca44317ffe7673fd2614c03c30c391ce7cf886b`
- direct sha256:
  `1dcac00bdd6bdf36ff8c64dc36d959bc52730a03a56c7f6ac219328dd98654b9`
- `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`

`flash-attn==2.8.3.post1` needed special handling. With this torch nightly,
building the CUDA extension from source reaches torch headers that use C++20
symbols while upstream flash-attn hardcodes `-std=c++17`. The legacy
`glm53_train` flash-attn CUDA extension is also not importable with this torch
nightly due an undefined torch/cuda symbol. `glm53_ep` therefore keeps the
pinned package identity but skips the flash-attn CUDA extension build.

Freeze diff:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01/glm53_train-to-glm53_ep-freeze.diff
```

Summary: `glm53_train` freeze had 304 lines; `glm53_ep` freeze has 162 lines.
`glm53_ep` keeps the GLM/Torch/CUDA stack and adds `torchao==0.18.0`, but omits
many unrelated legacy packages from `glm53_train`. Some resolver-selected
support packages are newer than the legacy freeze.

## Runtime verification

Command:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store \
  -- bash -lc 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && python experiments/glm53_flash_sft_ladder/runtime/verify_runtime.py --report .scratch/glm53-flash-sft-ladder/evidence/r01/glm53_ep-runtime-verification.json'
```

Receipt:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01/glm53_ep-runtime-verification.json
```

Passed:

- torch `2.16.0.dev20261013+cu132`
- torch CUDA `13.2`
- `torch.cuda.is_available() == true`
- device 0 capability `[10, 0]`, `NVIDIA B200`
- Triton import
- `torchao.prototype.mx_formats` import
- `nvidia.nvshmem` import
- CUDA headers/libs under `/opt/cuda-synth`
- `/opt/cuda-synth/bin/nvcc` CUDA 13.2 `V13.2.86`

Failed blocker:

```text
fla.ops.kda import fails before KDA itself imports.
Trace: fla.ops.__init__ -> fla.ops.gated_delta_rule -> fla.ops.cp.chunk_delta_h:321
Error: Triton autotune key contains BT, but the kernel argument list omits BT.
```

`fla-core==0.5.2` is the latest PyPI release, so there was no clean L1 package
bump to try.

## Test results

Syntax checks:

```text
bash -n scripts/rootfs/build_rootfs.sh
bash -n experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh
```

Passed.

Focused rootfs/runtime tests:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_rootfs_build_store_shell.py tests/unit_tests/test_rootfs_runtime_env_shell.py tests/unit_tests/test_rootfs_bwrap_plan.py'
```

Result:

```text
31 passed, 1 skipped in 65.97s
```

GLM Flash tests:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store \
  -- bash -lc 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && pytest -q torchtitan/experiments/glm53_flash/tests'
```

Evidence:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01/glm53_flash_tests.log
```

Result: failed during collection before test bodies ran. Import path reaches
`torchao.prototype.moe_training.kernels.float8_rowwise.py:211`, where Triton 3.9
rejects `key=["num_elements"]` because `num_elements` is not a kernel argument.
`torchao==0.18.0` is the latest PyPI release.

## Two-step debug train

All eight B200s were busy at high utilization before launch; the snapshot is:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01/nvidia-smi-before-train.txt
```

To avoid disturbing active jobs, and because import/config setup already failed
without launching GPU work, the official two-step command was captured with
`COMM_BACKEND=fake`:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store \
  -- bash -lc 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && GLM53_FLASH_DEBUG_STEPS=2 GLM53_FLASH_DUMP_FOLDER=.scratch/glm53-flash-sft-ladder/evidence/r01/debug_train NGPU=1 LOG_RANK=0 COMM_BACKEND=fake MODULE=torchtitan.experiments.glm53_flash.config_registry CONFIG=glm53_flash_debugmodel ./run_train.sh'
```

Evidence:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01/glm53_debug_train_2step.log
.scratch/glm53-flash-sft-ladder/evidence/r01/glm53_debug_train_import_probe.log
```

Result: failed before training starts with the same torchao/Triton autotune-key
error (`num_elements` is not a kernel argument). A real GPU train was not
started because the system was busy and the import blocker happens before useful
GPU execution.

## Remaining blockers

1. `torchao==0.18.0` is incompatible with Triton 3.9 import-time autotune key
   validation for `torchao.prototype.moe_training.kernels.float8_rowwise`. This
   blocks TorchTitan import paths that traverse quantization/NVFP4/Dist-MoE,
   including GLM Flash test collection and debug training.
2. `fla-core==0.5.2` is incompatible with Triton 3.9 import-time autotune key
   validation for `fla.ops.cp.chunk_delta_h`. This blocks `fla.ops.kda` import
   in the runtime verifier.
3. `flash-attn==2.8.3.post1` source CUDA extension does not build cleanly
   against the torch nightly headers; the env currently records a pure Python
   package install with CUDA extension build skipped.

The next fix should be a package compatibility patch or pinned compatible
nightly set for Triton/torchao/fla-core before treating `glm53_ep` as a
validated training environment.
