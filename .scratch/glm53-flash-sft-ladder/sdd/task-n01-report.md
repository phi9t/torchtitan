# N01 report: L2 native-artifact layer and GLM kernel package convention

Date: 2026-10-14

Surface: repo-local GLM native/runtime infrastructure and
`torchtitan/experiments/glm53_kernels/`. Core Trainer behavior was not changed.
No commit was made.

## Outcome

N01 is partially complete.

Implemented:

- `experiments/glm53_native/` L2 native package lock and build/install scripts.
- HDFS publish/stage/install flow through Waystone under
  `torchtitan/native-wheels/<pkg>/<key>/`, with `MANIFEST.jsonl`, per-file md5,
  `_COMPLETE`, local `.staged.json`, and installed build-key receipt.
- Package entries for `deepep_v2`, `torchao`, and `glm53_kernel_examples`.
- `torchtitan/experiments/glm53_kernels/` convention docs and checks for:
  - Triton Python JIT example with `torch.library`, fake impl, autograd, compile
    smoke, fwd+bwd numerics, and microbench template;
  - CuTe DSL package/decorator convention check against CUTLASS DSL 4.8.0;
  - C++ `torch.library` native wheel example built through `build_native.sh`.
- `torchtitan.experiments.glm53_flash_sft_ladder.native_keys`, a helper that
  writes installed native build keys into training `attempt.json`.

Blocked:

- DeepEP v2 CUDA wheel did not build.
- torchao source CUDA wheel did not build.
- Therefore `from deep_ep import ElasticBuffer`, DeepEP intranode tests,
  `qwen3_moe_deepep`, and torchao MXFP8 grouped-GEMM import/numeric checks could
  not be run.

Root cause for the blocker is the L0 CUDA assembler/toolchain, not DeepEP or
torchao source selection: `/opt/cuda-synth/bin/nvcc` V13.2.86 emits PTX
`.version 9.4`, but the paired `ptxas` rejects it with
`Unsupported .version 9.4; current version is '9.2'`.

Evidence:

```text
.scratch/glm53-native/toolchain-probe/probe.log
.scratch/glm53-native/deepep-build-attempt.log
.scratch/glm53-native/torchao-build-attempt.log
```

## Native package layer

Added:

```text
experiments/glm53_native/packages.lock.json
experiments/glm53_native/build_native.sh
experiments/glm53_native/install_native.sh
experiments/glm53_native/run_kernel_checks.sh
experiments/glm53_native/patches/{deepep_v2,torchao,glm53_kernel_examples}/README.md
```

Normal build behavior:

- host wrapper re-enters:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep \
scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store
```

- source is fetched or copied inside rootfs;
- ordered patches under `patches/<pkg>/*.patch` are applied;
- wheel is built with `--no-build-isolation`;
- build key is SHA-256 over package metadata, source tree hash, patch hash,
  torch version, CUDA version, arch, rootfs id, and build args;
- normal builds publish wheel plus receipt to HDFS through Waystone;
- `--dev` installs editable source with ccache under rootfs env state and does
  not publish.

`install_native.sh` stages from HDFS through Waystone, verifies
`MANIFEST.jsonl` md5s locally, installs the wheel into `glm53_ep`, runs the
package import check, and writes:

```text
/project/venvs/b200-runtime/glm53_native_build_keys.json
```

## Published proof artifact

The in-repo native example wheel was built, published, staged, verified, and
installed:

```text
package: glm53_kernel_examples
build_key: 39fef457855c97035e9ad3065c6e7b9b07818052af01306f53fe1737abb19447
HDFS_URI: hdfs://harunava/user/tiger/waystone/torchtitan/native-wheels/glm53_kernel_examples/39fef457855c97035e9ad3065c6e7b9b07818052af01306f53fe1737abb19447
manifest_md5: 64e25220ff53e636d930ea637e051635
wheel: glm53_kernel_examples-0.1.0-cp312-cp312-linux_x86_64.whl
wheel_md5: 5db5b79e8d64aa44b36c8bb4b39efdbc
wheel_sha256: b171fd51e7c6b6ca549581468b2c2550ae7159fcc68c3518785fe8b5f8d0433f
rootfs_store_id: rootfs-e0c18cff68fc8ed06ca494cf00c807cca3d714819e00d5b824d89269079f4ec5
torch: 2.16.0.dev20261013+cu132
CUDA_HOME: /opt/cuda-synth
TORCH_CUDA_ARCH_LIST: 10.0
```

Receipt:

```text
.scratch/glm53-native/publish/glm53_kernel_examples/39fef457855c97035e9ad3065c6e7b9b07818052af01306f53fe1737abb19447/glm53_kernel_examples-39fef457855c97035e9ad3065c6e7b9b07818052af01306f53fe1737abb19447.json
```

Installed key receipt:

```text
.scratch/glm53-native/stage/glm53_kernel_examples/39fef457855c97035e9ad3065c6e7b9b07818052af01306f53fe1737abb19447/.staged.json
```

## DeepEP and torchao blocker

DeepEP v2 was attempted at the required pin:

```text
01dc3aaac82068020353dce2c302e38153c0bfaa
```

It failed while compiling CUDA objects such as `layout.cu`, `intranode.cu`,
`cuda_driver.cu`, `nvshmem.cu`, and `nccl.cu`:

```text
ptxas ... line 9; fatal : Unsupported .version 9.4; current version is '9.2'
```

torchao was attempted at source commit:

```text
9d0c4946b11af5e484f3ac36d61ec970c40c68cf
```

It failed in the MXFP8 CUDA extension with the same assembler error for
`compute_100` and `compute_120` PTX.

Because neither wheel exists, there are no DeepEP or torchao HDFS URIs/keys yet.
HybridEP was not attempted; the binding brief made DeepEP v2 first priority and
the L0 assembler blocks native CUDA wheels before HybridEP coexistence can be
meaningfully evaluated.

## Kernel convention

Added docs and examples under:

```text
torchtitan/experiments/glm53_kernels/
```

Current candidates listed in the README, but not implemented:

- fused mHC Sinkhorn/HcPost;
- fused indexer top-k plus IndexPool;
- sparse MLA gather attention, checking FlashMLA sparse first;
- KDA, checking `fla` first;
- MoE permute for the selected dispatcher.

Verification command:

```bash
CUDA_VISIBLE_DEVICES=6 experiments/glm53_native/run_kernel_checks.sh --allow-missing-native
```

Result:

```text
STATUS=0
.scratch/glm53-native/kernel-checks-final.log
```

Summary:

- Triton example: PASS for output, `dx`, and `dy`; torch.compile smoke passed
  with max diff `0.0`; microbench template ran.
- CuTe DSL convention: CUTLASS DSL `4.8.0`, `cute.kernel` and `cute.jit`
  present; numerics wrapper PASS but marked `PASS_WITH_LIMITATIONS` because no
  real CuTe kernel candidate was implemented.
- Native C++ example: PASS for output, `dx`, and `dy` after HDFS install.

## Attempt metadata helper

Added:

```text
torchtitan/experiments/glm53_flash_sft_ladder/native_keys.py
```

Smoke command:

```bash
python -m torchtitan.experiments.glm53_flash_sft_ladder.native_keys \
  --attempt-json .scratch/glm53-native/attempt-helper/attempt.json
```

Result: wrote the installed `glm53_kernel_examples` build key and HDFS source
URI into `attempt.json`.

## Test results

Passed:

```text
bash -n experiments/glm53_native/build_native.sh
bash -n experiments/glm53_native/install_native.sh
bash -n experiments/glm53_native/run_kernel_checks.sh
python -m py_compile torchtitan/experiments/glm53_kernels/triton_examples.py \
  torchtitan/experiments/glm53_kernels/cute_examples.py \
  torchtitan/experiments/glm53_kernels/checks.py \
  torchtitan/experiments/glm53_flash_sft_ladder/native_keys.py
experiments/glm53_native/build_native.sh glm53_kernel_examples
experiments/glm53_native/install_native.sh glm53_kernel_examples --key 39fef457855c97035e9ad3065c6e7b9b07818052af01306f53fe1737abb19447
CUDA_VISIBLE_DEVICES=6 experiments/glm53_native/run_kernel_checks.sh --allow-missing-native
python -m torchtitan.experiments.glm53_flash_sft_ladder.native_keys --attempt-json ...
```

Failed due L0 CUDA assembler mismatch:

```text
experiments/glm53_native/build_native.sh deepep_v2 --no-publish
experiments/glm53_native/build_native.sh torchao --no-publish
```

Not run because DeepEP wheel was not available:

```text
from deep_ep import ElasticBuffer
DeepEP intranode 2-GPU test
qwen3_moe_deepep EP=4 few-step comparison
torchao.prototype.moe_training MXFP8 grouped-GEMM import/numeric check
```

## Recommendation

Fix or replace the native L0 CUDA assembler/toolchain before continuing N01/T18:
the selected L0 must compile a one-line `sm_100` CUDA probe without
`Unsupported .version 9.4`. After that, rerun:

```bash
experiments/glm53_native/build_native.sh deepep_v2
experiments/glm53_native/install_native.sh deepep_v2 --key <key>
experiments/glm53_native/build_native.sh torchao
experiments/glm53_native/install_native.sh torchao --key <key>
```

Then run the deferred DeepEP and torchao GPU checks.

