# R01b report: make `glm53_ep` reproduce the working GLM runtime behavior

Date: 2026-10-14

Surface: repo-local GLM Flash SFT runtime infrastructure. This work did not
change core Trainer behavior, model math, or the live `glm53_train` venv. It
changed only the declared `glm53_ep` L1 runtime files plus scratch evidence and
docs.

## Outcome

`glm53_ep` is train-ready for the GLM Flash, GLM Flash EP, and GLM architecture
speedrun paths verified here.

The R01 blockers were caused by over-installing packages into `glm53_ep` rather
than by a torch/Triton wheel mismatch:

- Live `glm53_train` passes the GLM Flash test suite with torch/triton in its
  venv and many GLM support packages coming from its legacy system-site layer.
- `glm53_train` does not have `torchao` installed in the effective environment.
  `glm53_ep` did, so TorchTitan's optional NVFP4 import reached
  `torchao.prototype.moe_training.kernels.float8_rowwise`, where Triton 3.9
  rejects `key=["num_elements"]` because `num_elements` is not a kernel
  argument.
- `flash-attn` is not referenced by the verified GLM/speedrun paths. The
  package was also not usable in either environment: the legacy
  `glm53_train` CUDA extension fails to import, and the prior `glm53_ep` source
  install skipped the CUDA extension.

The fix is to make `glm53_ep` reproduce the working effective behavior:
exclude `torchao` and `flash-attn` from the L1 lock, keep the torch/triton/GLM
packages hash-pinned, and verify the KDA path the GLM code actually uses
(`attn_gym`) instead of treating direct `fla.ops.kda` as the green gate.

Direct `fla.ops.kda` import still fails under Triton 3.9 because
`fla.ops.cp.chunk_delta_h` has an autotune key containing `BT` while the kernel
argument list omits `BT`. `verify_runtime.py` now records that as a diagnostic,
not a GLM runtime blocker.

## Changes

- Removed `torchao==0.18.0` and `flash-attn==2.8.3.post1` from
  `experiments/glm53_flash_sft_ladder/runtime/requirements.direct.txt`.
- Regenerated `experiments/glm53_flash_sft_ladder/runtime/requirements.lock`
  with `uv pip compile --generate-hashes`; the lock now resolves 128 packages
  and contains no `torchao` or `flash-attn` entries.
- Removed the `flash-attn` build special case from
  `experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh`.
- Updated `verify_runtime.py` to:
  - add the repo root to `sys.path` when run by path;
  - require `attn-gym==0.0.16`, `fla-core==0.5.2`, torch/triton, and NVSHMEM;
  - fail if excluded packages `torchao` or `flash-attn` are installed;
  - gate on `attn_gym.linear.kda` and `torchtitan.models.common.attn_gym`;
  - record direct `fla.ops.kda` import as a diagnostic.
- Updated the runtime README and `env-facts.md`.

## Evidence

Baseline/fingerprint receipts:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_train_legacy_probe.json
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_train_native_probe.json
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep_native_probe.json
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep_native_post_sync_probe.json
```

Key baseline facts:

- `glm53_train` on the live legacy rootfs passes
  `pytest -q torchtitan/experiments/glm53_flash/tests`:
  `26 passed, 1 skipped, 14 warnings in 14.69s`.
- `glm53_train` has `torch==2.16.0.dev20261013+cu132` and
  `triton==3.9.0+gitfc8d31a5` in the venv, but no `torchao`; `fla-core`,
  `attn-gym`, `spmd_types`, and `torch_remat` resolve from the legacy
  `/usr/local/lib/python3.12/dist-packages` layer.
- Entering `glm53_train` through the managed native store uses a different
  `/project/venvs` state and only exposes torch/triton, so it is not the useful
  comparator for GLM package behavior. The useful comparator is live
  `glm53_train` on the legacy rootfs.

Runtime sync:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01b/sync_python_env.log
```

Post-sync verifier:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep-runtime-verification.json
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep-runtime-verification-summary.json
```

Summary:

```text
ok: true
torch: 2.16.0.dev20261013+cu132
triton: 3.9.0+gitfc8d31a5
attn-gym: 0.0.16
fla-core: 0.5.2
nvidia-nvshmem-cu13: 3.7.2
torchao: not installed
flash-attn: not installed
device: NVIDIA B200, sm_100
```

Test matrix:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep_glm53_flash_tests.log
26 passed, 1 skipped, 1 warning in 91.31s

.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep_glm53_flash_ep_tests.log
15 passed in 7.09s

.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep_glm53_arch_speedrun_tests.log
32 passed in 90.65s
```

Two-step debug train:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep_debug_train_2step.log
```

Command used:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep CUDA_VISIBLE_DEVICES=6 \
  scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store \
  -- bash -lc 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && GLM53_FLASH_DEBUG_STEPS=2 GLM53_FLASH_DUMP_FOLDER=.scratch/glm53-flash-sft-ladder/evidence/r01b/debug_train NGPU=1 LOG_RANK=0 MODULE=torchtitan.experiments.glm53_flash.config_registry CONFIG=glm53_flash_debugmodel ./run_train.sh'
```

Result:

```text
step 1 loss 6.25553 grad_norm 5.0312
step 2 loss 5.08131 grad_norm 4.0312
Training completed
```

This debug train uses `Glm53SyntheticDataLoader` and
`Glm53PretokenizedTokenizer`, so it has no real dataset/checkpoint input and no
HDFS input requirement.

Focused final checks:

```text
bash -n experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh
compile() check for verify_runtime.py and env_probe.py inside glm53_ep rootfs

.scratch/glm53-flash-sft-ladder/evidence/r01b/precommit_focused.log
flake8 passed
ufmt passed
pydoclint passed
codespell passed
```

Freeze comparison:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_train_uv_pip_freeze.txt
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_ep_uv_pip_freeze.txt
.scratch/glm53-flash-sft-ladder/evidence/r01b/glm53_train_to_glm53_ep_freeze.diff
```

The freezes are intentionally not identical. `glm53_ep` is the declared L1 env,
so its freeze contains GLM/test dependencies directly in the venv. Live
`glm53_train` depends on the legacy system-site layer for several of those
packages. The important reproduced behavior is the effective package set for
the GLM paths: no `torchao`, no usable/needed `flash-attn`, and matching
torch/triton/GLM dependency versions.

Flash-attn usage:

```text
.scratch/glm53-flash-sft-ladder/evidence/r01b/flash_attn_usage_search.txt
```

The search found no `flash_attn` use in the verified GLM Flash, GLM Flash EP,
GLM architecture speedrun, or ladder runner paths outside the runtime docs and
verifier exclusion. It should stay out of L1. If a future path needs it, make it
an L2/native package with a compatible build.

## Remaining caveat

Direct `fla.ops.kda` remains incompatible with Triton 3.9 import-time
autotune-key validation. This does not block the verified GLM paths, which use
`attn_gym` for KDA, but it should be fixed upstream or isolated in L2 before
any future task depends on importing `fla.ops.kda` directly.
