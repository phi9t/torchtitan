# GLM-5.3 Flash SFT Runtime

`glm53_ep` is the declared L1 Python environment for GLM-5.3 Flash SFT ladder
training on the TorchTitan Insula rootfs. It is separate from the live
`glm53_train` environment and must be synced through the rootfs wrapper.

Build or select the opt-in native L0 rootfs when native extension work needs the
extra toolchain pieces:

```bash
scripts/rootfs/build_rootfs.sh \
  --rebuild \
  --variant native \
  --store /data05/philip.yang/torchtitan-rootfs-store
```

The current R01 native store selection is
`rootfs-e0c18cff68fc8ed06ca494cf00c807cca3d714819e00d5b824d89269079f4ec5`.

Sync `glm53_ep` against the selected store:

```bash
experiments/glm53_flash_sft_ladder/runtime/sync_python_env.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store
```

The sync script recreates the dedicated `glm53_ep` venv from
`requirements.lock` and records
`/project/venvs/b200-runtime/python_env_report.json`. `torchao` and
`flash-attn` are intentionally excluded from this L1 environment: `glm53_train`
does not need either package for the GLM Flash, GLM Flash EP, or GLM
architecture speedrun paths, and installing them in `glm53_ep` triggers import
failures with Triton 3.9 (`torchao.prototype.moe_training` autotune-key
validation and a non-importable flash-attn CUDA extension).

Verify the environment:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store \
  -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && python experiments/glm53_flash_sft_ladder/runtime/verify_runtime.py'
```

Launch GLM debug training in the environment:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store \
  -- bash -lc 'unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && GLM53_FLASH_DEBUG_STEPS=2 NGPU=1 MODULE=torchtitan.experiments.glm53_flash.config_registry CONFIG=glm53_flash_debugmodel ./run_train.sh'
```

R01b status:

- `verify_runtime.py` is green for the GLM runtime gate: torch/CUDA/sm_100,
  Triton, `attn_gym.linear.kda`, TorchTitan's `attn_gym` bridge, NVSHMEM, and
  CUDA headers/libs.
- Direct `fla.ops.kda` import still trips a Triton 3.9 autotune-key validation
  error in `fla.ops.cp.chunk_delta_h`; this is recorded as a diagnostic because
  the GLM paths verified here use `attn_gym` for KDA.
- `flash-attn` is not used by the verified GLM/speedrun paths. If a future GLM
  path requires it, promote it to an L2/native package with a compatible build
  instead of re-adding the broken source install to L1.
