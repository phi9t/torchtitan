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
`requirements.lock`, records `/project/venvs/b200-runtime/python_env_report.json`,
and installs `flash-attn==2.8.3.post1` with
`FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`. The skipped CUDA extension is deliberate:
with this torch nightly, the upstream flash-attn source build hardcodes
`-std=c++17` while torch headers require C++20 symbols, and the legacy
`glm53_train` flash-attn CUDA extension is not importable either.

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

Known R01 blockers:

- `verify_runtime.py` currently exits 21 because `fla.ops.kda` import trips a
  Triton 3.9 autotune-key validation error in `fla.ops.cp.chunk_delta_h` before
  KDA itself imports.
- `pytest -q torchtitan/experiments/glm53_flash/tests` and the two-step
  `glm53_flash_debugmodel` launch fail during import on a torchao
  `torchao.prototype.moe_training.kernels.float8_rowwise` Triton 3.9 autotune
  key mismatch.
