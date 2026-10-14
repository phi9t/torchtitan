# T16 brief: GLM EP phase 2, optional Dist-MoE evaluation

Repo: `/data02/home/philip.yang/workspace/torchtitan` (branch `ultron/mainline`). Read `env-facts.md`,
`docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`, `task-12-report.md`, and the latest EP GPU reports
before editing. Do NOT commit. This ticket is optional and should start only after plain all-to-all EP has passed the
T13 and T14 correctness gates.

Scope:

- Work in `torchtitan/experiments/glm53_flash_ep/` for GLM-specific Dist-MoE wrappers/configs. Do not edit existing files
  under `torchtitan/experiments/glm53_flash/` or `torchtitan/models/common/`; propose common Dist-MoE changes in the
  report.
- Install optional `dist_moe` only inside the `glm53_train` rootfs environment, without changing the torch version.
- First prove BF16 Dist-MoE correctness on the debug/truncated GLM EP config. Only then run full-depth scratch/VMM
  capacity sweeps. Only consider MXFP8 after BF16 passes.
- Requires NVIDIA SM100-or-newer CUDA devices.

Dependency and device preflight:

```bash
nvidia-smi

TORCHTITAN_ROOTFS_NETWORK=networked TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  constraint="$(mktemp)"
  python - <<PY >"${constraint}"
import torch
print("torch==" + torch.__version__)
PY
  uv pip install --system --break-system-packages -c "${constraint}" \
    "git+https://github.com/meta-pytorch/dist_moe.git@main"
'

TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  python - <<PY
import dist_moe
import torch
assert torch.cuda.is_available(), "Dist-MoE requires CUDA"
capability = torch.cuda.get_device_capability()
assert capability[0] >= 10, f"Dist-MoE requires SM100 or newer, got {capability}"
print({"dist_moe": getattr(dist_moe, "__version__", "unknown"), "capability": capability})
PY
'
```

BF16 debug/truncated correctness run:

```bash
RUN_ID="t16-glm53-dist-moe-bf16-$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RESULT_ROOT=".scratch/glm53-flash-sft-ladder/evidence/task-16/${RUN_ID}"
ROOTFS_RESULT_ROOT="/workspace/torchtitan/${HOST_RESULT_ROOT}"
HOST_CACHE_ROOT="${GLM_LADDER_CACHE:-/data05/${USER}/glm53-ladder-cache}"
mkdir -p "${HOST_RESULT_ROOT}/logs"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
TORCHTITAN_ROOTFS_ENV=glm53_train \
scripts/rootfs/enter_rootfs.sh -- bash -lc "
  unset CC CXX
  cd /workspace/torchtitan
  export GLM53_FLASH_COUNTDOWN_DATA_ROOT=/project/formal-cache/datasets/countdown-sft/v1
  export GLM53_FLASH_COUNTDOWN_DATA_FILE=/project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl
  export GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH=/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16
  export GLM53_FLASH_EP_DEGREE=4
  export GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS=8
  export GLM53_FLASH_EP_DIST_MOE_PRECISION=bf16
  export GLM53_FLASH_EP_DIST_MOE_SCRATCH_FACTOR=2.0
  export GLM53_FLASH_EP_DIST_MOE_VMM_FACTOR=none
  export GLM53_FLASH_SEED=42
  export GLM53_FLASH_DETERMINISTIC=1
  export GLM53_FLASH_FULL_SEQ_LEN=512
  export GLM53_FLASH_FULL_LOCAL_BATCH_SIZE=1
  export GLM53_FLASH_FULL_TOKENS_PER_STEP=32768
  export GLM53_FLASH_FULL_STEPS=3
  export GLM53_FLASH_EP_FULL_DUMP_FOLDER='${ROOTFS_RESULT_ROOT}/bf16_debug'
  torchrun --nproc_per_node=8 --rdzv_id '${RUN_ID}-bf16-debug' --rdzv_backend c10d \
    --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module torchtitan.experiments.glm53_flash_ep.config_registry \
    --config glm53_flash_ep_countdown_lora_truncated_dist_moe \
    --output-dir '${ROOTFS_RESULT_ROOT}/bf16_debug'
" 2>&1 | tee "${HOST_RESULT_ROOT}/logs/bf16_debug.log"
```

Full-depth BF16 scratch/VMM sweep, only after the correctness run passes:

```bash
for scratch in 1.0 1.5 2.0 4.0; do
  for vmm in none 4.0; do
    label="scratch${scratch}_vmm${vmm}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
    TORCHTITAN_ROOTFS_ENV=glm53_train \
    scripts/rootfs/enter_rootfs.sh -- bash -lc "
      unset CC CXX
      cd /workspace/torchtitan
      export GLM53_FLASH_COUNTDOWN_DATA_ROOT=/project/formal-cache/datasets/countdown-sft/v1
      export GLM53_FLASH_COUNTDOWN_DATA_FILE=/project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl
      export GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH=/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16
      export GLM53_FLASH_EP_DEGREE=8
      export GLM53_FLASH_EP_DIST_MOE_PRECISION=bf16
      export GLM53_FLASH_EP_DIST_MOE_SCRATCH_FACTOR=${scratch}
      export GLM53_FLASH_EP_DIST_MOE_VMM_FACTOR=${vmm}
      export GLM53_FLASH_SEED=42
      export GLM53_FLASH_DETERMINISTIC=0
      export GLM53_FLASH_FULL_SEQ_LEN=512
      export GLM53_FLASH_FULL_LOCAL_BATCH_SIZE=1
      export GLM53_FLASH_FULL_TOKENS_PER_STEP=32768
      export GLM53_FLASH_FULL_STEPS=10
      export GLM53_FLASH_EP_FULL_DUMP_FOLDER='${ROOTFS_RESULT_ROOT}/${label}'
      torchrun --nproc_per_node=8 --rdzv_id '${RUN_ID}-${label}' --rdzv_backend c10d \
        --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --local-ranks-filter 0 --role rank --tee 3 \
        -m torchtitan.train \
        --module torchtitan.experiments.glm53_flash_ep.config_registry \
        --config glm53_flash_ep_countdown_lora_dist_moe \
        --output-dir '${ROOTFS_RESULT_ROOT}/${label}'
    " 2>&1 | tee "${HOST_RESULT_ROOT}/logs/${label}.log"
  done
done
```

MXFP8 gate, only if BF16 passes and the report explains why MXFP8 is now worth spending:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
TORCHTITAN_ROOTFS_ENV=glm53_train \
scripts/rootfs/enter_rootfs.sh -- bash -lc "
  unset CC CXX
  cd /workspace/torchtitan
  export GLM53_FLASH_EP_DEGREE=8
  export GLM53_FLASH_EP_DIST_MOE_PRECISION=mxfp8
  export GLM53_FLASH_EP_DIST_MOE_SCRATCH_FACTOR=2.0
  export GLM53_FLASH_EP_DIST_MOE_VMM_FACTOR=4.0
  export GLM53_FLASH_FULL_STEPS=3
  export GLM53_FLASH_EP_FULL_DUMP_FOLDER='${ROOTFS_RESULT_ROOT}/mxfp8_gate'
  torchrun --nproc_per_node=8 --rdzv_id '${RUN_ID}-mxfp8-gate' --rdzv_backend c10d \
    --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module torchtitan.experiments.glm53_flash_ep.config_registry \
    --config glm53_flash_ep_countdown_lora_dist_moe \
    --output-dir '${ROOTFS_RESULT_ROOT}/mxfp8_gate'
"
```

Report: write `.scratch/glm53-flash-sft-ladder/sdd/task-16-report.md` with installed package revision, SM capability,
BF16 correctness result, scratch/VMM capacity table, memory evidence, and a recommendation on whether to continue toward
MXFP8.
