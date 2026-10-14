# T15 brief: GLM EP phase 2, full-depth 4-GPU feasibility with memory direction

Repo: `/data02/home/philip.yang/workspace/torchtitan` (branch `ultron/mainline`). Read `env-facts.md`,
`docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`, `task-12-report.md`, and the T09 memory-direction
reports before editing. Do NOT commit. This ticket is GPU-required and must use exactly 4 visible GPUs.

Default paired direction for this brief is T09c lean BF16 overhead reduction. If `sdd/progress.md` records a different
accepted 4-GPU direction before this ticket starts, keep the EP acceptance criteria below and update the package/config
names in the report with the exact reason.

Scope:

- Add the 4-GPU EP + selected-memory-direction wrapper under `torchtitan/experiments/glm53_flash_ep/`.
- Do not edit existing files under `torchtitan/experiments/glm53_flash/` or `torchtitan/models/common/`.
- Use config name `glm53_flash_ep_countdown_lora_4gpu_lean` for the default T09c pairing.
- Run EP=4 on 4 GPUs with seq length 512, 32,768 tokens per optimizer step, seed 42, LoRA rank 16, BF16, and 12
  optimizer steps.
- Record peak GPU memory by rank, host RSS, rank-local loader bytes/time, steps 3-12 step time, loss, grad norm,
  all-to-all balance, and any allocation or checkpoint-load warnings.
- Claim one of: smoke, deterministic comparison, representative training, or performance evidence. Do not upgrade the
  claim label without matched baseline conditions.

Preflight:

```bash
nvidia-smi

TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  pytest -q torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py -x
  python -m torchtitan.experiments.glm53_flash_ep.validate_ep_matrix --world-size 4 --ep 4
  GLM53_FLASH_EP_DEGREE=4 NGPU=4 COMM_BACKEND=fake \
    MODULE=torchtitan.experiments.glm53_flash_ep.config_registry \
    CONFIG=glm53_flash_ep_countdown_lora_4gpu_lean \
    ./run_train.sh --print-config >/tmp/t15_glm53_ep_4gpu_lean_print_config.txt
'
```

GPU feasibility run:

```bash
RUN_ID="t15-glm53-ep-4gpu-lean-$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RESULT_ROOT=".scratch/glm53-flash-sft-ladder/evidence/task-15/${RUN_ID}"
ROOTFS_RESULT_ROOT="/workspace/torchtitan/${HOST_RESULT_ROOT}"
HOST_CACHE_ROOT="${GLM_LADDER_CACHE:-/data05/${USER}/glm53-ladder-cache}"
mkdir -p "${HOST_RESULT_ROOT}/logs"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
TORCHTITAN_ROOTFS_ENV=glm53_train \
scripts/rootfs/enter_rootfs.sh -- bash -lc "
  unset CC CXX
  cd /workspace/torchtitan
  export GLM53_FLASH_COUNTDOWN_DATA_ROOT=/project/formal-cache/datasets/countdown-sft/v1
  export GLM53_FLASH_COUNTDOWN_DATA_FILE=/project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl
  export GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH=/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16
  export GLM53_FLASH_EP_DEGREE=4
  export GLM53_FLASH_SEED=42
  export GLM53_FLASH_DETERMINISTIC=0
  export GLM53_FLASH_FULL_SEQ_LEN=512
  export GLM53_FLASH_FULL_LOCAL_BATCH_SIZE=1
  export GLM53_FLASH_FULL_TOKENS_PER_STEP=32768
  export GLM53_FLASH_FULL_STEPS=12
  export GLM53_FLASH_FULL_CHECKPOINT_INTERVAL=99
  export GLM53_FLASH_FULL_KEEP_LATEST_K=2
  export GLM53_LEAN_SEQ_LEN=512
  export GLM53_LEAN_LOCAL_BATCH_SIZE=1
  export GLM53_LEAN_TOKENS_PER_STEP=32768
  export GLM53_LEAN_STEPS=12
  export GLM53_LEAN_RECORD_CUDA_MEMORY=1
  export GLM53_LEAN_MEMORY_SNAPSHOT_STEPS=0,1,12
  export GLM53_FLASH_EP_FULL_DUMP_FOLDER='${ROOTFS_RESULT_ROOT}/ep4_lean'
  torchrun --nproc_per_node=4 --rdzv_id '${RUN_ID}-ep4-lean' --rdzv_backend c10d \
    --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module torchtitan.experiments.glm53_flash_ep.config_registry \
    --config glm53_flash_ep_countdown_lora_4gpu_lean \
    --output-dir '${ROOTFS_RESULT_ROOT}/ep4_lean'
" 2>&1 | tee "${HOST_RESULT_ROOT}/logs/ep4_lean.log"
```

Summary artifact:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc "
  unset CC CXX
  cd /workspace/torchtitan
  python -m torchtitan.experiments.glm53_flash_ep.feasibility_report \
    --run-root '${ROOTFS_RESULT_ROOT}/ep4_lean' \
    --output-json '${ROOTFS_RESULT_ROOT}/feasibility.json' \
    --output-md '${ROOTFS_RESULT_ROOT}/feasibility.md'
"
```

Report: write `.scratch/glm53-flash-sft-ladder/sdd/task-15-report.md` with the selected T09 direction, whether full-depth
4-GPU EP fits, all memory evidence paths, and the claim label. A negative result with measured lower bounds is valid.
