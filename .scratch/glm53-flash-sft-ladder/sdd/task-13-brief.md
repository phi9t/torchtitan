# T13 brief: GLM EP phase 2, truncated-depth full-width numerical parity

Repo: `/data02/home/philip.yang/workspace/torchtitan` (branch `ultron/mainline`). Read `env-facts.md`,
`docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`, and `task-12-report.md` first. Do NOT commit.
This ticket may use GPUs, but only for the exact truncated-depth parity runs below. Do not run full-depth GLM here.

Scope:

- Work in `torchtitan/experiments/glm53_flash_ep/` unless a focused test or report needs to read an existing package.
- Do not edit existing files under `torchtitan/experiments/glm53_flash/` or `torchtitan/models/common/`; propose shared
  changes in the report instead.
- Add a full-width, first-8-layer GLM-5.3-Flash EP Countdown LoRA config named
  `glm53_flash_ep_countdown_lora_truncated`.
- Keep layers 0-7 exactly: dense layers 0-2, MoE layers 3-7. Keep seq length 512, 32,768 tokens per optimizer step,
  seed 42, deterministic mode on, LoRA rank 16, and BF16 training.
- Compare against the FSDP-only first-8-layer baseline from
  `torchtitan.experiments.glm53_flash_4gpu_lean.config_registry:glm53_flash_countdown_lora_truncated_plain`.
- Run at least 3 optimizer steps. Report full-precision loss, grad norm, and max parameter-gradient deltas. Bitwise is
  required only if the implementation promises identical grouped expert math; otherwise state the exact tolerance and the
  grouped-kernel reason.

Preflight and focused checks:

```bash
nvidia-smi

TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  pytest -q torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py -x
  python -m torchtitan.experiments.glm53_flash_ep.validate_ep_matrix --world-size 8 --ep 1 --ep 2 --ep 4 --ep 8
  GLM53_FLASH_EP_DEGREE=4 NGPU=8 COMM_BACKEND=fake \
    MODULE=torchtitan.experiments.glm53_flash_ep.config_registry \
    CONFIG=glm53_flash_ep_countdown_lora_truncated \
    ./run_train.sh --print-config >/tmp/t13_glm53_ep_truncated_print_config.txt
'
```

GPU parity run:

```bash
RUN_ID="t13-glm53-ep-truncated-$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RESULT_ROOT=".scratch/glm53-flash-sft-ladder/evidence/task-13/${RUN_ID}"
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
  export GLM53_FLASH_SEED=42
  export GLM53_FLASH_DETERMINISTIC=1
  export GLM53_FLASH_FULL_SEQ_LEN=512
  export GLM53_FLASH_FULL_LOCAL_BATCH_SIZE=1
  export GLM53_FLASH_FULL_TOKENS_PER_STEP=32768
  export GLM53_FLASH_FULL_CHECKPOINT_INTERVAL=99
  export GLM53_FLASH_FULL_KEEP_LATEST_K=2
  export GLM53_LEAN_NUM_LAYERS=8
  export GLM53_LEAN_STEPS=3
  export GLM53_LEAN_RECORD_CUDA_MEMORY=1
  torchrun --nproc_per_node=8 --rdzv_id '${RUN_ID}-plain' --rdzv_backend c10d \
    --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module torchtitan.experiments.glm53_flash_4gpu_lean.config_registry \
    --config glm53_flash_countdown_lora_truncated_plain \
    --output-dir '${ROOTFS_RESULT_ROOT}/plain'
" 2>&1 | tee "${HOST_RESULT_ROOT}/logs/plain.log"

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
  export GLM53_FLASH_SEED=42
  export GLM53_FLASH_DETERMINISTIC=1
  export GLM53_FLASH_FULL_SEQ_LEN=512
  export GLM53_FLASH_FULL_LOCAL_BATCH_SIZE=1
  export GLM53_FLASH_FULL_TOKENS_PER_STEP=32768
  export GLM53_FLASH_FULL_STEPS=3
  export GLM53_FLASH_FULL_CHECKPOINT_INTERVAL=99
  export GLM53_FLASH_FULL_KEEP_LATEST_K=2
  export GLM53_FLASH_EP_FULL_DUMP_FOLDER='${ROOTFS_RESULT_ROOT}/ep4'
  torchrun --nproc_per_node=8 --rdzv_id '${RUN_ID}-ep4' --rdzv_backend c10d \
    --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module torchtitan.experiments.glm53_flash_ep.config_registry \
    --config glm53_flash_ep_countdown_lora_truncated \
    --output-dir '${ROOTFS_RESULT_ROOT}/ep4'
" 2>&1 | tee "${HOST_RESULT_ROOT}/logs/ep4.log"
```

Comparison artifact:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc "
  unset CC CXX
  cd /workspace/torchtitan
  python -m torchtitan.experiments.glm53_flash_ep.parity_report \
    --plain '${ROOTFS_RESULT_ROOT}/plain' \
    --ep '${ROOTFS_RESULT_ROOT}/ep4' \
    --output-json '${ROOTFS_RESULT_ROOT}/parity.json' \
    --output-md '${ROOTFS_RESULT_ROOT}/parity.md'
"
```

Report: write `.scratch/glm53-flash-sft-ladder/sdd/task-13-report.md` with the command logs, environment class from the
GPU snapshot, loss/grad deltas, peak memory per rank, and a clear claim label: deterministic comparison if the trace is
matched, otherwise smoke plus bounded numerical delta.

## Addendum 2026-10-14 18:50Z (binding; overrides conflicting text above)
- User direction: replace the per-expert-loop + FSDP-over-experts MoE with real expert parallelism (token all-to-all,
  DeepEP-class dispatch). EP is now on the critical path for rung 4, not optional.
- Local expert compute must be grouped GEMM (T17 adds `GLM53_FLASH_EXPERTS_IMPL=grouped`; the common
  GroupedLinear path in `glm53_flash_ep` is equivalent) -- never the per-expert Python loop.
- A slow 8-GPU T08d GLM run (~136-155 GiB/GPU) is live as a fallback. Co-run only: <= 30 GiB per process via
  `torch.cuda.set_per_process_memory_fraction`, truncated-depth (e.g. first 8 layers) configs, never kill it or foreign
  processes. `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause you while T08d waits between legs.
- Inputs strictly from waystone HDFS staged roots verified by `experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`.
- Report step time (ms) and tokens/s per config alongside numerics; that is the point of this work.
