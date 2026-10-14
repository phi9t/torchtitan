# T14 brief: GLM EP phase 2, full-depth 8-GPU EP ladder

Repo: `/data02/home/philip.yang/workspace/torchtitan` (branch `ultron/mainline`). Read `env-facts.md`,
`docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md`, `task-12-report.md`, and `task-13-report.md` if it
exists. Do NOT commit. This ticket is GPU-required and should run only when all 8 B200s are available enough for a
full-depth GLM probe.

Scope:

- Work in `torchtitan/experiments/glm53_flash_ep/` for EP-owned code. Do not edit existing files under
  `torchtitan/experiments/glm53_flash/` or `torchtitan/models/common/`.
- Add or reuse a full-depth Countdown LoRA config named `glm53_flash_ep_countdown_lora`.
- Sweep `GLM53_FLASH_EP_DEGREE` in `2 4 8` on 8 GPUs.
- Keep seq length 512, 32,768 tokens per optimizer step, seed 42, LoRA rank 16, BF16, and no CPU offload.
- Run at least 12 optimizer steps per EP setting so steps 3-12 are available as warmed steady-state samples.
- Record peak GPU memory by rank, model static memory, largest unshard size if available, rank-local loader bytes/time,
  all-to-all split histograms, step time, loss, and grad norm.
- Compare with T08b one-step smoke and any completed T08d seed-42 full-depth run. Do not call this performance evidence
  unless the GPU baseline is uncontended and the comparison is matched.

Preflight:

```bash
nvidia-smi

TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc '
  unset CC CXX
  cd /workspace/torchtitan
  pytest -q torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py -x
  python -m torchtitan.experiments.glm53_flash_ep.validate_ep_matrix --world-size 8 --ep 2 --ep 4 --ep 8
  for ep in 2 4 8; do
    GLM53_FLASH_EP_DEGREE=$ep NGPU=8 COMM_BACKEND=fake \
      MODULE=torchtitan.experiments.glm53_flash_ep.config_registry \
      CONFIG=glm53_flash_ep_countdown_lora \
      ./run_train.sh --print-config >/tmp/t14_glm53_ep_full_ep${ep}_print_config.txt
  done
'
```

GPU ladder:

```bash
RUN_ID="t14-glm53-ep-full8-$(date -u +%Y%m%dT%H%M%SZ)"
HOST_RESULT_ROOT=".scratch/glm53-flash-sft-ladder/evidence/task-14/${RUN_ID}"
ROOTFS_RESULT_ROOT="/workspace/torchtitan/${HOST_RESULT_ROOT}"
HOST_CACHE_ROOT="${GLM_LADDER_CACHE:-/data05/${USER}/glm53-ladder-cache}"
mkdir -p "${HOST_RESULT_ROOT}/logs"

for ep in 2 4 8; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
  TORCHTITAN_ROOTFS_ENV=glm53_train \
  scripts/rootfs/enter_rootfs.sh -- bash -lc "
    unset CC CXX
    cd /workspace/torchtitan
    export GLM53_FLASH_COUNTDOWN_DATA_ROOT=/project/formal-cache/datasets/countdown-sft/v1
    export GLM53_FLASH_COUNTDOWN_DATA_FILE=/project/formal-cache/datasets/countdown-sft/v1/train/formatting.jsonl
    export GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH=/project/formal-cache/model-weights/zai-org/GLM-5.3-Flash-BF16
    export GLM53_FLASH_EP_DEGREE=${ep}
    export GLM53_FLASH_SEED=42
    export GLM53_FLASH_DETERMINISTIC=0
    export GLM53_FLASH_FULL_SEQ_LEN=512
    export GLM53_FLASH_FULL_LOCAL_BATCH_SIZE=1
    export GLM53_FLASH_FULL_TOKENS_PER_STEP=32768
    export GLM53_FLASH_FULL_STEPS=12
    export GLM53_FLASH_FULL_CHECKPOINT_INTERVAL=99
    export GLM53_FLASH_FULL_KEEP_LATEST_K=2
    export GLM53_FLASH_EP_RECORD_ROUTER_STATS=1
    export GLM53_FLASH_EP_FULL_DUMP_FOLDER='${ROOTFS_RESULT_ROOT}/ep${ep}'
    torchrun --nproc_per_node=8 --rdzv_id '${RUN_ID}-ep${ep}' --rdzv_backend c10d \
      --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 --local-ranks-filter 0 --role rank --tee 3 \
      -m torchtitan.train \
      --module torchtitan.experiments.glm53_flash_ep.config_registry \
      --config glm53_flash_ep_countdown_lora \
      --output-dir '${ROOTFS_RESULT_ROOT}/ep${ep}'
  " 2>&1 | tee "${HOST_RESULT_ROOT}/logs/ep${ep}.log"
done
```

Summary artifact:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc "
  unset CC CXX
  cd /workspace/torchtitan
  python -m torchtitan.experiments.glm53_flash_ep.ladder_report \
    --run-root '${ROOTFS_RESULT_ROOT}' \
    --ep 2 --ep 4 --ep 8 \
    --compare-t08b .scratch/glm53-flash-sft-ladder/sdd/task-08b-report.md \
    --output-json '${ROOTFS_RESULT_ROOT}/summary.json' \
    --output-md '${ROOTFS_RESULT_ROOT}/summary.md'
"
```

Report: write `.scratch/glm53-flash-sft-ladder/sdd/task-14-report.md` with one row per EP degree, raw log paths,
environment class, warm-step timing, memory, loader stats, all-to-all balance, and the exact claim label.
