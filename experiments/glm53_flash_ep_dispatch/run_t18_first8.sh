#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
rootfs_repo_root="/workspace/torchtitan"

RUN_ID="${RUN_ID:-t18-glm53-deepep-$(date -u +%Y%m%dT%H%M%SZ)}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
RESULT_ROOT="${RESULT_ROOT:-${script_dir}/results/${RUN_ID}/${ATTEMPT_ID}}"
ROOTFS_RESULT_ROOT="${ROOTFS_RESULT_ROOT:-${rootfs_repo_root}/experiments/glm53_flash_ep_dispatch/results/${RUN_ID}/${ATTEMPT_ID}}"
LOG_DIR="${RESULT_ROOT}/logs"
HOST_CACHE_ROOT="${GLM_LADDER_CACHE:-/data05/${USER:-user}/glm53-ladder-cache}"
ROOTFS_CACHE_ROOT="/project/formal-cache"
HOST_DATA_ROOT="${HOST_CACHE_ROOT}/datasets/countdown-sft/v1"
HOST_MODEL_ROOT="${HOST_CACHE_ROOT}/model-weights/zai-org/GLM-5.3-Flash-BF16"
ROOTFS_DATA_ROOT="${ROOTFS_CACHE_ROOT}/datasets/countdown-sft/v1"
ROOTFS_MODEL_ROOT="${ROOTFS_CACHE_ROOT}/model-weights/zai-org/GLM-5.3-Flash-BF16"
TRAIN_DATA_FILE="${ROOTFS_DATA_ROOT}/train/formatting.jsonl"
ROOTFS_STORE="${TORCHTITAN_ROOTFS_STORE:-/data05/philip.yang/torchtitan-rootfs-store}"
ROOTFS_TRAIN_ENV="${ROOTFS_TRAIN_ENV:-glm53_ep}"
TRAIN_NGPU="${TRAIN_NGPU:-8}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-30720}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-${rootfs_repo_root}/.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_cache_prewarm}"
TRAIN_MODULE_REGISTRY="torchtitan.experiments.glm53_flash_ep.config_registry"
ALLTOALL_CONFIG="glm53_flash_ep_countdown_lora_truncated"
DEEPEP_CONFIG="glm53_flash_ep_countdown_lora_truncated_deepep"
STEPS="${STEPS:-20}"
SEQ_LEN="${SEQ_LEN:-512}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-32768}"
MAX_DOCS="${MAX_DOCS:-64}"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

usage() {
  cat >&2 <<'USAGE'
Usage: run_t18_first8.sh [preflight|deepep|alltoall|both|microbench|summary]

Runs the T18 first-8-layer GLM EP8 all-to-all vs DeepEP checks. Training and
microbenchmarks run inside the glm53_ep rootfs. Inputs must already be staged
from HDFS with .staged.json and MANIFEST.jsonl.
USAGE
}

rootfs_train() {
  TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
  TORCHTITAN_ROOTFS_ENV="${ROOTFS_TRAIN_ENV}" \
    "${repo_root}/scripts/rootfs/enter_rootfs.sh" \
      --rootfs-store "${ROOTFS_STORE}" -- bash -lc \
      "unset CC CXX CUDAHOSTCXX; cd ${rootfs_repo_root} && $*"
}

check_inputs() {
  "${repo_root}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh" --quick \
    "${HOST_DATA_ROOT}" "${HOST_MODEL_ROOT}" | tee "${LOG_DIR}/hdfs_strict_inputs.log"
}

gpu_min_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
    awk 'NR == 1 {min=$1} $1 < min {min=$1} END {print min+0}'
}

gpu_snapshot() {
  {
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,compute_mode --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
  }
}

wait_for_gpus() {
  local label="$1"
  local log="${LOG_DIR}/gpu_wait_${label}.log"
  while true; do
    local min_free
    min_free="$(gpu_min_free_mib || printf 0)"
    {
      printf '%s label=%s min_free_mib=%s threshold_mib=%s\n' "$(date -u +%FT%TZ)" "${label}" "${min_free}" "${GPU_MIN_FREE_MIB}"
      gpu_snapshot | sed 's/^/  /'
    } >> "${log}"
    if (( min_free >= GPU_MIN_FREE_MIB )); then
      break
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

train_one() {
  local name="$1"
  local config_name="$2"
  local rootfs_out="${ROOTFS_RESULT_ROOT}/train/${name}"
  local log="${LOG_DIR}/train_${name}.log"
  wait_for_gpus "train_${name}"
  rootfs_train "GLM53_FLASH_COUNTDOWN_DATA_ROOT='${ROOTFS_DATA_ROOT}' \
GLM53_FLASH_COUNTDOWN_DATA_FILE='${TRAIN_DATA_FILE}' \
GLM53_FLASH_COUNTDOWN_SPLIT=train \
GLM53_FLASH_COUNTDOWN_MAX_DOCS='${MAX_DOCS}' \
GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH='${ROOTFS_MODEL_ROOT}' \
GLM53_FLASH_EP_DEGREE=8 \
GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS=8 \
GLM53_FLASH_FULL_STEPS='${STEPS}' \
GLM53_FLASH_FULL_SEQ_LEN='${SEQ_LEN}' \
GLM53_FLASH_FULL_LOCAL_BATCH_SIZE='${LOCAL_BATCH_SIZE}' \
GLM53_FLASH_FULL_TOKENS_PER_STEP='${TOKENS_PER_STEP}' \
GLM53_FLASH_FULL_CHECKPOINT_INTERVAL=99 \
GLM53_FLASH_FULL_KEEP_LATEST_K=2 \
GLM53_FLASH_LOSS_CHUNKS=8 \
GLM53_FLASH_SEED=42 \
GLM53_FLASH_DETERMINISTIC=1 \
GLM53_FLASH_EP_VERIFY_FROZEN_BASE_LOAD=1 \
TRITON_CACHE_DIR='${TRITON_CACHE_ROOT}' \
CUDA_VISIBLE_DEVICES='${GPU_DEVICES}' \
EP_DISABLE_GIN='${EP_DISABLE_GIN:-1}' \
EP_REUSE_NCCL_COMM='${EP_REUSE_NCCL_COMM:-0}' \
NVSHMEM_REMOTE_TRANSPORT='${NVSHMEM_REMOTE_TRANSPORT:-none}' \
NVSHMEM_DISABLE_MNNVL='${NVSHMEM_DISABLE_MNNVL:-1}' \
torchrun --nproc_per_node='${TRAIN_NGPU}' \
  --rdzv_id '${RUN_ID}-${name}' \
  --rdzv_backend c10d --rdzv_endpoint='127.0.0.1:0' \
  --local_addr='127.0.0.1' --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train \
  --module '${TRAIN_MODULE_REGISTRY}' \
  --config '${config_name}' \
  --output-dir '${rootfs_out}'" 2>&1 | tee "${log}"
}

microbench() {
  wait_for_gpus "microbench"
  rootfs_train "CUDA_VISIBLE_DEVICES='${GPU_DEVICES}' \
EP_DISABLE_GIN='${EP_DISABLE_GIN:-1}' \
EP_REUSE_NCCL_COMM='${EP_REUSE_NCCL_COMM:-0}' \
NVSHMEM_REMOTE_TRANSPORT='${NVSHMEM_REMOTE_TRANSPORT:-none}' \
NVSHMEM_DISABLE_MNNVL='${NVSHMEM_DISABLE_MNNVL:-1}' \
torchrun --nproc_per_node='${TRAIN_NGPU}' \
  --rdzv_id '${RUN_ID}-microbench' \
  --rdzv_backend c10d --rdzv_endpoint='127.0.0.1:0' \
  --local_addr='127.0.0.1' --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.experiments.glm53_flash_ep.dispatch_microbench \
  --output '${ROOTFS_RESULT_ROOT}/microbench/dispatch_microbench.json' \
  --seq-len '${SEQ_LEN}' --local-batch-size '${LOCAL_BATCH_SIZE}' \
  --num-experts 288 --top-k 8 --hidden-dim 4096 --ep-size 8 \
  --warmup 5 --iters 20" 2>&1 | tee "${LOG_DIR}/microbench.log"
}

summary() {
  rootfs_train "python -m torchtitan.experiments.glm53_flash_ep.dispatch_summary \
    --result-root '${ROOTFS_RESULT_ROOT}' \
    --output-json '${ROOTFS_RESULT_ROOT}/summary/t18_summary.json' \
    --output-md '${ROOTFS_RESULT_ROOT}/summary/t18_summary.md'" \
    2>&1 | tee "${LOG_DIR}/summary.log"
}

cmd="${1:-both}"
case "${cmd}" in
  preflight)
    check_inputs
    rootfs_train "python - <<'PY'
from deep_ep import ElasticBuffer
from torchtitan.experiments.glm53_flash_ep.config_registry import glm53_flash_ep_countdown_lora_truncated_deepep
cfg = glm53_flash_ep_countdown_lora_truncated_deepep()
print('ElasticBuffer', ElasticBuffer)
print('config', cfg.model.token_dispatcher)
PY"
    ;;
  deepep)
    check_inputs
    train_one deepep_first8 "${DEEPEP_CONFIG}"
    ;;
  alltoall)
    check_inputs
    train_one alltoall_first8 "${ALLTOALL_CONFIG}"
    ;;
  both)
    check_inputs
    train_one alltoall_first8 "${ALLTOALL_CONFIG}"
    train_one deepep_first8 "${DEEPEP_CONFIG}"
    summary
    ;;
  microbench)
    microbench
    ;;
  summary)
    summary
    ;;
  *)
    usage
    exit 2
    ;;
esac
