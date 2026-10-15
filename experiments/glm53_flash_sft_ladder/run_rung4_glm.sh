#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
rootfs_repo_root="/workspace/torchtitan"
rootfs_script_dir="${rootfs_repo_root}/experiments/glm53_flash_sft_ladder"

RUN_ID="${RUN_ID:-t08f-rung4-glm53-ep8-$(date -u +%Y%m%dT%H%M%SZ)}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt-01}"
RESULT_ROOT="${RESULT_ROOT:-${script_dir}/results/${RUN_ID}/${ATTEMPT_ID}}"
ROOTFS_RESULT_ROOT="${ROOTFS_RESULT_ROOT:-${rootfs_script_dir}/results/${RUN_ID}/${ATTEMPT_ID}}"
LOG_DIR="${RESULT_ROOT}/logs"
TRAIN_NGPU="${TRAIN_NGPU:-8}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-92160}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
HOST_CACHE_ROOT="${GLM_LADDER_CACHE:-/data05/${USER:-user}/glm53-ladder-cache}"
ROOTFS_STORE="${TORCHTITAN_ROOTFS_STORE:-/data05/philip.yang/torchtitan-rootfs-store}"
ROOTFS_STORE_SELECTION="${ROOTFS_STORE}/selected.json"
ROOTFS_CACHE_ROOT="/project/formal-cache"
HOST_DATA_ROOT="${HOST_CACHE_ROOT}/datasets/countdown-sft/v1"
HOST_MODEL_ROOT="${HOST_CACHE_ROOT}/model-weights/zai-org/GLM-5.3-Flash-BF16"
ROOTFS_DATA_ROOT="${ROOTFS_CACHE_ROOT}/datasets/countdown-sft/v1"
ROOTFS_MODEL_ROOT="${ROOTFS_CACHE_ROOT}/model-weights/zai-org/GLM-5.3-Flash-BF16"
TRAIN_DATA_FILE="${ROOTFS_DATA_ROOT}/train/formatting.jsonl"
DEV_PROBLEMS="${ROOTFS_DATA_ROOT}/dev/problems.jsonl"
TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-${rootfs_repo_root}/.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_cache_prewarm}"
TRAIN_MODULE_REGISTRY="${TRAIN_MODULE_REGISTRY:-torchtitan.experiments.glm53_flash_ep.config_registry}"
TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-glm53_flash_ep_countdown_lora_truncated}"
ROOTFS_TRAIN_ENV="${ROOTFS_TRAIN_ENV:-glm53_ep}"
GLM53_FLASH_EXPERTS_IMPL="${GLM53_FLASH_EXPERTS_IMPL:-grouped}"
GLM53_FLASH_EP_DEGREE="${GLM53_FLASH_EP_DEGREE:-8}"
GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS="${GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS:-45}"
GLM53_FLASH_FULL_SEQ_LEN="${GLM53_FLASH_FULL_SEQ_LEN:-512}"
GLM53_FLASH_FULL_LOCAL_BATCH_SIZE="${GLM53_FLASH_FULL_LOCAL_BATCH_SIZE:-1}"
GLM53_FLASH_FULL_TOKENS_PER_STEP="${GLM53_FLASH_FULL_TOKENS_PER_STEP:-32768}"
GLM53_FLASH_FULL_CHECKPOINT_INTERVAL="${GLM53_FLASH_FULL_CHECKPOINT_INTERVAL:-31}"
GLM53_FLASH_FULL_KEEP_LATEST_K="${GLM53_FLASH_FULL_KEEP_LATEST_K:-10}"
GLM53_FLASH_LOSS_CHUNKS="${GLM53_FLASH_LOSS_CHUNKS:-8}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
EVAL_DECODE_BATCH_SIZE="${EVAL_DECODE_BATCH_SIZE:-8}"
TRAIN_COUNTDOWN_SPLIT="${TRAIN_COUNTDOWN_SPLIT:-train}"
TRAIN_MAX_DOCS="${TRAIN_MAX_DOCS:-64}"
DEEPEP_NATIVE_BUILD_KEY="${DEEPEP_NATIVE_BUILD_KEY:-6b7e14e72362f6ad219ba107036eebb765ee4dd3c03bd43c8be5535f75f47e45}"
DEEPEP_NATIVE_HDFS_URI="${DEEPEP_NATIVE_HDFS_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/native-wheels/deepep_v2/6b7e14e72362f6ad219ba107036eebb765ee4dd3c03bd43c8be5535f75f47e45}"
DEEPEP_NATIVE_MANIFEST_MD5="${DEEPEP_NATIVE_MANIFEST_MD5:-616464c9be66a87090aea61222ed3660}"
TORCHAO_NATIVE_BUILD_KEY="${TORCHAO_NATIVE_BUILD_KEY:-09637433cfe6f6f49fb4a2c36274a1bb77f96373012e15f3683216d9a90386cb}"
TORCHAO_NATIVE_HDFS_URI="${TORCHAO_NATIVE_HDFS_URI:-hdfs://harunava/user/tiger/waystone/torchtitan/native-wheels/torchao/09637433cfe6f6f49fb4a2c36274a1bb77f96373012e15f3683216d9a90386cb}"
TORCHAO_NATIVE_MANIFEST_MD5="${TORCHAO_NATIVE_MANIFEST_MD5:-9614b784ac23b09e857d623a8496d84d}"
mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"

usage() {
  cat >&2 <<'USAGE'
Usage: run_rung4_glm.sh [all|preflight|train|resume|seed43|export|eval|publish|prune]

Runs the T08f GLM-5.3-Flash EP8 rung-4 gates. Every torchrun leg waits until
all 8 GPUs have at least 90 GiB free unless GPU_MIN_FREE_MIB is overridden,
and until no Exclusive_Process GPU holder is present. Default-mode foreign
processes are logged and make the leg shared_gpu, but do not block launch.
USAGE
}

rootfs_train() {
  TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
  TORCHTITAN_ROOTFS_ENV="${ROOTFS_TRAIN_ENV}" \
    "${repo_root}/scripts/rootfs/enter_rootfs.sh" \
      --rootfs-store "${ROOTFS_STORE}" -- bash -lc \
      "unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && $*"
}

rootfs_default() {
  TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${HOST_CACHE_ROOT}" \
    "${repo_root}/scripts/rootfs/enter_rootfs.sh" -- bash -lc \
      "unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && $*"
}

check_inputs() {
  [[ -f "${HOST_DATA_ROOT}/train/formatting.jsonl" ]] || {
    printf 'missing staged dataset: %s\n' "${HOST_DATA_ROOT}" >&2
    exit 1
  }
  [[ -f "${HOST_MODEL_ROOT}/.staged.json" ]] || {
    printf 'missing staged GLM checkpoint marker: %s/.staged.json\n' "${HOST_MODEL_ROOT}" >&2
    exit 1
  }
  "${script_dir}/lib/hdfs_strict.sh" --quick "${HOST_DATA_ROOT}" "${HOST_MODEL_ROOT}" \
    | tee -a "${LOG_DIR}/hdfs_strict_inputs.log"
  write_attempt_json
}

gpu_free_csv() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /, "", $1); gsub(/ /, "", $2); printf "%s:%s%s", $1, $2, NR == 8 ? "" : ","}'
}

gpu_min_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
    awk 'NR == 1 {min=$1} $1 < min {min=$1} END {print min+0}'
}

gpu_process_snapshot() {
  local output
  output="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${output}" ]]; then
    printf 'none\n'
    return
  fi
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    local pid user cmd
    pid="$(awk -F, '{gsub(/ /, "", $2); print $2}' <<<"${line}")"
    user="$(ps -o user= -p "${pid}" 2>/dev/null | awk '{print $1}' || true)"
    cmd="$(ps -o args= -p "${pid}" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g' || true)"
    printf 'user=%s pid=%s %s cmd=%s\n' "${user:-unknown}" "${pid}" "${line}" "${cmd:-unknown}"
  done <<<"${output}"
}

gpu_compute_process_count() {
  local output
  output="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  awk 'NF {count++} END {print count+0}' <<<"${output}"
}

gpu_foreign_process_count() {
  local output current_user pid user
  current_user="${USER:-$(id -un)}"
  output="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${output}" ]]; then
    printf '0\n'
    return
  fi
  local count=0
  while IFS= read -r pid; do
    pid="$(tr -d '[:space:]' <<<"${pid}")"
    [[ -n "${pid}" ]] || continue
    user="$(ps -o user= -p "${pid}" 2>/dev/null | awk '{print $1}' || true)"
    if [[ -n "${user}" && "${user}" != "${current_user}" ]]; then
      count=$((count + 1))
    fi
  done <<<"${output}"
  printf '%s\n' "${count}"
}

gpu_compute_modes_csv() {
  nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader |
    awk -F, '{gsub(/ /, "", $1); sub(/^ /, "", $2); printf "%s:%s%s", $1, $2, NR == 8 ? "" : ","}'
}

gpu_exclusive_process_mode_count() {
  nvidia-smi --query-gpu=compute_mode --format=csv,noheader |
    awk '/Exclusive_Process/ {count++} END {print count+0}'
}

wait_for_gpus() {
  local label="$1" log="$2"
  mkdir -p "$(dirname "${log}")"
  while true; do
    local timestamp free_csv min_free process_count foreign_count exclusive_count mode_csv decision
    timestamp="$(date -u +%FT%TZ)"
    process_count="$(gpu_compute_process_count)"
    foreign_count="$(gpu_foreign_process_count)"
    exclusive_count="$(gpu_exclusive_process_mode_count)"
    mode_csv="$(gpu_compute_modes_csv || true)"
    if ! free_csv="$(gpu_free_csv)"; then
      free_csv="query_error"
      min_free=0
      decision="wait_query_error"
    elif ! min_free="$(gpu_min_free_mib)"; then
      min_free=0
      decision="wait_query_error"
    elif (( min_free >= GPU_MIN_FREE_MIB )) && (( exclusive_count == 0 )); then
      decision="launch"
    elif (( min_free >= GPU_MIN_FREE_MIB )) && (( exclusive_count > 0 )); then
      decision="wait_exclusive_compute_mode"
    else
      decision="wait_low_free"
    fi
    {
      printf '%s label=%s decision=%s min_free_mib=%s threshold_mib=%s active_compute_processes=%s foreign_compute_processes=%s exclusive_mode_gpus=%s compute_modes=%s free_mib=%s\n' \
        "${timestamp}" "${label}" "${decision}" "${min_free}" "${GPU_MIN_FREE_MIB}" "${process_count}" "${foreign_count}" "${exclusive_count}" "${mode_csv:-query_error}" "${free_csv}"
      gpu_process_snapshot | sed 's/^/  proc /'
    } >> "${log}"
    if [[ "${decision}" == "launch" ]]; then
      break
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

monitor_gpu_processes() {
  local label="$1" log="$2"
  while true; do
    {
      printf '%s label=%s\n' "$(date -u +%FT%TZ)" "${label}"
      gpu_process_snapshot | sed 's/^/  proc /'
    } >> "${log}"
    sleep 60
  done
}

environment_class_from_gpu_logs() {
  local baseline_log="$1" gpu_log="$2"
  if [[ -f "${baseline_log}" ]] && grep -Ev '^(none)?$' "${baseline_log}" >/dev/null 2>&1; then
    printf 'shared_gpu\n'
    return
  fi
  if grep -E "proc user=" "${gpu_log}" | grep -Fv "${RUN_ID}" >/dev/null 2>&1; then
    printf 'shared_gpu\n'
  else
    printf 'exclusive\n'
  fi
}

max_rank_rss_gb() {
  local log="$1"
  if [[ ! -f "${log}" ]]; then
    printf '0\n'
    return
  fi
  awk -F'rss_gb=' '/rss_gb=/ {split($2, a, " "); if (a[1] > max) max=a[1]} END {if (max == "") print 0; else printf "%.2f\n", max}' "${log}"
}

run_with_monitors() {
  local label="$1" guard_pattern="$2" command="$3"
  local wait_log="${LOG_DIR}/gpu_wait_${label}.log"
  local baseline_log="${LOG_DIR}/gpu_baseline_${label}.log"
  local gpu_log="${LOG_DIR}/gpu_processes_${label}.log"
  local guard_log="${LOG_DIR}/host_ram_guard_${label}.log"
  local gpu_pid="" guard_pid="" status

  wait_for_gpus "${label}" "${wait_log}"
  gpu_process_snapshot > "${baseline_log}"
  monitor_gpu_processes "${label}" "${gpu_log}" &
  gpu_pid="$!"
  GLM53_HOST_RAM_GUARD_INTERVAL_SECONDS=30 \
  GLM53_HOST_RAM_GUARD_PATTERN="${guard_pattern}" \
    "${script_dir}/host_ram_guard.sh" "${GLM53_HOST_RAM_FLOOR_GB:-600}" "${guard_log}" "${guard_pattern}" &
  guard_pid="$!"

  set +e
  bash -o pipefail -lc "${command}"
  status="$?"
  set -e

  kill "${gpu_pid}" "${guard_pid}" 2>/dev/null || true
  wait "${gpu_pid}" "${guard_pid}" 2>/dev/null || true

  {
    printf 'name=%s\n' "${label}"
    printf 'exit_status=%s\n' "${status}"
    printf 'environment_class=%s\n' "$(environment_class_from_gpu_logs "${baseline_log}" "${gpu_log}")"
    printf 'max_rank_rss_gb=%s\n' "$(max_rank_rss_gb "${guard_log}")"
    printf 'gpu_wait_log=%s\n' "${wait_log}"
    printf 'gpu_baseline_log=%s\n' "${baseline_log}"
    printf 'gpu_process_log=%s\n' "${gpu_log}"
    printf 'host_ram_guard_log=%s\n' "${guard_log}"
  } > "${LOG_DIR}/leg_${label}.env"

  return "${status}"
}

verify_train_leg() {
  local name="$1" expected_step="$2" checkpoint_step="$3"
  local summary="${ROOTFS_RESULT_ROOT}/logs/verify_${name}.json"
  rootfs_train "python -m torchtitan.experiments.glm53_flash_sft_ladder.check_training_leg --run-dir '${ROOTFS_RESULT_ROOT}/train/${name}' --expected-step '${expected_step}' --checkpoint-step '${checkpoint_step}' --summary-json '${summary}'" \
    2>&1 | tee "${LOG_DIR}/verify_${name}.log"
}

print_config_one() {
  local name="$1" seed="$2" deterministic="$3"
  mkdir -p "${RESULT_ROOT}/configs"
  rootfs_train "export GLM53_FLASH_COUNTDOWN_DATA_ROOT='${ROOTFS_DATA_ROOT}' \
GLM53_FLASH_COUNTDOWN_DATA_FILE='${TRAIN_DATA_FILE}' \
GLM53_FLASH_COUNTDOWN_SPLIT='${TRAIN_COUNTDOWN_SPLIT}' \
GLM53_FLASH_COUNTDOWN_MAX_DOCS='${TRAIN_MAX_DOCS}' \
GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH='${ROOTFS_MODEL_ROOT}' \
GLM53_FLASH_EXPERTS_IMPL='${GLM53_FLASH_EXPERTS_IMPL}' \
GLM53_FLASH_EP_DEGREE='${GLM53_FLASH_EP_DEGREE}' \
GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS='${GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS}' \
GLM53_FLASH_FULL_SEQ_LEN='${GLM53_FLASH_FULL_SEQ_LEN}' \
GLM53_FLASH_FULL_LOCAL_BATCH_SIZE='${GLM53_FLASH_FULL_LOCAL_BATCH_SIZE}' \
GLM53_FLASH_FULL_TOKENS_PER_STEP='${GLM53_FLASH_FULL_TOKENS_PER_STEP}' \
GLM53_FLASH_FULL_CHECKPOINT_INTERVAL='${GLM53_FLASH_FULL_CHECKPOINT_INTERVAL}' \
GLM53_FLASH_FULL_KEEP_LATEST_K='${GLM53_FLASH_FULL_KEEP_LATEST_K}' \
GLM53_FLASH_LOSS_CHUNKS='${GLM53_FLASH_LOSS_CHUNKS}' \
GLM53_FLASH_EP_VERIFY_FROZEN_BASE_LOAD=1 \
GLM53_FLASH_SEED='${seed}' \
GLM53_FLASH_COUNTDOWN_SEED='${seed}' \
GLM53_FLASH_DETERMINISTIC='${deterministic}' \
GLM53_FLASH_FULL_STEPS=94; \
python -m torchtitan.train --module '${TRAIN_MODULE_REGISTRY}' --config '${TRAIN_CONFIG_NAME}' --output-dir '${ROOTFS_RESULT_ROOT}/train/${name}' --print-config" \
    > "${RESULT_ROOT}/configs/${name}_config.txt" \
    2> "${LOG_DIR}/print_config_${name}.log"
}

print_configs() {
  print_config_one seed42 42 1
  print_config_one seed43 43 1
}

train_one() {
  local name="$1" seed="$2" deterministic="$3" steps="$4" resume_step="${5:-}" stop_after="${6:-}"
  local out="${RESULT_ROOT}/train/${name}"
  local rootfs_out="${ROOTFS_RESULT_ROOT}/train/${name}"
  local train_module="torchtitan.train"
  local stop_args=""
  local resume_args=""
  local expected_step="${steps}"
  mkdir -p "${out}"
  if [[ -n "${stop_after}" ]]; then
    train_module="torchtitan.experiments.glm53_flash_sft_ladder.train_until"
    stop_args="--stop-after-step ${stop_after}"
    expected_step="${stop_after}"
  fi
  if [[ -n "${resume_step}" ]]; then
    resume_args="--resume-step ${resume_step}"
  fi

  local rootfs_cmd
  rootfs_cmd="export GLM53_FLASH_COUNTDOWN_DATA_ROOT='${ROOTFS_DATA_ROOT}' \
GLM53_FLASH_COUNTDOWN_DATA_FILE='${TRAIN_DATA_FILE}' \
GLM53_FLASH_COUNTDOWN_SPLIT='${TRAIN_COUNTDOWN_SPLIT}' \
GLM53_FLASH_COUNTDOWN_MAX_DOCS='${TRAIN_MAX_DOCS}' \
GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH='${ROOTFS_MODEL_ROOT}' \
GLM53_FLASH_EXPERTS_IMPL='${GLM53_FLASH_EXPERTS_IMPL}' \
GLM53_FLASH_EP_DEGREE='${GLM53_FLASH_EP_DEGREE}' \
GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS='${GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS}' \
GLM53_FLASH_FULL_SEQ_LEN='${GLM53_FLASH_FULL_SEQ_LEN}' \
GLM53_FLASH_FULL_LOCAL_BATCH_SIZE='${GLM53_FLASH_FULL_LOCAL_BATCH_SIZE}' \
GLM53_FLASH_FULL_TOKENS_PER_STEP='${GLM53_FLASH_FULL_TOKENS_PER_STEP}' \
GLM53_FLASH_FULL_CHECKPOINT_INTERVAL='${GLM53_FLASH_FULL_CHECKPOINT_INTERVAL}' \
GLM53_FLASH_FULL_KEEP_LATEST_K='${GLM53_FLASH_FULL_KEEP_LATEST_K}' \
GLM53_FLASH_LOSS_CHUNKS='${GLM53_FLASH_LOSS_CHUNKS}' \
GLM53_FLASH_EP_VERIFY_FROZEN_BASE_LOAD=1 \
GLM53_FLASH_SEED='${seed}' \
GLM53_FLASH_COUNTDOWN_SEED='${seed}' \
GLM53_FLASH_DETERMINISTIC='${deterministic}' \
GLM53_FLASH_FULL_STEPS='${steps}' \
TRITON_CACHE_DIR='${TRITON_CACHE_ROOT}' \
CUDA_VISIBLE_DEVICES='${GPU_DEVICES}' \
TORCHTITAN_ROOTFS_ENV='${ROOTFS_TRAIN_ENV}'; \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST='${HOST_CACHE_ROOT}' '${repo_root}/scripts/rootfs/enter_rootfs.sh' --rootfs-store '${ROOTFS_STORE}' -- bash -lc \"unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && torchrun --nproc_per_node=${TRAIN_NGPU} --rdzv_id '${RUN_ID}-${name}' --rdzv_backend c10d --rdzv_endpoint='127.0.0.1:0' --local_addr='127.0.0.1' --local-ranks-filter 0 --role rank --tee 3 -m ${train_module} ${stop_args} --module ${TRAIN_MODULE_REGISTRY} --config ${TRAIN_CONFIG_NAME} --output-dir '${rootfs_out}' ${resume_args}\" 2>&1 | tee '${LOG_DIR}/train_${name}.log'"
  run_with_monitors "train_${name}" "${RUN_ID}-${name}|${rootfs_out}" "${rootfs_cmd}"
  verify_train_leg "${name}" "${expected_step}" "${expected_step}"
}

vllm_probe() {
  rootfs_default "python - <<'PY'
import json
from pathlib import Path

result = {}
path = Path('${ROOTFS_MODEL_ROOT}') / 'config.json'
config = json.loads(path.read_text())
result['architectures'] = config.get('architectures')
try:
    import vllm
    result['vllm'] = getattr(vllm, '__version__', 'unknown')
except Exception as exc:
    result['selected'] = False
    result['can_serve'] = False
    result['error_type'] = type(exc).__name__
    result['error'] = str(exc)
else:
    result['selected'] = False
    result['can_serve'] = False
    try:
        import transformers
        from transformers import AutoConfig
        result['transformers'] = transformers.__version__
        cfg = AutoConfig.from_pretrained(str(path.parent), trust_remote_code=True)
        result['autoconfig'] = type(cfg).__name__
        result['reason'] = 'probe stops before loading 355B GLM into vLLM; native TorchTitan EP8 generator is selected'
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['error'] = str(exc)
        result['reason'] = 'vLLM/Transformers cannot load this GLM config; native TorchTitan EP8 generator is selected'
out = Path('${ROOTFS_RESULT_ROOT}') / 'eval' / 'vllm_probe.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\\n')
print(json.dumps(result, sort_keys=True))
PY" 2>&1 | tee "${LOG_DIR}/vllm_probe.log"
}

resume_gate() {
  train_one seed42_stop62 42 1 94 "" 62
  "${script_dir}/publish.sh" "${RESULT_ROOT}/train/seed42_stop62/checkpoint/step-62" \
    "torchtitan/checkpoints/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/step62" \
    | tee "${LOG_DIR}/publish_resume_step62.log"
  rm -rf "${RESULT_ROOT}/train/seed42_stop62/checkpoint/step-62"
  mkdir -p "${RESULT_ROOT}/train/seed42_resume_from62/checkpoint"
  "${script_dir}/stage.sh" \
    "hdfs://harunava/user/tiger/waystone/torchtitan/checkpoints/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/step62" \
    "${RESULT_ROOT}/train/seed42_resume_from62/checkpoint/step-62" \
    | tee "${LOG_DIR}/stage_resume_step62.log"
  "${script_dir}/lib/hdfs_strict.sh" --quick "${RESULT_ROOT}/train/seed42_resume_from62/checkpoint/step-62" \
    | tee "${LOG_DIR}/hdfs_strict_resume_step62.log"
  train_one seed42_resume_from62 42 1 94 62
  rootfs_train "python -m torchtitan.experiments.glm53_flash_sft_ladder.compare_resume_loss --full '${ROOTFS_RESULT_ROOT}/train/seed42_full' --resumed '${ROOTFS_RESULT_ROOT}/train/seed42_resume_from62' --output-json '${ROOTFS_RESULT_ROOT}/logs/resume_loss_compare.json' --output-md '${ROOTFS_RESULT_ROOT}/logs/resume_loss_compare.md'" \
    2>&1 | tee "${LOG_DIR}/compare_resume_loss.log"
}

export_one() {
  local train_name="$1" adapter_name="$2"
  rootfs_train "python -m torchtitan.experiments.glm53_flash.export_glm_lora --checkpoint '${ROOTFS_RESULT_ROOT}/train/${train_name}/checkpoint/step-94' --output '${ROOTFS_RESULT_ROOT}/adapters/${adapter_name}' --rank 16 --alpha 64 --base-model-name-or-path '${ROOTFS_MODEL_ROOT}'" \
    2>&1 | tee "${LOG_DIR}/export_${adapter_name}.log"
}

export_adapters() {
  export_one seed42_full seed42
  export_one seed43_full seed43
}

eval_one() {
  local name="$1" condition="$2" checkpoint_step_dir="${3:-}"
  local rootfs_out="${ROOTFS_RESULT_ROOT}/eval/dev/${name}"
  local checkpoint_args=""
  if [[ -n "${checkpoint_step_dir}" ]]; then
    checkpoint_args="--checkpoint-step-dir '${checkpoint_step_dir}'"
  fi
  local rootfs_cmd
  rootfs_cmd="export GLM53_FLASH_COUNTDOWN_DATA_ROOT='${ROOTFS_DATA_ROOT}' \
GLM53_FLASH_COUNTDOWN_DATA_FILE='${TRAIN_DATA_FILE}' \
GLM53_FLASH_COUNTDOWN_HF_ASSETS_PATH='${ROOTFS_MODEL_ROOT}' \
GLM53_FLASH_EXPERTS_IMPL='${GLM53_FLASH_EXPERTS_IMPL}' \
GLM53_FLASH_EP_DEGREE='${GLM53_FLASH_EP_DEGREE}' \
GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS='${GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS}' \
GLM53_FLASH_FULL_SEQ_LEN='${GLM53_FLASH_FULL_SEQ_LEN}' \
GLM53_FLASH_FULL_LOCAL_BATCH_SIZE='${GLM53_FLASH_FULL_LOCAL_BATCH_SIZE}' \
GLM53_FLASH_FULL_TOKENS_PER_STEP='${GLM53_FLASH_FULL_TOKENS_PER_STEP}' \
GLM53_FLASH_FULL_CHECKPOINT_LOAD_ONLY=1 \
TRITON_CACHE_DIR='${TRITON_CACHE_ROOT}' \
CUDA_VISIBLE_DEVICES='${GPU_DEVICES}' \
TORCHTITAN_ROOTFS_ENV='${ROOTFS_TRAIN_ENV}'; \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST='${HOST_CACHE_ROOT}' '${repo_root}/scripts/rootfs/enter_rootfs.sh' --rootfs-store '${ROOTFS_STORE}' -- bash -lc \"unset CC CXX CUDAHOSTCXX; cd /workspace/torchtitan && torchrun --nproc_per_node=${TRAIN_NGPU} --rdzv_id '${RUN_ID}-eval-${name}' --rdzv_backend c10d --rdzv_endpoint='127.0.0.1:0' --local_addr='127.0.0.1' --local-ranks-filter 0 --role rank --tee 3 -m torchtitan.experiments.glm53_flash.generate --module '${TRAIN_MODULE_REGISTRY}' --config '${TRAIN_CONFIG_NAME}' --problems '${DEV_PROBLEMS}' --output-dir '${rootfs_out}' ${checkpoint_args} --model-id '${name}' --condition '${condition}' --limit '${EVAL_LIMIT}' --num-samples '${EVAL_SAMPLES}' --temperature 0.8 --top-p 0.95 --max-new-tokens 192 --decode-batch-size '${EVAL_DECODE_BATCH_SIZE}'\" 2>&1 | tee '${LOG_DIR}/eval_${name}.log'"
  run_with_monitors "eval_${name}" "${RUN_ID}-eval-${name}|${rootfs_out}" "${rootfs_cmd}"
}

eval_gate() {
  vllm_probe
  eval_one base base
  eval_one seed42 adapter "${ROOTFS_RESULT_ROOT}/train/seed42_full/checkpoint/step-94"
  eval_one seed43 adapter "${ROOTFS_RESULT_ROOT}/train/seed43_full/checkpoint/step-94"
}

write_attempt_json() {
  local git_revision dirty selected_json selected_store_id native_lock_sha runtime_lock_sha
  git_revision="$(git -C "${repo_root}" rev-parse HEAD)"
  dirty="false"
  if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    dirty="true"
  fi
  selected_json="${ROOTFS_STORE_SELECTION}"
  if [[ -f "${selected_json}" ]]; then
    selected_store_id="$(jq -r '.store_id // ""' "${selected_json}")"
  else
    selected_store_id=""
  fi
  native_lock_sha="$(md5sum "${repo_root}/scripts/rootfs/native_variant.lock" | awk '{print $1}')"
  runtime_lock_sha="$(md5sum "${repo_root}/experiments/glm53_flash_sft_ladder/runtime/requirements.lock" | awk '{print $1}')"

  jq -n \
    --arg run_id "${RUN_ID}" \
    --arg attempt_id "${ATTEMPT_ID}" \
    --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg git_revision "${git_revision}" \
    --argjson git_dirty "${dirty}" \
    --arg result_root "${RESULT_ROOT}" \
    --arg rootfs_result_root "${ROOTFS_RESULT_ROOT}" \
    --arg rootfs_env "${ROOTFS_TRAIN_ENV}" \
    --arg rootfs_store "${ROOTFS_STORE}" \
    --arg rootfs_store_selection "${selected_json}" \
    --arg rootfs_store_id "${selected_store_id}" \
    --arg native_variant_lock_md5 "${native_lock_sha}" \
    --arg runtime_requirements_lock_md5 "${runtime_lock_sha}" \
    --arg formal_cache_host "${HOST_CACHE_ROOT}" \
    --arg train_module_registry "${TRAIN_MODULE_REGISTRY}" \
    --arg train_config_name "${TRAIN_CONFIG_NAME}" \
    --arg experts_impl "${GLM53_FLASH_EXPERTS_IMPL}" \
    --argjson ep_degree "${GLM53_FLASH_EP_DEGREE}" \
    --argjson num_layers "${GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS}" \
    --argjson seq_len "${GLM53_FLASH_FULL_SEQ_LEN}" \
    --argjson local_batch_size "${GLM53_FLASH_FULL_LOCAL_BATCH_SIZE}" \
    --argjson tokens_per_step "${GLM53_FLASH_FULL_TOKENS_PER_STEP}" \
    --argjson checkpoint_interval "${GLM53_FLASH_FULL_CHECKPOINT_INTERVAL}" \
    --argjson keep_latest_k "${GLM53_FLASH_FULL_KEEP_LATEST_K}" \
    --argjson loss_chunks "${GLM53_FLASH_LOSS_CHUNKS}" \
    --argjson eval_limit "${EVAL_LIMIT}" \
    --argjson eval_samples "${EVAL_SAMPLES}" \
    --argjson eval_decode_batch_size "${EVAL_DECODE_BATCH_SIZE}" \
    --arg train_split "${TRAIN_COUNTDOWN_SPLIT}" \
    --argjson train_max_docs "${TRAIN_MAX_DOCS}" \
    --arg deepep_native_build_key "${DEEPEP_NATIVE_BUILD_KEY}" \
    --arg deepep_native_hdfs_uri "${DEEPEP_NATIVE_HDFS_URI}" \
    --arg deepep_native_manifest_md5 "${DEEPEP_NATIVE_MANIFEST_MD5}" \
    --arg torchao_native_build_key "${TORCHAO_NATIVE_BUILD_KEY}" \
    --arg torchao_native_hdfs_uri "${TORCHAO_NATIVE_HDFS_URI}" \
    --arg torchao_native_manifest_md5 "${TORCHAO_NATIVE_MANIFEST_MD5}" \
    --slurpfile dataset "${HOST_DATA_ROOT}/.staged.json" \
    --slurpfile model "${HOST_MODEL_ROOT}/.staged.json" \
    '{
      run_id: $run_id,
      attempt_id: $attempt_id,
      created_at: $created_at,
      source: {
        revision: $git_revision,
        dirty: $git_dirty
      },
      result_root: $result_root,
      rootfs_result_root: $rootfs_result_root,
      rootfs: {
        env: $rootfs_env,
        store_root: $rootfs_store,
        selected_json: $rootfs_store_selection,
        selected_store_id: $rootfs_store_id,
        native_variant_lock_md5: $native_variant_lock_md5,
        runtime_requirements_lock_md5: $runtime_requirements_lock_md5,
        native_wheels: {
          deepep_v2: {
            build_key: $deepep_native_build_key,
            hdfs_uri: $deepep_native_hdfs_uri,
            manifest_md5: $deepep_native_manifest_md5
          },
          torchao: {
            build_key: $torchao_native_build_key,
            hdfs_uri: $torchao_native_hdfs_uri,
            manifest_md5: $torchao_native_manifest_md5
          }
        }
      },
      formal_cache_host: $formal_cache_host,
      staged_inputs: {
        dataset: $dataset[0],
        model: $model[0]
      },
      training: {
        module: $train_module_registry,
        config: $train_config_name,
        experts_impl: $experts_impl,
        ep_degree: $ep_degree,
        num_layers: $num_layers,
        seq_len: $seq_len,
        local_batch_size: $local_batch_size,
        tokens_per_step: $tokens_per_step,
        steps: 94,
        checkpoint_interval: $checkpoint_interval,
        keep_latest_k: $keep_latest_k,
        loss_chunks: $loss_chunks,
        countdown_split: $train_split,
        countdown_max_docs: $train_max_docs,
        countdown_seed_matches_debug_seed: true,
        lora: {rank: 16, alpha: 64}
      },
      eval: {
        split: "dev",
        subset: "first N problems",
        num_problems: $eval_limit,
        num_samples_per_problem: $eval_samples,
        temperature: 0.8,
        top_p: 0.95,
        max_new_tokens: 192,
        decode_batch_size: $eval_decode_batch_size
      }
    }' > "${RESULT_ROOT}/attempt.json"
}

write_manifest() {
  write_attempt_json
  jq -n \
    --arg run_id "${RUN_ID}" \
    --arg attempt_id "${ATTEMPT_ID}" \
    --arg dataset_root "${HOST_DATA_ROOT}" \
    --arg model_root "${HOST_MODEL_ROOT}" \
    --arg triton_cache "${TRITON_CACHE_ROOT}" \
    --argjson eval_limit "${EVAL_LIMIT}" \
    --argjson eval_samples "${EVAL_SAMPLES}" \
    --argjson eval_decode_batch_size "${EVAL_DECODE_BATCH_SIZE}" \
    --arg train_split "${TRAIN_COUNTDOWN_SPLIT}" \
    --argjson train_max_docs "${TRAIN_MAX_DOCS}" \
    --arg rootfs_env "${ROOTFS_TRAIN_ENV}" \
    --arg rootfs_store "${ROOTFS_STORE}" \
    --arg train_module_registry "${TRAIN_MODULE_REGISTRY}" \
    --arg train_config_name "${TRAIN_CONFIG_NAME}" \
    --arg experts_impl "${GLM53_FLASH_EXPERTS_IMPL}" \
    --argjson ep_degree "${GLM53_FLASH_EP_DEGREE}" \
    --argjson num_layers "${GLM53_FLASH_EP_TRUNCATED_NUM_LAYERS}" \
    --argjson seq_len "${GLM53_FLASH_FULL_SEQ_LEN}" \
    --argjson tokens_per_step "${GLM53_FLASH_FULL_TOKENS_PER_STEP}" \
    --argjson checkpoint_interval "${GLM53_FLASH_FULL_CHECKPOINT_INTERVAL}" \
    --slurpfile dataset "${HOST_DATA_ROOT}/.staged.json" \
    --slurpfile model "${HOST_MODEL_ROOT}/.staged.json" \
    '{
      run_id: $run_id,
      attempt_id: $attempt_id,
      surface: "repo-local GLM experiment",
      model: "zai-org/GLM-5.3-Flash-BF16",
      config: $train_config_name,
      module: $train_module_registry,
      dataset_root: $dataset_root,
      model_root: $model_root,
      triton_cache: $triton_cache,
      staged_inputs: {dataset: $dataset[0], model: $model[0]},
      rootfs: {env: $rootfs_env, store_root: $rootfs_store},
      training: {
        experts_impl: $experts_impl,
        ep_degree: $ep_degree,
        num_layers: $num_layers,
        steps: 94,
        seq_len: $seq_len,
        tokens_per_step: $tokens_per_step,
        checkpoint_interval: $checkpoint_interval,
        countdown_split: $train_split,
        countdown_max_docs: $train_max_docs,
        lora: {rank: 16, alpha: 64}
      },
      eval: {
        split: "dev",
        subset: "first N problems",
        num_problems: $eval_limit,
        num_samples_per_problem: $eval_samples,
        temperature: 0.8,
        top_p: 0.95,
        max_new_tokens: 192,
        decode_batch_size: $eval_decode_batch_size
      }
    }' > "${RESULT_ROOT}/run_manifest.json"
}

publish_outputs() {
  write_manifest
  local publish_root="${script_dir}/cache/publish-bundles/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}"
  rm -rf "${publish_root}"
  mkdir -p "${publish_root}"
  cp "${RESULT_ROOT}/attempt.json" "${publish_root}/attempt.json"
  cp "${RESULT_ROOT}/run_manifest.json" "${publish_root}/run_manifest.json"
  cp -R "${LOG_DIR}" "${publish_root}/logs"
  [[ -d "${RESULT_ROOT}/configs" ]] && cp -R "${RESULT_ROOT}/configs" "${publish_root}/configs"
  [[ -d "${RESULT_ROOT}/adapters" ]] && cp -R "${RESULT_ROOT}/adapters" "${publish_root}/adapters"
  [[ -d "${RESULT_ROOT}/eval" ]] && cp -R "${RESULT_ROOT}/eval" "${publish_root}/eval"
  if [[ -d "${RESULT_ROOT}/train" ]]; then
    mkdir -p "${publish_root}/train"
    rsync -a --exclude checkpoint "${RESULT_ROOT}/train/" "${publish_root}/train/"
  fi
  "${script_dir}/publish.sh" "${RESULT_ROOT}/adapters" \
    "torchtitan/runs/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/adapters" \
    | tee "${LOG_DIR}/publish_adapters.log"
  "${script_dir}/publish.sh" "${RESULT_ROOT}/eval" \
    "torchtitan/runs/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/eval" \
    | tee "${LOG_DIR}/publish_eval.log"
  "${script_dir}/publish.sh" "${LOG_DIR}" \
    "torchtitan/runs/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/logs" \
    | tee "${LOG_DIR}/publish_logs.log"
  if [[ -d "${RESULT_ROOT}/configs" ]]; then
    "${script_dir}/publish.sh" "${RESULT_ROOT}/configs" \
      "torchtitan/runs/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/configs" \
      | tee "${LOG_DIR}/publish_configs.log"
  fi
  "${script_dir}/publish.sh" "${publish_root}" \
    "torchtitan/runs/glm53-ladder/rung4/${RUN_ID}/${ATTEMPT_ID}/bundle" \
    | tee "${LOG_DIR}/publish_bundle.log"
}

prune_checkpoints() {
  if [[ -d "${RESULT_ROOT}/train" ]]; then
    find "${RESULT_ROOT}/train" -mindepth 2 -maxdepth 2 -type d -name checkpoint -print -exec rm -rf {} +
  fi
}

cmd="${1:-all}"
case "${cmd}" in
  all)
    check_inputs
    print_configs
    train_one seed42_full 42 1 94
    resume_gate
    train_one seed43_full 43 1 94
    export_adapters
    eval_gate
    publish_outputs
    prune_checkpoints
    ;;
  preflight) check_inputs; print_configs; vllm_probe ;;
  train) check_inputs; train_one seed42_full 42 1 94 ;;
  resume) check_inputs; resume_gate ;;
  seed43) check_inputs; train_one seed43_full 43 1 94 ;;
  export) check_inputs; export_adapters ;;
  eval) check_inputs; eval_gate ;;
  publish) publish_outputs ;;
  prune) prune_checkpoints ;;
  --help|-h) usage ;;
  *) usage; exit 2 ;;
esac
