#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
queue_dir="${script_dir}/queue"
cells_file="${GLM53_ARCH_CELLS_FILE:-${queue_dir}/cells.jsonl}"
state_dir="${GLM53_ARCH_STATE_DIR:-${queue_dir}/state}"
results_dir="${GLM53_ARCH_RESULTS_DIR:-${script_dir}/results}"
logs_dir="${GLM53_ARCH_LOGS_DIR:-${script_dir}/logs}"
runner="${GLM53_ARCH_RUNNER:-${script_dir}/run_arm.sh}"
publish_script="${GLM53_ARCH_PUBLISH_SCRIPT:-${repo_root}/experiments/glm53_flash_sft_ladder/publish.sh}"
analysis_script="experiments/glm53_arch_speedrun/analyze_results.py"
s01_report="${GLM53_ARCH_S01_REPORT:-.scratch/glm53-arch-speedrun/sdd/s01-report.md}"
s02_report="${GLM53_ARCH_S02_REPORT:-.scratch/glm53-arch-speedrun/sdd/s02-report.md}"
poll_sec="${GLM53_ARCH_POLL_SEC:-120}"
max_concurrent="${GLM53_ARCH_MAX_CONCURRENT:-6}"
max_per_gpu="${GLM53_ARCH_MAX_PER_GPU:-2}"
min_free_gib="${GLM53_ARCH_MIN_FREE_GIB:-30}"
current_user="${GLM53_ARCH_USER:-$(id -un)}"
dry_run=0
once=0
no_wait=0

usage() {
  cat >&2 <<'USAGE'
Usage: experiments/glm53_arch_speedrun/scheduler.sh [--dry-run] [--once] [--no-wait]

Host-only opportunistic scheduler for GLM-5.3 architecture speedrun ablations.
It waits for green S01/S02 reports, reads queue/cells.jsonl, and launches the
next runnable cell through experiments/glm53_arch_speedrun/run_arm.sh.

Environment overrides:
  GLM53_ARCH_RUNNER          runner path (default: experiments/glm53_arch_speedrun/run_arm.sh)
  GLM53_ARCH_RUNNER_CMD      shell command to execute instead of the default runner call
  GLM53_ARCH_MAX_CONCURRENT  max concurrent cells (default: 6)
  GLM53_ARCH_MAX_PER_GPU     max cells per GPU launched by this scheduler (default: 2)
  GLM53_ARCH_MIN_FREE_GIB    required nvidia-smi free memory per GPU (default: 30)
  GLM53_ARCH_POLL_SEC        loop sleep seconds (default: 120)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --once)
      once=1
      shift
      ;;
    --no-wait)
      no_wait=1
      shift
      ;;
    --poll-sec)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      poll_sec="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${state_dir}" "${results_dir}" "${logs_dir}"

require_tool() {
  local tool="$1"
  if ! command -v "${tool}" >/dev/null 2>&1; then
    printf 'missing required tool: %s\n' "${tool}" >&2
    exit 2
  fi
}

log_decision() {
  local action="$1"
  shift || true
  printf 'ts=%s action=%s' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${action}"
  local item
  for item in "$@"; do
    printf ' %s' "${item}"
  done
  printf '\n'
}

json_update() {
  local path="$1"
  shift
  local tmp="${path}.tmp.$$"
  jq "$@" "${path}" > "${tmp}"
  mv "${tmp}" "${path}"
}

cell_count() {
  jq -r 'select(.runnable != false) | .cell_id' "${cells_file}" | wc -l | tr -d '[:space:]'
}

cell_json() {
  local cell_id="$1"
  jq -c --arg cell_id "${cell_id}" 'select(.cell_id == $cell_id)' "${cells_file}"
}

state_path() {
  printf '%s/%s.json\n' "${state_dir}" "$1"
}

state_status() {
  local cell_id="$1" path
  path="$(state_path "${cell_id}")"
  if [[ ! -f "${path}" ]]; then
    printf 'missing\n'
    return
  fi
  jq -r '.status' "${path}"
}

init_states() {
  local cell_id path
  while IFS= read -r cell_id; do
    [[ -n "${cell_id}" ]] || continue
    path="$(state_path "${cell_id}")"
    if [[ -f "${path}" ]]; then
      continue
    fi
    jq -n --arg cell_id "${cell_id}" --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{cell_id:$cell_id,status:"pending",attempts:0,created_at:$created_at}' > "${path}"
  done < <(jq -r 'select(.runnable != false) | .cell_id' "${cells_file}")
}

report_has_failed_gate() {
  local report="$1"
  rg -iq 'gate[^[:alnum:]]*(failed|fail|red)|status[^[:alnum:]]*(failed|fail)|failed gate|red harness' "${report}"
}

report_has_passed_gate() {
  local report="$1"
  rg -iq 's03_gate[=: ][[:space:]]*pass|gate[^[:alnum:]]*(passed|pass|green)|status[^[:alnum:]]*(passed|pass|green)' "${report}"
}

report_path() {
  local report="$1"
  if [[ "${report}" == /* ]]; then
    printf '%s\n' "${report}"
  else
    printf '%s/%s\n' "${repo_root}" "${report}"
  fi
}

gate_status() {
  local report path
  for report in "${s01_report}" "${s02_report}"; do
    path="$(report_path "${report}")"
    if [[ ! -f "${path}" ]]; then
      printf 'waiting missing=%s\n' "${report}"
      return
    fi
    if report_has_failed_gate "${path}"; then
      printf 'failed report=%s\n' "${report}"
      return
    fi
    if ! report_has_passed_gate "${path}"; then
      printf 'waiting report=%s reason=no_explicit_pass\n' "${report}"
      return
    fi
  done
  if [[ ! -x "${runner}" && -z "${GLM53_ARCH_RUNNER_CMD:-}" ]]; then
    printf 'waiting missing_runner=%s\n' "${runner}"
    return
  fi
  printf 'passed\n'
}

running_count() {
  local count=0 path pid
  for path in "${state_dir}"/*.json; do
    [[ -f "${path}" ]] || continue
    if [[ "$(jq -r '.status' "${path}")" != "running" ]]; then
      continue
    fi
    pid="$(jq -r '.pid // empty' "${path}")"
    if [[ -n "${pid}" && -d "/proc/${pid}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

running_on_gpu_count() {
  local gpu="$1"
  local count=0 path pid devices
  for path in "${state_dir}"/*.json; do
    [[ -f "${path}" ]] || continue
    if [[ "$(jq -r '.status' "${path}")" != "running" ]]; then
      continue
    fi
    pid="$(jq -r '.pid // empty' "${path}")"
    if [[ -z "${pid}" || ! -d "/proc/${pid}" ]]; then
      continue
    fi
    devices="$(jq -r '.devices // ""' "${path}")"
    IFS=',' read -r -a parts <<< "${devices}"
    local part
    for part in "${parts[@]}"; do
      if [[ "${part}" == "${gpu}" ]]; then
        count=$((count + 1))
      fi
    done
  done
  printf '%s\n' "${count}"
}

refresh_running_states() {
  local path status pid exit_file exit_code attempts cell_id failure_class
  for path in "${state_dir}"/*.json; do
    [[ -f "${path}" ]] || continue
    status="$(jq -r '.status' "${path}")"
    [[ "${status}" == "running" ]] || continue
    cell_id="$(jq -r '.cell_id' "${path}")"
    exit_file="$(jq -r '.exit_code_file // empty' "${path}")"
    if [[ -n "${exit_file}" && -f "${exit_file}" ]]; then
      exit_code="$(tr -d '[:space:]' < "${exit_file}")"
      if [[ "${exit_code}" == "0" ]]; then
        json_update "${path}" \
          --arg finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          --argjson exit_code 0 \
          '.status="done" | .exit_code=$exit_code | .finished_at=$finished_at'
        log_decision "done" "cell=${cell_id}" "exit_code=0"
        publish_cell "${cell_id}" || true
      else
        failure_class="$(classify_failure "${path}")"
        attempts="$(jq -r '.attempts' "${path}")"
        if [[ "${failure_class}" == "infrastructure" && "${attempts}" -lt 2 ]]; then
          json_update "${path}" \
            --arg finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --arg failure_class "${failure_class}" \
            --argjson exit_code "${exit_code}" \
            '.status="pending" | .exit_code=$exit_code | .last_failure_class=$failure_class | .last_finished_at=$finished_at'
          log_decision retry_pending "cell=${cell_id}" "exit_code=${exit_code}" "class=${failure_class}" "attempts=${attempts}"
        else
          json_update "${path}" \
            --arg finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --arg failure_class "${failure_class}" \
            --argjson exit_code "${exit_code}" \
            '.status="failed" | .exit_code=$exit_code | .failure_class=$failure_class | .finished_at=$finished_at'
          log_decision failed "cell=${cell_id}" "exit_code=${exit_code}" "class=${failure_class}"
        fi
      fi
      continue
    fi
    pid="$(jq -r '.pid // empty' "${path}")"
    if [[ -n "${pid}" && -d "/proc/${pid}" ]]; then
      continue
    fi
    attempts="$(jq -r '.attempts' "${path}")"
    if [[ "${attempts}" -lt 2 ]]; then
      json_update "${path}" \
        --arg finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '.status="pending" | .last_failure_class="infrastructure" | .last_failure_reason="stale_running_without_exit_code" | .last_finished_at=$finished_at'
      log_decision stale_retry "cell=${cell_id}" "attempts=${attempts}"
    else
      json_update "${path}" \
        --arg finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '.status="failed" | .failure_class="infrastructure" | .failure_reason="stale_running_without_exit_code" | .finished_at=$finished_at'
      log_decision stale_failed "cell=${cell_id}" "attempts=${attempts}"
    fi
  done
}

classify_failure() {
  local state_path="$1"
  local attempt_dir log_file
  attempt_dir="$(jq -r '.attempt_dir // empty' "${state_path}")"
  log_file="$(jq -r '.log_file // empty' "${state_path}")"
  if [[ -n "${attempt_dir}" ]] && rg -iq '"failure_class"[[:space:]]*:[[:space:]]*"numeric"|non[- ]?finite|nan|inf|loss.*not finite|numeric' "${attempt_dir}" 2>/dev/null; then
    printf 'numeric\n'
    return
  fi
  if [[ -n "${log_file}" && -f "${log_file}" ]] && rg -iq 'nccl|rendezvous|torchrun|address already in use|connection reset|timed out|launch|cuda.*busy|processgroup' "${log_file}"; then
    printf 'infrastructure\n'
    return
  fi
  if [[ -n "${attempt_dir}" ]] && rg -iq '"failure_class"[[:space:]]*:[[:space:]]*"infrastructure"|nccl|rendezvous|torchrun|address already in use|connection reset|timed out|launch' "${attempt_dir}" 2>/dev/null; then
    printf 'infrastructure\n'
    return
  fi
  printf 'unknown\n'
}

gpu_free_table() {
  nvidia-smi --query-gpu=index,uuid,compute_mode,memory.free --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); gsub(/[[:space:]]/, "", $4); print $1 " " $2 " " $3 " " $4}'
}

gpu_has_foreign_process() {
  local uuid="$1"
  local app_uuid pid owner
  while IFS=',' read -r app_uuid pid; do
    app_uuid="${app_uuid//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    [[ -n "${app_uuid}" && -n "${pid}" && "${app_uuid}" == "${uuid}" ]] || continue
    owner="$(ps -o user:64= -p "${pid}" 2>/dev/null | awk 'NR==1 {print $1}')"
    if [[ -z "${owner}" || "${owner}" != "${current_user}" ]]; then
      return 0
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
  return 1
}

rootfs_repo_path() {
  local path="$1"
  case "${path}" in
    "${repo_root}"/*)
      printf '%s\n' "${path#"${repo_root}/"}"
      ;;
    *)
      printf '%s\n' "${path}"
      ;;
  esac
}

select_devices() {
  local needed="$1"
  local selected="" index uuid compute_mode free_mb per_count min_free_mb
  min_free_mb=$((min_free_gib * 1024))
  while read -r index uuid compute_mode free_mb; do
    [[ -n "${index}" ]] || continue
    if [[ "${compute_mode}" == *Exclusive_Process* ]]; then
      continue
    fi
    if gpu_has_foreign_process "${uuid}"; then
      continue
    fi
    per_count="$(running_on_gpu_count "${index}")"
    if (( free_mb >= min_free_mb && per_count < max_per_gpu )); then
      if [[ -n "${selected}" ]]; then
        selected+=","
      fi
      selected+="${index}"
      needed=$((needed - 1))
      if (( needed == 0 )); then
        printf '%s\n' "${selected}"
        return 0
      fi
    fi
  done < <(gpu_free_table)
  return 1
}

slice_orders() {
  jq -r 'select(.runnable != false) | .slice_order' "${cells_file}" | sort -n -u
}

slice_name_for_order() {
  local order="$1"
  jq -r --argjson order "${order}" 'select(.slice_order == $order) | .slice' "${cells_file}" | head -n 1
}

slice_has_failed() {
  local order="$1" cell_id
  while IFS= read -r cell_id; do
    [[ -n "${cell_id}" ]] || continue
    if [[ "$(state_status "${cell_id}")" == "failed" ]]; then
      return 0
    fi
  done < <(jq -r --argjson order "${order}" 'select(.runnable != false and .slice_order == $order) | .cell_id' "${cells_file}")
  return 1
}

slice_all_done() {
  local order="$1" cell_id
  while IFS= read -r cell_id; do
    [[ -n "${cell_id}" ]] || continue
    if [[ "$(state_status "${cell_id}")" != "done" ]]; then
      return 1
    fi
  done < <(jq -r --argjson order "${order}" 'select(.runnable != false and .slice_order == $order) | .cell_id' "${cells_file}")
  return 0
}

current_slice_order() {
  local order
  while IFS= read -r order; do
    [[ -n "${order}" ]] || continue
    if ! slice_all_done "${order}"; then
      printf '%s\n' "${order}"
      return 0
    fi
  done < <(slice_orders)
  return 1
}

next_pending_cell_for_slice() {
  local order="$1" cell_id
  while IFS= read -r cell_id; do
    [[ -n "${cell_id}" ]] || continue
    if [[ "$(state_status "${cell_id}")" == "pending" ]]; then
      printf '%s\n' "${cell_id}"
      return 0
    fi
  done < <(jq -r --argjson order "${order}" 'select(.runnable != false and .slice_order == $order) | .cell_id' "${cells_file}")
  return 1
}

publish_cell() {
  local cell_id="$1" path attempt_dir run_id publish_log publish_tmp_log
  path="$(state_path "${cell_id}")"
  attempt_dir="$(jq -r '.attempt_dir // empty' "${path}")"
  run_id="$(jq -r '.run_id // empty' "${path}")"
  if [[ -z "${attempt_dir}" || -z "${run_id}" || ! -d "${attempt_dir}" ]]; then
    log_decision publish_skip "cell=${cell_id}" "reason=missing_attempt_dir"
    return 0
  fi
  if [[ ! -x "${publish_script}" ]]; then
    log_decision publish_skip "cell=${cell_id}" "reason=missing_publish_script"
    return 0
  fi
  publish_log="${attempt_dir}/publish.log"
  publish_tmp_log="${logs_dir}/publish_${cell_id}_${run_id}.log"
  if "${publish_script}" --parallel "${GLM53_ARCH_PUBLISH_PARALLEL:-2}" "${attempt_dir}" "runs/glm53-arch-speedrun/${run_id}" > "${publish_tmp_log}" 2>&1; then
    cp "${publish_tmp_log}" "${publish_log}"
    json_update "${path}" \
      --arg published_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --arg hdfs_uri "$(awk -F= '/^HDFS_URI=/{print $2}' "${publish_tmp_log}" | tail -n 1)" \
      '.published=true | .published_at=$published_at | .hdfs_uri=$hdfs_uri'
    log_decision published "cell=${cell_id}" "run_id=${run_id}"
  else
    cp "${publish_tmp_log}" "${publish_log}" 2>/dev/null || true
    json_update "${path}" '.published=false | .publish_failed=true'
    log_decision publish_failed "cell=${cell_id}" "run_id=${run_id}"
    return 1
  fi
}

run_slice_analysis_if_ready() {
  local order="$1" slice_name marker analysis_log rootfs_cells rootfs_state rootfs_analysis rootfs_doc rootfs_report
  slice_name="$(slice_name_for_order "${order}")"
  marker="${state_dir}/.${slice_name}.analysis.done"
  if [[ -f "${marker}" ]]; then
    return 0
  fi
  if ! slice_all_done "${order}"; then
    return 0
  fi
  analysis_log="${logs_dir}/analysis_${slice_name}.log"
  rootfs_cells="$(rootfs_repo_path "${cells_file}")"
  rootfs_state="$(rootfs_repo_path "${state_dir}")"
  rootfs_analysis="$(rootfs_repo_path "${GLM53_ARCH_ANALYSIS_DIR:-${results_dir}/analysis}")"
  rootfs_doc="$(rootfs_repo_path "${GLM53_ARCH_DOC_PATH:-${repo_root}/docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md}")"
  rootfs_report="$(rootfs_repo_path "${GLM53_ARCH_REPORT_PATH:-${repo_root}/.scratch/glm53-arch-speedrun/sdd/s03-report.md}")"
  if TORCHTITAN_ROOTFS_ENV=glm53_train "${repo_root}/scripts/rootfs/enter_rootfs.sh" -- bash -lc \
    "unset CC CXX; cd /workspace/torchtitan && GLM53_ARCH_CELLS_FILE='${rootfs_cells}' GLM53_ARCH_STATE_DIR='${rootfs_state}' GLM53_ARCH_ANALYSIS_DIR='${rootfs_analysis}' GLM53_ARCH_DOC_PATH='${rootfs_doc}' GLM53_ARCH_REPORT_PATH='${rootfs_report}' GLM53_ARCH_REPORT_MODE='replace' python ${analysis_script} --slice '${slice_name}' --check-inputs" \
    > "${analysis_log}" 2>&1; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "${marker}"
    log_decision analysis_done "slice=${slice_name}"
  else
    log_decision analysis_failed "slice=${slice_name}" "log=${analysis_log}"
    return 1
  fi
}

run_ready_analyses() {
  local order
  while IFS= read -r order; do
    [[ -n "${order}" ]] || continue
    run_slice_analysis_if_ready "${order}"
  done < <(slice_orders)
}

launch_cell() {
  local cell_id="$1" devices="$2" path attempts attempt_id run_id attempt_dir log_file exit_file cell_file wrapper
  path="$(state_path "${cell_id}")"
  attempts="$(jq -r '.attempts' "${path}")"
  attempts=$((attempts + 1))
  attempt_id="$(printf 'attempt-%02d' "${attempts}")"
  run_id="${cell_id}-$(date -u +%Y%m%dT%H%M%SZ)"
  attempt_dir="${results_dir}/${run_id}/${attempt_id}"
  log_file="${attempt_dir}/cell.log"
  exit_file="${attempt_dir}/exit_code"
  cell_file="${attempt_dir}/cell.json"
  wrapper="${attempt_dir}/run_wrapper.sh"
  if (( dry_run == 1 )); then
    log_decision dry_launch "cell=${cell_id}" "devices=${devices}" "attempt=${attempt_id}"
    return 0
  fi
  mkdir -p "${attempt_dir}"
  cell_json "${cell_id}" > "${cell_file}"
  cat > "${wrapper}" <<'WRAPPER'
#!/usr/bin/env bash
set +e
printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${GLM53_ARCH_ATTEMPT_DIR}/attempt.env"
if [[ -n "${GLM53_ARCH_RUNNER_CMD:-}" ]]; then
  bash -lc "${GLM53_ARCH_RUNNER_CMD}"
else
  "${GLM53_ARCH_RUNNER}" run-cell --cell-json "${GLM53_ARCH_CELL_JSON}" --output-dir "${GLM53_ARCH_ATTEMPT_DIR}"
fi
rc=$?
printf '%s\n' "${rc}" > "${GLM53_ARCH_EXIT_CODE_FILE}"
printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${GLM53_ARCH_ATTEMPT_DIR}/attempt.env"
exit "${rc}"
WRAPPER
  chmod +x "${wrapper}"
  env \
    GLM53_ARCH_RUNNER="${runner}" \
    GLM53_ARCH_CELL_ID="${cell_id}" \
    GLM53_ARCH_CELL_JSON="${cell_file}" \
    GLM53_ARCH_ATTEMPT_DIR="${attempt_dir}" \
    GLM53_ARCH_RUN_ID="${run_id}" \
    GLM53_ARCH_ATTEMPT_ID="${attempt_id}" \
    GLM53_ARCH_GPU_MEMORY_CAP_GIB=24 \
    GLM53_ARCH_EXIT_CODE_FILE="${exit_file}" \
    CUDA_VISIBLE_DEVICES="${devices}" \
    GLM53_SPEEDRUN_BEHAVIOR_READS="${GLM53_SPEEDRUN_BEHAVIOR_READS:-1}" \
    "${wrapper}" > "${log_file}" 2>&1 &
  local pid=$!
  jq -n \
    --arg cell_id "${cell_id}" \
    --arg status "running" \
    --arg run_id "${run_id}" \
    --arg attempt_id "${attempt_id}" \
    --arg attempt_dir "${attempt_dir}" \
    --arg log_file "${log_file}" \
    --arg exit_code_file "${exit_file}" \
    --arg devices "${devices}" \
    --arg started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson attempts "${attempts}" \
    --argjson pid "${pid}" \
    '{cell_id:$cell_id,status:$status,attempts:$attempts,run_id:$run_id,attempt_id:$attempt_id,attempt_dir:$attempt_dir,log_file:$log_file,exit_code_file:$exit_code_file,devices:$devices,pid:$pid,started_at:$started_at}' \
    > "${path}"
  log_decision launched "cell=${cell_id}" "devices=${devices}" "pid=${pid}" "run_id=${run_id}" "attempt=${attempt_id}"
}

try_launch_one() {
  local gate order slice_name cell_id needed devices active
  gate="$(gate_status)"
  if [[ "${gate}" != "passed" ]]; then
    log_decision gate_wait "${gate}"
    return 0
  fi
  refresh_running_states
  active="$(running_count)"
  if (( active >= max_concurrent )); then
    log_decision capacity_wait "active=${active}" "max=${max_concurrent}"
    return 0
  fi
  run_ready_analyses
  if ! order="$(current_slice_order)"; then
    log_decision queue_complete
    return 0
  fi
  slice_name="$(slice_name_for_order "${order}")"
  if slice_has_failed "${order}"; then
    log_decision slice_failed_stop "slice=${slice_name}"
    return 1
  fi
  if ! cell_id="$(next_pending_cell_for_slice "${order}")"; then
    log_decision slice_wait "slice=${slice_name}" "reason=running_or_analysis"
    return 0
  fi
  needed="$(cell_json "${cell_id}" | jq -r '.resources.num_gpus // 1')"
  if ! devices="$(select_devices "${needed}")"; then
    log_decision gpu_wait "cell=${cell_id}" "needed=${needed}" "min_free_gib=${min_free_gib}"
    return 0
  fi
  launch_cell "${cell_id}" "${devices}"
}

main() {
  require_tool jq
  require_tool rg
  require_tool awk
  require_tool nvidia-smi
  if [[ ! -f "${cells_file}" ]]; then
    printf 'cells file does not exist: %s\n' "${cells_file}" >&2
    exit 2
  fi
  init_states
  log_decision scheduler_start "cells=$(cell_count)" "dry_run=${dry_run}" "once=${once}" "poll_sec=${poll_sec}"
  while true; do
    try_launch_one
    if (( once == 1 )); then
      break
    fi
    if (( no_wait == 1 )); then
      break
    fi
    sleep "${poll_sec}"
  done
}

main "$@"
