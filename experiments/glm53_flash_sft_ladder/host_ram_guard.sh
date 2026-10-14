#!/usr/bin/env bash
# Kill this user's current GLM ladder job if host available RAM drops too low.
set -u

floor_gb="${1:-${GLM53_HOST_RAM_FLOOR_GB:-800}}"
log="${2:-$(dirname "$0")/logs/host_ram_guard.log}"
pattern="${3:-${GLM53_HOST_RAM_GUARD_PATTERN:-torchtitan.experiments.glm53_flash.config_registry}}"
interval_seconds="${GLM53_HOST_RAM_GUARD_INTERVAL_SECONDS:-30}"
kill_grace_seconds="${GLM53_HOST_RAM_GUARD_KILL_GRACE_SECONDS:-20}"

mkdir -p "$(dirname "${log}")"

children_of() {
  local parent="$1"
  local child
  pgrep -P "${parent}" 2>/dev/null || true
  for child in $(pgrep -P "${parent}" 2>/dev/null || true); do
    children_of "${child}"
  done
}

guard_pids() {
  local pid
  {
    pgrep -u "$(id -u)" -f -- "${pattern}" 2>/dev/null || true
    for pid in $(pgrep -u "$(id -u)" -f -- "${pattern}" 2>/dev/null || true); do
      children_of "${pid}"
    done
  } | awk -v self="$$" '$1 != self { seen[$1]=1 } END { for (pid in seen) print pid }'
}

log_snapshot() {
  local timestamp="$1"
  local avail_gb="$2"
  local pids="$3"
  local pid ppid rss_kb rss_gb state cmd
  {
    printf '%s available_gb=%s floor_gb=%s pattern=%q pids=%q\n' \
      "${timestamp}" "${avail_gb}" "${floor_gb}" "${pattern}" "${pids}"
    for pid in ${pids}; do
      [[ -r "/proc/${pid}/status" ]] || continue
      ppid="$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status")"
      rss_kb="$(awk '/^VmRSS:/ {print $2}' "/proc/${pid}/status")"
      state="$(awk '/^State:/ {print $2}' "/proc/${pid}/status")"
      rss_kb="${rss_kb:-0}"
      rss_gb="$(awk -v kb="${rss_kb}" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
      cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)"
      printf '  pid=%s ppid=%s state=%s rss_gb=%s cmd=%s\n' \
        "${pid}" "${ppid:-?}" "${state:-?}" "${rss_gb}" "${cmd:-?}"
    done
  } >> "${log}"
}

while true; do
  timestamp="$(date -u +%FT%TZ)"
  avail_gb="$(free -g | awk 'NR==2{print $7}')"
  pids="$(guard_pids | sort -n | xargs)"
  log_snapshot "${timestamp}" "${avail_gb}" "${pids}"
  if (( avail_gb < floor_gb )); then
    echo "${timestamp} available_gb=${avail_gb} below floor_gb=${floor_gb}; killing guarded job pids: ${pids}" >> "${log}"
    for pid in ${pids}; do
      kill "${pid}" 2>/dev/null || true
    done
    sleep "${kill_grace_seconds}"
    for pid in ${pids}; do
      kill -9 "${pid}" 2>/dev/null || true
    done
    exit 75
  fi
  sleep "${interval_seconds}"
done
