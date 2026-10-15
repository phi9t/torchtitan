#!/usr/bin/env bash
# Give the GLM rung-4 ticket (T${TICKET}) GPU priority. While its runner is waiting at a GPU gate
# (latest gpu_wait log line says decision=wait_* and is fresh), SIGSTOP the small-GPU traecli
# workers (speedrun/ladder) so they stop launching new GPU jobs; their in-flight jobs finish.
# While GLM ranks are up or T${TICKET} is not gating, SIGCONT them. Host orchestration only; only
# touches this user's traecli worker processes. Exits when the T${TICKET} worker exits.
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PAT="${PRIORITY_PAUSE_PATTERN:-^traecli exec.*(glm53-arch-speedrun/sdd/s(06|07|07b|09|1[0-9])-brief|glm53-flash-sft-ladder/sdd/task-(1[3-8]|k0[0-9])-brief)}"
TICKET="${PRIORITY_TICKET:-08d}"
RESULTS_GLOB="${PRIORITY_RESULTS_GLOB:-t${TICKET}-*}"
log() { echo "$(date -u +%FT%TZ) $*"; }
state=running
while ps -eo args= | grep -q "^traecli exec.*task-${TICKET}-brief"; do
  latest="$(ls -t "${REPO}"/experiments/glm53_flash_sft_ladder/results/${RESULTS_GLOB}/attempt-*/logs/gpu_wait*.log 2>/dev/null | head -1)"
  gating=0
  if [ -n "${latest}" ] && [ $(( $(date +%s) - $(stat -c %Y "${latest}") )) -lt 180 ] \
     && tail -n 5 "${latest}" | grep -q "decision=wait"; then gating=1; fi
  pids="$(ps -u "$(id -u)" -o pid=,args= | grep -E " ${PAT#^}" | awk '{print $1}')"
  if (( gating )) && [ "${state}" = running ] && [ -n "${pids}" ]; then
    kill -STOP ${pids} && log "T${TICKET} gating -> paused workers: ${pids}"; state=paused
  elif (( ! gating )) && [ "${state}" = paused ]; then
    kill -CONT ${pids} 2>/dev/null; log "T${TICKET} not gating -> resumed workers: ${pids}"; state=running
  fi
  sleep 30
done
pids="$(ps -u "$(id -u)" -o pid=,args= | grep -E " ${PAT#^}" | awk '{print $1}')"
[ -n "${pids}" ] && kill -CONT ${pids} 2>/dev/null; log "T${TICKET} worker gone -> resumed all, exiting"
