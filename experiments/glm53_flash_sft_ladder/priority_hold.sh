#!/usr/bin/env bash
# Hold the shared full-size GLM lock on behalf of the rung-4 ticket (T08d) so that
# 4-GPU side tickets (T09*) queue behind it. Releases when the T08d report exists or
# the T08d worker exits. Host orchestration only.
set -uo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK=/data05/philip.yang/glm53-ladder-cache/glm-fullsize.lock
REPORT="${REPO}/.scratch/glm53-flash-sft-ladder/sdd/task-08d-report.md"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for lock"
exec 9>>"${LOCK}"
flock 9
log "lock acquired for T08d"
while [ ! -f "${REPORT}" ] && ps -eo args= | grep -q "^traecli exec.*task-08d-brief"; do sleep 60; done
log "T08d finished (report or worker exit); releasing lock"
