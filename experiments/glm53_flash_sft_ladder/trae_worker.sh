#!/usr/bin/env bash
# Launch a traecli worker on a campaign ticket brief.
# Usage: trae_worker.sh <ticket-id> <brief-path>   (logs to experiments/glm53_flash_sft_ladder/logs/<ticket-id>.log)
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ticket="$1"; brief="$2"
mkdir -p "${REPO}/experiments/glm53_flash_sft_ladder/logs"
exec traecli exec -m gpt-5.5 \
  -c model_reasoning_effort=xhigh -c model_reasoning_summary=detailed \
  --yolo -C "${REPO}" \
  -o "${REPO}/experiments/glm53_flash_sft_ladder/logs/${ticket}.last.md" \
  "Execute the ticket brief at ${brief} exactly. Work autonomously until verification and the report are done. Do not commit." \
  < /dev/null > "${REPO}/experiments/glm53_flash_sft_ladder/logs/${ticket}.log" 2>&1
