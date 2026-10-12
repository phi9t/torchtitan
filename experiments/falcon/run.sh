#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/falcon/run.sh"

command="${1:-overfit}"

_hero_kv() {
  HERO_ARM=""
  HERO_SEED=0
  HERO_TAG="${HERO_TAG:-hero_20k}"
  HERO_STEPS="${HERO_STEPS:-20000}"
  HERO_INTERVAL="${HERO_INTERVAL:-500}"
  shift || true
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --arm) HERO_ARM="$2"; shift 2 ;;
      --seed) HERO_SEED="$2"; shift 2 ;;
      --tag) HERO_TAG="$2"; shift 2 ;;
      --steps) HERO_STEPS="$2"; shift 2 ;;
      --checkpoint-interval) HERO_INTERVAL="$2"; shift 2 ;;
      --master-port) shift 2 ;;
      --dry-run|--fresh|--pause) shift ;;
      *) shift ;;
    esac
  done
}

_hero_dump() {
  local tag="$1" arm="$2" seed="$3"
  printf '%s\n' "${REPO_ROOT}/experiments/falcon/results/hero/${tag}/${arm}_seed${seed}"
}

if [[ "${command}" == "pause" ]]; then
  _hero_kv "$@"
  if [[ -z "${HERO_ARM}" ]]; then
    echo "usage: experiments/falcon/run.sh pause --arm A0|A1|A5 [--seed 0] [--tag hero_20k]" >&2
    exit 2
  fi
  dump="$(_hero_dump "${HERO_TAG}" "${HERO_ARM}" "${HERO_SEED}")"
  mkdir -p "${dump}"
  touch "${dump}/PAUSE"
  echo "pause requested: ${dump}/PAUSE"
  exit 0
fi

if [[ "${command}" == "hero-status" ]]; then
  _hero_kv "$@"
  for arm in A5 A0 A1; do
    dump="$(_hero_dump "${HERO_TAG}" "${arm}" 0)"
    echo "=== ${arm} ${dump} ==="
    if [[ -f "${dump}/PAUSE" ]]; then
      echo "PAUSE file present"
    fi
    if [[ -f "${dump}/hero.pid" ]]; then
      echo "pid=$(cat "${dump}/hero.pid")"
    fi
    if [[ -f "${dump}/hero.host.pid" ]]; then
      echo "host_pid=$(cat "${dump}/hero.host.pid")"
    fi
    if [[ -f "${dump}/hero_status.json" ]]; then
      cat "${dump}/hero_status.json"
    fi
    if [[ -f "${dump}/hero_outcome.json" ]]; then
      cat "${dump}/hero_outcome.json"
    fi
    if [[ -d "${dump}/checkpoint" ]]; then
      ls -1 "${dump}/checkpoint" || true
    fi
  done
  exit 0
fi

if [[ "${command}" == "hero-all" ]]; then
  if [[ "${TORCHTITAN_IN_ROOTFS:-0}" == "1" ]]; then
    echo "hero-all must run on the host so each arm gets its own rootfs" >&2
    exit 2
  fi
  _hero_kv "$@"
  tag="${HERO_TAG}"
  steps="${HERO_STEPS}"
  interval="${HERO_INTERVAL}"
  arms=(A5 A0 A1)
  gpus=(0 1 2)
  ports=(29580 29581 29582)
  for i in 0 1 2; do
    arm="${arms[$i]}"
    gpu="${gpus[$i]}"
    port="${ports[$i]}"
    dump="$(_hero_dump "${tag}" "${arm}" 0)"
    mkdir -p "${dump}"
    rm -f "${dump}/PAUSE"
    nohup "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" --share-pid -- bash -lc \
      "cd /workspace/torchtitan && export CUDA_VISIBLE_DEVICES=${gpu} && \
       python -m experiments.falcon.hero_runner --arm ${arm} --seed 0 \
       --steps ${steps} --checkpoint-interval ${interval} --tag ${tag} \
       --master-port ${port}" \
      > "${dump}/hero.log" 2>&1 &
    echo $! > "${dump}/hero.host.pid"
    echo "launched ${arm} gpu=${gpu} host_pid=$! log=${dump}/hero.log"
  done
  echo "pause: experiments/falcon/run.sh pause --arm A5 --tag ${tag}"
  echo "resume: re-run the same hero or hero-all command (clears PAUSE, loads latest ckpt)"
  echo "status: experiments/falcon/run.sh hero-status --tag ${tag}"
  exit 0
fi

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- \
    /bin/bash -lc \
    'cd /workspace/torchtitan && exec experiments/falcon/run.sh "$@"' \
    "${SCRIPT_REL}" "$@"
fi

cd "${REPO_ROOT}"

if [[ "${command}" == "overfit" ]]; then
  shift || true
  exec python -m experiments.falcon.overfit "$@"
fi

if [[ "${command}" == "train" ]]; then
  shift || true
  export MODULE="${MODULE:-falcon}"
  export CONFIG="${CONFIG:-falcon_tiny_overfit}"
  export NGPU="${NGPU:-1}"
  export COMM_MODE="${COMM_MODE:-fake_backend}"
  exec ./run_train.sh "$@"
fi

if [[ "${command}" == "hero" ]]; then
  shift || true
  exec python -m experiments.falcon.hero_runner "$@"
fi

if [[ "${command}" == "addition" ]]; then
  shift || true
  exec python -m experiments.falcon.addition_runner "$@"
fi

if [[ "${command}" == "addition-preflight" ]]; then
  shift || true
  exec python -m experiments.falcon.campaign_driver addition-preflight "$@"
fi

if [[ "${command}" == "mechanism" ]]; then
  shift || true
  exec python -m experiments.falcon.mechanism_runner "$@"
fi

if [[ "${command}" == "mechanism-preflight" ]]; then
  shift || true
  exec python -m experiments.falcon.campaign_driver mechanism-screen-preflight "$@"
fi

if [[ "${command}" == "checkpoint-eval" ]]; then
  shift || true
  exec python -m experiments.falcon.checkpoint_eval "$@"
fi

if [[ "${command}" == "perf" ]]; then
  shift || true
  exec python -m experiments.falcon.perf_base "$@"
fi

if [[ "${command}" == "nsys" ]]; then
  shift || true
  nsys_bin="$(command -v nsys || true)"
  if [[ -z "${nsys_bin}" ]]; then
    echo "nsys is not on PATH inside the rootfs; bind /usr/local/cuda into enter_rootfs.sh" >&2
    exit 2
  fi
  out="${NSYS_OUTPUT:-experiments/falcon/results/tooling/nsys_$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname -- "$out")"
  export TMPDIR="${TMPDIR:-/project/tmp}"
  exec "${nsys_bin}" profile \
    --output "$out" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --sample cpu \
    -- python -m experiments.falcon.perf_base "$@"
fi

echo "unknown Falcon command: ${command}" >&2
echo "usage: experiments/falcon/run.sh [overfit|train|hero|hero-all|pause|hero-status|addition|addition-preflight|mechanism|mechanism-preflight|checkpoint-eval|perf|nsys]" >&2
exit 2
