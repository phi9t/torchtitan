#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "${TORCHTITAN_IN_ROOTFS:-}" != "1" ]]; then
  exec "$ROOT/scripts/rootfs/enter_rootfs.sh" -- \
    experiments/rl_batch_invariance_async/run_reference.sh "$@"
fi

INSTALL_DEPS=0
SKIP_PREFLIGHT=0
SKIP_FULL=0
ONLY_ARM=""
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-2}"
FULL_STEPS="${FULL_STEPS:-150}"
OUT_ROOT="${OUT_ROOT:-outputs/rl_batch_invariance_async_rootfs}"
PREFLIGHT_VALIDATION_SAMPLES="${PREFLIGHT_VALIDATION_SAMPLES:-0}"
PREFLIGHT_NUM_PROMPTS="${PREFLIGHT_NUM_PROMPTS:-1}"
PREFLIGHT_NUM_SAMPLES="${PREFLIGHT_NUM_SAMPLES:-1}"
PREFLIGHT_MAX_TOKENS="${PREFLIGHT_MAX_TOKENS:-512}"
PREFLIGHT_SEQ_LEN="${PREFLIGHT_SEQ_LEN:-2048}"
PREFLIGHT_TARGET_OFFPOLICY_STEPS="${PREFLIGHT_TARGET_OFFPOLICY_STEPS:-0}"

usage() {
  cat <<'EOF'
Usage: experiments/rl_batch_invariance_async/run_reference.sh [options]

Options:
  --install-deps          Install RL runtime dependencies into the active env.
  --skip-preflight        Skip 2-step startup checks.
  --skip-full             Skip 150-step reference runs.
  --only ARM              Run only no_bi or bi.
  --preflight-steps N     Override preflight step count (default: 2).
  --full-steps N          Override full run step count (default: 150).
  --out-root PATH         Override output root.
  --preflight-validation-samples N
                          Override preflight validation samples (default: 0).
  --preflight-num-prompts N
                          Override preflight prompts per train step (default: 1).
  --preflight-num-samples N
                          Override preflight samples per prompt (default: 1).
  --preflight-max-tokens N
                          Override preflight generation cap (default: 512).
  --preflight-seq-len N   Override preflight packed sequence length (default: 2048).
  --preflight-target-offpolicy-steps N
                          Override preflight target off-policy steps (default: 0).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      shift
      ;;
    --skip-full)
      SKIP_FULL=1
      shift
      ;;
    --only)
      ONLY_ARM="${2:?--only requires an arm}"
      shift 2
      ;;
    --preflight-steps)
      PREFLIGHT_STEPS="${2:?--preflight-steps requires a value}"
      shift 2
      ;;
    --full-steps)
      FULL_STEPS="${2:?--full-steps requires a value}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:?--out-root requires a path}"
      shift 2
      ;;
    --preflight-validation-samples)
      PREFLIGHT_VALIDATION_SAMPLES="${2:?--preflight-validation-samples requires a value}"
      shift 2
      ;;
    --preflight-num-prompts)
      PREFLIGHT_NUM_PROMPTS="${2:?--preflight-num-prompts requires a value}"
      shift 2
      ;;
    --preflight-num-samples)
      PREFLIGHT_NUM_SAMPLES="${2:?--preflight-num-samples requires a value}"
      shift 2
      ;;
    --preflight-max-tokens)
      PREFLIGHT_MAX_TOKENS="${2:?--preflight-max-tokens requires a value}"
      shift 2
      ;;
    --preflight-seq-len)
      PREFLIGHT_SEQ_LEN="${2:?--preflight-seq-len requires a value}"
      shift 2
      ;;
    --preflight-target-offpolicy-steps)
      PREFLIGHT_TARGET_OFFPOLICY_STEPS="${2:?--preflight-target-offpolicy-steps requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$ONLY_ARM" && "$ONLY_ARM" != "no_bi" && "$ONLY_ARM" != "bi" ]]; then
  echo "--only must be 'no_bi' or 'bi'" >&2
  exit 2
fi

if ! command -v uv >/dev/null; then
  echo "uv is required inside the TorchTitan rootfs" >&2
  exit 1
fi

if [[ ! -x "$ROOT/.venv-rootfs/bin/python" ]]; then
  uv venv --python 3.12 --system-site-packages "$ROOT/.venv-rootfs"
fi

# shellcheck disable=SC1091
source "$ROOT/.venv-rootfs/bin/activate"
PYTHON="$ROOT/.venv-rootfs/bin/python"
if [[ "$(command -v python)" != "$PYTHON" ]]; then
  echo "Refusing to run with unexpected python on PATH: $(command -v python)" >&2
  exit 1
fi
PYTHON_EXECUTABLE="$("$PYTHON" -c 'import sys; print(sys.executable)')"
PYTHON_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)')"
case "$PYTHON_EXECUTABLE:$PYTHON_PREFIX" in
  "$ROOT/.venv-rootfs/bin/python:$ROOT/.venv-rootfs") ;;
  *)
    echo "Refusing to run outside .venv-rootfs: executable=$PYTHON_EXECUTABLE prefix=$PYTHON_PREFIX" >&2
    exit 1
    ;;
esac

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/torchinductor}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "$OUT_ROOT" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  "$PYTHON" -m pip install "setuptools<81.0.0,>=77.0.3"
  "$PYTHON" -m pip install --pre \
    "torch==2.14.0.dev20260805+cu130" \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130
  "$PYTHON" -m pip install -r requirements.txt -r requirements-dev.txt
  "$PYTHON" -m pip install -r torchtitan/experiments/rl/examples/dapo_math/requirements.txt
  "$PYTHON" -m pip install pygtrie portpicker
  "$PYTHON" -m pip install torchmonarch
  "$PYTHON" -m pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
  "$PYTHON" -m pip install "git+https://github.com/PrimeIntellect-ai/renderers.git@main"
  "$PYTHON" -m pip install --no-deps "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"
  "$PYTHON" -m pip install --no-deps \
    "vllm==1.0.0.dev20260805+cu130" \
    "torchcomms==0.3.0.dev20260805+cu130" \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --upgrade-strategy eager
  "$PYTHON" -m pip install \
    anthropic apache-tvm-ffi==0.1.11 blake3 cachetools cbor2 compressed-tensors==0.17.0 \
    depyf==0.20.0 "fastapi[standard]<0.137.0,>=0.133.0" fastsafetensors \
    flashinfer-python==0.6.15.post1 humming-kernels==0.1.10 ijson "jsonschema>=4.23.0" \
    "llguidance<1.8.0,>=1.7.0" lm-format-enforcer==0.11.3 mcp "mistral_common[image]>=1.11.6" \
    "model-hosting-container-standards<1.0.0,>=0.1.14" msgspec numba==0.65.0 \
    "nvidia-cudnn-frontend>=1.19.1" "nvidia-cutlass-dsl[cu13]==4.6.0" nvtx==0.2.15 \
    "opencv-python-headless>=4.13.0" "opentelemetry-exporter-otlp>=1.27.0" \
    "opentelemetry-sdk>=1.27.0" "opentelemetry-semantic-conventions-ai>=0.4.1" \
    outlines_core==0.2.14 partial-json-parser "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=8.0.0" py-cpuinfo pybase64 PyNvVideoCodec==2.0.4 \
    python-json-logger "quack-kernels>=0.6.1" sentencepiece setproctitle "starlette>=1.0.1" \
    tilelang==0.1.12 tokenspeed-mla==0.1.8 watchfiles "xgrammar<1.0.0,>=0.2.1" \
    lark==1.2.2 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --extra-index-url https://pypi.nvidia.com
  "$PYTHON" -m pip install "transformers>=5.5.3" --pre \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130
  "$PYTHON" -m pip install --no-deps torchvision==0.28.0
  "$PYTHON" -m pip install flash_attn_3 --extra-index-url=https://download.pytorch.org/whl/test/cu130
fi

snapshot_env() {
  local snapshot_dir="$OUT_ROOT/env"
  mkdir -p "$snapshot_dir"
  {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_status_short<<EOF"
    git status --short
    echo "EOF"
    echo "TORCHTITAN_IN_ROOTFS=${TORCHTITAN_IN_ROOTFS:-}"
    echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
    echo "python=$(command -v python)"
    "$PYTHON" --version
    echo "python_executable=$PYTHON_EXECUTABLE"
    echo "python_prefix=$PYTHON_PREFIX"
    echo "uv=$(command -v uv)"
    echo "PYTHONPATH=$PYTHONPATH"
    echo "CC=$CC"
    echo "CXX=$CXX"
    echo "HF_HOME=$HF_HOME"
    echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
    echo "VLLM_DEEP_GEMM_WARMUP=$VLLM_DEEP_GEMM_WARMUP"
    echo "VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM"
    echo "VLLM_MOE_USE_DEEP_GEMM=$VLLM_MOE_USE_DEEP_GEMM"
    echo "VLLM_USE_DEEP_GEMM_E8M0=$VLLM_USE_DEEP_GEMM_E8M0"
    echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
    echo "PREFLIGHT_STEPS=$PREFLIGHT_STEPS"
    echo "PREFLIGHT_VALIDATION_SAMPLES=$PREFLIGHT_VALIDATION_SAMPLES"
    echo "PREFLIGHT_NUM_PROMPTS=$PREFLIGHT_NUM_PROMPTS"
    echo "PREFLIGHT_NUM_SAMPLES=$PREFLIGHT_NUM_SAMPLES"
    echo "PREFLIGHT_MAX_TOKENS=$PREFLIGHT_MAX_TOKENS"
    echo "PREFLIGHT_SEQ_LEN=$PREFLIGHT_SEQ_LEN"
    echo "PREFLIGHT_TARGET_OFFPOLICY_STEPS=$PREFLIGHT_TARGET_OFFPOLICY_STEPS"
    echo "FULL_STEPS=$FULL_STEPS"
  } | tee "$snapshot_dir/environment.txt"
  nvidia-smi | tee "$snapshot_dir/nvidia-smi.txt"
  "$PYTHON" - <<'PY' | tee "$snapshot_dir/python_packages.txt"
import importlib
mods = [
    "torch",
    "vllm",
    "monarch",
    "torchstore",
    "torchcomms",
    "batch_invariant_ops",
    "flash_attn_interface",
    "flash_attn_3",
    "torchvision",
    "math_verify",
]
for name in mods:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}: ok version={version}")
    except Exception as exc:
        print(f"{name}: FAIL {type(exc).__name__}: {exc}")
print(f"cuda_available={importlib.import_module('torch').cuda.is_available()}")
if importlib.import_module('torch').cuda.is_available():
    torch = importlib.import_module("torch")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
PY
}

verify_preconditions() {
  local stale_workers
  stale_workers="$(
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
      | awk -F, '/VLLM::Worker/ {gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $3); print $1 " (" $3 " MiB)"}'
  )"
  if [[ -n "$stale_workers" ]]; then
    {
      echo "Preconditions failed: stale vLLM worker processes are already using GPUs."
      echo "$stale_workers"
      echo "Clean these PIDs before launching the reference run."
    } >&2
    exit 1
  fi

  "$PYTHON" - <<'PY'
import importlib
from pathlib import Path
import torch

required = [
    "vllm",
    "monarch",
    "torchstore",
    "torchcomms",
    "batch_invariant_ops",
    "flash_attn_interface",
    "flash_attn_3",
    "torchvision",
    "math_verify",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")

checkpoint = Path("torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Base")
if not checkpoint.exists():
    missing.append(f"missing checkpoint directory: {checkpoint}")
if not torch.cuda.is_available():
    missing.append("torch.cuda.is_available() is false")
elif torch.cuda.device_count() < 8:
    missing.append(f"need 8 visible GPUs, saw {torch.cuda.device_count()}")
if missing:
    raise SystemExit("Preconditions failed:\n" + "\n".join(missing))
from torchtitan.tools.utils import activate_cuda_flash_attention_impl

print(f"flash_attention_impl={activate_cuda_flash_attention_impl()}")
print(f"torch={torch.__version__}")
print(f"cuda_devices={torch.cuda.device_count()}")
PY
}

choose_master_port() {
  "$PYTHON" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

run_arm() {
  local arm="$1"
  local config="$2"
  local steps="$3"
  local out_dir="$4"
  shift 4
  mkdir -p "$out_dir"
  local log_file="$out_dir/run_${steps}_steps.log"
  local master_addr="127.0.0.1"
  local master_port
  master_port="$(choose_master_port)"
  echo "Running $arm ($config) for $steps steps -> $out_dir"
  echo "MASTER_ADDR=$master_addr MASTER_PORT=$master_port"
  MASTER_ADDR="$master_addr" MASTER_PORT="$master_port" \
    "$PYTHON" -m torchtitan.experiments.rl.train \
    --module dapo_math \
    --config "$config" \
    --dump-folder "$out_dir" \
    --async-loop.num-training-steps "$steps" \
    "$@" \
    2>&1 | tee "$log_file"
}

snapshot_env
verify_preconditions

arms=(no_bi bi)
if [[ -n "$ONLY_ARM" ]]; then
  arms=("$ONLY_ARM")
fi

for arm in "${arms[@]}"; do
  case "$arm" in
    no_bi) config="rl_dapo_qwen3_4b_math_8k_reference_no_bi" ;;
    bi) config="rl_dapo_qwen3_4b_math_8k_reference_bi" ;;
  esac
  if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
    run_arm "$arm-preflight" "$config" "$PREFLIGHT_STEPS" "$OUT_ROOT/preflight_${arm}" \
      --async-loop.validation.num-samples "$PREFLIGHT_VALIDATION_SAMPLES" \
      --async-loop.num-prompts-per-train-step "$PREFLIGHT_NUM_PROMPTS" \
      --async-loop.num-samples-per-prompt "$PREFLIGHT_NUM_SAMPLES" \
      --async-loop.target-offpolicy-steps "$PREFLIGHT_TARGET_OFFPOLICY_STEPS" \
      --async-loop.batcher.batch.seq-len "$PREFLIGHT_SEQ_LEN" \
      --generator.sampling.max_tokens "$PREFLIGHT_MAX_TOKENS"
  fi
  if [[ "$SKIP_FULL" -eq 0 ]]; then
    run_arm "$arm" "$config" "$FULL_STEPS" "$OUT_ROOT/$arm"
  fi
done

"$PYTHON" experiments/rl_batch_invariance_async/summarize.py --root "$OUT_ROOT"
