#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_REL="experiments/mini_kimi_k3/run.sh"

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  exec "${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh" -- \
    /bin/bash -lc \
    'cd /workspace/torchtitan && exec experiments/mini_kimi_k3/run.sh "$@"' \
    "${SCRIPT_REL}" "$@"
fi

cd "${REPO_ROOT}"

if [[ "${1:-}" == "preflight" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.preflight "$@"
fi

if [[ "${1:-}" == "init-evidence" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.init_evidence "$@"
fi

if [[ "${1:-}" == "init-manifest" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.init_manifest "$@"
fi

if [[ "${1:-}" == "register-shards" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.register_shards "$@"
fi

if [[ "${1:-}" == "import-shards" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.import_shards "$@"
fi

if [[ "${1:-}" == "build-local-shards" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.build_local_shards "$@"
fi

if [[ "${1:-}" == "prepare-stage1-local" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.prepare_stage1_local "$@"
fi

if [[ "${1:-}" == "materialize-stage1-local" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.materialize_stage1_local "$@"
fi

if [[ "${1:-}" == "write-stage1-input-manifest" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.write_stage1_input_manifest "$@"
fi

if [[ "${1:-}" == "plan-r1-corpus" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.plan_r1_corpus "$@"
fi

if [[ "${1:-}" == "probe-stage1-sources" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.probe_stage1_sources "$@"
fi

if [[ "${1:-}" == "stage1-remote-readiness" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.stage1_remote_readiness "$@"
fi

if [[ "${1:-}" == "run-stage1-remote" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.run_stage1_remote "$@"
fi

if [[ "${1:-}" == "completion-audit" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.completion_audit "$@"
fi

if [[ "${1:-}" == "audit-r1-corpus-plan" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.audit_r1_corpus_plan "$@"
fi

if [[ "${1:-}" == "corpus-disk-readiness" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.corpus_disk_readiness "$@"
fi

if [[ "${1:-}" == "inspect-stage1-inputs" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.inspect_stage1_inputs "$@"
fi

if [[ "${1:-}" == "decontaminate-local" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.decontaminate_local "$@"
fi

if [[ "${1:-}" == "build-decontamination-index" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.build_decontamination_index "$@"
fi

if [[ "${1:-}" == "install-tokenizer-assets" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.install_tokenizer_assets "$@"
fi

if [[ "${1:-}" == "compare-forward-oracle" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.compare_forward_oracle "$@"
fi

if [[ "${1:-}" == "trace-forward-oracle" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.trace_forward_oracle "$@"
fi

if [[ "${1:-}" == "review-launch-backend" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.review_launch_backend "$@"
fi

if [[ "${1:-}" == "dump-torchtitan-logits" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.dump_torchtitan_logits "$@"
fi

if [[ "${1:-}" == "dump-first-party-logits" ]]; then
  shift
  exec "${MINI_KIMI_K3_ORACLE_PYTHON:-python}" -m experiments.mini_kimi_k3.dump_first_party_logits "$@"
fi

if [[ "${1:-}" == "tiny-smoke" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.training_smoke "$@"
fi

if [[ "${1:-}" == "r1-training-smoke" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.r1_training_smoke "$@"
fi

if [[ "${1:-}" == "r1-smoke-gpu-readiness" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.r1_smoke_gpu_readiness "$@"
fi

if [[ "${1:-}" == "launch" ]]; then
  shift
  exec python -m experiments.mini_kimi_k3.launch "$@"
fi

cat >&2 <<'EOF'
Mini Kimi K3 runner requires an explicit command.

Use preflight before launch. The full r1 launch remains gated on model
fidelity, corpus, and run-attempt evidence, and this fallback path does not
start training or evaluation.
EOF

exit 21
