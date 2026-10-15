#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Rebuild variant/ from pinned upstream modded-nanogpt plus variant.patch.
# Upstream source is fetched (not committed) by experiments/modded_nanogpt_b200.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream="${here}/../modded_nanogpt_b200/sources/modded-nanogpt"
pin="ecbb586296d3dac36fd206211f25d63bad4a6b35"
out="${1:-${here}/variant}"
actual="$(git -C "$upstream" rev-parse HEAD)"
if [[ "$actual" != "$pin" ]]; then
  echo "upstream at $actual, expected $pin" >&2
  exit 1
fi
mkdir -p "$out/evals"
for f in train_gpt.py triton_kernels.py dc_triton_kernels.py evals/hellaswag.py; do
  cp "$upstream/$f" "$out/$f"
done
patch --quiet -p1 -d "$out" < "${here}/variant.patch"
echo "materialized $out"
