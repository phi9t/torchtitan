#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source_dir="experiments/countdown_search_distill/sweeps/formatting_replication/formatting_seed43_train2000_rank16/data"
cache_root="${GLM_LADDER_CACHE:-/data05/philip.yang/glm53-ladder-cache}"
tmp_dir="${cache_root}/dataset-v1-tmp/countdown-sft-v1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "${tmp_dir}/train" "${tmp_dir}/dev" "${tmp_dir}/iid_test" "${tmp_dir}/ood_test" "${tmp_dir}/calibration"

required_files=(
  "train/formatting.jsonl"
  "train/problems.jsonl"
  "dev/problems.jsonl"
  "iid_test/problems.jsonl"
  "ood_test/problems.jsonl"
  "calibration/problems.jsonl"
  "split_registry.json"
)

for rel in "${required_files[@]}"; do
  if [[ ! -f "${source_dir}/${rel}" ]]; then
    printf 'required source file missing: %s\n' "${source_dir}/${rel}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${tmp_dir}/${rel}")"
  cp "${source_dir}/${rel}" "${tmp_dir}/${rel}"
done

readme="${tmp_dir}/README.md"
{
  printf '# countdown-sft/v1\n\n'
  printf 'Published for the GLM-5.3-Flash SFT ladder storage smoke.\n\n'
  printf '%s\n' '- claim_label: smoke'
  printf "%s\n" "- source_dir: \`${source_dir}\`"
  printf "%s\n\n" "- built_at_utc: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
  printf '## Files\n\n'
  printf '| file | rows | bytes | md5 |\n'
  printf '|---|---:|---:|---|\n'
  for rel in "${required_files[@]}"; do
    rows="$(wc -l < "${tmp_dir}/${rel}" | tr -d '[:space:]')"
    bytes="$(wc -c < "${tmp_dir}/${rel}" | tr -d '[:space:]')"
    md5="$(md5sum "${tmp_dir}/${rel}" | awk '{print $1}')"
    printf "| \`%s\` | %s | %s | \`%s\` |\n" "$rel" "$rows" "$bytes" "$md5"
  done
} > "$readme"

"${script_dir}/publish.sh" "$tmp_dir" "torchtitan/datasets/countdown-sft/v1"
