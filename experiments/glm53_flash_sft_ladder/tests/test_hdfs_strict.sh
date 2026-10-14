#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
checker="${repo_root}/experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh"

tmp_root="$(mktemp -d)"
trap 'rm -rf "${tmp_root}"' EXIT

md5_file() {
  md5sum "$1" | awk '{print $1}'
}

write_fixture() {
  local dir="$1"
  mkdir -p "${dir}/nested"
  printf 'alpha\n' > "${dir}/alpha.txt"
  printf 'beta\n' > "${dir}/nested/beta.txt"
  : > "${dir}/MANIFEST.jsonl"
  for rel in alpha.txt nested/beta.txt; do
    jq -nc \
      --arg file "$rel" \
      --argjson size "$(stat -c '%s' "${dir}/${rel}")" \
      --arg md5 "$(md5_file "${dir}/${rel}")" \
      '{file:$file,size:$size,md5:$md5}' >> "${dir}/MANIFEST.jsonl"
  done
  jq -n \
    --arg source_uri "hdfs://example/user/tiger/waystone/torchtitan/test-fixture" \
    --arg manifest_md5 "$(md5_file "${dir}/MANIFEST.jsonl")" \
    --arg staged_at "2026-10-14T00:00:00Z" \
    --argjson file_count 2 \
    '{source_uri:$source_uri,manifest_md5:$manifest_md5,staged_at:$staged_at,file_count:$file_count}' \
    > "${dir}/.staged.json"
}

expect_fail() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>"${tmp_root}/${label}.err"; then
    printf 'expected failure but command passed: %s\n' "$label" >&2
    exit 1
  fi
}

fixture="${tmp_root}/fixture"
write_fixture "$fixture"

"$checker" "$fixture" > "${tmp_root}/full.out"
grep -q 'MODE=full' "${tmp_root}/full.out"

"$checker" --quick "$fixture" > "${tmp_root}/quick.out"
grep -q 'MODE=quick' "${tmp_root}/quick.out"
jq -e '.files["alpha.txt"].md5 and .files["nested/beta.txt"].md5' \
  "${fixture}/.verified.json" >/dev/null

rm -f "${fixture}/.staged.json"
expect_fail missing_marker "$checker" "$fixture"
write_fixture "$fixture"

printf 'BETAX\n' > "${fixture}/nested/beta.txt"
expect_fail md5_mismatch "$checker" "$fixture"

write_fixture "$fixture"
printf 'too long\n' > "${fixture}/nested/beta.txt"
expect_fail size_mismatch "$checker" --quick "$fixture"

printf 'hdfs_strict tests passed\n'
