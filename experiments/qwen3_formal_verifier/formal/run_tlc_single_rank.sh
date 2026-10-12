#!/usr/bin/env bash
# Check the valid observed Single-rank trace and one controlled lifecycle mutation.

set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: run_tlc_single_rank.sh <java> <tla2tools.jar>" >&2
  exit 2
}

resolve_runfile() {
  local requested="$1"
  local candidate
  for candidate in \
    "${requested}" \
    "${TEST_SRCDIR:-}/${requested}" \
    "${TEST_SRCDIR:-}/${requested#external/}"; do
    if [[ -f "${candidate}" ]]; then
      readlink -f -- "${candidate}"
      return 0
    fi
  done
  echo "runfile not found: ${requested}" >&2
  return 1
}

workspace="${TEST_WORKSPACE:-_main}"
fixture_dir="${TEST_SRCDIR}/${workspace}/experiments/qwen3_formal_verifier/formal"
[[ -d "${fixture_dir}" ]] || {
  echo "Single-rank formal fixtures not found: ${fixture_dir}" >&2
  exit 2
}
# shellcheck source=experiments/qwen3_formal_verifier/formal/checker_contract.sh
source "${fixture_dir}/checker_contract.sh"

java_bin="$(resolve_runfile "$1")"
tla_jar="$(resolve_runfile "$2")"
java_version_output="$("${java_bin}" -version 2>&1)" || {
  echo "Java runtime version check failed" >&2
  exit 1
}
grep -Eq ' version "17([."]|$)' <<<"${java_version_output}" || {
  echo "unexpected Java runtime: ${java_version_output}" >&2
  exit 1
}
printf 'SINGLE_RANK_TLA_TOOLCHAIN checker=tlc release=1.7.4 java_major=17\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/tlc-single-rank"
mkdir -p "${work_dir}/valid" "${work_dir}/invalid"
for name in TraceLifecycle SingleRankFacts SingleRankValid; do
  cp "${fixture_dir}/${name}.tla" "${work_dir}/valid/${name}.tla"
done
cp "${fixture_dir}/SingleRankValid.cfg" "${work_dir}/valid/SingleRankValid.cfg"
for name in TraceLifecycle SingleRankBadFacts SingleRankInvalid; do
  cp "${fixture_dir}/${name}.tla" "${work_dir}/invalid/${name}.tla"
done
cp "${fixture_dir}/SingleRankInvalid.cfg" "${work_dir}/invalid/SingleRankInvalid.cfg"

set +e
(
  cd "${work_dir}/valid"
  timeout 60 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/valid/states" \
    SingleRankValid.tla -config SingleRankValid.cfg
) >"${work_dir}/valid.log" 2>&1
valid_status=$?
set -e
valid_output="$(<"${work_dir}/valid.log")"
formal_classify_tlc_valid "${valid_status}" "${valid_output}" || {
  cat "${work_dir}/valid.log" >&2
  echo "TLC Single-rank valid trace was not a clean checker success" >&2
  exit 1
}
printf 'SINGLE_RANK_TLA_VALID invariant_set=lifecycle result=success exit=%s\n' \
  "${valid_status}"

set +e
(
  cd "${work_dir}/invalid"
  timeout 60 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/invalid/states" \
    SingleRankInvalid.tla -config SingleRankInvalid.cfg
) >"${work_dir}/invalid.log" 2>&1
invalid_status=$?
set -e
invalid_output="$(<"${work_dir}/invalid.log")"
formal_classify_tlc_transition_negative \
  "${invalid_status}" "${invalid_output}" \
  "SingleRankGradientReadyBeforeOptimizer" || {
  cat "${work_dir}/invalid.log" >&2
  echo "TLC Single-rank negative did not violate exactly the gradient invariant" >&2
  exit 1
}
printf 'SINGLE_RANK_TLA_NEGATIVE invariant=SingleRankGradientReadyBeforeOptimizer result=named_violation exit=%s\n' \
  "${invalid_status}"
