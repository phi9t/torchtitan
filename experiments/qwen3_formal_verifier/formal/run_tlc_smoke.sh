#!/usr/bin/env bash
# Run the bounded positive and controlled-negative TLC smoke models.

set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: run_tlc_smoke.sh <java> <tla2tools.jar>" >&2
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
  echo "formal fixtures not found: ${fixture_dir}" >&2
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
printf 'TLA_TOOLCHAIN checker=tlc release=1.7.4 java_major=17\n'
work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/tlc"
mkdir -p "${work_dir}/good" "${work_dir}/bad"

set +e
(
  cd "${work_dir}/good"
  timeout 60 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/good/states" \
    "${fixture_dir}/TlcSmokeGood.tla" -config "${fixture_dir}/TlcSmokeGood.cfg"
) >"${work_dir}/good.log" 2>&1
good_status=$?
set -e
good_output="$(<"${work_dir}/good.log")"
formal_classify_tlc_valid "${good_status}" "${good_output}" || {
  cat "${work_dir}/good.log" >&2
  echo "TLC valid smoke was not a clean checker success" >&2
  exit 1
}
printf 'TLA_VALID checker=tlc invariant=SmokeInvariant result=success exit=%s\n' \
  "${good_status}"

set +e
(
  cd "${work_dir}/bad"
  timeout 60 "${java_bin}" -XX:+UseParallelGC -cp "${tla_jar}" tlc2.TLC \
    -workers 1 -metadir "${work_dir}/bad/states" \
    "${fixture_dir}/TlcSmokeBad.tla" -config "${fixture_dir}/TlcSmokeBad.cfg"
) >"${work_dir}/bad.log" 2>&1
bad_status=$?
set -e
bad_output="$(<"${work_dir}/bad.log")"
formal_classify_tlc_negative \
  "${bad_status}" "${bad_output}" "SmokeInvariant" || {
  cat "${work_dir}/bad.log" >&2
  echo "TLC negative smoke did not violate exactly SmokeInvariant" >&2
  exit 1
}
printf 'TLA_NEGATIVE checker=tlc invariant=SmokeInvariant result=named_violation exit=%s\n' \
  "${bad_status}"
