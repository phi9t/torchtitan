#!/usr/bin/env bash
# Compile the valid theorem and classify the controlled invalid proposition.

set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: run_lean_smoke.sh <lean>" >&2
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

lean_bin="$(resolve_runfile "$1")"
version_output="$("${lean_bin}" --version 2>&1)" || {
  echo "Lean version check failed" >&2
  exit 1
}
grep -Fq 'Lean (version 4.34.0,' <<<"${version_output}" || {
  echo "unexpected Lean toolchain: ${version_output}" >&2
  exit 1
}
printf 'LEAN_TOOLCHAIN checker=lean version=4.34.0\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/lean"
mkdir -p "${work_dir}"
# Bazel source runfiles are symlinks back to the checkout. Lean canonicalizes
# them and requires inputs to remain below its invocation root, so stage just
# these two small fixtures in the external test directory.
cp "${fixture_dir}/LeanSmokeValid.lean" "${work_dir}/LeanSmokeValid.lean"
cp "${fixture_dir}/LeanSmokeInvalid.lean" "${work_dir}/LeanSmokeInvalid.lean"

set +e
(
  cd "${work_dir}"
  timeout 60 "${lean_bin}" -o "${work_dir}/LeanSmokeValid.olean" \
    LeanSmokeValid.lean
) >"${work_dir}/valid.log" 2>&1
valid_status=$?
set -e
valid_output="$(<"${work_dir}/valid.log")"
formal_classify_lean_valid \
  "${valid_status}" "${valid_output}" "FormalSmoke.validStepReady" || {
  cat "${work_dir}/valid.log" >&2
  echo "Lean valid smoke was not a clean axiom-free kernel check" >&2
  exit 1
}
printf 'LEAN_VALID checker=lean theorem=FormalSmoke.validStepReady kind=smoke scope=toolchain axioms=[] exit=%s\n' \
  "${valid_status}"

set +e
(
  cd "${work_dir}"
  timeout 60 "${lean_bin}" LeanSmokeInvalid.lean
) >"${work_dir}/invalid.log" 2>&1
invalid_status=$?
set -e
invalid_output="$(<"${work_dir}/invalid.log")"
formal_classify_lean_negative \
  "${invalid_status}" "${invalid_output}" \
  "FormalSmoke.ControlledInvalidProposition" || {
  cat "${work_dir}/invalid.log" >&2
  echo "Lean negative smoke did not reject the named proposition" >&2
  exit 1
}
printf 'LEAN_NEGATIVE checker=lean proposition=FormalSmoke.ControlledInvalidProposition kind=smoke scope=toolchain result=rejected exit=%s\n' \
  "${invalid_status}"
