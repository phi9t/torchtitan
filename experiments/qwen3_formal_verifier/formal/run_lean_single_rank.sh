#!/usr/bin/env bash
# Kernel-check Single-rank lifecycle facts and reject one controlled false fact.
#
# Every result here is an EVALUATION of the observed trace: Bool-valued
# predicates over literal SingleRankFacts data, closed by `rfl` or `decide`. The
# tokens carry kind=evaluation scope=observed-trace so the sealed log
# distinguishes them from the quantified protocol theorems emitted by
# run_lean_device_mesh_protocol.sh.

set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: run_lean_single_rank.sh <lean>" >&2
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

lean_bin="$(resolve_runfile "$1")"
version_output="$("${lean_bin}" --version 2>&1)" || {
  echo "Lean version check failed" >&2
  exit 1
}
grep -Fq 'Lean (version 4.34.0,' <<<"${version_output}" || {
  echo "unexpected Lean toolchain: ${version_output}" >&2
  exit 1
}
printf 'SINGLE_RANK_LEAN_TOOLCHAIN checker=lean version=4.34.0\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/lean-single-rank"
mkdir -p "${work_dir}/valid" "${work_dir}/invalid"
for name in TraceLifecycle SingleRankFacts SingleRankValid; do
  cp "${fixture_dir}/${name}.lean" "${work_dir}/valid/${name}.lean"
done
for name in TraceLifecycle SingleRankBadFacts SingleRankInvalid; do
  cp "${fixture_dir}/${name}.lean" "${work_dir}/invalid/${name}.lean"
done

set +e
(
  cd "${work_dir}/valid"
  LEAN_PATH=. timeout 60 "${lean_bin}" -o TraceLifecycle.olean TraceLifecycle.lean &&
    LEAN_PATH=. timeout 60 "${lean_bin}" -o SingleRankFacts.olean SingleRankFacts.lean &&
    LEAN_PATH=. timeout 60 "${lean_bin}" -o SingleRankValid.olean SingleRankValid.lean
) >"${work_dir}/valid.log" 2>&1
valid_status=$?
set -e
valid_output="$(<"${work_dir}/valid.log")"
formal_classify_lean_valid \
  "${valid_status}" "${valid_output}" \
  "Qwen3Formal.SingleRank.validLifecycle" || {
  cat "${work_dir}/valid.log" >&2
  echo "Lean Single-rank valid theorem was not a clean axiom-free kernel check" >&2
  exit 1
}
printf 'SINGLE_RANK_LEAN_VALID theorem=Qwen3Formal.SingleRank.validLifecycle kind=evaluation scope=observed-trace axioms=[] exit=%s\n' \
  "${valid_status}"

set +e
(
  cd "${work_dir}/invalid"
  LEAN_PATH=. timeout 60 "${lean_bin}" -o TraceLifecycle.olean TraceLifecycle.lean &&
    LEAN_PATH=. timeout 60 "${lean_bin}" -o SingleRankBadFacts.olean SingleRankBadFacts.lean &&
    LEAN_PATH=. timeout 60 "${lean_bin}" SingleRankInvalid.lean
) >"${work_dir}/invalid.log" 2>&1
invalid_status=$?
set -e
invalid_output="$(<"${work_dir}/invalid.log")"
formal_classify_lean_negative \
  "${invalid_status}" "${invalid_output}" \
  "Qwen3Formal.SingleRank.ControlledInvalidProposition" || {
  cat "${work_dir}/invalid.log" >&2
  echo "Lean Single-rank negative did not reject the named proposition" >&2
  exit 1
}
printf 'SINGLE_RANK_LEAN_NEGATIVE proposition=Qwen3Formal.SingleRank.ControlledInvalidProposition kind=evaluation scope=observed-trace result=rejected exit=%s\n' \
  "${invalid_status}"
