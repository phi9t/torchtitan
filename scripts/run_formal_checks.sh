#!/usr/bin/env bash
# Run the opt-in TLC and Lean smoke checks through Bazel inside Insula.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_REL="scripts/run_formal_checks.sh"
ROOTFS_ENTRYPOINT="${TORCHTITAN_ROOTFS_ENTRYPOINT:-${REPO_ROOT}/scripts/rootfs/enter_rootfs.sh}"
BAZEL_VERSION="9.2.0"
BAZEL_SHA256="7668a95db1250f12c40407251e4e203b4ec8bf39bc495d2f485b2d8c99048694"
BAZEL_URL="https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

resolve_formal_cache() {
  local requested="$1"
  [[ -n "${requested}" ]] || die "formal cache path is empty"
  [[ -d "${requested}" ]] \
    || die "formal cache directory does not exist: ${requested}"
  [[ ! -L "${requested}" ]] \
    || die "formal cache directory must not be a symlink: ${requested}"

  local canonical repo_canonical managed_path candidate managed_canonical
  canonical="$(cd -P -- "${requested}" && pwd)" \
    || die "cannot resolve formal cache directory: ${requested}"
  repo_canonical="$(cd -P -- "${REPO_ROOT}" && pwd)"
  case "${canonical}/" in
    "${repo_canonical}/"*) die "formal cache must be outside the checkout: ${canonical}" ;;
  esac
  [[ -w "${canonical}" ]] \
    || die "formal cache directory is not writable: ${canonical}"

  for managed_path in bazel tools locks; do
    candidate="${canonical}/${managed_path}"
    [[ ! -L "${candidate}" ]] \
      || die "formal cache managed directory must not be a symlink: ${candidate}"
    if [[ -e "${candidate}" ]]; then
      [[ -d "${candidate}" ]] \
        || die "formal cache managed path is not a directory: ${candidate}"
      managed_canonical="$(cd -P -- "${candidate}" && pwd)" \
        || die "cannot resolve formal cache managed directory: ${candidate}"
      case "${managed_canonical}/" in
        "${canonical}/"*) ;;
        *) die "formal cache managed directory escapes cache root: ${candidate}" ;;
      esac
    fi
  done
  printf '%s\n' "${canonical}"
}

ensure_cache_dir() {
  local relative="$1"
  local current="${formal_cache}"
  local component candidate canonical
  local -a components
  IFS='/' read -r -a components <<<"${relative}"

  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != "." && "${component}" != ".." ]] \
      || die "invalid managed formal cache path: ${relative}"
    candidate="${current}/${component}"
    [[ ! -L "${candidate}" ]] \
      || die "formal cache managed directory must not be a symlink: ${candidate}"
    if [[ -e "${candidate}" ]]; then
      [[ -d "${candidate}" ]] \
        || die "formal cache managed path is not a directory: ${candidate}"
    else
      mkdir -- "${candidate}" \
        || die "cannot create formal cache managed directory: ${candidate}"
    fi
    [[ ! -L "${candidate}" ]] \
      || die "formal cache managed directory became a symlink: ${candidate}"
    canonical="$(cd -P -- "${candidate}" && pwd)" \
      || die "cannot resolve formal cache managed directory: ${candidate}"
    case "${canonical}/" in
      "${formal_cache}/"*) ;;
      *) die "formal cache managed directory escapes cache root: ${candidate}" ;;
    esac
    current="${canonical}"
  done
  printf '%s\n' "${current}"
}

usage() {
  cat <<'EOF'
Usage: scripts/run_formal_checks.sh --networked|--no-fetch [--suite smoke|scout-a|scout-b]

  --networked  Materialize integrity-pinned formal dependencies, then test.
  --no-fetch   Disable network in Insula and Bazel; reuse materialized caches.
  --suite      Select the smoke, Scout A, or Scout B regression suite.

Environment:
  TORCHTITAN_FORMAL_CACHE_HOST  Persistent host cache outside the checkout.
  TORCHTITAN_FORMAL_ROOTFS      Optional explicit Insula rootfs directory.
EOF
}

[[ $# -eq 1 || $# -eq 3 ]] || {
  usage >&2
  exit 2
}
mode_arg="$1"
suite="smoke"
reentry_args=("${mode_arg}")
if [[ $# -eq 3 ]]; then
  if [[ "$2" != "--suite" ]] \
    || [[ "$3" != "smoke" && "$3" != "scout-a" && "$3" != "scout-b" ]]; then
    usage >&2
    exit 2
  fi
  suite="$3"
  reentry_args+=(--suite "${suite}")
fi
case "${mode_arg}" in
  --networked)
    mode="networked"
    rootfs_network="networked"
    ;;
  --no-fetch)
    mode="no-fetch"
    rootfs_network="offline"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "${TORCHTITAN_IN_ROOTFS:-0}" != "1" ]]; then
  host_cache="${TORCHTITAN_FORMAL_CACHE_HOST:-${HOME:?HOME is required}/.cache/torchtitan/formal}"
  mkdir -p -- "${host_cache}"
  host_cache="$(cd -P -- "${host_cache}" && pwd)"
  case "${host_cache}/" in
    "${REPO_ROOT}/"*) die "formal cache must be outside the checkout: ${host_cache}" ;;
  esac

  entrypoint_args=()
  if [[ -n "${TORCHTITAN_FORMAL_ROOTFS:-}" ]]; then
    entrypoint_args+=(--rootfs "${TORCHTITAN_FORMAL_ROOTFS}")
  fi
  export TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST="${host_cache}"
  export TORCHTITAN_FORMAL_REENTRY_NETWORK="${rootfs_network}"
  export TORCHTITAN_ROOTFS_NETWORK="${rootfs_network}"
  exec "${ROOTFS_ENTRYPOINT}" "${entrypoint_args[@]}" -- \
    "${SCRIPT_REL}" "${reentry_args[@]}"
fi

[[ "${TORCHTITAN_IN_ROOTFS}" == "1" ]] \
  || die "${SCRIPT_REL} must run with TORCHTITAN_IN_ROOTFS=1"
formal_cache_requested="${TORCHTITAN_FORMAL_CACHE:-}"
[[ -n "${formal_cache_requested}" ]] \
  || die "formal cache contract is missing; re-enter through ${SCRIPT_REL}"
formal_cache="$(resolve_formal_cache "${formal_cache_requested}")"
[[ "${TORCHTITAN_FORMAL_REENTRY_NETWORK:-}" == "${rootfs_network}" ]] \
  || die "${mode} must be launched from outside Insula to establish a fresh ${rootfs_network} sandbox"
[[ "${TORCHTITAN_ROOTFS_NETWORK:-}" == "${rootfs_network}" ]] \
  || die "${mode} requires Insula network mode ${rootfs_network}"

cd "${REPO_ROOT}"
declared_version="$(tr -d '[:space:]' < .bazelversion)"
[[ "${declared_version}" == "${BAZEL_VERSION}" ]] \
  || die ".bazelversion must declare ${BAZEL_VERSION}, found ${declared_version}"

bazel_dir="$(ensure_cache_dir "tools/bazel/${BAZEL_VERSION}")"
bazel_bin="${bazel_dir}/bazel"
locks_dir="$(ensure_cache_dir locks)"
bazel_lock="${locks_dir}/bazel-${BAZEL_VERSION}.lock"
[[ ! -L "${bazel_lock}" ]] \
  || die "formal cache lock must not be a symlink: ${bazel_lock}"
[[ ! -e "${bazel_lock}" || -f "${bazel_lock}" ]] \
  || die "formal cache lock is not a regular file: ${bazel_lock}"
exec 9>>"${bazel_lock}"
flock 9

[[ ! -L "${bazel_bin}" ]] \
  || die "cached Bazel must not be a symlink: ${bazel_bin}"
[[ ! -e "${bazel_bin}" || -f "${bazel_bin}" ]] \
  || die "cached Bazel is not a regular file: ${bazel_bin}"

if [[ ! -x "${bazel_bin}" ]]; then
  [[ "${mode}" == "networked" ]] \
    || die "pinned Bazel ${BAZEL_VERSION} is not cached for --no-fetch"
  command -v curl >/dev/null 2>&1 || die "curl is required to materialize Bazel"
  temp_bin="$(mktemp "${bazel_dir}/.bazel.download.XXXXXX")"
  trap 'rm -f -- "${temp_bin:-}"' EXIT
  curl --fail --location --retry 3 --output "${temp_bin}" "${BAZEL_URL}"
  printf '%s  %s\n' "${BAZEL_SHA256}" "${temp_bin}" | sha256sum --check --status \
    || die "Bazel ${BAZEL_VERSION} integrity verification failed"
  chmod 0755 "${temp_bin}"
  mv "${temp_bin}" "${bazel_bin}"
  trap - EXIT
fi
printf '%s  %s\n' "${BAZEL_SHA256}" "${bazel_bin}" | sha256sum --check --status \
  || die "cached Bazel ${BAZEL_VERSION} failed integrity verification"

output_root="$(ensure_cache_dir bazel/output-user-root)"
repository_cache="$(ensure_cache_dir bazel/repository-cache)"
symlink_prefix="$(ensure_cache_dir bazel/workspace-links)/"
vendor_dir="$(ensure_cache_dir bazel/vendor)"
local_registry="file://${vendor_dir}/_registries/bcr.bazel.build"
vendor_registry="${vendor_dir}/_registries/bcr.bazel.build/bazel_registry.json"
formal_target="//experiments/qwen3_formal_verifier/formal:formal_smoke_tests"
if [[ "${suite}" == "scout-a" ]]; then
  formal_target="//experiments/qwen3_formal_verifier/formal:scout_a_formal_tests"
elif [[ "${suite}" == "scout-b" ]]; then
  formal_target="//experiments/qwen3_formal_verifier/formal:scout_b_formal_tests"
fi
common_flags=(
  --repository_cache="${repository_cache}"
  --symlink_prefix="${symlink_prefix}"
  --config=formal
  --vendor_dir="${vendor_dir}"
)
test_flags=(
  "${common_flags[@]}"
  --registry="${local_registry}"
  --repository_disable_download
  --cache_test_results=no
  --test_output=all
)

printf 'formal rootfs sentinel: TORCHTITAN_IN_ROOTFS=%s\n' "${TORCHTITAN_IN_ROOTFS}"
printf 'formal cache: %s\n' "${formal_cache}"
printf 'formal mode: %s\n' "${mode}"
printf 'formal suite: %s\n' "${suite}"
if [[ "${mode}" == "no-fetch" && ! -f "${vendor_registry}" ]]; then
  die "--no-fetch requires materialized Bazel dependencies; run --networked first"
fi
if [[ "${mode}" == "networked" ]]; then
  "${bazel_bin}" --output_user_root="${output_root}" vendor \
    "${common_flags[@]}" "${formal_target}"
  [[ -f "${vendor_registry}" ]] \
    || die "Bazel did not vendor the BCR metadata required for --no-fetch"
  "${bazel_bin}" --output_user_root="${output_root}" vendor \
    "${common_flags[@]}" --registry="${local_registry}" "${formal_target}"
fi
exec "${bazel_bin}" --output_user_root="${output_root}" test \
  "${test_flags[@]}" "${formal_target}"
