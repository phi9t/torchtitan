#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Shared canonical target resolver for the TorchTitan bwrap rootfs.
#
# The builder and launcher must agree on exactly which directories may be
# created, replaced, or deleted. An arbitrary --dest or --rootfs must never
# become a deletion target through a typo, a symlink, or a broad directory.
# This library provides one fail-closed resolver and an ownership-marker
# contract that both scripts source.
#
# F0 policy (runtime_preflight_roadmap.md Section 8.2): the allowlist contains
# only the canonical legacy target scripts/rootfs/rootfs. F3 later narrows
# managed builds to content entries beneath a dedicated store.

ROOTFS_TARGET_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOTFS_OWNERSHIP_MARKER_NAME=".torchtitan-rootfs-owner"

# The F0 allowlist: canonical real paths that may be built or replaced.
rootfs_allowlist() {
  printf '%s\n' "${ROOTFS_TARGET_REPO_ROOT}/scripts/rootfs/rootfs"
}

rootfs_die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

# Canonicalize a requested destination without following a final symlink and
# refuse hostile shapes. Prints the canonical absolute path on success.
#
# The parent directory must already exist and be a real, non-symlinked
# directory; the basename is appended without traversal. This lets us resolve a
# not-yet-created target while refusing a symlinked or non-directory parent.
rootfs_resolve_managed_dest() {
  local requested="$1"
  if [[ -z "${requested}" ]]; then
    rootfs_die "empty rootfs destination is not allowed"
  fi
  if [[ "${requested}" == "/" ]]; then
    rootfs_die "refusing / as a rootfs destination"
  fi

  local parent basename
  parent="$(dirname -- "${requested}")"
  basename="$(basename -- "${requested}")"
  if [[ -z "${basename}" || "${basename}" == "." || "${basename}" == ".." ]]; then
    rootfs_die "invalid rootfs basename in ${requested}"
  fi
  if [[ ! -d "${parent}" ]]; then
    rootfs_die "rootfs parent is not an existing directory: ${parent}"
  fi
  # Refuse a symlinked parent component: canonicalize the parent with -P (no
  # symlinks) and compare against a plain resolution. A symlinked parent would
  # otherwise let a delete target escape the allowlist.
  local canonical_parent
  canonical_parent="$(cd -P -- "${parent}" && pwd)" \
    || rootfs_die "cannot resolve rootfs parent: ${parent}"
  local logical_parent
  logical_parent="$(cd -- "${parent}" && pwd)" \
    || rootfs_die "cannot resolve rootfs parent: ${parent}"
  if [[ "${canonical_parent}" != "${logical_parent}" ]]; then
    rootfs_die "refusing symlinked parent component for ${requested}"
  fi

  local canonical="${canonical_parent}/${basename}"

  # Refuse broad or sensitive roots outright.
  case "${canonical}" in
    "/"|"${HOME}"|"${ROOTFS_TARGET_REPO_ROOT}")
      rootfs_die "refusing broad rootfs destination: ${canonical}"
      ;;
  esac

  # If the exact path already exists it must be a real directory, not a symlink
  # or other file type.
  if [[ -L "${canonical}" ]]; then
    rootfs_die "refusing symlink rootfs destination: ${canonical}"
  fi
  if [[ -e "${canonical}" && ! -d "${canonical}" ]]; then
    rootfs_die "rootfs destination exists and is not a directory: ${canonical}"
  fi

  local allowed
  while IFS= read -r allowed; do
    if [[ "${canonical}" == "${allowed}" ]]; then
      printf '%s\n' "${canonical}"
      return 0
    fi
  done < <(rootfs_allowlist)

  rootfs_die "rootfs destination is not in the allowlist: ${canonical}"
}

rootfs_ownership_marker_path() {
  printf '%s/%s\n' "$1" "${ROOTFS_OWNERSHIP_MARKER_NAME}"
}

# Record builder ownership before extraction so a later replace can prove the
# target is a managed rootfs and not an unrelated directory.
rootfs_write_ownership_marker() {
  local dest="$1"
  [[ -d "${dest}" ]] || rootfs_die "cannot mark non-directory: ${dest}"
  printf '%s\n' "torchtitan-rootfs" > "$(rootfs_ownership_marker_path "${dest}")"
}

# Refuse to remove or replace a directory unless its exact, non-symlink path
# carries a valid builder ownership marker.
rootfs_assert_removable() {
  local dest="$1"
  if [[ -z "${dest}" ]]; then
    rootfs_die "empty rootfs destination is not removable"
  fi
  if [[ -L "${dest}" ]]; then
    rootfs_die "refusing to remove symlink rootfs destination: ${dest}"
  fi
  if [[ ! -d "${dest}" ]]; then
    rootfs_die "rootfs destination is not a directory: ${dest}"
  fi
  local marker
  marker="$(rootfs_ownership_marker_path "${dest}")"
  if [[ ! -f "${marker}" ]]; then
    rootfs_die "refusing to remove rootfs without ownership marker: ${dest}"
  fi
}
