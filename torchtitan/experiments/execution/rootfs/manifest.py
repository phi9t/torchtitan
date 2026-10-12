# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build manifest and content-addressed store ID (roadmap Section 8.1).

The hardened builder must record exactly which locked inputs produced a rootfs
so a scientific report can reference an explainable identity instead of the
mutable ``torchtitan-rootfs:local`` tag. Two builds from the same locked inputs
must compute the same store ID; changing any locked input must change it.

The store ID is a content address over the *locked build inputs* only. It is
deliberately independent of build-provenance fields (build time, host tool
versions, source revision) so an incidental rebuild on a different day at the
same inputs is recognized as the same environment. Provenance is still recorded
in the manifest for auditing; it just does not participate in the identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# Manifest schema version. Bump when the manifest layout changes in a way that
# older readers cannot interpret.
MANIFEST_SCHEMA_VERSION = 1

# The store-ID prefix keeps content addresses self-describing in paths and logs.
STORE_ID_PREFIX = "rootfs"

# Locked-input fields that define environment identity (roadmap 8.1). Every
# field here must be a resolved, immutable value: a digest, a hash, or a pinned
# version. Mutable tags or unpinned ranges are not scientific inputs.
_LOCKED_INPUT_FIELDS = (
    "base_image_digest",
    "uv_image_digest",
    "apt_lock_digest",
    "python_lock_digest",
    "cuda_package_versions",
    "dockerfile_digest",
)


@dataclass(frozen=True)
class LockedBuildInputs:
    """The resolved, immutable inputs that define a rootfs environment.

    Every field is a digest, a content hash, or a pinned version list. These are
    the only inputs that participate in the store ID; anything mutable (a tag, a
    branch, an unhashed range) is rejected at construction so it can never leak
    into the content address.
    """

    base_image_digest: str
    uv_image_digest: str
    apt_lock_digest: str
    python_lock_digest: str
    cuda_package_versions: tuple[str, ...]
    dockerfile_digest: str

    def __post_init__(self) -> None:
        for name in (
            "base_image_digest",
            "uv_image_digest",
            "apt_lock_digest",
            "python_lock_digest",
            "dockerfile_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"LockedBuildInputs.{name} must be a non-empty string")
        # A bare mutable image reference (no digest) is not a scientific input.
        for name in ("base_image_digest", "uv_image_digest"):
            value = getattr(self, name)
            if "@sha256:" not in value and not value.startswith("sha256:"):
                raise ValueError(
                    f"LockedBuildInputs.{name} must be digest-pinned, got {value!r}"
                )
        if not self.cuda_package_versions:
            raise ValueError(
                "LockedBuildInputs.cuda_package_versions must be non-empty"
            )
        for version in self.cuda_package_versions:
            if not isinstance(version, str) or not version:
                raise ValueError(
                    "cuda_package_versions entries must be non-empty strings"
                )

    def canonical_payload(self) -> dict[str, object]:
        """Return the identity payload in a stable, order-insensitive form."""

        payload: dict[str, object] = {}
        for name in _LOCKED_INPUT_FIELDS:
            value = getattr(self, name)
            if name == "cuda_package_versions":
                # Sort so the CUDA package order does not affect identity.
                value = sorted(value)
            payload[name] = value
        return payload


@dataclass(frozen=True)
class BuildProvenance:
    """Auditing metadata that does not participate in environment identity.

    Recorded in the manifest for traceability. Kept out of the store ID so a
    reproducible rebuild at identical locked inputs is the same environment even
    when built at a different time on a different host.
    """

    build_time: str
    host_tool_versions: dict[str, str] = field(default_factory=dict)
    source_revision: str = ""
    source_tree_clean: bool = True


def compute_store_id(inputs: LockedBuildInputs) -> str:
    """Compute the content-addressed store ID for a set of locked inputs.

    The digest is taken over a canonical JSON encoding of the locked-input
    payload with sorted keys, so field order and CUDA-package order never
    change the result.
    """

    encoded = json.dumps(
        inputs.canonical_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{STORE_ID_PREFIX}-{digest}"


def build_manifest(
    *,
    inputs: LockedBuildInputs,
    provenance: BuildProvenance,
    image_id: str,
    exported_rootfs_digest: str,
) -> dict[str, object]:
    """Assemble the build manifest a report references by digest (roadmap 8.1).

    ``image_id`` and ``exported_rootfs_digest`` are build *outputs*: the Docker
    image ID and the digest of the exported rootfs tree. They are recorded and
    validated but do not define identity, because identity comes from the locked
    inputs; a matching store ID with a mismatched exported digest signals a
    corrupt or tampered build, which validation must catch.
    """

    if not image_id:
        raise ValueError("build_manifest requires a non-empty image_id")
    if not exported_rootfs_digest:
        raise ValueError("build_manifest requires a non-empty exported_rootfs_digest")

    store_id = compute_store_id(inputs)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "rootfs_build_manifest",
        "store_id": store_id,
        "locked_inputs": inputs.canonical_payload(),
        "outputs": {
            "image_id": image_id,
            "exported_rootfs_digest": exported_rootfs_digest,
        },
        "provenance": asdict(provenance),
    }


def validate_manifest(manifest: dict[str, object]) -> str:
    """Validate a manifest's internal consistency and return its store ID.

    Recomputes the store ID from the recorded locked inputs and rejects a
    manifest whose recorded store ID disagrees, since that means the identity no
    longer matches the inputs it claims. Returns the validated store ID.
    """

    if manifest.get("kind") != "rootfs_build_manifest":
        raise ValueError("manifest kind is not rootfs_build_manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version: {manifest.get('schema_version')!r}"
        )
    locked = manifest.get("locked_inputs")
    if not isinstance(locked, dict):
        raise ValueError("manifest has no locked_inputs object")
    recorded_id = manifest.get("store_id")
    if not isinstance(recorded_id, str) or not recorded_id:
        raise ValueError("manifest has no store_id")

    inputs = LockedBuildInputs(
        base_image_digest=locked["base_image_digest"],
        uv_image_digest=locked["uv_image_digest"],
        apt_lock_digest=locked["apt_lock_digest"],
        python_lock_digest=locked["python_lock_digest"],
        cuda_package_versions=tuple(locked["cuda_package_versions"]),
        dockerfile_digest=locked["dockerfile_digest"],
    )
    recomputed = compute_store_id(inputs)
    if recomputed != recorded_id:
        raise ValueError(
            f"manifest store_id {recorded_id!r} does not match recomputed "
            f"{recomputed!r}: locked inputs were altered after signing"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get("exported_rootfs_digest"):
        raise ValueError("manifest has no outputs.exported_rootfs_digest")
    return recorded_id
