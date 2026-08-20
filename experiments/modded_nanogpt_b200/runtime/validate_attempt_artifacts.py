#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate launch-governing artifacts in one modded-nanogpt attempt directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200.runtime import schema_validation


UPSTREAM_COMMIT = "ecbb586296d3dac36fd206211f25d63bad4a6b35"

REQUIRED_SIDECARS = (
    ("attempt", Path("attempt.json")),
    ("command_env", Path("command.env.json")),
    ("command_argv", Path("command.argv.json")),
    ("preflight_report", Path("preflight_report.json")),
    ("launch_readiness", Path("launch_readiness.json")),
    ("runtime_verification", Path("runtime/runtime_verification.json")),
    ("summary", Path("summary.json")),
)


@dataclass(frozen=True)
class SidecarResult:
    schema: str
    path: str
    ok: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": self.path,
            "ok": self.ok,
            "message": self.message,
        }


def validate_attempt_dir(result_dir: Path) -> dict[str, Any]:
    """Validate required launch-governing sidecars under result_dir."""

    sidecars = [
        _validate_sidecar(result_dir=result_dir, schema_name=schema, rel_path=rel_path)
        for schema, rel_path in REQUIRED_SIDECARS
    ]
    sidecars.append(_validate_data_manifest_pointer(result_dir))
    sidecars.append(_validate_hardware_sidecar(result_dir))
    sidecars.append(_validate_environment_sidecar(result_dir))
    sidecars.append(_validate_rootfs_environment(result_dir))
    sidecars.extend(_validate_referenced_data_manifest(result_dir))
    sidecars.extend(_validate_referenced_sidecars(result_dir))
    sidecars.extend(_validate_cross_artifact_consistency(result_dir))
    sidecars.extend(_validate_local_digest_integrity(result_dir))
    return {
        "schema_version": 1,
        "kind": "attempt_artifact_validation",
        "result_dir": str(result_dir),
        "ok": all(sidecar.ok for sidecar in sidecars),
        "sidecars": [sidecar.to_dict() for sidecar in sidecars],
    }


def _validate_sidecar(
    *, result_dir: Path, schema_name: str, rel_path: Path
) -> SidecarResult:
    path = result_dir / rel_path
    if not path.is_file():
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message="missing",
        )
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise schema_validation.SchemaValidationError("artifact must be an object")
        schema_validation.validate(schema_name, payload)
    except (
        json.JSONDecodeError,
        OSError,
        schema_validation.SchemaValidationError,
    ) as exc:
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=str(exc),
        )
    return SidecarResult(
        schema=schema_name,
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_cross_artifact_consistency(result_dir: Path) -> list[SidecarResult]:
    payloads: dict[str, dict[str, Any]] = {}
    for schema_name, rel_path in REQUIRED_SIDECARS:
        path = result_dir / rel_path
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            payloads[schema_name] = payload
    data_manifest = _load_json_object(result_dir / "data_manifest.json")
    if data_manifest is not None:
        payloads["data_manifest_pointer"] = data_manifest
    hardware = _load_json_object(result_dir / "hardware.json")
    if hardware is not None:
        payloads["hardware_sidecar"] = hardware
    environment = _load_json_object(result_dir / "environment.json")
    if environment is not None:
        payloads["environment_sidecar"] = environment
    rootfs_environment = _load_json_object(
        result_dir / "telemetry" / "rootfs_environment.json"
    )
    if rootfs_environment is not None:
        payloads["rootfs_environment"] = rootfs_environment

    sidecars: list[SidecarResult] = []
    sidecars.extend(_validate_identity_consistency(result_dir, payloads))
    sidecars.extend(_validate_command_env_digest_consistency(result_dir, payloads))
    sidecars.extend(_validate_data_manifest_consistency(result_dir, payloads))
    sidecars.extend(_validate_data_manifest_metadata_consistency(result_dir, payloads))
    sidecars.extend(_validate_gpu_consistency(result_dir, payloads))
    sidecars.extend(_validate_environment_consistency(result_dir, payloads))
    sidecars.extend(_validate_rootfs_consistency(result_dir, payloads))
    return sidecars


def _validate_data_manifest_pointer(result_dir: Path) -> SidecarResult:
    path = result_dir / "data_manifest.json"
    if not path.is_file():
        return SidecarResult(
            schema="data_manifest_pointer",
            path=str(path),
            ok=False,
            message="missing",
        )
    pointer = _load_json_object(path)
    if pointer is None:
        return SidecarResult(
            schema="data_manifest_pointer",
            path=str(path),
            ok=False,
            message="artifact must be a JSON object",
        )
    validation = _validate_data_manifest_pointer_shape(path, pointer)
    if validation is not None:
        return validation
    return SidecarResult(
        schema="data_manifest_pointer",
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_referenced_data_manifest(result_dir: Path) -> list[SidecarResult]:
    pointer_path = result_dir / "data_manifest.json"
    pointer = _load_json_object(pointer_path)
    if pointer is None:
        return []
    shape_validation = _validate_data_manifest_pointer_shape(pointer_path, pointer)
    if shape_validation is not None:
        return []

    manifest_path = _resolve_artifact_path(result_dir, str(pointer["path"]))
    if not manifest_path.is_file():
        return [
            SidecarResult(
                schema="referenced_data_manifest",
                path=str(manifest_path),
                ok=False,
                message="missing",
            )
        ]
    try:
        payload = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return [
            SidecarResult(
                schema="referenced_data_manifest",
                path=str(manifest_path),
                ok=False,
                message=str(exc),
            )
        ]
    if not isinstance(payload, dict):
        return [
            SidecarResult(
                schema="referenced_data_manifest",
                path=str(manifest_path),
                ok=False,
                message="artifact must be an object",
            )
        ]

    shape_validation = _validate_full_data_manifest_shape(manifest_path, payload)
    if shape_validation is not None:
        return [shape_validation]
    return [
        SidecarResult(
            schema="referenced_data_manifest",
            path=str(manifest_path),
            ok=True,
            message="ok",
        )
    ]


def _validate_hardware_sidecar(result_dir: Path) -> SidecarResult:
    path = result_dir / "hardware.json"
    if not path.is_file():
        return SidecarResult(
            schema="hardware_sidecar",
            path=str(path),
            ok=False,
            message="missing",
        )
    payload = _load_json_object(path)
    if payload is None:
        return SidecarResult(
            schema="hardware_sidecar",
            path=str(path),
            ok=False,
            message="artifact must be a JSON object",
        )
    validation = _validate_hardware_sidecar_shape(path, payload)
    if validation is not None:
        return validation
    return SidecarResult(
        schema="hardware_sidecar",
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_environment_sidecar(result_dir: Path) -> SidecarResult:
    path = result_dir / "environment.json"
    if not path.is_file():
        return SidecarResult(
            schema="environment_sidecar",
            path=str(path),
            ok=False,
            message="missing",
        )
    payload = _load_json_object(path)
    if payload is None:
        return SidecarResult(
            schema="environment_sidecar",
            path=str(path),
            ok=False,
            message="artifact must be a JSON object",
        )
    validation = _validate_environment_sidecar_shape(path, payload)
    if validation is not None:
        return validation
    return SidecarResult(
        schema="environment_sidecar",
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_rootfs_environment(result_dir: Path) -> SidecarResult:
    path = result_dir / "telemetry" / "rootfs_environment.json"
    if not path.is_file():
        return SidecarResult(
            schema="rootfs_environment",
            path=str(path),
            ok=False,
            message="missing",
        )
    payload = _load_json_object(path)
    if payload is None:
        return SidecarResult(
            schema="rootfs_environment",
            path=str(path),
            ok=False,
            message="artifact must be a JSON object",
        )
    validation = _validate_rootfs_environment_shape(path, payload)
    if validation is not None:
        return validation
    return SidecarResult(
        schema="rootfs_environment",
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_local_digest_integrity(result_dir: Path) -> list[SidecarResult]:
    sidecars: list[SidecarResult] = []
    command_env = _load_json_object(result_dir / "command.env.json")
    if command_env is not None:
        sidecars.append(
            _validate_embedded_digest(
                schema_name="command_env_digest_integrity",
                path=result_dir / "command.env.json",
                payload=command_env,
                data_field="environment",
                digest_field="environment_digest",
            )
        )
    command_argv = _load_json_object(result_dir / "command.argv.json")
    if command_argv is not None:
        sidecars.append(
            _validate_embedded_digest(
                schema_name="command_argv_digest_integrity",
                path=result_dir / "command.argv.json",
                payload=command_argv,
                data_field="argv",
                digest_field="argv_digest",
            )
        )
    return sidecars


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _validate_embedded_digest(
    *,
    schema_name: str,
    path: Path,
    payload: dict[str, Any],
    data_field: str,
    digest_field: str,
) -> SidecarResult:
    expected = payload.get(digest_field)
    data = payload.get(data_field)
    if not isinstance(expected, dict):
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=f"missing {digest_field}",
        )
    actual = _json_digest(data)
    if actual != expected:
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=f"{digest_field} mismatch; expected {expected}; actual {actual}",
        )
    return SidecarResult(
        schema=schema_name,
        path=str(path),
        ok=True,
        message="ok",
    )


def _json_digest(data: object) -> dict[str, str]:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"algorithm": "sha256", "sha256": hashlib.sha256(payload).hexdigest()}


def _json_report_digest(data: dict[str, Any], *, digest_field: str) -> dict[str, str]:
    payload = dict(data)
    payload.pop(digest_field, None)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest()}


def _validate_identity_consistency(
    result_dir: Path, payloads: dict[str, dict[str, Any]]
) -> list[SidecarResult]:
    identities: list[tuple[str, str, str]] = []
    for name, payload in payloads.items():
        for label, nested in _identity_sources(payload):
            run_id = nested.get("run_id")
            attempt_id = nested.get("attempt_id")
            if isinstance(run_id, str) and isinstance(attempt_id, str):
                identities.append((f"{name}.{label}", run_id, attempt_id))
    if len(identities) < 2:
        return []

    _, expected_run_id, expected_attempt_id = identities[0]
    mismatches = [
        f"{label}={run_id}/{attempt_id}"
        for label, run_id, attempt_id in identities
        if run_id != expected_run_id or attempt_id != expected_attempt_id
    ]
    if not mismatches:
        return [
            SidecarResult(
                schema="cross_artifact_identity",
                path=str(result_dir),
                ok=True,
                message="ok",
            )
        ]
    return [
        SidecarResult(
            schema="cross_artifact_identity",
            path=str(result_dir),
            ok=False,
            message=(
                f"identity mismatch; expected {expected_run_id}/{expected_attempt_id}; "
                + "; ".join(mismatches)
            ),
        )
    ]


def _identity_sources(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    sources: list[tuple[str, dict[str, Any]]] = [("top_level", payload)]
    classification = payload.get("classification")
    if isinstance(classification, dict):
        sources.append(("classification", classification))
    launch_readiness = payload.get("launch_readiness")
    if isinstance(launch_readiness, dict):
        sources.append(("launch_readiness", launch_readiness))
        nested_classification = launch_readiness.get("classification")
        if isinstance(nested_classification, dict):
            sources.append(("launch_readiness.classification", nested_classification))
    return sources


def _validate_command_env_digest_consistency(
    result_dir: Path, payloads: dict[str, dict[str, Any]]
) -> list[SidecarResult]:
    digests: list[tuple[str, dict[str, Any]]] = []
    command_env = payloads.get("command_env", {})
    environment_digest = command_env.get("environment_digest")
    if isinstance(environment_digest, dict):
        digests.append(("command_env.environment_digest", environment_digest))
    runtime = payloads.get("runtime_verification", {})
    runtime_digest = runtime.get("command_env_digest")
    if isinstance(runtime_digest, dict):
        digests.append(("runtime_verification.command_env_digest", runtime_digest))
    launch_readiness = payloads.get("launch_readiness", {})
    launch_digest = launch_readiness.get("command_env_digest")
    if isinstance(launch_digest, dict):
        digests.append(("launch_readiness.command_env_digest", launch_digest))
    runtime_ref = launch_readiness.get("runtime_verification")
    if isinstance(runtime_ref, dict) and isinstance(
        runtime_ref.get("command_env_digest"), dict
    ):
        digests.append(
            (
                "launch_readiness.runtime_verification.command_env_digest",
                runtime_ref["command_env_digest"],
            )
        )
    summary = payloads.get("summary", {})
    summary_readiness = summary.get("launch_readiness")
    if isinstance(summary_readiness, dict):
        summary_digest = summary_readiness.get("command_env_digest")
        if isinstance(summary_digest, dict):
            digests.append(
                ("summary.launch_readiness.command_env_digest", summary_digest)
            )
    if len(digests) < 2:
        return []

    _, expected = digests[0]
    mismatches = [
        f"{label}={digest}" for label, digest in digests if digest != expected
    ]
    if not mismatches:
        return [
            SidecarResult(
                schema="cross_artifact_command_env_digest",
                path=str(result_dir),
                ok=True,
                message="ok",
            )
        ]
    return [
        SidecarResult(
            schema="cross_artifact_command_env_digest",
            path=str(result_dir),
            ok=False,
            message=(
                f"command_env_digest mismatch; expected {expected}; "
                + "; ".join(mismatches)
            ),
        )
    ]


def _validate_data_manifest_consistency(
    result_dir: Path, payloads: dict[str, dict[str, Any]]
) -> list[SidecarResult]:
    pointer = payloads.get("data_manifest_pointer")
    if pointer is None:
        return []

    pointer_path = result_dir / "data_manifest.json"
    shape_validation = _validate_data_manifest_pointer_shape(pointer_path, pointer)
    if shape_validation is not None:
        return [
            SidecarResult(
                schema="cross_artifact_data_manifest",
                path=shape_validation.path,
                ok=False,
                message=shape_validation.message,
            )
        ]

    manifest_path = pointer.get("path")
    assert isinstance(manifest_path, str)

    manifest_paths = [("data_manifest_pointer.path", manifest_path)]
    attempt_path = payloads.get("attempt", {}).get("data_manifest")
    if isinstance(attempt_path, str):
        manifest_paths.append(("attempt.data_manifest", attempt_path))
    launch_readiness = payloads.get("launch_readiness", {})
    readiness_path = launch_readiness.get("data_manifest")
    if isinstance(readiness_path, str):
        manifest_paths.append(("launch_readiness.data_manifest", readiness_path))
    summary = payloads.get("summary", {})
    summary_path = summary.get("data_manifest")
    if isinstance(summary_path, str):
        manifest_paths.append(("summary.data_manifest", summary_path))
    summary_readiness = summary.get("launch_readiness")
    if isinstance(summary_readiness, dict):
        summary_readiness_path = summary_readiness.get("data_manifest")
        if isinstance(summary_readiness_path, str):
            manifest_paths.append(
                ("summary.launch_readiness.data_manifest", summary_readiness_path)
            )

    mismatches = [
        f"{label}={path}" for label, path in manifest_paths if path != manifest_path
    ]
    if mismatches:
        return [
            SidecarResult(
                schema="cross_artifact_data_manifest",
                path=str(pointer_path),
                ok=False,
                message=(
                    f"data_manifest mismatch; expected {manifest_path}; "
                    + "; ".join(mismatches)
                ),
            )
        ]
    return [
        SidecarResult(
            schema="cross_artifact_data_manifest",
            path=str(pointer_path),
            ok=True,
            message="ok",
        )
    ]


def _validate_data_manifest_metadata_consistency(
    result_dir: Path, payloads: dict[str, dict[str, Any]]
) -> list[SidecarResult]:
    pointer = payloads.get("data_manifest_pointer")
    summary = payloads.get("summary")
    if pointer is None or summary is None:
        return []
    pointer_path = result_dir / "data_manifest.json"
    if _validate_data_manifest_pointer_shape(pointer_path, pointer) is not None:
        return []

    manifest_path = _resolve_artifact_path(result_dir, str(pointer["path"]))
    manifest = _load_json_object(manifest_path)
    if manifest is None:
        return []
    if _validate_full_data_manifest_shape(manifest_path, manifest) is not None:
        return []

    checks: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    data_manifest_summary = summary.get("data_manifest_summary")
    if isinstance(data_manifest_summary, dict):
        checks.append(
            (
                "summary.data_manifest_summary",
                data_manifest_summary,
                {
                    "schema_version": manifest["schema_version"],
                    "dataset": manifest["dataset"],
                    "token_budget": manifest["token_budget"],
                    "num_files": manifest["num_files"],
                    "total_bytes": manifest["total_bytes"],
                    "verified_sha": manifest["verified_sha"],
                    "sha256_entries": len(manifest["files"]),
                },
            )
        )
    preflight_manifest = summary.get("preflight_data_manifest")
    if isinstance(preflight_manifest, dict):
        checks.append(
            (
                "summary.preflight_data_manifest",
                preflight_manifest,
                {
                    "dataset": manifest["dataset"],
                    "token_budget": manifest["token_budget"],
                    "num_files": manifest["num_files"],
                    "total_bytes": manifest["total_bytes"],
                    "verified_sha": manifest["verified_sha"],
                    "manifest_verified_sha": manifest["verified_sha"],
                },
            )
        )
    if not checks:
        return []

    mismatches: list[str] = []
    for label, observed, expected_fields in checks:
        for field, expected in expected_fields.items():
            if field in observed and observed[field] != expected:
                mismatches.append(
                    f"{label}.{field}={observed[field]} expected {expected}"
                )
    if mismatches:
        return [
            SidecarResult(
                schema="cross_artifact_data_manifest_metadata",
                path=str(manifest_path),
                ok=False,
                message="data manifest metadata mismatch; " + "; ".join(mismatches),
            )
        ]
    return [
        SidecarResult(
            schema="cross_artifact_data_manifest_metadata",
            path=str(manifest_path),
            ok=True,
            message="ok",
        )
    ]


def _validate_gpu_consistency(
    result_dir: Path, payloads: dict[str, dict[str, Any]]
) -> list[SidecarResult]:
    hardware = payloads.get("hardware_sidecar")
    if hardware is None:
        return []
    hardware_path = result_dir / "hardware.json"
    if _validate_hardware_sidecar_shape(hardware_path, hardware) is not None:
        return []

    hardware_gpus = hardware["gpus"]
    gpu_sources: list[tuple[str, list[Any]]] = [("hardware.gpus", hardware_gpus)]
    summary = payloads.get("summary", {})
    summary_gpus = summary.get("gpus")
    if isinstance(summary_gpus, list):
        gpu_sources.append(("summary.gpus", summary_gpus))
    summary_hardware = summary.get("hardware_sidecar")
    if isinstance(summary_hardware, dict) and isinstance(
        summary_hardware.get("gpus"), list
    ):
        gpu_sources.append(("summary.hardware_sidecar.gpus", summary_hardware["gpus"]))
    preflight = payloads.get("preflight_report", {})
    preflight_gpus = preflight.get("gpus")
    if isinstance(preflight_gpus, list):
        gpu_sources.append(("preflight_report.gpus", preflight_gpus))
    for check in preflight.get("checks", []):
        if not isinstance(check, dict) or check.get("name") != "gpu_inventory":
            continue
        detail = check.get("detail")
        if isinstance(detail, list):
            gpu_sources.append(("preflight_report.checks.gpu_inventory.detail", detail))

    expected = _normalize_gpus(hardware_gpus)
    mismatches: list[str] = []
    for label, gpus in gpu_sources:
        normalized = _normalize_gpus(gpus)
        if normalized != expected:
            mismatches.append(f"{label}={normalized} expected {expected}")
    if mismatches:
        return [
            SidecarResult(
                schema="cross_artifact_gpu_inventory",
                path=str(hardware_path),
                ok=False,
                message="gpu inventory mismatch; " + "; ".join(mismatches),
            )
        ]
    return [
        SidecarResult(
            schema="cross_artifact_gpu_inventory",
            path=str(hardware_path),
            ok=True,
            message="ok",
        )
    ]


def _validate_environment_consistency(
    result_dir: Path, payloads: dict[str, dict[str, Any]]
) -> list[SidecarResult]:
    environment_sidecar = payloads.get("environment_sidecar")
    if environment_sidecar is None:
        return []
    environment_path = result_dir / "environment.json"
    if (
        _validate_environment_sidecar_shape(environment_path, environment_sidecar)
        is not None
    ):
        return []

    expected = environment_sidecar["environment"]
    environment_sources: list[tuple[str, dict[str, Any]]] = [
        ("environment.environment", expected)
    ]
    preflight = payloads.get("preflight_report", {})
    preflight_environment = preflight.get("environment")
    if isinstance(preflight_environment, dict):
        environment_sources.append(
            ("preflight_report.environment", preflight_environment)
        )
    summary = payloads.get("summary", {})
    summary_environment = summary.get("environment")
    if isinstance(summary_environment, dict):
        environment_sources.append(("summary.environment", summary_environment))
    summary_environment_sidecar = summary.get("environment_sidecar")
    if isinstance(summary_environment_sidecar, dict) and isinstance(
        summary_environment_sidecar.get("environment"), dict
    ):
        environment_sources.append(
            (
                "summary.environment_sidecar.environment",
                summary_environment_sidecar["environment"],
            )
        )

    mismatches: list[str] = []
    for label, observed in environment_sources:
        for field, expected_value in expected.items():
            if field in observed and observed[field] != expected_value:
                mismatches.append(
                    f"{label}.{field}={observed[field]} expected {expected_value}"
                )
    if mismatches:
        return [
            SidecarResult(
                schema="cross_artifact_environment",
                path=str(environment_path),
                ok=False,
                message="environment mismatch; " + "; ".join(mismatches),
            )
        ]
    return [
        SidecarResult(
            schema="cross_artifact_environment",
            path=str(environment_path),
            ok=True,
            message="ok",
        )
    ]


def _validate_rootfs_consistency(
    result_dir: Path, payloads: dict[str, dict[str, Any]]
) -> list[SidecarResult]:
    rootfs_environment = payloads.get("rootfs_environment")
    if rootfs_environment is None:
        return []
    rootfs_path = result_dir / "telemetry" / "rootfs_environment.json"
    if _validate_rootfs_environment_shape(rootfs_path, rootfs_environment) is not None:
        return []

    expected = {
        "torchtitan_in_rootfs": rootfs_environment["torchtitan_in_rootfs"],
        "cwd": rootfs_environment["cwd"],
        "workspace_sentinel_exists": rootfs_environment["workspace_sentinel_exists"],
    }
    python_executable = rootfs_environment.get("python_executable")
    if isinstance(python_executable, str):
        expected["python_executable"] = python_executable
    summary = payloads.get("summary", {})
    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, dict):
        return []
    summary_rootfs = telemetry.get("rootfs")
    if not isinstance(summary_rootfs, dict):
        return []

    mismatches = [
        f"summary.telemetry.rootfs.{field}={summary_rootfs.get(field)} expected {value}"
        for field, value in expected.items()
        if summary_rootfs.get(field) != value
    ]
    if mismatches:
        return [
            SidecarResult(
                schema="cross_artifact_rootfs_environment",
                path=str(rootfs_path),
                ok=False,
                message="rootfs environment mismatch; " + "; ".join(mismatches),
            )
        ]
    return [
        SidecarResult(
            schema="cross_artifact_rootfs_environment",
            path=str(rootfs_path),
            ok=True,
            message="ok",
        )
    ]


def _validate_data_manifest_pointer_shape(
    path: Path, pointer: dict[str, Any]
) -> SidecarResult | None:
    if pointer.get("schema_version") != 1:
        return SidecarResult(
            schema="data_manifest_pointer",
            path=str(path),
            ok=False,
            message="data_manifest pointer schema_version must be 1",
        )
    if pointer.get("kind") != "data_manifest_pointer":
        return SidecarResult(
            schema="data_manifest_pointer",
            path=str(path),
            ok=False,
            message="data_manifest pointer kind must be data_manifest_pointer",
        )
    manifest_path = pointer.get("path")
    if not isinstance(manifest_path, str) or not manifest_path:
        return SidecarResult(
            schema="data_manifest_pointer",
            path=str(path),
            ok=False,
            message="data_manifest pointer path must be a non-empty string",
        )
    extra_fields = set(pointer) - {"schema_version", "kind", "path"}
    if extra_fields:
        return SidecarResult(
            schema="data_manifest_pointer",
            path=str(path),
            ok=False,
            message=f"data_manifest pointer has unknown fields: {sorted(extra_fields)}",
        )
    return None


def _validate_environment_sidecar_shape(
    path: Path, payload: dict[str, Any]
) -> SidecarResult | None:
    if payload.get("schema_version") != 1:
        return SidecarResult(
            schema="environment_sidecar",
            path=str(path),
            ok=False,
            message="environment sidecar schema_version must be 1",
        )
    if payload.get("kind") != "preflight_environment":
        return SidecarResult(
            schema="environment_sidecar",
            path=str(path),
            ok=False,
            message="environment sidecar kind must be preflight_environment",
        )
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return SidecarResult(
            schema="environment_sidecar",
            path=str(path),
            ok=False,
            message="environment sidecar environment must be an object",
        )
    required_fields = ("python", "torch", "cuda_runtime", "triton", "flash_attention")
    for field in required_fields:
        value = environment.get(field)
        if not isinstance(value, str) or not value:
            return SidecarResult(
                schema="environment_sidecar",
                path=str(path),
                ok=False,
                message=f"environment.{field} must be a non-empty string",
            )
    extra_fields = set(payload) - {"schema_version", "kind", "environment"}
    if extra_fields:
        return SidecarResult(
            schema="environment_sidecar",
            path=str(path),
            ok=False,
            message=f"environment sidecar has unknown fields: {sorted(extra_fields)}",
        )
    return None


def _validate_rootfs_environment_shape(
    path: Path, payload: dict[str, Any]
) -> SidecarResult | None:
    expected_fields = {
        "torchtitan_in_rootfs": "1",
        "cwd": "/workspace/torchtitan",
        "workspace_sentinel_exists": True,
    }
    for field, expected in expected_fields.items():
        actual = payload.get(field)
        if actual != expected:
            return SidecarResult(
                schema="rootfs_environment",
                path=str(path),
                ok=False,
                message=f"{field} mismatch; expected {expected}; actual {actual}",
            )
    sentinel = payload.get("workspace_sentinel")
    if sentinel is not None and sentinel != "scripts/rootfs/enter_rootfs.sh":
        return SidecarResult(
            schema="rootfs_environment",
            path=str(path),
            ok=False,
            message=(
                "workspace_sentinel mismatch; expected "
                "scripts/rootfs/enter_rootfs.sh"
            ),
        )
    python_executable = payload.get("python_executable")
    if python_executable is not None and (
        not isinstance(python_executable, str) or not python_executable
    ):
        return SidecarResult(
            schema="rootfs_environment",
            path=str(path),
            ok=False,
            message="python_executable must be a non-empty string when present",
        )
    return None


def _validate_hardware_sidecar_shape(
    path: Path, payload: dict[str, Any]
) -> SidecarResult | None:
    if payload.get("schema_version") != 1:
        return SidecarResult(
            schema="hardware_sidecar",
            path=str(path),
            ok=False,
            message="hardware sidecar schema_version must be 1",
        )
    if payload.get("kind") != "preflight_gpus":
        return SidecarResult(
            schema="hardware_sidecar",
            path=str(path),
            ok=False,
            message="hardware sidecar kind must be preflight_gpus",
        )
    gpus = payload.get("gpus")
    if _normalize_gpus(gpus) != (
        (0, "NVIDIA B200", (10, 0)),
        (1, "NVIDIA B200", (10, 0)),
    ):
        return SidecarResult(
            schema="hardware_sidecar",
            path=str(path),
            ok=False,
            message="hardware sidecar must record exactly two visible NVIDIA B200 GPUs",
        )
    extra_fields = set(payload) - {"schema_version", "kind", "gpus"}
    if extra_fields:
        return SidecarResult(
            schema="hardware_sidecar",
            path=str(path),
            ok=False,
            message=f"hardware sidecar has unknown fields: {sorted(extra_fields)}",
        )
    return None


def _normalize_gpus(
    gpus: object,
) -> tuple[tuple[int, str, tuple[int, int]], ...] | None:
    if not isinstance(gpus, list):
        return None
    normalized: list[tuple[int, str, tuple[int, int]]] = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            return None
        index = gpu.get("index")
        name = gpu.get("name")
        compute_capability = gpu.get("compute_capability")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(name, str)
            or not isinstance(compute_capability, list)
            or len(compute_capability) != 2
            or not all(isinstance(part, int) for part in compute_capability)
        ):
            return None
        normalized.append((index, name, (compute_capability[0], compute_capability[1])))
    return tuple(sorted(normalized))


def _validate_full_data_manifest_shape(
    path: Path, manifest: dict[str, Any]
) -> SidecarResult | None:
    expected_fields = {
        "schema_version": 1,
        "dataset": "fineweb10B",
        "token_budget": "900M",
        "num_files": 10,
        "total_bytes": 2000010240,
        "verified_sha": True,
    }
    for field, expected in expected_fields.items():
        actual = manifest.get(field)
        if actual != expected:
            return SidecarResult(
                schema="referenced_data_manifest",
                path=str(path),
                ok=False,
                message=f"{field} mismatch; expected {expected}; actual {actual}",
            )
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 10:
        return SidecarResult(
            schema="referenced_data_manifest",
            path=str(path),
            ok=False,
            message="files must contain 10 entries",
        )
    source = manifest.get("source")
    source_commit = source.get("commit") if isinstance(source, dict) else None
    if source_commit != UPSTREAM_COMMIT:
        return SidecarResult(
            schema="referenced_data_manifest",
            path=str(path),
            ok=False,
            message=(
                f"source.commit mismatch; expected {UPSTREAM_COMMIT}; "
                f"actual {source_commit}"
            ),
        )
    command = manifest.get("command")
    if not isinstance(command, list) or not command:
        return SidecarResult(
            schema="referenced_data_manifest",
            path=str(path),
            ok=False,
            message="command must be a non-empty list",
        )
    total_bytes = 0
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            return SidecarResult(
                schema="referenced_data_manifest",
                path=str(path),
                ok=False,
                message=f"files[{index}] must be an object",
            )
        shard_path = entry.get("path")
        if not isinstance(shard_path, str) or not shard_path.endswith(".bin"):
            return SidecarResult(
                schema="referenced_data_manifest",
                path=str(path),
                ok=False,
                message=f"files[{index}].path must be a .bin string",
            )
        bytes_value = entry.get("bytes")
        if (
            not isinstance(bytes_value, int)
            or isinstance(bytes_value, bool)
            or bytes_value <= 0
        ):
            return SidecarResult(
                schema="referenced_data_manifest",
                path=str(path),
                ok=False,
                message=f"files[{index}].bytes must be a positive integer",
            )
        total_bytes += bytes_value
        sha256 = entry.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(char not in "0123456789abcdef" for char in sha256)
        ):
            return SidecarResult(
                schema="referenced_data_manifest",
                path=str(path),
                ok=False,
                message=(
                    f"files[{index}].sha256 must be a 64-character "
                    "lowercase hex string"
                ),
            )
    if total_bytes != manifest["total_bytes"]:
        return SidecarResult(
            schema="referenced_data_manifest",
            path=str(path),
            ok=False,
            message=(
                f"files bytes sum mismatch; expected {manifest['total_bytes']}; "
                f"actual {total_bytes}"
            ),
        )
    return None


def _validate_referenced_sidecars(result_dir: Path) -> list[SidecarResult]:
    launch_readiness_path = result_dir / "launch_readiness.json"
    try:
        payload = json.loads(launch_readiness_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []

    sidecars: list[SidecarResult] = []
    runtime_ref = payload.get("runtime_verification")
    if isinstance(runtime_ref, dict) and runtime_ref.get("path"):
        path = _resolve_artifact_path(result_dir, str(runtime_ref["path"]))
        expected = result_dir / "runtime" / "runtime_verification.json"
        sidecars.append(
            _validate_referenced_runtime_verification(
                path=path,
                expected_path=expected,
                runtime_ref=runtime_ref,
            )
        )
    optimized_kernel = payload.get("optimized_kernel_report")
    if isinstance(optimized_kernel, dict) and optimized_kernel.get("path"):
        path = _resolve_artifact_path(result_dir, str(optimized_kernel["path"]))
        sidecars.append(
            _validate_sidecar_path(schema_name="optimized_kernel_report", path=path)
        )
        sidecars.append(
            _validate_referenced_digest(
                schema_name="referenced_optimized_kernel_digest",
                path=path,
                expected_digest=optimized_kernel.get("report_digest"),
                digest_field="report_digest",
            )
        )
    return sidecars


def _resolve_artifact_path(result_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return result_dir / path


def _validate_sidecar_path(*, schema_name: str, path: Path) -> SidecarResult:
    if not path.is_file():
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message="missing",
        )
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise schema_validation.SchemaValidationError("artifact must be an object")
        schema_validation.validate(schema_name, payload)
    except (
        json.JSONDecodeError,
        OSError,
        schema_validation.SchemaValidationError,
    ) as exc:
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=str(exc),
        )
    return SidecarResult(
        schema=schema_name,
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_referenced_path(
    *, schema_name: str, path: Path, expected_path: Path
) -> SidecarResult:
    if path != expected_path:
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=f"unexpected path; expected {expected_path}",
        )
    if not path.is_file():
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message="missing",
        )
    return SidecarResult(
        schema=schema_name,
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_referenced_runtime_verification(
    *, path: Path, expected_path: Path, runtime_ref: dict[str, Any]
) -> SidecarResult:
    path_validation = _validate_referenced_path(
        schema_name="referenced_runtime_verification",
        path=path,
        expected_path=expected_path,
    )
    if not path_validation.ok:
        return path_validation
    payload = _load_json_object(path)
    if payload is None:
        return SidecarResult(
            schema="referenced_runtime_verification",
            path=str(path),
            ok=False,
            message="artifact must be a JSON object",
        )
    fields = ("ok", "training_launch_allowed", "command_env_digest")
    mismatches = [
        f"{field}={runtime_ref.get(field)} expected {payload.get(field)}"
        for field in fields
        if runtime_ref.get(field) != payload.get(field)
    ]
    if mismatches:
        return SidecarResult(
            schema="referenced_runtime_verification",
            path=str(path),
            ok=False,
            message="runtime verification reference mismatch; " + "; ".join(mismatches),
        )
    return SidecarResult(
        schema="referenced_runtime_verification",
        path=str(path),
        ok=True,
        message="ok",
    )


def _validate_referenced_digest(
    *,
    schema_name: str,
    path: Path,
    expected_digest: Any,
    digest_field: str,
) -> SidecarResult:
    if not isinstance(expected_digest, dict):
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=f"missing expected {digest_field}",
        )
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=str(exc),
        )
    if not isinstance(payload, dict):
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message="artifact must be an object",
        )
    actual_digest = payload.get(digest_field)
    if actual_digest != expected_digest:
        return SidecarResult(
            schema=schema_name,
            path=str(path),
            ok=False,
            message=(
                f"{digest_field} mismatch; expected {expected_digest}; "
                f"actual {actual_digest}"
            ),
        )
    if schema_name == "referenced_optimized_kernel_digest":
        content_digest = _json_report_digest(payload, digest_field=digest_field)
        if content_digest != expected_digest:
            return SidecarResult(
                schema=schema_name,
                path=str(path),
                ok=False,
                message=(
                    f"content digest mismatch; expected {expected_digest}; "
                    f"actual {content_digest}"
                ),
            )
    return SidecarResult(
        schema=schema_name,
        path=str(path),
        ok=True,
        message="ok",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate launch-governing modded-nanogpt attempt artifacts."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    report = validate_attempt_dir(args.result_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
    return 0 if report["ok"] else 21


if __name__ == "__main__":
    raise SystemExit(main())
