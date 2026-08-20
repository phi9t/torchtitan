#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Certify optimized kernels for modded-nanogpt B200 full-launch gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import textwrap
import time
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.modded_nanogpt_b200 import cli_guard, preflight


SCHEMA_VERSION = 1
SCHEMA_NAME = "optimized_kernel_report"
WRAPPER = "experiments/modded_nanogpt_b200/certify_optimized_kernels.sh"
DEFAULT_ENVIRONMENT_CLASS = "torchtitan-rootfs-b200"
SCHEMA_PATH = Path(
    "experiments/modded_nanogpt_b200/optimized_kernel_report.schema.json"
)
TAIL_CHARS = 4000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(_json_bytes(data))
    tmp.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _source_commit(source: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _source_digest(path: Path) -> dict[str, str] | None:
    if not path.exists() or not path.is_file():
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def _tail(text: str) -> str:
    return text[-TAIL_CHARS:]


def _blocker(failure_class: str, message: str) -> dict[str, str]:
    return {"class": failure_class, "message": message}


def _status_launch_eligible(
    *, support_status: str, build_status: str, smoke_status: str
) -> bool:
    return (
        support_status in {"supported", "absent", "installed_unselected"}
        and build_status in {"passed", "not_required"}
        and smoke_status in {"passed", "not_required"}
    )


def make_row(
    *,
    name: str,
    kind: str,
    role: str,
    requested: bool,
    selected_for_launch: bool = False,
    required_for_selected_launch: bool = False,
    diagnostic_only: bool = False,
    support_status: str,
    build_status: str,
    smoke_status: str,
    failure_class: str | None = None,
    blockers: list[dict[str, str]] | None = None,
    detail: dict[str, Any] | None = None,
    packages: dict[str, str | None] | None = None,
    artifact_paths: list[str] | None = None,
    stdout_tail: str | None = None,
) -> dict[str, Any]:
    row_blockers = list(blockers or [])
    launch_eligible = _status_launch_eligible(
        support_status=support_status,
        build_status=build_status,
        smoke_status=smoke_status,
    )
    row: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "role": role,
        "requested": requested,
        "selected_for_launch": selected_for_launch,
        "required_for_selected_launch": required_for_selected_launch,
        "diagnostic_only": diagnostic_only,
        "support_status": support_status,
        "build_status": build_status,
        "smoke_status": smoke_status,
        "launch_eligible": launch_eligible,
        "blockers": row_blockers,
    }
    if failure_class is not None:
        row["failure_class"] = failure_class
    if detail is not None:
        row["detail"] = detail
    if packages is not None:
        row["packages"] = packages
    if artifact_paths is not None:
        row["artifact_paths"] = artifact_paths
    if stdout_tail is not None:
        row["stdout_tail"] = stdout_tail
    return row


def _row_from_probe(
    *,
    name: str,
    kind: str,
    role: str,
    requested: bool,
    selected_for_launch: bool = False,
    required_for_selected_launch: bool = False,
    diagnostic_only: bool = False,
    probe,
) -> dict[str, Any]:
    try:
        detail = probe()
    except preflight.CheckFailure as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        failure_class = str(
            detail.get("failure_class") or f"{name.replace('.', '_')}_failed"
        )
        return make_row(
            name=name,
            kind=kind,
            role=role,
            requested=requested,
            selected_for_launch=selected_for_launch,
            required_for_selected_launch=required_for_selected_launch,
            diagnostic_only=diagnostic_only,
            support_status="supported",
            build_status="passed",
            smoke_status="failed",
            failure_class=failure_class,
            blockers=[_blocker(failure_class, str(exc))],
            detail=detail,
            stdout_tail=_tail(str(detail.get("stdout", ""))) if detail else None,
        )
    except Exception as exc:  # noqa: BLE001
        failure_class = f"{name.replace('.', '_')}_runtime_error"
        return make_row(
            name=name,
            kind=kind,
            role=role,
            requested=requested,
            selected_for_launch=selected_for_launch,
            required_for_selected_launch=required_for_selected_launch,
            diagnostic_only=diagnostic_only,
            support_status="supported",
            build_status="unknown",
            smoke_status="failed",
            failure_class=failure_class,
            blockers=[_blocker(failure_class, f"{type(exc).__name__}: {exc}")],
        )
    if not isinstance(detail, dict):
        detail = {"value": detail}
    return make_row(
        name=name,
        kind=kind,
        role=role,
        requested=requested,
        selected_for_launch=selected_for_launch,
        required_for_selected_launch=required_for_selected_launch,
        diagnostic_only=diagnostic_only,
        support_status=str(detail.get("support_status", "supported")),
        build_status=str(detail.get("build_status", "passed")),
        smoke_status=str(detail.get("smoke_status", "passed")),
        failure_class=detail.get("failure_class"),
        blockers=detail.get("blockers")
        if isinstance(detail.get("blockers"), list)
        else [],
        detail=detail.get("detail", detail),
        packages=detail.get("packages")
        if isinstance(detail.get("packages"), dict)
        else None,
    )


def _unsupported_row(
    name: str,
    *,
    kind: str,
    role: str,
    requested: bool,
    selected_for_launch: bool = False,
    required_for_selected_launch: bool = False,
    reason: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_row(
        name=name,
        kind=kind,
        role=role,
        requested=requested,
        selected_for_launch=selected_for_launch,
        required_for_selected_launch=required_for_selected_launch,
        diagnostic_only=not selected_for_launch and not required_for_selected_launch,
        support_status="unsupported",
        build_status="not_run",
        smoke_status="not_run",
        failure_class="unsupported_unselected",
        blockers=[_blocker("unsupported_unselected", reason)]
        if selected_for_launch or required_for_selected_launch
        else [],
        detail={"reason": reason, **(detail or {})},
    )


def _probe_attention_fa2() -> dict[str, Any]:
    preflight.check_fa2_smoke()
    return {
        "support_status": "supported",
        "build_status": "passed",
        "smoke_status": "passed",
        "packages": {"flash-attn": _package_version("flash-attn")},
        "detail": {"smoke": "causal_bf16_varlen_window"},
    }


def _probe_mlp_triton(source: Path) -> dict[str, Any]:
    detail = preflight._run_triton_mlp_smoke(source)
    return {
        "support_status": "supported",
        "build_status": "passed",
        "smoke_status": "passed",
        "detail": detail,
    }


def _probe_mlp_torch(source: Path) -> dict[str, Any]:
    detail = preflight._run_torch_mlp_smoke(source)
    return {
        "support_status": "supported",
        "build_status": "not_required",
        "smoke_status": "passed",
        "detail": {
            **detail,
            "full_mode_policy": "diagnostic_fallback_only_unless_promoted",
        },
    }


def _probe_torch_scaled_mm() -> dict[str, Any]:
    preflight.check_torch_primitives()
    return {
        "support_status": "supported",
        "build_status": "not_required",
        "smoke_status": "passed",
        "detail": {"primitive": "torch._scaled_mm fp8"},
    }


def _probe_inductor_cache() -> dict[str, Any]:
    cache_dir = Path(
        os.environ.get(
            "TORCHINDUCTOR_CACHE_DIR",
            "experiments/modded_nanogpt_b200/results/.torchinductor_cache",
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "support_status": "supported",
        "build_status": "passed",
        "smoke_status": "not_required",
        "detail": {"cache_dir": str(cache_dir), "exists": cache_dir.exists()},
    }


def _probe_triton_tensor_descriptor() -> dict[str, Any]:
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor

        has_descriptor = TensorDescriptor is not None
    except Exception:
        has_descriptor = False
    import triton

    return {
        "support_status": "supported" if has_descriptor else "unknown",
        "build_status": "not_required",
        "smoke_status": "passed" if has_descriptor else "unknown",
        "detail": {
            "triton": triton.__version__,
            "import": "triton.tools.tensor_descriptor.TensorDescriptor",
            "has_tensor_descriptor": has_descriptor,
        },
        "blockers": []
        if has_descriptor
        else [
            {
                "class": "triton_tensor_descriptor_unknown",
                "message": "could not confirm triton tensor descriptor API",
            }
        ],
    }


def _probe_nccl(expected_gpus: int) -> dict[str, Any]:
    preflight.check_nccl(expected_gpus)
    return {
        "support_status": "supported",
        "build_status": "not_required",
        "smoke_status": "passed",
        "detail": {"expected_gpus": expected_gpus, "collective": "all_reduce"},
    }


def _probe_source_triton(source: Path) -> dict[str, Any]:
    digest = _source_digest(source / "triton_kernels.py")
    if digest is None:
        return {
            "support_status": "unsupported",
            "build_status": "failed",
            "smoke_status": "not_run",
            "failure_class": "missing_source_triton_kernels",
            "blockers": [
                {
                    "class": "missing_source_triton_kernels",
                    "message": f"missing source-local triton_kernels.py under {source}",
                }
            ],
        }
    old_path = list(sys.path)
    sys.path.insert(0, str(source))
    try:
        importlib.invalidate_caches()
        module = importlib.import_module("triton_kernels")
    finally:
        sys.path[:] = old_path
    return {
        "support_status": "supported",
        "build_status": "passed",
        "smoke_status": "passed",
        "detail": {
            "module_file": str(getattr(module, "__file__", "")),
            "digest": digest,
        },
    }


def _probe_source_dc_triton(source: Path) -> dict[str, Any]:
    digest = _source_digest(source / "triton_kernels.py")
    if digest is None:
        return {
            "support_status": "unsupported",
            "build_status": "failed",
            "smoke_status": "not_run",
            "failure_class": "missing_source_dc_triton_kernels",
            "blockers": [
                {
                    "class": "missing_source_dc_triton_kernels",
                    "message": f"missing source-local dynamic-correction kernels under {source}",
                }
            ],
        }
    text = (source / "triton_kernels.py").read_text()
    has_dynamic_correction = (
        "MODDED_NANOGPT_CE_COMPUTE_CAPABILITY" in text or "dynamic" in text.lower()
    )
    return {
        "support_status": "supported" if has_dynamic_correction else "unknown",
        "build_status": "passed" if has_dynamic_correction else "not_run",
        "smoke_status": "passed" if has_dynamic_correction else "not_run",
        "detail": {
            "digest": digest,
            "has_dynamic_correction_marker": has_dynamic_correction,
        },
        "blockers": []
        if has_dynamic_correction
        else [
            {
                "class": "source_dc_triton_unknown",
                "message": "source-local dynamic-correction marker was not found",
            }
        ],
    }


def _gpu_arch_evidence(expected_gpus: int) -> list[dict[str, Any]]:
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    count = min(torch.cuda.device_count(), expected_gpus)
    return [
        {
            "index": idx,
            "name": torch.cuda.get_device_name(idx),
            "compute_capability": list(torch.cuda.get_device_capability(idx)),
        }
        for idx in range(count)
    ]


def _forbidden_component_rows() -> list[dict[str, Any]]:
    return _forbidden_component_rows_for_source(None)


def _source_texts(source: Path) -> list[tuple[Path, str]]:
    texts = []
    for relative in ("train_gpt.py", "train_gpt_medium.py", "triton_kernels.py"):
        path = source / relative
        if path.exists() and path.is_file():
            texts.append((path, path.read_text()))
    return texts


def _source_imports_flashinfer(source: Path) -> list[str]:
    matches = []
    for path, text in _source_texts(source):
        if "flashinfer" in text or "flash_infer" in text or "flash-infer" in text:
            matches.append(str(path))
    return matches


def _forbidden_component_rows_for_source(source: Path | None) -> list[dict[str, Any]]:
    rows = []
    source_matches = _source_imports_flashinfer(source) if source is not None else []
    try:
        module = importlib.import_module("flashinfer")
    except ImportError:
        rows.append(
            make_row(
                name="forbidden.flashinfer",
                kind="package",
                role="forbidden_component",
                requested=True,
                required_for_selected_launch=True,
                support_status="absent",
                build_status="not_required",
                smoke_status="not_required",
                detail={"package": "flashinfer", "status": "not_importable"},
            )
        )
    else:
        if not source_matches:
            rows.append(
                make_row(
                    name="forbidden.flashinfer",
                    kind="package",
                    role="forbidden_component",
                    requested=True,
                    required_for_selected_launch=True,
                    support_status="installed_unselected",
                    build_status="not_required",
                    smoke_status="not_required",
                    detail={
                        "package": "flashinfer",
                        "module_file": str(getattr(module, "__file__", "")),
                        "source_import_matches": [],
                        "policy": "installed but absent from selected source manifest",
                    },
                )
            )
            return rows
        rows.append(
            make_row(
                name="forbidden.flashinfer",
                kind="package",
                role="forbidden_component",
                requested=True,
                required_for_selected_launch=True,
                support_status="forbidden_present",
                build_status="failed",
                smoke_status="not_run",
                failure_class="forbidden_component_present",
                blockers=[
                    {
                        "class": "forbidden_component_present",
                        "message": "flashinfer is selected by the current source manifest",
                    }
                ],
                detail={
                    "package": "flashinfer",
                    "module_file": str(getattr(module, "__file__", "")),
                    "source_import_matches": source_matches,
                },
            )
        )
    return rows


def _attention_rows(args: Any) -> list[dict[str, Any]]:
    selected = args.attention_backend
    rows: list[dict[str, Any]] = []
    if selected == "fa2":
        rows.append(
            _row_from_probe(
                name="attention.fa2",
                kind="attention_backend",
                role="selected_attention",
                requested=True,
                selected_for_launch=True,
                required_for_selected_launch=True,
                probe=_probe_attention_fa2,
            )
        )
    else:
        rows.append(
            _unsupported_row(
                "attention.fa2",
                kind="attention_backend",
                role="unselected_attention",
                requested=False,
                reason="FA2 is not selected for this certification tuple",
                detail={"packages": {"flash-attn": _package_version("flash-attn")}},
            )
        )

    rows.append(
        _unsupported_row(
            "attention.fa3",
            kind="attention_backend",
            role="selected_attention" if selected == "fa3" else "unselected_attention",
            requested=selected == "fa3",
            selected_for_launch=selected == "fa3",
            required_for_selected_launch=selected == "fa3",
            reason=(
                "FA3 kernels path remains unsupported on B200 until a provider "
                "kernel image is certified"
            ),
            detail={"package": _package_version("kernels")},
        )
    )
    rows.append(
        _unsupported_row(
            "attention.fa4",
            kind="attention_backend",
            role="unselected_attention",
            requested=False,
            reason="no FA4 source, profile, package, or provider is selected",
        )
    )
    rows.append(
        _unsupported_row(
            "attention.flex",
            kind="attention_backend",
            role="unselected_attention",
            requested=selected == "flex",
            selected_for_launch=selected == "flex",
            required_for_selected_launch=selected == "flex",
            reason="flex attention is diagnostic-only for this ticket",
        )
    )
    rows.append(
        _unsupported_row(
            "attention.torch_sdpa",
            kind="attention_backend",
            role="unselected_attention",
            requested=False,
            reason="PyTorch SDPA is recorded as an unselected fallback, not a full-mode provider",
        )
    )
    return rows


def _mlp_rows(args: Any) -> list[dict[str, Any]]:
    selected = args.mlp_backend
    rows: list[dict[str, Any]] = []
    if selected == "triton":
        rows.append(
            _row_from_probe(
                name="mlp.triton",
                kind="mlp_backend",
                role="selected_mlp",
                requested=True,
                selected_for_launch=True,
                required_for_selected_launch=True,
                probe=lambda: _probe_mlp_triton(args.source),
            )
        )
    else:
        rows.append(
            _unsupported_row(
                "mlp.triton",
                kind="mlp_backend",
                role="unselected_mlp",
                requested=False,
                reason="Triton MLP is not selected for this certification tuple",
            )
        )
    if selected == "torch":
        rows.append(
            _row_from_probe(
                name="mlp.torch",
                kind="mlp_backend",
                role="selected_mlp",
                requested=True,
                selected_for_launch=True,
                required_for_selected_launch=True,
                probe=lambda: _probe_mlp_torch(args.source),
            )
        )
    else:
        rows.append(
            _row_from_probe(
                name="mlp.torch",
                kind="mlp_backend",
                role="diagnostic_fallback",
                requested=False,
                diagnostic_only=True,
                probe=lambda: _probe_mlp_torch(args.source),
            )
        )
    return rows


def _prereq_rows(args: Any) -> list[dict[str, Any]]:
    return [
        _row_from_probe(
            name="prereq.torch_scaled_mm_fp8",
            kind="prerequisite",
            role="core_prerequisite",
            requested=True,
            required_for_selected_launch=True,
            probe=_probe_torch_scaled_mm,
        ),
        _row_from_probe(
            name="prereq.torchinductor_cache",
            kind="prerequisite",
            role="core_prerequisite",
            requested=True,
            required_for_selected_launch=True,
            probe=_probe_inductor_cache,
        ),
        _row_from_probe(
            name="prereq.triton_tensor_descriptor",
            kind="prerequisite",
            role="core_prerequisite",
            requested=True,
            required_for_selected_launch=True,
            probe=_probe_triton_tensor_descriptor,
        ),
        _row_from_probe(
            name="prereq.nccl_all_reduce",
            kind="prerequisite",
            role="core_prerequisite",
            requested=True,
            required_for_selected_launch=True,
            probe=lambda: _probe_nccl(args.expected_gpus),
        ),
    ]


def _source_rows(args: Any) -> list[dict[str, Any]]:
    return [
        _row_from_probe(
            name="source_triton_kernels",
            kind="source_kernel",
            role="selected_source_kernel",
            requested=True,
            required_for_selected_launch=True,
            probe=lambda: _probe_source_triton(args.source),
        ),
        _row_from_probe(
            name="source_dc_triton_kernels",
            kind="source_kernel",
            role="selected_source_kernel",
            requested=True,
            required_for_selected_launch=True,
            probe=lambda: _probe_source_dc_triton(args.source),
        ),
    ]


def _report_blockers(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    blockers = []
    for row in rows:
        if not (
            row.get("selected_for_launch") is True
            or row.get("required_for_selected_launch") is True
        ):
            continue
        if row.get("launch_eligible") is True:
            continue
        row_blockers = row.get("blockers")
        if isinstance(row_blockers, list) and row_blockers:
            for blocker in row_blockers:
                if isinstance(blocker, dict):
                    blockers.append(
                        {
                            "row": str(row["name"]),
                            "class": str(
                                blocker.get(
                                    "class", row.get("failure_class", "blocked")
                                )
                            ),
                            "message": str(
                                blocker.get("message", "row is not launch eligible")
                            ),
                        }
                    )
        else:
            blockers.append(
                {
                    "row": str(row["name"]),
                    "class": str(row.get("failure_class", "not_launch_eligible")),
                    "message": f"{row['name']} is not launch eligible",
                }
            )
    return blockers


def build_report(args: Any) -> dict[str, Any]:
    preflight.check_rootfs()
    rootfs_detail = preflight.collect_rootfs_detail()
    environment = preflight.collect_environment_detail()
    rows = []
    rows.extend(_attention_rows(args))
    rows.extend(_mlp_rows(args))
    rows.extend(_source_rows(args))
    rows.extend(_prereq_rows(args))
    rows.extend(_forbidden_component_rows_for_source(args.source))

    blockers = _report_blockers(rows)
    schema_abs = REPO_ROOT / SCHEMA_PATH
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_digest": {
            "path": str(SCHEMA_PATH),
            "sha256": sha256_file(schema_abs),
        },
        "generated_at_epoch": time.time(),
        "launch_eligible": not blockers,
        "selected_tuple": {
            "attention_backend": args.attention_backend,
            "mlp_backend": args.mlp_backend,
        },
        "classification": {
            "run_id": args.run_id,
            "attempt_id": args.attempt_id,
            "environment_class": args.environment_class,
        },
        "environment": environment,
        "rootfs": rootfs_detail,
        "source": {
            "path": str(args.source),
            "commit": _source_commit(args.source),
            "digests": [
                item
                for item in (
                    _source_digest(args.source / "train_gpt.py"),
                    _source_digest(args.source / "triton_kernels.py"),
                )
                if item is not None
            ],
        },
        "gpu_arch_evidence": _gpu_arch_evidence(args.expected_gpus),
        "selected_backend_env": {
            "MODDED_NANOGPT_ATTN_BACKEND": args.attention_backend,
            "MODDED_NANOGPT_MLP_BACKEND": args.mlp_backend,
            "MODDED_NANOGPT_CE_COMPUTE_CAPABILITY": os.environ.get(
                "MODDED_NANOGPT_CE_COMPUTE_CAPABILITY"
            ),
        },
        "cache_directories": {
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
        },
        "rows": rows,
        "blockers": blockers,
    }


def _add_report_digest(report: dict[str, Any]) -> dict[str, Any]:
    report = dict(report)
    report.setdefault("schema_name", SCHEMA_NAME)
    report.setdefault(
        "schema_digest",
        {
            "path": str(SCHEMA_PATH),
            "sha256": sha256_file(REPO_ROOT / SCHEMA_PATH),
        },
    )
    report.pop("report_digest", None)
    report["report_digest"] = {
        "sha256": hashlib.sha256(_json_bytes(report)).hexdigest()
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=["fa2", "fa3", "fa4", "flex", "torch_sdpa"],
        required=True,
    )
    parser.add_argument("--mlp-backend", choices=["triton", "torch"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--environment-class", default=DEFAULT_ENVIRONMENT_CLASS)
    parser.add_argument("--expected-gpus", type=int, default=2)
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.run_id is None:
        args.run_id = f"optimized_kernel_cert_{timestamp}"
    if args.attempt_id is None:
        args.attempt_id = f"{args.run_id}_attempt_001"
    return args


def main(*, enforce_rootfs: bool = False) -> int:
    if enforce_rootfs:
        guard_exit = cli_guard.guard_rootfs_cli(WRAPPER)
        if guard_exit is not None:
            return guard_exit
    args = parse_args()
    try:
        report = _add_report_digest(build_report(args))
        _write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["launch_eligible"] else 21
    except Exception as exc:  # noqa: BLE001
        failure = {
            "schema_version": SCHEMA_VERSION,
            "schema_name": SCHEMA_NAME,
            "launch_eligible": False,
            "selected_tuple": {
                "attention_backend": getattr(args, "attention_backend", None),
                "mlp_backend": getattr(args, "mlp_backend", None),
            },
            "classification": {
                "run_id": getattr(args, "run_id", None),
                "attempt_id": getattr(args, "attempt_id", None),
                "environment_class": getattr(args, "environment_class", None),
            },
            "rows": [],
            "blockers": [
                {
                    "row": "certifier",
                    "class": "certifier_runtime_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
        try:
            _write_json_atomic(args.output, _add_report_digest(failure))
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        print(
            "\noptimized kernel certification failed:\n"
            + textwrap.indent(f"{type(exc).__name__}: {exc}", "  "),
            file=sys.stderr,
        )
        return 21


if __name__ == "__main__":
    raise SystemExit(main(enforce_rootfs=True))
