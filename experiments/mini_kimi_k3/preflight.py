#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preflight readiness report for the Mini Kimi K3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mini_kimi_k3.evidence import (  # noqa: E402
    inspect_launch_evidence_bundle,
)
from torchtitan.config import ConfigManager  # noqa: E402
from torchtitan.experiments.mini_kimi_k3.model_contract import (  # noqa: E402
    compare_with_first_party_oracle,
    estimate_parameter_budget,
    mini_k3_r1_config,
    to_kimi_linear_config_dict,
)
from torchtitan.experiments.mini_kimi_k3.recipe import (  # noqa: E402
    mini_k3_r1_launch_recipe,
)
from torchtitan.experiments.mini_kimi_k3.token_shards import (  # noqa: E402
    load_manifest,
    TokenShardLoader,
)

SCHEMA_VERSION = 1
ROOTFS_REPO_CWD = "/workspace/torchtitan"


class CheckFailure(Exception):  # noqa: N818
    def __init__(self, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def build_readiness_report(
    *,
    mode: str,
    token_manifest: Path | None = None,
    tokens_dir: Path | None = None,
    seq_len: int = 4096,
    tiny_smoke_report: Path | None = None,
    evidence_results_root: Path | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    oracle_root: Path | None = None,
    oracle_r1_config: Path | None = None,
    forward_oracle_report: Path | None = None,
    launch_backend_report: Path | None = None,
    r1_training_smoke_report: Path | None = None,
    r1_corpus_plan_audit_report: Path | None = None,
    corpus_disk_readiness_report: Path | None = None,
    stage1_source_resolution_report: Path | None = None,
    stage1_input_inspection_report: Path | None = None,
    stage1_remote_readiness_report: Path | None = None,
) -> dict[str, Any]:
    rootfs_runner = REPO_ROOT / "experiments" / "mini_kimi_k3" / "run.sh"
    token_shards = (
        REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "token_shards.py"
    )
    token_shard_tests = (
        REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_token_shards.py"
    )
    model_contract = (
        REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "model_contract.py"
    )
    model_contract_tests = (
        REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_model_contract.py"
    )
    activation = (
        REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "activation.py"
    )
    activation_tests = (
        REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_activation.py"
    )
    router = REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "router.py"
    router_tests = REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_router.py"
    moe = REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "moe.py"
    moe_tests = REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_moe.py"
    kda = REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "kda.py"
    kda_tests = REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_kda.py"
    mla = REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "mla.py"
    mla_tests = REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_mla.py"
    model = REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "model.py"
    model_tests = REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_model.py"
    recipe_contract = (
        REPO_ROOT / "torchtitan" / "experiments" / "mini_kimi_k3" / "recipe.py"
    )
    recipe_contract_tests = (
        REPO_ROOT / "tests" / "unit_tests" / "test_mini_kimi_k3_recipe.py"
    )
    corpus_gate = _corpus_manifest_gate(
        mode=mode,
        token_manifest=token_manifest,
        tokens_dir=tokens_dir,
        seq_len=seq_len,
        r1_corpus_plan_audit_report=r1_corpus_plan_audit_report,
        corpus_disk_readiness_report=corpus_disk_readiness_report,
        stage1_source_resolution_report=stage1_source_resolution_report,
        stage1_input_inspection_report=stage1_input_inspection_report,
        stage1_remote_readiness_report=stage1_remote_readiness_report,
    )
    model_gate = _model_fidelity_gate(
        model_contract=model_contract,
        model_contract_tests=model_contract_tests,
        activation=activation,
        activation_tests=activation_tests,
        router=router,
        router_tests=router_tests,
        moe=moe,
        moe_tests=moe_tests,
        kda=kda,
        kda_tests=kda_tests,
        mla=mla,
        mla_tests=mla_tests,
        model=model,
        model_tests=model_tests,
        oracle_root=oracle_root,
        oracle_r1_config=oracle_r1_config,
        forward_oracle_report=forward_oracle_report,
        launch_backend_report=launch_backend_report,
    )
    launch_evidence_gate = _launch_evidence_bundle_gate(
        evidence_results_root=evidence_results_root,
        run_id=run_id,
        attempt_id=attempt_id,
        mode=mode,
    )
    gates = [
        _rootfs_runner_gate(rootfs_runner),
        _gate(
            "token_shard_loader",
            "pass"
            if token_shards.is_file() and token_shard_tests.is_file()
            else "missing",
            "experiment-owned uint32 token-shard loader and tests exist",
            evidence=[
                str(token_shards.relative_to(REPO_ROOT)),
                str(token_shard_tests.relative_to(REPO_ROOT)),
            ],
        ),
        _trainable_config_gate(
            mode=mode,
            recipe_contract=recipe_contract,
            recipe_contract_tests=recipe_contract_tests,
            model_gate=model_gate,
            launch_backend_report=launch_backend_report,
            r1_training_smoke_report=r1_training_smoke_report,
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
            seq_len=seq_len,
        ),
        model_gate,
        corpus_gate,
        _tiny_training_smoke_gate(
            mode=mode,
            tiny_smoke_report=tiny_smoke_report,
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
            seq_len=seq_len,
        ),
        launch_evidence_gate,
    ]
    status = (
        "ready"
        if all(gate["status"] in {"pass", "not_applicable"} for gate in gates)
        else "blocked"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_preflight",
        "mode": mode,
        "status": status,
        "rootfs": {
            "marker": os.environ.get("TORCHTITAN_IN_ROOTFS"),
            "cwd": str(Path.cwd()),
            "workspace_sentinel": "scripts/rootfs/enter_rootfs.sh",
        },
        "gates": gates,
    }


def run_preflight(
    *,
    mode: str,
    report_path: Path,
    token_manifest: Path | None = None,
    tokens_dir: Path | None = None,
    seq_len: int = 4096,
    tiny_smoke_report: Path | None = None,
    evidence_results_root: Path | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    oracle_root: Path | None = None,
    oracle_r1_config: Path | None = None,
    forward_oracle_report: Path | None = None,
    launch_backend_report: Path | None = None,
    r1_training_smoke_report: Path | None = None,
    r1_corpus_plan_audit_report: Path | None = None,
    corpus_disk_readiness_report: Path | None = None,
    stage1_source_resolution_report: Path | None = None,
    stage1_input_inspection_report: Path | None = None,
    stage1_remote_readiness_report: Path | None = None,
) -> dict[str, Any]:
    report = build_readiness_report(
        mode=mode,
        token_manifest=token_manifest,
        tokens_dir=tokens_dir,
        seq_len=seq_len,
        tiny_smoke_report=tiny_smoke_report,
        evidence_results_root=evidence_results_root,
        run_id=run_id,
        attempt_id=attempt_id,
        oracle_root=oracle_root,
        oracle_r1_config=oracle_r1_config,
        forward_oracle_report=forward_oracle_report,
        launch_backend_report=launch_backend_report,
        r1_training_smoke_report=r1_training_smoke_report,
        r1_corpus_plan_audit_report=r1_corpus_plan_audit_report,
        corpus_disk_readiness_report=corpus_disk_readiness_report,
        stage1_source_resolution_report=stage1_source_resolution_report,
        stage1_input_inspection_report=stage1_input_inspection_report,
        stage1_remote_readiness_report=stage1_remote_readiness_report,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "ready":
        raise CheckFailure(f"Mini Kimi K3 {mode} launch is not ready", report=report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a Mini Kimi K3 launch readiness preflight report."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "tiny-plumbing"),
        default="full",
        help="Readiness target to evaluate.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/results/preflight.json"),
        help="Path to write the JSON readiness report.",
    )
    parser.add_argument(
        "--token-manifest",
        type=Path,
        help="JSON token-shard manifest to validate for the corpus gate.",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        help="Directory containing manifest source subdirectories and shard files.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Sequence length used to validate shard usability.",
    )
    parser.add_argument(
        "--tiny-smoke-report",
        type=Path,
        help="JSON report produced by experiments/mini_kimi_k3/run.sh tiny-smoke.",
    )
    parser.add_argument(
        "--evidence-results-root",
        type=Path,
        help="Results root containing the initialized run-attempt evidence bundle.",
    )
    parser.add_argument(
        "--run-id",
        help="Run id for the initialized evidence bundle.",
    )
    parser.add_argument(
        "--attempt-id",
        help="Attempt id for the initialized evidence bundle.",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        help="Optional first-party Mini-K3 checkout root for model-contract comparison.",
    )
    parser.add_argument(
        "--oracle-r1-config",
        type=Path,
        help="Optional first-party r1 KimiLinearConfig JSON for model-contract comparison.",
    )
    parser.add_argument(
        "--forward-oracle-report",
        type=Path,
        help="Optional first-party forward comparison report for Mini-K3 r1 logits.",
    )
    parser.add_argument(
        "--launch-backend-report",
        type=Path,
        help="Optional reviewed launch-backend report for Mini-K3 r1 training.",
    )
    parser.add_argument(
        "--r1-training-smoke-report",
        type=Path,
        help="Optional run_train.sh smoke report for Mini-K3 r1 Trainer execution.",
    )
    parser.add_argument(
        "--r1-corpus-plan-audit-report",
        type=Path,
        help="Optional audit-r1-corpus-plan report for source-level corpus deficits.",
    )
    parser.add_argument(
        "--corpus-disk-readiness-report",
        type=Path,
        help="Optional corpus-disk-readiness report for raw shard filesystem capacity.",
    )
    parser.add_argument(
        "--stage1-source-resolution-report",
        type=Path,
        help="Optional probe-stage1-sources report for Stage 1 source resolution.",
    )
    parser.add_argument(
        "--stage1-input-inspection-report",
        type=Path,
        help="Optional inspect-stage1-inputs report for local Stage 1 inputs.",
    )
    parser.add_argument(
        "--stage1-remote-readiness-report",
        type=Path,
        help="Optional stage1-remote-readiness report for remote corpus access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_preflight(
            mode=args.mode,
            report_path=args.report,
            token_manifest=args.token_manifest,
            tokens_dir=args.tokens_dir,
            seq_len=args.seq_len,
            tiny_smoke_report=args.tiny_smoke_report,
            evidence_results_root=args.evidence_results_root,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            oracle_root=args.oracle_root,
            oracle_r1_config=args.oracle_r1_config,
            forward_oracle_report=args.forward_oracle_report,
            launch_backend_report=args.launch_backend_report,
            r1_training_smoke_report=args.r1_training_smoke_report,
            r1_corpus_plan_audit_report=args.r1_corpus_plan_audit_report,
            corpus_disk_readiness_report=args.corpus_disk_readiness_report,
            stage1_source_resolution_report=args.stage1_source_resolution_report,
            stage1_input_inspection_report=args.stage1_input_inspection_report,
            stage1_remote_readiness_report=args.stage1_remote_readiness_report,
        )
    except CheckFailure as exc:
        print(str(exc), file=sys.stderr)
        print(f"wrote preflight report: {args.report}", file=sys.stderr)
        return 21
    print(f"Mini Kimi K3 {args.mode} launch is ready", file=sys.stderr)
    print(f"wrote preflight report: {args.report}", file=sys.stderr)
    return 0


def _gate(
    name: str,
    status: str,
    description: str,
    *,
    evidence: str | list[str] | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "name": name,
        "status": status,
        "description": description,
    }
    if evidence is not None:
        gate["evidence"] = evidence
    if detail is not None:
        gate["detail"] = detail
    return gate


def _rootfs_runner_gate(rootfs_runner: Path) -> dict[str, Any]:
    if not rootfs_runner.is_file():
        return _gate(
            "rootfs_runner",
            "missing",
            "rootfs-aware top-level runner is missing",
            evidence=str(rootfs_runner.relative_to(REPO_ROOT)),
        )
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        return _gate(
            "rootfs_runner",
            "fail",
            "rootfs-aware top-level runner exists, but preflight is not inside rootfs",
            evidence=str(rootfs_runner.relative_to(REPO_ROOT)),
            detail="not running inside the TorchTitan rootfs",
        )
    return _gate(
        "rootfs_runner",
        "pass",
        "rootfs-aware top-level runner exists and preflight is inside rootfs",
        evidence=str(rootfs_runner.relative_to(REPO_ROOT)),
    )


def _corpus_manifest_gate(
    *,
    mode: str,
    token_manifest: Path | None,
    tokens_dir: Path | None,
    seq_len: int,
    r1_corpus_plan_audit_report: Path | None,
    corpus_disk_readiness_report: Path | None,
    stage1_source_resolution_report: Path | None,
    stage1_input_inspection_report: Path | None,
    stage1_remote_readiness_report: Path | None,
) -> dict[str, Any]:
    if token_manifest is None:
        return _gate(
            "real_corpus_manifest",
            "missing",
            "no real r1 corpus/token manifest or tokenizer fingerprint exists",
        )
    if tokens_dir is None:
        return _gate(
            "real_corpus_manifest",
            "fail",
            "--tokens-dir is required when --token-manifest is provided",
        )

    try:
        manifest = load_manifest(token_manifest)
        loader = TokenShardLoader(
            manifest,
            tokens_dir,
            seq_len=seq_len,
            rank=0,
            world_size=1,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return _gate(
            "real_corpus_manifest",
            "fail",
            "token manifest failed validation",
            evidence={"manifest": str(token_manifest), "tokens_dir": str(tokens_dir)},
            detail=str(exc),
        )

    required_tokens = (
        mini_k3_r1_launch_recipe().target_tokens if mode == "full" else seq_len
    )
    metadata_evidence: dict[str, Any] = {}
    metadata_errors: list[str] = []
    if mode == "full":
        corpus_route = _full_corpus_route(manifest)
        requires_stage1_input_inspection = _has_local_stage1_route(manifest)
        metadata_evidence["corpus_route"] = corpus_route
        metadata_evidence[
            "requires_stage1_input_inspection"
        ] = requires_stage1_input_inspection
        _validate_full_corpus_metadata(
            manifest,
            tokens_dir=tokens_dir,
            evidence=metadata_evidence,
            errors=metadata_errors,
        )
        audit_evidence, audit_error = _r1_corpus_plan_audit_evidence(
            r1_corpus_plan_audit_report,
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
            available_tokens=loader.assigned_token_count,
            required_tokens=required_tokens,
        )
        if audit_evidence is not None:
            metadata_evidence["r1_corpus_plan_audit"] = audit_evidence
        elif r1_corpus_plan_audit_report is None:
            metadata_errors.append("corpus plan audit report is required for full r1")
        if audit_error is not None:
            metadata_errors.append(audit_error)
        disk_evidence, disk_error = _corpus_disk_readiness_evidence(
            corpus_disk_readiness_report,
            required_tokens=required_tokens,
        )
        if disk_evidence is not None:
            metadata_evidence["corpus_disk_readiness"] = disk_evidence
        if disk_error is not None:
            metadata_errors.append(disk_error)
        (
            source_resolution_evidence,
            source_resolution_error,
        ) = _stage1_source_resolution_evidence(stage1_source_resolution_report)
        if source_resolution_evidence is not None:
            metadata_evidence["stage1_source_resolution"] = source_resolution_evidence
        elif stage1_source_resolution_report is None:
            metadata_errors.append(
                "stage1 source resolution report is required for full r1"
            )
        if source_resolution_error is not None:
            metadata_errors.append(source_resolution_error)
        (
            input_inspection_evidence,
            input_inspection_error,
        ) = _stage1_input_inspection_evidence(
            stage1_input_inspection_report,
            required_tokens=required_tokens,
            required=requires_stage1_input_inspection,
        )
        if input_inspection_evidence is not None:
            metadata_evidence["stage1_input_inspection"] = input_inspection_evidence
        if input_inspection_error is not None:
            metadata_errors.append(input_inspection_error)
        (
            remote_readiness_evidence,
            remote_readiness_error,
        ) = _stage1_remote_readiness_evidence(stage1_remote_readiness_report)
        if remote_readiness_evidence is not None:
            if corpus_route == "local_or_imported":
                remote_readiness_evidence = {
                    **remote_readiness_evidence,
                    "status": "not_required",
                    "route": corpus_route,
                }
                remote_readiness_error = None
            metadata_evidence["stage1_remote_readiness"] = remote_readiness_evidence
        if remote_readiness_error is not None:
            metadata_errors.append(remote_readiness_error)

    evidence: dict[str, Any] = {
        "manifest": str(token_manifest),
        "tokens_dir": str(tokens_dir),
        "seq_len": seq_len,
        "available_tokens": loader.assigned_token_count,
        "required_tokens": required_tokens,
        "sources": sorted(loader.assigned_shards),
    }
    evidence.update(metadata_evidence)
    if metadata_errors:
        evidence["metadata_errors"] = metadata_errors

    if loader.assigned_token_count < required_tokens:
        return _gate(
            "real_corpus_manifest",
            "fail",
            "token manifest is readable but too small for the requested launch mode",
            evidence=evidence,
            detail=(
                "token manifest has fewer tokens than the r1 target "
                f"({loader.assigned_token_count} < {required_tokens})"
            ),
        )

    if metadata_errors:
        return _gate(
            "real_corpus_manifest",
            "fail",
            "token manifest is readable but missing full r1 corpus metadata",
            evidence=evidence,
            detail="; ".join(metadata_errors),
        )

    return _gate(
        "real_corpus_manifest",
        "pass",
        "token manifest is readable by the Mini-K3 uint32 shard loader",
        evidence=evidence,
    )


def _validate_full_corpus_metadata(
    manifest: dict[str, Any],
    *,
    tokens_dir: Path,
    evidence: dict[str, Any],
    errors: list[str],
) -> None:
    tokenizer = manifest.get("tokenizer")
    tokenizer_name: str | None = None
    tokenizer_family: str | None = None
    tokenizer_fingerprint: str | None = None
    if not isinstance(tokenizer, dict):
        errors.append("full Mini-K3 manifest must define a tokenizer object")
    else:
        tokenizer_name = _optional_str(tokenizer.get("name"))
        tokenizer_family = _optional_str(tokenizer.get("family"))
        tokenizer_label = " ".join(
            value.lower() for value in (tokenizer_name, tokenizer_family) if value
        )
        if not tokenizer_label or (
            "kimi" not in tokenizer_label and "k3" not in tokenizer_label
        ):
            errors.append(
                "full Mini-K3 manifest tokenizer must identify the Kimi K3 tokenizer"
            )
        if tokenizer.get("vocab_size") != mini_k3_r1_config().vocab_size:
            errors.append(
                "full Mini-K3 manifest tokenizer vocab_size must be "
                f"{mini_k3_r1_config().vocab_size}"
            )
        tokenizer_fingerprint = _first_nonempty_str(
            tokenizer, ("sha256", "hash", "fingerprint", "checksum")
        )
        if tokenizer_fingerprint is None or tokenizer_fingerprint.startswith(
            "pending-"
        ):
            errors.append(
                "full Mini-K3 manifest tokenizer must include a non-placeholder "
                "fingerprint or hash"
            )
        asset_report = _optional_str(tokenizer.get("asset_report"))
        if asset_report is not None:
            report_path = _resolve_manifest_local_path(
                asset_report,
                manifest_dir=tokens_dir.parent,
            )
            try:
                report = json.loads(report_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                errors.append(
                    "full Mini-K3 manifest tokenizer asset report could not be read: "
                    f"{exc}"
                )
            else:
                if report.get("kind") != "mini_kimi_k3_tokenizer_assets":
                    errors.append(
                        "full Mini-K3 manifest tokenizer asset report kind must be "
                        "mini_kimi_k3_tokenizer_assets"
                    )
                if report.get("schema_version") != 1:
                    errors.append(
                        "full Mini-K3 manifest tokenizer asset report schema_version "
                        "must be 1"
                    )
                report_fingerprint = _optional_str(report.get("sha256"))
                if report_fingerprint != tokenizer_fingerprint:
                    errors.append(
                        "full Mini-K3 manifest tokenizer asset report sha256 "
                        "mismatch"
                    )

    decontamination = manifest.get("decontamination")
    status: str | None = None
    report: str | None = None
    report_fingerprint: str | None = None
    report_scope: str | None = None
    if not isinstance(decontamination, dict):
        errors.append("full Mini-K3 manifest must define a decontamination object")
    else:
        status = _optional_str(decontamination.get("status"))
        if status is None or status.lower() not in {
            "complete",
            "completed",
            "pass",
            "passed",
            "reviewed",
            "verified",
        }:
            errors.append(
                "full Mini-K3 manifest decontamination status must be completed "
                "or reviewed"
            )
        report = _optional_str(decontamination.get("report")) or _optional_str(
            decontamination.get("report_path")
        )
        report_fingerprint = _first_nonempty_str(
            decontamination, ("sha256", "hash", "fingerprint", "checksum")
        )
        if (
            report is None
            or report.startswith("pending-")
            or report_fingerprint is None
            or report_fingerprint.startswith("pending-")
        ):
            errors.append(
                "full Mini-K3 manifest decontamination must include a non-placeholder "
                "report path and report hash"
            )
        elif "://" not in report:
            report_path = _resolve_manifest_local_path(
                report,
                manifest_dir=tokens_dir.parent,
            )
            try:
                actual_fingerprint = _sha256(report_path)
            except FileNotFoundError as exc:
                errors.append(
                    "full Mini-K3 manifest decontamination report could not be read: "
                    f"{exc}"
                )
            else:
                if actual_fingerprint != report_fingerprint:
                    errors.append(
                        "full Mini-K3 manifest decontamination report sha256 mismatch"
                    )
                report_scope_error, report_scope = _decontamination_report_scope_error(
                    report_path
                )
                if report_scope_error is not None:
                    errors.append(report_scope_error)

    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("full Mini-K3 manifest must define corpus sources")
        return
    for source_name, source_data in sources.items():
        if not isinstance(source_name, str) or not source_name:
            errors.append("full Mini-K3 manifest source names must be nonempty strings")
            continue
        if not isinstance(source_data, dict):
            errors.append(
                f"full Mini-K3 manifest source {source_name} must be an object"
            )
            continue
        source_id = _first_nonempty_str(
            source_data, ("dataset_id", "id", "name", "uri", "url")
        )
        provenance = source_data.get("provenance")
        source_hash = _first_nonempty_str(
            source_data, ("sha256", "hash", "fingerprint", "checksum")
        )
        has_provenance = isinstance(provenance, dict) and bool(provenance)
        provenance_report_error = _source_provenance_report_error(
            source_name=source_name,
            provenance=provenance,
            manifest_dir=tokens_dir.parent,
        )
        if provenance_report_error is not None:
            errors.append(provenance_report_error)
        has_hashed_provenance_report = (
            has_provenance
            and isinstance(provenance, dict)
            and _optional_str(provenance.get("report")) is not None
            and _first_nonempty_str(
                provenance, ("sha256", "hash", "fingerprint", "checksum")
            )
            is not None
            and provenance_report_error is None
        )
        has_placeholder_provenance = has_provenance and any(
            isinstance(value, str) and value.startswith("pending-")
            for value in provenance.values()
        )
        if (
            source_id is None
            or source_id.startswith("pending-")
            or (not has_provenance and source_hash is None)
            or (has_placeholder_provenance and not has_hashed_provenance_report)
        ):
            errors.append(
                f"full Mini-K3 manifest source {source_name} must include a "
                "dataset id and non-placeholder provenance or hash"
            )
        local_stage1_error = _local_stage1_report_error(
            source_name=source_name,
            local_stage1=source_data.get("local_stage1"),
            manifest_dir=tokens_dir.parent,
        )
        if local_stage1_error is not None:
            errors.append(local_stage1_error)
        local_stage1_decontamination_error = _local_stage1_decontamination_error(
            source_name=source_name,
            source_data=source_data,
        )
        if local_stage1_decontamination_error is not None:
            errors.append(local_stage1_decontamination_error)
        shard_error = _validate_registered_shard_metadata(
            source_name=source_name,
            source_data=source_data,
            tokens_dir=tokens_dir,
        )
        if shard_error is not None:
            errors.append(shard_error)

    if isinstance(tokenizer, dict):
        evidence["tokenizer"] = {
            "name": tokenizer_name,
            "family": tokenizer_family,
            "vocab_size": tokenizer.get("vocab_size"),
            "fingerprint": tokenizer_fingerprint,
        }
    if isinstance(decontamination, dict):
        evidence["decontamination"] = {
            "status": status,
            "report": report,
            "fingerprint": report_fingerprint,
        }
        if report_scope is not None:
            evidence["decontamination"]["scope"] = report_scope
    evidence["source_count"] = len(sources)
    evidence["source_ids"] = sorted(
        _first_nonempty_str(source_data, ("dataset_id", "id", "name", "uri", "url"))
        for source_data in sources.values()
        if isinstance(source_data, dict)
        and _first_nonempty_str(source_data, ("dataset_id", "id", "name", "uri", "url"))
        is not None
    )


def _validate_registered_shard_metadata(
    *,
    source_name: str,
    source_data: dict[str, Any],
    tokens_dir: Path,
) -> str | None:
    shards = source_data.get("shards")
    if not isinstance(shards, list) or not shards:
        return f"full Mini-K3 manifest source {source_name!r} has no shards"
    shard_metadata = source_data.get("shard_metadata")
    if not isinstance(shard_metadata, dict):
        return (
            f"full Mini-K3 manifest source {source_name!r} must define "
            "shard_metadata for every shard"
        )
    for shard_name in shards:
        if not isinstance(shard_name, str) or not shard_name:
            return f"full Mini-K3 manifest source {source_name!r} has an invalid shard"
        metadata = shard_metadata.get(shard_name)
        if not isinstance(metadata, dict):
            return (
                f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
                "is missing shard_metadata"
            )
        shard_path = tokens_dir / source_name / shard_name
        size = shard_path.stat().st_size
        expected_bytes = metadata.get("bytes")
        if expected_bytes != size:
            return (
                f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
                f"metadata bytes mismatch ({expected_bytes} != {size})"
            )
        expected_tokens = metadata.get("num_tokens")
        if expected_tokens != size // 4:
            return (
                f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
                f"metadata num_tokens mismatch ({expected_tokens} != {size // 4})"
            )
        expected_sha256 = _optional_str(metadata.get("sha256"))
        actual_sha256 = _sha256(shard_path)
        if expected_sha256 != actual_sha256:
            return (
                f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
                "metadata sha256 mismatch"
            )
        source_sidecar = _optional_str(metadata.get("source_sidecar"))
        if source_sidecar is not None:
            sidecar_error = _validate_source_sidecar(
                source_name=source_name,
                shard_name=shard_name,
                sidecar_path=Path(source_sidecar),
                expected_tokens=size // 4,
                expected_sha256=_optional_str(metadata.get("source_sidecar_sha256")),
            )
            if sidecar_error is not None:
                return sidecar_error
    return None


def _validate_source_sidecar(
    *,
    source_name: str,
    shard_name: str,
    sidecar_path: Path,
    expected_tokens: int,
    expected_sha256: str | None,
) -> str | None:
    if expected_sha256 is None:
        return (
            f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
            "source_sidecar_sha256 is required for imported Stage 1 sidecars"
        )
    try:
        sidecar_payload = sidecar_path.read_bytes()
        sidecar = json.loads(sidecar_payload)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return (
            f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
            f"source_sidecar could not be read: {exc}"
        )
    actual_sha256 = hashlib.sha256(sidecar_payload).hexdigest()
    if actual_sha256 != expected_sha256:
        return (
            f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
            "source_sidecar sha256 mismatch"
        )
    sidecar_source = _optional_str(sidecar.get("source"))
    if sidecar_source != source_name:
        return (
            f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
            f"source_sidecar source mismatch ({sidecar_source!r} != {source_name!r})"
        )
    if sidecar.get("dtype") != "uint32":
        return (
            f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
            "source_sidecar dtype must be uint32"
        )
    if sidecar.get("tokens") != expected_tokens:
        return (
            f"full Mini-K3 manifest source {source_name!r} shard {shard_name} "
            f"source_sidecar token count mismatch ({sidecar.get('tokens')} != "
            f"{expected_tokens})"
        )
    return None


def _r1_corpus_plan_audit_evidence(
    path: Path | None,
    *,
    token_manifest: Path,
    tokens_dir: Path,
    available_tokens: int,
    required_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"report": str(path), "status": "unreadable"}, str(exc)

    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "target_tokens": report.get("target_tokens"),
        "planned_tokens": report.get("planned_tokens"),
        "available_tokens": report.get("available_tokens"),
        "deficit_tokens": report.get("deficit_tokens"),
    }
    sources = report.get("sources")
    if isinstance(sources, dict):
        evidence["sources"] = sources
    if report.get("kind") != "mini_kimi_k3_r1_corpus_plan_audit":
        return (
            evidence,
            "corpus plan audit report kind must be "
            "mini_kimi_k3_r1_corpus_plan_audit",
        )
    if report.get("schema_version") != 1:
        return evidence, "corpus plan audit report schema_version must be 1"
    if report.get("manifest") != str(token_manifest):
        return (
            evidence,
            "corpus plan audit report does not match requested token manifest: "
            f"{report.get('manifest')} != {token_manifest}",
        )
    if report.get("tokens_dir") != str(tokens_dir):
        return (
            evidence,
            "corpus plan audit report does not match requested tokens dir: "
            f"{report.get('tokens_dir')} != {tokens_dir}",
        )
    if report.get("target_tokens") != required_tokens:
        return (
            evidence,
            "corpus plan audit report target_tokens does not match r1 target: "
            f"{report.get('target_tokens')} != {required_tokens}",
        )
    if report.get("available_tokens") != available_tokens:
        return (
            evidence,
            "corpus plan audit report available_tokens does not match token "
            f"manifest: {report.get('available_tokens')} != {available_tokens}",
        )
    status = report.get("status")
    if available_tokens < required_tokens and status != "blocked":
        return (
            evidence,
            "corpus plan audit report must be blocked while manifest is below "
            "the r1 target",
        )
    if available_tokens >= required_tokens and status != "ready":
        return (
            evidence,
            "corpus plan audit report must be ready when manifest meets the r1 "
            "target",
        )
    deficit_tokens = report.get("deficit_tokens")
    if not isinstance(deficit_tokens, int) or deficit_tokens != max(
        0, required_tokens - available_tokens
    ):
        return evidence, "corpus plan audit report deficit_tokens is inconsistent"
    return evidence, None


def _corpus_disk_readiness_evidence(
    path: Path | None,
    *,
    required_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"report": str(path), "status": "unreadable"}, str(exc)

    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "path": report.get("path"),
        "target_tokens": report.get("target_tokens"),
        "required_bytes": report.get("required_bytes"),
        "free_bytes": report.get("free_bytes"),
        "deficit_bytes": report.get("deficit_bytes"),
    }
    if report.get("kind") != "mini_kimi_k3_corpus_disk_readiness":
        return (
            evidence,
            "corpus disk readiness report kind must be "
            "mini_kimi_k3_corpus_disk_readiness",
        )
    if report.get("schema_version") != 1:
        return evidence, "corpus disk readiness report schema_version must be 1"
    if report.get("target_tokens") != required_tokens:
        return (
            evidence,
            "corpus disk readiness report target_tokens does not match r1 target: "
            f"{report.get('target_tokens')} != {required_tokens}",
        )
    status = report.get("status")
    if status != "ready":
        return evidence, "corpus disk readiness report must be ready"
    required_bytes = report.get("required_bytes")
    free_bytes = report.get("free_bytes")
    deficit_bytes = report.get("deficit_bytes")
    if not isinstance(required_bytes, int) or required_bytes < 0:
        return evidence, "corpus disk readiness report required_bytes is invalid"
    if not isinstance(free_bytes, int) or free_bytes < 0:
        return evidence, "corpus disk readiness report free_bytes is invalid"
    if not isinstance(deficit_bytes, int) or deficit_bytes != 0:
        return evidence, "corpus disk readiness report deficit_bytes must be 0"
    if free_bytes < required_bytes:
        return evidence, "corpus disk readiness report free_bytes is insufficient"
    return evidence, None


def _stage1_source_resolution_evidence(
    path: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"report": str(path), "status": "unreadable"}, str(exc)

    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "num_benchmarks": report.get("num_benchmarks"),
        "num_corpora": report.get("num_corpora"),
        "unresolved_benchmarks": report.get("unresolved_benchmarks"),
        "unresolved_corpora": report.get("unresolved_corpora"),
    }
    if report.get("kind") != "mini_kimi_k3_stage1_source_resolution":
        return (
            evidence,
            "stage1 source resolution report kind must be "
            "mini_kimi_k3_stage1_source_resolution",
        )
    if report.get("schema_version") != 1:
        return evidence, "stage1 source resolution report schema_version must be 1"
    unresolved_benchmarks = report.get("unresolved_benchmarks")
    unresolved_corpora = report.get("unresolved_corpora")
    if not isinstance(unresolved_benchmarks, list) or not all(
        isinstance(name, str) for name in unresolved_benchmarks
    ):
        return (
            evidence,
            "stage1 source resolution report unresolved_benchmarks must be a string list",
        )
    if not isinstance(unresolved_corpora, list) or not all(
        isinstance(name, str) for name in unresolved_corpora
    ):
        return (
            evidence,
            "stage1 source resolution report unresolved_corpora must be a string list",
        )
    if report.get("status") != "ready":
        return evidence, "stage1 source resolution report must be ready"
    if unresolved_benchmarks or unresolved_corpora:
        return (
            evidence,
            "stage1 source resolution report must have no unresolved benchmarks "
            "or corpora",
        )
    return evidence, None


def _stage1_input_inspection_evidence(
    path: Path | None,
    *,
    required_tokens: int,
    required: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        if required:
            return None, "stage1 input inspection report is required for local Stage 1"
        return None, None
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"report": str(path), "status": "unreadable"}, str(exc)

    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "input_manifest": report.get("input_manifest"),
        "input_format": report.get("input_format"),
        "min_total_tokens": report.get("min_total_tokens"),
        "total_files": report.get("total_files"),
        "total_documents": report.get("total_documents"),
        "total_tokens": report.get("total_tokens"),
    }
    if report.get("kind") != "mini_kimi_k3_stage1_input_inspection":
        return (
            evidence,
            "stage1 input inspection report kind must be "
            "mini_kimi_k3_stage1_input_inspection",
        )
    if report.get("schema_version") != 1:
        return evidence, "stage1 input inspection report schema_version must be 1"
    if report.get("status") != "pass":
        return evidence, "stage1 input inspection report must pass"
    min_total_tokens = report.get("min_total_tokens")
    if not isinstance(min_total_tokens, int) or min_total_tokens < required_tokens:
        return (
            evidence,
            "stage1 input inspection report min_total_tokens must cover the r1 "
            f"target ({min_total_tokens} < {required_tokens})",
        )
    total_tokens = report.get("total_tokens")
    if not isinstance(total_tokens, int) or total_tokens < required_tokens:
        return (
            evidence,
            "stage1 input inspection report total_tokens must cover the r1 "
            f"target ({total_tokens} < {required_tokens})",
        )
    for key in ("total_files", "total_documents"):
        value = report.get(key)
        if not isinstance(value, int) or value <= 0:
            return evidence, f"stage1 input inspection report {key} must be positive"
    input_manifest = report.get("input_manifest")
    if not isinstance(input_manifest, dict):
        return evidence, "stage1 input inspection report input_manifest is required"
    input_manifest_path = _optional_str(input_manifest.get("path"))
    input_manifest_paths = input_manifest.get("paths")
    has_input_manifest_paths = (
        isinstance(input_manifest_paths, list)
        and len(input_manifest_paths) > 0
        and all(_optional_str(path) is not None for path in input_manifest_paths)
    )
    if input_manifest_path is None and not has_input_manifest_paths:
        return (
            evidence,
            "stage1 input inspection report input_manifest.path or "
            "input_manifest.paths is required",
        )
    if _optional_str(input_manifest.get("sha256")) is None:
        return (
            evidence,
            "stage1 input inspection report input_manifest.sha256 is required",
        )
    return evidence, None


def _stage1_remote_readiness_evidence(
    path: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"report": str(path), "status": "unreadable"}, str(exc)

    checks = report.get("checks")
    rootfs = report.get("rootfs")
    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "huggingface_dns": _nested_status(checks, "huggingface_dns"),
        "modal_cli": _nested_status(checks, "modal_cli"),
        "modal_auth": _nested_status(checks, "modal_auth"),
        "rootfs_marker": rootfs.get("marker") if isinstance(rootfs, dict) else None,
        "network_mode": rootfs.get("network_mode")
        if isinstance(rootfs, dict)
        else None,
    }
    env_file_check = checks.get("env_file") if isinstance(checks, dict) else None
    if isinstance(env_file_check, dict):
        evidence["env_file"] = env_file_check
    elif isinstance(report.get("env_file"), dict):
        evidence["env_file"] = report["env_file"]
    if report.get("kind") != "mini_kimi_k3_stage1_remote_readiness":
        return (
            evidence,
            "stage1 remote readiness report kind must be "
            "mini_kimi_k3_stage1_remote_readiness",
        )
    if report.get("schema_version") != 1:
        return evidence, "stage1 remote readiness report schema_version must be 1"
    if not isinstance(rootfs, dict) or rootfs.get("marker") != "1":
        return evidence, "stage1 remote readiness report rootfs marker must be 1"
    if rootfs.get("network_mode") != "networked":
        return evidence, "stage1 remote readiness report network_mode must be networked"
    if not isinstance(checks, dict):
        return evidence, "stage1 remote readiness report checks must be an object"
    if isinstance(env_file_check, dict) and env_file_check.get("status") != "pass":
        return (
            evidence,
            "stage1 remote readiness env file is blocked "
            f"({env_file_check.get('detail')})",
        )
    required_checks = ("huggingface_dns", "modal_cli", "modal_auth")
    for check_name in required_checks:
        check = checks.get(check_name)
        if not isinstance(check, dict):
            return (
                evidence,
                f"stage1 remote readiness report missing {check_name} check",
            )
        if check.get("status") != "pass":
            return (
                evidence,
                "stage1 remote readiness report must be ready "
                f"({check_name}={check.get('status')})",
            )
    if report.get("status") != "ready":
        return evidence, "stage1 remote readiness report must be ready"
    return evidence, None


def _nested_status(mapping: object, key: str) -> object:
    if not isinstance(mapping, dict):
        return None
    nested = mapping.get(key)
    if not isinstance(nested, dict):
        return None
    return nested.get("status")


def _full_corpus_route(manifest: dict[str, Any]) -> str:
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources:
        return "unknown"

    for source_data in sources.values():
        if not isinstance(source_data, dict):
            return "unknown"
        if isinstance(source_data.get("local_stage1"), dict):
            continue
        shard_metadata = source_data.get("shard_metadata")
        if (
            isinstance(shard_metadata, dict)
            and shard_metadata
            and all(
                isinstance(metadata, dict)
                and _optional_str(metadata.get("source_sidecar")) is not None
                for metadata in shard_metadata.values()
            )
        ):
            continue
        return "remote_stage1"
    return "local_or_imported"


def _has_local_stage1_route(manifest: dict[str, Any]) -> bool:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        return False
    return any(
        isinstance(source_data, dict)
        and isinstance(source_data.get("local_stage1"), dict)
        for source_data in sources.values()
    )


def _local_stage1_decontamination_error(
    *,
    source_name: str,
    source_data: dict[str, Any],
) -> str | None:
    if not isinstance(source_data.get("local_stage1"), dict):
        return None

    local_builds = source_data.get("local_builds")
    if not isinstance(local_builds, list) or not local_builds:
        return (
            f"full Mini-K3 manifest source {source_name!r} local_stage1 "
            "decontamination evidence is required"
        )

    for index, local_build in enumerate(local_builds):
        if not isinstance(local_build, dict):
            return (
                f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                f"decontamination evidence build {index} must be an object"
            )
        decontamination = local_build.get("decontamination")
        if not isinstance(decontamination, dict):
            return (
                f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                "decontamination evidence is required for every local build"
            )
        if decontamination.get("applied") is not True:
            return (
                f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                "decontamination evidence must record applied=true"
            )
        index_path = _optional_str(decontamination.get("index"))
        if index_path is None or index_path.startswith("pending-"):
            return (
                f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                "decontamination evidence must include the index path"
            )
        min_matches = decontamination.get("min_matches")
        if not isinstance(min_matches, int) or min_matches <= 0:
            return (
                f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                "decontamination evidence must include positive min_matches"
            )
        for field in ("documents_seen", "documents_kept", "documents_removed"):
            value = decontamination.get(field)
            if not isinstance(value, int) or value < 0:
                return (
                    f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                    f"decontamination evidence must include nonnegative {field}"
                )
        documents = local_build.get("documents")
        if (
            isinstance(documents, int)
            and documents != decontamination["documents_kept"]
        ):
            return (
                f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                "decontamination evidence documents_kept must match local build "
                "documents"
            )
        hits_by_benchmark = decontamination.get("hits_by_benchmark")
        if not isinstance(hits_by_benchmark, dict):
            return (
                f"full Mini-K3 manifest source {source_name!r} local_stage1 "
                "decontamination evidence must include hits_by_benchmark"
            )
    return None


def _source_provenance_report_error(
    *,
    source_name: str,
    provenance: object,
    manifest_dir: Path,
) -> str | None:
    if not isinstance(provenance, dict):
        return None
    report = _optional_str(provenance.get("report"))
    if report is None or "://" in report:
        return None
    expected_fingerprint = _first_nonempty_str(
        provenance,
        ("sha256", "hash", "fingerprint", "checksum"),
    )
    if expected_fingerprint is None or expected_fingerprint.startswith("pending-"):
        return (
            f"full Mini-K3 manifest source {source_name} provenance report must "
            "include a non-placeholder hash"
        )
    report_path = _resolve_manifest_local_path(report, manifest_dir=manifest_dir)
    try:
        actual_fingerprint = _sha256(report_path)
    except FileNotFoundError as exc:
        return (
            f"full Mini-K3 manifest source {source_name} provenance report "
            f"could not be read: {exc}"
        )
    if actual_fingerprint != expected_fingerprint:
        return (
            f"full Mini-K3 manifest source {source_name} provenance report "
            "sha256 mismatch"
        )
    return None


def _local_stage1_report_error(
    *,
    source_name: str,
    local_stage1: object,
    manifest_dir: Path,
) -> str | None:
    if local_stage1 is None:
        return None
    if not isinstance(local_stage1, dict):
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 must be "
            "an object"
        )
    report = _optional_str(local_stage1.get("report"))
    if report is None or report.startswith("pending-") or "://" in report:
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 must include "
            "a local report path"
        )
    expected_fingerprint = _first_nonempty_str(
        local_stage1,
        ("sha256", "hash", "fingerprint", "checksum"),
    )
    if expected_fingerprint is None or expected_fingerprint.startswith("pending-"):
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 report must "
            "include a non-placeholder hash"
        )
    report_path = _resolve_manifest_local_path(report, manifest_dir=manifest_dir)
    try:
        report_payload = report_path.read_bytes()
        report_data = json.loads(report_payload)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 report "
            f"could not be read: {exc}"
        )
    actual_fingerprint = hashlib.sha256(report_payload).hexdigest()
    if actual_fingerprint != expected_fingerprint:
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 report "
            "sha256 mismatch"
        )
    if report_data.get("kind") != "mini_kimi_k3_source_provenance":
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 report kind "
            "must be mini_kimi_k3_source_provenance"
        )
    if report_data.get("schema_version") != 1:
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 report "
            "schema_version must be 1"
        )
    if report_data.get("source") != source_name:
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 report "
            f"source mismatch ({report_data.get('source')!r} != {source_name!r})"
        )
    expected_tokens = local_stage1.get("total_tokens")
    if not isinstance(expected_tokens, int) or expected_tokens < 0:
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 total_tokens "
            "must be a nonnegative integer"
        )
    if report_data.get("total_tokens") != expected_tokens:
        return (
            f"full Mini-K3 manifest source {source_name} local_stage1 total_tokens "
            f"mismatch ({report_data.get('total_tokens')} != {expected_tokens})"
        )
    return None


def _decontamination_report_scope_error(
    report_path: Path,
) -> tuple[str | None, str | None]:
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return None, None
    if report.get("kind") != "mini_kimi_k3_decontamination_report":
        return None, None
    scope = _optional_str(report.get("scope"))
    if scope != "launch_grade_benchmark_suite":
        return (
            "full Mini-K3 manifest decontamination report scope must be "
            "launch_grade_benchmark_suite",
            scope,
        )
    return None, scope


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_nonempty_str(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _optional_str(data.get(key))
        if value is not None:
            return value
    return None


def _resolve_manifest_local_path(value: str, *, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_fidelity_gate(
    *,
    model_contract: Path,
    model_contract_tests: Path,
    activation: Path,
    activation_tests: Path,
    router: Path,
    router_tests: Path,
    moe: Path,
    moe_tests: Path,
    kda: Path,
    kda_tests: Path,
    mla: Path,
    mla_tests: Path,
    model: Path,
    model_tests: Path,
    oracle_root: Path | None,
    oracle_r1_config: Path | None,
    forward_oracle_report: Path | None,
    launch_backend_report: Path | None,
) -> dict[str, Any]:
    if (
        model_contract.is_file()
        and model_contract_tests.is_file()
        and activation.is_file()
        and activation_tests.is_file()
        and router.is_file()
        and router_tests.is_file()
        and moe.is_file()
        and moe_tests.is_file()
        and kda.is_file()
        and kda_tests.is_file()
        and mla.is_file()
        and mla_tests.is_file()
        and model.is_file()
        and model_tests.is_file()
    ):
        evidence: dict[str, Any] = {
            "contract": str(model_contract.relative_to(REPO_ROOT)),
            "tests": str(model_contract_tests.relative_to(REPO_ROOT)),
            "activation": str(activation.relative_to(REPO_ROOT)),
            "activation_tests": str(activation_tests.relative_to(REPO_ROOT)),
            "router": str(router.relative_to(REPO_ROOT)),
            "router_tests": str(router_tests.relative_to(REPO_ROOT)),
            "moe": str(moe.relative_to(REPO_ROOT)),
            "moe_tests": str(moe_tests.relative_to(REPO_ROOT)),
            "kda": str(kda.relative_to(REPO_ROOT)),
            "kda_tests": str(kda_tests.relative_to(REPO_ROOT)),
            "mla": str(mla.relative_to(REPO_ROOT)),
            "mla_tests": str(mla_tests.relative_to(REPO_ROOT)),
            "model": str(model.relative_to(REPO_ROOT)),
            "model_tests": str(model_tests.relative_to(REPO_ROOT)),
            "r1_kimi_linear_config": to_kimi_linear_config_dict(mini_k3_r1_config()),
        }
        parameter_budget = estimate_parameter_budget(mini_k3_r1_config())
        evidence["parameter_budget"] = parameter_budget.to_dict()
        forward_oracle = _forward_oracle_evidence(forward_oracle_report)
        evidence["forward_oracle"] = forward_oracle["evidence"]
        if forward_oracle["status"] == "fail":
            return _gate(
                "model_fidelity",
                "fail",
                "Mini-K3 first-party forward oracle comparison did not pass",
                evidence=evidence,
                detail=forward_oracle["detail"],
            )
        launch_backend = _launch_backend_evidence(
            launch_backend_report,
            forward_oracle_report=forward_oracle_report,
        )
        evidence["launch_backend"] = launch_backend["evidence"]
        if launch_backend["status"] == "fail":
            return _gate(
                "model_fidelity",
                "fail",
                "Mini-K3 launch backend review did not pass",
                evidence=evidence,
                detail=launch_backend["detail"],
            )
        if oracle_root is not None:
            try:
                evidence["oracle"] = compare_with_first_party_oracle(
                    oracle_root,
                    r1_config_path=oracle_r1_config,
                )
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                return _gate(
                    "model_fidelity",
                    "fail",
                    "Mini-K3 first-party oracle comparison could not be read",
                    evidence={
                        "oracle_root": str(oracle_root),
                        "oracle_r1_config": (
                            str(oracle_r1_config)
                            if oracle_r1_config is not None
                            else None
                        ),
                    },
                    detail=str(exc),
                )
            mismatches = {
                key: value
                for key, value in evidence["oracle"].items()
                if value != "match"
            }
            if mismatches:
                return _gate(
                    "model_fidelity",
                    "fail",
                    "Mini-K3 exported config does not match the first-party oracle",
                    evidence=evidence,
                    detail=json.dumps(mismatches, sort_keys=True),
                )
        if parameter_budget.budget_status == "fail":
            return _gate(
                "model_fidelity",
                "fail",
                "Mini-K3 r1 parameter budget does not match the source-derived claim",
                evidence=evidence,
                detail=parameter_budget.budget_detail,
            )
        if forward_oracle["status"] == "pass" and launch_backend["status"] == "pass":
            return _gate(
                "model_fidelity",
                "pass",
                "Mini-K3 r1 architecture and launch backend evidence passed",
                evidence=evidence,
            )
        return _gate(
            "model_fidelity",
            "partial",
            "Mini-K3 r1 architecture exists, but launch-backend evidence is incomplete",
            evidence=evidence,
            detail=(
                forward_oracle["detail"]
                if forward_oracle["status"] == "missing"
                else launch_backend["detail"]
            ),
        )
    return _gate(
        "model_fidelity",
        "missing",
        "KDA, MLA, SITU, MoE routing, and parameter-count checks are not implemented",
    )


def _forward_oracle_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "missing",
            "evidence": {"status": "missing"},
            "detail": "no Mini-K3 r1 first-party forward oracle report was provided",
        }
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "evidence": {"report": str(path), "status": "unreadable"},
            "detail": str(exc),
        }

    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "model_flavor": report.get("model_flavor"),
    }
    report_error = _optional_str(report.get("error"))
    if report_error is not None:
        evidence["error"] = report_error
    results = report.get("results")
    if isinstance(results, dict):
        evidence["max_abs_diff"] = results.get("max_abs_diff")
        evidence["max_rel_diff"] = results.get("max_rel_diff")

    if report.get("kind") != "mini_kimi_k3_forward_oracle":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "forward oracle report kind must be mini_kimi_k3_forward_oracle",
        }
    if report.get("schema_version") != 1:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "forward oracle report schema_version must be 1",
        }
    if report.get("model_flavor") != "r1":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "forward oracle report must target model_flavor r1",
        }
    if report.get("status") != "pass":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": report_error or "forward oracle report did not pass",
        }

    tolerances = report.get("tolerances")
    if not isinstance(tolerances, dict) or not isinstance(results, dict):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "forward oracle report must include tolerances and results",
        }
    for key in ("max_abs_diff", "max_rel_diff"):
        tolerance = tolerances.get(key)
        measured = results.get(key)
        if not isinstance(tolerance, int | float) or not isinstance(
            measured, int | float
        ):
            return {
                "status": "fail",
                "evidence": evidence,
                "detail": f"forward oracle report {key} must be numeric",
            }
        if measured > tolerance:
            return {
                "status": "fail",
                "evidence": evidence,
                "detail": (
                    f"forward oracle report {key} exceeds tolerance "
                    f"({measured} > {tolerance})"
                ),
            }
    return {
        "status": "pass",
        "evidence": evidence,
        "detail": "forward oracle report passed",
    }


def _launch_backend_evidence(
    path: Path | None,
    *,
    forward_oracle_report: Path | None,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "missing",
            "evidence": {"status": "missing"},
            "detail": "no reviewed Mini-K3 r1 launch-backend report was provided",
        }
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "evidence": {"report": str(path), "status": "unreadable"},
            "detail": str(exc),
        }

    report_evidence = report.get("evidence")
    if not isinstance(report_evidence, dict):
        report_evidence = {}
    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "model_flavor": report.get("model_flavor"),
        "backend": report.get("backend"),
        "config": report.get("config"),
        "forward_oracle_report": report_evidence.get("forward_oracle_report"),
        "forward_trace_report": report_evidence.get("forward_trace_report"),
        "final_logit_max_abs_diff": report_evidence.get("final_logit_max_abs_diff"),
    }

    if report.get("kind") != "mini_kimi_k3_launch_backend_review":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "launch backend report kind must be "
                "mini_kimi_k3_launch_backend_review"
            ),
        }
    if report.get("schema_version") != 1:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "launch backend report schema_version must be 1",
        }
    if report.get("status") != "pass":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "launch backend report did not pass",
        }
    if report.get("model_flavor") != "r1":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "launch backend report must target model_flavor r1",
        }
    if report.get("backend") != "torchtitan.experiments.mini_kimi_k3":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "launch backend report backend must be "
                "torchtitan.experiments.mini_kimi_k3"
            ),
        }
    if report.get("config") != "mini_kimi_k3_r1_contract":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "launch backend report config must be " "mini_kimi_k3_r1_contract"
            ),
        }
    if forward_oracle_report is None:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "launch backend report requires a matching forward oracle report",
        }
    if report_evidence.get("forward_oracle_report") != str(forward_oracle_report):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "launch backend report does not match forward oracle report",
        }
    final_logit_diff = report_evidence.get("final_logit_max_abs_diff")
    if not isinstance(final_logit_diff, int | float) or final_logit_diff != 0.0:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "launch backend report must record exact zero final logit drift",
        }
    if _optional_str(report_evidence.get("forward_trace_report")) is None:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "launch backend report must include a forward trace report",
        }
    return {
        "status": "pass",
        "evidence": evidence,
        "detail": "launch backend report passed",
    }


def _trainable_config_gate(
    *,
    mode: str,
    recipe_contract: Path,
    recipe_contract_tests: Path,
    model_gate: dict[str, Any],
    launch_backend_report: Path | None,
    r1_training_smoke_report: Path | None,
    token_manifest: Path | None,
    tokens_dir: Path | None,
    seq_len: int,
) -> dict[str, Any]:
    if not recipe_contract.is_file() or not recipe_contract_tests.is_file():
        return _gate(
            "trainable_config",
            "missing",
            "no Mini-K3 launch recipe or trainable config is registered yet",
        )

    recipe_evidence = {
        "recipe": str(recipe_contract.relative_to(REPO_ROOT)),
        "tests": str(recipe_contract_tests.relative_to(REPO_ROOT)),
        **mini_k3_r1_launch_recipe().to_dict(),
    }
    if mode == "tiny-plumbing":
        try:
            config = ConfigManager().parse_args(
                [
                    "--module",
                    "mini_kimi_k3",
                    "--config",
                    "mini_kimi_k3_tiny_plumbing",
                ]
            )
            model_spec = config.model_spec
            assert model_spec is not None
            model = model_spec.model.build()
            model.verify_module_protocol()
        except Exception as exc:
            return _gate(
                "trainable_config",
                "fail",
                "Mini-K3 tiny Trainer config could not be loaded and built",
                evidence=recipe_evidence,
                detail=str(exc),
            )
        return _gate(
            "trainable_config",
            "pass",
            "Mini-K3 tiny-plumbing Trainer config loads and builds",
            evidence={
                **recipe_evidence,
                "module": "mini_kimi_k3",
                "config": "mini_kimi_k3_tiny_plumbing",
                "model_spec": {
                    "name": model_spec.name,
                    "flavor": model_spec.flavor,
                },
            },
        )
    try:
        config = ConfigManager().parse_args(
            [
                "--module",
                "mini_kimi_k3",
                "--config",
                "mini_kimi_k3_r1_contract",
            ]
        )
        model_spec = config.model_spec
        assert model_spec is not None
        model = model_spec.model.build()
        model.verify_module_protocol()
        nparams, _ = model_spec.model.get_nparams_and_flops(
            model,
            config.training.seq_len,
        )
        evidence = {
            **recipe_evidence,
            "module": "mini_kimi_k3",
            "config": "mini_kimi_k3_r1_contract",
            "global_batch_size": config.training.global_batch_size,
            "weight_decay_policy": model_spec.model.weight_decay_policy,
            "model_spec": {
                "name": model_spec.name,
                "flavor": model_spec.flavor,
            },
            "model_fidelity_status": model_gate["status"],
            "parameter_count": nparams,
        }
        if launch_backend_report is not None:
            evidence["launch_backend_report"] = str(launch_backend_report)
        training_smoke = _r1_training_smoke_evidence(
            r1_training_smoke_report,
            token_manifest=token_manifest,
            tokens_dir=tokens_dir,
            seq_len=seq_len,
        )
        evidence["r1_training_smoke"] = training_smoke["evidence"]
        if training_smoke["status"] == "fail":
            return _gate(
                "trainable_config",
                "fail",
                "Mini-K3 r1 Trainer smoke evidence did not pass",
                evidence={
                    **evidence,
                    "build_status": "buildable",
                },
                detail=training_smoke["detail"],
            )
        if model_gate["status"] == "pass" and training_smoke["status"] == "pass":
            return _gate(
                "trainable_config",
                "pass",
                "Mini-K3 r1 Trainer config loads, builds, and has execution evidence",
                evidence={
                    **evidence,
                    "build_status": "launchable",
                },
            )
        missing_reasons = []
        if model_gate["status"] != "pass":
            missing_reasons.append("passing model_fidelity")
        if training_smoke["status"] != "pass":
            missing_reasons.append("r1 training smoke")
        return _gate(
            "trainable_config",
            "partial",
            "Mini-K3 r1 Trainer config loads and builds, but execution evidence is incomplete",
            evidence={
                **evidence,
                "build_status": "buildable",
            },
            detail=(
                "Mini-K3 r1 trainable_config requires "
                f"{' and '.join(missing_reasons)} before full launch."
            ),
        )
    except Exception as exc:
        return _gate(
            "trainable_config",
            "partial",
            "Mini-K3 r1 launch recipe exists, but no executable r1 TorchTitan Trainer.Config is registered yet",
            evidence=recipe_evidence,
            detail=str(exc),
        )
    return _gate(
        "trainable_config",
        "partial",
        "Mini-K3 r1 contract config unexpectedly built a model; full launch still requires fidelity review",
        evidence=recipe_evidence,
        detail="r1 contract config did not fail closed during model build",
    )


def _r1_training_smoke_evidence(
    path: Path | None,
    *,
    token_manifest: Path | None,
    tokens_dir: Path | None,
    seq_len: int,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "missing",
            "evidence": {"status": "missing"},
            "detail": "no Mini-K3 r1 Trainer smoke report was provided",
        }
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "evidence": {"report": str(path), "status": "unreadable"},
            "detail": str(exc),
        }

    data = report.get("data")
    optimization = report.get("optimization")
    rootfs = report.get("rootfs")
    command = report.get("command")
    evidence = {
        "report": str(path),
        "status": report.get("status"),
        "config": report.get("config"),
        "model_flavor": report.get("model_flavor"),
    }
    report_error = _optional_str(report.get("error"))
    if report_error is not None:
        evidence["error"] = report_error
    gpu_memory = report.get("gpu_memory")
    if isinstance(gpu_memory, dict):
        evidence["gpu_memory"] = gpu_memory
    if isinstance(rootfs, dict):
        evidence["rootfs"] = {
            "marker": rootfs.get("marker"),
            "cwd": rootfs.get("cwd"),
        }
    if isinstance(command, dict):
        evidence["command"] = {
            "argv": command.get("argv"),
            "cwd": command.get("cwd"),
            "return_code": command.get("return_code"),
        }
    if isinstance(optimization, dict):
        evidence["steps"] = optimization.get("steps")
        evidence["max_grad_norm"] = optimization.get("max_grad_norm")
    if report.get("kind") != "mini_kimi_k3_r1_training_smoke":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report kind must be mini_kimi_k3_r1_training_smoke",
        }
    if report.get("schema_version") != 1:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report schema_version must be 1",
        }
    if report.get("status") != "pass":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": report_error or "r1 training smoke report did not pass",
        }
    if report.get("config") != "mini_kimi_k3_r1_contract":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report config must be mini_kimi_k3_r1_contract",
        }
    if report.get("model_flavor") != "r1":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report must target model_flavor r1",
        }
    if not isinstance(gpu_memory, dict):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report has no GPU capacity evidence",
        }
    if gpu_memory.get("status") != "pass" or gpu_memory.get("selected") is None:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report has no selected GPU capacity evidence",
        }
    if not isinstance(rootfs, dict) or rootfs.get("marker") != "1":
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report was not produced inside TorchTitan rootfs",
        }
    if rootfs.get("cwd") != ROOTFS_REPO_CWD:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke report has the wrong rootfs cwd: "
                f"{rootfs.get('cwd')} != {ROOTFS_REPO_CWD}"
            ),
        }
    command_argv = command.get("argv") if isinstance(command, dict) else None
    required_argv_items = {
        "MODULE=mini_kimi_k3",
        "CONFIG=mini_kimi_k3_r1_contract",
        "COMM_MODE=fake_backend",
        "NGPU=1",
        "./run_train.sh",
    }
    if (
        not isinstance(command, dict)
        or command.get("return_code") != 0
        or not isinstance(command_argv, list)
        or any(item not in command_argv for item in required_argv_items)
    ):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report was not produced by the expected run_train.sh command",
        }
    if command.get("cwd") != ROOTFS_REPO_CWD:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke report has the wrong command cwd: "
                f"{command.get('cwd')} != {ROOTFS_REPO_CWD}"
            ),
        }
    if not isinstance(data, dict):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report has no data section",
        }
    if token_manifest is None or tokens_dir is None:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke validation requires requested token manifest "
                "and tokens dir"
            ),
        }
    command_contract_error = _r1_smoke_command_contract_error(
        command_argv,
        data=data,
        optimization=optimization,
        gpu_memory=gpu_memory,
    )
    if command_contract_error is not None:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": command_contract_error,
        }
    command_options = _argv_options(command_argv)
    if token_manifest is not None and data.get("manifest") != str(token_manifest):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke report does not match requested token manifest: "
                f"{data.get('manifest')} != {token_manifest}"
            ),
        }
    if token_manifest is not None and command_options.get(
        "--dataloader.token_manifest"
    ) != str(token_manifest):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke command does not match requested token manifest: "
                f"{command_options.get('--dataloader.token_manifest')} != "
                f"{token_manifest}"
            ),
        }
    if tokens_dir is not None and data.get("tokens_dir") != str(tokens_dir):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke report does not match requested tokens dir: "
                f"{data.get('tokens_dir')} != {tokens_dir}"
            ),
        }
    if tokens_dir is not None and command_options.get("--dataloader.tokens_dir") != str(
        tokens_dir
    ):
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke command does not match requested tokens dir: "
                f"{command_options.get('--dataloader.tokens_dir')} != {tokens_dir}"
            ),
        }
    try:
        report_seq_len = int(data.get("seq_len", -1))
    except (TypeError, ValueError):
        report_seq_len = -1
    if report_seq_len != seq_len:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke report does not match requested seq_len: "
                f"{data.get('seq_len')} != {seq_len}"
            ),
        }
    try:
        command_seq_len = int(command_options.get("--training.seq_len", -1))
    except (TypeError, ValueError):
        command_seq_len = -1
    if command_seq_len != seq_len:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": (
                "r1 training smoke command does not match requested seq_len: "
                f"{command_options.get('--training.seq_len')} != {seq_len}"
            ),
        }
    try:
        smoke_steps = (
            int(optimization.get("steps", 0)) if isinstance(optimization, dict) else 0
        )
    except (TypeError, ValueError):
        smoke_steps = 0
    if not isinstance(optimization, dict) or smoke_steps < 1:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": "r1 training smoke report has no optimizer step evidence",
        }
    try:
        max_grad_norm = float(optimization["max_grad_norm"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "fail",
            "evidence": evidence,
            "detail": str(exc),
        }
    evidence["steps"] = smoke_steps
    evidence["max_grad_norm"] = max_grad_norm
    return {
        "status": "pass",
        "evidence": evidence,
        "detail": "r1 Trainer smoke report passed",
    }


def _argv_options(argv: list[Any]) -> dict[str, str]:
    options: dict[str, str] = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        if not isinstance(item, str):
            index += 1
            continue
        if item.startswith("--") and "=" in item:
            key, value = item.split("=", 1)
            options[key] = value
        elif item.startswith("--") and index + 1 < len(argv):
            value = argv[index + 1]
            if isinstance(value, str) and not value.startswith("--"):
                options[item] = value
                index += 1
        index += 1
    return options


def _argv_values(argv: list[Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        if not isinstance(item, str):
            index += 1
            continue
        if item.startswith("--") and "=" in item:
            key, value = item.split("=", 1)
            values.setdefault(key, []).append(value)
        elif item.startswith("--"):
            value = "true"
            if index + 1 < len(argv):
                next_item = argv[index + 1]
                if isinstance(next_item, str) and not next_item.startswith("--"):
                    value = next_item
                    index += 1
            values.setdefault(item, []).append(value)
        elif "=" in item:
            key, value = item.split("=", 1)
            values.setdefault(key, []).append(value)
        index += 1
    return values


def _single_argv_value(
    values: dict[str, list[str]], key: str, description: str
) -> tuple[str | None, str | None]:
    key_values = values.get(key, [])
    if len(key_values) != 1:
        return (
            None,
            (
                "r1 training smoke command must include exactly one "
                f"{description} argument"
            ),
        )
    return key_values[0], None


def _r1_smoke_command_contract_error(
    argv: list[Any],
    *,
    data: dict[str, Any],
    optimization: Any,
    gpu_memory: dict[str, Any],
) -> str | None:
    if not argv or argv[0] != "/usr/bin/env":
        return "r1 training smoke command must start with /usr/bin/env"
    launcher_indices = [
        index for index, item in enumerate(argv) if item == "./run_train.sh"
    ]
    if len(launcher_indices) != 1:
        return "r1 training smoke command must include exactly one ./run_train.sh"
    launcher_index = launcher_indices[0]
    env_argv = argv[1:launcher_index]
    train_argv = argv[launcher_index + 1 :]
    if any(isinstance(item, str) and item.startswith("--") for item in env_argv):
        return "r1 training smoke command has run arguments before ./run_train.sh"
    bare_train_arg = _first_bare_train_arg(train_argv)
    if bare_train_arg is not None:
        return (
            "r1 training smoke command has environment arguments after ./run_train.sh"
        )
    values = _argv_values(argv)
    expected_values = {
        "MODULE": "mini_kimi_k3",
        "CONFIG": "mini_kimi_k3_r1_contract",
        "COMM_MODE": "fake_backend",
        "NGPU": "1",
        "--training.local_batch_size": "1",
        "--training.global_batch_size": "1",
        "--training.seq_len": str(data.get("seq_len")),
        "--hf_assets_path": _optional_str(data.get("hf_assets_path")),
        "--dump_folder": "__required__",
        "--dataloader.token_manifest": _optional_str(data.get("manifest")),
        "--dataloader.tokens_dir": _optional_str(data.get("tokens_dir")),
    }
    descriptions = {
        "MODULE": "MODULE",
        "CONFIG": "CONFIG",
        "COMM_MODE": "COMM_MODE",
        "NGPU": "NGPU",
        "--training.local_batch_size": "local batch size",
        "--training.global_batch_size": "global batch size",
        "--training.seq_len": "seq_len",
        "--hf_assets_path": "hf assets path",
        "--dump_folder": "dump folder",
        "--dataloader.token_manifest": "token manifest",
        "--dataloader.tokens_dir": "tokens dir",
    }
    for key, expected_value in expected_values.items():
        if expected_value is None:
            return f"r1 training smoke command cannot validate missing {key}"
        actual_value, error = _single_argv_value(values, key, descriptions[key])
        if error is not None:
            return error
        if expected_value == "__required__":
            continue
        if actual_value != expected_value:
            return (
                "r1 training smoke command does not match expected "
                f"{descriptions[key]}: {actual_value} != {expected_value}"
            )

    steps_value, error = _single_argv_value(
        values, "--training.steps", "training steps"
    )
    if error is not None:
        return error
    expected_steps = (
        str(optimization.get("steps"))
        if isinstance(optimization, dict) and optimization.get("steps") is not None
        else None
    )
    if steps_value != expected_steps:
        return (
            "r1 training smoke command does not match reported training steps: "
            f"{steps_value} != {expected_steps}"
        )

    if values.get("--checkpoint.no-enable") != ["true"]:
        return "r1 training smoke command must include checkpoint disabling"

    selected = gpu_memory.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("index"), int):
        return "r1 training smoke report has no selected GPU index"
    cuda_visible_devices, error = _single_argv_value(
        values, "CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"
    )
    if error is not None:
        return error
    if cuda_visible_devices != str(selected["index"]):
        return (
            "r1 training smoke command does not match selected GPU selection: "
            f"{cuda_visible_devices} != {selected['index']}"
        )
    return None


def _first_bare_train_arg(argv: list[Any]) -> str | None:
    index = 0
    while index < len(argv):
        item = argv[index]
        if not isinstance(item, str):
            index += 1
            continue
        if not item.startswith("--"):
            return item
        if "=" not in item and index + 1 < len(argv):
            next_item = argv[index + 1]
            if isinstance(next_item, str) and not next_item.startswith("--"):
                index += 1
        index += 1
    return None


def _tiny_training_smoke_gate(
    *,
    mode: str,
    tiny_smoke_report: Path | None,
    token_manifest: Path | None,
    tokens_dir: Path | None,
    seq_len: int,
) -> dict[str, Any]:
    if mode != "tiny-plumbing":
        return _gate(
            "tiny_training_smoke",
            "not_applicable",
            "fixture-scale training smoke is only required for tiny-plumbing mode",
        )
    if tiny_smoke_report is None:
        return _gate(
            "tiny_training_smoke",
            "missing",
            "no fixture-scale forward/backward training smoke report was provided",
        )
    try:
        report = json.loads(tiny_smoke_report.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report is not readable",
            evidence=str(tiny_smoke_report),
            detail=str(exc),
        )
    if report.get("kind") != "mini_kimi_k3_training_smoke":
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report has the wrong kind",
            evidence=str(tiny_smoke_report),
            detail=str(report.get("kind")),
        )
    if report.get("status") != "pass":
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke did not pass",
            evidence=str(tiny_smoke_report),
            detail=str(report.get("status")),
        )
    data = report.get("data")
    if not isinstance(data, dict):
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report has no data section",
            evidence=str(tiny_smoke_report),
        )
    if token_manifest is not None and data.get("manifest") != str(token_manifest):
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report does not match requested token manifest",
            evidence=str(tiny_smoke_report),
            detail=(
                "does not match requested token manifest: "
                f"{data.get('manifest')} != {token_manifest}"
            ),
        )
    if tokens_dir is not None and data.get("tokens_dir") != str(tokens_dir):
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report does not match requested tokens dir",
            evidence=str(tiny_smoke_report),
            detail=f"{data.get('tokens_dir')} != {tokens_dir}",
        )
    try:
        report_seq_len = int(data.get("seq_len", -1))
    except (TypeError, ValueError):
        report_seq_len = -1
    if report_seq_len != seq_len:
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report does not match requested seq_len",
            evidence=str(tiny_smoke_report),
            detail=f"{data.get('seq_len')} != {seq_len}",
        )
    optimization = report.get("optimization")
    try:
        smoke_steps = (
            int(optimization.get("steps", 0)) if isinstance(optimization, dict) else 0
        )
    except (TypeError, ValueError):
        smoke_steps = 0
    if not isinstance(optimization, dict) or smoke_steps < 1:
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report has no optimizer step evidence",
            evidence=str(tiny_smoke_report),
        )
    try:
        tokens_emitted = int(data["tokens_emitted"])
        max_grad_norm = float(optimization["max_grad_norm"])
    except (KeyError, TypeError, ValueError) as exc:
        return _gate(
            "tiny_training_smoke",
            "fail",
            "fixture-scale training smoke report is missing measurement evidence",
            evidence=str(tiny_smoke_report),
            detail=str(exc),
        )
    return _gate(
        "tiny_training_smoke",
        "pass",
        "fixture-scale token loader, model forward/backward, and optimizer step passed",
        evidence={
            "report": str(tiny_smoke_report),
            "steps": smoke_steps,
            "tokens_emitted": tokens_emitted,
            "max_grad_norm": max_grad_norm,
        },
    )


def _launch_evidence_bundle_gate(
    *,
    evidence_results_root: Path | None,
    run_id: str | None,
    attempt_id: str | None,
    mode: str,
) -> dict[str, Any]:
    if evidence_results_root is None and run_id is None and attempt_id is None:
        return _gate(
            "launch_evidence_bundle",
            "missing",
            "no initialized run-attempt evidence bundle was provided",
        )
    if evidence_results_root is None or run_id is None or attempt_id is None:
        return _gate(
            "launch_evidence_bundle",
            "fail",
            "--evidence-results-root, --run-id, and --attempt-id must be provided together",
        )

    try:
        evidence = inspect_launch_evidence_bundle(
            results_root=evidence_results_root,
            run_id=run_id,
            attempt_id=attempt_id,
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        return _gate(
            "launch_evidence_bundle",
            "fail",
            "initialized run-attempt evidence bundle is not readable",
            evidence={
                "results_root": str(evidence_results_root),
                "run_id": run_id,
                "attempt_id": attempt_id,
            },
            detail=str(exc),
        )

    expected = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "family": "mini_kimi_k3",
        "task": "pretraining",
        "lane": mode,
        "model_variant": "r1",
    }
    for key, expected_value in expected.items():
        actual_value = evidence.get(key)
        if actual_value != expected_value:
            return _gate(
                "launch_evidence_bundle",
                "fail",
                "initialized run-attempt evidence bundle does not match this launch",
                evidence=evidence,
                detail=(
                    f"run-attempt evidence bundle {key} {actual_value!r} "
                    f"does not match preflight {key} {expected_value!r}"
                ),
            )

    return _gate(
        "launch_evidence_bundle",
        "pass",
        "run-attempt evidence bundle is initialized and readable",
        evidence=evidence,
    )


if __name__ == "__main__":
    raise SystemExit(main())
