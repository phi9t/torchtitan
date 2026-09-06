#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Audit whether the full Mini Kimi K3 launch objective actually completed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit completion of the full Mini Kimi K3 launch objective."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/mini_kimi_k3/reports/completion-audit.json"),
        help="Machine-readable completion audit report to write.",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        required=True,
        help="Full-mode preflight report to inspect.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/mini_kimi_k3/results"),
        help="Results root containing runs/<run-id>/<attempt-id>/.",
    )
    parser.add_argument("--run-id", required=True, help="Mini-K3 run id.")
    parser.add_argument("--attempt-id", required=True, help="Mini-K3 attempt id.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_completion_audit(
            report_path=args.report,
            preflight_report=args.preflight_report,
            results_root=args.results_root,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(f"Mini Kimi K3 completion audit: {report['status']}", file=sys.stderr)
    if report["status"] != "complete" and report["missing"]:
        print("missing: " + ", ".join(report["missing"]), file=sys.stderr)
        print("next actions:", file=sys.stderr)
        for action in report["next_actions"]:
            print(
                f"- {action['requirement']} -> {action['action']}",
                file=sys.stderr,
            )
            guidance = action.get("guidance")
            if isinstance(guidance, dict) and guidance:
                print(
                    f"  guidance: {_format_guidance(guidance)}",
                    file=sys.stderr,
                )
    print(f"wrote completion audit report: {args.report}", file=sys.stderr)
    return 0 if report["status"] == "complete" else 21


def run_completion_audit(
    *,
    report_path: Path,
    preflight_report: Path,
    results_root: Path,
    run_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise ValueError("Mini-K3 completion audit must run inside rootfs")

    preflight = _read_json(preflight_report)
    bundle_dir = Path(results_root) / "runs" / run_id / attempt_id
    outcome_path = bundle_dir / "outcome.json"
    events_path = bundle_dir / "processes" / "coordinator" / "events.jsonl"
    outcome = _read_json(outcome_path)
    events, event_stream_errors = _read_jsonl(events_path)

    gates = _gates_by_name(preflight)
    checks = [
        _preflight_ready_check(preflight_report, preflight),
        _r1_smoke_gpu_capacity_check(gates),
        _gate_status_check(gates, "trainable_config"),
        _gate_status_check(gates, "model_fidelity"),
        _gate_status_check(gates, "real_corpus_manifest"),
        _r1_corpus_plan_audit_check(gates),
        _corpus_disk_capacity_check(gates),
        _stage1_input_inspection_check(gates),
        _stage1_source_resolution_check(gates),
        _stage1_remote_readiness_check(gates),
        _launch_evidence_bundle_check(
            gates,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        _guarded_launch_completed_check(
            outcome_path,
            outcome,
            attempt_id=attempt_id,
        ),
        _training_measurement_check(
            outcome,
            events,
            event_stream_errors=event_stream_errors,
        ),
        _train_stage_succeeded_check(
            events_path,
            events,
            outcome=outcome,
            event_stream_errors=event_stream_errors,
        ),
    ]
    status = (
        "complete" if all(check["status"] == "pass" for check in checks) else "blocked"
    )
    checklist = _prompt_to_artifact_checklist(
        checks=checks,
        preflight_report=preflight_report,
        outcome_path=outcome_path,
        events_path=events_path,
    )
    summary = _checklist_summary(checklist)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mini_kimi_k3_completion_audit",
        "status": status,
        "objective": "proceed to launch the full Mini Kimi-K3 run end to end",
        "preflight_report": str(preflight_report),
        "attempt": {
            "results_root": str(results_root),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "bundle": str(bundle_dir),
        },
        "success_criteria": _success_criteria(checklist),
        "prompt_to_artifact_checklist": checklist,
        **summary,
        "next_actions": _next_actions(checklist=checklist, checks=checks),
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _preflight_ready_check(path: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "report": str(path),
        "schema_version": preflight.get("schema_version"),
        "kind": preflight.get("kind"),
        "mode": preflight.get("mode"),
        "status": preflight.get("status"),
    }
    status = (
        "pass"
        if preflight.get("schema_version") == 1
        and preflight.get("kind") == "mini_kimi_k3_preflight"
        and preflight.get("mode") == "full"
        and preflight.get("status") == "ready"
        else "fail"
    )
    detail = "full preflight must be ready"
    return _check("full_preflight_ready", status, detail, evidence)


def _gate_status_check(
    gates: dict[str, dict[str, Any]], gate_name: str
) -> dict[str, Any]:
    gate = gates.get(gate_name)
    evidence = {
        "gate": gate_name,
        "status": gate.get("status") if isinstance(gate, dict) else "missing",
    }
    if isinstance(gate, dict) and "detail" in gate:
        evidence["detail"] = gate["detail"]
    gate_evidence = gate.get("evidence") if isinstance(gate, dict) else None
    trainable_smoke_ready = True
    if (
        gate_name == "trainable_config"
        and isinstance(gate_evidence, dict)
        and isinstance(gate_evidence.get("r1_training_smoke"), dict)
    ):
        r1_training_smoke = _normalized_r1_training_smoke_evidence(
            gate_evidence["r1_training_smoke"]
        )
        evidence["r1_training_smoke"] = r1_training_smoke
        trainable_smoke_ready = _r1_training_smoke_ready(r1_training_smoke)
    elif gate_name == "trainable_config":
        trainable_smoke_ready = False
        evidence["r1_training_smoke"] = {"status": "missing"}
    status = (
        "pass" if evidence["status"] == "pass" and trainable_smoke_ready else "fail"
    )
    return _check(
        f"{gate_name}_passed", status, f"{gate_name} gate must pass", evidence
    )


def _normalized_r1_training_smoke_evidence(
    smoke_evidence: dict[str, Any],
) -> dict[str, Any]:
    evidence = dict(smoke_evidence)
    if not isinstance(evidence.get("rootfs"), dict):
        evidence["rootfs"] = {"status": "missing"}
    if not isinstance(evidence.get("command"), dict):
        evidence["command"] = {"status": "missing"}
    return evidence


def _r1_training_smoke_ready(smoke_evidence: dict[str, Any]) -> bool:
    rootfs = smoke_evidence.get("rootfs")
    command = smoke_evidence.get("command")
    argv = command.get("argv") if isinstance(command, dict) else None
    return (
        smoke_evidence.get("status") == "pass"
        and isinstance(smoke_evidence.get("steps"), int)
        and smoke_evidence["steps"] >= 1
        and isinstance(rootfs, dict)
        and rootfs.get("marker") == "1"
        and rootfs.get("cwd") == "/workspace/torchtitan"
        and isinstance(command, dict)
        and command.get("return_code") == 0
        and command.get("cwd") == "/workspace/torchtitan"
        and isinstance(argv, list)
        and "./run_train.sh" in argv
    )


def _launch_evidence_bundle_check(
    gates: dict[str, dict[str, Any]],
    *,
    run_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    gate = gates.get("launch_evidence_bundle")
    gate_evidence = gate.get("evidence") if isinstance(gate, dict) else None
    evidence = gate_evidence if isinstance(gate_evidence, dict) else {}
    if isinstance(gate, dict) and "status" not in evidence:
        evidence = {**evidence, "status": gate.get("status")}
    if not isinstance(gate, dict):
        evidence = {"status": "missing"}
    expected = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "family": "mini_kimi_k3",
        "task": "pretraining",
        "lane": "full",
        "model_variant": "r1",
    }
    mismatches = {
        key: {"expected": expected_value, "actual": evidence.get(key)}
        for key, expected_value in expected.items()
        if evidence.get(key) != expected_value
    }
    status = (
        "pass"
        if isinstance(gate, dict) and gate.get("status") == "pass" and not mismatches
        else "fail"
    )
    if mismatches:
        evidence = {**evidence, "mismatches": mismatches}
    return _check(
        "launch_evidence_bundle_passed",
        status,
        "launch_evidence_bundle gate must pass for this full r1 attempt",
        evidence,
    )


def _r1_smoke_gpu_capacity_check(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trainable_gate = gates.get("trainable_config")
    trainable_evidence = (
        trainable_gate.get("evidence") if isinstance(trainable_gate, dict) else None
    )
    smoke_evidence = (
        trainable_evidence.get("r1_training_smoke")
        if isinstance(trainable_evidence, dict)
        else None
    )
    gpu_evidence = (
        smoke_evidence.get("gpu_memory") if isinstance(smoke_evidence, dict) else None
    )
    evidence = gpu_evidence if isinstance(gpu_evidence, dict) else {"status": "missing"}
    status = (
        "pass"
        if isinstance(gpu_evidence, dict)
        and gpu_evidence.get("status") in {"pass", "ready"}
        and gpu_evidence.get("selected") is not None
        else "fail"
    )
    return _check(
        "r1_smoke_gpu_capacity_ready",
        status,
        "r1 Trainer smoke must have a selected GPU with enough free memory",
        evidence,
    )


def _corpus_disk_capacity_check(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corpus_gate = gates.get("real_corpus_manifest")
    corpus_evidence = (
        corpus_gate.get("evidence") if isinstance(corpus_gate, dict) else None
    )
    disk_evidence = (
        corpus_evidence.get("corpus_disk_readiness")
        if isinstance(corpus_evidence, dict)
        else None
    )
    evidence = (
        disk_evidence if isinstance(disk_evidence, dict) else {"status": "missing"}
    )
    status = (
        "pass"
        if isinstance(disk_evidence, dict)
        and disk_evidence.get("status") == "ready"
        and disk_evidence.get("deficit_bytes") == 0
        else "fail"
    )
    return _check(
        "raw_shard_disk_capacity_ready",
        status,
        "raw shard target filesystem must have capacity for the r1 token plan",
        evidence,
    )


def _r1_corpus_plan_audit_check(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corpus_gate = gates.get("real_corpus_manifest")
    corpus_evidence = (
        corpus_gate.get("evidence") if isinstance(corpus_gate, dict) else None
    )
    plan_evidence = (
        corpus_evidence.get("r1_corpus_plan_audit")
        if isinstance(corpus_evidence, dict)
        else None
    )
    evidence = (
        plan_evidence if isinstance(plan_evidence, dict) else {"status": "missing"}
    )
    status = (
        "pass"
        if evidence.get("status") == "ready" and evidence.get("deficit_tokens") == 0
        else "fail"
    )
    return _check(
        "r1_corpus_plan_audit_ready",
        status,
        "r1 corpus plan audit must be ready with no token deficit",
        evidence,
    )


def _stage1_input_inspection_check(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corpus_gate = gates.get("real_corpus_manifest")
    corpus_evidence = (
        corpus_gate.get("evidence") if isinstance(corpus_gate, dict) else None
    )
    required = (
        corpus_evidence.get("requires_stage1_input_inspection") is True
        if isinstance(corpus_evidence, dict)
        else False
    )
    input_evidence = (
        corpus_evidence.get("stage1_input_inspection")
        if isinstance(corpus_evidence, dict)
        else None
    )
    if not required:
        evidence = {
            "status": "not_required",
            "requires_stage1_input_inspection": required,
        }
        if isinstance(corpus_evidence, dict) and "corpus_route" in corpus_evidence:
            evidence["corpus_route"] = corpus_evidence["corpus_route"]
        return _check(
            "local_stage1_input_inspection_ready",
            "pass",
            "local Stage 1 input inspection is required only for local Stage 1 routes",
            evidence,
        )

    evidence = (
        input_evidence if isinstance(input_evidence, dict) else {"status": "missing"}
    )
    min_total_tokens = evidence.get("min_total_tokens")
    total_tokens = evidence.get("total_tokens")
    status = (
        "pass"
        if evidence.get("status") == "pass"
        and isinstance(min_total_tokens, int)
        and isinstance(total_tokens, int)
        and total_tokens >= min_total_tokens
        else "fail"
    )
    return _check(
        "local_stage1_input_inspection_ready",
        status,
        "local Stage 1 corpus route must include passing input inspection evidence",
        evidence,
    )


def _stage1_source_resolution_check(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corpus_gate = gates.get("real_corpus_manifest")
    corpus_evidence = (
        corpus_gate.get("evidence") if isinstance(corpus_gate, dict) else None
    )
    source_evidence = (
        corpus_evidence.get("stage1_source_resolution")
        if isinstance(corpus_evidence, dict)
        else None
    )
    evidence = (
        source_evidence if isinstance(source_evidence, dict) else {"status": "missing"}
    )
    unresolved_benchmarks = evidence.get("unresolved_benchmarks")
    unresolved_corpora = evidence.get("unresolved_corpora")
    status = (
        "pass"
        if evidence.get("status") == "ready"
        and unresolved_benchmarks == []
        and unresolved_corpora == []
        else "fail"
    )
    return _check(
        "stage1_source_resolution_ready",
        status,
        "Stage 1 source resolution must be ready with no unresolved sources",
        evidence,
    )


def _stage1_remote_readiness_check(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corpus_gate = gates.get("real_corpus_manifest")
    corpus_evidence = (
        corpus_gate.get("evidence") if isinstance(corpus_gate, dict) else None
    )
    remote_evidence = (
        corpus_evidence.get("stage1_remote_readiness")
        if isinstance(corpus_evidence, dict)
        else None
    )
    evidence = (
        remote_evidence if isinstance(remote_evidence, dict) else {"status": "missing"}
    )
    if isinstance(remote_evidence, dict):
        evidence = {
            **remote_evidence,
            "rootfs_marker": remote_evidence.get("rootfs_marker"),
            "network_mode": remote_evidence.get("network_mode"),
        }
    status = (
        "pass"
        if isinstance(remote_evidence, dict)
        and (
            remote_evidence.get("status") == "not_required"
            or (
                remote_evidence.get("status") == "ready"
                and remote_evidence.get("huggingface_dns") == "pass"
                and remote_evidence.get("modal_cli") == "pass"
                and remote_evidence.get("modal_auth") == "pass"
                and remote_evidence.get("rootfs_marker") == "1"
                and remote_evidence.get("network_mode") == "networked"
            )
        )
        else "fail"
    )
    return _check(
        "stage1_remote_readiness_ready",
        status,
        "remote Stage 1 corpus path must be ready or explicitly not required",
        evidence,
    )


def _guarded_launch_completed_check(
    path: Path,
    outcome: dict[str, Any],
    *,
    attempt_id: str,
) -> dict[str, Any]:
    evidence = {
        "report": str(path),
        "schema_version": outcome.get("schema_version"),
        "kind": outcome.get("kind"),
        "attempt_id": outcome.get("attempt_id"),
        "execution_outcome": outcome.get("execution_outcome"),
    }
    status = (
        "pass"
        if outcome.get("schema_version") == 1
        and outcome.get("kind") == "attempt_outcome"
        and outcome.get("attempt_id") == attempt_id
        and outcome.get("execution_outcome") == "completed"
        else "fail"
    )
    return _check(
        "guarded_launch_completed",
        status,
        "guarded launch attempt must complete",
        evidence,
    )


def _training_measurement_check(
    outcome: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    event_stream_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    training = outcome.get("evaluations", {}).get("training")
    outcome_stage_invocation_ids = _outcome_stage_invocation_ids(outcome)
    training_stage_invocation_id = (
        training.get("stage_invocation_id") if isinstance(training, dict) else None
    )
    matched_train_stage_invocation_ids = _successful_train_stage_invocation_ids(
        events, outcome_stage_invocation_ids=outcome_stage_invocation_ids
    )
    training_invocation_matches = (
        isinstance(training_stage_invocation_id, str)
        and training_stage_invocation_id in matched_train_stage_invocation_ids
    )
    run_gate = outcome.get("run_gate")
    has_real_measurement = (
        run_gate.get("has_real_measurement") is True
        if isinstance(run_gate, dict)
        else False
    )
    evidence = training.copy() if isinstance(training, dict) else {"status": "missing"}
    evidence["outcome_stage_invocation_ids"] = outcome_stage_invocation_ids
    evidence["matched_train_stage_invocation_ids"] = matched_train_stage_invocation_ids
    evidence["stage_invocation_id"] = training_stage_invocation_id
    evidence["stage_invocation_match"] = training_invocation_matches
    evidence["run_gate"] = (
        run_gate if isinstance(run_gate, dict) else {"status": "missing"}
    )
    if event_stream_errors:
        evidence["event_stream_errors"] = event_stream_errors
    status = (
        "pass"
        if isinstance(training, dict)
        and training.get("execution_outcome") == "completed"
        and training.get("measurement") == "real"
        and training_invocation_matches
        and has_real_measurement
        and not event_stream_errors
        else "fail"
    )
    return _check(
        "training_measurement_real",
        status,
        "training evaluation must be a completed real measurement",
        evidence,
    )


def _train_stage_succeeded_check(
    path: Path,
    events: list[dict[str, Any]],
    *,
    outcome: dict[str, Any],
    event_stream_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    train_events = [event for event in events if event.get("stage_id") == "train"]
    outcome_stage_invocation_ids = _outcome_stage_invocation_ids(outcome)
    evidence = {
        "event_stream": str(path),
        "outcome_stage_invocation_ids": outcome_stage_invocation_ids,
        "terminal_events": [
            {
                "kind": event.get("kind"),
                "stage_invocation_id": event.get("stage_invocation_id"),
                "return_code": event.get("return_code"),
            }
            for event in events
            if event.get("kind") != "stage_started"
        ],
        "train_terminal_events": [
            {
                "kind": event.get("kind"),
                "stage_invocation_id": event.get("stage_invocation_id"),
                "return_code": event.get("return_code"),
            }
            for event in train_events
            if event.get("kind") != "stage_started"
        ],
    }
    if event_stream_errors:
        evidence["event_stream_errors"] = event_stream_errors
    status = (
        "pass"
        if any(
            event.get("kind") == "stage_succeeded"
            and event.get("return_code") == 0
            and event.get("stage_invocation_id") in outcome_stage_invocation_ids
            for event in train_events
        )
        and not event_stream_errors
        else "fail"
    )
    return _check(
        "train_stage_succeeded",
        status,
        "coordinator event stream must include a successful train stage",
        evidence,
    )


def _outcome_stage_invocation_ids(outcome: dict[str, Any]) -> list[str]:
    stage_invocation_ids = outcome.get("stage_invocation_ids")
    if not isinstance(stage_invocation_ids, list) or not all(
        isinstance(stage_invocation_id, str)
        for stage_invocation_id in stage_invocation_ids
    ):
        return []
    return stage_invocation_ids


def _successful_train_stage_invocation_ids(
    events: list[dict[str, Any]],
    *,
    outcome_stage_invocation_ids: list[str],
) -> list[str]:
    return [
        event["stage_invocation_id"]
        for event in events
        if event.get("stage_id") == "train"
        and event.get("kind") == "stage_succeeded"
        and event.get("return_code") == 0
        and isinstance(event.get("stage_invocation_id"), str)
        and event["stage_invocation_id"] in outcome_stage_invocation_ids
    ]


def _gates_by_name(preflight: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gates = preflight.get("gates")
    if not isinstance(gates, list):
        return {}
    return {
        gate["name"]: gate
        for gate in gates
        if isinstance(gate, dict) and isinstance(gate.get("name"), str)
    }


def _check(
    name: str,
    status: str,
    detail: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "evidence": evidence,
    }


def _prompt_to_artifact_checklist(
    *,
    checks: list[dict[str, Any]],
    preflight_report: Path,
    outcome_path: Path,
    events_path: Path,
) -> list[dict[str, Any]]:
    checks_by_name = {check["name"]: check for check in checks}
    return [
        _checklist_item(
            requirement="full_preflight_ready",
            evidence=str(preflight_report),
            checks=[checks_by_name["full_preflight_ready"]],
        ),
        _checklist_item(
            requirement="r1_smoke_gpu_capacity",
            evidence=str(preflight_report),
            checks=[checks_by_name["r1_smoke_gpu_capacity_ready"]],
        ),
        _checklist_item(
            requirement="r1_trainer_optimizer_step",
            evidence=str(preflight_report),
            checks=[checks_by_name["trainable_config_passed"]],
        ),
        _checklist_item(
            requirement="model_fidelity",
            evidence=str(preflight_report),
            checks=[checks_by_name["model_fidelity_passed"]],
        ),
        _checklist_item(
            requirement="real_5b_corpus_manifest",
            evidence=str(preflight_report),
            checks=[checks_by_name["real_corpus_manifest_passed"]],
        ),
        _checklist_item(
            requirement="r1_corpus_plan_audit",
            evidence=str(preflight_report),
            checks=[checks_by_name["r1_corpus_plan_audit_ready"]],
        ),
        _checklist_item(
            requirement="raw_shard_disk_capacity",
            evidence=str(preflight_report),
            checks=[checks_by_name["raw_shard_disk_capacity_ready"]],
        ),
        _checklist_item(
            requirement="local_stage1_input_inspection",
            evidence=str(preflight_report),
            checks=[checks_by_name["local_stage1_input_inspection_ready"]],
        ),
        _checklist_item(
            requirement="stage1_source_resolution",
            evidence=str(preflight_report),
            checks=[checks_by_name["stage1_source_resolution_ready"]],
        ),
        _checklist_item(
            requirement="stage1_remote_readiness",
            evidence=str(preflight_report),
            checks=[checks_by_name["stage1_remote_readiness_ready"]],
        ),
        _checklist_item(
            requirement="launch_evidence_bundle",
            evidence=str(preflight_report),
            checks=[checks_by_name["launch_evidence_bundle_passed"]],
        ),
        _checklist_item(
            requirement="guarded_launch_completed",
            evidence=str(outcome_path),
            checks=[checks_by_name["guarded_launch_completed"]],
        ),
        _checklist_item(
            requirement="training_measurement_real",
            evidence=str(outcome_path),
            checks=[checks_by_name["training_measurement_real"]],
        ),
        _checklist_item(
            requirement="train_stage_succeeded",
            evidence=str(events_path),
            checks=[checks_by_name["train_stage_succeeded"]],
        ),
    ]


def _checklist_item(
    *,
    requirement: str,
    evidence: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    status = (
        "covered"
        if checks and all(check["status"] == "pass" for check in checks)
        else "missing"
    )
    if (
        status == "covered"
        and checks
        and all(
            isinstance(check.get("evidence"), dict)
            and check["evidence"].get("status") == "not_required"
            for check in checks
        )
    ):
        status = "not_required"
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "checks": [check["name"] for check in checks],
    }


def _checklist_summary(checklist: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        status: [
            item["requirement"] for item in checklist if item.get("status") == status
        ]
        for status in ("missing", "covered", "not_required")
    }


def _success_criteria(checklist: list[dict[str, Any]]) -> list[str]:
    return [item["requirement"] for item in checklist]


_NEXT_ACTIONS = {
    "full_preflight_ready": (
        "Refresh the full preflight after the missing prerequisite artifacts are ready."
    ),
    "r1_smoke_gpu_capacity": (
        "Free or select a GPU with enough memory for the r1 Trainer smoke, then "
        "rerun the r1 smoke and full preflight."
    ),
    "r1_trainer_optimizer_step": (
        "Produce r1 Trainer-smoke evidence with at least one real optimizer step."
    ),
    "model_fidelity": (
        "Regenerate and review the strict r1 forward-oracle and launch-backend "
        "evidence."
    ),
    "real_5b_corpus_manifest": (
        "Build or import the full 5B-token Kimi-tokenized corpus manifest with "
        "provenance and decontamination evidence."
    ),
    "r1_corpus_plan_audit": (
        "Refresh the r1 corpus-plan audit after the manifest covers the full "
        "5B-token target."
    ),
    "raw_shard_disk_capacity": (
        "Provide enough filesystem capacity for the r1 raw token shard plan and "
        "refresh disk readiness."
    ),
    "local_stage1_input_inspection": (
        "Run inspect-stage1-inputs for the local Stage 1 input manifest and "
        "refresh full preflight with the passing report."
    ),
    "stage1_source_resolution": (
        "Refresh the Stage 1 source-resolution report until every benchmark and "
        "corpus source resolves."
    ),
    "stage1_remote_readiness": (
        "Authenticate Modal inside the rootfs or provide a ready local corpus path, "
        "then refresh Stage 1 readiness."
    ),
    "launch_evidence_bundle": (
        "Initialize a full-mode immutable launch evidence bundle for the guarded "
        "attempt."
    ),
    "guarded_launch_completed": (
        "Run the guarded launch only after full preflight is ready and preserve "
        "its outcome artifact."
    ),
    "training_measurement_real": (
        "Complete a real training stage so the guarded attempt records a real "
        "measurement."
    ),
    "train_stage_succeeded": (
        "Complete the train stage successfully so the coordinator event stream "
        "records stage_succeeded for train."
    ),
}


def _next_actions(
    *,
    checklist: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks_by_name = {check["name"]: check for check in checks}
    actions = []
    for item in checklist:
        if item["status"] in {"covered", "not_required"}:
            continue
        item_checks = [checks_by_name[name] for name in item["checks"]]
        guidance = _next_action_guidance(
            requirement=item["requirement"],
            item_evidence=item["evidence"],
            blocking_checks=item_checks,
            checks_by_name=checks_by_name,
            checklist=checklist,
        )
        action = {
            "requirement": item["requirement"],
            "action": _NEXT_ACTIONS[item["requirement"]],
            "evidence": item["evidence"],
            "blocking_checks": [
                {
                    "name": check["name"],
                    "detail": check["detail"],
                    "evidence": check["evidence"],
                }
                for check in item_checks
                if check["status"] != "pass"
            ],
        }
        if guidance:
            action["guidance"] = guidance
        actions.append(action)
    return actions


def _next_action_guidance(
    *,
    requirement: str,
    item_evidence: str,
    blocking_checks: list[dict[str, Any]],
    checks_by_name: dict[str, dict[str, Any]],
    checklist: list[dict[str, Any]],
) -> dict[str, Any]:
    if requirement == "full_preflight_ready":
        return _full_preflight_guidance(checklist)
    if requirement == "real_5b_corpus_manifest":
        return _corpus_token_guidance(
            [checks_by_name.get("r1_corpus_plan_audit_ready", {})]
        )
    if requirement == "r1_corpus_plan_audit":
        return _corpus_token_guidance(blocking_checks)
    if requirement == "r1_smoke_gpu_capacity":
        return _r1_smoke_gpu_guidance(
            [checks_by_name.get("trainable_config_passed", {})]
        )
    if requirement == "r1_trainer_optimizer_step":
        return _r1_training_smoke_guidance(
            [checks_by_name.get("trainable_config_passed", {})]
        )
    if requirement == "guarded_launch_completed":
        return _guarded_launch_guidance(item_evidence, blocking_checks)
    if requirement == "training_measurement_real":
        return _training_measurement_guidance(item_evidence, blocking_checks)
    if requirement == "train_stage_succeeded":
        return _train_stage_guidance(blocking_checks)
    if requirement != "stage1_remote_readiness":
        return {}
    for check in blocking_checks:
        evidence = check.get("evidence")
        if not isinstance(evidence, dict):
            continue
        env_file = evidence.get("env_file")
        if not isinstance(env_file, dict):
            continue
        guidance: dict[str, Any] = {}
        path = env_file.get("path")
        if isinstance(path, str) and path:
            guidance["env_file"] = path
        template = env_file.get("template")
        if isinstance(template, str) and template:
            guidance["env_file_template"] = template
        required_keys = env_file.get("required_keys")
        if isinstance(required_keys, list) and all(
            isinstance(key, str) for key in required_keys
        ):
            guidance["required_keys"] = required_keys
        return guidance
    return {}


def _format_guidance(guidance: dict[str, Any]) -> str:
    parts = []
    for key in sorted(guidance):
        value = guidance[key]
        if isinstance(value, str):
            rendered = value
        elif value is None:
            rendered = "null"
        else:
            rendered = json.dumps(value, sort_keys=True)
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _full_preflight_guidance(checklist: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_preflight_requirements = [
        item["requirement"]
        for item in checklist
        if item.get("status") == "missing"
        and item.get("requirement")
        not in {
            "full_preflight_ready",
            "guarded_launch_completed",
            "training_measurement_real",
            "train_stage_succeeded",
        }
    ]
    if not blocked_preflight_requirements:
        return {}
    return {"blocked_preflight_requirements": blocked_preflight_requirements}


def _corpus_token_guidance(blocking_checks: list[dict[str, Any]]) -> dict[str, Any]:
    for check in blocking_checks:
        evidence = check.get("evidence")
        if not isinstance(evidence, dict):
            continue
        guidance = {
            key: evidence[key]
            for key in ("target_tokens", "available_tokens", "deficit_tokens")
            if isinstance(evidence.get(key), int)
        }
        source_deficits = _source_deficit_guidance(evidence)
        if source_deficits:
            guidance["source_deficits"] = source_deficits
        if guidance:
            return guidance
    return {}


def _source_deficit_guidance(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = evidence.get("sources")
    if not isinstance(sources, dict):
        return {}
    deficits: dict[str, dict[str, Any]] = {}
    for source, source_evidence in sources.items():
        if not isinstance(source, str) or not isinstance(source_evidence, dict):
            continue
        deficit_tokens = source_evidence.get("deficit_tokens")
        if not isinstance(deficit_tokens, int) or deficit_tokens <= 0:
            continue
        item = {
            key: source_evidence[key]
            for key in (
                "available_tokens",
                "deficit_tokens",
                "required_tokens",
                "status",
            )
            if key in source_evidence
        }
        deficits[source] = item
    return dict(sorted(deficits.items()))


def _r1_training_smoke_evidence(
    blocking_checks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for check in blocking_checks:
        evidence = check.get("evidence")
        if not isinstance(evidence, dict):
            continue
        smoke = evidence.get("r1_training_smoke")
        if isinstance(smoke, dict):
            return smoke
    return None


def _r1_smoke_gpu_guidance(blocking_checks: list[dict[str, Any]]) -> dict[str, Any]:
    smoke = _r1_training_smoke_evidence(blocking_checks)
    if not isinstance(smoke, dict):
        return {}
    gpu_memory = smoke.get("gpu_memory")
    if not isinstance(gpu_memory, dict):
        return {}
    guidance: dict[str, Any] = {}
    report = smoke.get("report")
    if isinstance(report, str) and report:
        guidance["r1_training_smoke_report"] = report
    required_free_mib = gpu_memory.get("required_free_mib")
    if isinstance(required_free_mib, int):
        guidance["required_free_mib"] = required_free_mib
    if "selected" in gpu_memory:
        guidance["selected"] = gpu_memory["selected"]
    status = gpu_memory.get("status")
    if isinstance(status, str) and status:
        guidance["status"] = status
    return guidance


def _r1_training_smoke_guidance(
    blocking_checks: list[dict[str, Any]]
) -> dict[str, Any]:
    smoke = _r1_training_smoke_evidence(blocking_checks)
    if not isinstance(smoke, dict):
        return {}
    guidance: dict[str, Any] = {}
    report = smoke.get("report")
    if isinstance(report, str) and report:
        guidance["r1_training_smoke_report"] = report
    steps = smoke.get("steps")
    if isinstance(steps, int):
        guidance["steps"] = steps
    status = smoke.get("status")
    if isinstance(status, str) and status:
        guidance["status"] = status
    return guidance


def _guarded_launch_guidance(
    item_evidence: str, blocking_checks: list[dict[str, Any]]
) -> dict[str, Any]:
    for check in blocking_checks:
        evidence = check.get("evidence")
        if not isinstance(evidence, dict):
            continue
        guidance: dict[str, Any] = {"attempt_outcome_report": item_evidence}
        kind = evidence.get("kind")
        if isinstance(kind, str) and kind:
            guidance["kind"] = kind
        execution_outcome = evidence.get("execution_outcome")
        if isinstance(execution_outcome, str) or execution_outcome is None:
            guidance["execution_outcome"] = execution_outcome
        status = check.get("status")
        if isinstance(status, str) and status:
            guidance["status"] = status
        return guidance
    return {}


def _training_measurement_guidance(
    item_evidence: str, blocking_checks: list[dict[str, Any]]
) -> dict[str, Any]:
    for check in blocking_checks:
        evidence = check.get("evidence")
        if not isinstance(evidence, dict):
            continue
        guidance: dict[str, Any] = {"attempt_outcome_report": item_evidence}
        execution_outcome = evidence.get("execution_outcome")
        if isinstance(execution_outcome, str) or execution_outcome is None:
            guidance["execution_outcome"] = execution_outcome
        measurement = evidence.get("measurement")
        if isinstance(measurement, str) or measurement is None:
            guidance["measurement"] = measurement
        evidence_status = evidence.get("status")
        if isinstance(evidence_status, str) and evidence_status:
            guidance["status"] = evidence_status
        else:
            status = check.get("status")
            if isinstance(status, str) and status:
                guidance["status"] = status
        event_stream_errors = evidence.get("event_stream_errors")
        if isinstance(event_stream_errors, list):
            guidance["event_stream_errors"] = event_stream_errors
        return guidance
    return {}


def _train_stage_guidance(blocking_checks: list[dict[str, Any]]) -> dict[str, Any]:
    for check in blocking_checks:
        evidence = check.get("evidence")
        if not isinstance(evidence, dict):
            continue
        guidance: dict[str, Any] = {}
        event_stream = evidence.get("event_stream")
        if isinstance(event_stream, str) and event_stream:
            guidance["event_stream"] = event_stream
        status = check.get("status")
        if isinstance(status, str) and status:
            guidance["status"] = status
        terminal_events = evidence.get("terminal_events")
        if isinstance(terminal_events, list):
            guidance["terminal_events"] = terminal_events
        event_stream_errors = evidence.get("event_stream_errors")
        if isinstance(event_stream_errors, list):
            guidance["event_stream_errors"] = event_stream_errors
        return guidance
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {"status": "missing", "path": str(path)}
    except json.JSONDecodeError as exc:
        return {"status": "unreadable", "path": str(path), "error": str(exc)}
    if not isinstance(data, dict):
        return {"status": "unreadable", "path": str(path), "error": "not an object"}
    return data


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return [], [{"line": None, "error": f"missing file: {path}"}]
    events = []
    errors = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": line_number, "error": exc.msg})
            continue
        if not isinstance(event, dict):
            errors.append({"line": line_number, "error": "not an object"})
            continue
        events.append(event)
    return events, errors


if __name__ == "__main__":
    raise SystemExit(main())
