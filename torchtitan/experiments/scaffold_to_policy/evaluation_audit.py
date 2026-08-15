# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Audit helpers for scaffold-to-policy benchmark evidence.

This module is intentionally host-testable. It validates already-written JSON
artifacts and does not rerun model generation, training, or external harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from torchtitan.experiments.execution import models


AUDIT_STATUSES = ("pass", "warn", "fail")


@dataclass(frozen=True)
class AuditFinding:
    audit_id: str
    status: str
    scope: str
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if self.status not in AUDIT_STATUSES:
            raise ValueError(f"invalid audit status: {self.status!r}")

    @property
    def blocking(self) -> bool:
        return self.status == "fail"

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "audit_id": self.audit_id,
            "status": self.status,
            "scope": self.scope,
            "message": self.message,
            "blocking": self.blocking,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


def audit_paths(
    *,
    report_inputs: Iterable[Path] = (),
    attempt_dirs: Iterable[Path] = (),
) -> dict[str, object]:
    findings: list[AuditFinding] = []
    report_input_paths = list(report_inputs)
    attempt_dir_paths = list(attempt_dirs)
    if not report_input_paths and not attempt_dir_paths:
        findings.append(
            _fail("audit.inputs", "no report inputs or attempt dirs matched", None)
        )
    for path in report_input_paths:
        findings.extend(audit_report_input(path))
    for path in attempt_dir_paths:
        findings.extend(audit_attempt_dir(path))
    return build_audit_report(findings)


def build_audit_report(findings: Iterable[AuditFinding]) -> dict[str, object]:
    finding_list = list(findings)
    status_counts = {status: 0 for status in AUDIT_STATUSES}
    for finding in finding_list:
        status_counts[finding.status] += 1
    return {
        "schema_version": 1,
        "kind": "scaffold_to_policy_evaluation_audit",
        "selected": status_counts["fail"] == 0,
        "status_counts": status_counts,
        "num_findings": len(finding_list),
        "findings": [finding.to_dict() for finding in finding_list],
    }


def audit_report_input(path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    payload = _load_object(path, findings, scope="report")
    if payload is None:
        return findings

    run_id = _run_id(payload)
    if run_id is None:
        findings.append(_fail("report.run_id", "report has no run.run_id", path))
    else:
        findings.append(
            _pass("report.run_id", f"report run_id is {run_id}", path, scope="report")
        )

    _audit_report_checks(payload, path, findings)
    _audit_metrics(payload, path, findings)
    _audit_artifact_records(payload, path, findings)
    return findings


def audit_attempt_dir(path: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    manifest_path = path / "manifest.json"
    outcome_path = path / "outcome.json"
    report_path = path / "derived" / "report_input.json"
    events_path = path / "processes" / "coordinator" / "events.jsonl"

    manifest = _load_required_object(manifest_path, findings, scope="attempt")
    outcome = _load_required_object(outcome_path, findings, scope="attempt")
    report = _load_required_object(report_path, findings, scope="attempt")

    if report is not None:
        findings.extend(audit_report_input(report_path))

    if manifest is not None and outcome is not None:
        _audit_attempt_identity(manifest, outcome, report, path, findings)
        _audit_attempt_outcome(outcome, outcome_path, findings)

    if outcome is not None:
        _audit_event_stream(events_path, outcome, findings)

    return findings


def _audit_report_checks(
    payload: dict[str, object],
    path: Path,
    findings: list[AuditFinding],
) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        findings.append(_fail("report.checks", "report has no checks object", path))
        return
    if not checks:
        findings.append(_warn("report.checks", "report checks object is empty", path))
        return

    false_checks = [name for name, value in checks.items() if not bool(value)]
    if not false_checks:
        findings.append(_pass("report.checks", "all report checks are true", path))
        return

    blockers = payload.get("blockers")
    if _is_typed_blocker(false_checks, blockers):
        findings.append(
            _warn(
                "report.typed_blocker",
                "benchmark execution did not complete, but blocker artifacts are present",
                path,
            )
        )
        return

    allowed_score_zero_checks = {
        "task_execution_probes_succeeded",
        "task_score_smokes_succeeded",
    }
    if set(false_checks) <= allowed_score_zero_checks:
        findings.append(
            _warn(
                "report.score_zero_probe",
                "external harness probe completed with an unsuccessful score",
                path,
            )
        )
        return

    findings.append(
        _fail(
            "report.checks_failed",
            "report checks failed: " + ", ".join(sorted(false_checks)),
            path,
        )
    )


def _is_typed_blocker(false_checks: list[str], blockers: object) -> bool:
    if not isinstance(blockers, dict):
        return False
    allowed_false_checks = {
        "benchmark_execution_completed",
        "blocker_selected",
        "preflight_ready",
    }
    return bool(false_checks) and set(false_checks) <= allowed_false_checks


def _audit_metrics(
    payload: dict[str, object],
    path: Path,
    findings: list[AuditFinding],
) -> None:
    metrics = payload.get("metrics")
    if metrics is None:
        findings.append(_warn("report.metrics", "report has no metrics section", path))
        return
    if not isinstance(metrics, dict):
        findings.append(_fail("report.metrics", "metrics is not an object", path))
        return
    splits = metrics.get("splits")
    if splits is None:
        findings.append(_warn("report.metrics.splits", "report has no split metrics", path))
        return
    if not isinstance(splits, dict):
        findings.append(_fail("report.metrics.splits", "metrics.splits is not an object", path))
        return

    for split_name, summary in splits.items():
        if not isinstance(summary, dict):
            findings.append(
                _fail(
                    "report.metrics.split",
                    f"split {split_name!r} summary is not an object",
                    path,
                )
            )
            continue
        _audit_split_summary(str(split_name), summary, path, findings)


def _audit_split_summary(
    split_name: str,
    summary: dict[str, object],
    path: Path,
    findings: list[AuditFinding],
) -> None:
    num_problems = summary.get("num_problems")
    total_rollouts = summary.get("total_rollouts")
    pass_at_k = summary.get("pass_at_k")
    if not isinstance(num_problems, int) or num_problems < 0:
        findings.append(
            _fail("report.metrics.num_problems", f"{split_name} has invalid num_problems", path)
        )
    if not isinstance(total_rollouts, int) or total_rollouts < 0:
        findings.append(
            _fail("report.metrics.total_rollouts", f"{split_name} has invalid total_rollouts", path)
        )
    if not isinstance(pass_at_k, dict):
        findings.append(
            _fail("report.metrics.pass_at_k", f"{split_name} has no pass_at_k object", path)
        )
        return
    if isinstance(num_problems, int) and num_problems > 0 and isinstance(total_rollouts, int):
        rollouts_per_problem = total_rollouts // num_problems
        over_budget = []
        for key in pass_at_k:
            try:
                k = int(key)
            except ValueError:
                findings.append(
                    _fail("report.metrics.pass_at_k", f"{split_name} has non-integer pass@k key {key!r}", path)
                )
                continue
            if k > rollouts_per_problem:
                over_budget.append(k)
        if over_budget:
            findings.append(
                _warn(
                    "report.pass_at_k_compatibility",
                    f"{split_name} reports pass@k beyond {rollouts_per_problem} rollouts per problem: {sorted(over_budget)}",
                    path,
                )
            )
        else:
            findings.append(
                _pass("report.pass_at_k_budget", f"{split_name} pass@k fits rollout budget", path)
            )


def _audit_artifact_records(
    payload: dict[str, object],
    path: Path,
    findings: list[AuditFinding],
) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        findings.append(_warn("report.artifacts", "report has no artifacts object", path))
        return
    records = list(_iter_artifact_records(artifacts))
    if not records:
        findings.append(_warn("report.artifacts.records", "report has no artifact records", path))
        return
    missing = []
    unlabeled = []
    unhashed = []
    digest_mismatches = []
    for record in records:
        record_path = record.get("path")
        actual_path = Path(record_path) if isinstance(record_path, str) else None
        if actual_path is None or not actual_path.is_file():
            missing.append(record)
            continue
        if not bool(record.get("exists")):
            missing.append(record)
        if not isinstance(record.get("run_binding"), dict):
            unlabeled.append(record)
        recorded_sha = record.get("sha256")
        if not isinstance(recorded_sha, str):
            unhashed.append(record)
        elif _sha256(actual_path) != recorded_sha:
            digest_mismatches.append(record)
    if missing:
        findings.append(
            _fail("report.artifacts.missing", f"{len(missing)} referenced artifacts are missing", path)
        )
    if unlabeled:
        findings.append(
            _fail("report.artifacts.unlabeled", f"{len(unlabeled)} artifact records lack run binding", path)
        )
    if unhashed:
        findings.append(
            _fail("report.artifacts.unhashed", f"{len(unhashed)} existing artifact records lack sha256", path)
        )
    if digest_mismatches:
        findings.append(
            _fail(
                "report.artifacts.sha256",
                f"{len(digest_mismatches)} artifact digests do not match current files",
                path,
            )
        )
    if not missing and not unlabeled and not unhashed and not digest_mismatches:
        findings.append(
            _pass("report.artifacts", f"{len(records)} artifact records are present and labeled", path)
        )


def _audit_attempt_identity(
    manifest: dict[str, object],
    outcome: dict[str, object],
    report: dict[str, object] | None,
    path: Path,
    findings: list[AuditFinding],
) -> None:
    manifest_run = manifest.get("run")
    manifest_attempt = manifest.get("attempt")
    run_id = manifest_run.get("run_id") if isinstance(manifest_run, dict) else None
    attempt_id = (
        manifest_attempt.get("attempt_id") if isinstance(manifest_attempt, dict) else None
    )
    if not isinstance(run_id, str) or not run_id:
        findings.append(_fail("attempt.run_id", "manifest has no run.run_id", path))
    if not isinstance(attempt_id, str) or not attempt_id:
        findings.append(_fail("attempt.attempt_id", "manifest has no attempt.attempt_id", path))
    if isinstance(attempt_id, str) and outcome.get("attempt_id") != attempt_id:
        findings.append(_fail("attempt.identity", "outcome attempt_id does not match manifest", path))
    if report is not None and isinstance(run_id, str) and _run_id(report) != run_id:
        findings.append(_fail("attempt.identity", "report run_id does not match manifest", path))
    if isinstance(run_id, str) and isinstance(attempt_id, str):
        findings.append(_pass("attempt.identity", "manifest, outcome, and report identities match", path))


def _audit_attempt_outcome(
    outcome: dict[str, object],
    path: Path,
    findings: list[AuditFinding],
) -> None:
    execution_outcome = outcome.get("execution_outcome")
    if execution_outcome not in models.ATTEMPT_EXECUTION_OUTCOMES:
        findings.append(_fail("attempt.outcome", "invalid attempt execution_outcome", path))
    evaluations = outcome.get("evaluations")
    if not isinstance(evaluations, dict) or not evaluations:
        findings.append(_fail("attempt.evaluations", "outcome has no evaluations object", path))
        return
    for name, status in evaluations.items():
        if not isinstance(status, dict):
            findings.append(_fail("attempt.evaluations", f"{name} status is not an object", path))
            continue
        try:
            models.ConditionStatus(**status)
        except TypeError as exc:
            findings.append(_fail("attempt.evaluations", f"{name} status has invalid fields: {exc}", path))
        except ValueError as exc:
            findings.append(_fail("attempt.evaluations", f"{name} status is invalid: {exc}", path))
    if not any(f.audit_id == "attempt.evaluations" and f.status == "fail" for f in findings):
        findings.append(_pass("attempt.evaluations", "all condition statuses are valid", path))


def _audit_event_stream(
    path: Path,
    outcome: dict[str, object],
    findings: list[AuditFinding],
) -> None:
    if not path.is_file():
        findings.append(_fail("attempt.events", "missing coordinator events", path))
        return
    if path.stat().st_size == 0:
        findings.append(_fail("attempt.events", "empty coordinator events", path))
        return
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append(
                _fail("attempt.events", f"invalid JSON on line {line_number}: {exc}", path)
            )
            return
        if not isinstance(row, dict):
            findings.append(
                _fail("attempt.events", f"event line {line_number} is not an object", path)
            )
            return
        rows.append(row)

    started: dict[str, dict[str, object]] = {}
    terminal: dict[str, dict[str, object]] = {}
    for row in rows:
        kind = row.get("kind")
        invocation_id = row.get("stage_invocation_id")
        if kind not in models.STAGE_EVENT_KINDS:
            findings.append(_fail("attempt.events", f"unknown event kind {kind!r}", path))
            return
        if not isinstance(invocation_id, str) or not invocation_id:
            findings.append(_fail("attempt.events", "event missing stage_invocation_id", path))
            return
        if kind == "stage_started":
            if invocation_id in started:
                findings.append(_fail("attempt.events", "duplicate stage_started event", path))
                return
            started[invocation_id] = row
        else:
            if invocation_id in terminal:
                findings.append(_fail("attempt.events", "duplicate terminal stage event", path))
                return
            terminal[invocation_id] = row

    if set(started) != set(terminal):
        findings.append(
            _fail("attempt.events", "stage_started and terminal events are not paired", path)
        )
        return
    outcome_invocations = outcome.get("stage_invocation_ids")
    if not isinstance(outcome_invocations, list) or not all(
        isinstance(value, str) and value for value in outcome_invocations
    ):
        findings.append(
            _fail("attempt.events", "outcome has invalid stage_invocation_ids", path)
        )
        return
    if set(outcome_invocations) != set(started):
        findings.append(
            _fail(
                "attempt.events",
                "outcome stage_invocation_ids do not match coordinator events",
                path,
            )
        )
        return
    findings.append(_pass("attempt.events", "coordinator events are paired", path))


def _load_required_object(
    path: Path,
    findings: list[AuditFinding],
    *,
    scope: str,
) -> dict[str, object] | None:
    if not path.is_file():
        findings.append(_fail(f"{scope}.missing", "missing required JSON artifact", path))
        return None
    return _load_object(path, findings, scope=scope)


def _load_object(
    path: Path,
    findings: list[AuditFinding],
    *,
    scope: str,
) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        findings.append(_fail(f"{scope}.json", f"invalid JSON: {exc}", path))
        return None
    if not isinstance(payload, dict):
        findings.append(_fail(f"{scope}.json", "JSON payload is not an object", path))
        return None
    return payload


def _run_id(payload: dict[str, object]) -> str | None:
    run = payload.get("run")
    if not isinstance(run, dict):
        return None
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    return run_id


def _iter_artifact_records(value: object):
    if isinstance(value, dict):
        if "path" in value and "exists" in value:
            yield value
            return
        for child in value.values():
            yield from _iter_artifact_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_artifact_records(child)


def _pass(audit_id: str, message: str, path: Path, *, scope: str = "report") -> AuditFinding:
    return AuditFinding(audit_id, "pass", scope, message, str(path))


def _warn(audit_id: str, message: str, path: Path, *, scope: str = "report") -> AuditFinding:
    return AuditFinding(audit_id, "warn", scope, message, str(path))


def _fail(audit_id: str, message: str, path: Path | None, *, scope: str = "report") -> AuditFinding:
    return AuditFinding(audit_id, "fail", scope, message, None if path is None else str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
