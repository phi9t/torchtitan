# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""RunAttempt lifecycle coordinator (roadmap Sections 3-7).

RunAttempt.create freezes a declaration into an attempt bundle; run_stage
launches one stage through the executor seam and records a stage_started plus
exactly one terminal event on the coordinator event stream; finish validates
per-condition measurement and promotion, derives the run-gate summary, and
commits the independent attempt outcome as an immutable terminal file.

The bundle layout follows roadmap 3.3:

    <results-root>/runs/<run-id>/<attempt-id>/
      manifest.json
      processes/<process-id>/events.jsonl
      outcome.json
      derived/report_input.json
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from torchtitan.experiments.execution import models, store
from torchtitan.experiments.execution.executor import Executor, SubprocessExecutor


# The lifecycle coordinator owns one process-local event stream. Nested runners
# attach their own process ids; they do not write this stream (roadmap 5).
COORDINATOR_PROCESS_ID = "coordinator"

_TERMINAL_EVENT_FOR_RETURN_CODE = {0: "stage_succeeded"}


class RunAttempt:
    """One operational execution of a frozen run declaration."""

    def __init__(
        self,
        *,
        attempt: models.Attempt,
        bundle_dir: Path,
        executor: Executor,
    ):
        self.attempt = attempt
        self.bundle_dir = bundle_dir
        self._executor = executor
        self._stage_invocations: list[str] = []
        self._terminal_events: list[models.StageEvent] = []

    @classmethod
    def create(
        cls,
        declaration: models.RunDeclaration,
        *,
        attempt_id: str,
        results_root: Path,
        parent_attempt_id: str | None = None,
        executor: Executor | None = None,
    ) -> "RunAttempt":
        attempt = models.Attempt(
            attempt_id=attempt_id,
            declaration=declaration,
            parent_attempt_id=parent_attempt_id,
        )
        bundle_dir = _bundle_dir(results_root, declaration.run_id, attempt_id)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "kind": "attempt_manifest",
            "run": declaration.normalized(),
            "attempt": {
                "attempt_id": attempt_id,
                "parent_attempt_id": parent_attempt_id,
            },
            "declaration_digest": declaration.digest(),
            "created_time": _now_iso(),
        }
        store.write_terminal_json(bundle_dir / "manifest.json", manifest)
        return cls(
            attempt=attempt,
            bundle_dir=bundle_dir,
            executor=executor or SubprocessExecutor(),
        )

    @classmethod
    def attach(
        cls,
        *,
        run_id: str,
        attempt_id: str,
        results_root: Path,
        executor: Executor | None = None,
    ) -> "RunAttempt":
        """Reattach to an existing attempt bundle across process boundaries.

        The begin/stage/finish CLI runs each subcommand as a separate process,
        so stage and finish reconstruct the attempt from its immutable manifest
        and append-only event stream rather than in-memory state (roadmap 3.3).
        """

        bundle_dir = _bundle_dir(results_root, run_id, attempt_id)
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no attempt bundle to attach at {manifest_path}; run begin first"
            )
        manifest = json.loads(manifest_path.read_text())
        run = manifest["run"]
        declaration = models.RunDeclaration(
            run_id=run["run_id"],
            family=run["family"],
            task=run["task"],
            lane=run["lane"],
            fields=run.get("fields", {}),
        )
        attempt = models.Attempt(
            attempt_id=attempt_id,
            declaration=declaration,
            parent_attempt_id=manifest["attempt"].get("parent_attempt_id"),
        )
        instance = cls(
            attempt=attempt,
            bundle_dir=bundle_dir,
            executor=executor or SubprocessExecutor(),
        )
        instance._replay_event_stream()
        return instance

    def run_stage(
        self,
        spec: models.StageSpec,
        *,
        stage_extra: dict | None = None,
        stage_extra_files: dict[str, str] | None = None,
        terminal_kind_by_return_code: dict[int, str] | None = None,
    ) -> models.StageEvent:
        stage_invocation_id = _new_stage_invocation_id(spec.stage_id)
        self._stage_invocations.append(stage_invocation_id)
        start_time = _now_iso()
        started = models.StageEvent(
            kind=models.STAGE_EVENT_STARTED,
            stage_id=spec.stage_id,
            stage_invocation_id=stage_invocation_id,
        )
        self._append_event(
            started, extra={"argv": spec.argv, "kind_of_stage": spec.kind}
        )

        cwd = Path(spec.cwd) if spec.cwd else None
        result = self._executor.run(spec.argv, cwd=cwd)
        # An executor may declare a terminal state a return code cannot express
        # (a gate that blocked the launch, or a cancellation that interrupted
        # it); otherwise derive succeeded/failed from the return code.
        terminal_overrides = terminal_kind_by_return_code or {}
        terminal_kind = (
            result.terminal_kind
            or terminal_overrides.get(result.return_code)
            or _TERMINAL_EVENT_FOR_RETURN_CODE.get(result.return_code, "stage_failed")
        )
        terminal = models.StageEvent(
            kind=terminal_kind,
            stage_id=spec.stage_id,
            stage_invocation_id=stage_invocation_id,
            return_code=result.return_code,
        )
        terminal_extra = {
            "return_code": result.return_code,
            "start_time": start_time,
            "end_time": _now_iso(),
        }
        # A caller may attach opaque per-stage metadata (for example a runner's
        # own status JSON) that the lifecycle records verbatim on the terminal
        # event without interpreting it. Reserved keys are not overwritten.
        if stage_extra:
            for key, value in stage_extra.items():
                terminal_extra.setdefault(key, value)
        # File-backed extras are resolved after the command runs so a caller can
        # attach a status file the command itself produced. A missing file is
        # skipped rather than failing the stage.
        if stage_extra_files:
            for key, path in stage_extra_files.items():
                status_path = Path(path)
                if status_path.is_file():
                    terminal_extra.setdefault(key, json.loads(status_path.read_text()))
        self._append_event(terminal, extra=terminal_extra)
        self._terminal_events.append(terminal)
        return terminal

    def finish(
        self,
        *,
        canonical_report_input: dict,
        evaluations: dict[str, models.ConditionStatus],
        attempt_outcome: str,
    ) -> models.AttemptOutcome:
        """Commit the terminal attempt outcome and derived report.

        ``evaluations`` maps a condition key (split/cell) to its own
        ConditionStatus. There is deliberately no run-wide measurement or
        promotion argument: passing one is a TypeError (roadmap Section 4/7).
        """

        for key, status in evaluations.items():
            if not isinstance(status, models.ConditionStatus):
                raise TypeError(
                    f"evaluation {key!r} must be a ConditionStatus, got {type(status)!r}"
                )

        outcome = models.AttemptOutcome(
            attempt_id=self.attempt.attempt_id,
            execution_outcome=attempt_outcome,
            stage_invocation_ids=list(self._stage_invocations),
        )
        outcome_payload = {
            "schema_version": 1,
            "kind": "attempt_outcome",
            "attempt_id": outcome.attempt_id,
            "execution_outcome": outcome.execution_outcome,
            "stage_invocation_ids": outcome.stage_invocation_ids,
            "evaluations": {
                key: status.to_dict() for key, status in evaluations.items()
            },
            "run_gate": _derive_run_gate(evaluations),
            "finished_time": _now_iso(),
        }
        # Terminal files are immutable; a second finish raises FileExistsError.
        store.write_terminal_json(self.bundle_dir / "outcome.json", outcome_payload)
        store.write_terminal_json(
            self.bundle_dir / "derived" / "report_input.json",
            canonical_report_input,
        )
        return outcome

    def _append_event(self, event: models.StageEvent, *, extra: dict) -> None:
        row = {
            "kind": event.kind,
            "stage_id": event.stage_id,
            "stage_invocation_id": event.stage_invocation_id,
            "monotonic_ns": time.monotonic_ns(),
            "time": _now_iso(),
        }
        if event.return_code is not None:
            row["return_code"] = event.return_code
        row.update(extra)
        store.append_jsonl(self._event_stream_path(), row)

    def _event_stream_path(self) -> Path:
        return self.bundle_dir / "processes" / COORDINATOR_PROCESS_ID / "events.jsonl"

    def _replay_event_stream(self) -> None:
        """Rebuild stage invocation ids from the append-only coordinator stream.

        Each stage_started row carries a unique stage_invocation_id; replaying
        them lets finish (a separate process) recover every stage the attempt
        launched without in-memory state.
        """

        for row in store.read_jsonl(self._event_stream_path()):
            if row.get("kind") == models.STAGE_EVENT_STARTED:
                self._stage_invocations.append(row["stage_invocation_id"])


def _derive_run_gate(evaluations: dict[str, models.ConditionStatus]) -> dict:
    """Summarize whether any condition is a real, promotable measurement.

    This is a derived view, not a score: it counts conditions by measurement so
    a blocked or fixture condition never masquerades as evidence.
    """

    measurements: dict[str, int] = {}
    promotions: dict[str, int] = {}
    for status in evaluations.values():
        measurements[status.measurement] = measurements.get(status.measurement, 0) + 1
        promotions[status.promotion] = promotions.get(status.promotion, 0) + 1
    return {
        "num_conditions": len(evaluations),
        "measurement_counts": measurements,
        "promotion_counts": promotions,
        "has_real_measurement": any(
            status.is_real_measurement for status in evaluations.values()
        ),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_stage_invocation_id(stage_id: str) -> str:
    return f"inv-{stage_id}-{_now_iso()}-{os.getpid()}-{time.monotonic_ns()}"


def _bundle_dir(results_root: Path, run_id: str, attempt_id: str) -> Path:
    return Path(results_root) / "runs" / run_id / attempt_id
