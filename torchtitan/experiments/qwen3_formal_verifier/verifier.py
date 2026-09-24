from __future__ import annotations

from dataclasses import dataclass

from .schema import Trace, TraceEvent, WorkRef


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    event_id: str | None = None
    work_id: str | None = None

    def to_jsonable(self) -> dict[str, object]:
        data: dict[str, object] = {"code": self.code, "message": self.message}
        if self.event_id is not None:
            data["event_id"] = self.event_id
        if self.work_id is not None:
            data["work_id"] = self.work_id
        return data


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    issues: tuple[VerificationIssue, ...]

    def to_jsonable(self) -> dict[str, object]:
        return {"ok": self.ok, "issues": [issue.to_jsonable() for issue in self.issues]}


def verify_trace(trace: Trace) -> VerificationReport:
    issues: list[VerificationIssue] = []
    issues.extend(_duplicate_event_issues(trace.events))
    issues.extend(_work_identity_issues(trace.events))
    issues.extend(_work_order_issues(trace.events))
    issues.extend(_p2p_metadata_issues(trace.events))
    issues.extend(_causal_reference_issues(trace.events))
    issues.extend(_causal_cycle_issues(trace.events))
    return VerificationReport(ok=not issues, issues=tuple(issues))


def _duplicate_event_issues(events: tuple[TraceEvent, ...]) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    seen: set[str] = set()
    for event in events:
        if event.event_id in seen:
            issues.append(
                VerificationIssue(
                    code="duplicate_event_id",
                    message=f"event id {event.event_id} appears more than once",
                    event_id=event.event_id,
                )
            )
        seen.add(event.event_id)
    return issues


def _work_identity_issues(events: tuple[TraceEvent, ...]) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    seen: dict[str, WorkRef] = {}
    for event in events:
        if event.work is None:
            continue
        previous = seen.get(event.work.id)
        if previous is None:
            seen[event.work.id] = event.work
            continue
        if (
            previous.sequence != event.work.sequence
            or previous.op != event.work.op
            or previous.peer != event.work.peer
            or previous.direction != event.work.direction
            or previous.is_batched != event.work.is_batched
        ):
            issues.append(
                VerificationIssue(
                    code="duplicate_work_id",
                    message=f"work id {event.work.id} has conflicting metadata",
                    event_id=event.event_id,
                    work_id=event.work.id,
                )
            )
    return issues


def _work_order_issues(events: tuple[TraceEvent, ...]) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    states: dict[str, set[str]] = {}
    for event in events:
        if event.work is None:
            continue
        work_states = states.setdefault(event.work.id, set())
        if event.kind.endswith(".started") and "enqueued" not in work_states:
            issues.append(
                VerificationIssue(
                    code="work_started_before_enqueued",
                    message=f"work {event.work.id} started before it was enqueued",
                    event_id=event.event_id,
                    work_id=event.work.id,
                )
            )
        if event.kind.endswith(".completed") and "started" not in work_states:
            issues.append(
                VerificationIssue(
                    code="work_completed_before_started",
                    message=f"work {event.work.id} completed before it was started",
                    event_id=event.event_id,
                    work_id=event.work.id,
                )
            )
        if event.kind.endswith(".enqueued"):
            work_states.add("enqueued")
        elif event.kind.endswith(".started"):
            work_states.add("started")
        elif event.kind.endswith(".completed"):
            work_states.add("completed")
    return issues


def _p2p_metadata_issues(events: tuple[TraceEvent, ...]) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    for event in events:
        if not event.kind.startswith("p2p.") or event.work is None:
            continue
        if event.work.peer is None:
            issues.append(
                VerificationIssue(
                    code="p2p_missing_peer",
                    message=f"P2P event {event.event_id} has no peer",
                    event_id=event.event_id,
                    work_id=event.work.id,
                )
            )
        if event.work.direction not in {"send", "recv"}:
            issues.append(
                VerificationIssue(
                    code="p2p_missing_direction",
                    message=f"P2P event {event.event_id} has no send/recv direction",
                    event_id=event.event_id,
                    work_id=event.work.id,
                )
            )
    return issues


def _causal_reference_issues(events: tuple[TraceEvent, ...]) -> list[VerificationIssue]:
    event_ids = {event.event_id for event in events}
    issues: list[VerificationIssue] = []
    for event in events:
        for predecessor in event.causal_predecessors:
            if predecessor not in event_ids:
                issues.append(
                    VerificationIssue(
                        code="missing_causal_predecessor",
                        message=f"event {event.event_id} references missing predecessor {predecessor}",
                        event_id=event.event_id,
                    )
                )
    return issues


def _causal_cycle_issues(events: tuple[TraceEvent, ...]) -> list[VerificationIssue]:
    graph = {event.event_id: tuple(event.causal_predecessors) for event in events}
    visiting: set[str] = set()
    visited: set[str] = set()
    issues: list[VerificationIssue] = []

    def visit(event_id: str) -> bool:
        if event_id in visiting:
            return True
        if event_id in visited:
            return False
        visiting.add(event_id)
        for predecessor in graph.get(event_id, ()):
            if predecessor in graph and visit(predecessor):
                return True
        visiting.remove(event_id)
        visited.add(event_id)
        return False

    for event in events:
        if visit(event.event_id):
            issues.append(
                VerificationIssue(
                    code="causal_cycle",
                    message="causal predecessors contain a cycle",
                    event_id=event.event_id,
                )
            )
            break
    return issues
