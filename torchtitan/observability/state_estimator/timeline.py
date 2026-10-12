# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Timeline helpers for offline training evidence graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from torchtitan.observability.state_estimator.observation import NormalizedObservation


TimelineWindowMode = Literal[
    "incident_centered",
    "explicit_time",
    "explicit_step",
    "last_n_steps",
]


@dataclass(frozen=True, slots=True)
class IncidentWindow:
    center_time_ns: int
    start_time_ns: int
    end_time_ns: int

    @classmethod
    def from_center(
        cls, *, center_time_ns: int, window_before_ns: int, window_after_ns: int
    ) -> "IncidentWindow":
        if window_before_ns < 0:
            raise ValueError("window_before_ns must be non-negative")
        if window_after_ns < 0:
            raise ValueError("window_after_ns must be non-negative")
        return cls(
            center_time_ns=center_time_ns,
            start_time_ns=center_time_ns - window_before_ns,
            end_time_ns=center_time_ns + window_after_ns,
        )

    def contains(self, event_time_ns: int) -> bool:
        return self.start_time_ns <= event_time_ns <= self.end_time_ns


@dataclass(frozen=True, slots=True)
class TimelineWindow:
    mode: TimelineWindowMode
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    center_time_ns: int | None = None
    start_step: int | None = None
    end_step: int | None = None
    num_steps: int | None = None

    @classmethod
    def incident_centered(
        cls, *, center_time_ns: int, window_before_ns: int, window_after_ns: int
    ) -> "TimelineWindow":
        window = IncidentWindow.from_center(
            center_time_ns=center_time_ns,
            window_before_ns=window_before_ns,
            window_after_ns=window_after_ns,
        )
        return cls(
            mode="incident_centered",
            start_time_ns=window.start_time_ns,
            end_time_ns=window.end_time_ns,
            center_time_ns=window.center_time_ns,
        )

    @classmethod
    def explicit_time(
        cls,
        *,
        start_time_ns: int,
        end_time_ns: int,
        center_time_ns: int | None = None,
    ) -> "TimelineWindow":
        if end_time_ns < start_time_ns:
            raise ValueError(
                "end_time_ns must be greater than or equal to start_time_ns"
            )
        return cls(
            mode="explicit_time",
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            center_time_ns=center_time_ns,
        )

    @classmethod
    def explicit_step(cls, *, start_step: int, end_step: int) -> "TimelineWindow":
        if end_step < start_step:
            raise ValueError("end_step must be greater than or equal to start_step")
        return cls(mode="explicit_step", start_step=start_step, end_step=end_step)

    @classmethod
    def last_n_steps(cls, *, num_steps: int) -> "TimelineWindow":
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        return cls(mode="last_n_steps", num_steps=num_steps)


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    observation_id: str
    record_type: str
    phase: str | None
    step: int | None
    entity: Mapping[str, str]
    event_time_ns: int | None
    ingestion_time_ns: int | None
    monotonic_ns: int | None
    event_seq: int | None
    artifact_seq: int | None
    source_path: str
    source_record: str
    topology_epoch: int | None
    relative_time_ns: int | None
    ordering_basis: str

    def to_json(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "record_type": self.record_type,
            "phase": self.phase,
            "step": self.step,
            "entity": dict(self.entity),
            "event_time_ns": self.event_time_ns,
            "ingestion_time_ns": self.ingestion_time_ns,
            "monotonic_ns": self.monotonic_ns,
            "event_seq": self.event_seq,
            "artifact_seq": self.artifact_seq,
            "source_path": self.source_path,
            "source_record": self.source_record,
            "topology_epoch": self.topology_epoch,
            "relative_time_ns": self.relative_time_ns,
            "ordering_basis": self.ordering_basis,
        }


@dataclass(frozen=True, slots=True)
class ClockQualityFinding:
    kind: str
    severity: str
    message: str
    source_path: str | None = None
    source_record: str | None = None
    entity: Mapping[str, str] | None = None
    metadata: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "source_path": self.source_path,
            "source_record": self.source_record,
            "entity": dict(self.entity) if self.entity else None,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class SemanticTimeline:
    window: TimelineWindow
    entries: tuple[TimelineEntry, ...]
    clock_quality_findings: tuple[ClockQualityFinding, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "window": {
                "mode": self.window.mode,
                "start_time_ns": self.window.start_time_ns,
                "end_time_ns": self.window.end_time_ns,
                "center_time_ns": self.window.center_time_ns,
                "start_step": self.window.start_step,
                "end_step": self.window.end_step,
                "num_steps": self.window.num_steps,
            },
            "entries": [entry.to_json() for entry in self.entries],
            "clock_quality_findings": [
                finding.to_json() for finding in self.clock_quality_findings
            ],
        }


@dataclass(frozen=True, slots=True)
class _ObservationRow:
    observation_id: str
    record_type: str
    phase: str | None
    step: int | None
    entity: Mapping[str, str]
    event_time_ns: int | None
    ingestion_time_ns: int | None
    monotonic_ns: int | None
    event_seq: int | None
    artifact_seq: int | None
    source_path: str
    source_record: str
    topology_epoch: int | None
    input_index: int


def build_semantic_timeline(
    observations_or_graph: Sequence[NormalizedObservation | Mapping[str, Any]]
    | Mapping[str, Any],
    *,
    window: TimelineWindow,
) -> SemanticTimeline:
    """Build a semantic timeline with explicit clock-quality findings."""

    rows = _coerce_observation_rows(observations_or_graph)
    filtered_rows = _filter_rows(rows, window)
    ordered_rows = sorted(filtered_rows, key=_semantic_sort_key)
    center_time_ns = _relative_center_time_ns(ordered_rows, window)
    entries = tuple(_timeline_entry(row, center_time_ns) for row in ordered_rows)
    return SemanticTimeline(
        window=window,
        entries=entries,
        clock_quality_findings=tuple(_clock_quality_findings(ordered_rows)),
    )


def summarize_incident_timeline(
    timeline: SemanticTimeline | Sequence[TimelineEntry],
    *,
    max_entries: int = 8,
) -> str:
    """Return compact Markdown bullets suitable for diagnosis reports."""

    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    entries = timeline.entries if isinstance(timeline, SemanticTimeline) else timeline
    lines = [_timeline_summary_line(entry) for entry in list(entries)[:max_entries]]
    remaining = len(entries) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} more event(s)")
    return "\n".join(lines)


def _coerce_observation_rows(
    observations_or_graph: Sequence[NormalizedObservation | Mapping[str, Any]]
    | Mapping[str, Any],
) -> list[_ObservationRow]:
    if isinstance(observations_or_graph, Mapping):
        observations = observations_or_graph.get("observations", [])
        if not isinstance(observations, Sequence):
            return []
    else:
        observations = observations_or_graph

    rows: list[_ObservationRow] = []
    for input_index, observation in enumerate(observations):
        if isinstance(observation, NormalizedObservation):
            rows.append(_row_from_normalized(observation, input_index=input_index))
        elif isinstance(observation, Mapping):
            rows.append(_row_from_mapping(observation, input_index=input_index))
    return rows


def _row_from_normalized(
    observation: NormalizedObservation, *, input_index: int
) -> _ObservationRow:
    envelope = observation.envelope
    return _ObservationRow(
        observation_id=observation.id,
        record_type=envelope.record_type,
        phase=envelope.phase,
        step=envelope.step,
        entity=_entity_from_envelope(observation),
        event_time_ns=envelope.event_time_ns,
        ingestion_time_ns=envelope.ingestion_time_ns,
        monotonic_ns=envelope.monotonic_ns,
        event_seq=envelope.event_seq,
        artifact_seq=envelope.artifact_seq,
        source_path=envelope.source_path,
        source_record=envelope.raw_record_id,
        topology_epoch=_read_int(observation.raw_record, "topology_epoch"),
        input_index=input_index,
    )


def _row_from_mapping(
    observation: Mapping[str, Any],
    *,
    input_index: int,
) -> _ObservationRow:
    payload = observation.get("payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    envelope = observation.get("envelope", {})
    if not isinstance(envelope, Mapping):
        envelope = {}
    raw_record = observation.get("raw_record", {})
    if not isinstance(raw_record, Mapping):
        raw_record = {}
    return _ObservationRow(
        observation_id=str(
            observation.get("id", observation.get("observation_id", ""))
        ),
        record_type=str(
            observation.get(
                "record_type",
                envelope.get("record_type", payload.get("record_type", "unknown")),
            )
        ),
        phase=_first_string(observation, envelope, payload, field_name="phase"),
        step=_first_int(observation, envelope, payload, field_name="step"),
        entity=_mapping_entity(observation.get("entity")),
        event_time_ns=_first_int(
            observation, envelope, payload, field_name="event_time_ns"
        ),
        ingestion_time_ns=_first_int(
            observation, envelope, payload, field_name="ingestion_time_ns"
        ),
        monotonic_ns=_first_int(
            observation, envelope, payload, field_name="monotonic_ns"
        ),
        event_seq=_first_int(observation, envelope, payload, field_name="event_seq"),
        artifact_seq=_first_int(
            observation, envelope, payload, field_name="artifact_seq"
        ),
        source_path=str(
            observation.get(
                "source_path",
                envelope.get("source_path", payload.get("source_path", "")),
            )
        ),
        source_record=str(
            observation.get(
                "source_record",
                observation.get(
                    "raw_record_id",
                    envelope.get("raw_record_id", payload.get("raw_record_id", "")),
                ),
            )
        ),
        topology_epoch=_first_int(
            observation, raw_record, payload, field_name="topology_epoch"
        ),
        input_index=input_index,
    )


def _entity_from_envelope(observation: NormalizedObservation) -> Mapping[str, str]:
    envelope = observation.envelope
    if envelope.process_id is not None:
        return {"kind": "process", "id": envelope.process_id}
    if envelope.global_rank is not None:
        return {"kind": "rank", "id": str(envelope.global_rank)}
    return {"kind": "unknown", "id": "unknown"}


def _mapping_entity(value: Any) -> Mapping[str, str]:
    if isinstance(value, Mapping):
        return {
            "kind": str(value.get("kind", "unknown")),
            "id": str(value.get("id", "unknown")),
        }
    return {"kind": "unknown", "id": "unknown"}


def _filter_rows(
    rows: Sequence[_ObservationRow],
    window: TimelineWindow,
) -> list[_ObservationRow]:
    if window.mode in ("incident_centered", "explicit_time"):
        return [
            row
            for row in rows
            if row.event_time_ns is not None
            and window.start_time_ns is not None
            and window.end_time_ns is not None
            and window.start_time_ns <= row.event_time_ns <= window.end_time_ns
        ]
    if window.mode == "explicit_step":
        return [
            row
            for row in rows
            if row.step is not None
            and window.start_step is not None
            and window.end_step is not None
            and window.start_step <= row.step <= window.end_step
        ]
    if window.mode == "last_n_steps":
        steps = sorted({row.step for row in rows if row.step is not None})
        selected_steps = set(steps[-(window.num_steps or 0) :])
        return [row for row in rows if row.step in selected_steps]
    raise ValueError(f"unsupported timeline window mode {window.mode!r}")


def _semantic_sort_key(row: _ObservationRow) -> tuple[Any, ...]:
    entity_kind = row.entity.get("kind", "")
    entity_id = row.entity.get("id", "")
    sequence = _local_sequence(row)
    if row.event_time_ns is not None:
        return (
            0,
            row.event_time_ns,
            entity_kind,
            entity_id,
            sequence if sequence is not None else -1,
            row.observation_id,
            row.input_index,
        )
    if row.monotonic_ns is not None:
        return (
            1,
            entity_kind,
            entity_id,
            row.monotonic_ns,
            sequence if sequence is not None else -1,
            row.observation_id,
            row.input_index,
        )
    if sequence is not None:
        return (
            2,
            entity_kind,
            entity_id,
            sequence,
            row.observation_id,
            row.input_index,
        )
    if row.ingestion_time_ns is not None:
        return (
            3,
            row.ingestion_time_ns,
            entity_kind,
            entity_id,
            row.observation_id,
            row.input_index,
        )
    return (4, entity_kind, entity_id, row.observation_id, row.input_index)


def _timeline_entry(row: _ObservationRow, center_time_ns: int | None) -> TimelineEntry:
    timestamp = _relative_timestamp(row)
    relative_time_ns = (
        None
        if timestamp is None or center_time_ns is None
        else timestamp - center_time_ns
    )
    return TimelineEntry(
        observation_id=row.observation_id,
        record_type=row.record_type,
        phase=row.phase,
        step=row.step,
        entity=row.entity,
        event_time_ns=row.event_time_ns,
        ingestion_time_ns=row.ingestion_time_ns,
        monotonic_ns=row.monotonic_ns,
        event_seq=row.event_seq,
        artifact_seq=row.artifact_seq,
        source_path=row.source_path,
        source_record=row.source_record,
        topology_epoch=row.topology_epoch,
        relative_time_ns=relative_time_ns,
        ordering_basis=_ordering_basis(row),
    )


def _relative_center_time_ns(
    rows: Sequence[_ObservationRow], window: TimelineWindow
) -> int | None:
    if window.center_time_ns is not None:
        return window.center_time_ns
    timestamps = [
        timestamp for row in rows if (timestamp := _relative_timestamp(row)) is not None
    ]
    if timestamps:
        return min(timestamps)
    if window.start_time_ns is not None:
        return window.start_time_ns
    return None


def _relative_timestamp(row: _ObservationRow) -> int | None:
    if row.event_time_ns is not None:
        return row.event_time_ns
    if row.monotonic_ns is not None:
        return row.monotonic_ns
    if row.ingestion_time_ns is not None:
        return row.ingestion_time_ns
    return None


def _ordering_basis(row: _ObservationRow) -> str:
    if row.event_time_ns is not None:
        return "event_time"
    if row.monotonic_ns is not None:
        return "monotonic_local"
    if _local_sequence(row) is not None:
        return "process_sequence"
    if row.ingestion_time_ns is not None:
        return "ingestion_time"
    return "input_order"


def _clock_quality_findings(
    rows: Sequence[_ObservationRow],
) -> Iterable[ClockQualityFinding]:
    for row in rows:
        if (
            row.event_time_ns is None
            and row.ingestion_time_ns is None
            and row.monotonic_ns is None
        ):
            yield ClockQualityFinding(
                kind="missing_clocks",
                severity="warning",
                message="Timeline observation has no event, ingestion, or monotonic clock",
                source_path=row.source_path,
                source_record=row.source_record,
                entity=row.entity,
                metadata={"observation_id": row.observation_id},
            )

    if any(row.event_time_ns is None for row in rows):
        yield ClockQualityFinding(
            kind="weak_cross_process_ordering",
            severity="warning",
            message=(
                "Timeline includes observations ordered by process-local or "
                "ingestion fallback; cross-process ordering is weak."
            ),
            metadata={"observation_count": len(rows)},
        )

    yield from _conflicting_sequence_time_findings(rows)


def _conflicting_sequence_time_findings(
    rows: Sequence[_ObservationRow],
) -> Iterable[ClockQualityFinding]:
    rows_by_entity: dict[tuple[str, str], list[_ObservationRow]] = {}
    for row in rows:
        if _local_sequence(row) is None:
            continue
        rows_by_entity.setdefault(
            (row.entity.get("kind", ""), row.entity.get("id", "")), []
        ).append(row)

    for entity_rows in rows_by_entity.values():
        sequence_rows = sorted(
            entity_rows,
            key=lambda row: (_local_sequence(row), row.input_index),
        )
        previous: _ObservationRow | None = None
        for row in sequence_rows:
            if previous is not None and _is_time_before(row, previous):
                yield ClockQualityFinding(
                    kind="conflicting_sequence_time_order",
                    severity="warning",
                    message=(
                        "Process-local sequence order conflicts with available "
                        "event or monotonic time."
                    ),
                    source_path=row.source_path,
                    source_record=row.source_record,
                    entity=row.entity,
                    metadata={
                        "previous_observation_id": previous.observation_id,
                        "observation_id": row.observation_id,
                    },
                )
                break
            previous = row


def _is_time_before(row: _ObservationRow, previous: _ObservationRow) -> bool:
    if row.event_time_ns is not None and previous.event_time_ns is not None:
        return row.event_time_ns < previous.event_time_ns
    if row.monotonic_ns is not None and previous.monotonic_ns is not None:
        return row.monotonic_ns < previous.monotonic_ns
    return False


def _timeline_summary_line(entry: TimelineEntry) -> str:
    relative_time = (
        "unknown"
        if entry.relative_time_ns is None
        else _format_relative_ns(entry.relative_time_ns)
    )
    step = "?" if entry.step is None else str(entry.step)
    phase = entry.phase or "unknown"
    entity = (
        f"{entry.entity.get('kind', 'unknown')}:{entry.entity.get('id', 'unknown')}"
    )
    source = f"{entry.source_path}#{entry.source_record}"
    return (
        f"- {relative_time} step={step} phase={phase} "
        f"entity={entity} source={source}"
    )


def _format_relative_ns(relative_time_ns: int) -> str:
    if relative_time_ns >= 0:
        return f"+{relative_time_ns}ns"
    return f"{relative_time_ns}ns"


def _local_sequence(row: _ObservationRow) -> int | None:
    if row.event_seq is not None:
        return row.event_seq
    return row.artifact_seq


def _first_string(*rows: Mapping[str, Any], field_name: str) -> str | None:
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, str):
            return value
    return None


def _first_int(*rows: Mapping[str, Any], field_name: str) -> int | None:
    for row in rows:
        value = _read_int(row, field_name)
        if value is not None:
            return value
    return None


def _read_int(row: Mapping[str, Any], field_name: str) -> int | None:
    value = row.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _entity_sort_key(observation: Mapping[str, Any]) -> tuple[str, str]:
    entity = observation.get("entity", {})
    if not isinstance(entity, Mapping):
        return ("", "")
    return (str(entity.get("kind", "")), str(entity.get("id", "")))


def _observation_sort_key(observation: Mapping[str, Any]) -> tuple[int, str, str, str]:
    event_time_ns = observation.get("event_time_ns")
    if not isinstance(event_time_ns, int):
        event_time_ns = 0
    entity_kind, entity_id = _entity_sort_key(observation)
    return (event_time_ns, entity_kind, entity_id, str(observation.get("id", "")))


def select_failure_center_time_ns(graph: Mapping[str, Any]) -> int | None:
    """Return a center time for failed attempts, if the graph has one."""

    has_failed_outcome = any(
        entity.get("kind") == "outcome"
        and entity.get("attrs", {}).get("outcome") == "failed"
        for entity in graph.get("entities", [])
        if isinstance(entity, Mapping)
    )
    if not has_failed_outcome:
        return None

    times = [
        observation.get("event_time_ns")
        for observation in graph.get("observations", [])
        if isinstance(observation, Mapping)
        and isinstance(observation.get("event_time_ns"), int)
    ]
    if not times:
        return None
    return max(times)


def build_incident_timeline(
    graph: Mapping[str, Any],
    *,
    center_time_ns: int | None,
    window_before_ns: int,
    window_after_ns: int,
) -> list[dict[str, Any]]:
    """Build a deterministic timeline around a selected incident center."""

    if center_time_ns is None:
        center_time_ns = select_failure_center_time_ns(graph)
    if center_time_ns is None:
        return []

    window = IncidentWindow.from_center(
        center_time_ns=center_time_ns,
        window_before_ns=window_before_ns,
        window_after_ns=window_after_ns,
    )
    rows: list[dict[str, Any]] = []
    for observation in graph.get("observations", []):
        if not isinstance(observation, Mapping):
            continue
        event_time_ns = observation.get("event_time_ns")
        if not isinstance(event_time_ns, int) or not window.contains(event_time_ns):
            continue
        row = dict(observation)
        payload = observation.get("payload", {})
        if isinstance(payload, Mapping):
            for field in ("monotonic_ns", "event_seq", "artifact_seq"):
                if field not in row and field in payload:
                    row[field] = payload[field]
        row["relative_time_ns"] = event_time_ns - center_time_ns
        rows.append(row)
    return sorted(rows, key=_observation_sort_key)
