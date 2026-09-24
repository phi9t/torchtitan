from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


JsonMap = dict[str, object]


def _tuple_strs(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return tuple(str(item) for item in value)


def _extensions(value: object) -> JsonMap:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("extensions must be an object")
    return dict(value)


@dataclass(frozen=True)
class DeviceRef:
    kind: str
    index: int
    id: str | None = None

    def to_jsonable(self) -> JsonMap:
        data: JsonMap = {"kind": self.kind, "index": self.index}
        if self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_jsonable(cls, data: Mapping[str, object]) -> "DeviceRef":
        return cls(kind=str(data["kind"]), index=int(data["index"]), id=data.get("id") and str(data["id"]))


@dataclass(frozen=True)
class ProcessGroupRef:
    id: str
    ranks: tuple[int, ...]

    def to_jsonable(self) -> JsonMap:
        return {"id": self.id, "ranks": list(self.ranks)}

    @classmethod
    def from_jsonable(cls, data: Mapping[str, object]) -> "ProcessGroupRef":
        ranks = data["ranks"]
        if not isinstance(ranks, list):
            raise TypeError("process_group.ranks must be a list")
        return cls(id=str(data["id"]), ranks=tuple(int(rank) for rank in ranks))


@dataclass(frozen=True)
class ExecutorRef:
    kind: str
    id: str

    def to_jsonable(self) -> JsonMap:
        return {"kind": self.kind, "id": self.id}

    @classmethod
    def from_jsonable(cls, data: Mapping[str, object]) -> "ExecutorRef":
        return cls(kind=str(data["kind"]), id=str(data["id"]))


@dataclass(frozen=True)
class WorkRef:
    id: str
    sequence: int
    op: str
    peer: int | None = None
    direction: str | None = None
    is_batched: bool | None = None

    def to_jsonable(self) -> JsonMap:
        data: JsonMap = {"id": self.id, "sequence": self.sequence, "op": self.op}
        if self.peer is not None:
            data["peer"] = self.peer
        if self.direction is not None:
            data["direction"] = self.direction
        if self.is_batched is not None:
            data["is_batched"] = self.is_batched
        return data

    @classmethod
    def from_jsonable(cls, data: Mapping[str, object]) -> "WorkRef":
        return cls(
            id=str(data["id"]),
            sequence=int(data["sequence"]),
            op=str(data["op"]),
            peer=None if data.get("peer") is None else int(data["peer"]),
            direction=None if data.get("direction") is None else str(data["direction"]),
            is_batched=None if data.get("is_batched") is None else bool(data["is_batched"]),
        )


@dataclass(frozen=True)
class Timing:
    clock_domain: str
    start: float | None = None
    end: float | None = None

    def to_jsonable(self) -> JsonMap:
        data: JsonMap = {"clock_domain": self.clock_domain}
        if self.start is not None:
            data["start"] = self.start
        if self.end is not None:
            data["end"] = self.end
        return data

    @classmethod
    def from_jsonable(cls, data: Mapping[str, object]) -> "Timing":
        return cls(
            clock_domain=str(data["clock_domain"]),
            start=None if data.get("start") is None else float(data["start"]),
            end=None if data.get("end") is None else float(data["end"]),
        )


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    kind: str
    logical_time: int
    rank: int
    local_rank: int
    world_size: int
    device: DeviceRef
    timing: Timing
    process_group: ProcessGroupRef | None = None
    work: WorkRef | None = None
    executor: ExecutorRef | None = None
    causal_predecessors: tuple[str, ...] = ()
    extensions: JsonMap = field(default_factory=dict)

    def to_jsonable(self) -> JsonMap:
        data: JsonMap = {
            "event_id": self.event_id,
            "kind": self.kind,
            "logical_time": self.logical_time,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device": self.device.to_jsonable(),
            "timing": self.timing.to_jsonable(),
            "causal_predecessors": list(self.causal_predecessors),
            "extensions": dict(self.extensions),
        }
        if self.process_group is not None:
            data["process_group"] = self.process_group.to_jsonable()
        if self.work is not None:
            data["work"] = self.work.to_jsonable()
        if self.executor is not None:
            data["executor"] = self.executor.to_jsonable()
        return data

    @classmethod
    def from_jsonable(cls, data: Mapping[str, object]) -> "TraceEvent":
        device_data = data["device"]
        timing_data = data["timing"]
        if not isinstance(device_data, dict):
            raise TypeError("device must be an object")
        if not isinstance(timing_data, dict):
            raise TypeError("timing must be an object")

        process_group = data.get("process_group")
        work = data.get("work")
        executor = data.get("executor")
        return cls(
            event_id=str(data["event_id"]),
            kind=str(data["kind"]),
            logical_time=int(data["logical_time"]),
            rank=int(data["rank"]),
            local_rank=int(data["local_rank"]),
            world_size=int(data["world_size"]),
            device=DeviceRef.from_jsonable(device_data),
            process_group=ProcessGroupRef.from_jsonable(process_group) if isinstance(process_group, dict) else None,
            work=WorkRef.from_jsonable(work) if isinstance(work, dict) else None,
            executor=ExecutorRef.from_jsonable(executor) if isinstance(executor, dict) else None,
            timing=Timing.from_jsonable(timing_data),
            causal_predecessors=_tuple_strs(data.get("causal_predecessors"), "causal_predecessors"),
            extensions=_extensions(data.get("extensions")),
        )


@dataclass(frozen=True)
class Trace:
    schema: str
    events: tuple[TraceEvent, ...]

    @classmethod
    def from_events(cls, events: Iterable[TraceEvent], schema: str = "accelerator.trace.v1") -> "Trace":
        return cls(schema=schema, events=tuple(events))

    def to_jsonable(self) -> JsonMap:
        return {"schema": self.schema, "events": [event.to_jsonable() for event in self.events]}

    @classmethod
    def from_jsonable(cls, data: Mapping[str, object]) -> "Trace":
        raw_events = data["events"]
        if not isinstance(raw_events, list):
            raise TypeError("events must be a list")
        return cls(
            schema=str(data.get("schema", "accelerator.trace.v1")),
            events=tuple(TraceEvent.from_jsonable(event) for event in raw_events if isinstance(event, dict)),
        )

    def to_json_text(self) -> str:
        import json

        return json.dumps(self.to_jsonable(), indent=2, sort_keys=True)
