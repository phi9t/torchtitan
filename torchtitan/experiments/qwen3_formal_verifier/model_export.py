from __future__ import annotations

import re

from .schema import Trace


def to_tla_constants(trace: Trace) -> dict[str, object]:
    ranks = sorted({event.rank for event in trace.events})
    devices = sorted({event.device.id or f"{event.device.kind}-{event.device.index}" for event in trace.events})
    executors = sorted({event.executor.id for event in trace.events if event.executor is not None})
    works = sorted({event.work.id for event in trace.events if event.work is not None})
    causal_edges = sorted(
        (predecessor, event.event_id) for event in trace.events for predecessor in event.causal_predecessors
    )
    return {
        "schema": trace.schema,
        "ranks": ranks,
        "devices": devices,
        "executors": executors,
        "works": works,
        "causal_edges": causal_edges,
    }


def to_lean_facts(trace: Trace) -> list[str]:
    facts: list[str] = []
    for event in trace.events:
        event_id = _lean_atom(event.event_id)
        facts.append(f"event_kind {event_id} {_lean_atom(event.kind)}")
        facts.append(f"event_rank {event_id} {event.rank}")
        if event.work is not None:
            facts.append(f"event_work {event_id} {_lean_atom(event.work.id)}")
        if event.executor is not None:
            facts.append(f"event_executor {event_id} {_lean_atom(event.executor.id)}")
        for predecessor in event.causal_predecessors:
            facts.append(f"causal_edge {_lean_atom(predecessor)} {event_id}")
    return facts


def _lean_atom(value: str) -> str:
    atom = re.sub(r"[^0-9A-Za-z_]", "_", value)
    if atom and atom[0].isdigit():
        return f"_{atom}"
    return atom
