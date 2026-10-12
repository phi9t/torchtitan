# Incident Records and Post-hoc Outcome Aggregation

TorchTitan's core run evidence recorder now emits a typed `incident` record and
supports a standalone post-hoc reducer that joins per-process outcomes into one
immutable `aggregate_outcome.json`. This closes the two items the core run
evidence foundation deferred (typed incident records and top-level multi-process
outcome aggregation) while preserving the v1 scope boundary: v1 diagnoses and
preserves evidence only.

## Decision

An `incident` record is appended to the same per-process artifact index through
the existing `_append_row`, built on `_event_context()` so it inherits the full
run/attempt/process/rank/mesh/device/clock/`event_seq`/phase envelope. It adds
the incident class, capture state, and disposition policy as `str`-valued enums,
plus best-effort first-fault attribution (`detected_locus`,
`attribution_confidence`) and optional progress-envelope fields. The nine
`IncidentClass` values enumerate the v1 fault suite, but only the non-finite
loss seam is wired as a real emitter in this step; the other eight are the
classification enum only.

Outcome aggregation is deliberately not on the training hot path. A separate
`reduce_attempt_outcome` reads every `processes/<process_id>/outcome.json` after
an attempt and classifies the attempt as failed (any failed, with `first_failure`
the lowest failed `global_rank`), interrupted (any interrupted, none failed),
succeeded (every expected process succeeded with a consistent positive world
size), or incomplete otherwise. Missing is not success, and an inconsistent
world size cannot certify success. `write_attempt_outcome` publishes the result
with the same `mkstemp`+`os.link` immutability the process outcome uses.

## Uncollected versus unknown

An uncollected optional field is omitted from the row, exactly like the existing
optional `step`/`phase`. An unknown-but-observed value is present with an
explicit sentinel (`FaultAttributionLocus.UNKNOWN`,
`FaultConfidence.COLLECTIVE_TIMEOUT_INSUFFICIENT_EVIDENCE`). This mirrors the
`source.dirty=null` semantics and avoids turning distributed ambiguity into
false certainty.

## Scope boundary

This is a CPU/static-verified schema extension with one proven fault class. It
is not GPU certification, not fault-injection certification of the other eight
classes, and not recovery, ETTR, overhead, or throughput evidence. It adds no
automatic retry, recovery, or fleet quarantine, no external tool or hardware
dependency, and no capture-trigger or policy engine. The aggregation reducer
runs post-hoc with no cross-rank barrier, preserving the no-central-collector
contract of the foundation. We accept a schema-first landing here because the
GPU-certification milestone needs the typed incident and progress envelope in
place before fault injection can produce meaningful evidence.
