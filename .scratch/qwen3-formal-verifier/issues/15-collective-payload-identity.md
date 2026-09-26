# 15 — Collective payload identity

**What to build:** Export collective input/output sizes and dtypes, and check
the shape and volume agreement that per-communicator order agreement does not
cover.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** ready-for-agent

## Why this is cheap

The data is already captured and then dropped. NCCL Flight Recorder entries
carry `input_sizes`, `output_sizes` and dtype per work;
`_collective_observations` never reads those fields, so they do not reach the
raw trace and therefore do not reach the formal facts. **No new
instrumentation and no new GPU run is needed** -- this is an exporter change
plus a predicate, the same shape as ticket 11's P1.

Confirm that before building: if the fields turn out to be absent rather than
merely unread, this becomes an instrumentation ticket and its cost changes,
which also makes it ticket 08's problem.

## What it buys

Matching op order on a communicator is necessary but does not make a
collective well-formed. Members must also agree on sizes and dtype; a mismatch
is a different failure than a hang -- it corrupts or errors. So this extends
the safety property along a dimension the current model cannot express at all,
since `Issue` carries only `comm` and `op`.

It also makes the model's abstraction auditable: with sizes present, an
all-gather's output should be the member count times its input, and a
reduce-scatter's the inverse. That relation is checkable against the observed
run and would catch an exporter that mislabelled an operation -- exactly the
class of error that produced the eight-vs-four communicator confusion.

## Acceptance criteria

- [ ] Verify first, in the raw Flight Recorder output, that sizes and dtype
      are present per work. Record what was found either way.
- [ ] `CollectivePayload` exported keyed the same way as `CollectiveComm`, on
      `process_group.canonical_id` -- never `runtime_pg_id`, which is per-rank
      local numbering and denotes different communicators on different ranks.
- [ ] A predicate requiring members of one collective to agree on dtype and on
  the size relation implied by the operation.
- [ ] A negative mutating one member's size, derived from the facts rather
      than hand-written, refused at the payload predicate and not at an
      earlier guard.
- [ ] This changes the raw schema, so every raw digest moves. Sequence it as
      its own commit with an intentional `--update-artifacts`, and say so in
      the ticket -- an unexplained digest change is indistinguishable from a
      regression.

## First criterion discharged, with two corrections to this ticket

Measured before starting the work, because the ticket's cost estimate turned on
it.

**The fields exist.** `torch 2.13.0`'s
`torch/include/torch/csrc/distributed/c10d/FlightRecorderDetail.hpp` emits
`input_sizes`, `output_sizes`, `input_dtypes` and `output_dtypes`. So the
premise holds: the Flight Recorder carries payload identity and
`_collective_observations` does not read it -- it reads `is_p2p`,
`profiling_name`, `state`, `thread_id` and `thread_name`, and nothing else.

**Correction 1: the fields are plural and per-tensor.** This ticket says
"dtype". It is `input_dtypes` and `output_dtypes`, lists parallel to the size
lists.
A single-dtype model would be wrong for a collective over multiple tensors.

**Correction 2: "no new instrumentation and no re-run" is wrong.** The raw
per-rank evidence holds only `device`, `events`, `identity`, `lineage`, `mesh`,
`process_groups`, `profile`, `schema` and `tensor_placements` -- there is
**no flight-recorder entries list preserved at all**. `_flight_snapshot()` calls
`_dump_nccl_trace_json()` at runtime and only the projection survives into the
artifact. So reading sizes requires a collector change, which changes the raw
schema, which requires a fresh 4-GPU run and a new canonical trace.

That is why this ticket is sequenced last, and the reason is now measured rather
than assumed. It also means the work pairs naturally with ticket 23, which needs
a fresh run for the same reason -- one run can serve both if the collector
changes land together.
