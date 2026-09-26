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
