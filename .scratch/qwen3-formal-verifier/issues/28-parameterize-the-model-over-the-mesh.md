# 28 — Parameterize the protocol model over the mesh

**What to build:** A generated instance of `ScoutBModel` derived from an
observed mesh, replacing the hand-written 2x2 instance, so the invariants can be
checked past one configuration.

**Blocked by:** the exporter prerequisite below. **Blocks:** any claim that the
safety invariants hold for a mesh other than 2x2 dp_shard x tp.

**Status:** open; prerequisite identified and unstarted

## The question this answers

Every invariant in this program has been checked on exactly one configuration:
four ranks, eight communicators, two non-trivial mesh axes. Whether they survive
past that is currently unanswerable, and a hand-written instance can drift from
the mesh actually observed.

## What ticket 27 already gives this ticket, and what it does not

Ticket 27's harness generates arbitrary topologies -- rank count, communicator
count, membership, admissible ops, stream map -- and renders each into both a
Lean module and a TLA+ probe with a cfg. **Reuse that generator.** Writing a
second one would duplicate the piece that already knows how to bind
`Ranks`/`CommIds`/`CommMembers`/`CommOps`, and, per ticket 27's own finding, the
two easily-missed overrides `Ops` and `StreamOfIssue`.

But it answers a different question. The differential asks whether the two
models *agree* on a state handed to them. This ticket asks whether the safety
invariants *hold* over all reachable states of a larger instance. Agreement on a
pinned state says nothing about reachability, and the differential's probe
deliberately pins `Next == UNCHANGED vars` so TLC explores exactly one state.
The state-space exploration is the new work.

## Correction to the plan this ticket comes from

The plan claimed "the communicator count is not derivable from the mesh". It is.
`ParallelDims.get_all_one_dimensional_meshes()` (`parallel_dims.py:612`) returns
`dp_replicate, fsdp, tp, batch, loss, efsdp` as first-class named axes -- its
docstring example prints exactly that -- and `batch` and `loss` are the two
extra names the observed eight-communicator run shows. So `ParallelDims` is a
sound generator source.

## The prerequisite, localized exactly

`_runtime_topology` in `scout_b.py` **already calls**
`trainer.parallel_dims.get_all_one_dimensional_meshes()`. It then throws most of
the answer away:

- it requires only `{"fsdp", "tp"}` to be present;
- it iterates a hardcoded two-tuple, `(("fsdp", "dp_shard"), ("tp", "tp"))`, so
  exactly two process groups are recorded per rank;
- it builds the coordinate as a hardcoded six-element list,
  `[0, fsdp, 0, tp, 0, 0]`, checked against `[0, rank // 2, 0, rank % 2, 0, 0]`.

So `batch`, `loss` and `efsdp` are returned by the helper and discarded. The
exporter is not missing access to the mesh; it is discarding it. That makes the
prerequisite a smaller change than "teach the exporter about ParallelDims", and
a more delicate one, because those three hardcoded shapes are what the 2x2
assumption is actually made of.

One thing this rules out: `_group_axis` is **not** the bug. `mesh_batch` and
`mesh_loss_mesh` are groups *over* the `dp_shard` axis, so returning `dp_shard`
for members `{0,2}` is correct. What distinguishes them is the mesh name, which
already reaches the `collective_id` through `description_key`. The gap is that
the topology never declares those meshes exist, not that the axis is
misreported.

## Why this needs a deliberate canonical-trace move

Recording more of the mesh changes `process_groups` from two entries per rank to
four or more, which changes the normalized bundle, which changes `trace_id`,
which makes all four checked-in fact modules stale. That is a legitimate
`--update-artifacts` -- the only kind this program should ever do -- but it
means:

- a schema decision, since `RAW_SCHEMA` currently pins `qwen3.formal.raw.v1`;
- the `MESH_AXES` / `MESH_DEGREES` literal assertions become wrong as written;
- two gate runs, one to accept the new trace and one to confirm check mode
  passes against it;
- an audit of the 20 cfgs and `ScoutDistributed.tla`, both of which encode 2x2
  independently.

That is a full ticket, not a quick fix, and it touches a converged gated path.
Doing it half-way would leave the canonical trace moved and the consumers not
updated, which is worse than not starting.

## The original framing, kept for context

What is not sound is the recorded evidence. The exporter writes only
`MeshAxisName` members, so `batch` and `loss` appear nowhere in the recorded
topology. They survive only inside Flight Recorder `desc` strings, which is
where `_group_axis` recovers them from by **hardcoded member tuple**. Recording
what `get_all_one_dimensional_meshes()` already knows is a real gap in the
evidence, not merely a phase-5 inconvenience, and it has to land first.

Changing the exporter moves the recorded trace, so it needs a 4-rank gate run to
regenerate fixtures. That is the expensive part of this ticket.

Name collision to beware when reading `scout_b.py`: `:281` and `:314` use the
key `"batch"` for `local_batch_size`/`global_batch_size` in the profile check.
That is the training batch size, not the `batch` mesh axis, and it is not
evidence the axis is covered.

## Four binding sites beyond the model file

1. **Twenty cfgs** bind `Ranks2x2 / CommIds2x2 / CommMembers2x2 / CommOps2x2` by
   definition override. Parameterizing the model alone changes nothing.
2. `run_tlc_scout_b_model.sh:189-200` **textually scrapes those four names with
   awk** to build its `bound_ranks=` / `bound_communicators=` token. Rename them
   and the token silently reports the wrong bound -- a result token claiming
   more scope than the run had, which is item 5 of the review checklist.
3. `ScoutDistributed.tla:7-27,207-215` **independently hardcodes 2x2** in
   `ExpectedCoordinates`, `AllowedGroups` and
   `DOMAIN meshAxisDegree = {"dp_shard","tp"}`. A second model to parameterize.
4. `MESH_DEGREES` in `scout_b.py` is **asserted, not observed**, and
   `_group_axis` hardcodes member tuples. Both must become derived.

## The trap

A generator emitting one communicator per mesh axis would **understate** the
set. Eight communicators over two non-trivial axes is four mesh names times two
rank pairs, because the runtime creates several groups over the same axis for
different purposes -- `mesh_batch` and `mesh_loss_mesh` both carry `axis:
dp_shard` with members `{0,2}`. Getting this wrong reintroduces the
eight-versus-four keying error from ticket 11 in a new guise.

## Needs no work

The Lean side is already fully topology-general: `Topology (R C O S)` quantifies
over arbitrary rank, communicator, op and stream types, so every theorem already
holds for any mesh. The unbounded proof is not the constraint; the TLA+
instances and the exporter are.

## Acceptance

- The invariants are checked on at least one mesh that is not 2x2, with the
  bound reported honestly in the token rather than scraped from a renamed
  definition.
- State counts reported per configuration, since a larger mesh may not be
  checkable exhaustively -- and if it is not, the token says so instead of
  implying it was.
- The generated instance is derived from recorded evidence, not from a second
  hand-written literal.

## Abandon criterion

If the recorded mesh facts do not determine the communicator set, stop.
Inferring it would reintroduce exactly the positional guessing that ticket 09
exists to remove.
