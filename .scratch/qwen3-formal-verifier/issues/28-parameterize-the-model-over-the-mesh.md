# 28 — Parameterize the protocol model over the mesh

**What to build:** A generated instance of `ScoutBModel` derived from an
observed mesh, replacing the hand-written 2x2 instance, so the invariants can be
checked past one configuration.

**Blocked by:** the exporter prerequisite below. **Blocks:** any claim that the
safety invariants hold for a mesh other than 2x2 dp_shard x tp.

**Status:** the payoff question is answered -- the invariants hold past 2x2, on
a derived instance proven equivalent to the shipped one. The exporter
prerequisite and replacing the hand-written quadruple remain.

## Answered, and how

`mesh_ladder.py` derives the instance for a dp_shard x tp mesh of any degree,
and a generated module EXTENDS `ScoutBModel` rather than editing it. That is the
seam that made this safe to do at all: the invariants are defined over the
CONSTANTS, so binding different ones checks the same properties against a
different mesh while the twenty shipped cfgs and the runner's awk scraper over
the 2x2 definition names stay untouched.

**Two things were wrong in my earlier framing of this ticket.**

First, the exporter change is not a prerequisite for the payoff question.
Checking whether the invariants survive a larger mesh needs no *observed* larger
run -- it is a model question, not an evidence question. The exporter fix is
needed to make the observed 2x2 instance derived rather than hand-written, which
is a separate and smaller goal.

Second, I recorded this as needing a canonical-trace move and two gate runs. It
does not. Nothing here touches the recorded trace, the fact modules, or any
existing file.

**The validation that matters.** Applying the observed convention at 2x2
reproduces the shipped instance and TLC explores the *same state graph*: 133881
states generated, 38321 distinct, matching the figure ticket 16 records for the
hand-written instance. Agreement on rank and communicator counts would not
establish that the generated instance means the same thing; equivalence of the
state graph does, and it is what makes replacing the hand-written quadruple
safe.

**The ladder, measured.** Exhaustive runs only; a run with states left on queue
explored a prefix and proves nothing, which the test helper now asserts rather
than assumes.

| mesh | ranks | comms | MaxIssues | distinct states | exhaustive | violations |
|---|---|---|---|---|---|---|
| 2x2 | 4 | 8 | 2 | 38321 | yes | none |
| 2x3 | 6 | 11 | 1 | 4625 | yes | none |
| 2x4 | 8 | 14 | 1 | 68449 | yes | none |
| 2x3 | 6 | 11 | 2 | 301862 reached | **no**, 151081 queued | none seen |
| 3x2 | 6 | 9 | 2 | 206343 reached | **no**, 29510 queued | none seen |

So the answer is yes, the invariants survive past 2x2 -- to eight ranks and
fourteen communicators at `MaxIssues = 1`. The binding constraint is
`MaxIssues`, not the mesh: at 2 the search closes at four ranks and does not
close at six inside ten minutes. The two incomplete rows are reported as
incomplete rather than as evidence.

## What remains

- The exporter still discards `batch` and `loss` (see below), so the *observed*
  2x2 instance is not yet derived from recorded evidence.
- The shipped hand-written quadruple is not yet replaced by generated
  definitions. The equivalence result above is what would make that safe, and it
  would also retire the awk scraper's dependency on those four names.
- `ScoutDistributed.tla` still hardcodes 2x2 independently.
- Meshes with a third non-trivial axis are not covered: the mesh-name convention
  for `dp_replicate` was not observed, and inventing one would be exactly the
  kind of unmeasured assumption this ticket is trying to remove.

## The original question this answers

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
