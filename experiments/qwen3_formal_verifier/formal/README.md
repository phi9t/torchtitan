# Qwen3 Step Formal Trace

This directory contains generated formal artifacts for a TorchTitan-style Qwen3
dense DPxTP training step.

- `Qwen3StepTrace.tla` captures the finite DPxTP mesh, process groups, work ids,
  work roles, and group membership used by the TLA+ transition model.
- `Qwen3StepTrace.lean` captures the same finite trace facts in a Lean-friendly
  shape for trace-level invariant proofs.

The source of truth is the Python trace IR emitted by
`build_torchtitan_qwen3_step_trace`; regenerate these files through
`write_qwen3_step_formal_artifacts` after schema or role changes.

## Hermetic smoke toolchains

Task 01 adds an opt-in repository-level Bazel surface for the formal checkers.
It does not make Bazel authoritative for TorchTitan Python, training, or shell
workflows. Run the supported wrapper from outside Insula so it can establish a
fresh network namespace. Calls from an existing Insula session fail closed;
this prevents `--no-fetch` from inheriting a networked sandbox:

```bash
# First use: allow Bazel to materialize pinned dependencies.
scripts/run_formal_checks.sh --networked

# Cache proof: disable both Insula networking and Bazel repository downloads.
scripts/run_formal_checks.sh --no-fetch
```

The wrapper pins Bazel 9.2.0, TLA+ 1.7.4, Bazel's JDK 17 runtime, and the
official Lean 4.34.0 Linux distribution. It never consults host `java`, `lean`,
`elan`, or `lake`. Set `TORCHTITAN_FORMAL_CACHE_HOST` to choose the persistent
host cache; it must resolve outside the checkout and is mounted inside Insula
at `/project/formal-cache`.

`MODULE.bazel.lock` records the reviewed Bzlmod resolution from the public BCR.
Supported runs use lockfile error mode, so dependency drift fails instead of
rewriting the lock. The networked run vendors that resolution into the external
cache; the no-fetch run selects the vendored registry, disables repository
downloads, and executes both smoke tests again rather than accepting cached test
results.

The TLC smoke accepts a bounded valid model and requires the controlled bad
model to violate exactly `SmokeInvariant`. The Lean smoke kernel-checks
`FormalSmoke.validStepReady`, requires its printed axiom set to be empty, and
requires the controlled invalid source to reject exactly
`FormalSmoke.ControlledInvalidProposition`. Tool absence, malformed input,
timeouts, or any other checker/infrastructure error fails the target.

## Tier 0: the trace-free suite

Most of the formal work does not depend on an observed run at all. The abstract
models, their negatives and the non-vacuity refutations are properties of the
specifications, so they can be checked on CPU without a trace, a GPU or a
sealed bundle -- minutes rather than seconds, since the Device-mesh model target
runs eleven checks over eight communicators, four of them liveness:

```bash
scripts/run_formal_checks.sh --networked --suite tier0
scripts/run_formal_checks.sh --no-fetch --suite tier0
experiments/qwen3_formal_verifier/run_formal_tier0.sh --no-fetch
```

The `tier0` suite is exactly the targets whose inputs are hand-written:
`tlc_smoke_test`, `lean_smoke_test`, `tlc_single_rank_model_abstract_test`,
`tlc_device_mesh_model_test`, and `lean_device_mesh_protocol_test`. The
Single-rank model checks are split by input set for this reason --
`tlc_single_rank_model_abstract_test` reads only `SingleRankModel.tla`, its two
configurations and `TraceLifecycle.tla`, while
`tlc_single_rank_model_refine_test` additionally reads the generated
`SingleRankFacts.tla`. Both halves remain in `single_rank_formal_tests` and
`device_mesh_formal_tests`, so the sealed gate runs every check it ran before
the split.

`tlc_device_mesh_model_test` is eleven checks over the abstract DPxTP protocol,
each with its own result token: `DEVICE_MESH_MODEL_SAFETY` (every safety
invariant up to the bound the token reports), `DEVICE_MESH_MODEL_NONVACUOUS`
(completion is reachable), `DEVICE_MESH_MODEL_DIVERGENT` (relaxing only
communicator-site agreement makes `DeadlockFreedom` false),
`DEVICE_MESH_MODEL_WITNESS` (and that deadlock is a balanced cross-communicator
cycle at full budget, not any other stuck state), `DEVICE_MESH_MODEL_OPMISMATCH`
(relaxing operation agreement instead hangs on an intra-communicator mismatch),
`DEVICE_MESH_MODEL_STREAM_EDGE` plus `DEVICE_MESH_MODEL_STREAM_EDGE_MUTANT`
(some rank holds two distinct communicators on one stream, and substituting
`StreamOfIssue(e) == e.comm` makes that stop being true), and
`DEVICE_MESH_MODEL_UNGUARDED` (dropping NCCL's matching guard corrupts the
rendezvous instead of hanging). Four more are liveness rather than safety,
because no safety invariant can say that a step finishes.
`DEVICE_MESH_MODEL_LIVENESS`: under weak fairness on `Start` and `Complete`,
with `Issue` unfair by design, `EveryIssuedCollectiveCompletes` holds,
exhaustively, at the bound the token reports.
`DEVICE_MESH_MODEL_LIVENESS_UNFAIR`: rerunning that same configuration against
the unfair `SPECIFICATION Spec` makes it fail, so the fairness is what the
result rests on. `DEVICE_MESH_MODEL_LIVENESS_NEGATIVE`: relaxing only
communicator-site agreement makes it false, and the counterexample is a lasso
rather than a reachable bad state, which is why it has its own classifier.
`DEVICE_MESH_MODEL_LIVENESS_UNCONDITIONAL`: the unconditional reading of the
same sentence is false even with every guard on, because `Issue` is unfair and a
rank may simply stop issuing -- which is what justifies conditioning the
property instead of weakening it. `DEVICE_MESH_MODEL_CONFIG_COVERAGE` reconciles
the configurations that ship with the configurations a stage actually ran. The
claim inventory in the header of `DeviceMeshModel.tla` states exactly what each
one establishes.

`run_formal_tier0.sh` adds changed-source lint and the focused
`tests/unit_tests/test_formal_toolchain.py` contracts. It is an iteration aid
and **not** a gate: it takes no `--output-root`, `--run-id` or `--attempt-id`,
writes nothing under `outputs/`, and seals no evidence, so a tier-0 pass cannot
be presented as a Single-rank or Device-mesh gate pass. The trace-dependent
checks -- the refinement bridge, the fact modules, artifact synchronization and
the real CUDA step -- run only in `run_single_rank.sh` and `run_device_mesh.sh`.

## Single-rank lifecycle suite

The Single-rank fact modules are generated from the normalized trace of a real
single-rank CUDA run:

- `SingleRankFacts.tla` and `SingleRankFacts.lean` contain observed identities,
  event values, causal predecessors, and raw-source links only.
- `TraceLifecycle.tla` and `TraceLifecycle.lean` are independently maintained
  semantics for forward/backward order, gradient readiness before optimizer
  mutation, optimizer mutation before step completion, and causal order.
- `SingleRankValid.*` applies those semantics to the observed facts.
- `SingleRankBadFacts.*` removes the gradient-ready fact for one controlled
  negative; `SingleRankInvalid.*` must reject the named property.

Run the complete suite through the same fresh-sandbox wrapper:

```bash
scripts/run_formal_checks.sh --networked --suite single-rank
scripts/run_formal_checks.sh --no-fetch --suite single-rank
```

TLC must accept all valid lifecycle invariants and report exactly
`SingleRankGradientReadyBeforeOptimizer` for the controlled transition mutation.
Lean must report that `Qwen3Formal.SingleRank.validLifecycle` has no axioms and
reject `Qwen3Formal.SingleRank.ControlledInvalidProposition`. Each checker
stages sources and writes state or `.olean` products only under Bazel's external
test directory.

## Device-mesh DPxTP suite

Device-mesh's generated modules contain fact values only. The exporter maps the
complete normalized four-rank bundle to all 1,340 event rows, 432 observed
rank-local collective work rows, and 222 synchronization edges.
`MeshTopology.tla` and `MeshTopology.lean` independently define the
bounded 2x2 coordinate, TP-peer input identity, DP/TP placement, group coverage,
exact enqueue/start/complete lifecycle, executor/stream, producer correlation,
event provenance, local predecessors, and cross-rank synchronization
predicates.

The proof boundary is deliberate. Python validates canonical normalized String
IDs and exact graph completeness before generating facts. TLA+ checks exact
String membership for event and edge references. Lean pairs every actual String
ID with its rank-local row and separately recomputes a collision-free bounded
identity from rank, order, and kind; its fixed-arity modules also check row
count/order/uniqueness and reference constraints without axioms. The maintained
Lean event-kind schedule is profile-specific: ten trainer events, zero or more
enqueue/start/complete triples, and one collection marker. It is not a general
training-event grammar.

`DeviceMeshEventRank*.lean`, `DeviceMeshSyncChunk*.lean`, and
`DeviceMeshCollectiveEventRank*.lean` partition the same complete facts so Lean
can kernel-reduce each bounded check. `DeviceMeshEventMutationChecks.lean`
proves that paired String-row drift, a wrong kind, and a wrong order are each
rejected. `DeviceMeshValid.lean` composes those axiom-free results.

The four generated `DeviceMeshFacts` and `DeviceMeshBadFacts` TLA+/Lean modules
exceed the repository's general 500 KiB added-file limit because each carries
the complete bounded graph. They have an exact-path exception in
`.pre-commit-config.yaml`: the supported checker runners consume these plain
modules directly, and artifact synchronization compares their complete bytes.
Compressing them would make them unusable as checker inputs, while splitting
them would change the generated module and import contract rather than repair
the lint gate. All other added files remain subject to the 500 KiB limit.

`DeviceMeshBadFacts.*` changes one completion producer while preserving every
other fact. TLC must report exactly the transition invariant
`DeviceMeshCollectiveProducerCorrelation`. Lean must report that
`Qwen3Formal.DeviceMesh.validDPxTP` has no axioms and must reject exactly
`Qwen3Formal.DeviceMesh.ControlledInvalidProducerProposition`.

`DeviceMeshIssueOrderInvalid.tla` is the second named TLA+ negative, for
per-communicator issue-order agreement -- the NCCL requirement whose violation
hangs a job. It is an override of the real `DeviceMeshFacts` rather than a
mutated 1.6 MB copy, and its target (rank 0's first collective in issue order)
is derived from the facts, so the control cannot drift away from the observed
run. TLC must report exactly `DeviceMeshPerCommunicatorIssueOrder`, with
`MutationIsIsolated` proving the override changed one operation and nothing
else. Its result token is `DEVICE_MESH_TLA_ISSUE_ORDER_NEGATIVE`.

### Structural DTensor placements

The facts carry every model parameter's placements with their mesh axis and
tensor dim, not the two per-rank booleans they used to collapse to. For the
observed four-rank run that is, PER RANK, 74 parameters and 148 placements --
115 `Shard`, 33 `Replicate`, 25 of the shards strided, zero `Partial` (across
the four ranks, 460 and 132). It is exported once because every rank agreed on
the whole list: `validate_normalized_bundle` recomputes each rank's
`placement_summary.schema_sha256` over that rank's own `tensor_placements` bytes
and then requires the four to match, and the per-rank digests are exported so a
checker asserts the agreement itself. Recomputation is the load-bearing half: a
digest only compared across ranks, never against the bytes it covers, agrees
with its peers just as happily when one rank's placements have been rewritten
underneath it.

`PlacementValid` and `placementValid` are unchanged and still mean only "some
parameter is DP-sharded and some parameter is TP-sharded somewhere".
`DeviceMeshPlacementBooleansAgree` / `placementBooleansAgreeObserved` tie the
two booleans they read to the per-parameter facts, so the weak check keeps its
original content instead of quietly becoming a different, stronger claim under
the same name. That is deliberately named for what it proves and no more: the
placements are exported once for all ranks, so for ranks 1-3 it is agreement
with that one list rather than a per-rank derivation, and it is one bit per
axis. The per-parameter content is carried by the well-formedness, divisibility
and local-shape invariants, not by this one. The new content is in separate
named invariants: schema agreement across ranks, mesh-axis degrees agreeing with
the exported rank coordinates, per-parameter well-formedness, no `Partial` at
the optimizer, divisibility of every sharded dim by its axis degree, local
shapes agreeing with which dims are sharded, and `_StridedShard` appearing
exactly where an inner axis shards the same tensor dim.

`RankCoordinates`, which the mesh-degree check compares against, is synthesized
by the exporter as `rank // 2` and `rank % 2` rather than read out of each
rank's observed `mesh.coordinate`. The link to observation is a Python assertion
in the collector, which refuses a run whose actual mesh coordinate is not that
value, so the formal layer checks a value the collector already pinned, not one
it read fresh. That is sound but worth saying, since a reader could otherwise
take `MeshAxisDegreeAgreesWithCoordinates` for two independent observations
agreeing.

Where the facts come from, because they are easy to over-read: one snapshot per
rank of `model_parts[0].named_parameters()`, taken after the model is built and
before `Trainer.train`. It is the parameter set the observed AdamW step updates,
and `OptimizerBoundaryObserved` pins that that step is in the trace, but the
snapshot is not re-taken inside the optimizer pre-hook and no gradient placement
is observed at all. A `Partial` *gradient* is therefore not visible here; what
is checkable is that no parameter the optimizer steps carries an unreduced
placement.

`DeviceMeshPlacementPartialInvalid.tla` is the third named TLA+ negative. Like
the issue-order control it is an override of the real `DeviceMeshFacts` rather
than a mutated 1.6 MB copy, and its target -- the first DP-shard placement in
`PlacementIds` order, on the outer FSDP axis whose reduce-scatter produces the
gradient the optimizer consumes -- is derived from the facts. TLC must report
exactly `DeviceMeshNoPartialAtOptimizer`. `MutationIsIsolated` proves the
override changed one kind, and dropped that placement from the shard-dim domain
as a Partial must, and nothing else; `StructureSurvivesTheMutation` and
`OtherPlacementChecksSurviveTheMutation` are asserted over the mutated facts, so
no other invariant caught it. Local-shape agreement is deliberately left out of
that survivor set: the mutated parameter's local shape is half its global shape
because the DP-shard axis shards it, so an override that stops that axis from
sharding contradicts that check too, and claiming otherwise would overstate the
control. Its result token is `DEVICE_MESH_TLA_PARTIAL_PLACEMENT_NEGATIVE`.
`DeviceMeshPlacementChecks.lean` is the Lean counterpart, with
`injectedPartialIsIsolated`, `rejectsInjectedPartial` and
`injectedPartialStaysWellFormed` under `DEVICE_MESH_LEAN_PLACEMENT_MUTATION`;
the positive results carry `DEVICE_MESH_LEAN_PLACEMENT`.

Parse cost is reported rather than assumed. `DEVICE_MESH_TLA_PLACEMENT_FACTS`
gives the placement share of `DeviceMeshFacts.tla`, and
`DEVICE_MESH_REFINE_PARSE` now carries `placement_bytes=` beside `facts_bytes=`
and `parse_ms=`, so a growing placement export shows up against the parse time
it is charged to. Adding the structural facts moved `DeviceMeshFacts.tla` from
1,642,832 to 1,689,740 bytes (+2.9%, 46,908 of them the delimited placement
block). An interleaved A/B of the SANY parse of `DeviceMeshRefine`, against the
same facts module with the placement block stripped, puts the cost at a real and
reproducible +35 to +50 ms, about 4-5% -- small, and measurable only with the
interleaved shape. `DEVICE_MESH_REFINE_PARSE`'s `parse_ms` is kept, with this
caveat attached rather than dropped, because it is still the only in-gate record
that the parse completed at all and roughly how long the fixed cost was. It is
NOT a measurement of parse cost: the refine target parses while other TLC
actions run, and 32 concurrent SANY parsers reproduce `parse_ms=1671` on the
UNCHANGED fixture while 96 bracket 2008, so a rise in that number is evidence
about machine load first and about the facts only after the load is ruled out.
Use an interleaved A/B like the one above for a cost claim. `placement_bytes=`
beside it is the number that tracks the export's growth, and it is exact.

### Collective payload identity

Matching operation order on a communicator is necessary but does not make a
collective well formed. Its members must also agree on dtype and on volume, and
a mismatch there is a different failure from a hang: NCCL errors or corrupts
instead of deadlocking, so the ordering predicates cannot express it at all.

The facts carry, per collective work, `input_sizes`, `output_sizes`,
`input_dtypes` and `output_dtypes` read off that rank's own NCCL Flight Recorder
entry. They are OBSERVED -- unlike `CollectiveProducers` and `CollectiveStream`,
which are attributed by a positional zip -- and the sealed bundle says so in the
provenance contract's `observed_payload` block. They are also PLURAL and per
tensor: one shape and one dtype name per tensor the collective was given, the
lists running in parallel, so a single-dtype model would be wrong for a
collective over several tensors. An empty shape is a 0-dim tensor: no
dimensions, one element, which is why the product of no sizes is 1 and not 0.

Element counts are DERIVED by the exporter, because the relation an operation
implies is between volumes rather than shapes -- an all-gather may concatenate
along any dimension. Both checkers recompute them from the exported shapes
rather than trusting the derivation, the same way the placement schema digest is
recomputed rather than compared.

Four invariants, taking `DeviceMeshValid` from 19 to 23:

- `DeviceMeshCollectivePayloadWellFormed` -- domains, the per-tensor pairing of
  shapes with dtype names, positive dimensions, and the recomputed element
  counts.
- `DeviceMeshPayloadCommKey` -- the payload is keyed on
  `process_group.canonical_id`, the same key as `CollectiveComm`, and NEVER on
  `runtime_pg_id`. The equality with `CollectiveComm` is a tripwire against the
  two exports drifting; the load-bearing half is that works sharing a payload
  key share a member set, which a per-rank local numbering cannot satisfy. That
  is the eight-versus-four communicator confusion, refuted rather than commented
  on.
- `DeviceMeshCollectivePayloadAgreement` -- THE property: the members of one
  collective agree on dtype and on volume, and a collective uses one dtype for
  both buffers, as NCCL requires.
- `DeviceMeshCollectivePayloadSizeRelation` -- THE second property: an
  all-gather's output volume is its member count times its input, a
  reduce-scatter's the inverse, an all-reduce's the same, with the exported
  relation label checked against the operation so the two cannot disagree
  silently. An operation the table does not know maps to a label outside the
  relation set and fails rather than passing vacuously.

Two named negatives, because the observed run satisfies both properties and a
positive result alone says nothing about what either would catch. Both are
overrides of the real `DeviceMeshFacts` rather than mutated multi-megabyte
copies, and both derive their target from the facts.

`DeviceMeshPayloadSizeInvalid.tla` doubles the first dimension of every shape of
ONE member of one collective, recomputing that member's element counts from the
mutated shapes. Doubling both sides keeps the operation's implied relation
intact, so the only thing it breaks is agreement between members. TLC must
report exactly `DeviceMeshCollectivePayloadAgreement`. Token:
`DEVICE_MESH_TLA_PAYLOAD_SIZE_NEGATIVE`.

`DeviceMeshPayloadOperationInvalid.tla` relabels one collective's operation on
EVERY member work, with the relation label moved to the one the wrong operation
implies -- what the exporter would have written had it read the operation
wrongly. TLC must report exactly `DeviceMeshCollectivePayloadSizeRelation`.
Token: `DEVICE_MESH_TLA_PAYLOAD_OPERATION_NEGATIVE`.

`PerCommunicatorIssueOrderAgreement` is in BOTH survivor sets, and that is the
whole argument for having these properties rather than treating ordering
agreement as sufficient. A volume mismatch leaves every communicator's operation
order exactly as observed. A symmetric mislabelling leaves it too, because all
four ranks then agree on the wrong label. Each control asserts that survival
over the mutated facts rather than claiming it in prose, and each cfg pins its
invariant order -- sentinel first, property under test last -- with the reason
in the file, because TLC reports only the first failing invariant and an
unpinned order silently decides which one a reader sees.

`DeviceMeshPayloadChecks.lean` and `DeviceMeshPayloadMutationChecks.lean` are
the Lean counterparts, under `DEVICE_MESH_LEAN_PAYLOAD` and
`DEVICE_MESH_LEAN_PAYLOAD_MUTATION`. Their facts are shaped differently from the
TLA ones on purpose, and the reason is measured. TLC evaluates an all-pairs
agreement predicate over 432 works without trouble; the Lean kernel does not --
a `rfl` over that form had not finished after four minutes, against a 120s
per-module budget. So the Lean facts add one row per collective, reached from
each work by a `Nat` index, plus one row per distinct communicator key, and the
predicates join through those instead. At the observed scale that is about 26s
for agreement and 13s for well-formedness per evaluation, which is also why the
positives and the mutations are separate modules with a 300s budget: `rfl`
caches nothing between theorems. Two things that form leaves elsewhere, stated
rather than hidden. The exporter takes each row's values from ONE member and
does NOT check that the others agree, because an exporter that refused a
mismatch would decide the property before any checker saw it. And the prefix
relation between the communicator key and the collective id -- the one check no
integer key could satisfy -- is asserted at the exporter seam by
`test_collective_payload_is_keyed_on_the_canonical_communicator_id`, because
every Lean route to a string prefix goes through `String.toList`, which is not
axiom-free under `rfl`.

`DEVICE_MESH_TLA_PAYLOAD_FACTS` reports the payload share of
`DeviceMeshFacts.tla` beside the whole-file size, for the same reason the
placement block is delimited. Read it with the same caveat: `payload_bytes=` is
exact, `parse_ms=` measures machine load first.

The Device-mesh suite includes all smoke and Single-rank targets:

```bash
scripts/run_formal_checks.sh --networked --suite device-mesh
scripts/run_formal_checks.sh --no-fetch --suite device-mesh
```

## Evaluations of the observed trace versus theorems about the protocol

These are two different kinds of result and the sealed log labels them
differently, because a reader meets the result there rather than in a source
comment.

The `DEVICE_MESH_MODEL_SAFETY` token additionally carries `invariants=`, the
list TLC was actually given, parsed out of the cfg by `formal_cfg_invariants`.
Without it the only invariant a log reader ever met by name was whichever one a
failure reported, and `StuckImpliesAllDone` was named nowhere at all.

Everything described above under Single-rank and Device-mesh is an EVALUATION of
the observed trace: a `Bool`-valued predicate applied to literal fact data and
closed by `rfl` or `decide`. That is a kernel-checked, axiom-free statement
about THIS run, and nothing more. Its tokens -- `SINGLE_RANK_LEAN_VALID`,
`SINGLE_RANK_LEAN_NEGATIVE`, `DEVICE_MESH_LEAN_VALID`,
`DEVICE_MESH_LEAN_MUTATION`, `DEVICE_MESH_LEAN_NEGATIVE`,
`DEVICE_MESH_LEAN_PLACEMENT`, `DEVICE_MESH_LEAN_PLACEMENT_MUTATION`,
`DEVICE_MESH_LEAN_PAYLOAD`, `DEVICE_MESH_LEAN_PAYLOAD_MUTATION` -- carry
`kind=evaluation scope=observed-trace`.

`lean_device_mesh_protocol_test` is the other kind. It reads no facts module.
Its three Lean modules encode the `DeviceMeshModel.tla` protocol and prove
theorems quantified over topologies, states and schedule lengths. It emits three
vocabularies, not one, because labelling all of its results as general theorems
would repeat the same category error one layer up:

- `kind=theorem scope=all-topologies-all-schedules bound=none` -- universally
  quantified over `Topology`, `State` and `Step`. `bound=none` means NO
  PARTICULAR BOUND IS ASSUMED, not that `maxIssues` is absent from the model:
  `Reachable` does depend on it through `issueAllowedB`, and what makes
  `bound=none` legitimate is that the `Topology`, hence its `maxIssues`, is
  universally quantified, so the result holds at every bound including 108.
- `kind=witness scope=fixed-instance bound=<n>` -- `by decide` over one fixed
  finite topology whose `maxIssues` is `<n>`. Real kernel-checked results about
  that instance and nothing more; two of them are existential rather than
  universal. They are what shows the general theorems' hypotheses and
  conjunctions carry weight, so they belong in the log under their own label.
- `kind=theorem-negative scope=fixed-instance bound=<n>` -- a proved NEGATION
  of a general statement, refuted by a fixed instance.

Three results additionally carry `conditional=acyclic-wait-for-graph`, because
"DeadlockFreedom holds" and "DeadlockFreedom holds where the wait-for graph is
acyclic" are different claims and the second must not read as the first.

The modules:

- `DeviceMeshProtocol.lean` -- the encoding. Its header holds the
  definition-by-definition correspondence to `DeviceMeshModel.tla` and ten named
  divergences, each with the direction it moves the claim. That correspondence
  is a SECOND reviewable claim: there is no mechanical link between the Lean
  file and the TLA+ module, so a reviewer has to check it by inspection. Three
  of the divergences are discharged as Lean lemmas
  (`idxOfOn_spec`, `idxOfOn_isSome_of_le`, `opAtOn_append_of_le`) rather than
  left to inspection, and those have their own tokens.
- `DeviceMeshInductiveInvariant.lean` -- `initiation`, `consecution` and
  `sufficiency`, each a separate theorem with a separate token because a
  missing obligation silently weakens the claim, plus `safetyOfReachable`,
  whose statement does not mention `maxIssues`. This is what lifts the
  `MaxIssues = 2` bound off `RendezvousOpAgreement`,
  `RendezvousMembership` and `CommFifo`. `commFifoAloneIsNotInductive` shows
  the conjunction is the strengthening rather than decoration.
- `DeviceMeshWaitGraph.lean` -- the general protocol theorem
  (`orderAgreementAndAcyclicWaitGraphExcludeBothHazards`): order agreement plus
  an acyclic communicator wait-for graph gives neither a rendezvous mismatch
  nor a stream-ordered circular wait. Its first component is the whole of
  `Safety T s`, not just `RendezvousOpAgreement`: that conjunct compares two
  `Option`s and holds vacuously where a running communicator is short of a
  member's issue, and `RendezvousMembership` -- which travels inside `Safety` --
  is what forces both sides to `some`. It pairs with
  `DeviceMeshModelDivergent.cfg`, which refutes the converse reading in TLC.
  `acyclicityIsLoadBearing` refutes the statement with the acyclicity
  hypothesis dropped, and `waitGraphWitnessIsNotVacuous` shows the witness's
  guards are satisfiable rather than false by construction.

`DeadlockFreedom` is proved only CONDITIONALLY, under the acyclicity
hypothesis. `StuckImpliesAllDone` has no bound-free version at all, since
`AllDone` is defined by `Len(issued[r]) = MaxIssues`. Both are stated in
`DeviceMeshInductiveInvariant.lean`'s header. The unbounded, unconditional
deadlock-freedom claim remains open.

`DeviceMeshProtocolInvalid.lean` is the controlled negative Lean must reject,
matching the shape of `SingleRankInvalid` and `DeviceMeshInvalid`: it asserts
the wait-graph witness is not stuck, `decide` proves that false, and
`DEVICE_MESH_PROTOCOL_LEAN_NEGATIVE` records the rejection.

No new toolchain was pinned for any of this and no Mathlib was added. The
proofs use core Lean 4.34.0 only, which is what keeps `#print axioms` empty:
`omega` depends on `propext` and `Quot.sound`, `simp` on `propext`, most core
`List` lemmas on `propext`, and anything classical on `Classical.choice`, so
none of them appears in these modules.

## Single-rank abstract model and the refinement bridge

`SingleRankValid` and `DeviceMeshValid` check the observed trace. They walk a
cursor along a constant sequence of generated facts and evaluate predicates over
it. That is a real check, and the negative controls show the predicates can
fail, but it is worth being precise about its reach: TLC explores one path, so
those modules say nothing about any execution that was not observed.
Device-mesh's spec in particular has a two-state stutter whose `cursor` no
invariant mentions.

`SingleRankModel` is a different kind of artifact. It is a transition system for
one single-rank training step, with guarded actions and no dependence on the
generated facts. Two sources of nondeterminism are modelled. Both are
possibilities the protocol has to be safe under, not orders the instrumentation
has recorded -- the Single-rank tracer treats each of them as an error:

- gradient readiness races backward completion, so `gradient.ready` may land
  before or after `backward.completed`. The tracer raises `model backward
  completed before the tracked parameter gradient was ready` if it observes the
  second order, so no exported trace contains it;
- gradients may fail to materialise at all, in which case the optimizer must not
  mutate parameters. `gradient.missing` is produced only by the synthetic
  corruptor that builds the negative fact modules.

Modelling them is the point of having a model rather than a replay: the
invariants then hold for executions nobody ran, including ones this
instrumentation could not record. What they are not is evidence that either
order occurs in practice.

TLC explores every admitted interleaving -- currently 25 distinct states, with
outdegree up to 3 -- and the safety invariants hold across all of them.

### The guard, and the negative control

`RequireReadyGradients` is exposed as a TLA+ constant so the negative
configuration relaxes exactly one guard and nothing else. TLC then discovers a
counterexample by exploration rather than by anyone editing a fact:

```text
gradient.missing -> optimizer.started -> optimizer.mutated
```

That is a genuine execution in which parameters are mutated after gradients were
reported missing. It is categorically different evidence from flipping a
constant in a generated fact module.

### Refinement

`SingleRankRefine` restricts the model's transitions so the emitted sequence
must stay a prefix of the observed one, and asks whether the complete observed
sequence is reachable. It is. Without this check the model and the trace would
be unrelated artifacts.

Two things about it are easy to overstate, so state them flatly.

TLC does **not** print the alignment of model actions to observed events. Every
step of the witness trace is labelled `<ConstrainedNext line ..., col ... of
module SingleRankRefine>`, because `ConstrainedNext` is `RSpec`'s only action.
Which model action produced each event is inferable from the state variables in
the printed states; the checker does not say it.

And the bridge is an agreement between two artifacts, not a per-run check.
`single_rank.py` requires the normalized event kinds to equal one 10-element
literal, so `EventKinds` is that same constant on every run and a deviating run
is refused by the Python validator before any facts are exported. What
`SingleRankRefine` establishes is therefore that *that constant sequence* lies
in the abstract model's language -- that the independently written model and the
independently written validator agree on what a legal step looks like. Catching
a lifecycle deviation is the validator's job, not this check's.

The check is still guarded against becoming vacuous: an empty `Observed` is
admitted by every model, and produces the same "exit 0, no violation" that a
*refused* trace produces. `ObservedIsNonEmpty` is checked alongside, and because
it mentions no variables TLC refutes a false one as a constant expression with
its own status and message, which is what lets the runner report the vacuous
case as vacuous instead of as a refusal.

`SingleRankRefineBad` is the control, and it must be refused **for the right
reason**. An earlier version moved `optimizer.mutated` ahead of
`gradient.ready`; the model refused it, but because the lifecycle was out of
order -- the optimizer had not started -- which exercises a different guard than
the one under test. The control now substitutes `gradient.missing` for
`gradient.ready` and changes nothing else, so every earlier event stays
admissible, the optimizer legitimately starts, and the trace dies exactly at
`OptimizerMutate`.

Three checks pin that down. `SingleRankRefineBad.cfg` shows the trace is not
admitted; `SingleRankRefineBadReach.cfg` shows it *was* admitted up to the
mutation, so a control refused too early fails the gate rather than counting as
a success; and `SingleRankRefineBadRelaxed.cfg` reruns the same corrupted trace
with `RequireReadyGradients = FALSE` and requires `ObservedTraceIsNotAdmitted`
to be violated. The third is the decisive one: it attributes the refusal to that
one constant directly, where the first two only triangulate where the refusal
landed. `CorruptionIsIsolated` additionally proves the trace differs from the
observed one at exactly one position, and `ControlEventsArePresent` refuses the
degenerate case in which the events the corruption targets are absent and the
index sentinels make the "corruption" a no-op.

One polarity note, stated loudly because it reads backwards. TLC proves
reachability by refutation, so the witness that the observed trace is admitted
is a *violation* of `ObservedTraceIsNotAdmitted`. The runner translates that
into `SINGLE_RANK_REFINEMENT result=admitted`. All four refinement
configurations set `CHECK_DEADLOCK FALSE` so that a refused trace completes
cleanly (exit 0, no witness) instead of surfacing as TLC exit 11, which is
indistinguishable from a genuine specification defect.

### What this does and does not establish

This section is about Single-rank. It establishes safety properties of a bounded
abstract model of one single-rank step, and that the one exported event-kind
sequence lies inside that model.

The abstract DPxTP protocol is a separate model, `DeviceMeshModel`, with its own
eleven checks and its own claim inventory; see "Tier 0: the trace-free suite"
above for what each of its result tokens establishes, and the header of
`DeviceMeshModel.tla` for the exact claims. Its refinement bridge to the
observed four-rank trace is `DeviceMeshRefine`, described in "DPxTP refinement
bridge" below; `DeviceMeshValid` and `DeviceMeshIssueOrderInvalid` remain cursor
walks over the generated facts, not refinement.

The bridge is over event **kinds** only. `SingleRankRefine` projects the
observed trace to its `EventKinds` sequence, so the refinement argument covers
the order and presence of lifecycle events and nothing else. Event identities,
phases, predecessor sets, raw-source provenance and projection digests are
checked by Python and by the `SingleRankValid`/Lean fact modules, but they are
not part of what the model admits. A trace could therefore be admitted here
while carrying corrupt provenance; that is the other checkers' job, and this one
does not subsume them.

`SingleRankModel` itself does not model the 2x2 DPxTP topology, collectives,
multi-step training, convergence, or performance. It is not claimed to be a
faithful model of TorchTitan -- only of the step lifecycle contract stated in
its own actions, and its value depends on that contract being the right one,
which no checker decides. The same caveat applies to `DeviceMeshModel` for the
collective protocol.

## DPxTP refinement bridge

`DeviceMeshRefine` asks whether the observed four-rank run is an admitted
behaviour of `DeviceMeshModel`. Its target is `tlc_device_mesh_refine_test`,
which runs in the sealed Device-mesh suite and not in tier 0, because it reads
the generated `DeviceMeshFacts`. Seven checks; the module header carries the
same list as its claim inventory and the two must agree.

**Only the issue order is replayed.** The observer appends each collective's
enqueued/started/completed triple contiguously, post hoc, from the Flight
Recorder, and the raw-event projection drops `observed_time_ns`, so per-rank
*event* order asserts a full serialization of every collective before the next.
That cannot have happened -- FSDP all-gathers overlap by construction -- and a
bridge replaying it would **pass**, because a serialized schedule satisfies
every guard trivially. Only the `collective.enqueued` sub-order is faithful, so
only `CollectiveIssueOrder` is read. `Start` and `Complete` are taken by the
model's own guards, and the start/complete schedule along the witness path is
model-chosen, not claimed to be the schedule that ran.

**What the positive establishes, exactly.** At each of the 108 positions the
four ranks' recorded issues form a *column*. The positive establishes that every
column is model-typable and cross-rank consistent -- each rank's issue names a
communicator it belongs to, carrying an operation that communicator admits, and
at every position the four ranks agree on the operation and are at the same
communicator wherever their communicators share a rank -- and that the
collectives those columns induce drain to completion under the model's own
`Start` and `Complete` guards.

It does **not** establish that the observed absolute order is the only order the
model admits, and it cannot. `OpsAgreeAtFront` quantifies over the members at a
communicator's front; `UniformProgramOpsOK` and `UniformProgramCommsOK` quantify
over the ranks at a position. All three are relational across ranks at a
position and none mentions which collective a position ought to carry. That is
correct NCCL semantics: members of a communicator must *agree* on their
collective sequence, not follow any particular one, so a permutation applied
uniformly to all four ranks is an equally valid SPMD program and must be
admitted. Measured: reversing every rank's sequence is admitted with the same
9,172 generated / 5,371 distinct / depth 865 the positive reports, and relaxing
`RequireMatchedIssueOrder`, `RequireStreamOrder` or either SPMD guard on the
observed order leaves those numbers unchanged.

So the limit is a checked pair rather than a caveat.
`DeviceMeshRefineUniformPermutation.cfg` reverses every rank's sequence and the
run is **admitted**; `DeviceMeshRefineSingleRankPermutation.cfg` reverses one
rank's sequence, which breaks the columns, and the run is **refused** at depth
6. The second half is the evidence that the bridge detects agreement violations
-- the property NCCL imposes and the one a real job hangs on. Both halves
assert, through the model's own `UniformProgram*OK` predicates, that their input
really does preserve or break agreement, so neither can pass for an unrelated
reason.

**Admission is completion, not length.** `Issue`'s guard never reads `doneOn`,
so "every rank reached its observed issue count" is satisfied by issuing
everything and running nothing -- for a corrupted issue order as much as for the
real one. The admission predicate is therefore `AllDone`: every rank issued the
replayed prefix, in order, and every collective completed.

**The communicator mapping is derived, not tabulated.** The facts key
collectives by canonical id (`tp:0,1:mesh_tp`, `dp_shard:0,2:mesh_fsdp`, ...)
and the model by its own eight names. The mapping is computed from the recorded
member sets and operations -- no communicator name of either side appears in the
derivation -- and then asserted total, injective, onto `CommIds`, member-set
preserving and operation-respecting. The derived mapping is printed by the
checker and echoed by the runner, so a reviewer reads the mapping that was used
rather than one quoted in prose. It is not unique and does not need to be:
members plus operations pin four of the eight images and leave two independent
two-element swaps, so there are four valid mappings. The `{0,1}` and `{2,3}`
classes hold one observed and one model communicator each; in `{0,2}` and
`{1,3}`, `mesh_fsdp` is pinned by carrying three operations and `mesh_batch` and
`mesh_loss_mesh` are interchangeable. The model gives that pair equal members
and equal `CommOps`, so swapping them is an automorphism of the whole
specification.

**Cost control lives in the bridge, not in the model.** `Next` is untouched.
`GreedyReplay` narrows the search to a drain-then-advance schedule and
`MaxReplaySkew = 1` holds the four ranks in lockstep, which makes the witness
linear in the trace: 865 steps, 5,371 distinct states for all 108 issues of all
four ranks. The same witness is found at looser spreads and costs more (9,199
states at 2, 13,261 at 4), so the bound is reported rather than assumed.

That is sound for a positive result without any
confluence argument -- every step of `ConstrainedNext` is a step of `Next`, so a
witness is a witness, and narrowing a search can only lose witnesses. The
converse direction, reading a greedy failure as non-admission, is **not**
claimed from that configuration.

The witness path is serialized: at most one collective in flight. A serialized
schedule satisfies every guard easily, so the positive result is evidence about
the issue order rather than about scheduling. That it is not vacuous is shown by
the negative control, which runs under the same serialized policy and is still
refused; overlapping schedules are covered by the confluence configuration
below.

**Confluence is machine-checked over a declared fragment.**
`DeviceMeshRefineSkew.cfg` drops the greedy policy and explores every schedule
whose per-rank issue counts stay within `MaxReplaySkew` of one another and which
keeps at most `MaxOutstanding` issues in flight per rank and communicator, over
the first `MaxIssues` issues of each rank. `CHECK_DEADLOCK` is on and this is
the only refinement configuration where it is: `Terminated` self-loops on
`AllDone` alone, so "no dead end anywhere" gives "every schedule in the fragment
reaches completion" **provided the graph is otherwise acyclic**. That proviso is
an unchecked lemma, stated as prose in the module header and argued from a
lexicographic measure; a finite state graph may perfectly well contain cycles,
so the inference is only as good as that argument. TLC checks the absence of
dead ends, not the acyclicity. A pass also certifies that the two cost-control
bounds are not themselves obstructive. `DeviceMeshRefineOverlap.cfg` then
refutes `NoTwoCollectivesRunConcurrently` over the same constants, which is what
keeps the result from being a statement about serialized schedules only. The
bound of 54 is chosen, not convenient: rank 0's first `reduce_scatter` is at
position 52 and the pair the negative control transposes is 52/53, so a shorter
prefix contains no `reduce_scatter` at all and the `rs` stream -- and with it
the alternation between FSDP2's two dedicated streams on one communicator --
never appears. `all_gather` is in play from position 2, so the `ag` stream is
not what this bound buys.

The fragment is a fragment: it says nothing about
positions beyond its bound or about wider skew, and the runner reports every
bound in its token.

**The negative control transposes two adjacent issues of one rank on one
communicator, with differing operations.** The site is derived from the facts,
minimal index first; on this trace it is rank 0, positions 52 and 53, the
`reduce_scatter`/`all_gather` pair on the FSDP communicator. Same-communicator
adjacency is what makes the rendezvous guard reachable at all: transposing
across different communicators would be refused because a member never reached
a communicator's front, which is the wrong guard.

A single-rank corruption is an agreement violation, and agreement is guarded in
two places, so which guard refuses it depends on the configuration. Under the
**positive's** guard set the operations disagree at that position with every
other rank, so `RequireUniformProgramOps` refuses every *peer's* issue at that
position and the corrupted column never forms: that is
`DeviceMeshRefineBadUniform.cfg`, where `CorruptedColumnIsNeverFormed` holds --
2,555 distinct states, depth 414. The corrupted record itself is still issued,
because `UniformProgramOpsOK` compares positions only up to the shorter of two
sequences and the frontier position is not yet compared with anything. That is
exactly the real-world shape of the fault: one rank runs ahead, its peers cannot
follow, the job stops. The refusal is correct, but it is not the guard the
control exists to exercise.

So the three configurations that isolate the rendezvous guard --
`DeviceMeshRefineBad`, `...BadReach` and `...BadRelaxed` -- all set
`RequireUniformProgramOps = FALSE`. That is necessary for `...BadRelaxed`: with
the SPMD guard on, flipping `RequireMatchedIssueOrder` does not admit the
corruption and the attribution to that one constant collapses. It is **not**
necessary for `...BadReach` -- measured, that witness appears at the same depth
with the guard either way -- and is set there only to keep the three on one
constant set, so the only differences among them are the ones the runner's cfg
diff checks. `DeviceMeshRefineBadRelaxed.cfg` then flips
`RequireMatchedIssueOrder` alone and the same corruption completes, which
attributes that refusal to NCCL's matching requirement directly.

**Why `-Xss` is raised.** `DeviceMeshModel`'s SPMD-program guards contain
`\A r1 \in Ranks : \A r2 \in Ranks : \A k \in 1..MinOf(len1, len2)` inside
`IssueAllowed`, which is evaluated in action position, where TLC recurses once
per bound element. At `MaxIssues = 2`, where every other configuration of that
model runs, the innermost range has two elements; replaying 108 issues per rank
makes it 4 x 4 x 108 and TLC overflows the default 1 MB thread stack at around
replay depth 258. Attribution was verified by rerunning the same configuration
with both uniformity guards relaxed, which completes the whole 865-step replay
at the default stack size. The runner uses `-Xss32m`; the replay completes at
`-Xss8m`.
