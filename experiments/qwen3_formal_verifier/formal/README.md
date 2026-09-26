# Qwen3 Step Formal Trace

This directory contains generated formal artifacts for a TorchTitan-style Qwen3 dense DPxTP training step.

- `Qwen3StepTrace.tla` captures the finite DPxTP mesh, process groups, work ids, work roles, and group membership used by the TLA+ transition model.
- `Qwen3StepTrace.lean` captures the same finite trace facts in a Lean-friendly shape for trace-level invariant proofs.

The source of truth is the Python trace IR emitted by `build_torchtitan_qwen3_step_trace`; regenerate these files through `write_qwen3_step_formal_artifacts` after schema or role changes.

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

## Scout A lifecycle suite

The Scout A fact modules are generated from the normalized trace of a real
single-rank CUDA run:

- `ScoutAFacts.tla` and `ScoutAFacts.lean` contain observed identities, event
  values, causal predecessors, and raw-source links only.
- `ScoutLifecycle.tla` and `ScoutLifecycle.lean` are independently maintained
  semantics for forward/backward order, gradient readiness before optimizer
  mutation, optimizer mutation before step completion, and causal order.
- `ScoutAValid.*` applies those semantics to the observed facts.
- `ScoutABadFacts.*` removes the gradient-ready fact for one controlled
  negative; `ScoutAInvalid.*` must reject the named property.

Run the complete suite through the same fresh-sandbox wrapper:

```bash
scripts/run_formal_checks.sh --networked --suite scout-a
scripts/run_formal_checks.sh --no-fetch --suite scout-a
```

TLC must accept all valid lifecycle invariants and report exactly
`ScoutAGradientReadyBeforeOptimizer` for the controlled transition mutation.
Lean must report that `Qwen3Formal.ScoutA.validLifecycle` has no axioms and
reject `Qwen3Formal.ScoutA.ControlledInvalidProposition`. Each checker stages
sources and writes state or `.olean` products only under Bazel's external test
directory.

## Scout B DPxTP suite

Scout B's generated modules contain fact values only. The exporter maps the
complete normalized four-rank bundle to all 1,340 event rows, 432 observed
rank-local collective work rows, and 222 synchronization edges.
`ScoutDistributed.tla` and `ScoutDistributed.lean` independently define the
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

`ScoutBEventRank*.lean`, `ScoutBSyncChunk*.lean`, and
`ScoutBCollectiveEventRank*.lean` partition the same complete facts so Lean can
kernel-reduce each bounded check. `ScoutBEventMutationChecks.lean` proves that
paired String-row drift, a wrong kind, and a wrong order are each rejected.
`ScoutBValid.lean` composes those axiom-free results.

The four generated `ScoutBFacts` and `ScoutBBadFacts` TLA+/Lean modules exceed
the repository's general 500 KiB added-file limit because each carries the
complete bounded graph. They have an exact-path exception in
`.pre-commit-config.yaml`: the supported checker runners consume these plain
modules directly, and artifact synchronization compares their complete bytes.
Compressing them would make them unusable as checker inputs, while splitting
them would change the generated module and import contract rather than repair
the lint gate. All other added files remain subject to the 500 KiB limit.

`ScoutBBadFacts.*` changes one completion producer while preserving every other
fact. TLC must report exactly the transition invariant
`ScoutBCollectiveProducerCorrelation`. Lean must report that
`Qwen3Formal.ScoutB.validDPxTP` has no axioms and must reject exactly
`Qwen3Formal.ScoutB.ControlledInvalidProducerProposition`.

The Scout B suite includes all smoke and Scout A targets:

```bash
scripts/run_formal_checks.sh --networked --suite scout-b
scripts/run_formal_checks.sh --no-fetch --suite scout-b
```

## Abstract model and the refinement bridge

`ScoutAValid` and `ScoutBValid` check the observed trace. They walk a cursor
along a constant sequence of generated facts and evaluate predicates over it.
That is a real check, and the negative controls show the predicates can fail,
but it is worth being precise about its reach: TLC explores one path, so those
modules say nothing about any execution that was not observed. Scout B's spec
in particular has a two-state stutter whose `cursor` no invariant mentions.

`ScoutAModel` is a different kind of artifact. It is a transition system for one
single-rank training step, with guarded actions and no dependence on the
generated facts. Two sources of nondeterminism are modelled because both are
real executions:

- gradient readiness races backward completion, so `gradient.ready` may land
  before or after `backward.completed`;
- gradients may fail to materialise at all, in which case the optimizer must not
  mutate parameters.

TLC explores every admitted interleaving -- currently 25 distinct states, with
outdegree up to 3 -- and the safety invariants hold across all of them. An
invariant that holds here holds for executions nobody ran.

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

`ScoutARefine` restricts the model's transitions so the emitted sequence must
stay a prefix of the observed one, and asks whether the complete observed
sequence is reachable. It is, and TLC prints the alignment of model actions to
observed events. This is the bridge that makes runtime evidence mean something:
without it, the model and the trace are unrelated artifacts.

`ScoutARefineBad` is the control, and it must be refused **for the right
reason**. An earlier version moved `optimizer.mutated` ahead of
`gradient.ready`; the model refused it, but because the lifecycle was out of
order -- the optimizer had not started -- which exercises a different guard than
the one under test. The control now substitutes `gradient.missing` for
`gradient.ready` and changes nothing else, so every earlier event stays
admissible, the optimizer legitimately starts, and the trace dies exactly at
`OptimizerMutate`.

Two checks pin that down: `ScoutARefineBad.cfg` shows the trace is not admitted,
and `ScoutARefineBadReach.cfg` shows it *was* admitted up to the mutation. A
control refused too early therefore fails the gate rather than counting as a
success. `CorruptionIsIsolated` additionally proves the trace differs from the
observed one at exactly one position.

One polarity note, stated loudly because it reads backwards. TLC proves
reachability by refutation, so the witness that the observed trace is admitted
is a *violation* of `ObservedTraceIsNotAdmitted`. The runner translates that
into `SCOUT_A_REFINEMENT result=admitted`. Both configurations set
`CHECK_DEADLOCK FALSE` so that a refused trace completes cleanly (exit 0, no
witness) instead of surfacing as TLC exit 11, which is indistinguishable from a
genuine specification defect.

### What this does and does not establish

It establishes safety properties of a bounded abstract model of one single-rank
step, and that one observed run lies inside that model.

The bridge is over event **kinds** only. `ScoutARefine` projects the observed
trace to its `EventKinds` sequence, so the refinement argument covers the order
and presence of lifecycle events and nothing else. Event identities, phases,
predecessor sets, raw-source provenance and projection digests are checked by
Python and by the `ScoutAValid`/Lean fact modules, but they are not part of what
the model admits. A trace could therefore be admitted here while carrying
corrupt provenance; that is the other checkers' job, and this one does not
subsume them.

It does not model the 2x2 DPxTP topology, collectives, multi-step training,
convergence, or performance. The abstract model is not claimed to be a faithful
model of TorchTitan -- only of the step lifecycle contract stated in its own
actions, and its value depends on that contract being the right one, which no
checker decides.
