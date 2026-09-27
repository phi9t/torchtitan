# 27 — Differential-test the TLA+ and Lean models against each other

**What to build:** A generator of random topologies and states, evaluated by
both TLC and Lean, asserting the shared predicates agree. Replaces a prose
correspondence table verified once by inspection.

**Blocked by:** none. **Blocks:** any future claim that the unbounded Lean proof
describes the system TLC checks.

**Status:** both halves proven on one instance with full agreement; generator
and comparator outstanding

## What is wrong

`ScoutBModel.tla` and `ScoutBProtocol.lean` are two hand-written models of one
protocol, joined by a prose correspondence table with ten named divergences,
checked once by a reviewer reading both. If they drift, the unbounded Lean proof
describes a different system from the one TLC checks, and nothing says so.

The Lean side already asserts the correspondence in its own docstrings --
`opsAgreeAtFrontB` is documented as "TLA's `OpsAgreeAtFront(c)`",
`uniformOpsOKB` as "TLA's `UniformProgramOpsOK(candidate)`". The intent is
recorded; only the agreement is unverified.

## Feasibility, measured rather than assumed

The plan's abandon criterion was: stop if the two models' predicates cannot be
evaluated on common inputs without a third translation layer. That criterion is
**not** triggered.

The Lean model is Bool-valued and computable, not opaque `Prop`. `Topology`
carries `requireStreamOrder : Bool` and friends, `Issue` derives
`DecidableEq`, and the nine shared predicates are `def ... : Bool`. A spike
evaluated all nine on a concrete instance and printed them:

```text
fullyPending_c0=true   fullyPending_c1=true   opsAgreeAtFront_c0=true
sameSite_c0_c1=false   startAllowed_c0=false  allDone=false
stuck=true             completed_r0_k1=false  uniformOps=true
uniformComms=false
```

Two facts the harness depends on, both established by that spike:

- Using `Fin n` for the rank, communicator, op and stream types makes
  `Topology`'s three proof obligations -- `membersInRanks`, `admissibleInOps`,
  `commsComplete` -- decidable, so each is discharged by `by decide`. The
  existing `cyclicTopology` in `ScoutBWaitGraph.lean` uses `Bool`/`Unit` with
  hand-written `cases` proofs, which does not generalise to generated
  instances; `Fin` does.
- `ScoutBProtocol.lean` compiles standalone, so the harness needs no other
  module: `lean -o ScoutBProtocol.olean ScoutBProtocol.lean` then `lean` the
  generated file with `LEAN_PATH` set. The toolchain is vendored at
  `/project/formal-cache/bazel/vendor/+http_archive+lean_4_34_0/bin/lean`
  inside the rootfs.

### The TLA+ half, also proven

The second abandon criterion was: stop if TLC cannot evaluate a predicate at an
externally supplied state without a hand-written translation per instance. It
also is **not** triggered.

A module that `EXTENDS ScoutBModel, TLC` inherits the model's variables and
operators, so it can pin the state and print the predicates directly:

```tla
PinnedInit ==
  /\ issued = [r \in Ranks |-> ...]      \* the supplied state
  /\ running = {}
  /\ doneOn  = [c \in CommIds |-> 0]
PinnedNext == UNCHANGED vars
PinnedSpec == PinnedInit /\ [][PinnedNext]_vars
Probe == /\ PrintT(<<"stuck", Stuck>>) /\ ... /\ TRUE
```

with a cfg carrying `SPECIFICATION PinnedSpec`, `INVARIANT Probe`, and the seven
constants bound by `CONSTANT X <- ProbeX` definition override. TLC evaluates the
invariant at the single initial state and prints each pair. Nothing is
hand-written per instance: the probe module and its cfg are both generated from
the same instance data as the Lean module.

`ScoutBModel.tla` extends only `Naturals`, `Sequences` and `FiniteSets`, so the
probe needs no other module. TLC runs under the Bazel-vendored JDK at
`/project/formal-cache/bazel/vendor/rules_java++toolchains+remotejdk17_linux/bin/java`
-- `java` is not on the rootfs PATH -- with
`/project/formal-cache/bazel/vendor/+http_file+tla2tools_1_7_4/file/tla2tools.jar`.

### The two engines already agree on one instance

Same topology and state driven through both, for a state where rank 0 issues
`c0` then `c1` and rank 1 issues them in the opposite order:

| predicate | Lean | TLC |
|---|---|---|
| `fullyPending_c0` | true | TRUE |
| `fullyPending_c1` | true | TRUE |
| `opsAgreeAtFront_c0` | true | TRUE |
| `sameSite_c0_c1` | false | FALSE |
| `startAllowed_c0` | false | FALSE |
| `allDone` | false | FALSE |
| `stuck` | true | TRUE |
| `completed_r0_k1` | false | FALSE |
| `uniformOps` | true | TRUE |
| `uniformComms` | false | FALSE |

Ten for ten. One instance is not evidence of fidelity -- it is evidence that the
instrument works, which is what the spike was for.

## The shared predicate set

Nine pairs, all boolean-valued on both sides:

| Lean | TLA+ |
|---|---|
| `completedB s r k` | `Completed(r, k)` |
| `fullyPendingB T s c` | `FullyPending(c)` |
| `opsAgreeAtFrontB T s c` | `OpsAgreeAtFront(c)` |
| `sameSiteB T c1 c2` | `SameSite(c1, c2)` |
| `uniformOpsOKB T f` | `UniformProgramOpsOK(candidate)` |
| `uniformCommsOKB T f` | `UniformProgramCommsOK(candidate)` |
| `startAllowedB T s c` | `StartAllowed(c)` |
| `allDoneB T s` | `AllDone` |
| `stuckB T s` | `Stuck` |

## Shape

1. A generator emitting N random instances as data: ranks, communicators,
   membership, admissible ops, stream assignment, `maxIssues`, the four
   `require*` flags, and a state of per-rank issue lists, `running` and
   `doneOn`.
2. Per instance, a Lean module and a TLA+ module plus cfg that evaluate the nine
   predicates at that exact state, each printing a comparable record.
3. A comparator asserting the two records are equal, reporting the instance and
   the predicate on any disagreement.

The ten prose divergences become the specification of what may legitimately
differ. Each is annotated with the direction it moves the claim, and three are
already Lean lemmas rather than inspection items -- those three must agree
exactly; the rest need their documented exemption encoded, not silently ignored.

## Acceptance

- A disagreement on any shared predicate fails the suite and names instance and
  predicate. A harness that cannot fail is worthless here.
- Prove it can fail: perturb one predicate on one side and show the comparator
  catches it. This is the mutation for this ticket.
- Fixed seed and recorded instance count in the result token, so a passing run
  states how much was actually compared rather than implying exhaustiveness.
- Generated instances must be small enough for TLC to evaluate quickly; the
  point is breadth of shapes, not depth of state space.

## Abandon criterion

If the TLA+ side cannot be driven to evaluate a predicate at an externally
supplied state without a hand-written translation per instance, stop and report.
A third hand-written layer would itself need verification and the exercise
becomes circular.
