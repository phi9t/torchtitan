# 27 — Differential-test the TLA+ and Lean models against each other

**What to build:** A generator of random topologies and states, evaluated by
both TLC and Lean, asserting the shared predicates agree. Replaces a prose
correspondence table verified once by inspection.

**Blocked by:** none. **Blocks:** any future claim that the unbounded Lean proof
describes the system TLC checks.

**Status:** BUILT. Generator, both emitters, both runners and the comparator
land in `torchtitan/experiments/qwen3_formal_verifier/fidelity_diff.py`, with
the runnable wrapper at `experiments/qwen3_formal_verifier/run_fidelity_diff.sh`
and the suite at `tests/unit_tests/test_qwen3_formal_fidelity_diff.py`. See
"Delivered" at the end of this file. Not yet wired into
`run_formal_tier0.sh`.

## What is wrong

`DeviceMeshModel.tla` and `DeviceMeshProtocol.lean` are two hand-written models
of one protocol, joined by a prose correspondence table with ten named
divergences, checked once by a reviewer reading both. If they drift, the
unbounded Lean proof describes a different system from the one TLC checks, and
nothing says so.

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
  existing `cyclicTopology` in `DeviceMeshWaitGraph.lean` uses `Bool`/`Unit`
  with hand-written `cases` proofs, which does not generalise to generated
  instances; `Fin` does.
- `DeviceMeshProtocol.lean` compiles standalone, so the harness needs no other
  module: `lean -o DeviceMeshProtocol.olean DeviceMeshProtocol.lean` then `lean`
  the generated file with `LEAN_PATH` set. The toolchain is vendored at
  `/project/formal-cache/bazel/vendor/+http_archive+lean_4_34_0/bin/lean` inside
  the rootfs.

### The TLA+ half, also proven

The second abandon criterion was: stop if TLC cannot evaluate a predicate at an
externally supplied state without a hand-written translation per instance. It
also is **not** triggered.

A module that `EXTENDS DeviceMeshModel, TLC` inherits the model's variables and
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

`DeviceMeshModel.tla` extends only `Naturals`, `Sequences` and `FiniteSets`, so
the probe needs no other module. TLC runs under the Bazel-vendored JDK at
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

## Delivered

### What runs

```
experiments/qwen3_formal_verifier/run_fidelity_diff.sh [--self-test]
python -m torchtitan.experiments.qwen3_formal_verifier.fidelity_diff \
  --seed 42 --instances 192 --work-dir <dir>
pytest -q tests/unit_tests/test_qwen3_formal_fidelity_diff.py
```

Four generator families, all drawn from one seeded stream and prefix-stable in
the instance count:

- `uniform_random` -- independent membership, admissible ops, stream map,
  `maxIssues`, four `require*` flags, per-rank issue lists, `running`,
  `doneOn`;
- `spmd_permutation` -- one base program permuted per rank, all ranks in all
  communicators, which is what makes `fullyPending` true and produces
  stream-ordered circular waits;
- `spmd_terminal` and `spmd_terminal_running` -- fully issued and fully
  counted, with and without one communicator left running. The second exists
  because without it, deleting `running = {}` from `AllDone` is undetectable;
- `front_op_mismatch` -- one communicator issued twice with two operations,
  permuted per rank, so `OpsAgreeAtFront` is false inside the region where TLA
  defines it. Without this family the draw compared that case 4 times in 192
  instances.

The TLA+ cfg binds `Ops` and `StreamOfIssue` by definition override, not only
the seven declared constants. Both are ordinary definitions in
`DeviceMeshModel.tla`; without overriding them the generator would inherit the
module's single hard-wired 2x2 instance and could vary neither the operation
alphabet nor the operation-to-stream map. Every cfg also asserts TLA's FULL
`TypeOK`, including the two conjuncts divergence 7 drops from the Lean
encoding, so a generated state that is not well formed aborts instead of
producing a meaningless agreement.

### Measured result, seed 42, 192 instances

```
QFV_FIDELITY_DIFF result=success seed=42 instances=192
  predicates_compared=3660 exempt_records=1377
  exemption_obligations_checked=1332 disagreements=0 vacuity_problems=0
  lean_seconds=4.61 tlc_seconds=9.88 wall_seconds=15.18
  scope=sampled-instances exhaustive=no
```

No disagreement was found. All nine shared predicates took both values:
`allDone` 39/153, `completed` 571/320, `fullyPending` 103/323,
`opsAgreeAtFront` 89/14, `sameSite` 524/522, `startAllowed` 64/362, `stuck`
54/138, `uniformComms` 114/78, `uniformOps` 149/43. A run in which some
predicate was constant is reported as a `vacuity_problem` and fails, because
such a run would pass equally well with a broken comparator.

Replicated across draws, all zero disagreements and zero vacuity problems:
seed 7 at 192 instances (3491 comparisons), seed 1234 at 192 (3570), and seed
42 at 512 (9706 comparisons, 36.1 s wall). One seed would not distinguish
agreement from a lucky draw.

Cost is dominated by JVM startup: one TLC probe is about 0.8 s serially, so
192 probes are run 16-way concurrently for 9.9 s wall. Lean is batched 24
instances per generated file, about 4.6 s wall for all 192 plus a one-time
0.7 s compile of `DeviceMeshProtocol.olean`.

Concurrent TLC probes need a private `java.io.tmpdir` and a private
`-metadir`. tla2tools extracts `Naturals`, `Sequences` and `TLC` into
`java.io.tmpdir` and parses them back; sharing `/tmp` makes SANY fail with an
internal `NullPointerException` on a half-written standard module.

### The two exemptions, both divergence 3

Only divergence 3 can make a shared predicate legitimately differ, and it does
so in exactly two places. Each carve-out is named, carries its divergence
number, and carries a one-sided obligation, so it cannot decay into a silent
skip. The remaining seven predicates are compared with no carve-out at all.

- **E1, `completed_outside_issue_domain`.** TLA's `Completed(r, k)` reads
  `issued[r][k]`. Measured rather than assumed: out of domain TLC reports
  `Attempted to apply tuple <<...>> to integer 5 which is out of domain` and
  abandons the run. The probe domain is therefore `1..Len(issued[r])`. Lean is
  additionally probed at `Len+1` and `Len+3`, where divergence 3 claims
  `completedB` is `true`, and the comparator asserts exactly that. 1054
  records, 1054 obligations checked.
- **E2, `ops_agree_at_front_not_fully_pending`.** TLA's `OpsAgreeAtFront(c)`
  reads `OpAt(r, c, Front(c))`, whose `IdxOf` is a `CHOOSE` with no witness when
  `CommCount(r, c) < Front(c)`. Measured: TLC reports `Attempted to compute the
  value of an expression of form CHOOSE x \in S: P, but no element of S
  satisfied P` at `DeviceMeshModel.tla`'s `IdxOf`. The predicate is compared
  wherever `FullyPending(c)` holds -- the guard is evaluated by TLC from the
  model's own definition, not recomputed by the harness. Where every member is
  short of the front, both of Lean's `opAtOn` reads are `none` and the
  obligation is that Lean is `true`. 323 records, 278 obligations checked.
  `startAllowed(c)` and `stuck` stay compared on those instances, and they are
  where the guard is actually used: `StartAllowed`'s members-ready conjunct
  implies `FullyPending`, so the composite is total on both sides.

### Proof that the harness can fail

Six named perturbations, each injected into the generated module the checker
actually consumes, so a detection exercises emission, execution, parsing and
comparison. At seed 42 and 24 instances each is reported, naming instance and
predicate:

| perturbation | side | predicate | disagreements |
|---|---|---|---|
| `lean_same_site_negate` | Lean | `sameSite` | 139 |
| `lean_stuck_drop_start_conjunct` | Lean | `stuck` | 5 |
| `lean_ops_agree_negate` | Lean | `opsAgreeAtFront` | 48 |
| `lean_completed_out_of_domain_negate` | Lean | `completedOutOfDomain` | 126 |
| `tla_all_done_drop_running` | TLA+ | `allDone` | 2 |
| `tla_completed_off_by_one` | TLA+ | `completed` | 28 |

`lean_completed_out_of_domain_negate` is caught only by E1's one-sided
obligation, since TLA compares nothing there. That is the check that
distinguishes a carve-out from a skip.

### Scope

A passing run says the two models agreed on every shared predicate at every
probed argument, over the recorded instance count drawn from the recorded
seed. It is not exhaustive and raising the instance count does not make it so,
which is why the token carries `scope=sampled-instances exhaustive=no`.
Divergences 8, 9 and 10 concern `BlockedBy`, liveness and stuttering; none is
a shared boolean predicate, so none is in scope here.
