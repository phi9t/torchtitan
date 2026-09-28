# 13 — Close the bounded-checking gap

**What to build:** An inductive invariant for `DeviceMeshModel`, so its safety
claim holds for all schedule lengths rather than up to `MaxIssues`.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** folded into 14 — the route was decided and the work is done there

## The gap, stated plainly

TLC explores 38321 distinct states at `MaxIssues = 2` (the figure was 3671 when
this ticket was written, before the instance grew to eight communicators; see
`DeviceMeshModel.tla`'s claim inventory for the current numbers). That
establishes safety for schedules of at most two issues per rank. The observed
run has 108 per rank.

The composed claim previously shipped was: the guards are safe for all
schedules up to the bound, **and separately** the observed 108-length schedule
satisfies the per-communicator agreement property. Both halves are true. Their
conjunction is weaker than "the protocol is safe", and a casual reader will not
notice the difference. That is the honesty problem this ticket exists to fix.

## The route chosen, and why it is not Apalache or TLAPS

This ticket originally proposed Apalache (cheapest) or TLAPS (machine-checked).
Neither was used. The decision, recorded here so the ticket stops describing a
plan nobody is following:

- **No new toolchain.** Apalache and TLAPS are each another pinned artifact to
  vet, mirror, digest-corroborate and keep hermetic, for one result. The
  `MODULE.bazel` standard for a pin is stated against TLA+ 1.7.4 and Lean
  4.34.0 and it is deliberately strict; adding a third toolchain for a single
  obligation does not pay for itself.
- **Lean 4.34.0 is already pinned, and it can carry the argument.** The three
  obligations are ordinary theorems about an encoded transition system, and
  Lean proves them over ALL topologies and schedule lengths rather than over a
  bounded state space. That is strictly more than Apalache's `--length=1`
  inductiveness check would give, because it is not tied to one instance.
- **The cost is an encoding-fidelity claim**, which Apalache and TLAPS would not
  incur since they read the TLA+ module itself. That cost is paid explicitly:
  the correspondence is written out definition by definition in
  `DeviceMeshProtocol.lean`'s header, the divergences are named with the
  direction each moves the claim, and three of them are discharged as Lean
  lemmas rather than left to inspection. A reviewer must check the encoding;
  that is the price of the route and it is stated rather than hidden.

## Where the work landed

All of it is in ticket 14, which now carries this ticket's acceptance criteria
as well as its own. In outline:

- `experiments/qwen3_formal_verifier/formal/DeviceMeshProtocol.lean` — the
  encoding, the correspondence table, the named divergences.
- `experiments/qwen3_formal_verifier/formal/DeviceMeshInductiveInvariant.lean` —
  `initiation`, `consecution`, `sufficiency`, each its own theorem and its own
  result token, plus `safetyOfReachable`, whose statement mentions no bound.
- `experiments/qwen3_formal_verifier/formal/DeviceMeshWaitGraph.lean` — the
  conditional treatment of `DeadlockFreedom`, which is the one safety property
  that was NOT made unconditional; see 14 for what was and was not reached.
- `experiments/qwen3_formal_verifier/formal/run_lean_device_mesh_protocol.sh`
  and the `lean_device_mesh_protocol_test` target, in both `tier0_formal_tests`
  and `device_mesh_formal_tests`. CPU-only, no GPU run, no attempt bundle.

## Acceptance criteria

All of these now live in 14 and are checked there. Kept here only so the
mapping is legible.

- [x] `Inv` is stated explicitly and separately from the reachability-checked
  invariants. `DeviceMeshProtocol.lean` defines `Inv`; TLC's configurations are
  untouched, so which claim rests on which method is visible.
- [x] Initiation, consecution and sufficiency are each checked and each is
  reported as its own result token (`DEVICE_MESH_PROTOCOL_LEAN_THEOREM ...
  role=obligation-1-initiation` and so on).
- [x] The bound `MaxIssues` no longer appears in the safety claim's statement:
  `safetyOfReachable` does not mention `Topology.maxIssues`.
- [x] The check needs no fresh GPU run: the modules read no facts module, and
  the target is in the trace-free `tier0_formal_tests` suite.
- [x] Recorded as a finding where an invariant could NOT be made inductive:
  `DeadlockFreedom` is proved only under an acyclicity hypothesis and
  `StuckImpliesAllDone` is a statement about the bound and cannot be made
  bound-free. Both are stated in `DeviceMeshInductiveInvariant.lean`'s header
  and in 14, not left as a silent gap.
- [x] No new toolchain was pinned, so the `MODULE.bazel` pinning obligation
  does not arise. No Mathlib either: the proofs use core Lean only, which is
  why `#print axioms` stays empty.

## Note

This was the single largest gap between what the artifacts state and what a
reader will assume they state. It is closed for the three state-predicate
safety invariants and explicitly not closed for deadlock freedom.

### Gate evidence

Full nine-stage Device-mesh gate, all stages exit 0, sealed and verified, after
the four fidelity-review findings were fixed.

- Device-mesh evidence ID:
  `sha256:c370f7163b86b8d90916d37f93fb12b15517d5d859a1fbd3c7cc41d414bbca0f`
- Device-mesh source ID:
  `sha256:2e84b97e26aace9081ff25d2c8d550f0964467a488c95c098bd8287359c66e08`
- Nested Single-rank evidence ID:
  `sha256:c331886cd5d27f4aa6904740e7efdfd95d1cd45ebf363319ec6fc7ffa554eea4`

The vocabulary split is visible in the sealed log, which was the point of the
relabelling:

```
consecution                        kind=theorem  scope=all-topologies-all-schedules bound=none
blockerOfStuckPending              kind=theorem  scope=all-topologies-all-schedules bound=none
cyclicWitnessIsStuck               kind=witness  scope=fixed-instance bound=2
cyclicWitnessHasTwoCycle           kind=witness  scope=fixed-instance bound=2
commFifoAloneIsNotInductive        kind=witness  scope=fixed-instance bound=1
acyclicityIsLoadBearing   kind=theorem-negative  scope=fixed-instance bound=2
```

A reader of the log can now tell a theorem over all topologies and schedules
from a `decide` over one fixed topology with a declared bound. Before the fix
all of these carried `bound=none`, and a test asserted that they must.
