# 11 — DPxTP collective protocol model

**What to build:** A TLA+ transition system for collective execution on the 2x2
DP-shard x TP mesh, checking the property whose violation hangs real jobs, plus
the same property checked against the observed run.

**Blocked by:** 03 — Scout B: 2x2 DPxTP tracer bullet (for the observed facts).

**Status:** review-pending

## The property

Per communicator, every member rank must issue the same collectives in the same
order. This is an NCCL requirement. Ordering *across* communicators is
unconstrained, but an order that creates circular wait across communicators
hangs just as surely.

The hazard is concrete here rather than theoretical: `mesh_batch`, `mesh_fsdp`
and `mesh_loss_mesh` are three separate communicators spanning the same rank set
`{0,2}`, and the observed issue order shows fsdp and tp collectives interleaving
densely, so ranks 0 and 2 sit on a genuine four-cycle wait-for graph.

## Acceptance criteria

- [x] Communicators are keyed on the global `canonical_id`. The observed
  `runtime_pg_id` is per-rank local and was measured to map to two distinct
  communicators each, so keying on it would silently merge them.
- [x] Operation and axis are first-class values, not substrings of identifiers.
- [x] Issue order is taken from the `collective.enqueued` event, which inherits
  the Flight Recorder record sequence. The `started`/`completed` sub-order is
  appended post hoc and is never used as ordering evidence.
- [x] `PerCommunicatorIssueOrderAgreement` holds on the real 108-collective run,
  and flipping exactly one operation makes it false.
- [x] `ScoutBModel` is a transition system with concurrent per-rank issue, not a
  replay: 3,671 distinct states at maximum outdegree 12.
- [x] Streams are modelled, because without them the model has no deadlock at
  all -- the wait edge exists only because a CUDA stream executes in issue order.
- [x] Five safety invariants hold across every admitted interleaving.
- [x] Non-vacuity is checked by refutation: the model reaches completion, so the
  safety invariants are not true merely because a guard is unsatisfiable.
- [x] A divergent program is discovered to deadlock, by exploration.
- [x] Without NCCL's matching guard, the same divergence instead runs a
  mismatched rendezvous -- showing order agreement is necessary, not sufficient.
- [ ] Refinement bridge replaying the observed issue order through the model's
  guards (planned P4), and its negative (P5).
- [ ] Full Scout B gate from the final state, then separate Standards and Spec
  reviews by a fresh clean-context reviewer.

## A modelling error worth recording

The first version constrained ranks to agree on the communicator *class* at each
position but not the *operation*. TLC immediately found a counterexample in
which two members of one communicator issued `all_gather` and `reduce_scatter`
at the same position: that communicator can never start, and the run hangs.

The model was wrong, not the protocol. A single program cannot diverge that way,
and NCCL requires that it does not. The fix was to constrain the (class,
operation) pair while deliberately leaving the communicator *instance*
unconstrained, since which instance a rank uses is fixed by its mesh coordinate
-- rank 0 uses `fsdp02` where rank 1 uses `fsdp13`.

This is the model doing its job on its first run: an invariant that could not be
satisfied would otherwise have sat in the suite looking green.

## Gate evidence

Complete Scout B gate, all nine stages exit 0:

```text
SCOUT_B_MODEL_SAFETY      result=success distinct_states=3671 max_outdegree=12
SCOUT_B_MODEL_NONVACUOUS  result=completion_reachable
SCOUT_B_MODEL_DIVERGENT   invariant=DeadlockFreedom result=named_violation
SCOUT_B_MODEL_UNGUARDED   invariant=RendezvousOpAgreement result=named_violation
SCOUT_B_TLA_ISSUE_ORDER_NEGATIVE invariant=ScoutBPerCommunicatorIssueOrder
```

- Scout B  `sha256:084426e69dff3ede0194d3e4e78465158bc7ae04eca9f4635144a51e936368f1`
- Scout A  `sha256:c4222e595de7dd2a002c46356be8a29813538da4837a7ef0f956578fcbc74a2d`
- source   `sha256:d2786828e12f6906718f8dea004ff3a96a1ac29e7cfc0b917ae670bf154f5b06`

## What committing does to the evidence contract

This attempt records 13 verified source entries, where pre-commit attempts
recorded 87. That is not a regression: the source manifest is built from
`git status`, so once work is committed it is clean and falls out of the
manifest entirely.

The consequence is worth stating because it changes what a sealed bundle means.
Before a commit, a bundle attests to the whole uncommitted experiment. After,
it attests only to the remaining delta, and the committed majority is covered by
Git history instead. Neither is wrong, but a reader comparing entry counts
across attempts will otherwise conclude that coverage collapsed.

## Bounded claim

Safety is established for schedules up to `MaxIssues` (currently 2) on four
ranks and four communicators. That is not a claim about the observed 108. The
composed claim is: the guards are safe for all schedules up to that bound, and
separately the observed run satisfies the same per-communicator agreement
property. Closing the gap properly needs an inductive invariant (Apalache or
TLAPS), which is not attempted here.

The model is not a faithful model of TorchTitan. It models the collective
rendezvous and stream-ordering contract stated in its own actions, and its value
depends on that contract being the right one.

Stream identity in the observed facts is inferred, not observed: Flight Recorder
entries are zipped positionally to Kineto kernels. The model's stream mapping is
a design constant so the model is unaffected, but no claim should rest on
observed stream identity.
