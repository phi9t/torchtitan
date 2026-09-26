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

### Review round 2026-09-26 (independent, clean context): FAIL

Three blocking findings, all produced by running TLC rather than reading, and
the first independently reconfirmed:

1. **"Streams are load-bearing" is false.** Neutralising the stream-head
   conjunct leaves the state graph bit-identical -- `12142 generated, 3671
   distinct`, max outdegree 12 for safety; `124620 / 35915 / 18308` with
   `DeadlockFreedom` violated for the divergent cfg. `AtStreamHead` is
   redundant by construction in the only instance any cfg binds.
2. **The cross-communicator wait cycle is in no checked configuration.** The
   wide instance that would contain it is dead code, guarded only by a test that
   counts substrings in text no checker loads.
3. **The divergent negative is refuted by an intra-communicator operation
   mismatch** -- the very divergence this ticket calls unrealizable. Order
   divergence cannot deadlock the model at all.

The machinery reproduces exactly and all eight formal targets pass. What fails
is the correspondence between the artifact and its claims: the model does not
express the hazard it was built to express, and the module header, this ticket's
criteria, the runner comment and the commit message each assert otherwise.

The repair is tracked as ticket 21, which blocks ticket 12 -- a refinement
bridge onto this model would otherwise attach to a deadlock property that is
vacuous in the direction that matters.

Also recorded there: `MaxSkew` is inert at `MaxIssues = 2`; the `CommClass`
conjunct is provably inert; four of the five "safety invariants" are
restatements of `Start`'s guard; the bound appears nowhere a reader meets the
result; and the SYMMETRY rationale is wrong for the reason given, though right
in its conclusion.

What the review confirmed sound, so it is not re-litigated: `Stuck` does not
drift from `Next` (checked by equivalence against an `ENABLED` formulation);
non-vacuity is genuinely sensitive (forcing `StartAllowed` false makes the stage
fail); the negative classifier is tight; the issue-order negative violates the
right invariant for the right reason; `RuntimePgIdAgreesWithinRank` is not
circular; and the model does branch, with `max_outdegree 12` being TLC's own
number.

One correction to the record: the criteria span two commits. `ScoutDistributed`
(both `.tla` and `.lean`), `scout_b.py` and `checker_contract.sh` are in
`25315472c`; `fee966fea` holds `ScoutBModel*`, `ScoutBIssueOrderInvalid*` and
the runner, BUILD and test changes.
