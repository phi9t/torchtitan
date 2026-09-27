# 10 — Abstract single-rank step model and refinement bridge

**What to build:** A TLA+ transition system for one single-rank training step
whose invariants hold over every admitted interleaving, plus a refinement check
proving the observed Scout A trace is a behaviour of that model.

**Blocked by:** None for the single-rank scope delivered here. Extending the
same shape to the 2x2 DPxTP topology is blocked by 03.

**Status:** review-pending

## Why this exists outside the planned sequence

The spec routes model deepening through Phase 2 (tickets 05, 06, 07), which
cannot begin until Scout B passes its full gate and a human approves the Phase 2
design checkpoint required by ticket 04. This work was directed explicitly by
the user ahead of that gate, after an assessment of what the formal layer
actually proved. It is recorded here rather than folded into ticket 03 so the
deviation is visible, and it does **not** close 05, 06, or 07, whose scope is
broader.

## The assessment that prompted it

The pre-existing formal layer checks the observed trace and nothing else:

- `ScoutAValid` walks a cursor along a constant sequence of generated facts;
  TLC explores one path, 12 states, one per observed event.
- `ScoutBValid` is a two-state stutter whose `cursor` variable appears in none
  of its invariants; every Scout B invariant is a predicate over constants.
- All 19 Lean theorems have the shape
  `someDecidableBool <literal data> = true`, closed by `rfl` (18) or `decide`
  (1). They are kernel-checked evaluation certificates over specific data, not
  quantified statements about executions.

That layer has real value -- an independent re-implementation of the predicates
in two languages, kernel-checked integrity, and negative controls that show the
predicates can fail. What it cannot do is say anything about an execution that
was not observed, which is the thing "formal verification" is usually taken to
mean.

## Acceptance criteria

- [x] The model is a transition system with guarded actions over real state,
  and does not import the generated facts.
- [x] It admits both orders of gradient readiness versus backward completion,
  and the branch in which gradients never materialise.
- [x] TLC explores a non-degenerate state space and the runner fails rather
  than reporting success on a trivial one. Observed: 25 distinct states,
  maximum outdegree 3, search depth 11.
- [x] Safety invariants hold across every admitted interleaving, covering
  mutation-requires-gradients, gradient resolution uniqueness, forward/backward
  bracketing, and terminal step completion.
- [x] The one load-bearing guard is a TLA+ constant, so the negative model
  relaxes exactly it. TLC discovers the counterexample
  `gradient.missing -> optimizer.started -> optimizer.mutated` by exploration.
- [x] The observed Scout A trace is proved to be an admitted behaviour of the
  model, with TLC exhibiting the action-to-event alignment.
- [x] A controlled corruption of that trace is refused, and refused for the
  right reason. The corruption substitutes `gradient.missing` for
  `gradient.ready` and changes nothing else, so every earlier event stays
  admissible and the trace dies exactly at `OptimizerMutate`. Two checks prove
  this: the trace is not admitted, and it WAS admitted up to the mutation.
  An earlier version swapped `gradient.ready` with `optimizer.mutated`, which
  was refused for an out-of-order lifecycle instead -- a different guard than
  the one under test. Independent review caught that; it looked correct.
- [x] Refinement polarity is unambiguous: rejection completes cleanly rather
  than surfacing as a deadlock that reads like a specification defect.
- [x] Wired into both the Scout A and Scout B Bazel formal suites, so the
  bridge is gate evidence rather than a side experiment.
- [x] Focused tests cover the model's shape, the guard isolation, the
  refinement configuration, the degenerate-state-space refusal, and the runner
  errexit regression.
- [x] Full Scout B gate from the final state: all nine stages exit 0, with the
  four model checks reported from inside the gate's own formal stages.
- [ ] Separate Standards and Spec reviews by a fresh clean-context reviewer.

## Evidence

Formal suite, `--networked --suite scout-a`, 5 of 5 targets pass:

```text
SCOUT_A_MODEL_TOOLCHAIN checker=tlc release=1.7.4 java_major=17
SCOUT_A_MODEL_SAFETY result=success distinct_states=25 max_outdegree=3 exit=0
SCOUT_A_MODEL_NEGATIVE invariant=MutationRequiresReadyGradients result=named_violation exit=12
SCOUT_A_REFINEMENT result=admitted witness=ObservedTraceIsNotAdmitted exit=12
SCOUT_A_REFINEMENT_NEGATIVE result=rejected_at_mutation_guard reject_exit=0 reach_exit=12
```

## Gate evidence

Complete Scout B gate at HEAD `366737c37478d2253491742a74179febcf4ea54f`, all
nine stages exit 0:

- Scout B evidence ID:
  `sha256:b95e3f3e74e6874eeeef47aaa70005b26328eacc5f5762b68ea096c7d05fa266`
- Nested Scout A evidence ID:
  `sha256:afa04eb38c64d7218abadd5dc32ff43171aae8981a0934f06d4de783823b50c4`
- Shared source ID:
  `sha256:63b39450b548fb39b13bafb062e2214350d3abb21f56926476da31bebdf8dab1`
  (87 verified entries, 12 process documents)

## Test quality: mutation-checked, not asserted

These tests read file contents, which is the shape that can decay into
restating the implementation. Both load-bearing tests were therefore checked by
mutating the source and confirming they fail:

- reintroducing the `set -e` inside the runner's `run_tlc` helper fails
  `test_model_runner_does_not_clobber_errexit_around_expected_failures`;
- narrowing `GradientReady` from `bwd \in {"started", "completed"}` to
  `bwd = "started"`, which removes the interleaving the safety argument depends
  on, fails `test_gradient_readiness_and_backward_completion_may_interleave`.

Both mutations were reverted byte-identically, confirmed by both sealed bundles
still verifying afterwards.

## Bounded claim

This establishes safety properties of a bounded abstract model of one
single-rank step, and that one observed run lies inside that model.

The bridge is over event **kinds** only: `ScoutARefine` projects the observed
trace to `EventKinds`, so identities, phases, predecessors, provenance and
projection digests are outside what the model admits. Those remain the job of
the Python validators and the fact-module checkers, and this bridge does not
subsume them. It does not model the 2x2 DPxTP topology, collectives, multi-step
training, convergence, or performance. The abstract model is not claimed to be a faithful model of
TorchTitan; it is a model of the step lifecycle contract stated in its own
actions, and its value depends on that contract being the right one.

## Incident: checker products sealed as source

An ad-hoc TLA+ compatibility probe run directly inside
`experiments/qwen3_formal_verifier/formal/` left TLC state products
(`states/<timestamp>/*.st`, `*.fp`) in the checkout, because TLC writes them
beside the checked module when no metadir is given. The following gate captured
those four files as **verified source** in its sealed manifest. The spec keeps
generated state spaces out of the checkout, and nothing in the gate refused
them.

The files were removed and `build_source_manifest` now fails closed on any
`.st`/`.fp` path or any path under a `states/` directory. Failing closed rather
than ignoring is deliberate: gitignoring them would let the pollution persist
invisibly, which is the condition that produced the defect.

Found by independent review, not by the gate -- which is the point of having
one, but also a gap the gate should not have had.

## Follow-on

The same shape should be ported to the 2x2 DPxTP model, where the interesting
concurrency actually lives: collective enqueue/start/complete lifecycles across
four ranks, and cross-rank synchronisation edges. That port belongs with
ticket 06, and the refinement machinery proved out here is the reusable part.

### Review round 2026-09-26 (independent, clean context): PASS WITH FINDINGS

Nothing blocking. The four areas flagged as most likely to be wrong -- genuine
transition system, vacuity, refinement polarity, and "refused at the right
guard" -- all survived direct experiment with real TLC, not reading:

- The branching floor is real end-to-end. A deliberately linearized model was
  run through the actual runner and refused: `abstract model never branches
  (max outdegree 1) ... is a replay, not a model`, exit 1.
- Eleven reachability probes found no invariant with an unreachable antecedent.
- Polarity is safe in both failure directions: an unadmitted trace and an empty
  trace each produce runner exit 1, not a false positive.
- The negative control is genuinely fixed. Flipping `RequireReadyGradients` to
  FALSE admits the corrupted trace in full, which proves the gradient guard is
  the sole reason for refusal. `RejectionHappensBeforeTheMutation` does real
  work and is not redundant.

Eight non-blocking findings are tracked in ticket 22. Two matter before ticket
12: the unguarded `IndexOf` (whose duplicate case becomes reachable in the DPxTP
port) and the missing TLC evaluation-error class in the classifiers.

Status stays `review-pending` for one reason only: the acceptance criterion
requiring a full nine-stage gate from the final state was out of the reviewer's
scope, and that evidence is stale for the same reason as ticket 03's. It is
deferred to the end of the formalization sequence.

### 2026-09-27: the final-state criterion is unsatisfiable, not merely unmet

The final re-seal was attempted from the clean committed tree at `d8b1016d5`,
after every other phase closed. It **fails**, in the nested Scout A gate's
`lint` stage, whose log is ten lines with no hook run.

The manifest from a clean tree has zero verified and zero process entries, so
`source_manifest_lint_paths` yields nothing and the lint stage's "at least one
file" assertion fails. Tracked as ticket 25.

So this ticket's criterion -- re-execute the supported commands from the final
worktree state -- cannot be satisfied as the gate is written, because the final
state is a committed state, a committed state is a clean tree, and a clean tree
has no lint paths. Every bundle this project has sealed came from a dirty tree.

This does not mean the work was ungated. Each phase ran a real nine-stage gate
whose manifest pinned the uncommitted bytes under review, and those bundles
verify. What is absent is the final seal from the committed state, which is a
narrower claim than "never gated" and should be read as such. This ticket stays
open until ticket 25 lands.
