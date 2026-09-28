# 16 — Liveness under fairness

**What to build:** A temporal liveness property for `DeviceMeshModel`: every
issued collective eventually completes, under weak fairness.

**Blocked by:** 11 — DPxTP collective protocol model.

**Status:** resolved -- the liveness property in `DeviceMeshModel.tla` landed in
`87d341c89`. This line read `ready-for-agent` until a status audit corrected it;
the work had been committed for some time and the ledger was advertising it as
available.

## Why safety is not enough here

`DeadlockFreedom` is a safety invariant. Since ticket 21 it says: no reachable
state is stuck with an admissible rendezvous that every member has pending and
that can never start. That rules out a terminal deadlock, but not a livelock in
which the system keeps taking steps while some collective never starts. For a
step model whose whole purpose is to reason about jobs that hang, "the job
always has something to do" is not the property anyone cares about; "the step
finishes" is.

## Shape

`Spec == Init /\ [][Next]_vars /\ WF_vars(...)` with fairness on `Start` and
`Complete`, then

```tla
EveryIssuedCollectiveCompletes == \A c \in CommIds : ...~> doneOn[c] = ...
```

`Issue` must **not** be fair -- it is unconstrained by design, and that is the
source of rank skew. Fair `Issue` would let a rank run ahead forever and the
property would be about an artifact of the fairness assumption.

## Acceptance criteria

- [ ] The liveness property is checked as its own run, reported with its own
      result token. Liveness checking is much slower than safety, so it likely
      needs a smaller bound than the safety runs -- report the bound actually
      achieved rather than implying the safety bound carried over.
- [ ] Fairness is on `Start`/`Complete` only, with the reason for excluding
  `Issue` in the module header.
- [ ] A negative: with `RequireUniformCommSchedule = FALSE`, TLC produces a
      counterexample trace to liveness. Liveness counterexamples are lassos,
      so `formal_classify_tlc_transition_negative` -- which expects exactly
      one `^Error:` line and a state-space summary -- almost certainly does
      not classify this correctly. Verify the real TLC output shape and add a
      classifier for it; **do not** hand-author the fixture.
- [ ] That last point is the standing lesson from this project: a check
      verified against hand-authored fixtures confirms the author's
      assumptions instead of testing the system. Fixtures come from the tool
      whose output they imitate.

## Outcome

**Status:** done at `DeviceMeshModelLive.cfg`,
`DeviceMeshModelLiveDivergent.cfg` and `DeviceMeshModelLiveUnconditional.cfg`,
checked by `run_tlc_device_mesh_model.sh` stages 8-10.

### Two corrections to this ticket

**The constant it names no longer exists.** `RequireUniformCommSchedule` was
replaced in `4f252960d` by `RequireUniformProgramOps` (operations agree at each
position) and `RequireUniformProgramComms` (the communicator SITE agrees
wherever two ranks' communicators share a member); `MaxSkew` was dropped
entirely. The liveness negative relaxes **`RequireUniformProgramComms`**, which
is the successor of the constant this ticket meant: it is the relaxation that
leaves operations agreeing and NCCL's own guard on, so the counterexample is a
stream-ordered wait cycle rather than an operation mismatch. Relaxing
`RequireUniformProgramOps` instead would also refute liveness, but for the
op-mismatch reason that `DEVICE_MESH_MODEL_OPMISMATCH` already owns.

**The property as written in this ticket is false**, and not because of a hang.
`Issue` is unfair by design, so a behaviour may simply stop issuing. A
collective one rank has issued and its peers have not never becomes
`FullyPending`, so `Start` is never enabled, so weak fairness on `Start`
promises nothing and the issue never completes. Measured, with every guard on:
TLC refutes the unconditional reading in a four-state prefix -- ranks 0 and 1
issue, ranks 2 and 3 never do, and the lasso closes by stuttering -- at exit 13
in 4s. That refutation is kept as `DeviceMeshModelLiveUnconditional.cfg` and
`DEVICE_MESH_MODEL_LIVENESS_UNCONDITIONAL`, so the narrowing below is a
measurement rather than a convenience.

### The property that shipped

```tla
EveryIssuedCollectiveCompletes ==
  (AllIssued /\ EveryRendezvousPopulated) ~> AllIssuesCompleted
```

`AllIssued` is what makes it true: every rank has issued its whole program, so
nothing is waiting on a rank that merely stopped. `EveryRendezvousPopulated` --
no communicator that some member is already waiting in is short of another
member's issue -- is there for the NEGATIVE, not the positive. Without it the
negative's first counterexample is the finite-budget artifact the module
documents as `StuckByBudget`: measured, rank 0 `<<batch02, loss02>>` against
rank 2 `<<loss02, fsdp02>>`, where `batch02` is short of rank 2's issue. With
it, the measured counterexample is a genuine four-communicator stream-ordered
circular wait at full budget: rank 0 `<<tp01, loss02>>`, rank 1 `<<fsdp13,
tp01>>`, rank 2 `<<loss02, tp23>>`, rank 3 `<<tp23, fsdp13>>`.

That conjunct costs the positive nothing, which is checked rather than argued:
the positive configuration also checks the invariant
`FullBudgetRendezvousPopulated == AllIssued => EveryRendezvousPopulated`, so at
full budget under one SPMD program the two antecedents coincide.

Fairness is `\A c \in CommIds : WF_vars(Start(c)) /\ WF_vars(Complete(c))` --
per communicator, so a behaviour cannot satisfy it by servicing one
communicator forever and another never. `Issue` has no fairness. Weak fairness
suffices because `Start(c)` and `Complete(c)`, once enabled, stay enabled:
`Issue` only appends, and `AtStreamHead(r, k)` quantifies over `1..(k - 1)`.

### What TLC actually prints, and the classifier

A liveness counterexample is a lasso, and it is a fourth TLC outcome:

```text
Error: Temporal properties were violated.

Error: The following behavior constitutes a counter-example:
State 1: <Initial predicate>
...
State 10: Stuttering
1599947 states generated, 401719 distinct states found, 78116 states left on queue.
Finished in 02min 05s
```

Exit **13**, two `^Error:` lines, no `Invariant X is violated` line at all, and
a summary reporting states still on the queue. So
`formal_classify_tlc_transition_negative` rejects it on the exactly-one-Error
rule and on the missing invariant line, `formal_classify_tlc_negative` and
`formal_classify_tlc_constant_false` reject it on exit status, and
`formal_classify_tlc_valid` rejects it outright. All four rejections are
asserted, against this real output, in
`test_tlc_liveness_negative_classifier_matches_the_real_checker_output`.

`formal_classify_tlc_liveness_negative` accepts exit 13 with both Error lines,
no other Error line, a search summary, `Finished in`, and a closed lasso --
either `State N: Stuttering` or `Back to state N`. The `Back to state` shape
comes from a real TLC run on a throwaway flip-flop, because this model's
variables only grow and so cannot produce a cyclic lasso; it was not authored.

TLC's liveness diagnostic names no property, so the classifier cannot check one.
`formal_cfg_declares_one_property` pins each stage to its cfg instead, and
`DEVICE_MESH_MODEL_CONFIG_COVERAGE` reconciles shipped configurations against
the ones a stage actually ran.

### The bound, and the cost

The liveness positive is exhaustive at the **same** bound as safety,
`MaxIssues = 2` on 4 ranks and 8 communicators: 133881 states generated, 38321
distinct, 0 left on queue, 23s. A smaller bound was not needed, and a smaller
one would not have worked for the negative anyway -- a cross-communicator cycle
needs two issues per rank.

The negative is a partial search by construction: TLC reports from a periodic
liveness check, so it stops with states on the queue (measured 397195 and 401719
distinct states on two runs, 78116 and 79813 left on queue, about 2min 05s
each). Its state count is not reproducible run to run, because the periodic
check's timing moves with TLC's fingerprint seed; nothing asserts the number.
The unconditional refutation costs 4s. The liveness stages run under a 420s
timeout rather than the safety stages' 120s, and the whole Device-mesh model
target went from about 2min to 4min 31s.

No CONSTRAINT in the liveness configurations: TLC warns that a state constraint
under liveness checking is unsound, because a pruned state becomes a terminal
node and can manufacture a stuttering lasso. `ModelBounded` would prune nothing
here, since `IssueAllowed` already enforces the bound, so it buys nothing.

### Acceptance criteria

- [x] Checked as its own run, with its own result tokens, reporting the bound
      actually achieved (`bound_max_issues_per_rank=2`) and, for the negative,
      that the search was partial.
- [x] Fairness on `Start`/`Complete` only, per communicator, with the reason for
      excluding `Issue` in the module header.
- [x] A negative with `RequireUniformProgramComms = FALSE` -- the successor of
      the constant this ticket named -- producing a lasso counterexample, with a
      classifier written against the real TLC output and asserted to be rejected
      by all four existing classifiers.
- [x] No hand-authored fixtures: both lasso shapes in the test come from real
      TLC runs, and the ticket's own property was refuted by a run before it was
      narrowed.

### One check beyond the ticket: the fairness is load-bearing

A liveness positive that would also pass without fairness would prove nothing
about `WF`. So the runner reruns the *same* `DeviceMeshModelLive.cfg` with
`SPECIFICATION LiveSpec` substituted to `SPECIFICATION Spec` -- the same pattern
as the `StreamOfIssue(e) == e.comm` mutant -- and requires it to FAIL. Measured:
exit 13, exhaustive over the same 38321 states, 0 left on queue, 13s, lasso
closed by stuttering at state 17. Reported as
`DEVICE_MESH_MODEL_LIVENESS_UNFAIR`. The substitution is asserted to have
applied, so the control cannot silently rerun the fair specification.

## Review round 2026-09-26 (independent, clean context): SOUND WITH CAVEATS

**The antecedent is reachable and the property is not vacuous.** This was the
question that decided whether the phase means anything, because strengthening
a `~>` antecedent weakens the property. Three negated predicates were checked
as invariants under the positive cfg's constants and all three were violated
at depth 9: `AllIssued` is reachable, `AllIssued /\ EveryRendezvousPopulated`
is reachable, and -- the one that matters -- there is a reachable state where
the antecedent holds and `AllIssuesCompleted` does not, with `doneOn` all zero
and eight collectives outstanding. So the property is not discharged by
antecedent-implies-consequent-in-place. `FullBudgetRendezvousPopulated` was
confirmed non-circular.

Also confirmed: fairness is on `Start`/`Complete` only and `Issue` carries
none anywhere; the stability argument licensing weak rather than strong
fairness holds against the source; the unfair control cannot silently pass
without substituting; the negative's counterexample is a genuine
four-communicator stream-ordered wait cycle at full budget with every
rendezvous populated; and dropping the second antecedent conjunct does produce
the `StuckByBudget` shape instead, as claimed.

The `Back to state` fixture was verified genuinely TLC-produced, by
reconstructing a flip-flop from the column geometry encoded in the fixture and
obtaining byte-identical output including state counts. This project's
signature defect -- a guard validated against hand-authored output -- is
absent here.

### The substantive caveat, now recorded in the module

`issued` and `doneOn` are monotone non-decreasing and `Complete` is the only
action that removes from `running`, so the reachable state graph has no cycles
beyond the `Terminated` self-loop: **there is no livelock in this model to
exclude**. And because the antecedent contains `AllIssued`, `Issue` -- the
unfair action -- is disabled everywhere downstream of it, leaving only the two
weakly fair actions. So under `LiveSpec` the property reduces to "every
successor-less state reachable from an antecedent state satisfies
`AllIssuesCompleted`", which is close to what `StuckImpliesAllDone` already
says over the same 38321 states.

That is not unsound and the property is not trivial: the unfair control proves
the fairness is required to discharge it, and the divergent negative proves it
discriminates. But the general claim that a safety invariant cannot express
"the step finishes" carries little weight *in this particular model*, and the
header said so without saying that. It now says both.

### Findings fixed

- **Coverage ledger keyed on the cfg name at invocation.** Two stages run
  mutated copies under a shipped cfg's name, so a mutant satisfied coverage
  for the cfg it mutates -- demonstrated by deleting the honest `live` stage
  and still getting `shipped=10 checked=10 unchecked=0`. The ledger now
  records only runs whose cfg **and** module are byte-identical to what ships,
  and marks the rest as mutated. Both mutation shapes are covered: one
  rewrites the cfg, the other the module.
- **The stated rejection mechanism was wrong.**
  `formal_classify_tlc_transition_negative` rejects a lasso on the exit-status
  pin, returning before it reaches the one-`Error:` rule or the
  missing-invariant rule. The conclusion (all four classifiers reject) is true
  and asserted against real output; only the explanation was wrong.
- **Counterexample identities are seed-dependent, not just state counts.** The
  specific traces quoted in the module and this ticket are not reproducible --
  the reviewer's run gave a different four-communicator cycle from the same
  cfg, and a different pair of idle ranks in the unconditional refutation. The
  class is stable and is what the argument rests on; the disclosure now covers
  trace identity as well as counts.
- **`formal_cfg_temporal_properties` omitted `CONSTRAINTS`** from its keyword
  set, so a cfg with `PROPERTY` followed by a `CONSTRAINTS` block would parse
  the block's contents as property names. Fail-closed, but fixed.

### Left as a follow-on

- [ ] The new runner logic -- stages 8 to 11, the per-stage timeouts, the four
  new tokens and the coverage reconciliation -- is asserted only by the Bazel
  `sh_test`, which is `manual`-tagged. That is the right trade against
  string-grepping a shell script, but the reconciliation in particular
  deserves a direct shell-level test of the kind the review improvised.

### Gate evidence

Full nine-stage Device-mesh gate, all stages exit 0, sealed and verified, after
the four review findings were fixed.

- Device-mesh evidence ID:
  `sha256:7d19a183856c4e6b137a8ee8bbf0ed088638310971044d83a603b288bb8158f5`
- Device-mesh source ID:
  `sha256:a6bc93e38480580ae69e6e5843bd02850f6aaf577904f537b47b6578d93e199b`
- Nested Single-rank evidence ID:
  `sha256:99617d7f27add67d41b666c9b9e43b0948a733d401a715843ed00354bdd15d1d`

Tokens in the sealed log:

```
DEVICE_MESH_MODEL_LIVENESS        result=holds
                              fairness=start_complete_per_communicator
                              issue_fairness=none distinct_states=38321
                              states_left_on_queue=0
                              bound_max_issues_per_rank=2
DEVICE_MESH_MODEL_LIVENESS_NEGATIVE  result=lasso_counterexample search=partial
                              distinct_states=409298
                              states_left_on_queue=74455
DEVICE_MESH_MODEL_LIVENESS_UNFAIR substitution=LiveSpec_is_Spec
                              result=lasso_counterexample
DEVICE_MESH_MODEL_LIVENESS_UNCONDITIONAL cause=unfair_issue
                              result=lasso_counterexample
DEVICE_MESH_MODEL_CONFIG_COVERAGE shipped=10 checked=10 unchecked=0
                              mutated_runs=2
```

The positive reports `states_left_on_queue=0`, so it is exhaustive at its bound;
the negative reports `search=partial` with its queue depth exposed rather than
presenting a periodic liveness check as a complete search.

The coverage fix was verified against the mutation it exists to catch: with only
the mutated run recorded, `DeviceMeshModelLive.cfg` reads UNCHECKED and the
reconciliation exits 1. Before the fix that case reported
`shipped=10 checked=10 unchecked=0`. The check was widened past what the review
reported, because one mutant stage rewrites the cfg and the other rewrites the
module, so comparing only the cfg would have closed half the hole while looking
closed.
