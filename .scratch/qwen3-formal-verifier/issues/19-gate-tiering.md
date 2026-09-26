# 19 — Gate tiering

**What to build:** Split the nine-stage gate so the model-only checks run
without a GPU, and only trace-dependent checks need a fresh run.

**Blocked by:** none.

**Status:** ready-for-agent

## The problem

All nine stages -- `source_manifest, focused_pytest, cuda_pytest,
owning_pytest, artifact_sync, scout_a_regression, formal_networked,
formal_no_fetch, lint` -- run together, so every change pays for a GPU run and
the full sealing path. But most of the formal work does not depend on the
trace at all: `ScoutAModel`, `ScoutBModel`, their negatives, the non-vacuity
refutations and (ticket 13) the inductive invariant are properties of the
models. They can be checked on CPU in seconds.

Iterating on a model currently costs a full gate. That is the main tax on
doing more formal work, so paying it down accelerates every other ticket here.

## Shape

- **Tier 0, CPU, always-on:** model checks, model negatives, non-vacuity
  refutations, inductive invariant, Lean, lint, source manifest. No GPU, no
  trace. Minutes.
- **Tier 1, trace-dependent:** facts checks, refinement bridges, artifact
  sync, Scout A regression. Needs the sealed bundle, not necessarily a fresh
  run -- the canonical `four-rank-dp2-tp2-v1` trace is reusable.
- **Tier 2, GPU:** `cuda_pytest` and producing a new trace. Only when
  instrumentation or the raw schema changes.

## Acceptance criteria

- [ ] Tiers are declared in one place, and the sealed result says which tier
      ran. A tier-0 pass must not be presentable as a full gate pass; that is
      the whole risk of this change.
- [ ] Tier 0 is runnable with a single command and no GPU.
- [ ] Sealing still requires the full set. Tiering is for iteration, not for
  lowering the bar at the commit boundary.
- [ ] The existing `--suite scout-b` formal-only path is the starting point;
  extend it rather than adding a parallel mechanism.
- [ ] Any change to `--attempt-id` handling is out of scope. A new attempt id
      changes every digest and breaks `artifact_sync`; that is correct
      behaviour and was diagnosed once already. Do not "fix" it.

## Implementation status (2026-09-26)

Delivered: a `tier0` Bazel suite of the four genuinely trace-free targets, the
`tlc_scout_a_model` split into an abstract half and a facts-reading refinement
half, `--suite tier0` in the wrapper, and a `run_formal_tier0.sh` entrypoint
that seals nothing. Tier 0 runs in 13-16s against 108s for the scout-b suite,
and the gap is structural rather than cache-dependent: tier 0 excludes
`lean_scout_b_test`, which is 92% of the scout-b critical path. Independent
review verified no coverage regression -- the file-level input sets of both
sealed suites are unchanged, nothing dropped, nothing added.

**Not met: "tiers are declared in one place."** Tier membership is spread across
three: the suite in `BUILD.bazel`, the `--suite` mapping in
`run_formal_checks.sh`, and the lint and pytest stages in
`run_formal_tier0.sh`. Tiers 1 and 2 from this ticket's shape are not declared
anywhere, and tier 0 omits `source_manifest`, which the shape lists. Recorded as
outstanding rather than satisfied.

**Met differently: "the sealed result says which tier ran."** No tier field was
added to the bundle, because that would mean changing sealing, which was out of
scope. The equivalent guarantee is structural and was verified: tier 0 produces
no `outputs/` delta at all, the required-stage lists still demand all 8 and 9
stages, the stage journal records the literal command so `--suite scout-b` is on
the record, and the tier-0 transcript self-labels `not a gate pass`. There is no
path by which a tier-0 pass becomes a bundle. If a tier field is wanted later it
belongs with ticket 18's sealing changes.

### Review findings fixed in the same phase

Four gaps the review found by mutation testing, each now closed and each
verified by re-running the mutation:

- The `abstract`/`refine` wiring had no test, and the coverage tests read only
  `srcs` and `data`, never `args` -- so a target could carry the refinement
  inputs while being told to run the abstract half, and the sealed gate would
  silently stop checking that the observed trace is admitted. This was the worst
  of the four: it is the project's recurring failure shape, a guard that cannot
  see the thing it guards.
- The `--suite` to Bazel-target mapping was untested, so pointing the sealed
  scout-b suite at the trace-free tier-0 suite passed every test.
- `test_tier0_runner_refuses_gate_and_unknown_arguments` asserted only a
  non-zero exit and the usage banner, so deleting the gate-argument arm entirely
  left all 15 cases passing. It now pins the specific diagnostic.
- Tier-0 lint runs a hook subset. A hook added to `.pre-commit-config.yaml`
  would have been skipped by tier 0 forever with nothing noticing; the split is
  now pinned against the config so a new hook forces a decision.

### Gate evidence

Full nine-stage Scout B gate, default attempt id, no `--update-artifacts`,
HEAD `6a8035ced282`, all stages exit 0: `source_manifest`, `focused_pytest`,
`cuda_pytest`, `owning_pytest`, `artifact_sync`, `scout_a_regression`,
`formal_networked`, `formal_no_fetch`, `lint`.

- Scout B evidence ID:
  `sha256:cadf97359c96f0c24adb21947924e9d055353529daa5a2fce27b4031dd4fa61a`
- Scout B source ID:
  `sha256:364f1bafa376dc86d06c5aa9bdc1df96d8d3516c40f4dbc525cbf1b97398e396`
- Nested Scout A evidence ID:
  `sha256:92f2895c19c57405d7cb37097e350f6496c33ea3c63a78356fb00a2f3b10f8b1`
- Source manifest: 7 code paths hashed, 2 tracker paths informational, matching
  the working tree exactly.
- Every sealed stage log is non-empty (checked directly, since sealing does not
  yet enforce it -- ticket 18 item 3).

Tier 0 end to end on the same state: 4/4 formal targets, 10 lint hooks passed,
pyrefly 0 errors, 66 focused tests passed, 25s.
