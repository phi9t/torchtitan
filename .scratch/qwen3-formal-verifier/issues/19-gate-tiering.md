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
