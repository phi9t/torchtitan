# 02 — Scout A: single-rank Qwen3 tracer bullet

**What to build:** Carry one real, deterministic single-rank Qwen3 optimizer
step through the unchanged core Trainer, experiment-owned observation,
normalization, generated facts, and both formal checkers.

**Blocked by:** 01 — Insula/Bazel formal toolchain spine.

**Status:** resolved

- [x] Configuration is selected through `ConfigManager` from
  `qwen3_debugmodel`; normal CLI overrides set one CUDA rank, DP=1, TP=1,
  every other mesh degree to 1, bfloat16, AdamW, local batch 1, sequence length
  128, global batch 1, 128 tokens per optimizer step, one gradient-accumulation
  step, one optimizer step, seed 42, and deterministic mode.
- [x] The run uses the repository-local `c4_test` data and test tokenizer,
  records the exact first-batch identity, starts from seed-42 model
  initialization, and disables checkpoint loading and saving.
- [x] The success path executes the core `Trainer`; it does not reconstruct a
  trainer loop or use the synthetic trace builder as success evidence.
- [x] Experiment-owned PyTorch contexts and hooks observe forward, backward,
  gradient readiness, optimizer mutation, and step completion without changing
  default core behavior or causing core to import experiment/formal packages.
- [x] Raw observed evidence is immutable and normalization emits deterministic
  `qwen3.formal.scout.v0` facts with complete run/attempt, rank, device,
  clock/order, step/phase, lineage, event, and causal identities.
- [x] Every normalized fact has a provenance link to raw observed evidence.
- [x] Exporters generate facts only; independently maintained TLA+ semantics and
  Lean definitions establish the single-rank step-lifecycle invariants.
- [x] TLC accepts the valid trace and rejects a controlled named lifecycle
  mutation; Lean checks the valid theorem without `sorryAx` or project axioms
  and rejects one controlled false lifecycle fact.
- [x] Checked-in generated modules match the normalized trace while raw traces,
  checker logs, state spaces, and Lean build trees remain ignored evidence.
- [x] The supported Insula command runs focused pytest checks, artifact sync,
  Bazel formal checks, and the no-fetch cache rerun. From that same final state,
  changed-file lint and separate Standards and Spec reviews by a fresh
  clean-context Codex reviewer also pass. Missing review output or unresolved
  blocking findings block the ticket.

## Evidence pointer

Scout A closure evidence lives outside this tracker, in the sibling worktree
`.worktrees/qwen3-formal-verifier` under `.superpowers/sdd/spec/`:
`task-02-report.md`, `task-02-codex-review.md`, `task-02-finalizer.md`, and the
two fix-round re-reviews. That directory is gitignored and worktree-local, so it
must not be deleted while this ticket's closure is load-bearing.

The canonical Task 02 run-attempt identities recorded there are:

```text
run_id     = qfv-scout-a-seed42
attempt_id = single-rank-cuda-v0
trace_id   = sha256:f8f07558565d342c8f0d262319c38135d8c84926daed273b31fa6edf38694259
```

Ticket 03 later re-executed Scout A as a nested regression under its own run and
attempt identities; those are recorded in the Task 03 report, not here.
