# 01 — Insula/Bazel formal toolchain spine

**What to build:** Enable the repository's additive formal build surface so one
Insula-aware command can invoke integrity-pinned TLC and Lean checks through
Bazel, retain their large dependencies outside the checkout, and distinguish a
real checker result from provisioning failure.

**Blocked by:** None — can start immediately.

**Status:** resolved

**Kata projection:** Import and claim this ticket before implementation; the
`.scratch` ticket remains the canonical requirements source.

- [x] The repository declares and pins its Bazel version without changing the
  authority of existing pytest, torchrun, or shell workflows.
- [x] The supported formal command re-enters Insula when needed and fails closed
  if the rootfs contract cannot be established.
- [x] Bazel materializes an integrity-pinned official TLA+ 1.8.0 artifact and
  uses its declared JDK 17 runtime rather than host Java.
- [x] Bazel materializes an integrity-pinned official Lean 4.34.0 distribution
  as an external dependency rather than through host `elan` or `PATH`.
- [x] The TLA+ smoke proves one valid bounded model succeeds and one controlled
  invalid model produces the expected named invariant violation; parse errors,
  missing tools, and download failures are failures, not negative-test passes.
- [x] The Lean smoke compiles and kernel-checks one valid proposition and proves
  one controlled invalid fact cannot establish its named proposition; the valid
  theorem has no `sorryAx` or project-declared axiom dependency.
- [x] Bazel output roots, repository caches, downloaded archives, extracted
  tools, TLC states, and Lean build products remain outside the checkout.
- [x] After one networked materialization, an explicit no-fetch run reuses the
  Bazel-managed dependencies and passes both formal smokes.
- [x] Focused tests, changed-file lint, and separate Standards and Spec reviews
  by a fresh clean-context Codex reviewer pass from the same final worktree
  state. Missing review output or unresolved blocking findings block the ticket.

## Superseded Trae review blocker

On 2026-09-24 the user explicitly replaced the Trae process gate with a fresh
Codex review. The following diagnosis is retained as historical evidence and no
longer blocks this ticket.

Implementation, focused verification, Standards review, and Spec review are
clean on the current uncommitted delta. Four exact direct Trae attempts have
failed to supply the required clean synchronous result. The first two exited
nonzero without a terminal verdict. The third printed a P1 finding and reached
its stop hook, then hung until terminated and exited 1. Its sole finding claimed
that pinned TLC 1.8.0 emits two `Error:` lines for the controlled initial-state
violation. A direct run of the exact pinned JAR and JDK disproved that premise:
TLC exited 12 and emitted exactly one `Error:` line, matching the classifier and
the already-passing real Bazel smoke. The ticket's nonzero and nonterminal
rules still prevent treating the third attempt as a pass. RoboRev cannot
substitute because the installed and current released versions do not provide
a Trae adapter.

The fourth attempt exhaustively reviewed the containment-repaired delta, ran 46
focused checks, and emitted a clean verdict with no actionable bugs after
`hook: Stop Completed`. The process then stayed asleep for more than 50 minutes,
ignored SIGINT, required SIGTERM, and exited 143. The expected worktree delta
was unchanged and its checkout-artifact search was clean. A fresh finalizer
therefore ruled Standards PASS but Spec FAIL: the clean text does not override
this ticket's explicit synchronous-completion and zero-exit requirements.

Closure originally required one of these explicit remedies:

1. restore or fix the Trae reviewer service and obtain a synchronous clean
   verdict for the unchanged diff; or
2. receive human authorization to amend or waive this ticket's terminal Trae
   acceptance gate. The user supplied this authorization on 2026-09-24 by
   replacing Trae with Codex review for every ticket.

The actual fresh-run cache at
`~/.cache/torchtitan-formal-qfv-task01-lockfresh` remains intact at 8.6 GiB,
including its vendored BCR registry marker, and backs the recorded consecutive
networked/no-fetch passes. No implementation or cache rerun is required unless
the worktree or that cache changes.

The nonterminal behavior is independently reproducible with the current stable
TraeCode CLI 0.207.1 in a temporary one-file Git repository: review prints a
clean final verdict and then times out, while generic `seed-evolving` exec runs
with and without a shell tool exit 0. App-server daemon mode and targeted
feature toggles do not repair review teardown. A syscall trace shows the hung
minimal review retaining 251 threads and live inotify watchers after final
output. Structured `exec review --json` emits a well-formed `turn.completed`
event and still remains alive, isolating the failure to post-terminal-event
shutdown. No supported local configuration found can make the exact ticket
command terminate. The later user-authorized Codex replacement superseded this
process gate.

## Closure

A fresh clean-context Codex reviewer returned Standards PASS and Spec PASS with
no blocking findings. A separate fresh finalizer returned closure PASS, required
no rerun, and found no blocker. Task 01 is complete without a commit; commit,
push, pull-request creation, and merge remain unauthorized.

## Evidence pointer

Closure evidence for this ticket lives outside this tracker, in the sibling
worktree `.worktrees/qwen3-formal-verifier` under `.superpowers/sdd/spec/`:
`task-01-report.md`, `task-01-codex-review.md`, and `task-01-codex-finalizer.md`.
That directory is gitignored and worktree-local, so it must not be deleted while
this ticket's closure is load-bearing. Task 01 sealed no run-attempt bundle; its
durable artifact is the populated external formal cache described in the report.
