# Multi-Agent Collaboration

TorchTitan follows the Corenius Collab Lite shape for local multi-agent work:
committed Git tickets, isolated worker workspaces, exact candidate verification,
and explicit operator landing. This guide is the active authority for
multi-agent coordination. It supersedes `CONSTITUTION.md`,
`docs/agents/agentic-engineering.md`, `docs/agents/active-work.md`, and
`docs/agents/skill-orchestration.md` wherever those files add ceremony or give a
different collaboration flow.

Direct user instructions and TorchTitan's surface-specific safety rules still
come first: preserve existing work, keep core and experiments separated, use the
rootfs for Python/model/test work, and require fresh verification for claims.

The executable helper lives in `tools/collab/collab.py`; its operator guide is
`tools/collab/README.md`. Use the helper for real multi-agent attempts unless
the user explicitly asks for a manual dry run.

## When To Use This

Use this workflow when work is split across more than one agent, worktree, or
review context. For a small single-agent change, work directly in the current
checkout, verify the result, and stop before any Git action that lacks explicit
authorization.

Do not use this document to create extra process around a task the user already
made simple. Its job is coordination, not ceremony.

## Roles

- **Lead:** the interactive agent working with the user. The lead owns intent,
  decomposition, worker prompts, boundaries, verification, integration choices,
  and the handoff.
- **Worker:** a bounded agent in its own private workspace and branch. The
  worker produces a nonempty, linear candidate commit and the requested evidence.
- **Reviewer:** a fresh context that audits an exact candidate against the
  ticket, diff, and verification evidence. The reviewer reports findings; it
  does not land or mutate the candidate.

The user talks to the lead. A worker cannot push, merge, rewrite below its base,
move shared branches, touch another worker's tree, or expand its allowed output
paths. A reviewer is read-only unless the lead explicitly assigns a repair task
in a new worker workspace.

## Tickets

Dispatchable multi-agent work is recorded as committed Git tickets under
`.tickets/<exact-id>.md`. Existing `.scratch/` notes may hold sketches, research,
or private planning, but they do not grant worker authority.

Use exact ticket IDs, not prefix resolution. Keep essential requirements in the
ticket because referenced documents are not pinned transitively.

```markdown
---
id: T-example
status: open
deps: [T-prerequisite]
---

# A narrow outcome

## Outcome
Describe the behavior or artifact to produce.

## Invariants
List constraints, protected paths, dependency direction, and authority limits.

## Allowed Outputs
List exact files or directories the worker may write.

## Acceptance
List the checks, evidence, and review conditions that prove completion.
```

Status values are `open`, `in_progress`, and `closed`. Dependencies must exist,
be acyclic, and be closed before dispatch. Closing a ticket requires fresh
verification evidence and an exact candidate identity.

## Worker Contract

Every worker launch fixes these before the worker starts:

- ticket ID and exact ticket bytes;
- trunk/base commit;
- private workspace path and worker branch;
- allowed output paths;
- prohibited actions and protected paths;
- verification commands or evidence expected;
- wall-clock, GPU, network, and rootfs boundaries;
- handoff format.

Workers commit their own candidate on their worker branch. They keep a clean
workspace at handoff and report the candidate commit, base commit, commands run,
files changed, and known gaps. If the base or ticket changes, abandon the old
attempt and start a fresh worker from current trunk; cherry-pick only the useful
candidate commits.

## Candidate Verification

The lead verifies the exact candidate, not the worker's summary. Review the
diff, confirm it stayed within allowed outputs, rerun focused checks from the
candidate state, and use a fresh reviewer when risk or scope warrants it.
Passing worker-run tests are evidence, not acceptance.

For TorchTitan work, rootfs and GPU rules still apply. Python business logic,
model code, tests, training, evaluation, CUDA probes, and log parsing run inside
`scripts/rootfs/enter_rootfs.sh` unless the repository has a more specific
rootfs-aware wrapper.

## Landing

Landing is an explicit operator action. The lead may prepare and recommend a
candidate, but commit, push, branch publication, pull-request creation, and
integration require the user's explicit authorization for that action.

Landing must preserve the exact verified candidate. Prefer fast-forward
integration. Do not rebase, force-update, or rewrite shared history unless the
user explicitly authorizes that specific operation.

## Recovery

If a worker stalls, violates its boundary, produces a dirty tree, or loses its
base, preserve the workspace and evidence, then mark the attempt failed or
abandoned in the ticket notes. Start a new attempt instead of repairing inside a
contaminated workspace.

If concurrent work conflicts, the lead decides the ordering. One writer owns one
workspace; no agent edits another agent's checkout to "help".
