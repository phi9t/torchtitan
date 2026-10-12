<!-- TorchTitan active-work guidance. Corenius Collab Lite style. -->
# Active work

Committed Git tickets under `.tickets/` are the durable system of record for
dispatchable multi-agent work. Agent sessions, Git worktrees, reviews, and pull
requests are evidence about work; they do not create or complete work by
themselves.

For the full workflow, read `docs/agents/multi-agent-collab.md`. That guide
supersedes this file if they conflict.

## Start or resume work

1. Search `.tickets/` before creating a new ticket.
2. Select one `open` ticket whose `deps` are closed and whose work is not
   already in progress.
3. Claim the issue before implementation by setting its `status:` field to
   `in_progress`. If it is already `in_progress`, select another issue.
4. Carry the issue path or map entry in the session context. A session may add
   evidence to an existing issue, but it must not create or close work
   implicitly.
5. For cross-repository work, use an Ultron parent issue and native linked child
   issues in affected repository projects.

## Pause, hand off, or finish

- Record decisions, partial attempts, and remaining work in the `.tickets/`
  issue before a pause, compaction, or handoff.
- Add a substantive `## Comments` entry when human intervention is required.
  Structural dependency blocking uses frontmatter `deps`.
- Close only after fresh verification. Supply a substantive message and typed
  evidence such as tests, reviewed paths, commits, or pull requests.
- Leave incomplete work `open` or `in_progress`. Session disappearance, a clean
  branch, passing tests, or a review result never closes an issue automatically.

## Human supervision

Use the `.tickets/` issue files, Git worktree status, and review
evidence for the local view. Resolve stale or inconsistent evidence through an
explicit human or agent action; the tracker does not assign, close, repair,
commit, push, open pull requests, or merge by itself.

Federation, remote listeners, and shared services require a separate approved
deployment design.
