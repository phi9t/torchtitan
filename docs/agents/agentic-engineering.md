# Agentic Engineering Workflow

TorchTitan no longer uses the imported gated agentic-engineering ceremony as
the default collaboration workflow. For multi-agent work, follow
`docs/agents/multi-agent-collab.md`.

This file is retained as a compatibility pointer for older references. The
active rules are:

1. Direct user instructions and TorchTitan repository guidance come first.
2. Single-agent work should stay direct: inspect the relevant surface, make the
   change, verify it, and report the result.
3. Multi-agent work uses committed Git tickets, isolated worker workspaces,
   exact candidate verification, and explicit operator landing.
4. `CONSTITUTION.md`, `docs/agents/active-work.md`, and
   `docs/agents/skill-orchestration.md` are lower-priority references wherever
   they add extra ceremony or conflict with `multi-agent-collab.md`.

Use skills when they are actually triggered by the task. Skill text can guide
implementation or review, but it does not grant authority to commit, push,
land, rewrite history, or edit another agent's workspace.
