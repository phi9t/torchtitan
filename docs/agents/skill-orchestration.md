# Skill Orchestration

Skills support the active workflow; they do not replace it. For multi-agent
collaboration, `docs/agents/multi-agent-collab.md` is authoritative.

## Rules

- Load a skill only when the user names it or the task clearly matches its
  description.
- If multiple skills apply, use the smallest set that changes the work.
- Do not duplicate ceremonies. A ticket, design note, review, or verification
  record should have one durable home.
- Skill text does not grant Git authority. Commit, push, pull-request creation,
  landing, history rewrite, and shared-branch movement require explicit user
  authorization for that action.
- For review, use the requested review skill or a fresh separate-context review
  against the exact candidate. Findings must cite concrete files, lines, or
  behavior.

## Relationship To Collab Lite

When a worker uses a skill, the worker's authority is still bounded by its
ticket, allowed outputs, workspace, and branch. When a reviewer uses a skill,
the reviewer remains read-only unless the lead assigns a new repair ticket in a
new worker workspace.
