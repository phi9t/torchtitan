# Collab Lite

Collab Lite coordinates local agents through committed Git tickets, isolated
worktrees, exact verification commits, and SQLite attempt receipts. It is an
optional Linux engineering tool for TorchTitan multi-agent work. It does not
start agents, push branches, authenticate reviewers, or replace human landing
authority.

Git owns `.tickets/<exact-id>.md` and accepted code. The Git common directory
owns `collab/state.sqlite3`, lock files, and check logs. Linked worktrees share
that state. SQLite records attempts and receipts; it is not a second backlog.

## Requirements

Use Python 3.10+, Git, and a local Linux filesystem with `fcntl.flock` support.
The CLI uses only the Python standard library plus the sibling
`tools/temp_root.py` helper.

Host commands need a safe, absolute, owned temporary root. `tools/temp_root.py`
validates `TMPDIR`, or selects this repo's `.scratch/tmp`. System temporary
directory output paths are refused.

## Operator Workflow

Start with a clean, committed repository on the intended trunk. For this
rewritten branch that is normally `ultron/mainline`.

```sh
PROJECT=/data02/home/philip.yang/workspace/torchtitan
collab() {
  python3 "$PROJECT/tools/temp_root.py" -- \
    python3 "$PROJECT/tools/collab/collab.py" --repo "$PROJECT" "$@"
}

collab init --trunk ultron/mainline \
  --check-json '["scripts/rootfs/enter_rootfs.sh","--","bash","-lc","cd /workspace/torchtitan && .venv-rootfs/bin/python -B -m pytest -q -p no:cacheprovider tests/test_ultron_mainline.py"]' \
  --workspace-root /data02/home/philip.yang/workspace/torchtitan-collab-worktrees
collab new T-example 'A narrow outcome'
# Edit, review, git-add, and commit the ticket on trunk.
collab ready
collab claim T-example --owner worker-a
```

Use the actual trunk and verification command for the effort. `--check-json`
must be a nonempty JSON array of nonempty strings. Workspace roots must be
outside the primary checkout. Initialization refuses existing or partial state.

The claim returns an attempt ID, a `collab/` branch, and a locked worktree.
Start the chosen agent in that worktree. The worker commits a nonempty, linear
candidate on its assigned branch, then the operator runs:

```sh
collab submit a-ACTUAL-ID
collab check a-ACTUAL-ID
# Review the checked diff, receipt, log, and ticket acceptance criteria.
collab land a-ACTUAL-ID
```

`submit` seals the exact candidate. It refuses dirty or substituted workspaces,
stale bases, changed ticket bytes, unmet dependencies, empty deltas, merge
commits, and changes to `.tickets/`, `.github/`, `.claude/`, `docs/agents/`,
`tools/collab/`, root `AGENTS.md`, root `CLAUDE.md`, or root `CONSTITUTION.md`.

`check` builds a fresh detached integration commit from the candidate delta and
the ticket closure. Its only parent is the admitted trunk base. It runs the
stored argv directly, without a shell. Success requires zero exit, a clean
workspace, unchanged detached HEAD and repository identity, and unchanged trunk.
Ignored check outputs may remain, but cleanup refuses them.

`land` is an explicit operator approval action. It requires the matching
receipt, unchanged check log, unchanged contract, the expected predecessor, and
a clean primary checkout. A fast-forward merge lands exactly the checked commit
and ticket closure together. No command pushes.

## Tickets

Only committed trunk tickets are dispatchable. IDs are exact, without prefix
resolution. Tickets must be regular files with mode `100644` and UTF-8 bytes:

```markdown
---
id: T-example
status: open
deps: [T-prerequisite]
---

# A narrow outcome

## Outcome
Describe the requested behavior.

## Invariants
State the constraints.

## Allowed Outputs
List exact files or directories the worker may write.

## Acceptance
Describe observable proof.
```

Status is `open`, `in_progress`, or `closed`. Dependencies must exist, have no
cycles, and be closed on trunk. The parser supports a narrow scalar and
inline-list frontmatter subset; it is not a general YAML parser. Keep essential
requirements in the ticket because referenced documents are not pinned
transitively.

## Trunk Movement

An attempt cannot change its admitted base. Preserve its workspace and branch.
Stop its processes, then abandon it and claim a new attempt at current trunk:

```sh
collab abandon a-OLD-ID --confirm-stopped
collab claim T-example --owner worker-a
# In the new assigned worktree, cherry-pick only the old contribution commits.
```

Changing the ticket also requires explicit abandonment and a fresh claim.

## Recovery, Cleanup, And Backup

Use `status` for committed tickets and attempt history, and `show <attempt>`
for one attempt. Before recovery or release, inspect and stop the old CLI, Git,
check, and agent processes. Confirmation does not stop them:

```sh
collab recover a-ACTUAL-ID --confirm-stopped
collab abandon a-ACTUAL-ID --confirm-stopped
collab gc a-ACTUAL-ID --confirm-stopped
collab backup /absolute/path/to/new-snapshot.sqlite3
```

`gc` requires a terminal attempt and validates every workspace before removal.
It refuses tracked modifications, untracked files, ignored outputs, repository
substitution, and changed check HEADs. Classify and preserve useful files first.
Branches, sealed refs, logs, database history, and older check worktrees remain.

`backup` creates an exclusive new SQLite snapshot and refuses overwrite,
including symlink destinations and competing writers. Preserve Git objects,
refs, live workspaces, and logs as well for a complete project backup.

## Limits And Verification

This is cooperative, single-machine coordination. Same-user agents can bypass
the CLI and edit shared state. Worktrees and advisory locks are not a hostile
code sandbox. Distributed execution, network filesystems, process supervision,
automatic leases or retries, scheduling, remote publication, authenticated
review, full YAML support, and power-loss guarantees are outside scope.

Run the public CLI suite in disposable real Git repositories:

```sh
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && .venv-rootfs/bin/python -B -m unittest discover -s tools/collab/tests -v'
```

Tests use `COLLAB_FAILPOINT` for process-death scenarios; leave it unset in
normal operation.
