# Rebase policy onto new trunk, then fast-forward

Type: task
Status: resolved
Blocked by: 01, 02
Parent: ../spec.md

## Requirements

- In `.worktrees/land-ultron-mainline-policy`, rebase
  `land/ultron-mainline-policy` onto the post-ticket-02 `ultron/mainline`
  tip (`253d05cae`).
- Abort and report if the rebase stops on a conflict. File sets were
  disjoint at replay time; a conflict means the ticket-01 docs edit
  drifted and must be resolved in the worktree, not with a merge commit
  on trunk.
- From the primary checkout, `git merge --ff-only land/ultron-mainline-policy`.
- Do not push.

## Exclusions

- Rewriting incident commits
- `git merge --no-ff`
- Deleting dirty worktrees or ancestor-only branches

## Verification

- `git merge-base --is-ancestor 253d05cae ultron/mainline`
- `git merge --ff-only` succeeded (exit 0)
- `git log --first-parent --merges origin/ultron/mainline..HEAD` is empty
- Policy tests from the primary checkout:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m pytest -q tests/test_ultron_mainline.py'
```

Expected: 47 passed.

- Incident tests from ticket 02 still pass on the combined tip.

## Answer

Rebased `land/ultron-mainline-policy` onto `253d05cae`, then
`--ff-only` to `2e68983f4`. First-parent merges since origin: empty.
Combined tests: 127 passed. Local trunk is ahead of origin by 7.
Not pushed.

## Comments
