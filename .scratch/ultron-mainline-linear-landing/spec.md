# Linear landing of remaining unique branches

Status: landed locally 2026-09-06 at `2e68983f4`. Not pushed.
Human-owned fast-forward onto `ultron/mainline` is the only integration
move. Execution proceeds only through the numbered tickets and the
session plan at `plan.md`.

Sources of intent:

- ADR 0006 (`docs/adr/0006-ultron-mainline-is-the-research-trunk.md`)
- Direct instruction: land ready branches only through rebase/FF so
  `ultron/mainline` stays linear
- Prior replay: `land/incident-progress-envelope` and
  `land/ultron-mainline-policy` already contain current trunk as ancestor

## Outcome

Fast-forward two already-replayed topic tips onto `ultron/mainline`
without a merge commit, squash, rebase of trunk, or rewrite of published
history.

Success is:

1. `git merge-base --is-ancestor bae4954ed ultron/mainline` still holds
   (old trunk is an ancestor of the new tip).
2. `git log --first-parent --oneline origin/ultron/mainline..ultron/mainline`
   is a straight list of the landed topic commits (plus one policy
   reconcile commit), no merge commits.
3. Focused tests for both landed surfaces pass from the primary checkout
   inside the bwrap rootfs.
4. `origin/ultron/mainline` is unchanged until a later, explicit push
   authorization.

## Ready branches (as of 2026-09-06)

| Branch | Tip | Unique commits | File overlap with the other |
| --- | --- | --- | --- |
| `land/incident-progress-envelope` | `253d05cae` | 5 | none |
| `land/ultron-mainline-policy` | `679529538` | 1 | none |

Both are descendants of `ultron/mainline` = `bae4954ed` =
`origin/ultron/mainline`. Each can `--ff-only` **now**. After the first
FF, the second is no longer a descendant of HEAD even though the file
sets are disjoint. The second **must** be rebased onto the new tip, then
fast-forwarded. A `git merge` that creates a merge commit is out of
scope.

## Order

1. Reconcile the policy branch with ADR 0006 (ticket 01). This commit
   happens on `land/ultron-mainline-policy` before any trunk movement.
2. Fast-forward incident onto `ultron/mainline` (ticket 02).
3. Rebase policy onto the new trunk tip, then fast-forward (ticket 03).

Incident goes first because it is core run-evidence and has no policy
contradiction. Policy goes second because it needs the ADR 0006
reconcile commit and a rebase after incident lands.

## Policy reconcile (required, docs-only)

The cherry-picked policy commit still says `ultron/mainline` is
local-only, has no upstream, and is never published. ADR 0006 already
published this trunk and this checkout tracks `origin/ultron/mainline`.

Do **not** change `tools/ultron_mainline.py` or the fleet contract tests
to allow a published upstream. Those files are Ultron-managed sibling
copies. TorchTitan records a local exception in
`docs/agents/ultron-mainline.md` and cites ADR 0006.

`python tools/ultron_mainline.py --check` is **not** a post-land gate on
this repository. Verification of the landed policy remains
`tests/test_ultron_mainline.py` (temp repos).

## Exclusions

- Merge commits, squash, rebase of `ultron/mainline`, force-push
- Push of trunk or `land/*` without a later explicit authorization
- Landing dirty worktrees (`codex/active-work-tracker`,
  `torchtitan_roborev_modded_nanogpt_review`)
- Landing `.scratch/production-release-readiness/` or
  `experiments/apex_skyrl_smallscale/`
- Deleting ancestor-only local branches (separate cleanup)
- Changing GitHub default branch or remotes
- Executing Falcon tickets 08/09
