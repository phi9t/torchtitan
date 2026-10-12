# Linear landing of remaining unique branches

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> (this is git integration, not feature TDD). Canonical intent is
> `.scratch/ultron-mainline-linear-landing/spec.md`. Do not create a
> `docs/superpowers/plans/` copy. Do not fast-forward `ultron/mainline`
> or push without explicit human authorization for that action.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fast-forward `land/incident-progress-envelope` then
`land/ultron-mainline-policy` onto `ultron/mainline` so the trunk history
stays strictly linear.

**Architecture:** Both land branches already contain current trunk
(`bae4954ed`) as ancestor. File sets are disjoint, but Git still
requires a rebase of the second branch after the first FF because the
second tip is not a descendant of the new HEAD. Never create a merge
commit. Reconcile the policy docs with ADR 0006 on the policy branch
before any trunk movement.

**Tech Stack:** git (`merge --ff-only`, `rebase`), bwrap rootfs
(`scripts/rootfs/enter_rootfs.sh`), pytest.

## Global Constraints

- Only `git merge --ff-only` may advance `ultron/mainline`. If that
  command refuses, stop.
- Do not squash, amend published commits, rebase `ultron/mainline`,
  `--no-ff`, or force-push.
- Primary checkout stays on `ultron/mainline` for every FF.
- Policy rebase runs in `.worktrees/land-ultron-mainline-policy`.
- Do not modify `tools/ultron_mainline.py` or
  `tests/test_ultron_mainline.py`.
- `python tools/ultron_mainline.py --check` is not a landing gate.
- Push is out of this plan.
- Dirty worktrees, APEX SkyRL, and production-release-readiness stay
  unlanded.

## File structure

- Modify (ticket 01 only):
  `.worktrees/land-ultron-mainline-policy/docs/agents/ultron-mainline.md`
- Landed by FF, not edited in this plan:
  `torchtitan/observability/run_evidence.py`,
  `torchtitan/observability/aggregate_outcome.py`,
  `torchtitan/trainer.py`,
  `docs/adr/0007-incident-records-and-post-hoc-outcome-aggregation.md`,
  `docs/run_evidence.md`, `CONTEXT.md`,
  `docs/agents/agentic-engineering.md`,
  `docs/agents/agentic-workflow-repos.json`,
  `docs/agents/ultron-mainline.md`,
  `tests/test_ultron_mainline.py`,
  `tools/ultron_mainline.py`,
  plus the incident unit tests already on the land branch.

---

### Task 1: Reconcile policy docs with ADR 0006

**Files:**
- Modify: `.worktrees/land-ultron-mainline-policy/docs/agents/ultron-mainline.md`
- Test: `.worktrees/land-ultron-mainline-policy/tests/test_ultron_mainline.py`
- Tracker: `.scratch/ultron-mainline-linear-landing/issues/01-reconcile-policy-with-adr-0006.md`

**Interfaces:**
- Consumes: ADR 0006 text (published research trunk, topic branches
  fast-forward, `main` deprecated).
- Produces: one new commit on `land/ultron-mainline-policy` that is a
  descendant of `679529538`. Trunk remains `bae4954ed`.

- [ ] **Step 1: Confirm starting refs**

```bash
cd ${REPO_ROOT}
git rev-parse --short ultron/mainline
git rev-parse --short land/incident-progress-envelope
git rev-parse --short land/ultron-mainline-policy
git status -sb
```

Expected:

```text
bae4954ed
253d05cae
679529538
## ultron/mainline...origin/ultron/mainline
```

Untracked `.scratch/production-release-readiness/` and
`experiments/apex_skyrl_smallscale/` are allowed. Any other dirty
tracked file on the primary checkout is a stop.

- [ ] **Step 2: Edit the policy operations guide in the policy worktree**

In `.worktrees/land-ultron-mainline-policy/docs/agents/ultron-mainline.md`,
replace the Dual mainlines section (lines 5–13) with:

```markdown
## Dual mainlines

Each canonical repository in the Ultron fleet keeps two distinct lines:

- `upstream_mainline` is the repository's native `main` or `master` and remains upstream-facing.
- `ultron_mainline` is the `ultron/mainline` integration line for Ultron-related work.
- `ultron_mainline_bootstrap` is the immutable approved commit used only for initial creation.

The fleet default is that `ultron/mainline` has no upstream and is never
implicitly published, rebased, reset, force-updated, or rewritten.

### TorchTitan exception (ADR 0006)

This repository already published `ultron/mainline` as the research trunk
on `origin`. GitHub's default branch is `ultron/mainline`. `main` is
deprecated and is not the integration line. Topic branches merge back
with fast-forward only.

Agents still do not commit on, rebase, reset, or force-update
`ultron/mainline`. A human authorizes each fast-forward and each push.
`python tools/ultron_mainline.py --check` encodes the fleet default
(no configured upstream) and will disagree with this published trunk;
do not treat a failed `--check` on TorchTitan as a reason to unpublish
or rewrite the trunk.
```

Leave the rest of the file, including the `--check` and bootstrap
sections, unchanged.

- [ ] **Step 3: Re-run the policy contract tests**

```bash
cd ${REPO_ROOT}
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan/.worktrees/land-ultron-mainline-policy && python -m pytest -q tests/test_ultron_mainline.py'
```

Expected: `47 passed`.

- [ ] **Step 4: Commit on the policy land branch only**

```bash
cd ${REPO_ROOT}/.worktrees/land-ultron-mainline-policy
git add docs/agents/ultron-mainline.md
git commit -m "$(cat <<'EOF'
Document the TorchTitan published-trunk exception

ADR 0006 already published ultron/mainline. Keep the Ultron-fleet
tool contract; record that --check is not a gate on this repo.
EOF
)"
git rev-parse --short HEAD
git merge-base --is-ancestor 679529538 HEAD && echo policy_tip_contains_cherry_pick
```

Expected: new short SHA (record it in ticket 01 Answer). Primary
checkout `git status -sb` still shows `ultron/mainline` at `bae4954ed`.

- [ ] **Step 5: Resolve ticket 01**

Set `Status: resolved` on
`.scratch/ultron-mainline-linear-landing/issues/01-reconcile-policy-with-adr-0006.md`
and append the new policy tip under `## Answer`. Copy that SHA into
`map.md` Decisions-so-far.

---

### Task 2: Fast-forward incident onto ultron/mainline

**Files:**
- No edits. FF moves `ultron/mainline` to `253d05cae`.
- Test: `tests/unit_tests/observability/test_run_evidence.py`,
  `tests/unit_tests/observability/test_aggregate_outcome.py`,
  `tests/unit_tests/test_invalid_loss.py`
- Tracker: `.scratch/ultron-mainline-linear-landing/issues/02-ff-incident.md`

**Interfaces:**
- Consumes: `land/incident-progress-envelope` = `253d05cae`, which
  already contains `bae4954ed`.
- Produces: primary `ultron/mainline` = `253d05cae`. Policy branch is
  not yet an ancestor of HEAD.

- [ ] **Step 1: Preflight ancestry (must be true or abort)**

```bash
cd ${REPO_ROOT}
test "$(git rev-parse --abbrev-ref HEAD)" = ultron/mainline
test "$(git rev-parse --short HEAD)" = bae4954ed
git merge-base --is-ancestor ultron/mainline land/incident-progress-envelope
git merge-base --is-ancestor bae4954ed land/incident-progress-envelope
```

If any test fails, stop. Do not run `git merge`.

- [ ] **Step 2: Fast-forward only**

```bash
git merge --ff-only land/incident-progress-envelope
git rev-parse --short HEAD
git log --oneline --first-parent bae4954ed..HEAD
git log --first-parent --merges bae4954ed..HEAD
```

Expected HEAD: `253d05cae`. Expected first-parent log:

```text
253d05cae Renumber incident ADR to 0007
5ee2ab00d Document incident records and outcome aggregation
475a71698 Emit a nonfinite_loss incident before aborting training
a887dbe5e Add post-hoc multi-process outcome aggregation
335b3be01 Add typed incident record and record_incident facade
```

The merges log must be empty. If `merge --ff-only` refuses, stop.

- [ ] **Step 3: Run incident tests from the primary checkout**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/observability/test_run_evidence.py tests/unit_tests/observability/test_aggregate_outcome.py tests/unit_tests/test_invalid_loss.py'
```

Expected: `80 passed`. Do not run this with cwd inside the land
worktree; git identity inside bwrap resolves through the primary
`.git` directory.

- [ ] **Step 4: Resolve ticket 02**

Set `Status: resolved` on
`.scratch/ultron-mainline-linear-landing/issues/02-ff-incident.md`.
Do not push. `origin/ultron/mainline` remains `bae4954ed`.

---

### Task 3: Rebase policy onto the new tip, then fast-forward

**Files:**
- No new edits if ticket 01 already landed on the policy branch.
- Test: `tests/test_ultron_mainline.py` plus the incident tests from
  Task 2 (regression on the combined tip).
- Tracker: `.scratch/ultron-mainline-linear-landing/issues/03-rebase-ff-policy.md`

**Interfaces:**
- Consumes: `ultron/mainline` = `253d05cae`; `land/ultron-mainline-policy`
  = ticket-01 tip (descendant of `679529538`).
- Produces: `ultron/mainline` at the rebased policy tip. Linear
  first-parent history from `bae4954ed` through incident commits, then
  the policy cherry-pick, then the ADR 0006 docs commit.

- [ ] **Step 1: Confirm incident is on trunk and policy is not yet**

```bash
cd ${REPO_ROOT}
test "$(git rev-parse --short ultron/mainline)" = 253d05cae
git merge-base --is-ancestor 253d05cae land/ultron-mainline-policy; echo policy_already_contains_incident:$?
```

Expected: `policy_already_contains_incident:1` (not an ancestor). That
is why rebase is required even though files do not overlap.

- [ ] **Step 2: Rebase the policy worktree onto ultron/mainline**

```bash
cd ${REPO_ROOT}/.worktrees/land-ultron-mainline-policy
git rebase ultron/mainline
git merge-base --is-ancestor ultron/mainline HEAD && echo rebase_ok
git log --oneline ultron/mainline..HEAD
```

Expected: two commits replayed (policy cherry-pick, then ADR 0006
docs). If the rebase stops, do not `git rebase --skip`. Fix only in
this worktree, or `git rebase --abort` and report.

- [ ] **Step 3: Fast-forward trunk to the rebased policy tip**

```bash
cd ${REPO_ROOT}
test "$(git rev-parse --abbrev-ref HEAD)" = ultron/mainline
git merge --ff-only land/ultron-mainline-policy
git log --first-parent --merges origin/ultron/mainline..HEAD
git log --oneline --first-parent origin/ultron/mainline..HEAD
```

The merges log must be empty. First-parent log should be the five
incident commits, then the two policy commits, newest last in
`git log` (newest first in `--oneline` default).

- [ ] **Step 4: Combined tests from the primary checkout**

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/observability/test_run_evidence.py tests/unit_tests/observability/test_aggregate_outcome.py tests/unit_tests/test_invalid_loss.py tests/test_ultron_mainline.py'
```

Expected: `127 passed` (80 incident + 47 policy).

- [ ] **Step 5: Resolve ticket 03 and leave push unauthorized**

Set `Status: resolved` on
`.scratch/ultron-mainline-linear-landing/issues/03-rebase-ff-policy.md`.
Report `git status -sb` (should show `ultron/mainline` ahead of
`origin/ultron/mainline` by 7 commits: 5 incident + 2 policy). Do not
push. Do not delete land worktrees unless the human asks.

---

## Out of plan

- Push `ultron/mainline` or `land/*`
- Delete ancestor-only local branches
- Land dirty worktrees or APEX / production-readiness trees
- Falcon campaign tickets 08 and 09
