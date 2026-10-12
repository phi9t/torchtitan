# 25 — The gate cannot run from a clean tree

**What to build:** Make the lint stage work when the working tree is clean, so a
committed state can be gated at all.

**Blocked by:** none. **Blocks:** closing tickets 03 and 10.

**Status:** resolved for criteria 1-3 and 4; the tier-0 residual below
remains open

## The finding

Measured on 2026-09-27, running the gate from the clean committed tree at
`d8b1016d5` -- the first time in this project's history that has been attempted.
It fails:

```
FINAL_GATE_EXIT=1
source_manifest exit=0   focused_pytest exit=0   cuda_pytest exit=0
owning_pytest   exit=0   artifact_sync  exit=0
error: stage command failed ... single-rank-regression.log
```

The failure is inside the nested Single-rank gate's `lint` stage, whose log is
ten lines long -- the `git init` hints and nothing else. No pre-commit hook ran.

Cause: the source manifest is derived from `git status`, so on a clean tree it
has
**zero verified entries and zero process entries**. `source_manifest_lint_paths`
therefore yields nothing, and the lint stage's `((${#source_files[@]} > 0))`
assertion fails. Confirmed from the sealed manifest:

```
verified entries: 0
process entries : 0
head            : d8b1016d5f00
```

## Why this is worse than the coverage note in ticket 18

Ticket 18 item 4 records that manifest coverage *narrows* after commit, 87
entries to 13. That understated it. On a fully clean tree coverage is **zero**
and the gate does not merely weaken -- it **fails**.

The consequence is structural, and it applies to every bundle this project has
sealed: all of them were produced from a dirty working tree. The evidence
contract has never gated a committed state, and as written it cannot. So:

- Ticket 03's and ticket 10's acceptance criterion -- "the supported commands
  re-execute from the final worktree state" -- is **unsatisfiable**, not merely
  unmet. That is why they are still open after every other phase closed.
- "Gate the final state" and "gate a dirty tree" are currently the same
  requirement, and only the second is achievable.

## Acceptance criteria

- [ ] The lint stage lints something meaningful when the tree is clean. The
  obvious candidate is the files the `HEAD` commit touched
  (`git diff-tree --no-commit-id --name-only -r HEAD`), computed on the host,
  since the rootfs lint stage runs against a throwaway bare repository with no
  real history.
- [ ] Say in the manifest which case applied -- dirty-tree bytes or HEAD-commit
  paths -- so a bundle states what its lint stage covered rather than leaving a
  reader to infer it from an entry count.
- [ ] A test that runs the gate's lint stage against a clean tree. Without it
  this returns the moment someone gates after committing, which is exactly
  when it matters and exactly when nobody tries.
- [ ] Only then rewrite tickets 03 and 10's gate-evidence sections from a bundle
  sealed at the committed state, and close them.

## Note on what this does not invalidate

Every phase in this sequence was gated, and those gates were real: nine stages,
sealed and verified, with the manifest pinning the uncommitted bytes under
review. What is missing is the *final* seal from the committed state, which is
a different and weaker claim than "the work was never gated". Ticket 03's
stale-evidence note should be read with that distinction.

## Residual after the fix: tier 0 still lints nothing on a clean tree

Disclosed by the implementation. `run_formal_tier0.sh` derives its own lint
list from git directly rather than from the source manifest, so on a clean
tree it prints `QFV_TIER0 lint result=skipped reason=no_changed_files`.

That is a different failure from the gate's. Tier 0 degrades **honestly**: it
says it skipped and why, rather than failing obscurely or passing while
checking nothing -- so it was correctly left alone. But the consequence is the
same in kind: iterating on a committed state, tier 0's lint stage covers
nothing.

- [ ] Give tier 0 the same fallback, or have it say plainly that lint coverage
  on a clean tree belongs to the gate. Either is fine; silently skipping while
  the gate has been taught to cope is the combination to avoid.

## A correction to this ticket's own prescription

`git diff-tree --no-commit-id --name-only -r HEAD`, which this ticket
suggested, is insufficient: it prints nothing for a merge commit and nothing
for a repository's root commit. Both were measured during implementation, and
the root-commit case broke the first test fixture. The capture uses `--root
-m`.

Recorded because the ticket stated the command as if it were the answer, and
it was not.

## Closed 2026-09-27

The fix landed in `47ebae4d9` and the gate then ran from the clean committed
tree at that commit: all nine stages exit 0 at both levels, with
`lint_coverage.case = head_commit_paths` over 10 paths spanning Python, shell
and tests. Device-mesh
evidence `sha256:ab0524d8...`, Single-rank nested `sha256:19572fec...`.

Tickets 03 and 10 are resolved against that bundle, which is what criterion 4
asked for and what had been unreachable since the project began.

Still open: the tier-0 residual above. `run_formal_tier0.sh` derives its own
lint list and still covers nothing on a clean tree, reported as skipped.
