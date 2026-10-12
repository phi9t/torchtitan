# Fast-forward incident onto ultron/mainline

Type: task
Status: resolved
Blocked by: 01
Parent: ../spec.md

## Requirements

- From the primary checkout on `ultron/mainline` at `bae4954ed`, run
  `git merge --ff-only land/incident-progress-envelope`.
- Abort if Git would create a merge commit or if HEAD is not `bae4954ed`
  when the command starts.
- Do not rebase, squash, or amend the five incident commits.
- Do not push.

## Exclusions

- Policy branch
- Worktree deletion (allowed after ticket 03, not required here)

## Verification

- `git rev-parse --short HEAD` equals `253d05cae`
- `git log --first-parent --merges -1 --oneline` does not show a new
  merge at HEAD (HEAD subject is `Renumber incident ADR to 0007`)
- `ls docs/adr/0007-incident-records-and-post-hoc-outcome-aggregation.md`
  exists and `docs/adr/0006-ultron-mainline-is-the-research-trunk.md`
  is unchanged
- Rootfs tests from the primary checkout:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/observability/test_run_evidence.py tests/unit_tests/observability/test_aggregate_outcome.py tests/unit_tests/test_invalid_loss.py'
```

Expected: 80 passed. Do not `cd` into the land worktree for this run;
`RunEvidence._source_state` needs the primary checkout's `.git`.

## Answer

`ultron/mainline` fast-forwarded `bae4954ed` -> `253d05cae`. First-parent
merges log empty. Incident tests: 80 passed. Not pushed.

## Comments
