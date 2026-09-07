# Reconcile policy docs with ADR 0006

Type: task
Status: resolved
Blocked by:
Parent: ../spec.md

## Requirements

- On `land/ultron-mainline-policy` only, add one docs commit that cites
  ADR 0006 and states the TorchTitan exception: this repo's
  `ultron/mainline` is the published research trunk; topic branches
  fast-forward onto it; agents do not commit on or rewrite it.
- Do not modify `tools/ultron_mainline.py` or
  `tests/test_ultron_mainline.py`.
- Do not fast-forward `ultron/mainline` in this ticket.

## Exclusions

- Fleet-wide published-trunk mode in the Ultron tool
- Changing `docs/adr/0006-ultron-mainline-is-the-research-trunk.md`
- Push

## Verification

- `git merge-base --is-ancestor bae4954ed land/ultron-mainline-policy`
- `git merge-base --is-ancestor 679529538 land/ultron-mainline-policy`
- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan/.worktrees/land-ultron-mainline-policy && python -m pytest -q tests/test_ultron_mainline.py'`
  expected: 47 passed
- `docs/agents/ultron-mainline.md` mentions ADR 0006 and no longer
  claims this repository's trunk is unpublished

## Answer

Policy tip is `f201e3c59` (`679529538` plus the ADR 0006 exception commit).
Primary `ultron/mainline` remained `bae4954ed`. Contract tests: 47 passed.

## Comments
