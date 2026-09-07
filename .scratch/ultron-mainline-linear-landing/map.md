# Linear landing map

## Notes

- Trunk: `ultron/mainline` at `bae4954ed`, tracking `origin/ultron/mainline`.
- Replay worktrees already exist:
  - `.worktrees/land-incident-progress-envelope` on `land/incident-progress-envelope`
  - `.worktrees/land-ultron-mainline-policy` on `land/ultron-mainline-policy`
- Primary checkout stays on `ultron/mainline` for every FF. Rebase happens
  in the policy worktree.

## Decisions-so-far

- Integration mechanism is rebase then `git merge --ff-only`. No merge
  commits.
- Incident lands before policy.
- Policy keeps the Ultron-fleet tool contract. TorchTitan documents an
  ADR 0006 exception instead of teaching the tool that published trunks
  are allowed.
- This spec does not authorize merge, push, or commit to trunk. Ticket
  execution still needs explicit go.
- Ticket 01 resolved: `land/ultron-mainline-policy` is `f201e3c59`.

## Frontier

All tickets resolved. Local `ultron/mainline` is `2e68983f4`, ahead of
origin by 7 linear commits. Push is not authorized.

## Fog

- After ticket 01, `land/ultron-mainline-policy` will not be `679529538`;
  record the new tip in ticket 01's Answer before ticket 03 rebases.
- `ultron_mainline.py --check` will still disagree with this published
  trunk; that is accepted, not a landing blocker.
