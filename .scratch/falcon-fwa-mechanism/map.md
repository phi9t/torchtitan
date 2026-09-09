# Falcon FWA mechanism campaign map

## Notes

- Methodology: `spec.md`, approved and technically reviewed 2026-09-07.
- Parent evidence and final closure remain in `../falcon-fast-weight-attention/`.
- This is a mechanism campaign, not a retune of Campaign B or a paper
  reproduction.

## Decisions-so-far

- Screen M0-M4 for three matched seeds at 2,000 steps, at most one summed B200
  GPU-hour; confirm only a surviving contrast at 8,000 steps within three B200
  GPU-hours.
- RMS/L2 equivalence uses the complete head-dimension coordinate transform,
  separate normalization and NLMS epsilon roles, scaled lambda, and
  coordinate-adjusted state.
- Directional effects require three paired seed signs and at least 0.01 CE;
  M2/M4 also has a predeclared practical-equivalence margin.
- Historical evidence normalization and repaired evaluation are owned by
  Campaign B. New science attempts consume its native evidence contract.
- A terminal no-run decision is a valid mechanism closeout when nothing
  promotes.

## Frontier

`01` (RMS/L2 scale transform) is ready and requires no GPU science run. The
human-enforced Campaign B `09` evidence gate for `02` is satisfied as of
2026-09-08. Actual-shape preflight `03` additionally waits for Campaign B `11`.

## Fog

- Cross-campaign gates are recorded in ticket text but are not automatically
  resolved by the local Markdown frontier scanner.
- Only about 24G was free at ticket publication; every GPU ticket needs a disk
  and artifact-retention preflight.
- The paper's exact `ctxeta-ctxlambda` parameterization remains unpublished and
  outside this campaign.
