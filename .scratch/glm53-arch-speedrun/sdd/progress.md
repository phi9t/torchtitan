# Progress ledger: glm53-arch-speedrun
- 2026-10-14 ~14:45Z campaign opened. S01 (harness + B0 baseline + data) and S02 (GLM element modules + identity tests +
  overfit gates) launched in parallel. S03 (ablation matrix) blocked on S01+S02.
- 14:55Z user: Step 4 is a slice-and-dice ablation study. Spec now defines factors F1-F6 and slices 4a add-one,
  4b leave-one-out from full stack G, 4c fractional factorial (res >= V) for 2-way interactions, 4d dose sweeps.
  S02 brief addendum: all combos composable + dose knobs exposed.
- ~15:05Z S03 launched: queue + opportunistic scheduler (phase A now; runs cells after S01/S02 gates). Doc: docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md
