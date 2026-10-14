# Progress ledger: glm53-arch-speedrun
- 2026-10-14 ~14:45Z campaign opened. S01 (harness + B0 baseline + data) and S02 (GLM element modules + identity tests +
  overfit gates) launched in parallel. S03 (ablation matrix) blocked on S01+S02.
- 14:55Z user: Step 4 is a slice-and-dice ablation study. Spec now defines factors F1-F6 and slices 4a add-one,
  4b leave-one-out from full stack G, 4c fractional factorial (res >= V) for 2-way interactions, 4d dose sweeps.
  S02 brief addendum: all combos composable + dose knobs exposed.
- ~15:05Z S03 launched: queue + opportunistic scheduler (phase A now; runs cells after S01/S02 gates). Doc: docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md
- ~15:20Z S02 DONE (fc412ef58): elements + 13 identity tests pass; overfit blocked on S01. Gaps: indexer no_grad (untrained in from-scratch runs), Python KDA/O(L^2) sparse refs, gated-identity mHC init. S04 launched to fix (indexer KL objective + dense warmup, fast kernels, GLM mHC init) and own the Step-3 gate.
- ~15:30Z S03 phase A DONE (committed): 93-cell queue + scheduler. Matrix not started (S01 missing). Launch plan: once S01 + S04 gates are green (reviewed by me), run scheduler.sh with GLM53_ARCH_S02_REPORT=.scratch/glm53-arch-speedrun/sdd/s04-report.md.
- NOTE: when launching the speedrun scheduler before T08d finishes, cap 1 cell/GPU (24 GiB) so the 150 GiB T08d gate can still open.
- ~15:25Z S04: elements gate PASS (committed); Step-3 BLOCKED on S01 (B0-only seam). Next: S05 = wire elements + aux-loss collector into S01 model, then Step-3 overfit gate; launch after S01 report.
- ~15:45Z S01 DONE (fce8a52fb): harness + FineWeb HDFS ecbb586-fineweb10b-gpt2-binonly-v1; B0 only 8.4M tokens (CE 6.958+/-0.012) at 53k tok/s, 4.2 GiB -> too early/slow for ablations. S05 launched: throughput + budget recalibration, element wiring, Step-3 gate.
- ~16:55Z S05 DONE (committed): B0 calibrated 268M tokens, seq4096 val CE 3.9028+/-0.0019 (thr 0.0039), ~220k tok/s quiet; elements wired. Step-3 FAIL (not run). Slow: A4/A5 ~2.3k tok/s, A1 21k, G 7k. S06 launched: Step-3 gate + masked-dense sparse path + mHC perf.
- ~17:45Z user: run a ladder of speedrun-scale model families (gpt2 -> qwen3 -> mini-kimi-k3 -> mini-glm-5.3f). Spec section added; S07 launched (new pkg glm53_arch_ladder; matched active non-embedding params; 2 seeds; avoids exclusive/foreign GPUs).
- ~17:55Z user: speedruns must do >= 1 full epoch of FineWeb-Edu. Decided (AskUserQuestion): sample-10BT; ladder L0-L4 + C1 finalists only; ablation queue stays 268M. S08 launched (download/tokenize/publish FineWeb-Edu 10BT). S09 (ladder epoch runs) to launch after S07 gates + S08 data.
- ~18:05Z S08 DONE: FineWeb-Edu 10BT on HDFS fineweb-edu-10BT-gpt2/87f0914-v1 (9.85B train + 100M val tokens). T08d seed42 full run up on 8 GPUs at ~18:00Z; priority_pause resumed S06/S07.
- ~20:10Z S06 PASS (ad0e5c58b). S10 launched (4a; A4/A5 at 33.5M-token side table). K01 launched (Triton top-k sparse attention fwd/bwd, torch.library op, speedrun wiring).
- ~20:40Z S07 DONE (committed): L1 3.8694 < L0 3.9028 (sig.); L2 MoE 4.044 worse (suspect harness); L3/L4 reduced only. S07b launched (L2 diagnosis, L4 grouped experts + K03 knobs, L3 profile, full reruns). S09 now blocked on S07b.
