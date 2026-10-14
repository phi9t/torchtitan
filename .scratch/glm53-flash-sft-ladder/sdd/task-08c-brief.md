# T08c brief: diagnose GLM-5.3-Flash bf16 drift (TT vs HF) against an fp32 ground truth -- DIAGNOSIS ONLY

Repo: /data02/home/philip.yang/workspace/torchtitan. Do NOT commit. Do NOT edit `torchtitan/experiments/glm53_flash/model.py`
or other files there (T08b owns them); write new scripts under `torchtitan/experiments/glm53_flash_bf16_diag/` and patches as files.
Read first: `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md`, `sdd/task-07c-report.md`, `sdd/task-08-report.md` (BF16 parity
open item), `torchtitan/experiments/glm53_flash/{model.py,hf_parity.py}`.

## Problem
Layers 20-27 real-weight slice on 4 Countdown prompts: TT-bf16 vs HF-bf16 top-1 0.43-0.54, KL 0.18-0.27, while TT-fp32 vs
HF-fp32 matches (top-1 1.0). Unknown which bf16 path is closer to the truth.

## Requirements (time-box 75 min, CPU and/or 1 free GPU; respect the host RAM guard: keep RSS < 200 GB)
1. Compute fp32 ground truth (HF fp32 == TT fp32) for the slice. Measure TT-bf16 vs truth and HF-bf16 vs truth (top-1, KL,
   per-layer hidden max/mean abs). Whichever is far from truth is the suspect.
2. If TT-bf16 is the outlier: bisect per layer and per sub-block (mHC pre/post, attention (KDA or sparse MLA), router, experts,
   shared expert, norms) by running TT with that sub-block in fp32 and the rest bf16. Name the sub-block(s) responsible and
   the exact op (e.g. a bf16 accumulation, a missing upcast, softmax/sigmoid in bf16, Sinkhorn in bf16, recurrence state dtype).
   Write a minimal patch `.scratch/glm53-flash-sft-ladder/evidence/t08c-bf16-fix.patch` and show the improvement using a
   monkeypatched copy (not by editing model.py).
3. If HF-bf16 is the outlier or both are equally far: say so; then the bf16 gap is reference noise and not a TT bug.

## Report
`.scratch/glm53-flash-sft-ladder/sdd/task-08c-report.md`: table TT-bf16/HF-bf16 vs truth, bisection results, patch path, claim
`deterministic_comparison`, no commit.
