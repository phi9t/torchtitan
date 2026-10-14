# S02 brief: GLM-5.3-Flash element modules for the speedrun substrate + identity tests + overfit gates
Read: `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY), the GLM architecture doc
`docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md`, and the verified GLM implementation in
`torchtitan/experiments/glm53_flash/model.py` (+ `torchtitan/models/deepseek_v4/mhc.py`, HcPost dim=-3 fix). Do NOT
edit those files (other tickets own them); import or copy-adapt. Do NOT commit.

You own: `torchtitan/experiments/glm53_arch_speedrun/elements/` and `.../elements/tests/`. S01 is concurrently building
the B0 model in `torchtitan/experiments/glm53_arch_speedrun/model.py`; read it as it lands and plug in via its config
seam; if the seam is missing something, write the adapter in elements/ and note the needed hook in your report.

Implement, at speedrun width (d768, head dim 128 or as S01 declares), each as a switchable element:
- A1 mHC 4-stream residual (HcPre/HcPost + Sinkhorn), identity-at-init option.
- A2 NoPE switch.
- A3 KDA layer (reuse fla/GLM KDA path) and 3:1 hybrid layout ((i+1)%4==0 softmax).
- A4 lightning indexer + top-k sparse attention for softmax layers (top-k configurable; causal; indexer trained with
  the GLM/DSA objective or straight-through as the GLM code does -- document which).
- A5 IndexPool (4->1 weighted pooling of indexer keys).
- A6 MTP head (1 layer, aux loss weight).
Identity/oracle tests (fwd AND bwd/finite grads at seq 1024 and 4096): mHC identity init == plain residual; indexer
top-k >= L == dense causal; IndexPool pool=1 == A4; KDA chunked vs recurrent reference; NoPE flag removes RoPE only.
Then overfit gate (Step 3) per arm on 1-2 GPUs with 24 GiB cap: fixed small FineWeb bank (e.g. 64 x 1024 tokens),
CE collapse, lr=0 flat, arms diverge. Measure tokens/s per arm vs B0 at seq 1024 and 4096 (shared_gpu label).
Report `.scratch/glm53-arch-speedrun/sdd/s02-report.md` with tests, overfit table, per-arm throughput/memory, and any
element that is too slow (O(L) Python) -- per skill, flag it for a side table rather than stalling the matrix.

## Addendum (14:55Z): this campaign is a factorial ablation study -- see spec "Ablation design"
Every element must be independently switchable AND composable with every other (all F1-F5 combinations must build and
pass a 2-step smoke), and expose its dose knobs as config fields: mHC num streams + Sinkhorn iters; KDA:softmax ratio;
indexer top-k; IndexPool ratio; NoPE scope (all vs softmax/sparse layers only); MTP weight/depth. Add a test that
builds every F1-F5 combination (48 cells) on a tiny width and runs fwd+bwd.
