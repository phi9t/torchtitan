# GLM-5.3-Flash architecture elements in a modded-nanogpt-style speedrun -- campaign spec

Status: approved by user 2026-10-14 ("see if we can do glm-5.3-flash kind of arch elements (mHC, indexer, etc.) on a
modded-nanogpt type of speedrun setting"). Methodology: `.claude/skills/running-experiment-campaigns/SKILL.md`.
Architecture source of truth: `docs/research/2026-10-14-glm-5.3-flash-architecture-and-finetuning.md` and
`torchtitan/experiments/glm53_flash/` (verified against HF reference in T07c).

## Question
Which GLM-5.3-Flash architecture elements help, hurt, or are neutral at speedrun scale (~124M params, FineWeb, a few
hundred M tokens), measured as validation CE at a fixed token budget and tokens/time-to-target?

## Substrate (decision D1)
TorchTitan-native small GPT in a new experiment package `torchtitan/experiments/glm53_arch_speedrun/`, modelled on the
classic modded-nanogpt speedrun recipe (12 layers, d768, 6 heads x 128, RoPE, QK-norm, ReLU^2 MLP, untied head,
logit softcap, Muon for matrices + Adam for embeddings/scalars, trapezoid LR, FineWeb GPT-2 .bin shards, val on the
standard 10M-token FineWeb val). Reuse `torchtitan/experiments/falcon/bin_reader.py` and Muon from
`torchtitan/components/optim`. GLM elements are switchable modules reusing `torchtitan/experiments/glm53_flash` code.
The upstream record script (`experiments/modded_nanogpt_b200`, upstream ecbb586) is only an anchor reference, not the
ablation substrate (its MUDD/bigram/FP8 tricks would confound single-knob arms).

## Arms (one knob each vs baseline B0)
- A1 mHC: 4-stream manifold hyper-connection residuals (GLM/DSv4 HcPre/HcPost, Sinkhorn comb), identity-at-init.
- A2 NoPE: no positional encoding (all layers) -- GLM uses NoPE.
- A3 KDA hybrid 3:1: KDA linear attention in 3 of 4 layers, full softmax attention every 4th ((i+1)%4==0).
- A4 Sparse attention: lightning indexer + top-k key selection on the softmax layers (DSA/GLM style), top-k declared.
- A5 IndexPool: A4 plus 4->1 weighted pooling of indexer keys.
- A6 MTP: one multi-token-prediction layer (aux loss weight declared).
- A7 MoE (optional, last): fine-grained MoE FFN with shared expert, matched active params.
- C1 combined: the winners, frozen before looking at C1.
Long-context relevance (KDA/indexer) needs seq >= 4096; declare the training seq len per arm family in the harness note.

## Gates (Steps 1-6 of the skill)
1 Harness: tests per element incl. identity reductions (mHC n=1/identity init == plain residual; indexer top-k >= L ==
  dense causal attention; IndexPool pool=1 == A4; KDA vs recurrent oracle fwd+bwd at science seq len); dry-run the
  whole matrix at science shape.
2 Systems: B0 finite, no hang; B0 seed noise from 3 seeds; B0 reaches a declared val CE target (e.g. 3.28 or the
  calibrated equivalent) in declared tokens.
3 Overfit: each arm collapses CE on a fixed small bank; lr=0 flat; arms' curves diverge (no silent collapse).
4 Ablate: matched tokens/batch/seq/optimizer, 2+ seeds (3 for B0), val CE at budget + tokens-to-target, arms must diverge.
  Winner rule declared before looking: an arm "wins" if mean val CE improves by more than 2x B0 seed std.
5/6 Combine winners (C1), replicate, write table. Transfer eval optional (HellaSwag-style or lm_eval) later.

## Constraints
- Rootfs for all Python/CUDA (`env-facts.md` of the GLM ladder campaign applies; use `TORCHTITAN_ROOTFS_ENV=glm53_train`
  for torch nightly + fla/KDA kernels; `unset CC CXX`).
- GPUs are shared with the GLM ladder (T08d needs all 8 GPUs with ~132 GiB each) and foreign users. Each speedrun job:
  1-2 GPUs, hard per-process GPU memory cap 24 GiB (`torch.cuda.set_per_process_memory_fraction`), so GLM jobs can
  co-run. Never kill foreign processes. Label all timing `shared_gpu`; primary metric is token-matched val CE.
- Data: FineWeb10B GPT-2 shards; store/verify via waystone HDFS (`torchtitan/datasets/fineweb10B-gpt2/<rev>`) with MD5.
- Attempt artifacts record lane, mode, arm, claim_label, evidence_tier, run_id, attempt_id, environment_class.
- No commit by workers. Claim labels per skill.

## Ablation design (user 2026-10-14: "ablation studies -- slice and dice the arch elements")
Step 4 is a structured ablation study, not a single add-one table. Factors (G = full GLM-like stack):
- F1 mHC {off, on}; F2 positions {RoPE, NoPE}; F3 mixer {softmax-only, KDA 3:1 hybrid};
- F4 sparse {dense, indexer top-k, indexer top-k + IndexPool} (IndexPool nested inside indexer);
- F5 MTP {off, on}; F6 MoE {off, on} (optional; only if S02 delivers it at matched active params).
Slices, in this order (each matched tokens/batch/seq/optimizer; seeds: B0 and G x3, everything else x2):
- 4a add-one: B0 + each single element (main effects in isolation).
- 4b leave-one-out: G minus each single element (main effects in context; catches elements that only help together,
  e.g. NoPE may only work when KDA supplies positional signal).
- 4c interactions: fractional factorial over F1-F5 (2^4 x 3 = 48 cells full; run a resolution >= V fraction, ~24
  cells, so all main effects and 2-way interactions are estimable unaliased; declare the design matrix before running).
  Pre-registered interaction hypotheses: NoPE x KDA, indexer x IndexPool, mHC x depth/MTP, KDA x sparse.
- 4d intra-element dose sweeps (one element at a time, on its best context from 4a/4b):
  mHC streams n in {2,4,8} and Sinkhorn iters; KDA:softmax ratio in {1:1, 3:1, 7:1}; indexer top-k in {L/16, L/8, L/4};
  IndexPool ratio in {2,4,8}; NoPE scope {all layers, softmax/sparse layers only}; MTP depth/weight {0.1, 0.3}.
Seq len: 4a-4d run at seq 4096 (KDA/indexer are long-context mechanisms); a seq-1024 replicate of 4a only.
Analysis: per-cell mean +/- seed std; factorial effects with bootstrap CIs; "effect" requires |delta| > 2x B0 seed std.
Report val CE at budget, tokens-to-target, tokens/s and memory per cell (shared_gpu), and behavior reads
(mHC comb matrices / stream usage, indexer selection recall vs dense attention top-k, KDA gate stats).
