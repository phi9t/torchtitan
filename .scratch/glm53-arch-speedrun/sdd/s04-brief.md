# S04 brief: make the GLM elements fair and fast for the ablation matrix, then run the Step-3 overfit gate
Read `../spec.md` (incl. "Ablation design"), `s02-report.md` (what exists + caveats), S01's files as they land, and
`.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY). Do NOT commit. 24 GiB per-process GPU cap, <= 2 GPUs,
never kill foreign processes. You own `torchtitan/experiments/glm53_arch_speedrun/elements/` from now on (S02 is done).
Do not edit S01-owned files; if S01's seam lacks a hook, add an adapter in elements/ and note it.
1. Indexer training (fairness): S02's `LightningIndexer` is no_grad, so in from-scratch training the indexer stays random.
   Implement the DSA/GLM-style indexer objective: KL(stop_grad(dense attention distribution, head-summed, L1-normalized)
   || softmax(indexer scores)) on the selected/all causal keys, with indexer inputs detached from the main graph,
   aux weight configurable; plus the dense-warmup option (indexer trained with dense attention for the first N steps,
   then sparse). Cite primary sources (DeepSeek-V3.2 DSA paper/code, GLM HF reference) for what GLM does; if GLM's exact
   recipe is unknown say so and make it a declared dose knob. Test: indexer recall of dense top-k rises on a toy task.
2. Fast paths: KDA via fla/Attention Gym kernels (check `torchtitan/models/common/attention`, `attn_gym.py`, the
   glm53_flash KDA path) with a fwd+bwd parity test against S02's recurrent oracle at seq 1024/4096; sparse top-k attention
   via a gather or flex-attention block-sparse path (no [B,L,L] materialization), parity vs S02's reference.
3. mHC init: add GLM's own mHC initialization as the default for A1 (match glm53_flash/DSv4 init), keep S02's
   gated-identity init as a dose knob `mhc_init in {glm, gated_identity}`.
4. After S01 B0 exists: Step-3 overfit gate per arm (B0, A1-A6, and G = all elements) on a fixed small FineWeb bank:
   CE collapse, lr=0 flat, arms diverge; plus tokens/s and peak memory per arm at seq 1024 and 4096 (`shared_gpu`).
Report `.scratch/glm53-arch-speedrun/sdd/s04-report.md`. S03's scheduler treats s04-report gates as the Step-3 gate.
