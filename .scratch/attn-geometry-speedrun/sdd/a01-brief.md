# A01 -- attention geometry arms on small modded-nanogpt: harness, systems, overfit gates

Read first: .scratch/attn-geometry-speedrun/spec.md, experiments/modded_nanogpt_b200/upstream_reference.md,
.claude/skills/running-experiment-campaigns/SKILL.md, CLAUDE.md (rootfs + modded_nanogpt_b200 artifact schema).

## Requirements
1. Vehicle: the pinned upstream modded-nanogpt train_gpt.py (fetch via experiments/modded_nanogpt_b200/fetch_upstream.sh
   if not cached), scaled down to SMALL models that train on one GPU (start e.g. ~6-8 layers, d_model 384-512,
   seq 1024; pick and record). Put the variant under experiments/attn_geometry_speedrun/ (runner .sh that re-enters
   scripts/rootfs/enter_rootfs.sh; python under torchtitan/experiments/attn_geometry_speedrun/ if reusable).
   Keep the upstream file unmodified; the variant is a copy with one switch ATTN_GEOMETRY in {anchor,dot,cosine,gauss}.
2. Arms exactly as in spec.md. Same RoPE, projections, backbone, data order, optimizer, schedule. Temperature/bandwidth
   are per-head learned positive params (log-parameterized) with a configurable init. Gauss via augmented SDPA.
3. Tests (rootfs pytest): gauss augmented-SDPA == explicit -||q-k||^2/(2 sigma^2) softmax (fwd + finite bwd at
   science seq_len); cosine == anchor RMS-norm path at matched temperature with eps->0; dot == plain SDPA at
   s=1/sqrt(d); RoPE norm preservation (||k|| unchanged) for gauss.
4. Associative-recall probe: held-out eval of key->value recall on FRESH random mappings, at trained and longer
   retrieval distances (synthetic sequences; report accuracy per distance). Add it as an eval hook, not as training data.
5. Data: FineWeb tokens as used by modded_nanogpt_b200; training inputs must come from HDFS via waystone
   (waystone layout-profile --project torchtitan; verified get with --verify-md5; record input URIs). If the shards
   are not on HDFS yet, put them there first (verified put) and stage to a local verified cache.
6. Gates: (a) all 4 arms dry-run a few steps at the science shape, finite loss, arms DIVERGE (not identical curves);
   (b) fixed small bank overfit per arm (CE collapses), lr=0 control fails; (c) one 5-minute timed run of anchor to
   define the target loss and validate timing fields (compile, eval, train, peak memory recorded separately).
7. Artifacts: every attempt records lane, mode, arm, claim_label, evidence_tier, run_id, attempt_id,
   environment_class (CLAUDE.md schema). Claim labels: smoke / learnability_smoke only in A01.
8. Write .scratch/attn-geometry-speedrun/sdd/a01-report.md: config chosen, test results, dry-run table,
   overfit table, anchor 5-min result, exact launch command for the A02 pilot (19 runs, 95 timed min), and
   anything that blocks A02.

## GPU rules
Single GPU per run. GPUs are shared with long GLM-5.3 eval jobs (do NOT touch, signal, or slow them; never kill
other processes). Pick a GPU with enough free memory (check nvidia-smi), cap your memory, and wait/poll if none
is free. If timing runs happen on a shared GPU, record environment_class=shared_gpu and say timing is not a
speedrun claim.

## Exclusions
No RoPE changes, no Q/K tying, no hard top-k, no unit-normalized RBF arm, no A02 pilot runs, no multi-GPU,
no edits to torchtitan core or experiments/modded_nanogpt_b200 existing files beyond additive helpers. Do not commit.
Python/CUDA only inside the rootfs.
