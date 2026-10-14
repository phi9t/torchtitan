# S11 brief: slice 4b leave-one-out from G (15 cells)
Blocked by: S10 complete (shared scheduler/GPU budget; analysis conventions). Same mechanics as S10 for slice 4b
(G x3 and G-minus-each-element x2, seq 4096, 268M tokens). Report which elements only help in context (compare 4a vs
4b effects per element). `s11-report.md`; fill the 4b row. Claim: `representative_small` + `replicated_eval`.

Common rules for every ticket: read `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY,
incl. HDFS-STRICT INPUT RULE: inputs only from waystone HDFS URIs staged with `.staged.json` and verified by
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`; outputs published to HDFS with md5). Rootfs for all Python/CUDA
(`TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`). 1 GPU per run unless stated, 24 GiB per-process cap, never use
GPUs in Exclusive_Process mode that hold another process, never kill foreign processes, label timing `shared_gpu`.
GLM rung 4 (T08d) has GPU priority: `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause this worker while
T08d waits for GPUs -- that is expected; resume where you left off. Do NOT commit. Every run records lane, mode, arm,
claim_label, evidence_tier, run_id, attempt_id, environment_class and its input HDFS URIs.

## Addendum (K03 landed, d4f4fa0a2)
- All cells containing the indexer (F4 != dense) run with `indexer_aux_compile=True`, `indexer_aux_query_stride=1`
  (exact objective; stride barely helps once compiled). Expected ~0.36-0.46x B0 tokens/s at seq 4096.
- FIRST re-queue slice 4a's A4 and A5 cells (x2 seeds) at the FULL 268,435,456-token budget with these settings (S10 ran
  them at the reduced 33.5M side-table budget) and fold them into the 4a analysis, then run slice 4b.
