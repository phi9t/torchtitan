# S14 brief: C1 combined winners -- freeze, replicate at 268M, then one FineWeb-Edu epoch
Blocked by: S13 complete; epoch leg also blocked by S08 + S09 (shared epoch harness).
1. Freeze C1 with the declared rule (include an element/dose only if its mean val CE gain > 2 x B0 seed std in 4a/4b/4c;
   interactions per 4c) and write the frozen config + rationale into DESIGN.md before running.
2. C1 x3 seeds at the 268M FineWeb budget vs B0 and G (already run).
3. C1 x2 seeds one full epoch of FineWeb-Edu 10BT with the S09 harness; compare against S09's L0 and L4.
`s14-report.md`; fill the C1 rows. Claims: `replicated_eval` (268M), `representative_training` (epoch).

Common rules for every ticket: read `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY,
incl. HDFS-STRICT INPUT RULE: inputs only from waystone HDFS URIs staged with `.staged.json` and verified by
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`; outputs published to HDFS with md5). Rootfs for all Python/CUDA
(`TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`). 1 GPU per run unless stated, 24 GiB per-process cap, never use
GPUs in Exclusive_Process mode that hold another process, never kill foreign processes, label timing `shared_gpu`.
GLM rung 4 (T08d) has GPU priority: `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause this worker while
T08d waits for GPUs -- that is expected; resume where you left off. Do NOT commit. Every run records lane, mode, arm,
claim_label, evidence_tier, run_id, attempt_id, environment_class and its input HDFS URIs.
