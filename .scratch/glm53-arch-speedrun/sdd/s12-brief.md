# S12 brief: slice 4c fractional factorial (24 rows x 2 seeds = 48 cells)
Blocked by: S11 complete. Run the declared design (`queue/DESIGN.md`, `alias_structure.json`, generator seed 5303,
rank 21/21) unchanged. Fit the registered model (intercept + all main effects + all two-way terms incl. binary x sparse)
by OLS on per-cell val CE; bootstrap CIs over seeds; report main effects and the pre-registered interactions
(NoPE x KDA, indexer x IndexPool, mHC x MTP, KDA x sparse) with the 2x seed-std rule; check consistency with 4a/4b.
`s12-report.md`; fill the 4c row. Claim: `representative_small` + `replicated_eval`.

Common rules for every ticket: read `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY,
incl. HDFS-STRICT INPUT RULE: inputs only from waystone HDFS URIs staged with `.staged.json` and verified by
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`; outputs published to HDFS with md5). Rootfs for all Python/CUDA
(`TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`). 1 GPU per run unless stated, 24 GiB per-process cap, never use
GPUs in Exclusive_Process mode that hold another process, never kill foreign processes, label timing `shared_gpu`.
GLM rung 4 (T08d) has GPU priority: `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause this worker while
T08d waits for GPUs -- that is expected; resume where you left off. Do NOT commit. Every run records lane, mode, arm,
claim_label, evidence_tier, run_id, attempt_id, environment_class and its input HDFS URIs.
