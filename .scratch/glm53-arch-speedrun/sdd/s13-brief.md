# S13 brief: slice 4d dose sweeps (36 templates)
Blocked by: S12 complete. Freeze each element's context per the deferred rule (better of add-one vs leave-one-out
context from S10/S11), instantiate `queue/deferred_4d_templates.jsonl` via the generator (record the frozen choices
in DESIGN.md BEFORE running), run: mHC streams {2,4,8} + Sinkhorn iters, KDA ratio {1:1,3:1,7:1}, top-k {L/16,L/8,L/4},
IndexPool {2,4,8}, NoPE scope {all, softmax-only}, MTP weight {0.1,0.3}, plus mhc_init {glm,gated_identity} and indexer
aux weight/warmup if S04 declared them as dose knobs. `s13-report.md`; fill the 4d row. Claim: `representative_small`.

Common rules for every ticket: read `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY,
incl. HDFS-STRICT INPUT RULE: inputs only from waystone HDFS URIs staged with `.staged.json` and verified by
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`; outputs published to HDFS with md5). Rootfs for all Python/CUDA
(`TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`). 1 GPU per run unless stated, 24 GiB per-process cap, never use
GPUs in Exclusive_Process mode that hold another process, never kill foreign processes, label timing `shared_gpu`.
GLM rung 4 (T08d) has GPU priority: `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause this worker while
T08d waits for GPUs -- that is expected; resume where you left off. Do NOT commit. Every run records lane, mode, arm,
claim_label, evidence_tier, run_id, attempt_id, environment_class and its input HDFS URIs.
