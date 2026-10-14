# S10 brief: slice 4a add-one (30 cells) via the scheduler
Blocked by: S06 report with `S06_STEP3_GATE=PASS` (if any arm is listed in `S06_SLOW_ARMS`, run it at the declared
reduced budget into a side table), H01. Cells: `experiments/glm53_arch_speedrun/queue/cells.jsonl` slice 4a (B0 x3 and
each add-one arm x2 at seq 4096 + the seq-1024 replicate) at 268,435,456 tokens.
Run with `experiments/glm53_arch_speedrun/scheduler.sh` (GLM53_ARCH_S02_REPORT pointed at s06-report.md,
GLM53_ARCH_MAX_PER_GPU=1, skip exclusive/foreign GPUs). Then analyze inside the rootfs with `analyze_results.py`:
per-cell val CE mean+/-std, delta vs B0 with the pre-registered threshold 2 x 0.00194 = 0.0039, behavior reads (mHC comb
matrices, indexer recall vs dense top-k, KDA gate stats, MTP aux loss). Publish the slice bundle to HDFS. Write
`s10-report.md` and fill the 4a row of the doc. Claim: `representative_small` + `replicated_eval`.

Common rules for every ticket: read `../spec.md` (binding), `.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY,
incl. HDFS-STRICT INPUT RULE: inputs only from waystone HDFS URIs staged with `.staged.json` and verified by
`experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh`; outputs published to HDFS with md5). Rootfs for all Python/CUDA
(`TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`). 1 GPU per run unless stated, 24 GiB per-process cap, never use
GPUs in Exclusive_Process mode that hold another process, never kill foreign processes, label timing `shared_gpu`.
GLM rung 4 (T08d) has GPU priority: `experiments/glm53_flash_sft_ladder/priority_pause.sh` may pause this worker while
T08d waits for GPUs -- that is expected; resume where you left off. Do NOT commit. Every run records lane, mode, arm,
claim_label, evidence_tier, run_id, attempt_id, environment_class and its input HDFS URIs.
