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

## Addendum at launch (S06 landed: S06_STEP3_GATE=PASS, S06_SLOW_ARMS=A4,A5,G)
- Run B0, A1, A2, A3, A6 (and the seq-1024 replicate) at the full 268,435,456-token budget.
- A4, A5 (and G when 4b starts) are ~0.05x B0 at seq 4096 (masked dense attention). Run them NOW at a declared reduced
  budget of 33,554,432 tokens (1/8, 128 steps) as a side table, comparing against B0 at the same token count (read
  B0's val CE at 33.5M tokens from the full-budget B0 curve). K01 is building a fast top-k sparse attention kernel; when it
  lands the full-budget A4/A5 cells are re-queued -- do not wait for it.
- Scheduler settings: GLM53_ARCH_MAX_CONCURRENT=5, GLM53_ARCH_MAX_PER_GPU=1, min free 40 GiB; EP work (T13/T18, <= 30
  GiB per process on all 8 GPUs) has priority -- never launch on a GPU with < 40 GiB free.
