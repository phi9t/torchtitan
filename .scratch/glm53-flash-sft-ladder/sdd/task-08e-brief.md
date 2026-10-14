# T08e brief: rung 4 full run on the grouped-GEMM experts path (supersedes T08d's slow loop-path run)
Read `env-facts.md` (MANDATORY: HDFS-STRICT INPUT RULE, waystone only), `task-08d-brief.md` (all gates still apply),
`task-08b-report.md`, `task-17-report.md`, and T08d's runner `experiments/glm53_flash_sft_ladder/run_rung4_glm.sh` plus its
results tree `experiments/glm53_flash_sft_ladder/results/t08d-rung4-glm53-20261014T143843Z/` (attempt-03 seed42 loop path
reached step 8 in 44 min, loss 1.678 -> 0.289 -- keep it as the loop-path reference curve; do not delete). You own
`run_rung4_glm.sh` now. Do NOT commit. Never kill foreign processes.
Changes vs T08d:
- Set `GLM53_FLASH_EXPERTS_IMPL=grouped` for every training/eval leg (T17: 39.5x faster MoE layer, fp32 parity <= 7e-8,
  bf16 fwd 4.9e-4). Record it in every attempt artifact.
- Apply H01's `run_rung4_glm.sh` patch (`task-h01-report.md`, "Patches For Active Ticket Files") so every input root is
  checked with `experiments/glm53_flash_sft_ladder/lib/hdfs_strict.sh --quick` and its HDFS URI recorded.
- GPU gate: keep >= 150 GiB free on all 8 GPUs, but do NOT require zero processes: our capped side jobs (<= 30 GiB) may
  co-run; foreign processes or Exclusive_Process mode with another process still block.
- New run_id (`t08e-rung4-glm53-<UTC>`).
Gates (unchanged): seed 42 94 steps (>= 10 steady steps throughput + memory recorded), step-62 trainable-only checkpoint
published to HDFS -> local deleted -> staged back -> resume to 94 with losses 63-94 compared (deterministic if feasible),
seed 43 94 steps, adapter export (PEFT-style), Countdown dev eval base vs both adapters (declared subset if vLLM cannot serve
GLM; T08d may already have `generate.py` / eval tooling), publish adapters/eval/logs/manifests to HDFS with md5. Also report
the first 8 steps' loss vs the loop-path reference (expect close, not bitwise).
Report `sdd/task-08e-report.md` with every gate PASS/FAIL/NOT RUN and evidence paths.
