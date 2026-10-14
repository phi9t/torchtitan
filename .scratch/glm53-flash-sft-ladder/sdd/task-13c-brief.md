# T13c brief: full-depth GLM-5.3-Flash EP8 readiness probe (last gate before the rung-4 rerun)
Read `env-facts.md` (MANDATORY, HDFS-strict), `task-13b-report.md` (EP8 correct on first 8 layers after the init-order fix;
5.65 s/step, 17.3 GiB/rank), `task-08b-report.md` (rank-local loader), `task-08e-brief.md` (rung-4 gates),
`torchtitan/experiments/glm53_flash_ep/`. Do NOT commit. Never kill foreign processes or other workers' jobs.
GPU gate: run the full-depth legs only when all 8 GPUs have >= 120 GiB free and are not Exclusive_Process-held by another
process; poll every 60 s and WAIT (our capped side jobs finish within ~30 min). Record the gate log.
1. Full depth (45 layers), EP8 + FSDP for non-expert params, grouped experts, LoRA rank 16, seq 512, 32,768 tokens/step,
   seed 42, deterministic, rank-local HF load (confirm each rank reads only its own experts: bytes/rank, keys/rank, load
   time), 15 steps: ms/step (steps 3-15), tokens/s, peak GiB/rank, host RSS/rank, loss/grad norm. Compare the first steps'
   loss with the loop-path reference (T08d attempt-03 seed42: step 1 1.6781, step 2 1.6739, ..., step 8 0.2885) --
   expect close, not bitwise (different data order? state it).
2. Trainable-only checkpoint under EP: save at step 10 (LoRA + optimizer + scheduler + dataloader + RNG + step; no frozen
   base), confirm size, reload into a fresh process with the rank-local base load, resume to 15, and compare losses
   11-15 with the uninterrupted run (bitwise if deterministic holds; else max |delta| with reason). Publish the checkpoint
   to HDFS via waystone, delete local, stage back, verify (the rung-4 resume gate depends on this path).
3. Adapter export under EP: produce the PEFT-style LoRA adapter (gather from EP/FSDP shards) and check key names/shapes
   against the T08b/T08d export path.
Report `sdd/task-13c-report.md` with a go/no-go for running rung 4 on EP8 and the exact launch config/env.
