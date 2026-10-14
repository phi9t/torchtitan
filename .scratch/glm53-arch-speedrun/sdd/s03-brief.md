# S03 brief: ablation queue + opportunistic GPU scheduler + execute slices 4a-4d
Read: `../spec.md` ("Ablation design" is binding), `docs/research/2026-10-14-glm-5.3-flash-arch-speedrun-ablations.md`,
`.scratch/glm53-flash-sft-ladder/sdd/env-facts.md` (MANDATORY). Do NOT commit. Never kill or signal processes you did
not start (shared host: GLM ladder jobs need ~132 GiB on all 8 GPUs; foreign users come and go).

Phase A (start now, no GPU): build
- `experiments/glm53_arch_speedrun/queue/` with a declared design: `cells.jsonl` (one line per run: cell id, slice,
  factor levels, dose knobs, seed, seq len, token budget, config overrides). Generate the 4c fractional factorial with a
  documented resolution >= V design (write the generator + the alias structure into the doc), and 4a/4b/4d from the spec.
- `experiments/glm53_arch_speedrun/scheduler.sh` (host orchestration only; every run re-enters the rootfs via the S01
  runner): loop -> read nvidia-smi free memory per GPU -> launch the next pending cell on 1-2 GPUs that each have
  >= 30 GiB free (job uses the 24 GiB cap) -> at most 6 concurrent cells, at most 2 per GPU -> per-cell state files
  (pending/running/done/failed with exit code, run_id, attempt_id) -> resumable after restart -> retry once only on
  infrastructure failure (NCCL/launch), never on numeric failure. Log one compact line per decision.
Phase B: wait (poll every 120 s) until `.scratch/glm53-arch-speedrun/sdd/s01-report.md` and `s02-report.md` both exist
and their gates pass. If a gate failed, stop and report -- do not run the matrix on a red harness.
Phase C: run the queue in slice order 4a -> 4b -> 4c -> 4d, then C1 (combined winners, frozen by the declared rule,
3 seeds). After each slice, write the analysis (per-cell mean +/- std, factorial effects with bootstrap CIs, behavior
reads) into the doc's results cells, replacing that slice's TODO(S03), and publish the slice bundle to HDFS
(`torchtitan/runs/glm53-arch-speedrun/<run_id>/`, md5). Python analysis runs inside the rootfs.
Report `.scratch/glm53-arch-speedrun/sdd/s03-report.md` after each slice (append), final at the end.
