# T08f brief: RUNG 4 -- GLM-5.3-Flash LoRA SFT on 8x B200 with EP8 (supersedes T08d/T08e) -- the campaign's main goal
Read `env-facts.md` (MANDATORY: HDFS-strict via waystone), `spec.md` (done definition), `task-08d-brief.md` (gates),
`task-13c-report.md` (GO; exact launch path, 15-step full-depth evidence, exact HDFS resume, adapter export caveat),
`task-13b-report.md`, `task-17-report.md`, `task-08b-report.md`, and T08d's tooling (`experiments/glm53_flash_sft_ladder/
run_rung4_glm.sh`, `torchtitan/experiments/glm53_flash/{generate.py,export_glm_lora.py}` -- untracked, from T08d).
You own `run_rung4_glm.sh` (adapt it), `export_glm_lora.py`, `generate.py`. Do NOT commit. Never kill foreign processes.
Training path (from T13c): `TORCHTITAN_ROOTFS_ENV=glm53_ep`, native store rootfs
(`--rootfs-store /data05/philip.yang/torchtitan-rootfs-store`), `TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/
glm53-ladder-cache`, `torchtitan.experiments.glm53_flash_ep.config_registry` at full depth (45 layers), EP8 + FSDP for
non-expert params, grouped experts, LoRA r16/alpha64, seq 512, local batch 1, 32,768 tokens/step, deterministic.
Expect ~29 s/step, ~81 GiB/GPU. GPU gate: >= 100 GiB free on all 8 GPUs, no foreign/exclusive holder; capped side jobs may
co-run. Input roots checked with `lib/hdfs_strict.sh --quick`; record input HDFS URIs + native build keys in attempt.json.
Gates (all required):
1. Seed 42, 94 steps: load time, steady throughput over >= 10 steps, per-GPU peak memory, host RSS, full loss curve.
2. Resume: stop-after-62 leg (same config), publish step-62 trainable-only checkpoint to HDFS, delete local, stage back,
   resume to 94; compare TensorBoard losses 63-94 with gate 1 (T13c showed exact equality at 11-15).
3. Seed 43, 94 steps.
4. Fix `export_glm_lora.py` to export only model keys ending exactly in `.lora_a`/`.lora_b` (it matched optimizer-state keys
   on the EP DCP checkpoint -- see T13c's strict exporter); export PEFT-style adapters for both seeds; key/shape check.
5. Eval on Countdown dev: base vs seed-42 vs seed-43 adapters. First try vLLM (default env) for GLM-5.3-Flash + LoRA; if it
   cannot serve it, use the TorchTitan-native `generate.py` (EP8, grouped) on a declared subset (e.g. first 100 dev problems
   x 4 samples, temperature 0.8, top-p 0.95, max 192 new tokens, GLM chat template), same subset for all three, countdown
   verifier, pass@1 + strict pass@1 with bootstrap CIs. `replicated_eval` only if both seeds beat base.
6. Publish adapters, eval outputs, logs, TensorBoard, configs, manifests to HDFS
   (`torchtitan/runs/glm53-ladder/rung4/<run_id>/<attempt_id>/`) with md5 + `_COMPLETE`.
Report `sdd/task-08f-report.md`: every gate PASS/FAIL/NOT RUN with evidence, HDFS URIs, and the campaign done-definition
checklist from spec.md.
