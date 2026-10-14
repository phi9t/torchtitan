# T12 brief: GLM EP phase 2, CPU/fake-backend tickets 1-3
Read `env-facts.md` (MANDATORY), `docs/research/2026-10-14-glm-5.3-flash-expert-parallelism.md` (T11 design; its phase-2
tickets 1-3 are this brief), `task-11-report.md`. Do NOT commit. No real-GPU training. Do NOT edit existing files under
`torchtitan/experiments/glm53_flash/` or `torchtitan/models/common/`; work in a new package
`torchtitan/experiments/glm53_flash_ep/` (wrap/subclass; propose shared patches in the report).
1. EP prototype: GLM MoE on the common token-dispatch EP path (ep axis, edp_shard sparse FSDP mesh, Shard(0) experts),
   debug and mini flavors, validated with `COMM_MODE=fake_backend` and `local_tensor` at EP in {1,2,4} on 4 ranks and
   {1,2,4,8} on 8 ranks.
2. Router/dispatch parity on tiny/truncated models: EP vs FSDP-only loss and grads (LoRA + activations) with
   local_tensor or CPU gloo; state bitwise or max |delta| with reason.
3. Rank-local loader + trainable-only checkpoint under EP: each EP rank reads only its experts' HF keys (tiny
   safetensors fixture), checkpoint save/load round-trip.
Then write the GPU-needing tickets 4-7 as ready-to-run briefs (`task-13..16-brief.md`) with exact commands, but do not
run them. Update the EP doc's phase-2 section with results. Report `sdd/task-12-report.md`.
