# T09c brief: 4-GPU GLM via bf16 overhead reduction (no offload, no quantization)
Read `task-09-common.md` first; it is binding. Your package: `torchtitan/experiments/glm53_flash_4gpu_lean/`.

Budget: 4-GPU static bf16 base is 145.9 GiB/GPU, so everything else must fit in ~30 GiB (8-GPU probe used 58.6 GiB).
Step 1: MEASURE where the 58.6 GiB goes (torch.cuda.memory snapshot / memory_summary at peak) using the truncated-depth
config on 4 GPUs and, if possible, one full-depth 8-GPU snapshot only when 8 GPUs are free and T08d is not running.
Step 2: cut it, one knob at a time with measurements: disable FSDP forward/backward prefetch; finer FSDP units (wrap
experts / attention separately so the largest all-gather shrinks from 13.8 GiB); reshard_after_forward; Triton/KDA
workspace and autotune limits; allocator settings (expandable_segments); smaller microbatch; chunked loss.
Step 3: decide with numbers whether full-depth fits on 4 GPUs. If the measured lower bound is > 178 GiB, stop and report
the breakdown as a NEGATIVE result (that is a valid outcome). If it fits, run the full-depth evidence from the common rules.
Numerics should be bf16-equivalent to the plain path; report the truncated-depth comparison.
