# T09a brief: 4-GPU GLM via frozen-base weight streaming (FSDP2 CPU offload + rank-local loader)
Read `task-09-common.md` first; it is binding. Your package: `torchtitan/experiments/glm53_flash_4gpu_stream/`.

Mechanism: keep each rank's frozen-base FSDP shard in pinned host memory (FSDP2 `CPUOffloadPolicy`, i.e. what
`training.enable_cpu_offload` wires) and stream units to GPU per layer. The T08 host-OOM came from the full-model HF/DCP
initial load, NOT from offload itself; the rank-local loader (T08b) must fill the CPU-resident local shards directly
(verify it never materializes a full tensor; expected host RSS ~146 GiB/rank at 4 ranks, ~584 GiB total).
The glm53_flash config currently raises on `GLM53_FLASH_FULL_CPU_OFFLOAD=1`; build your own config in your package instead.
Then tune for throughput: with ~120 GiB GPU free, raise tokens per microbatch (local batch x seq) to cut the number of
weight passes per optimizer step (keep 32768 tokens/step and seq 512 fixed), and try AC off vs FullAC. Report H2D bandwidth
achieved, step time per setting, and the best setting. Numerics should match plain bf16 (offload moves bytes, not math):
report the truncated-depth comparison and whether it is bitwise.
