# T09b brief: 4-GPU GLM via FP8 frozen base (quantize-at-load from bf16), bf16 LoRA
Read `task-09-common.md` first; it is binding. Your package: `torchtitan/experiments/glm53_flash_4gpu_fp8/`.

Mechanism: store frozen base linear/expert weights as float8_e4m3fn with block-wise scales (128x128, like the official
GLM/DeepSeek FP8 layout; read the official repo's config.json quantization_config via HF API only, do not download weights)
computed at load time from the staged bf16 slices by the rank-local loader. Embeddings, norms, router/gate, KDA small
params, indexer and LoRA stay bf16/fp32. Forward dequantizes (or uses an FP8 GEMM such as torch._scaled_mm / torchao
blockwise if available and correct for frozen weights) per layer; the backward must give correct grads to LoRA and to
activations (dgrad through frozen W), not to W. FSDP all-gather of fp8 + scale tensors (check NCCL/FSDP2 support; fall
back to uint8 views if needed). Expected static base ~73 GiB/GPU at 4 GPUs.
Numerics are NOT identical to bf16: report truncated-depth logits drift vs bf16 (max/mean abs, top-1 agreement on Countdown
prompts, as in T07c) and loss/grad-norm deltas. Label honestly.
