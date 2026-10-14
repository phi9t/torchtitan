# T09b Report: 4-GPU GLM FP8 Frozen Base

## Claim

T09b implemented a new package,
`torchtitan/experiments/glm53_flash_4gpu_fp8/`, that stores selected frozen
GLM-5.3-Flash base weights persistently as `torch.float8_e4m3fn` with 128x128
block scales loaded from the staged BF16 Hugging Face checkpoint. LoRA,
embeddings, norms, router/gate/indexer/KDA small parameters, and other official
exclusions stay BF16/FP32.

Claim label: `partial_full_depth_training_smoke`, not 12-step proof and not
94-step readiness.

The mechanism fits and trains full-depth on 4x B200 for two optimizer steps
before the 45-minute ticket cap, with TensorBoard peak rank-0 active memory
about 102.36 GiB and reserved memory about 116.44 GiB. It does not satisfy the
required `>=12` full-depth steps because the current forward dequantizes FP8
weights to BF16/FP32 and uses regular `F.linear`; it is a storage/load/backprop
proof, not an optimized FP8 GEMM path.

## Mechanism

- New package only: `torchtitan/experiments/glm53_flash_4gpu_fp8/`.
- Base `glm53_flash/` package was not edited.
- `FP8FrozenLoRALinear` stores frozen base `weight` as FP8 qdata plus FP32
  `weight_scale`; LoRA `lora_a`/`lora_b` remain trainable BF16.
- `FP8FrozenExperts` stores grouped MoE expert `gate_up_proj` and `down_proj`
  as FP8 qdata plus FP32 block scales.
- Quantization is 128x128 blockwise E4M3FN. Forward dequantizes then calls
  `F.linear`; gradients flow to activations and LoRA, not to frozen qdata/scales.
- FSDP dtype override keeps FP8 qdata and scale tensors from being converted by
  mixed-precision param policy.
- Rank-local loader reads staged BF16 safetensors slices and quantizes them into
  FP8 storage at load time. No FP8 HF checkpoint was downloaded.
- Official API evidence is captured at
  `.scratch/glm53-flash-sft-ladder/evidence/task-09b/hf_api/glm53_flash_quantization_config.json`:
  `quant_method="fp8"` and `modules_to_not_convert` count `1509`.

## Static Memory

Evidence:
`.scratch/glm53-flash-sft-ladder/evidence/task-09b/static_param_accounting.json`.

| Model storage | Total bytes | Static GiB/rank on 4 GPUs |
| --- | ---: | ---: |
| BF16 frozen-base path | 626,787,981,308 | 145.935 |
| FP8 frozen-base path | 319,839,885,308 | 74.469 |
| Saved | 306,948,096,000 | 71.467 |

FP8 storage breakdown:

- `float8_e4m3fn` qdata: 307,023,052,800 bytes.
- BF16 non-quantized/trainable storage: 12,741,875,708 bytes.
- FP32 block scales: 74,956,800 bytes.
- Trainable LoRA params remain 67,190,784.

## Truncated Full-Width Comparison

Claim label: `fp8_truncated_full_width_comparison_not_bitwise`.
Environment class: `4gpu_b200_rootfs_functional`.

Config:

- First 8 full-width layers, layer offset 0.
- Seq len 512, 32768 tokens per optimizer step, local batch size 1.
- 4 ranks, DP shard 4, seed 42, Countdown SFT data.
- BF16 baseline:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09b/truncated_plain_4gpu_attempt02/`.
- FP8:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09b/truncated_fp8_4gpu_attempt01/`.
- Parsed comparison:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09b/truncated_4gpu_parity.json`.

| Step | BF16 loss | FP8 loss | Abs delta | BF16 grad | FP8 grad | Abs grad delta | BF16 mem GiB | FP8 mem GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.81229 | 10.81194 | 0.00035 | 1.8672 | 1.9844 | 0.1172 | 76.27 | 53.13 |
| 2 | 10.72388 | 10.71379 | 0.01009 | 1.9531 | 2.0781 | 0.1250 | 76.28 | 53.14 |
| 3 | 10.47549 | 10.44695 | 0.02854 | 2.4688 | 2.6250 | 0.1562 | 76.28 | 53.20 |

The FP8 run converted 32 linear modules and 5 expert modules in the first-8
model. Rank-0 load verification checked:

- `layers.0.mlp.gate_proj.weight`
- `layers.3.self_attn.q_a_proj.weight`
- `layers.3.mlp.experts.gate_up_proj`

Max dequant error on verified keys was `0.0004185`; mean dequant error was
`2.42e-05`.

Logits drift note: the successful full-width truncated evidence above covers
loss, grad norm, memory, and throughput. A one-GPU full-width logits probe was
attempted but hit CUDA OOM under current host GPU pressure before producing an
artifact. The available logits/top-1 drift artifact remains debug-depth only:
`.scratch/glm53-flash-sft-ladder/evidence/task-09b/debug_parity.json`.
Its claim label is `fp8_truncated_debug_comparison_not_bitwise`, environment
class `cpu_functional`, with step-1 max abs logit diff `0.67587`, mean abs diff
`0.07449`, and top-1 agreement `0.5625`.

## Full-Depth 4-GPU Attempt

Claim label: `full_depth_fp8_timeout_smoke`.
Environment class: `4gpu_b200_rootfs_full_depth_timeout`.

Evidence root:
`.scratch/glm53-flash-sft-ladder/evidence/task-09b/full_4gpu_attempt02/`.

Launch:

- Start: `2026-10-14T14:53:19Z`.
- End: `2026-10-14T16:14:39Z`.
- Exit status: `143` from the 45-minute timeout after lock wait and execution.
- GPUs: `CUDA_VISIBLE_DEVICES=2,3,6,7`.
- Rootfs env: `TORCHTITAN_ROOTFS_ENV=glm53_train`, `unset CC CXX`.
- Formal cache: `/data05/philip.yang/glm53-ladder-cache`.
- Config:
  `torchtitan.experiments.glm53_flash_4gpu_fp8.config_registry::glm53_flash_countdown_lora`.
- Full GLM: 45 layers, seq len 512, 32768 tokens per optimizer step, seed 42,
  LoRA rank 16 alpha 64, FullAC, no CPU offload.

Result:

- Completed 2 optimizer steps before timeout; did not reach the required 12.
- No OOMs recorded in TensorBoard.
- TensorBoard scalar artifact:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09b/full_4gpu_attempt02/tb_scalars.json`.

| Step | Global avg loss | Global max loss | Grad norm | Throughput tps | End-to-end s | Max active GiB | Max reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.64437 | 1.65428 | 3.3750 | 10.775 | 760.29 | 102.304 | 116.379 |
| 2 | 1.17805 | 1.20010 | 2.6875 | 14.402 | 568.82 | 102.361 | 116.436 |

Comparison anchor:

- T08b 8-GPU BF16 step-1 loss: `1.67569`, grad norm `3.2812`, peak memory
  `131.56 GiB`, model CUDA memory `72.99 GiB`.
- T09b FP8 full-depth step-1 loss: `1.64437`, grad norm `3.3750`.

Memory scope notes:

- TensorBoard rank-0 max active/reserved was `102.36/116.44 GiB`.
- External `nvidia-smi` sampling during full-depth saw selected GPUs rise above
  rank-0 TensorBoard scope, roughly up to `~144 GiB` on GPUs 2/3 and `~121 GiB`
  on GPUs 6/7.
- Host RAM guard stayed healthy; minimum available host RAM was about `3535 GiB`.
- Max sampled per-rank RSS was about `155-162 GiB` in the guard log.

## Verification

All verification was run through:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && ...'
```

Passed:

- Focused FP8 tests:
  `pytest -q torchtitan/experiments/glm53_flash_4gpu_fp8/tests/test_glm53_flash_4gpu_fp8.py -x`
  -> `7 passed, 14 warnings in 6.24s`.
- Existing GLM tests:
  `pytest -q torchtitan/experiments/glm53_flash/tests -x`
  -> `22 passed, 1 skipped, 14 warnings in 11.44s`.
- Syntax:
  `python -m compileall -q torchtitan/experiments/glm53_flash_4gpu_fp8`
  -> status `0`.
- Debug 1-GPU FP8 smoke:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09b/debug_train_ipv4/`
  completed 3 steps after switching torchrun rendezvous to explicit IPv4.

Not run:

- `ruff`, because the rootfs environment did not have `ruff` installed
  (`ruff: command not found` / `No module named ruff` in the earlier T09b check).

## Known Limitations

- The implementation uses dequantize-then-`F.linear`; it saves persistent
  storage but is slow. It is not an optimized FP8 compute implementation.
- `FP8FrozenExperts.forward` dequantizes selected expert weights and loops over
  active experts. This is correct enough for LoRA/activation gradients but is
  not the right final performance path.
- Full-depth 4-GPU did not meet the ticket target of `>=12` steps. It reached
  2 steps before the 45-minute cap.
- Full-width logits max/mean/top-1 drift was not produced; the single-GPU
  first-8 logits probe OOMed under current GPU pressure. Debug-depth logits
  drift is available and labeled separately.
- The FP8 loader telemetry field `bf16_source_bytes_read` is not reliable in
  the parsed first-8 artifact because the BF16 copy stats proxy can overwrite
  it; qdata/scale write counts and explicit dequant verification remain useful.

## Remaining Work for a 94-Step 4-GPU Rung

1. Replace dequantize-then-`F.linear` with a correct blockwise FP8 GEMM path
   (`torch._scaled_mm`, torchao blockwise, or a local kernel) that preserves
   LoRA and activation gradients.
2. Add a full-width logits drift artifact on Countdown prompts once GPU memory
   pressure allows it, reporting max/mean abs diff and top-1 agreement.
3. Re-run full-depth 4-GPU for at least 12 steps and collect steps 3-12 timing,
   memory, loss, and grad norm.
4. Only after 12-step stability/performance is acceptable, run the 94-step rung
   with the same seq len, tokens per step, seed, LoRA rank, BF16 source
   checkpoint, host RAM guard, and full-size lock discipline.
