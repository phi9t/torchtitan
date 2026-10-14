# T09c report: 4-GPU GLM bf16 overhead reduction

## Result

T09c is a **positive memory-fit smoke result** and a **negative readiness result**
for the full 4-GPU rung.

The split FSDP package under
`torchtitan/experiments/glm53_flash_4gpu_lean/` fit full-depth
GLM-5.3-Flash LoRA SFT on 4x B200 without offload or quantization for six
optimizer steps. Peak reserved CUDA memory was **174.7285 GiB/rank** on
178.3512 GiB B200s, with zero recorded OOMs. The fit is real but extremely
tight: step 1 had a minimum reported free memory of **0.5017 GiB** across ranks.

The run did **not** satisfy the common-rule full-depth `>=12` step target. The
bounded full-depth leg hit the required 45-minute timeout after step 6. Steps
2-6 each took about 6.3 minutes, so this mechanism is not ready for a 94-step
4-GPU rung without further throughput work.

Claim labels:

- `comparison_not_deterministic`: truncated-depth plain vs split, same seed and
  data, with tiny loss drift after step 1 from changed FSDP unit boundaries.
- `smoke`: full-depth 4-GPU split path, six optimizer steps before timeout.
- `negative_readiness`: not enough evidence or throughput for a 94-step rung.

## Environment

- `environment_class`: local bwrap rootfs `TORCHTITAN_ROOTFS_ENV=glm53_train`
- hardware: 4x NVIDIA B200 via `CUDA_VISIBLE_DEVICES=2,3,6,7`
- weights: `/data05/philip.yang/glm53-ladder-cache/model-weights/zai-org/GLM-5.3-Flash-BF16`
- cache: reused `.scratch/glm53-flash-sft-ladder/evidence/task-08b/triton_cache_prewarm/`
- allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- training shape: full-depth run used seq 512, 32768 tokens/optimizer step,
  seed 42, LoRA rank 16, bf16 frozen base, no offload, no quantization
- 8-GPU snapshot: not run; all 8 GPUs were not free while T08d/other work was
  active.

## Mechanism

New package:

- `torchtitan/experiments/glm53_flash_4gpu_lean/model.py`
- `torchtitan/experiments/glm53_flash_4gpu_lean/config_registry.py`
- `torchtitan/experiments/glm53_flash_4gpu_lean/trainer.py`
- `torchtitan/experiments/glm53_flash_4gpu_lean/memory.py`
- `torchtitan/experiments/glm53_flash_4gpu_lean/tests/`

The lean model inherits GLM forward math, state-dict names, and checkpoint
loading from `torchtitan.experiments.glm53_flash`. It only changes FSDP grouping
inside the experiment-owned package:

- layer policy: delegate to the plain GLM wrapper.
- split policy: wrap attention, attention norm/highway, FFN norm/highway,
  MoE gate, shared experts, and grouped experts as separate bottom-up FSDP
  units before wrapping the decoder layer.
- clear explicit FSDP forward/backward prefetch lists where the installed torch
  exposes those APIs.
- emit rank-local JSON plus `torch.cuda.memory_summary()` after trainer init,
  after checkpoint load, and after every train step.

Important limitation: this package can wrap the existing `GlmExperts` module,
but cannot split `gate_up_proj` and `down_proj` into separate FSDP units without
a shared-model patch. That deeper split remains a proposed follow-up rather than
a T09c edit, per the ownership rules.

## Truncated-depth measurements

Both truncated runs used first 8 full-width GLM layers, including sparse layers
3 and 7, 4 GPUs, seed 42, seq 512, and 32768 tokens/step.

Evidence:

- plain baseline:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09c/t09c-trunc-plain-20261014T143226Z/`
- split probe:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09c/t09c-trunc-split-20261014T143902Z/`

Peak CUDA memory per rank:

| Run | Static reserved after load | Step peak active | Step peak reserved | Dynamic reserved over static |
| --- | ---: | ---: | ---: | ---: |
| plain layer FSDP | 18.2676 GiB | 46.5036 GiB | 61.8164 GiB | 43.5488 GiB |
| split FSDP | 18.2676 GiB | 46.1715 GiB | 46.7422 GiB | 28.4746 GiB |

Breakdown from the truncated run:

- active dynamic peak was about 28 GiB in both paths.
- the plain path reserved an additional about 15.31 GiB above active peak.
- the split path reduced that reserved slack to about 0.57 GiB.
- reported `inactive_split_bytes` was 0.0 GiB in the rank JSON; the saved
  memory is allocator reserved high-water/prefetch/grouping behavior rather
  than inactive split blocks.

Truncated-depth numerics:

| Step | Plain loss | Split loss | Loss delta | Plain grad | Split grad | Memory delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.81229 | 10.81229 | 0.00000 | 1.8672 | 1.8672 | -15.08 GiB |
| 2 | 10.72406 | 10.72353 | -0.00053 | 1.9531 | 1.9531 | -15.08 GiB |
| 3 | 10.47525 | 10.47539 | 0.00014 | 2.4688 | 2.4688 | -15.08 GiB |

This is not labeled `deterministic_comparison`: the forward math is inherited
and unit-test bitwise equivalent on the debug model, but FSDP unit boundaries
change distributed reduction/order enough that the step loss is not bitwise
identical after step 1.

## Full-depth 4-GPU attempt

Evidence:

- failed setup attempt with invalid checkpoint retention:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09c/t09c-full-split-20261014T144342Z/`
- completed bounded attempt:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09c/t09c-full-split-20261014T144439Z/`
- TensorBoard:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09c/t09c-full-split-20261014T144439Z/tb/20261014-1444/`
- memory JSON:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09c/t09c-full-split-20261014T144439Z/lean_memory/`
- host guard:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09c/logs/host_ram_guard_t09c-full-split-20261014T144439Z.log`

The first full-depth launch used `GLM53_LEAN_KEEP_LATEST_K=1` and failed before
training because TorchTitan requires at least two checkpoint replicas. The
successful bounded launch used `GLM53_LEAN_KEEP_LATEST_K=2`.

Full-depth memory:

| Phase | Max active/allocated | Max reserved | Min free | Current rank RSS |
| --- | ---: | ---: | ---: | ---: |
| after trainer init | 145.9357 GiB | 145.9453 GiB | 31.2908 GiB | 1.8820 GiB |
| after checkpoint load | 145.9356 GiB | 145.9453 GiB | 30.7479 GiB | 1.9243 GiB |
| after step 1 | 174.4681 GiB | 174.6699 GiB | 0.5017 GiB | 3.7315 GiB |
| after step 2 | 174.5256 GiB | 174.7285 GiB | 1.8924 GiB | 3.8035 GiB |
| after step 3 | 174.5256 GiB | 174.7285 GiB | 1.8924 GiB | 3.8098 GiB |
| after step 4 | 174.5256 GiB | 174.7285 GiB | 1.8924 GiB | 3.8133 GiB |
| after step 5 | 174.5256 GiB | 174.7285 GiB | 1.8924 GiB | 3.8167 GiB |
| after step 6 | 174.5255 GiB | 174.7285 GiB | 1.8924 GiB | 3.8178 GiB |

Host memory:

- guard minimum available host RAM: 3461 GiB
- guard maximum rank RSS: 153.38 GiB during rank-local HF loading
- JSON `maxrss_gib` maximum: 153.56 GiB/rank
- steady post-load rank RSS: about 3.7-3.8 GiB/rank

Full-depth scalar curve from TensorBoard:

| Step | Loss | Grad norm | TPS | End-to-end seconds | Peak reserved | OOMs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.678362 | 3.34375 | 19.1769 | 427.1805 | 174.6699 GiB | 0 |
| 2 | 1.197009 | 2.703125 | 21.6896 | 377.6932 | 174.7266 GiB | 0 |
| 3 | 0.633783 | 2.078125 | 21.6893 | 377.6982 | 174.7285 GiB | 0 |
| 4 | 0.330455 | 2.125 | 21.3258 | 384.1364 | 174.7285 GiB | 0 |
| 5 | 0.152030 | 1.609375 | 21.0422 | 389.3122 | 174.7285 GiB | 0 |
| 6 | 0.094730 | 0.291016 | 21.5118 | 380.8144 | 174.7285 GiB | 0 |

Comparison to T08b 8-GPU one-step probe:

- T08b step 1: loss 1.67569, grad norm 3.2812, peak memory 131.56 GiB/rank.
- T09c full split step 1: loss 1.678362, grad norm 3.34375,
  peak reserved 174.6699 GiB/rank.
- T08d seed-42 comparison: unavailable; no `task-08d-report.md` was present.

The full-depth run exited via the configured `timeout 45m` with status 124. It
had written step 6 evidence before timeout. No T09c torchrun remained after the
timeout.

## Fit decision

Measured lower-bound before full-depth:

- prior full-depth 4-GPU static floor from T07b: about 146.09 GiB/rank
- measured truncated split dynamic reserved overhead:
  46.7422 - 18.2676 = 28.4746 GiB/rank
- predicted peak: about 174.56 GiB/rank
- B200 capacity in this environment: 178.3512 GiB

Observed full-depth:

- static reserved after load: 145.9453 GiB/rank
- peak reserved through step 6: 174.7285 GiB/rank
- observed dynamic reserved overhead: 28.7832 GiB/rank

Conclusion: full-depth bf16 GLM LoRA SFT can technically fit on 4x B200 with
this split FSDP package, but only with about 3.62 GiB nominal capacity margin
and as little as 0.50 GiB observed free memory at step 1. That is too little
headroom to call robust.

## Verification

Passed:

```bash
CUDA_VISIBLE_DEVICES= TORCHTITAN_ROOTFS_ENV=glm53_train \
  scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan &&
   python -m py_compile torchtitan/experiments/glm53_flash_4gpu_lean/*.py \
     torchtitan/experiments/glm53_flash_4gpu_lean/tests/*.py &&
   pytest -q torchtitan/experiments/glm53_flash_4gpu_lean/tests/test_glm53_flash_4gpu_lean.py -x'
```

Result: `5 passed, 14 warnings`.

Passed:

```bash
CUDA_VISIBLE_DEVICES=2 TORCHTITAN_ROOTFS_ENV=glm53_train \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache \
  scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'unset CC CXX; cd /workspace/torchtitan &&
   pytest -q torchtitan/experiments/glm53_flash/tests -x'
```

Result: `23 passed, 14 warnings`.

Blocked:

- `ruff format --check torchtitan/experiments/glm53_flash_4gpu_lean`
- `ruff check torchtitan/experiments/glm53_flash_4gpu_lean`

Reason: `ruff` is not installed in the `glm53_train` rootfs
(`/project/venvs/b200-runtime/bin/python: No module named ruff`).

## What remains for a 94-step 4-GPU rung

The current package is enough to prove memory feasibility, not enough to run a
useful 94-step rung.

Remaining work:

- Complete a >=12-step full-depth leg; current evidence stops at step 6 because
  the 45-minute bound fired.
- Recover throughput. At about 380 seconds/step after warmup, a 94-step run
  would take roughly 10 hours before evaluation/checkpoint overhead.
- Add more headroom. The run has only about 3.6 GiB nominal peak margin and
  observed step-1 free memory dipped to 0.5 GiB.
- Measure the prefetch-only knob separately when 4 GPUs are free. The completed
  comparison isolates the practical split package against the plain path, but
  does not separately quantify explicit-prefetch clearing.
- Consider a shared-model patch for finer expert splitting
  (`gate_up_proj` vs `down_proj`) or an expert-parallelism path. The current
  package cannot reduce the largest expert all-gather below the existing
  `GlmExperts` module boundary without editing shared GLM code.
- Revisit Triton/KDA/autotune workspace controls only after a separate snapshot
  run, because the observed active dynamic overhead is already near the
  predicted 28-29 GiB and the main remaining gap is robustness/performance.

No commit was made.
