# T09a report: 4-GPU GLM frozen-base CPU-offload streaming

Status: PARTIAL. The T09a package was implemented and verified on focused
tests, the inherited GLM tests still pass, and the truncated-depth stream path
completed 3 optimizer steps on 4 GPUs. The required full-depth 4-GPU 12-step
tuning run was not launched because the shared full-size GLM lock was held by
another active worker command, with another worker already queued behind it.
No commit was made.

Surface: repo-local experiment package
`torchtitan/experiments/glm53_flash_4gpu_stream/`. No existing file under
`torchtitan/experiments/glm53_flash/` was edited.

## Mechanism

The stream config enables `training.enable_cpu_offload=True`, which routes
TorchTitan's GLM FSDP2 path through `CPUOffloadPolicy()`. The frozen-base local
FSDP shards are materialized on CPU, and the inherited T08b
`load_rank_local_hf_safetensors` fills only each rank's local shard directly
from safetensors slices. It does not call the DCP/HF full-model materialization
path.

One CPU-offload incompatibility appeared in the truncated probe: the base GLM
model ignored `buffer_device`, so persistent router buffers such as
`e_score_correction_bias` stayed on CPU while routed MoE scores were CUDA. The
package fixes this locally with `StreamGlm53FlashForCausalLM`, which reuses the
GLM implementation but honors `buffer_device` for buffers. Plain baseline
configs still use the original `Glm53FlashForCausalLM`.

## Code

- `torchtitan/experiments/glm53_flash_4gpu_stream/config_registry.py`
  - Adds full and truncated stream configs plus a truncated plain baseline.
  - Keeps seq length fixed at 512 and tokens/step fixed at 32768.
  - Exposes local batch size as the throughput tuning knob.
  - Supports FullAC or AC-off through `GLM53_STREAM_ACTIVATION_CHECKPOINT`.
- `torchtitan/experiments/glm53_flash_4gpu_stream/model.py`
  - Adds the stream-owned buffer-device wrapper needed for FSDP CPU offload.
- `torchtitan/experiments/glm53_flash_4gpu_stream/analysis.py`
  - Computes lower-bound H2D traffic from local frozen shard size, weight
    passes, and accumulation count.
- `torchtitan/experiments/glm53_flash_4gpu_stream/tests/`
  - Covers config defaults, truncated parity config shape, local-batch math,
    CPU-local safetensor slice copying, and H2D estimate scaling.

## Verification

Rootfs training env was used for all Python/test/training commands:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && ...'
```

Passed:

- `python -m py_compile` on the new package files.
- `pytest -q torchtitan/experiments/glm53_flash_4gpu_stream/tests/test_glm53_flash_4gpu_stream.py -x`
  - Result: `5 passed, 14 warnings`.
- `pytest -q torchtitan/experiments/glm53_flash/tests -x`
  - Result after the final stream-wrapper patch: `22 passed, 1 skipped, 14 warnings`.
- Selected hooks on new package files:
  - `trailing-whitespace`
  - `end-of-file-fixer`
  - `check-ast`
  - `check-merge-conflict`
  - `ufmt`
  - `flake8`
  - `pydoclint`
  - `codespell`

Full repo pre-commit was not run because `env-facts.md` says repo-wide pyrefly
currently fails on pre-existing errors and can auto-edit unrelated files.

## Config Evidence

Config prints:

- `.scratch/glm53-flash-sft-ladder/evidence/task-09a/configs/full_stream_lb4_print.log`
- `.scratch/glm53-flash-sft-ladder/evidence/task-09a/configs/truncated_stream_lb2_noac_print.log`
- `.scratch/glm53-flash-sft-ladder/evidence/task-09a/configs/truncated_plain_lb2_noac_print.log`

Important resolved values:

| Config | CPU offload | AC | local batch | seq | tokens/step | steps |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| full stream lb4 | yes | FullAC | 4 | 512 | 32768 | 12 |
| truncated stream lb2 | yes | off | 2 | 512 | 32768 | 3 |
| truncated plain lb2 | no | off | 2 | 512 | 32768 | 3 |

## Truncated-Depth Evidence

Scope: first 8 full-width layers, including dense layers 0-2 and sparse MoE
layers 3-7. Environment class: `shared_gpu`, with other workers/jobs present on
the host.

Accepted artifacts:

- Plain baseline log:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09a/logs/truncated_plain_lb2_noac_attempt02.log`
- Stream log:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09a/logs/truncated_stream_lb2_noac_attempt02.log`
- Host RSS guards:
  `.scratch/glm53-flash-sft-ladder/evidence/task-09a/logs/host_ram_guard_truncated_plain_lb2_noac_attempt02.log`
  `.scratch/glm53-flash-sft-ladder/evidence/task-09a/logs/host_ram_guard_truncated_stream_lb2_noac_attempt02.log`

Loader evidence:

| Path | tensors loaded | LoRA skipped | bytes loaded/rank | load time | verified keys |
| --- | ---: | ---: | ---: | ---: | --- |
| plain | 229 | 146 | 19,587,436,096 | 11.28s | embedding, q_proj, sparse wk, grouped experts |
| stream | 229 | 146 | 19,587,436,096 | 11.62s | embedding, q_proj, sparse wk, grouped experts |

Memory evidence:

| Path | model CUDA memory | peak logged CUDA memory | max sampled rank RSS |
| --- | ---: | ---: | ---: |
| plain | 18.26 GiB | 83.74 GiB | 3.50 GiB |
| stream | 0.00 GiB | 68.05 GiB | 47.07 GiB |

The stream path moves the truncated frozen shard from GPU to host and still
loads only rank-local slices. It did not show the T08 full-model host
materialization pattern.

Numerics, 3 optimizer steps:

| Step | plain loss | stream loss | loss delta | plain grad norm | stream grad norm | grad delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.79574 | 10.79574 | 0.00000 | 1.8828 | 1.7734 | -0.1094 |
| 2 | 10.70416 | 10.70815 | +0.00399 | 1.9844 | 1.8672 | -0.1172 |
| 3 | 10.44788 | 10.45599 | +0.00811 | 2.5312 | 2.4062 | -0.1250 |

Claim label: `smoke` plus `numerical_comparison`. This is not bitwise
deterministic evidence. CPU offload itself should move bytes rather than change
math, but this run used default non-deterministic bf16/distributed execution and
printed metric precision only. A stricter parity gate should enable deterministic
settings and compare full-precision TensorBoard or `loss_compare.py` values.

## H2D Estimate

The estimate is a lower bound: local frozen-base shard GiB times two weight
passes per microbatch times gradient accumulation steps. Actual FSDP scheduling,
prefetch, buffers, and runtime overhead can add traffic.

Full-depth 4-GPU lower-bound estimates:

| local batch | accumulation steps | frozen shard GiB/rank | H2D GiB/rank/step |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 145.904 | 4668.933 |
| 2 | 8 | 145.904 | 2334.467 |
| 4 | 4 | 145.904 | 1167.233 |
| 8 | 2 | 145.904 | 583.617 |
| 16 | 1 | 145.904 | 291.808 |

Truncated stream lb2 used an 18.245 GiB/rank local frozen shard. With 8
accumulation steps, the lower-bound H2D is about 291.9 GiB/rank/step. Using
the logged step timestamps for steps 2-3, observed lower-bound bandwidth was
about 5.8 GiB/s per rank, or about 23 GiB/s aggregate across four ranks. This
is `shared_gpu` evidence only.

## Full-Depth Status

Not run. At the time of the full-depth attempt decision:

- `/data05/philip.yang/glm53-ladder-cache/glm-fullsize.lock` was held.
- `lslocks` showed an active holder for T09c and a second queued T09b holder.
- GPUs 2, 3, 6, and 7 were effectively full; GPUs 0, 1, 4, and 5 became free
  briefly and then picked up other work again.
- There was no live `torchtitan.experiments.glm53_flash.config_registry`
  training process after filtering out the lock-holder command text, but the
  lock itself was actively held by another worker process. I did not kill,
  signal, bypass, or clear another worker's lock.

Therefore no full-depth 4-GPU >=12-step claim, no FullAC-vs-AC-off tuning
claim, no full-depth H2D achieved measurement, and no best full-depth setting
are reported for T09a.

## What Remains

- Launch full-depth 4-GPU stream runs once the GLM full-size lock is free:
  local batch 4 first, then try local batch 8/16 if GPU memory leaves enough
  headroom; compare AC off and FullAC.
- Record peak GPU memory per rank, host RSS per rank, steps 3-12 step time,
  loss/grad curve, and lower-bound H2D bandwidth.
- Compare full-depth step-1 loss against T08b's 8-GPU probe (`1.67569`) and
  T08d seed-42 if available.
- Run a deterministic truncated-depth parity gate if a `deterministic_comparison`
  claim is needed.
