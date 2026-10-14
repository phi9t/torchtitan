# T13 report: GLM EP first-8-layer full-width Countdown LoRA parity

Date: 2026-10-14

Claim label: `smoke_plus_bounded_numerical_delta`

The EP4 truncated-depth GLM-5.3-Flash Countdown LoRA config now runs 3 optimizer steps on 8 B200 GPUs with first-8-layer full-width weights, seq length 512, BF16, deterministic seed 42, 32,768 tokens per optimizer step, and LoRA rank 16. The successful EP run used expert-parallel sharding plus grouped expert compute and stayed under the addendum cap at 21.58 GiB reserved per rank.

This is not a deterministic or bitwise parity claim. The EP path uses grouped expert compute and token-dispatch ordering, while the plain baseline uses the current local expert representation. Gradient probe snapshots were not produced, so parameter-gradient tensor deltas are unavailable; only scalar grad-norm deltas and final checkpoint parameter deltas are reported.

## Scope and implementation

Edited only the EP package plus this report:

- `torchtitan/experiments/glm53_flash_ep/config_registry.py`
  - Added `glm53_flash_ep_countdown_lora_truncated`.
  - Uses full GLM-5.3-Flash width, layers 0-7, rank-16 LoRA, BF16, seq 512, 32,768 tokens/step, seed 42, deterministic default, EP degree from `GLM53_FLASH_EP_DEGREE`, TensorBoard metrics, the EP LoRA checkpointer, and `LeanMemoryTrainer`.
- `torchtitan/experiments/glm53_flash_ep/flavors.py`
  - Registered the full `glm53_flash_ep` flavor used by the truncated config.
- `torchtitan/experiments/glm53_flash_ep/model.py`
  - Added an EP-specific FSDP override that forwards `DataParallelMeshDims` and sparse mesh dims for SPMD expert parameters.
- `torchtitan/experiments/glm53_flash_ep/moe.py`
  - Added distinct grouped-linear region labels for `w13` and `w2` so remat fallback names do not collide.
- `torchtitan/experiments/glm53_flash_ep/parity_report.py`
  - Added TensorBoard, memory, checkpoint-parameter, and optional grad-probe comparison artifact generation.
- `torchtitan/experiments/glm53_flash_ep/sitecustomize.py`
  - Added opt-in CUDA per-process memory cap hook used only when its directory is prepended to `PYTHONPATH` and `TORCHTITAN_CUDA_MEMORY_CAP_GIB` is set.
- `torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py`
  - Added tests for the T13 config contract, parity-report helpers, and distinct grouped remat names.

No files under `torchtitan/experiments/glm53_flash/` or `torchtitan/models/common/` were edited.

## Inputs

Strict HDFS staged roots were used through the local cache at `/data05/philip.yang/glm53-ladder-cache`, mounted in rootfs as `/project/formal-cache`.

`logs/hdfs_strict_inputs.log` records:

- Countdown SFT data:
  - source: `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1`
  - manifest md5: `66f71a353be516264fcc02fc08116ddb`
  - files: 8
  - mode: `quick`, using the helper's cached md5 validation by file mtime and size.
- GLM-5.3-Flash BF16 weights/tokenizer:
  - source: `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538`
  - manifest md5: `365857f8b1d7cb0841992fff91240ccf`
  - files: 130
  - mode: `quick`, using the helper's cached md5 validation by file mtime and size.

A full refresh was started after the runs. It completed the dataset root and was stopped during the large GLM checkpoint hash pass to avoid blocking the report; the completed quick strict log above is the input evidence used for this ticket.

## Environment

Rootfs/training environment:

- `TORCHTITAN_ROOTFS_ENV=glm53_train`
- `unset CC CXX`
- `TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache`
- training launched from `/workspace/torchtitan`

GPU environment:

- Initial addendum-era snapshot: T08d/fallback full-depth GLM was live on all 8 B200s with roughly 136-148 GiB already used per GPU. The initial baseline attempt was therefore run under a 30 GiB/process cap and failed with CUDA OOM.
- Final EP evidence: `logs/nvidia-smi-before-ep4_rank01_cache_retry.txt` showed the high-memory T08d job gone; GPUs 0-2 still had small repo-local jobs, so this is `shared_gpu`, not exclusive performance proof. EP was still run with `TORCHTITAN_CUDA_MEMORY_CAP_GIB=30`.
- Final plain baseline evidence: `logs/nvidia-smi-before-plain_uncapped_cache_retry.txt` showed the high-memory T08d job gone; GPUs 0-2 had repo-local jobs using about 9-12 GiB. The plain baseline was run uncapped because the capped baseline already proved it cannot fit under 30 GiB.

## Runs

Evidence root:

`.scratch/glm53-flash-sft-ladder/evidence/task-13/t13-glm53-ep-truncated-20261014T184711Z`

Important logs:

- `logs/plain.log`: capped plain FSDP-only baseline failed with CUDA OOM before step 1.
- `logs/ep4.log`: first EP attempt failed during FSDP init with missing SPMD `dp_mesh_dims`.
- `logs/ep4_retry.log`: second EP attempt failed on duplicate remat region name.
- `logs/ep4_retry2.log`: rank-filtered retry hid the nonzero rank traceback.
- `logs/ep4_rank01_cache_retry.log`: final capped EP4 run, 3 optimizer steps completed.
- `logs/plain_uncapped_cache_retry.log`: final uncapped plain baseline run, 3 optimizer steps completed.
- `logs/parity_report.log`: comparison artifact generation.
- `logs/preflight_post_gpu.log`: post-GPU focused preflight.
- `logs/precommit_ep_files.log`: narrow pre-commit hooks.

Run directories:

- Plain baseline: `plain_uncapped_cache_retry`
- EP4: `ep4_rank01_cache_retry`
- Comparison artifacts: `parity.json`, `parity.md`

## Numerics

From `parity.md`:

| step | plain loss | EP loss | loss delta | plain grad norm | EP grad norm | grad norm delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.81232166 | 10.81223869 | -0.00008296966553 | 1.7890625 | 1.849831104 | 0.06076860428 |
| 2 | 10.72580910 | 10.72077274 | -0.005036354065 | 1.8750000 | 1.952562451 | 0.07756245136 |
| 3 | 10.47611618 | 10.46335316 | -0.01276302338 | 2.3906250 | 2.487468481 | 0.09684348106 |

Scalar maxima:

- max `|loss delta|`: `0.01276302338`
- max `|grad_norm delta|`: `0.09684348106`
- final LoRA checkpoint max `|parameter delta|`: `0.076904296875`
- max `|parameter-gradient tensor delta|`: unavailable, because `grad_probe` rank snapshots were not produced.

The scalar trajectories are close enough for a smoke plus bounded-delta claim, but not enough for deterministic numerical identity.

## Memory and throughput

From `parity.md` and run logs:

| run | cap | max reserved GiB/rank | max allocated GiB/rank | step 1 ms | step 2 ms | step 3 ms | step 1 tok/s | step 2 tok/s | step 3 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain FSDP-only | uncapped final run | 67.1171875 | 37.37747192 | 114456.2073 | 111284.2407 | 109924.2096 | 35.78661346 | 36.80664825 | 37.26203537 |
| EP4 | 30 GiB/process | 21.58203125 | 20.24496984 | 22062.17384 | 5383.858204 | 5297.014713 | 185.657135 | 760.7927246 | 773.2657471 |

Throughput deltas are large, but this is still `shared_gpu` evidence rather than a clean matched performance proof. The final EP run did satisfy the 30 GiB/process cap; the plain baseline did not fit under that cap and required the later uncapped rerun after the high-memory fallback job was gone.

## Verification

Passed inside `glm53_train` rootfs:

```bash
pytest -q torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py -x
```

Result: `14 passed, 14 warnings`.

```bash
python -m torchtitan.experiments.glm53_flash_ep.validate_ep_matrix --world-size 8 --ep 1 --ep 2 --ep 4 --ep 8
```

Result: passed for debug and mini configs at EP degrees 1, 2, 4, and 8.

```bash
GLM53_FLASH_EP_DEGREE=4 NGPU=8 COMM_BACKEND=fake \
  MODULE=torchtitan.experiments.glm53_flash_ep.config_registry \
  CONFIG=glm53_flash_ep_countdown_lora_truncated \
  ./run_train.sh --print-config
```

Result: passed; config printed to `/tmp/t13_glm53_ep_truncated_print_config.txt` inside rootfs.

Narrow pre-commit hooks on changed EP files:

```bash
pre-commit run ufmt --files ...
pre-commit run flake8 --files ...
pre-commit run pydoclint --files ...
pre-commit run codespell --files ...
```

Result: all passed. Repo-wide `pyrefly` and full pre-commit were not run because `env-facts.md` says repo-wide pyrefly has pre-existing failures and may auto-edit unrelated files.

## Notes and follow-ups

- The final capped EP run is the useful T13 result: 3 steps completed at 21.58 GiB reserved/rank, with steady steps around 5.3 seconds and 760-773 tokens/s.
- The plain baseline cannot co-run under the 30 GiB cap. The capped attempt failed on FSDP all-gather with `torch.OutOfMemoryError: Tried to allocate 9.00 GiB ... 30.00 GiB allowed`.
- FSDP2 view warnings appeared in both plain and EP runs. A follow-up should audit the shared GLM forward path for view outputs around FSDP-wrapped decoder/model/norm/linear modules and use out-of-place ops or clones where required. This was not changed here because the ticket forbids editing `torchtitan/experiments/glm53_flash/`.
- A future deterministic comparison should add a true one-step gradient probe for both the plain and EP packages, save rank-local LoRA gradients before optimizer step, and compare those snapshots. This report intentionally does not claim parameter-gradient tensor parity.
- A future clean performance run should reserve the GPUs for just the two matched jobs, keep cap policy identical where possible, and report `exclusive_foreign_free` or equivalent. This ticket's timing is useful for directionality but remains shared-GPU evidence.
