# T18 report: GLM-5.3-Flash EP8 DeepEP dispatcher evaluation

Date: 2026-10-14

Verdict: NO-GO for switching rung 4 to DeepEP yet. TorchTitan's built-in
`DeepEPTokenDispatcher` now has an experiment-local GLM config path and passes
import/config/unit construction checks, but the GLM DeepEP training comparison
did not complete. The first DeepEP first-8-layer train attempt exposed a real
integration bug: `AssertionError: Buffer must be initialized before dispatch`.
I fixed that by initializing GLM EP expert communication buffers after the
custom GLM RNG-ordered weight initializer. The post-fix rerun was then blocked
by the live full-depth T08f run occupying all 8 GPUs, so no DeepEP loss,
grad-norm, ms/step, tokens/s, microbenchmark, or profile claim is made here.

Recommendation for rung 4: keep the validated EP8 all-to-all path from T13c.
Schedule a new exclusive or explicitly gated T18 continuation after T08f
finishes: first rerun the first-8-layer DeepEP 20-step comparison, then the
dispatch/combine microbenchmark and one profiled first-8 step, and only then
consider a full-depth DeepEP run.

## Evidence roots

- T18 paired result root:
  `experiments/glm53_flash_ep_dispatch/results/t18-glm53-deepep-20261014T2142Z/attempt-01`
- T18 summary:
  `experiments/glm53_flash_ep_dispatch/results/t18-glm53-deepep-20261014T2142Z/attempt-01/summary/t18_summary.{json,md}`
- DeepEP source/install scratch evidence:
  `.scratch/glm53-flash-sft-ladder/evidence/task-18/`
- Inherited N01b native-wheel and Qwen3 smoke evidence:
  `.scratch/glm53-flash-sft-ladder/evidence/n01b/`

## Inputs

The T18 runner used HDFS-strict staged inputs from
`/data05/philip.yang/glm53-ladder-cache`, mounted inside the rootfs at
`/project/formal-cache`.

| Input | HDFS source | manifest md5 | files | mode |
| --- | --- | --- | ---: | --- |
| Countdown SFT | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` | `66f71a353be516264fcc02fc08116ddb` | 8 | quick |
| GLM-5.3-Flash BF16 | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538` | `365857f8b1d7cb0841992fff91240ccf` | 130 | quick |

## DeepEP install and smoke status

Per the launch addendum, I did not reinstall DeepEP. I reused N01b's native L2
wheel in `glm53_ep`.

N01b wheel facts:

```text
DeepEP source: https://github.com/deepseek-ai/DeepEP.git
DeepEP commit: 01dc3aaac82068020353dce2c302e38153c0bfaa
wheel: deep_ep-2.1.0+local-cp312-cp312-linux_x86_64.whl
wheel_sha256: 7c679ee793650d7741576e2f53f95af555641bee560656725cc45d88aed154a5
rootfs_env: glm53_ep
rootfs_store_id: rootfs-0239a373638b39bca200ff29a39aa21b1737fadb267711e6b1e15572ffff5dd9
torch: 2.16.0.dev20261013+cu132
CUDA_HOME: /opt/cuda-synth
TORCH_CUDA_ARCH_LIST: 10.0
```

Current T18 rootfs metadata probe:

```text
torch 2.16.0.dev20261013+cu132
cuda 13.2
deep_ep 2.1.0+local
```

Current import/config probe passed:

```text
ElasticBuffer <class 'deep_ep.buffers.elastic.ElasticBuffer'>
dispatcher DeepEPTokenDispatcher.Config(... num_experts=288, top_k=8,
  num_max_tokens_per_rank=512, hidden_dim=4096)
ep_degree 8
```

Inherited N01b Qwen3 EP=4 smoke status: both all-to-all and DeepEP completed
two deterministic optimizer steps. Rounded console scalars were close but not
bitwise identical:

| step | all-to-all loss | DeepEP loss | loss delta | all-to-all grad norm | DeepEP grad norm | grad norm delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.72119 | 7.72118 | -0.00001 | 23.2793 | 23.2767 | -0.0026 |
| 2 | 6.89898 | 6.89897 | -0.00001 | 7.7720 | 7.7713 | -0.0007 |

That is a DeepEP training smoke pass, not a full-precision parity proof.

## Code changes

Changed files for T18:

- `torchtitan/experiments/glm53_flash_ep/model.py`
  - added optional `token_dispatcher` config plumbing;
  - passes the dispatcher config into `GlmEPMoE`;
  - initializes `GlmEPExperts` communication buffers after the custom GLM EP
    weight initializer.
- `torchtitan/experiments/glm53_flash_ep/moe.py`
  - accepts an optional `LocalTokenDispatcher.Config`;
  - keeps `AllToAllTokenDispatcher.Config` as the default, preserving the T13
    behavior.
- `torchtitan/experiments/glm53_flash_ep/config_registry.py`
  - added `glm53_flash_ep_countdown_lora_truncated_deepep()`;
  - defaults EP degree to 8 when `GLM53_FLASH_EP_DEGREE` is unset;
  - uses `DeepEPTokenDispatcher.Config(num_experts=288, top_k=8,
    num_max_tokens_per_rank=512, hidden_dim=4096)`.
- `torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py`
  - added DeepEP config factory and construction coverage.
- `torchtitan/experiments/glm53_flash_ep/dispatch_microbench.py`
  - added an experiment-local 8-rank dispatch/combine microbenchmark for
    all-to-all vs DeepEP at GLM first-8 shapes.
- `torchtitan/experiments/glm53_flash_ep/dispatch_summary.py`
  - added TensorBoard and optional microbenchmark summarization.
- `experiments/glm53_flash_ep_dispatch/run_t18_first8.sh`
  - added the rootfs runner for preflight, all-to-all, DeepEP, both,
    microbench, and summary modes.

No core TorchTitan files were edited.

## GLM all-to-all baseline completed

Run:

```text
run_id: t18-glm53-deepep-20261014T2142Z
config: glm53_flash_ep_countdown_lora_truncated
layers: first 8
EP degree: 8
seq_len: 512
tokens/optimizer step: 32768
local_batch_size: 1
steps: 20
seed: 42
deterministic: true
```

Rank 0 completed 20 steps and saved the final checkpoint. TensorBoard summary:

| run | final loss | final grad norm | mean ms/step, steps 2-20 | mean tokens/s, steps 2-20 | printed peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| T18 EP8 all-to-all first-8 | 4.166499615 | 9.617691994 | 7820.945087 | 523.8951769 | 17.30 GiB/rank |

Selected printed rank0 metrics:

| step | loss | grad norm | memory | printed tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.83415 | 1.7810 | 17.30 GiB | 224 |
| 20 | 4.16650 | 9.6177 | 17.30 GiB | 518 |

This was shared-GPU evidence. It is slower than the T13b EP8 first-8 baseline
(`5652.821390` ms/step, `728.82885` tokens/s over steps 2-20) because T18 ran
while other campaign jobs were active.

## DeepEP GLM status

Pre-patch result: the first DeepEP GLM train attempt failed before step 1 with:

```text
AssertionError: Buffer must be initialized before dispatch
```

Root cause: `Glm53FlashEPForCausalLM.init_states()` intentionally overrides the
normal module initialization path to preserve T13b's GLM RNG order. That custom
path initialized weights but bypassed child `Module.init_states()`, so
`RoutedExperts._init_self_buffers()` never initialized the DeepEP
`ElasticBuffer` before the first dispatch.

Fix: after `_initialize_glm53_flash_ep_weights(...)`,
`Glm53FlashEPForCausalLM.init_states()` now calls
`_initialize_glm53_flash_ep_buffers(...)`, which invokes
`GlmEPExperts._init_self_buffers(...)`. This keeps the T13b all-to-all RNG order
unchanged while enabling dispatcher-owned buffers.

Post-fix verification completed:

- DeepEP import/config probe passed.
- Focused GLM EP tests passed: `18 passed in 6.81s`.
- `dispatch_summary` ran successfully and reports DeepEP metrics as missing
  rather than inventing values.

Post-fix DeepEP training did not launch. The retry waited on the runner GPU
gate until the T08f full-depth EP8 ranks were active on all eight GPUs and the
minimum free memory was only `7650` MiB against the `30720` MiB threshold:

```text
2026-10-14T21:48:10Z label=train_deepep_first8 min_free_mib=7650 threshold_mib=30720
```

The parked T18 wait wrapper was terminated; no foreign process was signaled.

## Requested timing table

Rows marked `reused` come from earlier reports and are included only to keep the
T18 decision table complete. The T18-owned performance row is the all-to-all
first-8 run above.

| path | source | scope | mean ms/step | mean tokens/s | status |
| --- | --- | --- | ---: | ---: | --- |
| FSDP-only loop | T13 report | first 8, 3-step uncapped smoke | step2 `111284.2407`, step3 `109924.2096` | step2 `36.80664825`, step3 `37.26203537` | reused context, not T18-owned |
| FSDP grouped | T13b report | first 8, steps 2-20 | 29579.53493 | 139.65425 | reused context, not T18-owned |
| EP all-to-all | T13b report | first 8, steps 2-20 | 5652.821390 | 728.82885 | cleanest prior EP8 baseline |
| EP all-to-all | T18 | first 8, steps 2-20 | 7820.945087 | 523.8951769 | completed under shared load |
| EP DeepEP | T18 | first 8 | missing | missing | blocked after buffer fix by T08f GPU occupancy |

## Microbenchmark, profile, and MXFP8 status

- Dispatch/combine microbenchmark: implemented, not run. It requires the same
  8-rank GPU window and was blocked by the live T08f full-depth run.
- One profiled first-8 step for dispatch/GEMM/attention/FSDP breakdown: not
  run for the same reason.
- Full-depth DeepEP: not run, per launch addendum and T08f occupancy.
- MXFP8 grouped GEMM: not evaluated or enabled. N01b verified torchao MXFP8
  import/tiny numeric behavior, but T18 made no GLM MXFP8 change.

## Verification

All Python verification ran inside the `glm53_ep` rootfs:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_ep \
scripts/rootfs/enter_rootfs.sh \
  --rootfs-store /data05/philip.yang/torchtitan-rootfs-store -- bash -lc '
    unset CC CXX CUDAHOSTCXX
    cd /workspace/torchtitan
    python -m py_compile \
      torchtitan/experiments/glm53_flash_ep/model.py \
      torchtitan/experiments/glm53_flash_ep/moe.py \
      torchtitan/experiments/glm53_flash_ep/config_registry.py \
      torchtitan/experiments/glm53_flash_ep/dispatch_microbench.py \
      torchtitan/experiments/glm53_flash_ep/dispatch_summary.py
    pytest -q torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py -x'
```

Result:

```text
18 passed in 6.81s
```

Additional checks:

```text
bash -n experiments/glm53_flash_ep_dispatch/run_t18_first8.sh
python -m torchtitan.experiments.glm53_flash_ep.dispatch_summary ...
DeepEP import/config probe
```

All passed.

## Remaining work

1. Rerun `RUN_ID=<new-id> bash experiments/glm53_flash_ep_dispatch/run_t18_first8.sh both`
   when T08f is not occupying the 8-GPU window.
2. Require DeepEP first-8 to complete 20 steps and compare full-precision
   TensorBoard loss and grad norm against all-to-all.
3. Run `run_t18_first8.sh microbench` and the first-8 profiler pass, then fill
   in the dispatch/combine and component-time breakdown.
4. Only after the first-8 DeepEP leg passes, run any full-depth DeepEP probe.
