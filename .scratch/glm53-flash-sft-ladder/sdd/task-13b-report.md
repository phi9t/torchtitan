# T13b report: GLM EP good-state pass

Date: 2026-10-14

Verdict: EP is in a good state for the first-8-layer GLM-5.3-Flash Countdown
LoRA ladder slice. The original T13 grad-norm gap was a real gradient mismatch,
not a core `clip_grad_norm_` accounting bug. It is fixed in the EP package by
making EP initialization consume RNG in the same order as the plain GLM model.

This is still not a final scale/performance claim. The required full-depth EP8
timing probe was skipped because the live GPU free-memory gate failed. DeepEP
and native communication work remain for N01/T18.

## Evidence root

`.scratch/glm53-flash-sft-ladder/evidence/task-13b/t13b-20261014T192730Z`

Key artifacts:

- `grad_probe_comparison.json`: pre-fix full-gradient probe comparison.
- `grad_probe_comparison_postfix.json`: post-fix EP8 full-gradient comparison.
- `task13b_metrics_summary.json`: TensorBoard loss, grad-norm, speed, and
  memory summary.
- `logs/hdfs_strict_inputs.log`: strict staged-input provenance.
- `full_depth_gpu_gate_20261014T2026Z.csv`: full-depth gate snapshot.

## Inputs

All training/probe runs used HDFS-staged inputs through
`/data05/philip.yang/glm53-ladder-cache`, mounted in rootfs as
`/project/formal-cache`.

Strict input check:

| Input | HDFS source | manifest md5 | files | mode |
| --- | --- | --- | ---: | --- |
| Countdown SFT | `hdfs://harunava/user/tiger/waystone/torchtitan/datasets/countdown-sft/v1` | `66f71a353be516264fcc02fc08116ddb` | 8 | quick |
| GLM-5.3-Flash BF16 | `hdfs://harunava/user/tiger/waystone/torchtitan/model-weights/zai-org/GLM-5.3-Flash-BF16/a5b45eb41df6402735dedc900be14a42e8d5e538` | `365857f8b1d7cb0841992fff91240ccf` | 130 | quick |

## Root Cause

The one-step full-gradient probes falsified the metric-bug hypothesis:

- The offline full fp32 gradient norm from gathered CPU tensors matched the
  logged scalar gap.
- EP1, EP4, and EP8 all had the same pre-fix high norm, so the issue was not
  sparse-mesh norm counting, EP all-to-all degree, `Partial` vs `Replicate`
  reduction, or double counting in `clip_grad_norm_`.
- Plain loop and plain grouped matched each other at BF16-scale tolerance, so
  grouped expert math itself was not the cause.

The actual cause was EP initialization order. Plain GLM initializes sparse-layer
expert/router weights before shared-expert LoRA modules. The EP model had
replaced the sparse modules, then called the plain initializer and initialized
EP-specific expert/router weights late. Because `lora_b` starts at zero, the
first forward loss still matched, but the first `lora_b` gradient depends on
the random `lora_a` values. Consuming RNG in a different order changed those
adapter gradients and moved the global grad norm.

The fix keeps EP initialization experiment-local and mirrors the plain GLM
initializer order, including LoRA, dense, embedding, norm, router, expert,
index-pool, and KDA modules.

## Code Changes

Changed files:

- `torchtitan/experiments/glm53_flash_ep/model.py`
  - replaced the old plain-init-plus-late-EP-extra flow with an EP-aware
    initializer that preserves plain GLM RNG traversal/order.
- `torchtitan/experiments/glm53_flash_ep/grad_probe.py`
  - added a one-step diagnostic entrypoint that captures full gathered CPU fp32
    gradients, offline L2 norm, and TorchTitan's no-clip grad norm.
- `torchtitan/experiments/glm53_flash_ep/parity_report.py`
  - pointed gradient comparison at the EP-local probe helper.
- `torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py`
  - added regression coverage for plain-vs-EP LoRA RNG order and gradient-probe
    summary math.

No core grad-norm patch is proposed because the gathered-gradient evidence did
not show a core norm bug.

## Gradient Probe Results

Reference for all rows: FSDP-only plus grouped experts.

Pre-fix:

| Candidate | core grad-norm delta | offline grad-norm delta | max gradient delta | differing params |
| --- | ---: | ---: | ---: | ---: |
| plain loop | 0 | 0.00008480795347 | 0.000244140625 | 75 |
| EP1 | 0.0625 | 0.06874639029 | 0.055908203125 | 75 |
| EP4 | 0.06076860428 | 0.06874639029 | 0.055908203125 | 75 |
| EP8 | 0.06076860428 | 0.06874639029 | 0.055908203125 | 75 |

Post-fix EP8:

| Candidate | core grad-norm delta | offline grad-norm delta | max gradient delta | differing params |
| --- | ---: | ---: | ---: | ---: |
| EP8 fixed | 0.0006651878357 | 0.000007412342242 | 0.000244140625 | 72 |

The post-fix offline norm moved from `1.8550615560953554` to
`1.7863225781484406`, matching the grouped reference
`1.7863151658061982` to BF16-scale tolerance.

## Fair Speed Baselines

Config: first 8 full-width layers, seq 512, 32,768 tokens/optimizer step,
deterministic seed 42, 10 steps, shared GPUs.

These are useful directionality numbers, not promotion-grade exclusive
performance evidence.

| Run | mean ms/step, steps 2-10 | mean tokens/s, steps 2-10 | final loss | final grad norm | peak reserved GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain grouped | 35756.39598 | 114.64255 | 7.513541698 | 5.03125 | 66.9765625 |
| EP4 fixed | 7983.433565 | 513.56882 | 7.513078213 | 5.037360191 | 23.375 |
| EP8 fixed | 6980.597178 | 587.16812 | 7.513078213 | 5.037360191 | 17.3046875 |

## 20-Step Numerics

Comparison: plain grouped vs fixed EP8, same config and seed.

Scalar deltas from TensorBoard full-precision values:

- max `|loss delta|`: `0.0023818016052246094`
- final `loss delta`: `-0.0004305839538574219`
- max `|grad-norm delta|`: `0.014719009399414062`
- final `grad-norm delta`: `-0.0032052993774414062`

| Step | plain loss | EP8 loss | loss delta | plain grad norm | EP8 grad norm | grad-norm delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.812238693 | 10.812238693 | 0 | 1.7890625 | 1.789727688 | 0.000665188 |
| 2 | 10.789027214 | 10.788232803 | -0.000794411 | 1.8125 | 1.811422110 | -0.001077890 |
| 3 | 10.590026855 | 10.590385437 | 0.000358582 | 2.265625 | 2.263846397 | -0.001778603 |
| 4 | 10.275610924 | 10.275522232 | -0.000088692 | 2.859375 | 2.861380816 | 0.002005816 |
| 5 | 9.919642448 | 9.919252396 | -0.000390053 | 3.484375 | 3.482097149 | -0.002277851 |
| 6 | 9.520675659 | 9.520909309 | 0.000233650 | 4.0625 | 4.062019348 | -0.000480652 |
| 7 | 9.072789192 | 9.072346687 | -0.000442505 | 4.4375 | 4.444097042 | 0.006597042 |
| 8 | 8.610684395 | 8.610392570 | -0.000291824 | 4.625 | 4.623310566 | -0.001689434 |
| 9 | 8.151907921 | 8.151750565 | -0.000157356 | 4.78125 | 4.782781601 | 0.001531601 |
| 10 | 7.686886311 | 7.686996937 | 0.000110626 | 5 | 5 | 0 |
| 11 | 7.222612381 | 7.222310066 | -0.000302315 | 5.125 | 5.123475552 | -0.001524448 |
| 12 | 6.776378632 | 6.776194096 | -0.000184536 | 5.1875 | 5.184110165 | -0.003389835 |
| 13 | 6.364573002 | 6.365211964 | 0.000638962 | 5.4375 | 5.442885399 | 0.005385399 |
| 14 | 5.965328693 | 5.965392113 | 0.000063419 | 5.84375 | 5.852349758 | 0.008599758 |
| 15 | 5.580036163 | 5.580075264 | 0.000039101 | 6.4375 | 6.442049503 | 0.004549503 |
| 16 | 5.225470066 | 5.225346088 | -0.000123978 | 7.40625 | 7.399324417 | -0.006925583 |
| 17 | 4.916821003 | 4.915642262 | -0.001178741 | 8.5 | 8.485280991 | -0.014719009 |
| 18 | 4.665769577 | 4.668151379 | 0.002381802 | 9.9375 | 9.949873924 | 0.012373924 |
| 19 | 4.409689426 | 4.409529209 | -0.000160217 | 10.5 | 10.488088608 | -0.011911392 |
| 20 | 4.109737873 | 4.109307289 | -0.000430584 | 9.75 | 9.746794701 | -0.003205299 |

20-step throughput and memory:

| Run | mean ms/step, steps 2-20 | mean tokens/s, steps 2-20 | final loss | final grad norm | peak reserved GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain grouped | 29579.53493 | 139.65425 | 4.109737873 | 9.75 | 66.9765625 |
| EP8 fixed | 5652.821390 | 728.82885 | 4.109307289 | 9.746794701 | 17.3046875 |

## Full-Depth EP8 Timing Probe

Skipped per the ticket gate. A fresh `nvidia-smi` check showed GPUs 1 and 7
below the required 150 GiB free threshold:

| GPU | free MiB | total MiB |
| ---: | ---: | ---: |
| 0 | 167278 | 183359 |
| 1 | 108422 | 183359 |
| 2 | 170000 | 183359 |
| 3 | 159814 | 183359 |
| 4 | 167278 | 183359 |
| 5 | 159814 | 183359 |
| 6 | 182628 | 183359 |
| 7 | 108422 | 183359 |

No full-depth 45-layer timing, loader-locality, peak-memory, or host-RSS claim
is made in this report.

## T09a and T09b

The same mechanism does not directly explain T09a or T09b based on this
evidence. The T13b bug is specific to the EP model substituting sparse expert
modules and then initializing EP expert/router tensors after the plain
initializer had already consumed RNG for later LoRA modules.

T09a's CPU-offload stream path and T09b's FP8 storage path are different
experiment packages and do not use this EP initialization flow. Their reported
grad-norm gaps remain plausible package-specific numerical/storage/offload
effects until they get the same style of full-gradient probe.

## Verification

All Python/training verification ran inside the rootfs training environment:

```bash
TORCHTITAN_ROOTFS_ENV=glm53_train \
TORCHTITAN_ROOTFS_FORMAL_CACHE_HOST=/data05/philip.yang/glm53-ladder-cache \
scripts/rootfs/enter_rootfs.sh -- bash -lc 'unset CC CXX; cd /workspace/torchtitan && ...'
```

Passed:

```bash
python -m py_compile \
  torchtitan/experiments/glm53_flash_ep/model.py \
  torchtitan/experiments/glm53_flash_ep/grad_probe.py \
  torchtitan/experiments/glm53_flash_ep/parity_report.py \
  torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py

pytest -q torchtitan/experiments/glm53_flash_ep/tests/test_glm53_flash_ep.py -x
```

Result: `16 passed, 14 warnings`.

Narrow pre-commit hooks on the changed EP files passed:

- `trailing-whitespace`
- `end-of-file-fixer`
- `check-ast`
- `check-merge-conflict`
- `check-added-large-files`
- `insert-license`
- `flake8`
- `ufmt`
- `pydoclint`
- `codespell`
- `check-no-pii`

Repo-wide `pyrefly` and full pre-commit were not run, per `env-facts.md`:
repo-wide pyrefly has pre-existing failures and can auto-edit unrelated files.

## Remaining Work

- Run the 45-layer EP8 timing/memory probe once all eight GPUs have at least
  150 GiB free.
- Keep the existing FSDP2 view warnings on the follow-up list. They were present
  before this fix and were not implicated by the gradient probes.
- Continue DeepEP/native communication work through N01/T18 before treating EP8
  as a final throughput path.
