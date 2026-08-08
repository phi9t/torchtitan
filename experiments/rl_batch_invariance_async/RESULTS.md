# RL Batch-Invariance Async Results

Status: rootfs-gated reference run stopped before the BI arm completed. The
no-BI arm completed 150/150 train steps. The BI arm reached one confirmed
console train step, then the Monarch root actor reported a SIGTERM-backed actor
failure. Do not treat this run as a completed 150-step BI comparison.

## Rootfs Run

Command:

```bash
OUT_ROOT=outputs/rl_batch_invariance_async_rootfs \
experiments/rl_batch_invariance_async/run_reference.sh
```

The runner re-entered `scripts/rootfs/enter_rootfs.sh`, set
`TORCHTITAN_IN_ROOTFS=1`, created or reused `.venv-rootfs`, and ran setup,
preflight, training, and summarization through `.venv-rootfs/bin/python`.

Environment snapshot:

- Date: `2026-08-07T18:00:00Z`
- Git commit: `d79766701e9e852624be1c762f8b3f57aad11eee`
- Rootfs path: `scripts/rootfs/rootfs`
- Marker: `TORCHTITAN_IN_ROOTFS=1`
- Virtualenv: `/workspace/torchtitan/.venv-rootfs`
- Python: `/workspace/torchtitan/.venv-rootfs/bin/python`, `Python 3.12.3`
- Torch: `2.14.0.dev20260805+cu130`
- vLLM: `0.1.dev1+gd6938b740.d20260806`
- `batch_invariant_ops`: `0.1.0`
- Output root: `outputs/rl_batch_invariance_async_rootfs`
- Full snapshots:
  - `outputs/rl_batch_invariance_async_rootfs/env/environment.txt`
  - `outputs/rl_batch_invariance_async_rootfs/env/nvidia-smi.txt`
  - `outputs/rl_batch_invariance_async_rootfs/env/python_packages.txt`

## Topology

| Arm | Config | Trainer | Generators | Off-policy target | Attention | BI debug |
| --- | --- | --- | --- | ---: | --- | --- |
| no_bi | `rl_dapo_qwen3_4b_math_8k_reference_no_bi` | TP=2 | 6 x TP=1 | 4 | varlen | disabled |
| bi | `rl_dapo_qwen3_4b_math_8k_reference_bi` | TP=2 | 3 x TP=2 | 0 | flex | `batch_invariant=True`, `deterministic=True` |

The BI arm intentionally uses matched trainer/generator tensor parallelism and
strict on-policy rollout because this checkout only supports bitwise logprob
parity under those constraints.

## Completion Status

| Arm | Confirmed console train steps | Final state | Primary log |
| --- | ---: | --- | --- |
| no_bi | 150/150 | completed full reference arm | `outputs/rl_batch_invariance_async_rootfs/no_bi/run_150_steps.log` |
| bi | 1/150 | stopped after Monarch actor failure | `outputs/rl_batch_invariance_async_rootfs/bi/run_150_steps.log` |

No new job was launched after the stop. The analysis below uses the captured
trace and the generated summary artifacts only.

## Console Metrics

These values come from `Train | Step:` console records in the full arm logs.

| Metric | no_bi step 1 | no_bi step 150 | bi step 1 |
| --- | ---: | ---: | ---: |
| `bit_wise/logprob_diff/max` | 0.32 | 0.47 | 0 |
| `rollout_reward/_mean` | 0.080 | 0.46 | 0.052 |
| `perf/trainer/tokens_per_second_full_step` | 169.7 | 6690.1 | 25.87 |
| `perf/trainer/tokens_per_second_fwd_bwd` | 5887.1 | 12621.7 | 2923.6 |
| `generator/inter_token_latency_ms/mean` | 70.92 | 77.82 | 124.7 |
| `generator/decode_time_ms/mean` | 53109.0 | 237703.6 | 110128.1 |
| `generator/queue_time_ms/mean` | 72.41 | 2116.4 | 1.81 |

Validation console records:

| Arm | Validation step | Response length mean | Timing |
| --- | ---: | ---: | ---: |
| no_bi | 0 | 853.1 | 133.5 |
| no_bi | 150 | 5033.1 | 585.7 |
| bi | 0 | 1117.2 | 1055.1 |

## Summary Artifacts

The runner wrote:

- `experiments/rl_batch_invariance_async/summary.md`
- `experiments/rl_batch_invariance_async/summary.csv`
- `experiments/rl_batch_invariance_async/summary.json`

Interpret the BI rows in these files with care. The summarizer combines
TensorBoard events and console logs, deduplicates by `(step, metric)` with the
first observation winning, and scans both full arm directories and
`preflight_<arm>` directories. As a result, the BI summary reports three metric
observations even though the full BI console log contains only one confirmed
`Train | Step:` line.

The aggregate summary still gives useful supporting evidence:

- no-BI `bit_wise/logprob_diff/max`: first `0.323327 @ 1`, last
  `0.465602 @ 150`, min `0.0676317`, max `2.72901`, mean `0.698053`,
  count `152`.
- no-BI `rollout_reward/_mean`: first `0.0803571 @ 1`, last
  `0.4625 @ 150`, mean `0.332471`, count `152`.
- no-BI `validation_reward/_mean`: first `0.0333333 @ 0`, last
  `0.266667 @ 150`.
- BI `bit_wise/logprob_diff/max`: first `0 @ 1`, last `0.0755811 @ 2`,
  min `0`, max `0.11875`, mean `0.0647769`, count `3`.
- BI `rollout_reward/_mean`: first `0.0520833 @ 1`, last `0 @ 2`,
  mean `0.0173611`, count `3`.

Only the BI step-1 console metric should be used for a full-run parity claim
from this stopped attempt.

## Stop Trace

The BI log reports the stop as a Monarch actor failure, not as a Python
exception in the trainer loop:

```text
Unhandled monarch error on the root actor, hostname=n116-077-207, PID=19581 at time 2026-08-08 04:30:02.710087:
The actor logger-dpTS2Qh7THp{'gpus': 0/2} and all its descendants have failed:
  the process this actor was running on failed: Killed(sig=15)
[actor=<root>] Interrupted; attempting graceful shutdown...
[actor=<root>] trainer.close failed
[actor=<root>] generator[0].close failed
[actor=<root>] generator[1].close failed
[actor=<root>] generator[2].close failed
```

The close failures were `monarch_hyperactor.supervision.SupervisionError`
reports because the trainer and generator actor processes had already been
killed with `SIGTERM`. The repeated Python `resource_tracker` warnings about
leaked semaphores and shared memory objects appeared immediately before the
Monarch error and are consistent with forced actor shutdown.

## Interpretation

The completed no-BI arm provides the 150-step reference trace. Its logprob
parity metric stayed nonzero: `bit_wise/logprob_diff/max` was `0.32` at step 1
and `0.47` at step 150, with the generated summary reporting a run mean of
`0.698053`.

The stopped BI arm confirms the intended first-step parity condition on the
full rootfs path: its only confirmed console train record had
`bit_wise/logprob_diff/max: 0`. That is evidence that the BI/flex attention
path reached clean first-step logprob parity after the B200 attention fixes.
It is not evidence that BI maintained parity over 150 steps, improved reward,
or established a stable throughput cost over the full run.

Reward comparison is inconclusive. The no-BI arm ended at
`rollout_reward/_mean: 0.46`; the BI arm stopped with only step-1
`rollout_reward/_mean: 0.052`.

Throughput comparison is also inconclusive as a 150-step statement. At the
only confirmed comparable first step, BI had lower full-step throughput
(`25.87` tokens/s versus no-BI `169.7`) and higher generator inter-token
latency (`124.7` ms versus `70.92` ms), but the BI arm did not run long enough
to separate startup, validation, rollout length, and steady-state effects.
