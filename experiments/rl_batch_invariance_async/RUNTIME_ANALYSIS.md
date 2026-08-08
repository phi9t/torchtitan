# Runtime Analysis

This note records the runtime shape of the rootfs-gated 150-step reference
attempt. The no-BI arm completed 150 steps; the BI arm reached one confirmed
console train step and then stopped after a Monarch actor failure reported
`Killed(sig=15)`.

## Run Shape

The full command path was:

```text
/workspace/torchtitan/.venv-rootfs/bin/python -m torchtitan.experiments.rl.train \
  --module dapo_math \
  --config <reference arm config> \
  --dump-folder outputs/rl_batch_invariance_async_rootfs/<arm> \
  --async-loop.num-training-steps 150
```

The no-BI arm used `rl_dapo_qwen3_4b_math_8k_reference_no_bi` and completed
150/150 console train steps. The BI arm used
`rl_dapo_qwen3_4b_math_8k_reference_bi` and has exactly one confirmed full-arm
console train step in `outputs/rl_batch_invariance_async_rootfs/bi/run_150_steps.log`.

## Why It Was Slow

The run was rollout-generation bound, not primarily trainer-compute bound.

The reference Qwen3-4B DAPO-Math 8K response settings use:

- `num_prompts_per_train_step=8`
- `num_samples_per_prompt=16`
- `max_response_tokens=8192`
- `max_total_tokens=10240`
- `num_training_steps=150`

That means one arm can require up to `8 * 16 = 128` sampled candidate responses
per optimizer step, or `150 * 128 = 19,200` sampled responses before validation.
The full two-arm comparison nominally asks for roughly `38,400` sampled training
responses before validation.

Autoregressive decoding is sequential over generated tokens. With DAPO-Math,
Qwen3 thinking enabled, `temperature=1.0`, `top_p=1.0`, and an 8192-token
response cap, many requests become long token-by-token decode workloads.

The no-BI metrics matched that explanation:

- Step 1: `generator/decode_time_ms/mean: 53109.0`,
  `generator/inter_token_latency_ms/mean: 70.92`,
  `perf/trainer/tokens_per_second_fwd_bwd: 5887.1`.
- Step 150: `generator/decode_time_ms/mean: 237703.6`,
  `generator/inter_token_latency_ms/mean: 77.82`,
  `perf/trainer/tokens_per_second_fwd_bwd: 12621.7`.
- The generated summary reports no-BI
  `generator/decode_time_ms/mean` with mean `208886` over 152 observations.

The trainer forward/backward path was much faster than the generator side could
feed batches. Full-step throughput dropped when the trainer waited for packed
rollout batches.

## Async Backpressure Shape

The controller's active rollout slots are:

```text
(target_offpolicy_steps + 1) * num_prompts_per_train_step
```

For no-BI, `target_offpolicy_steps=4` and `num_prompts_per_train_step=8`, so
there are `40` active prompt-group slots. Each prompt group has `16` samples,
so peak rollout concurrency is sized around `40 * 16 = 640` sequences, bounded
and routed across the generator shards.

The BI arm uses strict on-policy rollout with `target_offpolicy_steps=0`, so it
has fewer active prompt-group slots. It also uses fewer generator replicas
because each generator is TP=2 to match the trainer TP degree.

## Hardware Layout Tradeoff

The no-BI reference config uses:

- trainer: tensor parallel degree 2
- generators: 6 generators, each tensor parallel degree 1

This gives six independent single-GPU Qwen3-4B vLLM generators. That improves
request throughput by spreading prompts across generators, but it does not make
each individual long response decode as quickly as a TP=2 generator might.

The BI arm uses:

- `num_generators=3`
- generator tensor parallel degree 2
- `target_offpolicy_steps=0`
- flex attention with `BatchInvariantFlexConverter`
- deterministic batch-invariant debug settings

This is the required topology for the first-step BI parity check in this
checkout, but it is not a like-for-like generator-replica topology.

## Stop Signature

The BI stop was not reported as a Python trainer exception. The decisive trace
is:

```text
Unhandled monarch error on the root actor, hostname=n116-077-207, PID=19581 at time 2026-08-08 04:30:02.710087:
The actor logger-dpTS2Qh7THp{'gpus': 0/2} and all its descendants have failed:
  the process this actor was running on failed: Killed(sig=15)
[actor=<root>] Interrupted; attempting graceful shutdown...
```

The subsequent `trainer.close` and `generator.close` failures were Monarch
supervision errors because those actor processes had already been killed with
`SIGTERM`.

## Interpretation

The no-BI arm is a valid completed 150-step reference trace. Its logprob parity
metric stayed nonzero across the run.

The BI arm validates only the first comparable rootfs train metric:
`bit_wise/logprob_diff/max: 0` at step 1. It does not establish 150-step BI
parity, reward behavior, or steady-state throughput cost because the BI arm
stopped before step 2 was confirmed in the console log.
