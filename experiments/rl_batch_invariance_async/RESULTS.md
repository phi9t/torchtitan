# RL Batch-Invariance Async Results

Status: bounded smoke parity completed; full 150-step reference deferred.

## Environment

Smoke run command:

```bash
PYTHONPATH=$PWD \
HF_HOME=$PWD/.cache/huggingface \
TRITON_CACHE_DIR=$PWD/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/torchinductor \
OUT_ROOT=outputs/rl_batch_invariance_async_smoke \
experiments/rl_batch_invariance_async/run_reference.sh --skip-full
```

Recorded environment:

- Date: `2026-08-07T06:52:12Z`
- Git commit: `633f53fa43593fb7a34be9c7aee563c511dd1f4b`
- Python: `.venv/bin/python`, `Python 3.11.2`
- Torch: `2.13.0+cu130`
- CUDA: available, 8 x NVIDIA B200 visible
- vLLM: `0.1.dev1+gd6938b740.d20260806`
- `batch_invariant_ops`: `0.1.0`
- Full snapshots:
  - `outputs/rl_batch_invariance_async_smoke/env/environment.txt`
  - `outputs/rl_batch_invariance_async_smoke/env/nvidia-smi.txt`
  - `outputs/rl_batch_invariance_async_smoke/env/python_packages.txt`

## Topology

| Arm | Trainer | Generators | Off-policy target | Attention | BI debug |
| --- | --- | --- | ---: | --- | --- |
| no_bi | TP=2 | 6 x TP=1 | full: 4, smoke: 0 | varlen | disabled |
| bi | TP=2 | 3 x TP=2 | 0 | flex | `batch_invariant=True`, `deterministic=True` |

The BI arm intentionally differs in generator topology because this checkout
only supports bitwise logprob parity when trainer and generator tensor
parallelism match.

## Smoke Settings

The completed smoke run used:

- `--async-loop.num-training-steps 2`
- `--async-loop.validation.num-samples 0`
- `--async-loop.num-prompts-per-train-step 1`
- `--async-loop.num-samples-per-prompt 1`
- `--async-loop.target-offpolicy-steps 0`
- `--async-loop.batcher.batch.seq-len 2048`
- `--generator.sampling.max_tokens 512`

These settings intentionally test the first-step generator/trainer logprob
parity path. They do not test reward improvement, async backlog behavior, or
long-context throughput.

## Smoke Metrics

Summary generated with:

```bash
PYTHONPATH=$PWD .venv/bin/python \
  experiments/rl_batch_invariance_async/summarize.py \
  --root outputs/rl_batch_invariance_async_smoke
```

Key results from `experiments/rl_batch_invariance_async/summary.md`:

| Metric | no_bi | bi |
| --- | ---: | ---: |
| `bit_wise/logprob_diff/max`, step 1 | 0.114651 | 0 |
| `bit_wise/logprob_diff/max`, step 2 | 0.0996521 | 0 |
| `bit_wise/logprob_diff/mean`, step 1 | 0.0124135 | 0 |
| `bit_wise/logprob_diff/mean`, step 2 | 0.000402601 | 0 |
| `bit_wise/ratio_tokens_different/mean`, steps 1-2 | 1 | 0 |
| `rollout_reward/_mean`, steps 1-2 | 0 | 0 |
| `generator/inter_token_latency_ms/mean`, step 2 | 67.3516 ms | 99.0855 ms |
| `generator/decode_time_ms/mean`, step 2 | 808.219 ms | 1189.03 ms |

The corresponding console logs are:

- `outputs/rl_batch_invariance_async_smoke/preflight_no_bi/run_2_steps.log`
- `outputs/rl_batch_invariance_async_smoke/preflight_bi/run_2_steps.log`

## Uncapped Preflight Evidence

The uncapped no-BI preflight reached real DAPO-style metrics with the base
prompt/sample/validation settings:

- Step 1: `rollout_reward/_mean: 0.045`,
  `bit_wise/logprob_diff/max: 0.56`,
  `perf/trainer/tokens_per_second_full_step: 200.0`
- Step 2: `rollout_reward/_mean: 0.027`,
  `bit_wise/logprob_diff/max: 0.33`,
  `perf/trainer/tokens_per_second_full_step: 449.0`
- Validation at step 2: `validation/response_length/mean: 879.8`,
  `timing/validate: 117.6`

The uncapped BI preflight successfully initialized BI on trainer and generators,
ran pre-training validation, and entered strict on-policy rollout generation:

- `max_num_seqs=43 per generator`
- `Validation | Step: 0 validation/response_length/mean: 1117.2`
- `timing/validate: 1003.6`
- `window_size=1, max_offpolicy_steps=0`

It was interrupted before the first BI train metric because the full validation
and long DAPO-Math rollout path made the run impractical for this turn.

## Interpretation

The bounded smoke result supports the local parity claim: with BI enabled,
matched trainer/generator TP, flex attention, deterministic mode, and the vLLM
logprob patch, the first comparable train metrics had
`bit_wise/logprob_diff/max == 0`; the no-BI arm remained nonzero.

This does not establish reward gain. Both smoke arms had zero reward, and
validation was disabled. The uncapped no-BI run produced reward and validation
metrics, but the uncapped BI run did not reach a train metric in the available
time.

The smoke run shows a local latency cost for BI on this tiny workload: generator
step-2 inter-token latency increased from 67.3516 ms to 99.0855 ms, and decode
time increased from 808.219 ms to 1189.03 ms. These numbers should not be used
as throughput conclusions for the full async DAPO setting.

Next full-reference attempt should keep validation and sampling bounded enough
to collect comparable BI metrics before restoring the full 8K/150-step settings.
