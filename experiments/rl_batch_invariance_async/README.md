# RL Batch-Invariance Async Reference

This experiment compares the current TorchTitan RL DAPO-Math Qwen3-4B recipe with
batch-invariant numerics disabled and enabled. It is a local proxy for the GDN
async-RL batch-invariance claim, not an exact GDN/Qwen3.5 reproduction.

The central local question is narrow: does TorchTitan's batch-invariant path make
the trainer recomputed logprobs match the generator logprobs, and what cost does
that impose on reward and throughput?

## Source Map

The DAPO-Math base recipe is `rl_dapo_qwen3_4b_math_8k`. It builds Qwen3-4B-Base
with varlen attention, an fp32 LM head converter, 150 train steps, 8 prompt groups
per optimizer step, 16 completions per prompt, `target_offpolicy_steps=4`, a
TP=2 trainer, and six TP=1 generators
([config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:65),
[config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:74),
[config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:95),
[config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:120),
[config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:140)).

The two experiment entrypoints live in the same example registry so they are
discoverable through the normal RL launcher:

```bash
python -m torchtitan.experiments.rl.train \
  --module dapo_math \
  --config rl_dapo_qwen3_4b_math_8k_reference_no_bi

python -m torchtitan.experiments.rl.train \
  --module dapo_math \
  --config rl_dapo_qwen3_4b_math_8k_reference_bi
```

The no-BI arm keeps the base topology and disables W&B while enabling
TensorBoard under `outputs/rl_batch_invariance_async/no_bi`
([config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:175),
[config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:194)).

The BI arm uses flex attention plus `BatchInvariantFlexConverter`, pads each
sample to the flex block size, sets `DebugConfig(batch_invariant=True,
deterministic=True)`, disables trainer sequence parallelism, and changes to
three TP=2 generators so trainer and generator tensor parallelism match
([config_registry.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/examples/dapo_math/config_registry.py:201)).
This topology still uses all eight local GPUs: two for the trainer and six for
generation.

## RL Dataflow

`ConfigManager` loads a config function selected by `--module` and `--config`,
then applies CLI overrides before validation and launch
([manager.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/config/manager.py:34),
[manager.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/config/manager.py:49)).
The RL launcher builds the controller, computes trainer and generator world
sizes, spawns separate Monarch proc meshes for trainer and generator actors, and
then calls `setup_async()` followed by `run()`
([train.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/train.py:291)).

Rollout generation starts from a dataset sample. The rollouter builds one
environment per sibling completion, drives all siblings concurrently through the
generator, scores the completed rollouts, and attaches group advantages
([rollouter.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/rollout/rollouter.py:158)).
Each rollout turn stores the prompt token ids, completion token ids, generator
completion logprobs, and the policy version range sampled by the generator
([types.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/rollout/types.py:82)).

The batcher consumes rollout groups rather than raw samples. It waits for
`num_prompts_per_train_step` trainable groups and packs them into microbatches
with the configured local batch size and sequence length
([batcher.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/components/batcher.py:64),
[batcher.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/components/batcher.py:103)).

## Off-Policy Windowing

The async controller sizes the active rollout pipeline as:

```text
max_active_rollout_groups = (target_offpolicy_steps + 1) * num_prompts_per_train_step
```

This is implemented directly in `AsyncLoopConfig.max_active_rollout_groups`
([controller.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/controller.py:220)).
`target_offpolicy_steps` is a target, not a guaranteed observed age; generation
bottlenecks can lower the observed age, while windowed FIFO may increase it up
to the derived max
([controller.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/controller.py:167)).

The work buffer charges active slots when work is admitted and only releases
them after training/weight sync, which prevents born-stale rollout groups from
accumulating beyond the configured window
([work_buffer.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/components/work_buffer.py:54),
[work_buffer.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/components/work_buffer.py:64)).
Windowed FIFO can choose a finalized group inside the anchored window while
leaving later groups blocked
([work_buffer.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/components/work_buffer.py:174)).

At consumption time, the trainer computes policy age from the live trainer
policy version and the sampled policy versions in the packed batch, then raises
if the max age exceeds the configured bound
([controller_metrics.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/controller_metrics.py:167)).

## Weight Sync

Weight sync is deliberately overlapped with the training step. The manager waits
for the previous trainer push before mutating weights, runs the optimizer step,
waits for the previous generator pull before overwriting the shared key, then
starts the next push-pull chain in the background
([weight_sync.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/components/weight_sync.py:24)).
After the generator pull completes, the manager releases active rollout slots so
new rollouts are admitted only after generators hold the latest policy version
([weight_sync.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/components/weight_sync.py:109)).

## Logprob Mismatch Measurement

DAPO recomputes trainer logprobs from current-policy logits, subtracts the
generator logprobs stored in the rollout sample, and logs:

```text
bit_wise/logprob_diff/mean
bit_wise/ratio_tokens_different/mean
bit_wise/logprob_diff/max
```

Those metrics are emitted by the DAPO loss after masking non-finite generator
logprobs and non-loss tokens
([dapo.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/losses/dapo.py:81),
[dapo.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/losses/dapo.py:107)).
The metrics processor prints `bit_wise/logprob_diff/max` to the console by
default and sends every reduced metric to TensorBoard when enabled
([processor.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/observability/metrics/processor.py:55),
[processor.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/observability/metrics/processor.py:165)).

## Batch-Invariant Wiring

TorchTitan's RL docs define batch-invariant mode as making a given input's
outputs identical regardless of what other inputs share the batch. The documented
mechanisms are batch-invariant kernels for `mm`, `addmm`, `log_softmax`, and
`mean.dim`; deterministic NCCL settings; reduced-precision/TF32 disablement; and
single-split flash attention
([README.md](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/README.md:93)).

This checkout only supports logprob bitwise parity when trainer and generator
parallelism match
([README.md](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/README.md:104)).
The controller also validates that batch-invariant mode is deterministic, that
the trainer forward uses bf16 via FSDP mixed precision, that the generator dtype
is bf16, and that trainer sequence parallelism is disabled
([controller.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/controller.py:368)).

Generator startup applies batch-invariant mode before vLLM initialization, then
patches `bmm` and routes vLLM's token-logprob path through the trainer logprob
function so the generator does not bypass the batch-invariant Aten overrides
([generator.py](/data02/home/philip.yang/workspace/torchtitan/torchtitan/experiments/rl/actors/generator.py:797)).

## Blog Mapping

The public GDN/Qwen3.5 blog experiment uses machinery that is not present on
this checkout's mainline. This experiment therefore tests the nearest current
TorchTitan claims:

| Blog concept | TorchTitan proxy |
| --- | --- |
| Async rollout/training overlap | Controller rollout, batcher, trainer, and overlapped weight sync loops |
| Policy staleness/off-policy window | `target_offpolicy_steps`, active rollout slots, windowed FIFO, and consume-time policy-age metrics |
| Generator/trainer logprob mismatch | DAPO `bit_wise/logprob_diff/*` metrics |
| Batch-invariant numerics | `DebugConfig(batch_invariant=True, deterministic=True)`, flex attention converter, vLLM generator patches |
| Throughput/reward tradeoff | TensorBoard/console metrics summarized by `summarize.py` |

Because current BI parity requires matched trainer/generator TP and strict
on-policy generation, the BI arm is a validity-first comparison. Treat any reward
or throughput delta as a measured local tradeoff, not as proof that the async
TP=1 baseline topology would behave identically under a future GDN branch.

## Run

Install or verify the RL runtime dependencies, including the DAPO-Math verifier
requirements and the Qwen3-4B-Base checkpoint:

```bash
python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-4B-Base \
  --local_dir torchtitan/experiments/rl/example_checkpoint \
  --all
```

Then run both arms:

```bash
experiments/rl_batch_invariance_async/run_reference.sh
```

Useful variants:

```bash
# Install dependencies into the active env before running.
experiments/rl_batch_invariance_async/run_reference.sh --install-deps

# Bounded two-step smoke/parity run. This skips validation, uses one prompt and
# one sample, forces on-policy windowing, and caps generation length.
experiments/rl_batch_invariance_async/run_reference.sh --skip-full

# One arm only.
experiments/rl_batch_invariance_async/run_reference.sh --only bi
```

The script writes environment snapshots under
`outputs/rl_batch_invariance_async/env/`, arm logs and TensorBoard events under
`outputs/rl_batch_invariance_async/{no_bi,bi}/`, and summary files under this
experiment folder.

The smoke defaults are intentionally not a reward or throughput benchmark. They
exist to validate startup, weight sync, trainer stepping, and the first
generator/trainer logprob parity metric before spending time on the full
150-step reference arms.

## Acceptance Criteria

The run supports the local BI claim only if:

1. Both full arms complete without unhandled worker shutdown failures.
2. The BI arm has `bit_wise/logprob_diff/max == 0` at the first comparable
   on-policy train metric, or the first nonzero source is documented.
3. The summary includes logprob diff, rollout reward, validation reward,
   trainer throughput/step time, generator latency/decode metrics, and wall time.
4. `RESULTS.md` states whether the local data shows cleaner logprob signal,
   ambiguous reward gain, and measurable throughput cost.
