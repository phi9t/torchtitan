# TorchTitan RL Experiment Stack Resources

## Knowledge

- [Local source: RL README](torchtitan/experiments/rl/README.md)
  Primary overview for the experiment's intent, dependencies, entrypoint, and reproducibility claims. Use for: first-pass orientation and setup assumptions.
- [Local source: Controller async loop](torchtitan/experiments/rl/controller.py)
  Source of truth for the prompt-group pipeline, backpressure, active rollout buffer, and top-level `Controller.Config`. Use for: understanding runtime behavior.
- [Local source: Train entrypoint](torchtitan/experiments/rl/train.py)
  Source of truth for Monarch process mesh spawning and trainer/generator GPU partitioning. Use for: launch and process-layout questions.
- [Local source: RL data types](torchtitan/experiments/rl/types.py)
  Compact vocabulary for completions, training samples, microbatches, and optimizer-step outputs. Use for: tracing data shape across boundaries.
- [Local source: Rollout data types](torchtitan/experiments/rl/rollout/types.py)
  Compact vocabulary for generation requests, rollout turns, rollout groups, rewards, and terminal statuses. Use for: task/env/rubric work.
- [Local source: DAPO-Math configs](torchtitan/experiments/rl/examples/dapo_math/config_registry.py)
  Concrete Qwen3/DAPO recipe showing model spec, rollouter, trainer, generator, routing, and batch-invariant reference configs. Use for: learning from a realistic config.
- [Paper: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
  Algorithmic background for group-relative advantage estimation. Use for: understanding why sibling rollouts per prompt matter.
- [Paper: DAPO](https://arxiv.org/abs/2503.14476)
  Algorithmic background for DAPO-style clipped policy optimization and long-CoT math RL. Use for: interpreting the DAPO-Math example.

## Wisdom (Communities)

- [PyTorch TorchTitan GitHub issues](https://github.com/pytorch/torchtitan/issues)
  Best public venue for current implementation questions and regressions.
- [vLLM GitHub issues](https://github.com/vllm-project/vllm/issues)
  Use for generator-side engine behavior, cudagraph, and kernel/runtime issues.
- [Monarch GitHub repository](https://github.com/meta-pytorch/monarch)
  Use for actor/process-mesh behavior when local source comments point at Monarch changes.

## Gaps

- Need a future lesson that follows a single real DAPO-Math prompt through `DapoMathRollouter`, `TokenEnv`, `DAPOLoss`, and trainer metrics.
