# Mission: TorchTitan RL Experiment Stack

## Why
Learn the `torchtitan.experiments.rl` stack well enough to modify, validate, and debug local RL training experiments without treating the controller, generator, trainer, and rollout code as a black box.

## Success looks like
- Trace a prompt group from dataset input through rollout generation, batching, optimization, and weight sync.
- Identify the right file to change for a new task, loss, router, generator setting, or reproducibility issue.
- Explain why generator/trainer logprob parity is hard and what batch-invariant mode changes.

## Constraints
- Ground every lesson in the current checkout.
- Prefer short lessons with source links and retrieval practice.
- Keep experiment-specific changes out of core TorchTitan unless the repo already has a general abstraction for them.

## Out of scope
- A general RLHF or policy-gradient course.
- A full Monarch, vLLM, or TorchStore deep dive unless needed for the RL stack.
