# Scaffold-To-Policy Experiments

This directory records the broader experiment family for compressing verified
scaffold behavior into policy behavior. Countdown remains the first implemented
task under `experiments/countdown_search_distill/`; this directory starts as the
shared planning, registry, and evidence-contract home for the next lanes.

See `agentic_eval_benchmarks.md` for the benchmark landscape, complexity ladder,
normalization rules, and adoption plan.

See `followup_experiment_plan.md` for the concrete staged experiment program
that follows from the Countdown pilot: split repair, Countdown ablations,
reasoning transfer, harness feasibility, and later agentic pilots.

## Current Ladder

1. **Countdown repair**: regenerate clean held-out same-regime splits, enforce
   global problem-key de-duplication, and rerun base plus exported adapters.
2. **Reasoning lane**: add small exact-verifier reasoning tasks before broader
   benchmark claims.
3. **Agentic task lane**: add tau2-bench and Terminal-Bench/Harbor after the
   registry supports external harness boundaries and at least one reasoning
   benchmark shows replicated held-out gain.

## Evidence Contract

Every serious run must record:

- runtime preflight;
- split registry and overlap validation;
- base metrics;
- adapter metrics;
- scaffold subset metrics, such as base-elicitable subsets when applicable;
- verifier failure-mode breakdown;
- representative examples;
- seed or split-draw replication before robust claims.

## Initial Reasoning Benchmarks

The first reasoning lane should include:

- GSM8K or a small GSM-style subset with parsed final-answer verification;
- a small MATH subset with task-appropriate answer normalization;
- verifier-first synthetic reasoning tasks where exact verification is cheap.

The first scaffold remains best-of-N sampling. Tool-assisted scaffolds are a
second-tier variant and must be reported separately.

## Agentic Benchmarks

tau2-bench and Terminal-Bench/Harbor are in scope, but they enter after the
reasoning lane because they add external harnesses, tool/user or terminal
interaction semantics, and slower artifact loops.

For Harbor/Terminal-Bench, this repo should own rootfs-managed launch,
registry entries, artifact paths, and result ingestion. Harbor should own task
execution.

For tau2-bench, this repo should own rootfs-managed install/run pinning,
trajectory and score ingestion, and report generation. The tau2 harness should
own task execution.

## Registry Shape

Each run family should be described by a machine-readable registry with:

- task name and lane;
- verifier type;
- model and asset path;
- split registry path;
- scaffold type and rollout budget;
- training arms;
- training budget;
- evaluation budget;
- success thresholds;
- artifact roots;
- external harness boundary, if any;
- report template.
