# Modded NanoGPT B200 Benchmark

Status: design-draft

## Intent

Design and later implement a repo-local benchmark harness for running
`kellerjordan/modded-nanogpt` on an 8x B200 host while preserving the upstream
speedrun claim boundaries.

The design source for this effort is:

- `experiments/modded_nanogpt_b200/benchmark_design.md`

## Classification

Route: gated path.

Reason:

- external source repository;
- real 8-GPU B200 runtime;
- benchmark and validation-loss claims;
- environment, data, and distributed-runtime evidence requirements;
- likely multi-ticket implementation.

Surface: repo-local research program. This effort must not change TorchTitan
core trainer semantics unless a later ticket explicitly justifies and
authorizes that change.

## Authority Boundary

This tracker authorizes design only. It does not authorize:

- cloning large source or data trees into git-tracked paths;
- running long GPU jobs;
- changing TorchTitan core training code;
- changing upstream `modded-nanogpt` source;
- committing, pushing, or opening a pull request.

Future code work should happen in an isolated worktree or a clearly bounded
dirty-tree-safe path.

## Acceptance Criteria

The v1 benchmark harness is complete when:

- the upstream source commit is pinned and verified;
- environment preflight fails loudly unless 8 B200 GPUs and required FP8, FA3,
  NCCL, compiler, and cache paths are present;
- FineWeb binary data preparation records a manifest with checksums;
- the reproduction wrapper can run a smoke path and a full path without editing
  upstream source;
- logs are parsed into a typed summary containing final validation loss,
  upstream `train_time`, `step_avg`, peak allocated memory, peak reserved
  memory, source commit, hardware, and environment;
- reports classify results as `smoke`, `B200 upstream reproduction`,
  `B200 compatibility patchset`, or `B200 optimization ablation`;
- no competition-comparable claim is made when data streams, validation count,
  model, optimizer, schedule, or banned compile policy changed.

## Child Tickets

Planned tickets:

- `issues/01-source-and-preflight.md`
- `issues/02-data-and-manifest.md`
- `issues/03-reproduction-wrapper-and-parser.md`
- `issues/04-b200-ablation-matrix.md`
