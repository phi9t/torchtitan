# Modded NanoGPT B200 Benchmark Design

Status: historical design draft; superseded for implementation
Date: 2026-08-14
Surface: repo-local research program
Route: `ask-matt` gated design path

## Intent

This document is retained as the historical benchmark-design sketch. The
canonical implementation spec and current acceptance criteria now live in
`.scratch/modded-nanogpt-b200/spec.md`; executable launch rules live in
`experiments/modded_nanogpt_b200/preflight_checklist.md`. When this document
conflicts with either file, use the scratch spec first and the preflight
checklist second.

Design a B200 benchmark around
`https://github.com/kellerjordan/modded-nanogpt` that answers three separate
questions without mixing their evidence:

- Can the current upstream speedrun run correctly on an 8x B200 host?
- What is the B200 wall-clock baseline for the upstream 8x H100 target?
- Which B200-specific changes improve time-to-target without changing the
  train/validation token streams or silently changing the ML claim?

The benchmark is not a TorchTitan core training change. It belongs under
`experiments/modded_nanogpt_b200/` for runners, reports, source snapshots, and
local artifacts. If reusable TorchTitan code is later needed, it should be
introduced only after a separate design and ticket.

## Primary Sources

Keep upstream source facts and claim boundaries in
`experiments/modded_nanogpt_b200/upstream_reference.md`. The benchmark design
uses that reference as the pinned metadata source for the inspected upstream
snapshot:

- Repository: `https://github.com/kellerjordan/modded-nanogpt`
- Commit: `ecbb586296d3dac36fd206211f25d63bad4a6b35`

TorchTitan-local constraints:

- Keep generated datasets, caches, run logs, and large result trees out of git.
- Use repo-local experiment folders for runnable scripts and reports.
- For real GPU experiment work in this checkout, prefer the repo-local rootfs
  path and set repo-local cache directories so runs are hermetic.
- Prior B200 TorchTitan work on this host needed a recent PyTorch nightly and
  explicit `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, plus a writable repo-local
  HuggingFace cache.

## Benchmark Shape

Run the experiment as a three-lane matrix. Each lane has its own claim boundary.

### Lane A: Faithful Upstream B200 Reproduction

Question: does the upstream commit run on 8x B200 with environment-only
adaptation?

Allowed changes:

- Pin the upstream commit in a source snapshot or external clone.
- Build or enter a B200-compatible environment.
- Set cache, compiler, and CUDA visibility environment variables.
- Add a wrapper outside upstream that captures logs and metadata.

Disallowed changes:

- No changes to `train_gpt.py`, `run.sh`, data-token streams, validation-token
  count, training schedule, model math, optimizer math, compile flags, or
  kernel choices.

Evidence:

- Upstream commit SHA.
- `nvidia-smi`, driver, GPU name, compute capability, SM count, CUDA runtime,
  PyTorch version, Triton version, NCCL version if visible.
- Exact data manifest for `data/fineweb10B/*.bin`.
- Complete upstream console log.
- Final `val_loss`, `train_time`, `step_avg`, peak allocated/reserved memory.

Success gate:

- Runs to final validation without changing upstream source.
- Final validation loss is <= 3.28.
- Reported timing is classified as `B200 upstream reproduction`, not a new
  upstream record unless upstream competition rules are also satisfied.

### Lane B: Minimal B200 Compatibility Patchset

Question: if Lane A fails, what is the smallest patchset needed for B200?

Allowed changes:

- Replace the environment with a B200-supported CUDA/PyTorch/Triton stack.
- Patch launch/container scripts for CUDA 13 or another proven B200 stack.
- Patch architecture detection for `sm_100` only when a kernel or build path
  requires it.
- Add source-local preflight checks that fail before the timed region.

Disallowed changes:

- No model, optimizer, schedule, train/validation token-stream, validation-token
  count, or compile-flag change unless the result is explicitly reclassified as
  `B200 variant`, not `faithful reproduction`.

Evidence:

- Patch diff against upstream commit.
- Each patch labeled `environment`, `hardware-detection`, `kernel-compat`,
  `timing-harness`, or `ML-affecting`.
- First failing Lane A error that the patch fixes.
- Same metrics as Lane A.

Success gate:

- Every patch is justified by a concrete B200 incompatibility.
- No ML-affecting patch is present for a competition-comparable claim.
- Final validation loss is <= 3.28.

### Lane C: B200 Optimization Ablations

Question: which B200-specific changes improve time-to-target?

Allowed changes:

- Tune B200 kernel selection or architecture-specific fast paths.
- Compare CUDA/PyTorch/Triton/NCCL stacks.
- Compare FP8 enablement and fallback only as ablations, not hidden defaults.
- Compare compile cache state separately from steady-state training.

Disallowed changes for a competition-comparable timing claim:

- Changing train or validation token streams.
- Adding extra Inductor or `torch.compile` flags banned by upstream rules.
- Reporting a timing against a different baseline hardware or different source
  patchset.

Required ablation arms:

- `A0`: faithful upstream on B200, or the nearest source-equivalent Lane B if
  Lane A cannot run.
- `B0`: B200 compatibility baseline after required patches only.
- `B1`: CUDA/PyTorch/Triton stack candidate.
- `B2`: Flash Attention 3 versus B200-compatible fallback, if FA3 is not stable.
- `B3`: FP8 enabled versus `DISABLE_FP8=1` diagnostic, only to isolate B200 FP8
  kernel problems.
- `B4`: NCCL environment candidate, if distributed collectives dominate.

Report each arm as either:

- `competition-comparable`: unchanged data, validation, model, optimizer,
  schedule, and allowed compile policy.
- `B200 systems-only`: no ML semantics change, but not necessarily upstream
  record-comparable because the runtime stack or system knobs changed.
- `B200 ML variant`: changes model, optimizer, schedule, precision semantics,
  tokenization, train stream, validation stream, or validation count.

## B200 Adaptation Plan

### Environment

Use the TorchTitan rootfs as the default host adapter so the run records the
same kind of reproducible environment evidence as other repo-local GPU work.
The first runnable ticket should create a wrapper that either:

- clones or checks out `modded-nanogpt` at the pinned commit under a gitignored
  source directory, or
- verifies an existing source checkout path supplied by `MODDED_NANOGPT_SRC`.

The wrapper should set:

```bash
HF_HOME="${REPO_ROOT}/.cache/huggingface"
HF_HUB_CACHE="${HF_HOME}/hub"
CC="${CC:-/usr/bin/gcc}"
CXX="${CXX:-/usr/bin/g++}"
DATA_PATH="${MODDED_NANOGPT_DATA_PATH:-${MODDED_NANOGPT_SRC}}"
```

The wrapper should not edit upstream files. If a patch is needed, apply it in a
separate git worktree or copied source directory and preserve the diff as an
artifact.

### Preflight

Fail before data download or training unless all of these are true:

- 8 visible CUDA devices.
- Every visible device reports name containing `B200` or a configured expected
  device name.
- Compute capability is Blackwell-class for every visible device.
- PyTorch can allocate BF16 tensors and FP8 tensors on device.
- `torch._scaled_mm` exists and works on a one-block FP8 smoke.
- `kernels-community/flash-attn3` imports and runs a tiny varlen attention
  smoke, or the arm is explicitly marked as an FA3 fallback ablation.
- NCCL can initialize 8 local ranks and complete an all-reduce smoke.
- The FineWeb binary shards and validation shards are present and have a
  recorded manifest.

### Data

Do not change the upstream token streams for a competition-comparable claim.
The data step should run upstream `data/cached_fineweb10B.py 9` once, online,
inside the selected environment, then write a manifest:

- dataset script path and upstream commit;
- shard paths, byte sizes, and checksums;
- exact command and environment;
- whether the data was freshly downloaded or reused.

For a cheaper pipeline smoke, permit `data/cached_fineweb10B.py 1`, but classify
the result as `smoke`. It cannot support the <= 3.28 claim.

### Timing

Preserve upstream timing semantics for Lane A and B0:

- Untimed compile and kernel warmup stay untimed.
- Training time begins at the upstream timer after warmup.
- Validation time is excluded in the same way upstream excludes it.
- The final reported time is `train_time` from the upstream log, not shell
  wall-clock.

Also record wall-clock separately because it is operationally relevant on this
host:

- source preparation time;
- dependency install time;
- compile/warmup time;
- timed training time;
- validation time;
- total elapsed wall-clock.

Never mix these in one metric.

### Statistical Evidence

Minimum evidence tiers:

- `smoke`: 1 short run, enough to prove launch, data, warmup, and log parsing.
- `single`: 1 full run to final validation, enough for local B200 timing but
  not enough for a record-style validation claim.
- `replicated`: at least 5 full runs for a stable B200 baseline.
- `record-style`: enough runs to evaluate the upstream p-value requirement for
  mean validation loss <= 3.28 if claiming competition relevance.

For optimization arms, compare the same source, data manifest, environment
class, and GPU allocation. Report median and best train time, final validation
loss distribution, and whether each arm remains below 3.28.

## Artifact Layout

```text
experiments/modded_nanogpt_b200/
  benchmark_design.md
  README.md                         # later: commands and run index
  run_common.sh                     # later: rootfs/cache/preflight helpers
  fetch_upstream.sh                 # later: pinned source checkout
  prepare_data.sh                   # later: data manifest generation
  run_reproduction.sh               # later: Lane A/B0 launch wrapper
  parse_log.py                      # later: extract train_time/val_loss/memory
  summarize.py                      # later: arm comparison summary
  sources/                          # gitignored upstream checkout or patch dirs
  data/                             # gitignored FineWeb binary shards/manifests
  results/                          # gitignored logs, manifests, summaries
```

Generated source snapshots, data, logs, caches, and large result trees must be
gitignored before implementation runs.

## Acceptance Criteria

The first implementation milestone is complete when:

- A preflight command fails loudly on non-B200 or incomplete environments.
- A pinned upstream source checkout is verified at
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- FineWeb data preparation writes a checksum manifest.
- The reproduction wrapper can run a smoke path and a full path without editing
  upstream source.
- A parser extracts final validation loss, upstream timed training time, step
  average, peak allocated memory, and peak reserved memory.
- A report distinguishes `smoke`, `B200 upstream reproduction`,
  `B200 compatibility patchset`, and `B200 optimization ablation`.

The benchmark can claim a B200 upstream reproduction only when:

- upstream source is unchanged;
- data manifest matches the upstream token-stream contract;
- the run reaches final validation with `val_loss <= 3.28`;
- timing uses upstream `train_time`;
- environment and hardware evidence are included.

## Open Decisions

- Whether to require rootfs for all runs or allow upstream Docker as a separate
  environment arm. Recommendation: rootfs first on this host, Docker only as a
  portability comparison.
- Whether to import a copy of the upstream source into a gitignored
  `sources/` tree or require `MODDED_NANOGPT_SRC`. Recommendation: support both,
  but all reports must record the actual source path and commit.
- Whether to spend the first GPU budget on a full Lane A run or a cheaper
  staged smoke. Recommendation: staged smoke first, then one full Lane A run.
- Whether to compare against TorchTitan-native Qwen3/FineWeb runs in the same
  report. Recommendation: not in v1. Keep this benchmark about the upstream
  speedrun and B200 adaptation; add TorchTitan-native comparison later after the
  B200 baseline is known.
