# Modded NanoGPT B200 Execution Prompt

Use this prompt to guide a fresh execution agent through the
`experiments/modded_nanogpt_b200` productionization run. Treat this file as the
user intent for the execution session, and treat the current files on disk as
authoritative.

## Role

You are the execution agent for the TorchTitan-local
`kellerjordan/modded-nanogpt` B200 harness. Your job is to productionize the
experiment path, not merely launch one job. Start by reading the previous run
artifacts and agent traces, identify the current highest-confidence blocker or
launch gate, repair the harness or environment only where the evidence points,
run the required rootfs preflights, launch authorized full B200 jobs only when
gates pass, babysit them while they run, preserve telemetry, analyze telemetry
and results, and stop with a precise blocker when a gate or runtime condition
fails.

The requested result is a B200 upstream reproduction or a clearly labeled B200
compatibility/diagnostic result. It is not an official upstream record claim.

The active RSI foundation target is one authorized, sequential, non-skip
two-GPU Lane B full attempt after the current dry prerequisite gate. Broader 8x
reproduction, repeatability, ablation, or a 10-run production campaign is
superseded until that first trial is stopped, parsed, summarized, and
classified from preserved evidence.

## Mandatory Sources

Read these before acting:

- `CONSTITUTION.md`
- `docs/agents/agentic-engineering.md`
- `docs/agents/domain.md`
- `docs/agents/issue-tracker.md`
- `AGENTS.md`
- `.scratch/modded-nanogpt-b200/spec.md`
- `.scratch/modded-nanogpt-b200/completion_audit.md`, if present
- `experiments/modded_nanogpt_b200/preflight_checklist.md`
- `experiments/modded_nanogpt_b200/upstream_reference.md`
- `experiments/modded_nanogpt_b200/results/run_index.json`, if present

Authority order for this effort:

1. Direct user instructions in the execution session.
2. `AGENTS.md` and the rootfs-only rule.
3. `.scratch/modded-nanogpt-b200/spec.md`.
4. `experiments/modded_nanogpt_b200/preflight_checklist.md`.
5. `experiments/modded_nanogpt_b200/upstream_reference.md`.
6. `experiments/modded_nanogpt_b200/benchmark_design.md`, which is historical
   context only.

## Non-Negotiable Boundaries

- Host shell is only for orchestration: create result directories, set
  environment variables, call repo-local shell wrappers, and inspect git status
  or generated logs.
- Python, pip installs, dataset code, CUDA/FlashAttention/Triton/NCCL probes,
  `torchrun`, training, validation, log parsing, telemetry parsing, and
  summarization must run inside `scripts/rootfs/enter_rootfs.sh`.
- Use repo-relative entrypoints when entering rootfs. Do not pass absolute host
  paths into `scripts/rootfs/enter_rootfs.sh`.
- Do not run upstream `pip install -r requirements.txt`; it can replace the
  B200-capable Torch stack.
- Do not modify upstream source for Lane A.
- Do not track generated source, data, logs, caches, telemetry, or result
  bundles.
- Do not claim success unless final validation is reached. If final validation
  is absent, `val_loss`, `train_time`, and `step_avg` are null or omitted.
- Do not treat `ready_to_launch=true` as a launched run. A launch-ready dry gate
  with `skip_run=true` is prerequisite evidence only.
- Do not count a run in baseline or production evidence unless the run index
  and the attempt artifacts prove `training_launched=true`, non-skip execution,
  final validation metrics, exit code `0`, rootfs sentinel evidence, the
  declared GPU allocation, NCCL over that declared world size, and SHA-verified
  900M data.
- Report upstream `train_time` separately from shell wall-clock.
- The active small-scale RSI foundation trial uses exactly two visible B200
  GPUs. A broader 8x B200 reproduction or production baseline still requires
  explicit authorization and must not be inferred from a two-GPU prerequisite.
  Full jobs require NCCL, full 900M FineWeb manifest, SHA verification, rootfs
  sentinel evidence, and no known-stall override.
- Diagnostic runs must be labeled diagnostic and excluded from successful
  baseline statistics.
- Commit, push, and PR creation require explicit user authorization.

## Current Known Runtime Facts

These facts may be stale; verify cheaply before relying on them.

- Pinned upstream commit:
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
- Rootfs environment previously observed:
  - Python `3.12.3`
  - Torch `2.13.0+cu132`
  - CUDA runtime `13.2`
  - Triton `3.7.1`
  - host inventory has included 8x `NVIDIA B200`, compute capability `(10, 0)`;
    the active RSI foundation launch allocation is exactly 2 visible B200 GPUs
- FA2 has previously built and smoked successfully in rootfs:
  `flash-attn==2.8.3.post1`, CUDA build wheels `13.2.86`,
  `TORCH_CUDA_ARCH_LIST=10.0`.
- Lane A FA3 has previously failed on B200 with
  `no kernel image is available for execution on the device`.
- FlexAttention full-width fallback previously attempted a 256 GiB allocation.
- Triton MLP previously failed for sm100 in
  `TritonNvidiaGPUOptimizeTMemLayoutsPass`.
- PyTorch MLP fallback previously passed a local smoke but has not passed a
  full-job warmup.
- An older data manifest may exist but may be rejected because it predates the
  schema-versioned manifest contract.
- The current run index has previously shown many diagnostic or blocked
  attempts, zero baseline statistics, and a dry full-mode Lane B prerequisite
  row with `skip_run=true`. Recompute the current state before acting.
- Historical failures include FA3 missing B200 kernel images, FlexAttention
  full-width allocation blow-up, Triton MLP sm100 compiler failures, stale or
  incomplete manifest SHA evidence, sidecar-only readiness over-reporting, and
  launch authorization gates.

## Result Classification Schema

Every artifact that represents an attempt must include:

```json
{
  "schema_version": 1,
  "classification": {
    "lane": "A|B|C",
    "mode": "full|smoke|diagnostic",
    "arm": "A0|B0|B1|...",
    "claim_label": "B200 upstream reproduction|B200 compatibility patchset|B200 systems-only|B200 ML variant|smoke|diagnostic",
    "evidence_tier": "full-single-attempt|smoke|diagnostic",
    "run_id": "...",
    "attempt_id": "...",
    "environment_class": "torchtitan-rootfs-b200",
    "claim_eligible": false
  }
}
```

These `claim_label` values are v1 legacy emitted strings. For RSI planning,
map `B200 compatibility patchset` and `B200 systems-only` to
`B200-compatible local setup`, `B200 ML variant` to `B200 local variant`,
`B200 upstream reproduction` to `Faithful upstream reproduction`, and
`diagnostic` to `Diagnostic`. Do not rename emitted labels without a separate
schema compatibility migration.

`claim_eligible` means only that the attempt or materialized plan is
structurally allowed to support a claim. Post-run claim validity still requires
successful full-mode preflight, final validation, valid source state, valid
data, and valid metrics.

## Execution Mode Selection

Start with Lane A full only if the user explicitly asked for a faithful
upstream full run in the execution session and all full gates can pass.

Use Lane B diagnostic when investigating the current B200 compatibility
blockers, including FA2 fallback, MLP fallback, or schema/data harness repair.

Use smoke mode only for cheap harness validation. A smoke run cannot support a
validation-loss or timing claim.

Do not launch Lane C ablations until a Lane A or Lane B baseline artifact exists
and ticket `04` is unblocked.

## Phase 0: Trace-First Resume

Before preparing a new result directory, reconstruct the current state from
existing artifacts. This phase prevents repeating old work and prevents dry
readiness from being mistaken for production evidence.

Read or regenerate, using rootfs-aware wrappers where they execute Python:

- `.scratch/modded-nanogpt-b200/completion_audit.md`, if present;
- `experiments/modded_nanogpt_b200/results/run_index.json`;
- the latest `summary.json`, `analysis.md`, `launch_readiness.json`,
  `preflight_report.json`, `blocker.json`, `exit_code.json`, and `run.log` for
  the newest relevant Lane A and Lane B attempts;
- prior rollout summaries or local agent traces only as historical hints. The
  current result artifacts and current source files win.

Produce a short resume note in the new `operator_notes.md` or in the final
response before launching anything. It must state:

- current `baseline_stats.count`;
- number of non-skip launch-ready rows;
- the newest prerequisite-only row, if any, and why it is not production
  evidence;
- the newest concrete blocker by phase;
- the next single gate to exercise.

If the latest state is already blocked on explicit launch authorization and no
material input changed, do not rerun equivalent audits or preflights merely to
reconfirm the same state. If the execution session explicitly grants full B200
launch authority, continue to the safety check and full launch path.

## Phase 1: State and Safety Check

From the host shell:

1. Confirm the checkout root:
   `test "$(pwd)" = "$(git rev-parse --show-toplevel)"`.
2. Record `git status --short --ignored=matching` for:
   - `.scratch/modded-nanogpt-b200`
   - `experiments/modded_nanogpt_b200`
   - `.gitignore`
   - `.claude/CLAUDE.md`
3. Confirm generated paths are ignored:
   - `experiments/modded_nanogpt_b200/sources/`
   - `experiments/modded_nanogpt_b200/data/`
   - `experiments/modded_nanogpt_b200/results/`
4. Check for active jobs with the rootfs-aware structured scanner:
   `experiments/modded_nanogpt_b200/check_active_jobs.sh --active-jobs-output experiments/modded_nanogpt_b200/results/<run_id>/active_jobs.json`.
5. Confirm the current run index does not already contain an accepted non-skip
   two-GPU Lane B full attempt for the active RSI foundation gate. If it does,
   verify the summary and stop with the evidence list instead of launching more
   jobs.

Stop before launching a job if another process is using the target GPUs and the
user has not authorized sharing the machine.

## Phase 2: Rootfs Repair Loop

Use this loop whenever rootfs entry, Python, CUDA, FlashAttention, Triton, NCCL,
or package installation fails.

1. Reproduce the failure through a repo-local wrapper or
   `scripts/rootfs/enter_rootfs.sh -- <repo-relative-command>`.
2. Classify the failure:
   - rootfs missing or unusable;
   - host NVIDIA device or driver library bind issue;
   - missing rootfs package;
   - wrong Torch/CUDA/Triton version;
   - missing CUDA build wheel or CUDA symlink;
   - stale compiled extension;
   - NCCL rendezvous or stale process;
   - source/data manifest contract failure.
3. Fix only the minimal rootfs or wrapper issue needed for the current gate.
4. Install Python packages only inside rootfs, using
   `experiments/modded_nanogpt_b200/runtime/sync_python_env.sh` for normal
   runtime sync.
5. Avoid Torch replacement. If a package tries to pull Torch, stop and update
   the direct non-Torch runtime lock through the networked-rootfs diagnostic
   path instead of running a broad upstream requirements install.
6. After repair, rerun the exact failing rootfs command and preserve the before
   and after evidence in the result directory.

Allowed rootfs repairs include:

- building the default rootfs through `scripts/rootfs/enter_rootfs.sh` if it is
  missing;
- installing missing direct runtime dependencies inside rootfs;
- restoring `/opt/cuda-synth` symlinks required for extension builds;
- rebuilding `flash-attn==2.8.3.post1` through
  `experiments/modded_nanogpt_b200/setup_flash_attention.sh`;
- killing stale processes only after confirming they belong to the current
  failed attempt or are clearly abandoned.

Not allowed without explicit user authorization:

- replacing the Torch stack;
- modifying TorchTitan core;
- changing upstream Lane A source;
- deleting generated result directories from previous attempts;
- using Docker as a replacement path for this run.

## Phase 3: Prepare Result Directory and Metadata

Create a unique result directory under ignored results:

```bash
RUN_ID="lane_<lane>_<mode>_<YYYYMMDDTHHMMSSZ>"
ATTEMPT_ID="${RUN_ID}_attempt_001"
RESULT_DIR="experiments/modded_nanogpt_b200/results/${RUN_ID}"
mkdir -p "${RESULT_DIR}"
```

Before any expensive step, create or capture:

- `attempt.json` with the common classification schema;
- `command.env.json` with schema-valid redacted environment variables;
- `command.argv.json` with the exact wrapper and training commands planned;
- `operator_notes.md` with the Phase 0 resume note, current known blockers, and
  the intended lane/mode.

Redact secrets. Do not dump tokens, credentials, or private endpoints.

## Phase 4: Source Preparation

For Lane A:

1. Ensure source exists under
   `experiments/modded_nanogpt_b200/sources/modded-nanogpt`.
2. Clone or fetch if missing.
3. Check out `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
4. Verify `git rev-parse HEAD`.
5. Verify `git status --short` is empty.
6. Record `source.json`.

For Lane B:

1. Use a distinct generated source tree under
   `experiments/modded_nanogpt_b200/sources/`.
2. Ensure it descends from the pinned commit.
3. Preserve `variant_patch.diff` before launch.
4. Classify every changed file as environment, hardware-detection,
   kernel-compat, timing-harness, or ML-affecting.
5. Record the first Lane A blocker each patch addresses.

Run any source helper through rootfs if it invokes Python, network-adjacent
helper code, or CUDA. Git-only clone/status commands may be orchestrated from
host.

## Phase 5: Dependency and FlashAttention Preparation

Inside rootfs:

1. Verify Torch remains `2.13.0+cu132`.
2. Verify CUDA runtime remains `13.2`.
3. Verify Triton imports.
4. Install only missing direct dependencies. Use `--no-deps` when needed to
   avoid Torch replacement.
5. For Lane A, verify the `kernels` dependency if FA3 is being tested.
6. For Lane B FA2, run:
   `experiments/modded_nanogpt_b200/setup_flash_attention.sh`.

If `setup_flash_attention.sh` fails:

- preserve full build output;
- inspect for CUDA wheel mismatch, `-lcudart`, compiler, or arch issues;
- repair inside rootfs;
- rerun the script until it passes or a precise blocker remains.

Success requires the FA2 B200 varlen/window smoke, not just import success.

## Phase 6: Data Preparation and Manifest

Full-mode runs require a schema-versioned 900M FineWeb manifest:

- `schema_version: 1`
- `dataset: fineweb10B`
- `token_budget: 900M`
- pinned source commit;
- preparation command;
- environment snapshot;
- fresh versus reused status;
- exactly 10 `.bin` shards;
- total bytes `2000010240`;
- per-shard byte sizes;
- per-shard SHA256 checksums.

Run or refresh data preparation through the repo-local rootfs-aware helper:

```bash
experiments/modded_nanogpt_b200/prepare_data.sh \
  --source "${SOURCE}" \
  --data-dir "${DATA_DIR}" \
  --output "${MANIFEST}" \
  --token-budget 900M \
  --freshness fresh
```

The helper records the upstream source command
`python data/cached_fineweb10B.py 9` as manifest provenance for full manifests.
Do not run that upstream command directly from the host. If a schema-valid
manifest is missing, create it through a rootfs-aware helper or one-off rootfs
Python command and write it atomically. Do not mutate an old result manifest in
place; write a new manifest in the current result directory or a clearly named
generated data-manifest path.

Before a full job, run preflight with `--verify-sha`. Expect this to take time;
preserve duration as data-verification wall-clock.

A manifest is full-job evidence only when the current attempt summary or
preflight report can prove top-level `verified_sha=true`, the pinned source
commit, exactly 10 shards, token budget `900M`, and total bytes `2000010240`.
If older artifacts point at a manifest but the summary cannot prove those
fields, refresh the manifest or summary through the rootfs-aware helpers before
launch.

## Phase 7: Preflight

Use `experiments/modded_nanogpt_b200/run_preflight.sh`; it re-enters rootfs.

Lane A full shape:

```bash
MODDED_NANOGPT_ATTN_BACKEND=fa3 \
experiments/modded_nanogpt_b200/run_preflight.sh \
  --mode full \
  --lane A \
  --run-id "${RUN_ID}" \
  --attempt-id "${ATTEMPT_ID}" \
  --arm A0 \
  --source "${SOURCE}" \
  --data-manifest "${DATA_MANIFEST}" \
  --attention-backend fa3 \
  --mlp-backend triton \
  --verify-sha \
  --report "${RESULT_DIR}/preflight_report.json"
```

Historical Lane B diagnostic shape for the FA2/PyTorch-MLP fallback
investigation:

```bash
MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100 \
MODDED_NANOGPT_ATTN_BACKEND=fa2 \
MODDED_NANOGPT_MLP_BACKEND=torch \
MODDED_NANOGPT_ALLOW_SKIP_NCCL=1 \
experiments/modded_nanogpt_b200/run_preflight.sh \
  --mode diagnostic \
  --lane B \
  --run-id "${RUN_ID}" \
  --attempt-id "${ATTEMPT_ID}" \
  --arm B0 \
  --source "${SOURCE}" \
  --data-manifest "${DATA_MANIFEST}" \
  --attention-backend fa2 \
  --mlp-backend torch \
  --skip-nccl \
  --allow-previous-stall \
  --report "${RESULT_DIR}/preflight_report.json"
```

Do not launch a full job unless full preflight exits `0`. If preflight fails,
preserve `preflight_report.json`, classify the blocker, and stop or repair the
failing prerequisite before retrying.

For the current Lane B production candidate, use the full-mode gate shape that
exercises FA2 and Triton MLP unless new trace evidence proves a different
backend is the next blocker. PyTorch MLP is diagnostic-only fallback evidence;
do not use it for the next full launch unless its patch class and metric impact
are explicitly classified.

## Phase 8: Telemetry Capture Plan

Capture telemetry without changing upstream source.

Required artifacts:

- `preflight_report.json`
- `attempt.json`
- `source.json`
- `data_manifest.json` or pointer record
- `command.env.json`
- `command.argv.json`
- `run.log`
- `wall_clock.json`
- `telemetry/`
- `summary.json`
- `analysis.md`

Telemetry directory:

```text
telemetry/
  nvidia_smi_query.csv
  nvidia_smi_dmon.csv
  process_watch.log
  disk_watch.log
  gpu_processes.log
  rootfs_environment.json
  source_status_before.txt
  source_status_after.txt
  optional_torch_profiler/
```

Start low-overhead host-side telemetry for operational observation only:

- `nvidia-smi --query-gpu=timestamp,index,name,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5`
- `nvidia-smi dmon -s pucvmet -d 5`
- process watcher for `torchrun`, `train_gpt.py`, Python children, CPU RSS, and
  elapsed time;
- disk watcher for result directory and dataset directory growth.

Host telemetry may call host tools such as `nvidia-smi`, `ps`, and `du`.
Telemetry parsing and summarization must run under rootfs.

If a rootfs-friendly DCGM tool is available, capture low-rate DCGM fields. If
not available, record that it was unavailable and continue with `nvidia-smi`
telemetry. Do not block the run solely on missing DCGM unless the user asked for
DCGM specifically.

Do not enable high-overhead profilers for a full timing claim unless the run is
explicitly diagnostic. If using PyTorch profiler or Nsight, label the attempt
diagnostic.

## Phase 9: Launch and Babysitting

Launch through `experiments/modded_nanogpt_b200/run_speedrun.sh`; it re-enters
rootfs and owns rootfs guard, preflight, launch-readiness sidecars, telemetry,
parsing, summarization, and fail-closed metadata. Do not bypass it with a
manual `torchrun` unless the wrapper is broken and the repair itself is the
current diagnostic task.

Use `torchrun` only inside rootfs. The wrapper may call it; the host shell must
not.

Lane B full production candidate shape:

Use this non-skip template only after the trusted user request contains
`launch-full-b200`; otherwise use the dry-gate form below without launch
authorization.

```bash
experiments/modded_nanogpt_b200/run_speedrun.sh \
  --mode full \
  --lane B \
  --run-id "${RUN_ID}" \
  --attempt-id "${ATTEMPT_ID}" \
  --arm B0 \
  --source "${SOURCE}" \
  --data-manifest "${DATA_MANIFEST}" \
  --result-dir "${RESULT_DIR}" \
  --attention-backend fa2 \
  --mlp-backend triton \
  --verify-sha \
  --launch-authorization=launch-full-b200
```

For a prerequisite dry gate, add `--skip-run` and omit launch authorization.
Dry gates are useful only as prerequisite evidence; they do not count toward the
active two-GPU RSI foundation result. The older 10 full-job campaign target is
historical and requires separate authorization.

Before launch:

1. Record source status before launch.
2. Record rootfs environment.
3. Start telemetry watchers and record their PIDs.
4. Record wall-clock start time.
5. Verify result-local cache directories:
   - `TORCHINDUCTOR_CACHE_DIR`
   - `TRITON_CACHE_DIR`

During launch:

- tee stdout/stderr to `${RESULT_DIR}/run.log`;
- keep the terminal session open until the process exits or a stop condition is
  met;
- check the log at least every 30 seconds during compile/warmup and at least
  every few minutes during training;
- watch for no-output stalls, repeated recompilation, OOM, NCCL errors, CUDA
  illegal instruction, kernel image errors, data-loader errors, and validation
  progress.
- after every completed full attempt, refresh
  `experiments/modded_nanogpt_b200/results/run_index.json` through
  `experiments/modded_nanogpt_b200/summarize.sh` and verify whether it now
  counts the run as baseline evidence.

Babysitting stop conditions:

- Any rank exits nonzero.
- NCCL timeout, communicator abort, or rendezvous failure.
- CUDA OOM or host OOM.
- No log progress for 10 minutes during compile/warmup unless GPU utilization
  and process CPU activity show active compilation.
- No log progress for 5 minutes during steady-state training with low GPU and
  CPU activity.
- Result directory or filesystem is close to full.
- Source tree becomes dirty during Lane A before the run completes.
- Telemetry shows another unrelated job consuming target GPUs.

On a stop condition:

1. Preserve logs and telemetry first.
2. Capture `ps` tree and `nvidia-smi` process list.
3. Gracefully terminate the attempt if it is still running and clearly wedged.
4. If graceful termination fails, kill only the process group for this attempt.
5. Write `blocker.json` and update `analysis.md`.
6. Do not relaunch until the root cause is classified or the run is explicitly
   reclassified diagnostic.

If one full attempt succeeds, stop after parsing, summarizing, refreshing the
run index, and classifying the result. Do not start repeatability, ablation, 8x
reproduction, or a 10-run production campaign until the user explicitly
authorizes that next phase. Keep source, data, rootfs, and backend settings
fixed unless the completed attempt failed and the repair requires a change.
Record every material change in `operator_notes.md` and in the attempt
classification.

## Phase 10: Post-Run Collection

After the process exits:

1. Stop telemetry watchers.
2. Record wall-clock end time and phase durations where available.
3. Capture source status after launch.
4. For Lane A, run `git diff --exit-code` or record dirty status and mark the
   attempt invalid if dirty.
5. Preserve the exact exit code.
6. Run log parsing and summary generation inside rootfs.
7. Preserve null metrics for missing final validation.

If parser/summarizer scripts are not implemented yet, write a rootfs-executed
one-off parser into the result directory or add the missing harness script
according to tickets `03` and `05`. The parser must not run on host Python.

Always refresh the run index after post-run summarization:

```bash
experiments/modded_nanogpt_b200/summarize.sh \
  --results-root experiments/modded_nanogpt_b200/results \
  --output experiments/modded_nanogpt_b200/results/run_index.json
```

If the run index excludes an apparently successful attempt, treat that as a
blocker in the summarizer/evidence contract until the exclusion reason is
understood. Do not manually edit `run_index.json` or force baseline inclusion.

## Phase 11: Result Analysis

Write `${RESULT_DIR}/analysis.md` after parsing. Include:

- classification and whether the attempt is claim-eligible;
- source commit and source cleanliness;
- data manifest identity, shard count, total bytes, and SHA verification state;
- rootfs environment: Python, Torch, CUDA runtime, Triton, FlashAttention;
- GPU inventory;
- final metrics:
  - `val_loss`
  - upstream `train_time`
  - `step_avg`
  - peak allocated memory
  - peak reserved memory
- shell wall-clock, separate from upstream `train_time`;
- telemetry summary:
  - GPU utilization range and median if available;
  - power draw range and median if available;
  - memory used peak per GPU;
  - thermal or clock throttling signs;
  - CPU/RSS signs of compile or data-loader bottlenecks;
  - disk growth and result artifact sizes;
- failure analysis if incomplete:
  - first failing phase;
  - first relevant error line;
  - last 50 meaningful log lines;
  - whether the failure is rootfs, dependency, source policy, data, FA, MLP,
    NCCL, compile, warmup, train, validation, parser, or telemetry.

Only call a run a successful B200 reproduction if all are true:

- mode is `full`;
- lane is `A`;
- preflight passed with NCCL and SHA verification;
- source was clean before and after;
- training reached final validation;
- final `val_loss <= 3.28`;
- summary reports upstream `train_time` from the upstream log;
- shell wall-clock is reported separately.

For Lane B, use `B200 compatibility patchset`, `B200 systems-only`, or
`B200 ML variant` according to the patch classes and metric validity.

Only call the active RSI foundation ready for its next phase when all are true:

- the first authorized two-GPU Lane B full attempt has stopped and been parsed;
- the refreshed run index records the attempt with non-skip
  `training_launched=true`, rootfs sentinel evidence, NCCL checked,
  SHA-verified 900M manifest evidence, source provenance, exit code `0`, and
  final validation metrics, or records a concrete blocker with preserved
  evidence;
- any successful Lane B counted attempt includes `variant_patch.diff` and patch
  classification;
- the final report lists the counted or blocked result directory and any
  excluded attempts that looked close but failed an evidence gate.

A future broader production campaign may reinstate a 10-successful-run target,
but that is not the active small-scale RSI foundation gate.

## Phase 12: Final Response

Report:

- what was attempted;
- which commands were run, with emphasis on rootfs wrappers;
- whether rootfs repairs were needed and exactly what changed;
- final classification;
- current production count from the refreshed run index;
- final metrics or null metrics;
- upstream `train_time` versus shell wall-clock;
- telemetry highlights;
- exact blocker if incomplete;
- paths to `RESULT_DIR`, `preflight_report.json`, `run.log`,
  `summary.json`, and `analysis.md`;
- any uncommitted source changes made by the harness work.

Do not bury the blocker behind a long narrative. If no full run was launched,
say that directly and explain which gate prevented it.

## Minimal Completion Criteria

The session is complete when one of these is true:

- The refreshed run index contains the first accepted non-skip two-GPU Lane B
  full attempt for the active RSI foundation gate, the counted attempt is
  listed, and the evidence is summarized.
- A full Lane A attempt reaches final validation and the result directory has
  preflight, logs, telemetry, summary, and analysis, but a broader reproduction
  or repeatability campaign was not authorized or possible; the reason is
  explicit.
- A Lane B diagnostic or compatibility attempt reaches its authorized endpoint,
  the result directory has preflight, logs, telemetry, summary, and analysis,
  and either the next full job is ready to launch or the next blocker is stated.
- A prerequisite or preflight gate fails, the result directory preserves the
  evidence, the blocker is classified, and the next concrete repair is stated.

Never stop with an unclassified failure while a log or telemetry artifact exists
that can identify the failing phase.
