# Reproduction wrapper and log parser

Type: task
Status: resolved
Blocked by: -

## Requirement

Add the faithful upstream reproduction wrapper and parser for the Modded NanoGPT
B200 benchmark.

Use `.scratch/modded-nanogpt-b200/spec.md` as the canonical spec.

## Scope

Allowed:

- run upstream `run.sh` or equivalent `torchrun` through an experiment-local
  wrapper;
- add rootfs-aware `run_speedrun.sh` and parser/summary wrappers as needed;
- capture complete logs and environment metadata;
- parse final validation loss, upstream `train_time`, `step_avg`, and CUDA
  memory lines;
- write a compact typed per-attempt `summary.json`.

Excluded:

- no upstream source edits for the Lane A reproduction path;
- no optimization ablations;
- no claim that shell wall-clock equals upstream train time;
- no long GPU run unless explicitly authorized for this ticket.
- no host-side Python, `torchrun`, parsing, or summary generation for real GPU
  attempts.

## Verification Evidence

- Parser test or fixture covering representative upstream log lines.
  - `python -m pytest tests/unit_tests/test_modded_nanogpt_b200_parse_log.py -q`
    passed for success, diagnostic/partial, skip-run blocker, analysis
    manifest/telemetry/source-commit/SHA-state reporting, Lane A and Lane B
    full-success claim boundaries, watcher-derived telemetry range rendering,
    separate manifest-declared versus preflight-verified SHA reporting, attempt
    source provenance rendering, environment/GPU/wall-clock/failure category
    rendering, PyTorch and Triton MLP preflight detail rendering, and atomic
    write cases.
- Preflight environment test.
  - `python -m pytest tests/unit_tests/test_modded_nanogpt_b200_preflight.py -q`
    passed for recording Torch, CUDA runtime, Triton, and FlashAttention package
    versions in the preflight environment snapshot.
- Runner test or fixture covering attempt metadata, preflight stop, skip-run
  summary generation, and CLI guard behavior.
  - `python -m pytest tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py -q`
    passed for skip-run attempt bundles, preflight failure summaries, full-mode
    policy rejection, direct-script import/CLI error handling, telemetry
    metadata, `operator_notes.md`, `blocker.json`, telemetry source-status
    copies, prompt-named telemetry logs, live `run.log` streaming through the
    default runner, no-output stall termination, and `exit_code.json`
    preservation, DCGM availability metadata, and conditional `dcgmi dmon`
    capture when a rootfs-friendly `dcgmi` command is available. The runner
    also has regression coverage for the launched-attempt path: after preflight
    passes and `--skip-run` is absent, it starts telemetry before invoking
    `torchrun`, stops telemetry in the `finally` block, writes
    `watcher_status.json` with `state=stopped` and
    `reason=training_finished`, captures stop-condition snapshots under
    `telemetry/stop_ps_tree.txt`, `telemetry/stop_nvidia_smi_processes.txt`,
    and `telemetry/stop_snapshot.json`, preserves `exit_code.json`, and still
    parses a nonzero launched diagnostic into a classified blocker. Parser
    coverage also verifies that `summary.json` embeds the stop snapshot and
    `analysis.md` renders its reason, exit code, and command count.
    Preflight-stop attempts now derive `blocker.json` and
    `summary.json["blocker"]` from the first structured
    `preflight_report.json["failures"]` entry, preserving the exact failing gate
    such as `attention_backend`.
    The runner now also treats the structured preflight report as authoritative:
    a preflight command that exits `0` but writes `preflight_report.json` with
    `ok=false` is converted into exit code `21`, records the first structured
    failure as `blocker.json`, writes launch readiness with
    `ready_to_launch=false`, and stops before any skip-run, launch-authority, or
    training path.
    Full-mode prerequisite reporting is now covered by a red-green regression:
    `launch_readiness.json` and `launch_readiness.md` are written after
    preflight for preflight-failed, `--skip-run`, launch-authority-blocked, and
    launched paths. The report records `ready_to_launch`, `training_launched`,
    `launch_authority_required`, full-mode SHA/NCCL/manifest gates, and
    `blocked_by` without treating an operator dry run or missing launch token as
    a baseline.
    Full-mode launches require the trusted user request itself to contain
    `launch-full-b200`; only after that may the lower-level
    `--launch-authorization=launch-full-b200` marker be supplied. Without the
    marker, the runner writes exit code `21`, `blocker.phase=launch_authority`,
    preserves the launch-readiness report, and stops before invoking `torchrun`.
    The launch-readiness JSON and Markdown also expose
    `launch_authorization_required_token=launch-full-b200` so downstream
    tooling does not have to parse blocker prose.
    Authorized full launches also run the structured active-job scanner after
    preflight and before telemetry/training. If another `torchrun`,
    `train_gpt.py`, or `cached_fineweb10B.py` job is active, the runner writes
    `active_jobs.json`, stops before invoking `torchrun`, preserves
    `blocker.phase=active_jobs`, records `exit_code.phase=active_jobs`, and
    embeds the active-job blocker in launch readiness. The scanner also fails
    closed if process-table discovery itself fails: a nonzero `ps` return code
    sets `ok=false`, records `scan_error`, persists the report, and blocks any
    authorized launch instead of treating an empty scan as safe. That path now
    writes an explicit `active-job scan failed: ...` blocker rather than the
    generic active-process message.
    `attempt.json` records only `launch_authorization_present` and no longer
    echoes any operator-supplied launch-authorization token into immutable
    attempt metadata. Persisted environment metadata is also redacted before it
    is written to `command.env` or `attempt.json["environment"]`; focused
    coverage injects `AWS_CREDENTIALS_FILE`, `WANDB_API_KEY`, and
    `SERVICE_AUTH_HEADER` and verifies that only `<REDACTED>` is stored while
    the real subprocess environment remains available to the command runner.
    Result-directory reuse now fails closed before any attempt artifacts,
    preflight report, or run log can be written: if `result_dir` already exists
    and is non-empty, `run_speedrun.py` returns exit code `21` with an
    `attempt_reuse` stderr blocker and leaves the existing directory contents
    untouched. Focused regression coverage proves preflight is not invoked and
    `attempt.json`, `command.env`, `command.argv`, and `run.log` are absent on
    that path.
    Result-local cache directories are now also fail-closed before preflight.
    A focused regression monkeypatches `TRITON_CACHE_DIR` outside the result
    bundle and proves the runner records `blocker.phase=cache_directories`,
    writes `exit_code.phase=cache_directories`, preserves launch-readiness
    evidence with the blocker, marks telemetry watchers `not_started`, and
    does not invoke preflight or training. The guard requires both
    `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to be actual directories
    under the attempt `result_dir`.
    New `command.argv` files record the replayable
    `run_speedrun.sh` invocation and `attempt.json["command"]["training_argv"]`
    preserves the inner `torchrun` command separately. New attempt directories
    also include result-local `data_manifest.json` pointer records for the
    supplied manifest. Parser summaries embed the command metadata under
    `attempt_command`, and `analysis.md` renders both the replayable attempt
    command and inner training command. Parser summaries also resolve
    result-local data-manifest pointer records and render both the pointer path
    and resolved manifest facts. After preflight returns, the runner now writes
    result-local `environment.json` and `hardware.json` sidecars copied from
    `preflight_report.json["environment"]` and `preflight_report.json["gpus"]`.
    Parser summaries now embed those sidecars, prefer them over stale preflight
    fields for top-level `environment` and `gpus`, render the sidecar kinds in
    `analysis.md`, and fall back to preflight fields for older bundles.
    Fresh rootfs evidence exists at
    `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_sidecars_20260816T025443Z/`:
    it includes `environment.json` kind `preflight_environment`,
    `hardware.json` kind `preflight_gpus`, a result-local
    `data_manifest.json` pointer, replayable `command.argv`, and
    `attempt.json["command"]["training_argv"]`; its launch readiness records
    `ready_to_launch=true`, `training_launched=false`, `skip_run=true`, and
    `verified_sha=true`.
    A later clean-context parser TDD slice fixed result-local manifest pointer
    resolution and SHA propagation. `parse_log.py` now resolves non-absolute
    `data_manifest_pointer.path` values relative to the pointer file
    directory. For pointer manifests, it propagates an explicit top-level
    `verified_sha` from the resolved target only when the pointer lacks
    top-level `verified_sha`; it does not infer `verified_sha` from shard
    `sha256` entries. RED failed because pointer tests showed relative
    result-local targets were not resolved and target manifest fields were not
    loaded. GREEN evidence: parse_log plus summarize tests passed with
    `71 passed in 0.14s`, clean-context verification recorded
    `71 passed in 0.13s`, `py_compile` passed for `parse_log.py` and
    `summarize.py`, and `git diff --check` passed for the scoped files, with a
    note that many scoped files are untracked and static verifier covers
    untracked content separately. Clean-context review found no blocking
    issues, confirmed summarize fail-closed gates remain intact, and confirmed
    `verified_sha` is not inferred from shard hashes.
    A later clean-context runner hardening slice fixed three wrapper root
    causes with unit/static evidence only: Lane B `attention_backend=fa2` now
    runs `setup_flash_attention.sh` before preflight and records
    setup-failure blocker/exit evidence without running preflight or training;
    `DATA_PATH` is now derived from manifest shard roots for absolute and
    repo-relative shard paths, with repo-relative paths resolved using the same
    repo/rootfs cwd semantics as preflight; and initial `attempt.json`
    classification starts with `claim_eligible: False` rather than becoming
    true solely because `mode=full`. TDD evidence recorded RED
    `4 failed, 24 passed` for the initial findings and RED
    `1 failed, 27 passed` for the repo-relative `DATA_PATH` regression, then
    GREEN `29 passed in 0.70s` for
    `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py`.
    Independent clean-context verification recorded `29 passed in 0.61s`,
    `py_compile` passing for `run_speedrun.py`, `git diff --check` passing for
    the runner and focused test file, and static `rg` confirmation of
    `_manifest_shard_paths`, `_data_root_from_manifest`,
    `setup_flash_attention`, `DATA_PATH`, and `claim_eligible: False`
    references. Independent review reported no blocking findings remain and
    that the previous repo-relative `DATA_PATH` blocker is fixed and covered.
    Residual risk: this evidence uses fake command runners/environment
    construction by design, not a live rootfs launch.
- `experiments/modded_nanogpt_b200/parse_log.sh` re-enters
  `scripts/rootfs/enter_rootfs.sh` before invoking parser Python for real
  attempt artifacts.
- `experiments/modded_nanogpt_b200/run_speedrun.sh` re-enters
  `scripts/rootfs/enter_rootfs.sh` before invoking runner Python; rootfs wrapper
  validation rejects invalid full-mode `--allow-previous-stall` with exit `21`
  and no traceback.
- Smoke wrapper run if GPU budget is authorized; otherwise exact blocker and
  dry-run evidence.
- Report uses the common schema fields: `lane`, `mode`, `arm`, `claim_label`,
  `evidence_tier`, `run_id`, `attempt_id`, `environment_class`, and
  `claim_eligible`.
- Report separates completed-run timing from claim-valid timing and sets
  `included_in_baseline_stats=false` unless every validity gate passes.
- Parser embeds structured MLP preflight detail in `summary.json` under
  `preflight_mlp_backend` and renders it in `analysis.md` when present. The
  refreshed
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T001547Z/analysis.md`
  shows `mlp_backend: torch`, `mlp_local_smoke_output_shape: [2, 16, 768]`,
  and `mlp_full_mode_policy: blocked_without_allow_previous_stall`.
  The refreshed
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_triton_20260816T002835Z/analysis.md`
  preserves the historical failed Triton gate with `mlp_backend: triton`,
  `mlp_blocked_kernel: linear_relu_square_kernel`,
  `mlp_blocked_arch: sm100`, `mlp_failure_class: triton_compile`, and
  `mlp_compiler_pass: TritonNvidiaGPUOptimizeTMemLayoutsPass`.
  The current launch-ready dry gate at
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_triton_20260816T005014Z/analysis.md`
  renders `preflight_ok: True`, `mlp_backend: triton`,
  `mlp_local_smoke_backend: triton`,
  `mlp_local_smoke_output_shape: [2, 16, 768]`, and null MLP blocker fields.
  The current authority-guard artifact at
  `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/analysis.md`
  renders the same green preflight state but preserves
  `failure_category: launch_authority` and the explicit launch-token blocker.
- Lane B prelaunch source collection now writes
  `variant_patch_classification.json` beside `variant_patch.diff`, so every
  changed source file has a machine-readable patch class and first Lane A
  blocker before preflight and before any possible training launch. The current
  rootfs authority-guard artifact
  `experiments/modded_nanogpt_b200/results/lane_b_full_prelaunch_patch_guard_20260816T012647Z/`
  records `train_gpt.py` as `hardware-detection` for `FA3 no kernel image on
  B200` and `triton_kernels.py` as `kernel-compat` for the Triton sm100 custom
  kernel blocker. The parser now embeds that sidecar in `summary.json` under
  `variant_patch_classification`, and `analysis.md` renders the diff path,
  changed-file count, patch class, and first blocker for each file.
  A focused runner regression now covers the negative case: a full Lane B
  attempt with an unclassified source patch writes
  `blocker.phase=variant_patch_classification`,
  `exit_code.phase=variant_patch_classification`, and `exit_code=21` before
  invoking preflight or training, even when
  `--launch-authorization=launch-full-b200` is supplied. The same early-stop
  path now writes `launch_readiness.json` with `ready_to_launch=false`,
  `training_launched=false`, and `blocked_by` set to the source-classification
  blocker, without inventing NCCL or SHA blockers before preflight has run.
  Parser coverage also invalidates full Lane B baseline eligibility when a
  `variant_patch_classification.json` sidecar contains any `unclassified`
  source patch, while preserving an explicit runner `blocker.json` message when
  present.
- Parser summaries now embed `launch_readiness.json` under
  `summary.json["launch_readiness"]`, and `analysis.md` renders
  `launch_ready_to_launch`, `launch_training_launched`,
  `launch_authority_required`, `launch_authorization_required_token`,
  `launch_preflight_ok`, and `launch_blocked_by`.
- Parser summaries now embed compact, allowlisted result artifact byte sizes
  under `summary.json["artifact_sizes"]` and render
  `artifact_total_known_bytes` plus one `artifact_size` row per known file in
  `analysis.md`. The collector intentionally covers top-level attempt sidecars
  and known telemetry files only, so Phase 11 can report result-artifact size
  evidence without recursively scanning generated source, data, or cache trees.
- Parser summaries now also embed `summary.json["claim_validation"]` and render
  a `Claim Validation` section in `analysis.md`. It reports the prompt's
  successful-reproduction gates directly, including full mode, Lane A,
  preflight, NCCL, SHA verification, source cleanliness, final validation,
  `val_loss <= 3.28`, upstream `train_time`, separate shell wall-clock, and the
  first claim blocker.
- Parser telemetry summaries now derive `telemetry.signs` and render explicit
  thermal-or-clock throttling and CPU-or-RSS bottleneck sign lines in
  `analysis.md`. The signs are informational and preserve the raw watcher
  ranges; focused parser coverage drives hot/low-clock and high-CPU/RSS
  fixtures.
- Parser success for full attempts now depends on the structured attempt
  sidecars, not only final metrics. A full attempt is excluded from `ok` and
  `included_in_baseline_stats` unless launch-readiness records an actual
  launched non-skip, non-stall-override attempt, `exit_code.json` records
  `phase=training` and `exit_code=0`, rootfs sentinel evidence is present, NCCL
  and SHA-verified full 900M manifest evidence passed, the data source commit
  is pinned, `launch_readiness.data_manifest` matches either the result-local
  manifest pointer or its resolved manifest target when present, and the
  visible GPU inventory matches the declared B200 allocation. The active RSI
  foundation trial requires exactly 2x B200; 8x B200 remains a separate broader
  reproduction claim. Focused regressions first proved a final-metric
  full summary with no launch/rootfs/data sidecars was incorrectly accepted,
  then proved a copied launch-readiness sidecar with a mismatched data-manifest
  path was incorrectly accepted, and now prove an otherwise valid full attempt
  with `exit_code.json` recording `exit_code=7` is blocked at `training`; those
  paths are now blocked at `launch_evidence`, `data_manifest`, or `training`
  respectively.
- Parser summaries now preserve the actual first relevant log error separately
  from the authoritative structured blocker. `analysis.md` renders both
  `first relevant error` and `blocker_message`, so a runner-side
  `blocker.json` no longer hides the first diagnostic error line from the log.
  The first-error scanner now uses bounded log-error patterns, so benign JSON
  keys such as `allow_previous_stall` do not mask a later real stall, timeout,
  or error line.
- `analysis.md` now normalizes `failure_category` to the execution prompt's
  vocabulary where possible. For example, an exact phase such as `mlp_backend`
  remains the first failing phase, while the category renders as `MLP`; NCCL
  log failures render as `NCCL`.
- Parser summaries now use captured `source_status_after.txt` when present.
  Full Lane A attempts with final validation metrics are still invalid if
  post-run source cleanliness evidence is missing or dirty; the summary records
  `blocker.phase=source_policy` instead of adding the attempt to baseline
  statistics.
- Host-launched wrapper re-enters rootfs before preflight, launch, parsing, and
  summary generation.
- Direct Python execution of wrapper-owned tools now fails closed outside
  rootfs. `cli_guard.py` is called from the `__main__` entrypoints for
  `run_speedrun.py`, `parse_log.py`, `summarize.py`, `prepare_data.py`,
  `fetch_upstream.py`, `preflight.py`, and `diagnose_mlp_backend.py`, so
  host-side `python experiments/modded_nanogpt_b200/<tool>.py ...` exits `21`
  before parsing logs, summarizing results, preparing data, cloning source,
  running preflight, scanning jobs, or writing diagnostic artifacts.
- Shell wrappers also source `rootfs_guard.sh` after the rootfs re-entry
  branch. A host process that forges `TORCHTITAN_IN_ROOTFS=1` now exits `21`
  before package installation, parsing, summarization, preflight, source fetch,
  data prep, active-job scanning, or diagnostics unless it is actually running
  from `/workspace/torchtitan` with the rootfs sentinel present.
- Wrapper-owned diagnostic skip-run evidence:
  `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_watchers_20260815T230238Z/`
  records `preflight_report.json`, `attempt.json`, `run.log`,
  `wall_clock.json`, `exit_code.json`, `blocker.json`, `operator_notes.md`,
  `summary.json`, `analysis.md`, `telemetry/rootfs_environment.json`, and
  telemetry source-status copies plus `process_watch.log`, `disk_watch.log`,
  and `gpu_processes.log`; it is classified Lane B diagnostic and blocked at
  `not_launched` because `--skip-run` intentionally did not start training.
  Its `summary.json` and `analysis.md` were refreshed via
  `experiments/modded_nanogpt_b200/parse_log.sh`, so parser execution re-entered
  the rootfs and the analysis includes explicit null telemetry range fields for
  the intentionally not-started watchers plus separate attempt-source and
  data-source commit fields.
- Historical wrapper-owned diagnostic skip-run evidence:
  `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_env_20260815T232745Z/`
  recorded the then-required bundle shape, re-ran rootfs preflight without
  launching training, recorded `flash_attention: 2.8.3.post1`, historical 8x
  B200 GPU inventory, shell wall-clock, and `failure_category: not_launched`,
  and remains excluded from baseline stats. It is superseded as current handoff
  evidence by the later two-GPU runtime-env prerequisite artifact.
- Historical DCGM-aware wrapper-owned diagnostic skip-run evidence:
  `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_dcgm_20260815T233425Z/`
  re-ran rootfs preflight without launching training and recorded
  `telemetry/dcgm_status.json` with `state=unavailable`,
  `reason="no rootfs-friendly DCGM command found"`, and checked commands
  `dcgmi` plus `dcgmproftester`. Its `summary.json` embeds the DCGM status
  under `telemetry.dcgm`, `analysis.md` renders `dcgm_state` and `dcgm_reason`,
  `rootfs_marker: 1`, `flash_attention: 2.8.3.post1`, historical 8x B200
  inventory, `manifest_verified_sha: null`, `preflight_verified_sha: False`,
  null final metrics, and `failure_category: not_launched`. It is superseded as
  current handoff evidence by the later two-GPU runtime-env prerequisite
  artifact.
