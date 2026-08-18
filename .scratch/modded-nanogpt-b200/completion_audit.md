# Modded NanoGPT B200 Completion Audit

Status: active
Updated: 2026-08-16

## Objective

Execute `.scratch/modded-nanogpt-b200/execution_prompt.md` until the
TorchTitan-local Modded NanoGPT B200 harness is productionized enough to carry
the experiment as far as current authority and gates safely permit.

Completion requires one of the prompt's terminal states:

- a full Lane A attempt reaches final validation with preflight, logs,
  telemetry, summary, and analysis;
- a Lane B diagnostic or compatibility attempt reaches its authorized endpoint
  with preflight, logs, telemetry, summary, and analysis;
- a prerequisite or preflight gate fails with preserved evidence, a classified
  blocker, and a concrete next repair.

The current session is not a full reproduction claim. No full long GPU job has
been authorized.

## Current Evidence Snapshot

Latest prerequisite artifact:

- `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z/`
- `summary.json` classification: Lane B full, arm `B0`,
  `claim_eligible=true`.
- `summary.json` blocker: `phase=not_launched`, message records that
  `--skip-run` was requested and training was not launched.
- `summary.json` final metrics: `val_loss`, `train_time`, `step_avg`,
  `peak_allocated_memory`, and `peak_reserved_memory` are null.
- `summary.json["claim_validation"]` records
  `successful_b200_reproduction=false` and
  `first_blocker="final validation was not reached"`, and the refreshed
  `analysis.md` renders the same `Claim Validation` section. This keeps the
  prompt's successful-reproduction gates visible even when the artifact is a
  launch-ready prerequisite rather than a completed run.
- `summary.json` environment: Python 3.12.3, Torch 2.13.0+cu132, CUDA runtime
  13.2, Triton 3.7.1, FlashAttention 2.8.3.post1.
- `summary.json` GPU inventory: 8x NVIDIA B200 with compute capability 10.0.
- `environment.json` is a schema-v1 `preflight_environment` sidecar copied
  from `preflight_report.json`.
- `hardware.json` is a schema-v1 `preflight_gpus` sidecar copied from
  `preflight_report.json` and records the 8x B200 inventory.
- `data_manifest.json` is a schema-v1 `data_manifest_pointer` to the refreshed
  full manifest.
- `telemetry/dcgm_status.json`: `state=unavailable`,
  `reason="no rootfs-friendly DCGM command found"`.
- `summary.json["telemetry"]["signs"]` records
  `thermal_or_clock_throttling=false` and `cpu_or_rss_bottleneck=false` for the
  latest skip-run artifact, and the refreshed `analysis.md` renders those sign
  lines. Launched attempts with watcher ranges get the same fields populated
  from temperature/SM-clock and process CPU/RSS evidence.
- `launch_readiness.json` records `ready_to_launch=true`,
  `training_launched=false`, `launch_authority_required=true`,
  `launch_authorization_required_token=launch-full-b200`, `nccl_checked=true`,
  `data_manifest_checked=true`, `verified_sha=true`, `skip_run=true`, and
  `blocked_by=[]`.
- `attempt.json["command"]["argv"]` records the replayable
  `run_speedrun.sh` command with `--skip-run` and without
  `--launch-authorization=launch-full-b200`.
  `launch_authorization_present=false`, and
  `attempt.json["command"]["training_argv"]` records the inner
  `torchrun --standalone --nproc_per_node=8 train_gpt.py` command that was not
  launched.
- `telemetry/rootfs_environment.json` records `cwd=/workspace/torchtitan`,
  `torchtitan_in_rootfs=1`, and `workspace_sentinel_exists=true`, proving the
  host-launched `run_speedrun.sh` wrapper re-entered rootfs before
  `run_speedrun.py` executed.
- `variant_patch_classification.json` records the Lane B source differences:
  `train_gpt.py` as `hardware-detection` for the first Lane A blocker
  `FA3 no kernel image on B200`, and `triton_kernels.py` as `kernel-compat`
  for the first Lane A blocker
  `Triton sm100 custom kernel compile/runtime blocker`.
  The paired `variant_patch.diff` and classification JSON are now written
  before preflight and before any possible training launch.
- `summary.json` embeds the same classification under
  `variant_patch_classification`, and `analysis.md` renders
  `variant_patch_diff`, `variant_patch_file_count`, and one
  `variant_patch_file` row per changed file.
- `preflight_report.json` records `mlp_backend` as passing with structured local
  smoke detail: `backend=triton`,
  `local_smoke.output_shape=[2, 16, 768]`.
- `summary.json` carries the same structured MLP evidence under
  `preflight_mlp_backend`, and the refreshed `analysis.md` renders
  `mlp_backend=triton`, `mlp_local_smoke_backend=triton`,
  `mlp_local_smoke_output_shape=[2, 16, 768]`, and null MLP blocker fields.
- The paired PyTorch MLP full-mode dry gate remains at
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T001547Z/`;
  it records the successful local PyTorch MLP smoke
  `local_smoke.output_shape=[2, 16, 768]` and the same full-mode policy block.
- A bounded rootfs Triton MLP diagnostic artifact now exists at
  `experiments/modded_nanogpt_b200/results/mlp_diag_triton_shared_20260816T005734Z/mlp_diagnostic.json`.
  It passes inside rootfs with `ok=true`, rootfs marker `1`, Torch
  `2.13.0+cu132`, Triton `3.7.1`, CUDA runtime `13.2`, and
  `detail.output_shape=[2, 16, 768]`.
  The diagnostic now delegates to the same Triton smoke helper used by
  full-mode preflight, so the bounded diagnostic and launch-readiness gate do
  not maintain duplicate Triton MLP probe code.
  The prior failing diagnostic at
  `experiments/modded_nanogpt_b200/results/mlp_diag_triton_20260816T004154Z/mlp_diagnostic.json`
  identified the fixed bug as a `triton_backward_shape_mismatch` at
  `FusedLinearReLUSquareFunction.backward`.

Current full manifest:

- `experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`
- `schema_version=1`, `token_budget=900M`, `num_files=10`,
  `total_bytes=2000010240`, top-level `verified_sha=true`, and source commit
  `ecbb586296d3dac36fd206211f25d63bad4a6b35`.

Current run index:

- `experiments/modded_nanogpt_b200/results/run_index.json`
- `total_attempts=24`
- `baseline_stats.count=0`
- `len(launch_prerequisite_attempts)=1`
- `len(launch_ready_attempts)=0`
- Latest refresh used the rootfs-aware wrapper
  `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`.
  This exercised the current run-index code, including the independent
  sibling `exit_code.json` baseline gate and active-job evidence gate, against
  real ignored artifacts. The refreshed index now has one
  `launch_prerequisite_attempts` row pointing to
  `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z/summary.json`.
  It is not a launch-ready row because `skip_run=true`; no non-skip
  launch-ready row exists.
  The refreshed index carries `environment`, `gpus`, `gpu_count`,
  `environment_sidecar`, `hardware_sidecar`, `claim_validation`,
  `telemetry_signs`, and `launch_readiness_exclusion` fields in attempt
  records.
  The newest indexed sidecar-backed attempt is
  `lane_b_full_skiprun_refresh_20260816T111056Z`, with
  `environment_sidecar.kind=preflight_environment`,
  `hardware_sidecar.kind=preflight_gpus`,
  `claim_validation.successful_b200_reproduction=false`,
  `telemetry_signs.thermal_or_clock_throttling=false`, `ready_to_launch=true`,
  `training_launched=false`, and `skip_run=true`.
- A later clean-context subagent refreshed the ignored real run index through
  the approved rootfs-aware summarizer command above; the command exited `0`.
  The subagent did not run launch, GPU, training, preflight, data-prep,
  download, pip, CUDA, NCCL, `torchrun`, or full-launch commands. The refreshed
  index now reports `total_attempts=24`, `baseline_stats.count=0`,
  `len(launch_prerequisite_attempts)=1`, and
  `len(launch_ready_attempts)=0`. The one prerequisite row points to the
  refreshed `lane_b_full_skiprun_refresh_20260816T111056Z` summary and remains
  non-launch evidence because `skip_run=true`. Historical SHA-gated exclusion
  examples include
  `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010335Z/summary.json`,
  `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/summary.json`,
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000411Z/summary.json`,
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000617Z/summary.json`,
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T001037Z/summary.json`,
  and
  `experiments/modded_nanogpt_b200/results/lane_b_full_rootfs_guard_20260816T031154Z/summary.json`.
  The requested SHA-exclusion `jq` query failed on a null phase during
  operator inspection, so the subagent used a null-safe equivalent; this was an
  operator-query adjustment, not evidence of a summarizer failure.
  These stricter SHA gates intentionally demote older generated artifacts. The
  refreshed index proves one non-launch prerequisite row, not runtime launch or
  baseline completion.
- Parser baseline eligibility is now fail-closed for full attempts. A full
  attempt with final validation metrics is not `ok` and is not included in
  baseline stats unless `launch_readiness.training_launched=true`, the attempt
  is not a skip-run, no known-stall override was used, rootfs sentinel evidence
  is present, NCCL was checked, the full 900M manifest is SHA-verified against
  the pinned source commit, `launch_readiness.data_manifest` matches either the
  result-local manifest pointer or its resolved manifest target when present,
  and the GPU inventory is 8x B200.
- Run-index aggregation now also fails closed. `summarize.py` does not trust a
  stale or malformed summary's `included_in_baseline_stats=true` bit unless the
  normalized classification is Lane A or B, `mode=full`,
  `claim_eligible=true`, and `summary.ok=true`. A focused regression proves a
  diagnostic summary with a forged baseline-inclusion bit is still excluded
  from both global and grouped baseline statistics.
- Run-index baseline aggregation independently rechecks full-baseline evidence
  before counting a summary row. A baseline row must have actual
  `launch_readiness.training_launched=true`, sibling `exit_code.json` with
  `phase=training` and `exit_code=0`, no skip-run or known-stall override,
  `preflight_ok=true`, rootfs sentinel evidence, full-mode SHA/NCCL gate
  evidence, 900M manifest gate shape, and 8x B200 GPU inventory. Focused
  regressions prove stale full-mode summaries with `ok=true`,
  `included_in_baseline_stats=true`, and good metrics but either missing
  launch/rootfs evidence or nonzero training-exit evidence are preserved with
  `baseline_evidence_error` and excluded from global and grouped baseline
  stats.
- Run-index baseline aggregation also independently validates baseline-critical
  metrics. Numeric `val_loss`, `train_time`, and `step_avg` are required, and
  `val_loss` must be `<= 3.28`; a focused regression proves a stale full-mode
  summary with `final_metrics.val_loss=3.29` is preserved with
  `metrics_error` but excluded from baseline stats and launch-ready rows.
- Run-index launch readiness now prefers sibling `launch_readiness.json` when
  present but falls back to embedded `summary.json["launch_readiness"]` when
  the sidecar is absent. A focused regression covers archived or copied
  summary bundles that retain embedded readiness evidence but not the sibling
  sidecar.
- Run-index launch readiness now requires parsed manifest evidence from the
  attempt summary before surfacing a row as launch-ready. This reflects the
  prelaunch failure pattern: readiness sidecars are useful operator gates, but
  a sidecar alone is not enough to make a copied or stale attempt auditable
  before a future full launch. A focused regression covers a sidecar-only
  ready attempt with missing `data_manifest_summary`; it remains preserved as a
  failed/not-launched attempt but is excluded from `launch_ready_attempts`.
- Run-index launch readiness now also requires parsed 8x B200 GPU evidence
  from the attempt summary before surfacing a row as launch-ready. A focused
  regression covers a sidecar-ready full attempt whose summary records only two
  B200 GPU entries; it remains preserved as a failed/not-launched attempt but
  is excluded from `launch_ready_attempts`.
- Run-index launch readiness now also requires parsed rootfs sentinel evidence
  from the attempt summary before surfacing a row as launch-ready. A focused
  regression covers a sidecar-ready full attempt whose summary lacks
  `telemetry.rootfs`; it remains preserved as a failed/not-launched attempt but
  is excluded from `launch_ready_attempts`.
- Run-index launch readiness now also requires parsed source provenance from
  the attempt summary before surfacing a row as launch-ready. A focused
  regression covers a sidecar-ready full attempt whose `summary.source.commit`
  is stale; it remains preserved as a failed/not-launched attempt but is
  excluded from `launch_ready_attempts`.
- Run-index launch readiness now also requires parsed Lane B patch provenance
  from the attempt summary before surfacing a Lane B row as launch-ready. A
  focused regression covers a sidecar-ready Lane B full attempt whose summary
  lacks `variant_patch_classification`; it remains preserved as a
  failed/not-launched attempt but is excluded from `launch_ready_attempts`.
- `launch_prerequisite_attempts` is now full-mode prerequisite readiness only.
  The run index requires Lane A or B, `mode=full`, `claim_eligible=true`,
  `ready_to_launch=true`, `training_launched!=true`, no readiness blockers,
  `preflight_ok=true`, `full_mode_gates.full_mode=true`,
  `full_mode_gates.data_manifest_checked=true`,
  `full_mode_gates.verified_sha=true`, and
  `full_mode_gates.nccl_checked=true`, plus the full-manifest shape
  `manifest_token_budget=900M`, `manifest_num_files=10`, and
  `manifest_total_bytes=2000010240`, and the required full-launch token marker
  `launch_authorization_required_token=launch-full-b200`; a focused regression
  proves a
  diagnostic ready-but-not-launched sidecar stays under
  `diagnostic_or_failed_attempts` and does not enter launch-prerequisite or
  launch-ready rows.
  The narrower `launch_ready_attempts` list excludes `skip_run=true` dry gates,
  so it contains only non-skip prerequisite rows that could launch after adding
  the explicit full-launch authorization token. A focused regression proves a
  dry prerequisite and a non-skip authority-guard prerequisite are both
  preserved in `launch_prerequisite_attempts`, while only the non-skip row
  enters `launch_ready_attempts`.
  When a launch-readiness sidecar records `lane` or `mode`, those fields must
  match the summary classification; a focused regression proves a stale or
  copied sidecar with mismatched `lane` is not surfaced as launch-ready.
  When a launch-readiness sidecar records `run_id` or `attempt_id` at top
  level or under its nested `classification` object, every present value must
  match the summary classification; focused regressions prove copied sidecars
  with mismatched attempt identity and contradictory top-level versus nested
  identity are preserved but excluded from launch-ready rows. New
  `run_speedrun.py` readiness sidecars write `run_id` and `attempt_id` at top
  level.
  Another focused regression proves an internally contradictory readiness
  sidecar with `ready_to_launch=true`, a `blocked_by` entry, and
  `verified_sha=false` is preserved but excluded from launch-ready rows.
  Another focused regression proves a sidecar with `ready_to_launch=true`,
  `verified_sha=true`, and `nccl_checked=true` but a smoke-shaped manifest is
  preserved but excluded from launch-ready rows.
  Another focused regression proves a sidecar with correct manifest shape but
  `data_manifest_checked=false` is preserved but excluded from launch-ready
  rows.
  Another focused regression proves a sidecar with all full-mode gates set but
  no `launch_authorization_required_token` is preserved but excluded from
  launch-ready rows.
  Another focused regression proves a sidecar with all full-mode gates set and
  the correct token but `launch_authority_required=false` is preserved but
  excluded from launch-ready rows.

Fresh verification for this audit:

- 2026-08-16 clean-context parser manifest-pointer SHA propagation slice:
  a clean-context TDD implementation changed only
  `experiments/modded_nanogpt_b200/parse_log.py` and
  `tests/unit_tests/test_modded_nanogpt_b200_parse_log.py`. The root cause was
  that `data_manifest_pointer.path` values that were relative to the pointer
  file directory were not resolved before loading the target manifest, so
  result-local pointer targets could leave target manifest fields absent from
  parser summaries. The fix makes `parse_log.py` resolve non-absolute
  `data_manifest_pointer.path` values relative to the pointer file directory.
  For pointer manifests, the parser propagates an explicit top-level
  `verified_sha` from the resolved target only when the pointer itself lacks a
  top-level `verified_sha`. It does not infer `verified_sha` from shard
  `sha256` entries. RED failed because the new pointer tests showed relative
  result-local pointer targets were not resolved and target manifest fields
  were not loaded. GREEN evidence: parse_log plus summarize tests passed with
  `71 passed in 0.14s` from implementation, and clean-context verification
  independently recorded `71 passed in 0.13s`. `py_compile` passed for
  `parse_log.py` and `summarize.py`. `git diff --check` passed for the scoped
  files, with the note that many scoped files are untracked and the static
  verifier covers untracked content separately. Clean-context review found no
  blocking issues, confirmed summarize fail-closed gates remain intact, and
  confirmed `verified_sha` is not inferred from shard hashes.
- 2026-08-16 clean-context parser-pointer real-index refresh:
  after the parser fix, the ignored real index was refreshed through the
  approved rootfs-aware summarizer:
  `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`.
  Historical refreshed facts, now superseded by the
  `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh:
  `total_attempts=23`, `baseline_count=0`, `launch_prerequisite_count=0`,
  `launch_ready_count=0`, `stale_count=20`, and
  `launch_readiness_exclusion_stats.count=20`.
  Exclusion phases were `data_manifest_summary.verified_sha=11`,
  `launch_readiness=7`, and `preflight_data_manifest.verified_sha=2`, plus
  3 diagnostic failed-attempt rows with no `launch_readiness_exclusion` phase.
  The current real demotions did not clear because existing resolved manifests
  still lack top-level `verified_sha`. The parser fix protects future pointer
  manifests that carry explicit `verified_sha`; it does not mutate stale
  generated artifacts and does not infer SHA from shard hashes. No launch, GPU,
  training, preflight, data-prep, download, pip, CUDA, NCCL, `torchrun`, or
  full-launch command was run. Overall completion remains incomplete: no full
  Lane A/B baseline exists. The later refreshed index now has one skip-run
  prerequisite row, no non-skip launch-ready row, and full launch still
  requires explicit authorization.
- 2026-08-16 clean-context manifest SHA evidence hardening slice:
  a clean TDD implementation changed only
  `experiments/modded_nanogpt_b200/prepare_data.py`,
  `experiments/modded_nanogpt_b200/preflight.py`,
  `experiments/modded_nanogpt_b200/summarize.py`,
  `tests/unit_tests/test_modded_nanogpt_b200_prepare_data.py`,
  `tests/unit_tests/test_modded_nanogpt_b200_preflight.py`, and
  `tests/unit_tests/test_modded_nanogpt_b200_summarize.py`. The root cause was
  that manifests carried per-shard `sha256` entries but no top-level
  `verified_sha` state; parser and run-index launch-prerequisite logic read the
  parsed manifest SHA state from top-level `verified_sha`, so prerequisite
  indexing could rely on sidecar or preflight SHA while parsed manifest SHA was
  null. The fix makes `prepare_data.build_manifest` write top-level
  `verified_sha: true` when it computes shard SHA entries, makes full-mode
  preflight with `verify_sha=True` reject manifests whose top-level
  `verified_sha` is missing or not true while still recomputing hashes for
  accepted manifests, and makes summarize launch-prerequisite gating require
  both parsed manifest SHA evidence and preflight SHA evidence.
  TDD evidence recorded RED `3 failed, 45 passed` for the prepare-data,
  preflight, and summarize SHA tests before implementation, then GREEN
  `48 passed in 0.15s` for the focused tests. `py_compile` passed for
  `prepare_data.py`, `preflight.py`, and `summarize.py`, and
  `git diff --check` passed for the scoped code and tests. Clean-context
  verification independently recorded `48 passed in 0.08s`, `py_compile`
  passing, `git diff --check` passing, and static `rg` confirmation of
  `verified_sha`, `manifest_verified_sha`, and `require_sha` references.
  Clean-context review found no blocking correctness bugs and noted only
  non-blocking coverage gaps. A follow-up test-only slice changed only
  `tests/unit_tests/test_modded_nanogpt_b200_preflight.py` and
  `tests/unit_tests/test_modded_nanogpt_b200_summarize.py`, adding a positive
  preflight recomputation test for top-level `verified_sha=true` and a
  summarize test for `preflight_data_manifest.verified_sha=false` with parsed
  manifest SHA true. That follow-up recorded `47 passed in 0.15s` and
  `git diff --check` passing; no production code change was needed. No launch,
  GPU, training, data-prep, download, pip, CUDA, NCCL, `torchrun`, or
  artifact-mutating command was run. This does not refresh the current
  `run_index.json`, does not create a Lane A or Lane B baseline, and does not
  authorize or imply full-launch readiness.
- 2026-08-16 clean-context run-index refresh after SHA hardening:
  a clean-context subagent refreshed the ignored real `run_index.json` through
  the approved rootfs-aware summarizer command
  `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`,
  which exited `0`. Historical refreshed facts, now superseded by the
  `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh:
  `total_attempts=23`, `baseline_count=0`, `launch_prerequisite_count=0`,
  `launch_ready_count=0`,
  `launch_readiness_exclusion_stats.count=20`,
  `by_phase.data_manifest_summary.verified_sha.count=11`,
  `by_phase.preflight_data_manifest.verified_sha.count=2`, and
  `by_phase.launch_readiness.count=7`; both prerequisite lists are empty.
  No launch, GPU, training, preflight, data-prep, download, pip, CUDA, NCCL,
  `torchrun`, or full-launch command was run in that clean-context refresh.
  This zero-prerequisite state is historical and was superseded by the later
  `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh, which reports
  one skip-run launch-prerequisite row and zero non-skip launch-ready rows.
  Overall completion remains blocked: no full Lane A/B baseline exists, full
  launch still requires explicit authorization, and older generated artifacts
  are intentionally demoted by stricter SHA evidence gates.
- 2026-08-16 clean-context stale/demoted artifact indexing and real-index
  refresh: a clean-context TDD implementation changed only
  `experiments/modded_nanogpt_b200/summarize.py` and
  `tests/unit_tests/test_modded_nanogpt_b200_summarize.py`, adding top-level
  `stale_or_demoted_artifacts` to `run_index.json`. Each stale/demoted record
  includes the `summary` path, `reason`, `message`, `operator_action`,
  lane/mode/claim-label context, and `launch_readiness_exclusion` when
  relevant. The list is bounded by
  `MAX_STALE_OR_DEMOTED_ARTIFACTS = 20`. Tests cover legacy summary schema, a
  ready-ish sidecar demoted by missing `active_jobs`, bounded list behavior,
  and malformed summaries included in stale/demoted artifacts. Evidence:
  RED `3 failed, 39 passed` before implementation, GREEN
  `42 passed in 0.08s` after implementation, and follow-up test-only gap
  closure `43 passed in 0.07s` with no production code needed. Clean-context
  verification recorded `42 passed in 0.07s`, `py_compile` passing,
  `git diff --check` passing, and static `rg` confirmation of
  `stale_or_demoted_artifacts`, `MAX_STALE_OR_DEMOTED_ARTIFACTS`,
  `legacy_summary_schema`, and `operator_action`. Clean-context review found no
  blocking findings. The ignored real index was refreshed through the approved
  rootfs-aware summarizer command
  `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`,
  which exited `0`. Historical refreshed facts, now superseded by the
  `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh:
  `total_attempts=23`, `baseline_count=0`, `launch_prerequisite_count=0`,
  `launch_ready_count=0`, `len(stale_or_demoted_artifacts)=20`, and
  `len(stale_or_demoted_artifacts) <= 20` returned true. Representative
  stale/demoted records include
  `experiments/modded_nanogpt_b200/results/lane_a_full_20260815T045618Z/summary.json`
  with `reason=legacy_summary_schema`,
  `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010335Z/summary.json`
  and
  `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/summary.json`
  with `data_manifest_summary.verified_sha must be true`, and
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000411Z/summary.json`
  and
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000617Z/summary.json`
  with `preflight_data_manifest.verified_sha must be true`. No launch, GPU,
  training, preflight, data-prep, download, pip, CUDA, NCCL, `torchrun`, or
  full-launch command was run. The overall objective remains incomplete: no
  full Lane A/B baseline exists. The later refreshed index now has one
  skip-run prerequisite row, no non-skip launch-ready row, and full launch still
  requires explicit authorization.
- 2026-08-16 clean-context runner hardening slice:
  a clean implementation subagent changed only
  `experiments/modded_nanogpt_b200/run_speedrun.py` and
  `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py`. It fixed three
  runner root causes without launching training: Lane B
  `attention_backend=fa2` now invokes `setup_flash_attention.sh` before
  preflight, and setup failure writes blocker/exit evidence before any
  preflight or training command; `DATA_PATH` is derived from manifest shard
  roots for both absolute and repo-relative shard paths, with repo-relative
  paths resolved according to the repo/rootfs cwd semantics used by preflight;
  and initial `attempt.json` classification starts with
  `claim_eligible: False` instead of becoming true merely because
  `mode=full`.
  TDD evidence from that subagent recorded RED runs of `4 failed, 24 passed`
  for the initial three findings and `1 failed, 27 passed` for the
  repo-relative `DATA_PATH` regression before the fix, then GREEN
  `29 passed in 0.70s` for
  `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py`.
  It also recorded `py_compile` passing for `run_speedrun.py` and
  `git diff --check` passing for `run_speedrun.py` plus the focused test file.
  Clean-context verification independently recorded
  `29 passed in 0.61s`, `py_compile` passing, `git diff --check` passing, and
  static `rg` confirmation of `_manifest_shard_paths`,
  `_data_root_from_manifest`, `setup_flash_attention`, `DATA_PATH`, and
  `claim_eligible: False` references. Clean-context review found no blocking
  findings remain; the previous repo-relative `DATA_PATH` blocker is fixed and
  covered by tests. Residual risk is intentional for this slice: tests use fake
  command runners and environment construction, not a live rootfs launch. No
  launch, GPU, training, data-prep, download, pip, CUDA, or `torchrun` command
  was run. The overall objective remains incomplete: no full Lane A/B baseline
  exists, the current run index remains zero baseline and zero launch-ready,
  and a full launch still requires explicit user authorization.
- 2026-08-16 clean-context static verifier slice:
  a clean-context TDD implementation added only
  `experiments/modded_nanogpt_b200/verify_static.py` and
  `tests/unit_tests/test_modded_nanogpt_b200_verify_static.py`. It closes the
  static-verification gap where `git diff --check` can skip untracked harness
  files. The verifier enumerates tracked modified files, staged files, and
  untracked files from the scoped paths `.scratch/modded-nanogpt-b200`,
  `experiments/modded_nanogpt_b200`, and
  `tests/unit_tests/test_modded_nanogpt_b200_*.py`; checks `.py`, `.sh`, and
  `.md`; excludes generated and ignored trees including
  `.scratch/trae-pytest-tmp`, experiment `data/`, `results/`, `sources/`, and
  `__pycache__`; fails on trailing whitespace, missing final newline, Python
  compile errors, and shell syntax errors via `bash -n`; and supports
  `--list-files` and `--json`. TDD evidence recorded RED `5 failed` before
  the verifier existed, GREEN `5 passed`, and after staged-change coverage was
  added, `6 passed`. `py_compile` passed for the verifier and test. Focused
  static verification recorded
  `python experiments/modded_nanogpt_b200/verify_static.py --list-files`
  printing 39 candidate files, `--json` reporting 39 candidate files, and the
  real verifier passing with `Static verification passed for 39 file(s).` The
  full focused Modded NanoGPT B200 unit suite then passed with
  `114 passed in 1.11s`. Clean-context review found no blocking issues and
  noted one staged-change coverage gap, which was closed by the test-only
  follow-up; production code did not need a change for staged files. No
  launch, GPU, training, preflight, data-prep, download, pip, CUDA, NCCL,
  `torchrun`, full-launch, or artifact-mutating command was run. The overall
  objective remains incomplete: no full Lane A/B baseline exists, the later
  refreshed index has one skip-run launch-prerequisite row and zero non-skip
  launch-ready rows, and full launch still requires explicit authorization.
- Rootfs artifact and direct-host guard check:
  a host-side JSON invariant script verified
  `lane_b_full_rootfs_guard_20260816T031154Z` has
  `blocker.phase=not_launched`, `ready_to_launch=true`,
  `training_launched=false`, `skip_run=true`, full-mode SHA/NCCL/manifest
  gates checked, and rootfs sentinel evidence
  `cwd=/workspace/torchtitan`, `torchtitan_in_rootfs=1`, and
  `workspace_sentinel_exists=true`. The same check invoked
  `run_speedrun.py` directly on the host and confirmed it fails closed with
  exit code `21` before run-attempt validation, including the expected
  messages that `run_speedrun.sh` is required and `TORCHTITAN_IN_ROOTFS=1`
  is missing.
- Python harness compile check:
  `python -m py_compile` passed for the Modded NanoGPT B200 harness Python
  files and focused unit-test files.
- Shell wrapper syntax check:
  `bash -n` passed for the Modded NanoGPT B200 rootfs-aware shell wrappers,
  including `run_speedrun.sh`, `check_active_jobs.sh`, `summarize.sh`,
  `parse_log.sh`, `prepare_data.sh`, `preflight.sh`,
  `setup_flash_attention.sh`, `rootfs_guard.sh`, and
  `diagnose_mlp_backend.sh`.
- Focused Modded NanoGPT B200 tests:
  `TMPDIR=$PWD/.scratch/trae-pytest-tmp python -m pytest --basetemp=.scratch/trae-pytest-tmp tests/unit_tests/test_modded_nanogpt_b200_*.py -q`
  passed with 111 tests, including runner, parser, preflight, data-manifest,
  direct-CLI guard, shell-wrapper forged-marker guard, and run-index
  regressions for active-job scanning, direct host fail-closed behavior,
  launch-readiness indexing, diagnostic exclusion from baseline stats,
  stale or malformed summary exclusion from baseline stats, full-baseline
  evidence gating at parser and run-index boundaries,
  active-job scanner `ps` failure fail-closed handling and explicit
  scan-failure blocker messages,
  structured preflight-report `ok=false` fail-closed handling even when the
  wrapper exits zero,
  result-local cache-directory validation before preflight or launch,
  allowlisted result-artifact size capture in parser summaries and analysis,
  explicit parser claim-validation summaries and analysis rendering,
  parser-derived thermal/clock and CPU/RSS telemetry signs,
  run-index preservation of claim validation and telemetry-sign summaries,
  full-baseline rejection when `exit_code.json` records a nonzero training exit,
  run-index baseline rejection when a stale successful summary is paired with
  nonzero `exit_code.json` training evidence,
  launch-readiness data-manifest consistency, embedded
  launch-readiness fallback, full-mode-only launch-ready indexing,
  launch-readiness sidecar classification consistency, top-level and nested
  sidecar attempt-identity consistency, contradictory sidecar identity
  rejection, and sidecar provenance.
- Rootfs run-index refresh:
  `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`
  completed successfully through the rootfs-aware wrapper. The refreshed real
  index reports `total_attempts=24`, `baseline_stats.count=0`,
  `len(launch_prerequisite_attempts)=1`, `len(launch_ready_attempts)=0`, zero
  `launch_readiness_error` records, zero `summary_error` records, zero
  `metrics_error` records, zero `baseline_evidence_error` records in the
  current artifact set, and zero non-skip launch-ready rows. The current
  prerequisite row is
  `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z/summary.json`.
  Because the stricter gate requires parsed per-attempt `active_jobs.ok=true`,
  `active_jobs.active_job_count=0`, and no active-job scan error, older
  sidecar-ready artifacts without embedded active-job evidence are preserved as
  failed/diagnostic history instead of being surfaced as launch prerequisites.
  The refreshed index now records top-level
  `launch_readiness_exclusion_stats.count=20`. Its `by_phase` entries are now
  objects with `count` plus bounded `example_summaries`: `active_jobs` has
  `count=4` with examples
  `experiments/modded_nanogpt_b200/results/lane_b_full_patch_class_guard_20260816T012032Z/summary.json`,
  `experiments/modded_nanogpt_b200/results/lane_b_full_prelaunch_patch_guard_20260816T012647Z/summary.json`,
  and
  `experiments/modded_nanogpt_b200/results/lane_b_full_rootfs_guard_20260816T031154Z/summary.json`;
  `launch_readiness` has `count=7` with examples
  `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_20260815T224326Z/summary.json`,
  `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_artifacts_20260815T230007Z/summary.json`,
  and
  `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_dcgm_20260815T233425Z/summary.json`;
  and `variant_patch_classification` has `count=9` with examples
  `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010335Z/summary.json`,
  `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/summary.json`,
  and
  `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000411Z/summary.json`,
  plus the per-attempt `launch_readiness_exclusion` rows. This makes the
  demotion distribution, the bounded example artifacts, and each first
  demotion reason visible without reopening every result directory.
  The index therefore has zero launch-ready blockers, zero launch-ready
  skip-run rows, zero launch-ready rows missing parsed manifest summaries, zero
  launch-ready rows with wrong GPU count or non-B200 GPU names, zero
  launch-ready rows missing rootfs sentinel evidence, zero launch-ready rows
  with stale source commit, zero launch-ready rows missing Lane B
  patch-classification evidence, zero bad launch tokens, zero failed
  preflights, and zero launched training rows.
- Diff whitespace check:
  `git diff --check -- <touched files>` passed.
- Generated `results/`, `sources/`, and `data/` paths are ignored by git.
  The required local pytest basetemp `.scratch/trae-pytest-tmp/` is also now
  explicitly ignored.
- Exact process checks found no active `torchrun`, `train_gpt.py`, or
  `cached_fineweb10B.py`.
- `experiments/modded_nanogpt_b200/check_active_jobs.sh` now provides the
  Phase 1 active-job check as a rootfs-aware structured scanner instead of a
  raw `pgrep` probe. It filters search-shell false positives, matches real
  target commands by process-token basename instead of arbitrary substrings,
  fails closed when process-table discovery itself fails, and can write a JSON
  report under ignored results. Fresh evidence:
  `experiments/modded_nanogpt_b200/results/phase1_active_jobs_launch_guard_20260816T030725Z/active_jobs.json`
  records `ok=true`, `active_job_count=0`, `ignored_match_count=0`, and
  `active_jobs=[]`.

## Prompt-To-Artifact Checklist

| Prompt requirement | Current evidence | Status |
| --- | --- | --- |
| Mandatory sources read | Current session read `CONSTITUTION.md`, `docs/agents/agentic-engineering.md`, `docs/agents/domain.md`, `docs/agents/issue-tracker.md`, `AGENTS.md`, `spec.md`, `preflight_checklist.md`, and `upstream_reference.md`. | Covered for this slice |
| Rootfs-only Python/CUDA/training/parsing/summarization | Rootfs-aware wrappers exist for preflight, FlashAttention setup, source fetch, data prep, speedrun, active-job scanning, parser, MLP diagnostics, and summarizer. `rootfs_guard.sh` is sourced by the shell wrappers after rootfs re-entry, so a forged `TORCHTITAN_IN_ROOTFS=1` marker on the host fails before wrapper work such as `pip install`. `cli_guard.py` is wired into the Python entrypoints so direct host invocation of `run_speedrun.py`, `parse_log.py`, `summarize.py`, `prepare_data.py`, `fetch_upstream.py`, `preflight.py`, or `diagnose_mlp_backend.py` fails closed with exit `21` before real work if `TORCHTITAN_IN_ROOTFS=1` is missing. Focused tests cover direct host fail-closed behavior while keeping import-level unit tests usable. Latest prerequisite refresh used `run_speedrun.sh`, and `telemetry/rootfs_environment.json` in `lane_b_full_skiprun_refresh_20260816T111056Z` records `torchtitan_in_rootfs=1`, `cwd=/workspace/torchtitan`, and `workspace_sentinel_exists=true`. | Covered for non-launch prerequisite path |
| No upstream `pip install -r requirements.txt` | No such command was run in this slice. | Covered |
| Do not modify Lane A source | Lane A generated source remains separate from Lane B generated source; latest diagnostic used Lane B source. Parser coverage now invalidates a final-valid Lane A summary if post-run source cleanliness evidence is missing or dirty. | Covered for latest attempt |
| Generated paths ignored | `git check-ignore -v` confirms generated `results/`, `sources/`, and `data/` paths are ignored. It also confirms `.scratch/trae-pytest-tmp/` is ignored for required local pytest runs. | Covered |
| Classification schema in attempt artifacts | Latest `attempt.json`, `preflight_report.json`, and `summary.json` include common classification fields. | Covered |
| Phase 1 state/safety checks | Latest verification included ignored-path checks and exact process checks for active training/data prep commands. The prompt and checklist now use `experiments/modded_nanogpt_b200/check_active_jobs.sh --active-jobs-output ...`, which re-enters rootfs and writes a structured JSON report; the latest real report at `phase1_active_jobs_launch_guard_20260816T030725Z/active_jobs.json` records `ok=true`, `active_job_count=0`, `ignored_match_count=0`, and no active jobs. Focused tests cover filtering raw `pgrep`/`rg` search-shell false positives, ignoring substring lookalikes such as `nottrain_gpt.py` and `mytorchrun_helper`, preserving real `torchrun`, `train_gpt.py`, and `cached_fineweb10B.py` matches, failing closed with `ok=false` plus `scan_error` when `ps` itself exits nonzero, and surfacing scan failure as an explicit `active-job scan failed: ...` blocker in `blocker.json` and launch readiness. | Covered |
| Phase 3 metadata before expensive work | `attempt.json`, `command.env`, `command.argv`, `data_manifest.json`, `environment.json`, `hardware.json`, and `operator_notes.md` exist in new preflight-passing attempt directories. New attempts record `command.launch_authorization_present` instead of echoing any operator-supplied launch-authorization token in `attempt.json`; the required token is still exposed in `launch_readiness.json` and Markdown as the operator-facing gate. Persisted environment metadata is redacted in both `command.env` and `attempt.json["environment"]`: focused coverage injects credential, API key, and auth-header variables and verifies that artifacts contain `<REDACTED>` rather than raw values while leaving subprocess execution environment construction unchanged. Result-directory reuse now fails closed before artifact creation: if `result_dir` already exists and is non-empty, the runner returns exit code `21` with an `attempt_reuse` stderr blocker, does not invoke preflight, leaves existing contents untouched, and does not write `attempt.json`, `command.env`, `command.argv`, or `run.log`. New `command.argv` files record the replayable `run_speedrun.sh` invocation with lane, mode, source, manifest, result directory, backends, and flags; `attempt.json["command"]["training_argv"]` separately preserves the inner `torchrun` command. Parser summaries now embed those command fields under `attempt_command`, and `analysis.md` renders both `attempt_command` and `training_command`. New result-local `data_manifest.json` files are pointer records to the supplied manifest, so attempt bundles carry the required data-manifest artifact without copying or mutating reused generated manifests. Parser summaries resolve those pointer records back to the underlying manifest and render both the pointer path and resolved manifest facts in `analysis.md`. New result-local `environment.json` and `hardware.json` sidecars copy `preflight_report.json["environment"]` and `preflight_report.json["gpus"]` under schema-v1 `preflight_environment` and `preflight_gpus` kinds. Parser summaries preserve those sidecars under `environment_sidecar` and `hardware_sidecar`, prefer them for top-level environment/GPU fields, render their kinds in `analysis.md`, and fall back to preflight fields for older bundles. | Covered |
| Phase 4 Lane B source evidence | Latest result has `source.json`, source status before/after, `variant_patch.diff`, and `variant_patch_classification.json`. The runner writes the Lane B diff and classification before preflight and before any possible launch. The classification artifact records each changed file's patch class and first Lane A blocker, and the parser embeds and renders the same data in `summary.json` and `analysis.md`. Full Lane B attempts with any `unclassified` patch class now stop before preflight with `blocker.phase=variant_patch_classification`, even if the full-launch authorization token is present. This path also writes `launch_readiness.json` with `ready_to_launch=false`, `training_launched=false`, and `blocked_by` set to the source-classification blocker. Parser coverage also invalidates Lane B baseline stats if a summary sidecar contains unclassified patches, while preserving an explicit runner `blocker.json` message when present. | Covered |
| Phase 5 dependency evidence | Latest preflight records Torch/CUDA/Triton/FlashAttention and B200 inventory. | Covered |
| Phase 6 full 900M manifest | Full schema-v1 manifest refreshed through `prepare_data.sh` at `experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`; it records 10 shards, `total_bytes=2000010240`, SHA256 entries for every shard, `token_budget=900M`, top-level `verified_sha=true`, and the pinned source commit. | Covered for reused full data |
| Phase 7 preflight | Latest full-mode Triton gate ran through `run_speedrun.sh` with `--verify-sha --skip-run`. Rootfs, Torch/CUDA/Triton/FlashAttention import, 8x B200 inventory, Torch primitive smoke, NCCL all-reduce, source policy, full data-manifest SHA verification, FA2 attention smoke, and Triton MLP local forward/backward smoke passed. The runner now fails closed on structured preflight report disagreement: if the preflight command exits `0` but `preflight_report.json["ok"]` is not true, it converts the attempt to preflight exit code `21`, records the first structured failure as the blocker, writes `ready_to_launch=false`, and stops before skip-run or launch paths. | Covered; launch-ready dry gate |
| MLP backend diagnostics | `diagnose_mlp_backend.sh` re-enters rootfs and writes a bounded JSON diagnostic under ignored `results/`. The latest Triton diagnostic records rootfs/environment evidence and passes with `output_shape=[2, 16, 768]`, after setting the B200 CE compile capability. The standalone diagnostic now delegates to the same preflight Triton smoke helper used by the full-mode launch gate. The preceding failing diagnostic identified the fixed backward shape mismatch. | Covered; current diagnostic passes |
| Full-mode prerequisite gate report | `run_speedrun.py` now writes `launch_readiness.json` and `launch_readiness.md` for preflight-failed, skip-run, launch-authority-blocked, launched, and preflight-not-run source-classification blocker paths. The report separates preflight readiness, unchecked downstream gates, full-mode SHA/NCCL/manifest gates, launch authorization, the required launch token, and actual training launch state. Preflight-not-run blockers do not invent missing NCCL or SHA blockers; those gates remain unchecked until preflight runs. `parse_log.py` now embeds `launch_readiness.json` in `summary.json` and renders readiness/token/blocker fields in `analysis.md`. Latest full-mode Triton skip-run gate records `training_launched=false`, `ready_to_launch=true`, `skip_run=true`, `launch_authorization_required_token=launch-full-b200`, `nccl_checked=true`, `data_manifest_checked=true`, `verified_sha=true`, and `blocked_by=[]`; the blocker is `not_launched` because this was an explicit non-launch artifact refresh. The latest non-skip authority guard remains `lane_b_full_prelaunch_patch_guard_20260816T012647Z`, blocked by `launch_authority` because the explicit full-launch token was absent. | Covered; awaiting launch authorization |
| Full baseline artifact validity | Parser success now requires more than final metrics. Full attempts are rejected from `ok` and `included_in_baseline_stats` unless launch-readiness says training was actually launched, `exit_code.json` records `phase=training` and `exit_code=0`, the attempt was not `--skip-run`, no previous-stall override was used, launch preflight passed, rootfs sentinel evidence records `/workspace/torchtitan`, the full 900M manifest has SHA verification in both preflight and manifest evidence, NCCL preflight evidence is present, the manifest source commit is pinned, and the GPU inventory is exactly 8x B200. A focused regression proved a final-metric full attempt without launch/rootfs/data sidecars used to be accepted and is now blocked at `launch_evidence`; a second regression proves an otherwise valid full attempt with `exit_code.json` recording `exit_code=7` is blocked at `training` and excluded from baseline stats. Run-index baseline aggregation now independently repeats the same training-exit check from the sibling `exit_code.json`, plus parsed manifest provenance checks: `launch_readiness.data_manifest` must match the parsed manifest pointer or resolved target when present, both preflight and manifest evidence must be SHA-verified, `data_manifest_summary` must have the full 900M shape, and `data_manifest_summary.source_commit` must be the pinned upstream commit. Focused regressions cover stale full summaries with otherwise valid launch/rootfs/NCCL/GPU evidence but either a wrong data source commit or nonzero training exit sidecar; both contribute zero baseline stats. | Covered by tests and refreshed rootfs-generated index |
| Run-index launch readiness supersession | The zero launch-prerequisite statement in the following historical row is superseded by the `lane_b_full_skiprun_refresh_20260816T111056Z` refresh. The current `run_index.json` records `total_attempts=24`, `baseline_stats.count=0`, one skip-run launch-prerequisite row pointing to `lane_b_full_skiprun_refresh_20260816T111056Z/summary.json`, and zero non-skip launch-ready rows. | Current |
| Run-index launch readiness surfacing | `summarize.py` now reads sibling `launch_readiness.json` files into attempt records and exposes `launch_prerequisite_attempts` for ready-but-not-launched prerequisite attempts. The narrower `launch_ready_attempts` list excludes `skip_run=true` dry gates, so it only contains non-skip rows that could launch after adding the explicit full-launch authorization token. If the sibling sidecar is absent, it falls back to embedded `summary.json["launch_readiness"]`, so archived summary bundles keep their readiness evidence. If the sibling sidecar is malformed or not a JSON object, the attempt is preserved with `launch_readiness_error`, contributes zero baseline stats, and is excluded from launch-prerequisite and launch-ready rows instead of aborting the entire index. If `summary.json` itself is malformed or not a JSON object, the attempt is preserved with `classification.mode=malformed`, `summary_error`, zero baseline contribution, and no launch-ready row. If `final_metrics` is malformed, the attempt is preserved with normalized empty metrics, `metrics_error`, zero baseline contribution, and no launch-ready row. If baseline-critical `final_metrics.val_loss`, `final_metrics.train_time`, or `final_metrics.step_avg` is nonnumeric, or if `final_metrics.val_loss > 3.28`, the attempt is preserved with `metrics_error`, zero baseline contribution, and no launch-ready row. Attempt records also preserve top-level `environment`, `gpus`, `gpu_count`, `environment_sidecar`, `hardware_sidecar`, `data_manifest_summary`, `preflight_data_manifest`, `claim_validation`, `variant_patch_classification`, `telemetry`, `telemetry_signs`, and `active_jobs`, so launch-ready and failed-attempt rows can be audited without opening every result directory. The run index independently rejects baseline inclusion for non-full, non-claim-eligible, `ok=false`, missing or malformed training-exit sidecar, or nonzero training-exit attempts even if a stale summary says `included_in_baseline_stats=true`; it also requires prerequisite rows to be Lane A/B full-mode, claim-eligible, no readiness blockers, `preflight_ok=true`, `full_mode_gates.full_mode=true`, `full_mode_gates.data_manifest_checked=true`, `full_mode_gates.verified_sha=true`, `full_mode_gates.nccl_checked=true`, `manifest_token_budget=900M`, `manifest_num_files=10`, `manifest_total_bytes=2000010240`, `launch_authority_required=true`, and `launch_authorization_required_token=launch-full-b200`, and rejects readiness sidecars whose present `lane`, `mode`, `run_id`, or `attempt_id` at top level or under `classification` disagrees with the summary classification or with each other. Launch-prerequisite rows now require parsed manifest evidence from `summary.json`: `launch_readiness.data_manifest` must match the parsed manifest pointer or resolved target when present, `data_manifest_summary` must have the full 900M shape, and `data_manifest_summary.source_commit` must be the pinned upstream commit. They also require parsed GPU evidence from `summary.json`: exactly eight GPU entries whose names include `B200`. They also require parsed rootfs sentinel evidence from `summary.json`: `TORCHTITAN_IN_ROOTFS=1`, `cwd=/workspace/torchtitan`, and the workspace sentinel flag. They also require parsed source provenance from `summary.json`: the source commit must equal the pinned upstream commit. Lane B prerequisite rows also require parsed patch provenance from `summary.json`: a diff artifact, a non-empty changed-file list, classified patch classes other than `unclassified`, and a first Lane A blocker for every changed file. Launch-prerequisite rows also require parsed active-job evidence from `summary.json`: `active_jobs.ok=true`, `active_jobs.active_job_count=0`, and no scan error. This keeps malformed summaries, malformed metrics, high-loss stale summaries, stale successful summaries with failed training exits, malformed diagnostic readiness, stale copied readiness sidecars, sidecar-only readiness without parsed manifest evidence, readiness without rootfs sentinel evidence, readiness without Lane B patch classification, readiness without active-job evidence, copied readiness sidecars with mismatched or contradictory attempt identity or data-manifest path, contradictory readiness sidecars, unchecked data manifests, smoke-shaped full manifests, incomplete or non-B200 GPU summaries, stale source summaries, tokenless launch-readiness sidecars, authority-not-required sidecars, and truncated readiness sidecars out of launch-prerequisite rows and baseline stats. Focused regressions cover diagnostic forged-inclusion, embedded-readiness fallback, full-mode-only launch-ready indexing, skip-run prerequisite separation, sidecar classification consistency, top-level and nested sidecar attempt-identity consistency, contradictory sidecar identity rejection, contradictory readiness sidecar rejection, full-manifest-shape enforcement, explicit data-manifest gate enforcement, copied data-manifest path rejection, missing parsed manifest summary rejection, incomplete GPU inventory rejection, missing rootfs sentinel rejection, stale source commit rejection, missing Lane B patch-classification rejection, missing active-job scan evidence rejection, launch-token marker enforcement, launch-authority gate enforcement, malformed readiness sidecar preservation, malformed summary preservation, malformed metrics preservation, nonnumeric baseline metric rejection, high validation loss rejection, nonzero training-exit sidecar rejection, and preservation of claim validation plus telemetry-sign summaries. A historical real `run_index.json` refresh regenerated through `summarize.sh` had zero launch-prerequisite entries because ready-ish artifacts lacked per-attempt active-job evidence; that zero-prerequisite state is superseded by the `lane_b_full_skiprun_refresh_20260816T111056Z` refresh, which surfaces one skip-run launch-prerequisite row and zero non-skip launch-ready rows. The historical refresh also had zero launch-readiness lane/mode mismatches, zero launch-ready rows with present top-level or nested `run_id` or `attempt_id` identity mismatches, zero launch-ready rows with contradictory top-level versus nested identity, zero launch-ready rows with blockers, zero launch-ready rows with failed preflight, zero launch-ready rows with failed SHA, zero launch-ready rows with failed NCCL, zero launch-ready rows with missing active-job evidence, zero launch-ready rows with failed active-job scans or nonzero active-job counts, zero launch-ready rows with non-full-mode gates, unchecked data manifests, missing parsed manifest summaries, incomplete or non-B200 GPU summaries, missing rootfs sentinel evidence, stale source commit, missing Lane B patch-classification evidence, missing or incorrect launch token, missing or false launch-authority gate, wrong token budget, wrong shard count, wrong total bytes, wrong parsed data source commit, mismatched readiness data-manifest path, or `skip_run=true`, zero `launch_readiness_error` records, zero `summary_error` records, and zero `metrics_error` records in that artifact set. Older ready-but-not-launched rows without per-attempt active-job evidence are preserved under diagnostic/failed attempts and are not the current skip-run prerequisite row. | Covered by tests and refreshed rootfs-generated index |
| Phase 8 telemetry | Latest diagnostic has telemetry directory, rootfs environment, watcher status, source-status copies, prompt-named watcher logs, and DCGM availability status. Watcher range files are intentionally not started because training was skipped. | Covered for skip-run diagnostic |
| Phase 9 launch/babysitting | Unit coverage verifies launched-attempt telemetry start/stop, no-output watchdog, and nonzero launched diagnostic classification. Full-mode launches now also require explicit `--launch-authorization=launch-full-b200`; the latest real rootfs guard run proved omission stops before `torchrun` with exit code 21 and a preserved `launch_authority` blocker. A unit regression proves full Lane B attempts with unclassified source patches stop before preflight/training and write launch-readiness evidence, so the launch token alone is not sufficient to launch an unexplained patchset. Full-mode attempts now capture active-job evidence after successful preflight and before returning for `--skip-run` or missing launch authorization; active jobs or scan errors block through the existing `active_jobs` blocker before skip-run or launch-authority success can be reported. The refreshed non-launch Lane B full-mode artifact records `active_jobs.ok=true`, `active_job_count=0`, and `active_jobs=[]` before returning for `--skip-run`. A unit regression also proves an authorized full launch stops after preflight and before telemetry/training if `active_jobs.json` reports another `torchrun`, `train_gpt.py`, or `cached_fineweb10B.py` job, preserving `blocker.phase=active_jobs` and `exit_code.phase=active_jobs`. A new unit regression proves `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` must be result-local directories before preflight or launch; an out-of-result cache path records `blocker.phase=cache_directories`, writes `exit_code.phase=cache_directories`, preserves launch-readiness evidence with the blocker, marks telemetry watchers `not_started`, and does not invoke preflight or training. Independent verifier evidence for the early active-job capture fix: `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py -q` reported `31 passed in 1.08s`, `tests/unit_tests/test_modded_nanogpt_b200_summarize.py -q` reported `43 passed in 0.10s`, `python -m py_compile experiments/modded_nanogpt_b200/run_speedrun.py experiments/modded_nanogpt_b200/summarize.py` passed, `git diff --check -- experiments/modded_nanogpt_b200/run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py` passed, and `python experiments/modded_nanogpt_b200/verify_static.py` reported `Static verification passed for 39 file(s).` No current real full launch is authorized. | Covered by tests and refreshed non-launch prerequisite artifact; unproven by a new real launch |
| Phase 9 stop-condition one-shot snapshots | Runner now captures `telemetry/stop_ps_tree.txt`, `telemetry/stop_nvidia_smi_processes.txt`, and `telemetry/stop_snapshot.json` for launched nonzero exits and no-output stalls before writing the blocker. Parser coverage verifies the snapshot is embedded in `summary.json` and rendered in `analysis.md`. | Covered by tests |
| Phase 10 post-run collection | Unit coverage and older artifacts show source-after, wall clock, exit code, parser, and null metrics. Parser coverage now uses captured `source_status_after.txt` when present and marks Lane A attempts invalid with `blocker.phase=source_policy` if source cleanliness evidence is missing or the source is dirty after launch. | Covered by tests and diagnostic artifacts |
| Phase 11 analysis | Latest analysis renders classification, launch readiness, source/data, Lane B patch classification, separate manifest/preflight SHA state, environment/GPU including sidecar kinds, final metrics, wall clock, telemetry, DCGM, stop snapshots when present, normalized prompt-vocabulary failure category, first relevant log error when present, and blocker message. Parser summaries now include `artifact_sizes` for allowlisted top-level attempt sidecars and known telemetry files, and `analysis.md` renders `artifact_total_known_bytes` plus `artifact_size` rows for those known files. Parser summaries also include `claim_validation`, and `analysis.md` renders the prompt's successful-reproduction gates plus the first claim blocker; the refreshed `lane_b_full_skiprun_refresh_20260816T111056Z` artifact records `successful_b200_reproduction=false` because final validation was not reached. Parser summaries now derive telemetry signs for thermal-or-clock throttling and CPU-or-RSS bottleneck suspicion, and `analysis.md` renders those sign booleans plus reasons; the latest skip-run artifact has both signs false because watcher ranges were not started. The first-error scanner now ignores benign JSON/config keys such as `allow_previous_stall`, so the refreshed `lane_b_full_skiprun_refresh_20260816T111056Z` summary records the authoritative blocker as `not_launched`. | Covered for non-launch prerequisite path |
| Phase 12 final reporting | Chat responses report current artifact paths and blockers. | Covered per turn |
| Minimal terminal state | Latest artifact is a Lane B full-mode skip-run gate with preserved evidence, all preflight gates passed, active-job scan clear, and classified `not_launched` blocker before training launch. The real run index now surfaces this artifact as one launch-prerequisite row and still has zero non-skip launch-ready rows. The latest non-skip authority guard remains `lane_b_full_prelaunch_patch_guard_20260816T012647Z`, blocked by `launch_authority` because the explicit launch token was absent. The broader productionization goal still has no full baseline. | Partially achieved |

## Current Missing Or Weak Requirements

1. Full Lane A and full Lane B baseline paths remain incomplete because
   `baseline_stats.count=0`; no full attempt has reached final validation.
2. Lane B full-mode preflight now passes with FA2 attention and Triton MLP under
   `--skip-run`; the latest sidecar-backed gate is
   `lane_b_full_skiprun_refresh_20260816T111056Z`. The runner now captures
   active-job evidence before skip-run or launch-authorization returns, and the
   real ignored `run_index.json` was refreshed afterward: it has one
   launch-prerequisite row pointing to this skip-run artifact and zero non-skip
   launch-ready rows. No full Lane B training launch has been authorized or
   completed.
   The PyTorch fallback local smoke passes with output shape `[2, 16, 768]`, but
   remains disallowed for full jobs because prior full-job options either OOMed
   or compile-stalled. `--allow-previous-stall` is diagnostic-only.
3. Lane C remains blocked until a Lane A or Lane B baseline exists.

## Next Concrete Action

The next meaningful action requires explicit user authorization: launch a
bounded Lane B full-mode training attempt by passing
`--launch-authorization=launch-full-b200` to `run_speedrun.sh` with the
launch-ready Lane B full configuration.
Without that authorization, continue only with non-launch hardening, review, or
documentation work. The latest non-launch hardening fixed early active-job
evidence capture for full-mode skip-run and launch-authority return paths,
after allowlisted parser artifact-size evidence for known result files and the
cache-directory guard
made runner preflight fail closed when TorchInductor or Triton cache directories
are not result-local directories. Earlier hardening also made structured
`preflight_report.json["ok"]` disagreement fail closed, made result-directory
reuse fail closed before artifact creation, made parser baseline eligibility
fail closed on missing launch/rootfs/data/NCCL/hardware evidence, added
run-index guards against stale or malformed baseline-inclusion bits and missing
launch-readiness sidecars, and refreshed the real run index through rootfs
wrappers.
