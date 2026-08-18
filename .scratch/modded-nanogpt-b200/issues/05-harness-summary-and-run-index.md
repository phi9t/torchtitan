# Harness summary and run index

Type: task
Status: resolved
Blocked by: -

## Requirement

Add the summary aggregation slice for the Modded NanoGPT B200 speedrun harness.
This ticket turns per-attempt `summary.json` files into a compact run index
without weakening any claim boundary.

Use `.scratch/modded-nanogpt-b200/spec.md` as the canonical spec.

## Scope

Allowed:

- add `experiments/modded_nanogpt_b200/summarize.py`;
- add a rootfs-aware shell wrapper if summary generation invokes Python;
- aggregate `summary.json` files under ignored result directories;
- produce compact Markdown or JSON run indexes under the experiment directory or
  ignored result paths;
- add parser/summary fixtures.

Excluded:

- no training launch;
- no source or data mutation;
- no generated logs, data, or large result bundles tracked in git;
- no inclusion of diagnostic attempts in successful baseline statistics.

## Behavior

The summarizer must:

- read `summary.json` when present instead of reparsing mixed historical logs;
- group attempts by lane, mode, arm, claim label, source commit, data manifest,
  environment class, and evidence tier;
- report final validation loss, upstream `train_time`, `step_avg`, memory, and
  shell wall-clock separately;
- list failed or diagnostic attempts with blocker phase and message;
- compute median and best train time only from attempts with
  `included_in_baseline_stats=true`;
- keep completed-run timing separate from claim-valid timing;
- preserve conservative labels: `smoke`, `diagnostic`,
  `B200 upstream reproduction`, `B200 compatibility patchset`,
  `B200 systems-only`, `B200 ML variant`, and `B200 optimization ablation`.

## Verification Evidence

- Focused tests or fixtures for successful, failed, diagnostic, and partial
  attempts.
  - `python -m pytest tests/unit_tests/test_modded_nanogpt_b200_summarize.py -q`
    passed for successful full attempts, diagnostic exclusion, grouping by
    source commit and data manifest, preferring captured attempt source commits
    over pinned upstream fallback, legacy summaries, launch-readiness indexing,
    environment/hardware sidecar preservation, and atomic index writes.
- Demonstrate that diagnostic attempts are excluded from successful baseline
  statistics.
  - `experiments/modded_nanogpt_b200/results/run_index.json` currently reports
    `total_attempts=24`, `baseline_stats.count=0`,
    `len(launch_prerequisite_attempts)=1`, and
    `len(launch_ready_attempts)=0`; diagnostic attempts remain excluded from
    baseline stats. The single prerequisite row points to
    `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z/summary.json`
    and is not a non-skip launch-ready row because `skip_run=true`.
  - Malformed or truncated `summary.json` files are preserved as failed
    attempt records instead of aborting the entire index. A focused regression
    covers one valid diagnostic summary beside one corrupted summary: the index
    still reports both attempts, records the corrupted file under
    `diagnostic_or_failed_attempts` with `classification.mode=malformed` and
    `summary_error`, contributes zero baseline stats, and exposes no
    launch-ready row for it.
  - Malformed `final_metrics` payloads are also fail-closed. A focused
    regression covers a full-mode summary that claims baseline inclusion while
    setting `final_metrics` to a string: the index normalizes metrics to `{}`,
    records `metrics_error`, contributes zero baseline stats, and exposes no
    launch-ready row.
  - Nonnumeric baseline-critical metrics are fail-closed even when
    `final_metrics` is a JSON object. A focused regression covers a full-mode
    summary that claims baseline inclusion while setting
    `final_metrics.train_time` to `"72 seconds"`: the index records
    `metrics_error`, contributes zero baseline stats, and exposes no
    launch-ready row. Baseline aggregation now requires numeric `val_loss`,
    `train_time`, and `step_avg`.
  - High validation loss is also fail-closed at the run-index boundary, not
    only in the parser. A focused regression covers a stale full-mode summary
    that claims baseline inclusion while setting `final_metrics.val_loss` to
    `3.29`: the index records `metrics_error`, contributes zero baseline
    stats, exposes no launch-ready row, and requires `val_loss <= 3.28` for
    baseline inclusion.
  - Parser baseline eligibility now rejects full attempts that have final
    metrics but lack structured launch/rootfs/data/NCCL/hardware sidecar
    evidence. The run index also rechecks the summary evidence before counting
    a baseline row: it must come from a launched, non-skip,
    non-stall-override full attempt with rootfs sentinel evidence,
    SHA-verified 900M manifest gate evidence, NCCL evidence, and 8x B200
    inventory. A focused regression covers a stale full-mode summary with
    `ok=true`, `included_in_baseline_stats=true`, and good metrics but no
    launch/rootfs sidecars; the attempt is preserved with
    `baseline_evidence_error` and contributes zero global or grouped baseline
    stats.
  - Run-index baseline aggregation now repeats the parser's full-manifest
    provenance checks before counting any baseline row. A focused regression
    covers a stale full-mode summary with otherwise valid launch/rootfs/NCCL/GPU
    evidence but `data_manifest_summary.source_commit` set to a non-pinned
    value; the attempt is preserved with `baseline_evidence_error` and
    contributes zero global or grouped baseline stats. Baseline rows now require
    `launch_readiness.data_manifest` to match the parsed manifest pointer or
    resolved target when present, SHA verification in both
    `preflight_data_manifest` and `data_manifest_summary`, the full 900M shape
    in `data_manifest_summary`, and the pinned upstream data source commit.
  - Clean-context manifest SHA hardening closed a launch-prerequisite evidence
    gap: manifests had per-shard `sha256` entries but no top-level
    `verified_sha`, while the parser and index read parsed manifest SHA state
    from top-level `verified_sha`. `summarize.py` launch-prerequisite gating
    now requires both parsed manifest SHA evidence and
    `preflight_data_manifest.verified_sha` evidence, so sidecar or preflight
    SHA alone cannot surface a row. TDD evidence recorded RED
    `3 failed, 45 passed`, GREEN `48 passed in 0.15s`, clean-context
    verification `48 passed in 0.08s`, `py_compile` passing for the three
    changed Python entrypoints, and `git diff --check` passing. A test-only
    follow-up added coverage for false `preflight_data_manifest.verified_sha`
    with parsed manifest SHA true and recorded `47 passed in 0.15s` plus
    `git diff --check` passing. This was code/test evidence only; the current
    `run_index.json` was not refreshed or mutated in that slice.
  - A later clean-context subagent refreshed the ignored real run index through
    the approved rootfs-aware summarizer:
    `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`,
    which exited `0`. It did not run launch, GPU, training, preflight,
    data-prep, download, pip, CUDA, NCCL, `torchrun`, or full-launch commands.
    Historical refreshed facts, now superseded by the
    `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh:
    `total_attempts=23`, `baseline_count=0`, `launch_prerequisite_count=0`,
    `launch_ready_count=0`, `launch_readiness_exclusion_stats.count=20`,
    `by_phase.data_manifest_summary.verified_sha.count=11`,
    `by_phase.preflight_data_manifest.verified_sha.count=2`, and
    `by_phase.launch_readiness.count=7`; both
    `launch_prerequisite_attempts=[]` and `launch_ready_attempts=[]`.
    Representative SHA-gated exclusion paths include
    `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010335Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000411Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000617Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T001037Z/summary.json`,
    and
    `experiments/modded_nanogpt_b200/results/lane_b_full_rootfs_guard_20260816T031154Z/summary.json`.
    The requested SHA-exclusion `jq` query failed on a null phase during
    operator inspection, so the subagent used a null-safe equivalent; this is
    recorded as an operator-query footnote, not a code failure. The refreshed
    index intentionally demotes older generated artifacts under stricter SHA
    evidence gates and does not prove runtime launch readiness.
  - Clean-context parser pointer hardening fixed result-local manifest pointer
    resolution without weakening the launch-prerequisite gates. `parse_log.py`
    now resolves non-absolute `data_manifest_pointer.path` values relative to
    the pointer file directory. For pointer manifests, it propagates explicit
    top-level `verified_sha` from the resolved target only when the pointer
    lacks top-level `verified_sha`; it does not infer `verified_sha` from shard
    `sha256` entries. RED failed because new pointer tests showed relative
    result-local pointer targets were not resolved and target manifest fields
    were not loaded. GREEN evidence: parse_log plus summarize tests passed with
    `71 passed in 0.14s`, clean-context verification recorded
    `71 passed in 0.13s`, `py_compile` passed for `parse_log.py` and
    `summarize.py`, and `git diff --check` passed for scoped files, with a note
    that many scoped files are untracked and static verifier covers untracked
    content separately. Clean-context review found no blocking issues and
    confirmed summarize fail-closed gates remain intact.
  - After the parser fix, the real ignored index was refreshed again through
    the approved rootfs-aware summarizer. Historical refreshed facts, now
    superseded by the `lane_b_full_skiprun_refresh_20260816T111056Z` index
    refresh: `total_attempts=23`, `baseline_count=0`,
    `launch_prerequisite_count=0`, `launch_ready_count=0`, `stale_count=20`,
    and `launch_readiness_exclusion_stats.count=20`.
    Exclusion phases were `data_manifest_summary.verified_sha=11`,
    `launch_readiness=7`, and
    `preflight_data_manifest.verified_sha=2`, plus 3 diagnostic failed-attempt
    rows with no `launch_readiness_exclusion` phase. The current real demotions
    did not clear because existing resolved manifests still lack top-level
    `verified_sha`. The parser fix protects future pointer manifests that
    carry explicit `verified_sha`; it does not mutate stale generated artifacts
    and does not infer SHA from shard hashes. No launch, GPU, training,
    preflight, data-prep, download, pip, CUDA, NCCL, `torchrun`, or full-launch
    command was run. The overall objective remains incomplete: no full Lane A/B
    baseline exists. The later refreshed index now has one skip-run
    prerequisite row, no non-skip launch-ready row, and full launch still
    requires explicit authorization.
  - The latest non-launch Lane B full-mode artifact refresh used
    `experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`,
    whose top-level manifest state records `verified_sha=true`,
    `token_budget=900M`, `num_files=10`, and
    `total_bytes=2000010240`. The paired result directory is
    `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z`.
    Its summary records `preflight_ok=true`,
    `launch_readiness.ready_to_launch=true`,
    `launch_readiness.training_launched=false`,
    `launch_readiness.skip_run=true`, `blocked_by=[]`,
    `full_mode_gates.verified_sha=true`,
    `full_mode_gates.nccl_checked=true`, and
    `full_mode_gates.data_manifest_checked=true`. The active-job scan is clear:
    `active_jobs.ok=true`, `active_job_count=0`, and `active_jobs=[]`.
    The blocker is `phase=not_launched` with a message that skip-run was
    requested and training was not launched. `attempt.json["command"]["argv"]`
    and `command.argv` include `--skip-run` and do not include
    `--launch-authorization=launch-full-b200`.
  - After that refresh, the ignored real index was regenerated through the
    rootfs-aware summarizer and now reports `total_attempts=24`,
    `len(launch_prerequisite_attempts)=1`,
    `len(launch_ready_attempts)=0`, and `baseline_stats.count=0`. The
    prerequisite row points to
    `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z/summary.json`.
    No full Lane A/B baseline exists, no non-skip launch-ready row exists, no
    training launch occurred, and a full Lane B launch still requires explicit
    `launch-full-b200` authorization.
  - Run-index baseline aggregation now also repeats the parser's training-exit
    evidence gate before counting any baseline row. A focused regression covers
    a stale full-mode summary with otherwise valid launch/rootfs/NCCL/GPU/data
    evidence but sibling `exit_code.json` set to
    `{"phase": "training", "exit_code": 7}`: the attempt is preserved with
    `baseline_evidence_error`, contributes zero global or grouped baseline
    stats, and is excluded from `launch_ready_attempts`. Baseline rows now
    require sibling `exit_code.json` to be a valid JSON object with
    `phase=training` and `exit_code=0`.
  - The run index also applies an independent classification guard so a
    stale or malformed summary cannot count a diagnostic attempt merely by
    setting `included_in_baseline_stats=true`. A focused regression covers a
    diagnostic summary with a forged inclusion bit; it remains listed under
    `diagnostic_or_failed_attempts` and contributes zero global or grouped
    baseline stats.
  - Clean-context stale/demoted artifact indexing added a top-level
    `stale_or_demoted_artifacts` section to `run_index.json`, bounded by
    `MAX_STALE_OR_DEMOTED_ARTIFACTS = 20`. Each record includes the
    `summary` path, `reason`, `message`, `operator_action`, lane/mode/claim
    label context, and `launch_readiness_exclusion` when relevant. TDD evidence
    recorded RED `3 failed, 39 passed` before implementation, GREEN
    `42 passed in 0.08s` after implementation, and a follow-up test-only gap
    closure `43 passed in 0.07s` with no production code needed. Coverage
    includes legacy summary schema, a ready-ish sidecar demoted by missing
    `active_jobs`, bounded list behavior, and malformed summaries included in
    stale/demoted artifacts. Clean-context verification recorded
    `42 passed in 0.07s`, `py_compile` passing, `git diff --check` passing,
    and static `rg` confirmation of `stale_or_demoted_artifacts`,
    `MAX_STALE_OR_DEMOTED_ARTIFACTS`, `legacy_summary_schema`, and
    `operator_action`. Clean-context review found no blocking findings.
  - Clean-context static verifier implementation added
    `experiments/modded_nanogpt_b200/verify_static.py` and
    `tests/unit_tests/test_modded_nanogpt_b200_verify_static.py` to close the
    gap where `git diff --check` can skip untracked harness files. The verifier
    enumerates tracked modified files, staged files, and untracked files from
    `.scratch/modded-nanogpt-b200`,
    `experiments/modded_nanogpt_b200`, and
    `tests/unit_tests/test_modded_nanogpt_b200_*.py`; checks `.py`, `.sh`, and
    `.md`; excludes generated and ignored trees including
    `.scratch/trae-pytest-tmp`, experiment data/results/sources, and
    `__pycache__`; fails on trailing whitespace, missing final newline, Python
    compile errors, and shell syntax errors through `bash -n`; and supports
    `--list-files` and `--json`. TDD evidence recorded RED `5 failed` before
    the verifier existed, GREEN `5 passed`, and `6 passed` after adding
    staged-change coverage. `py_compile` passed for the verifier and test.
    `python experiments/modded_nanogpt_b200/verify_static.py --list-files`
    printed 39 candidate files, `--json` reported 39 candidate files, the real
    verifier passed with `Static verification passed for 39 file(s).`, and the
    full focused suite passed with `114 passed in 1.11s`. Clean-context review
    found no blocking issues and the noted staged-change coverage gap was
    closed by test-only follow-up; production code did not need a change for
    staged files.
- Surface full-mode prerequisite readiness without counting it as a baseline.
  - `summarize.py` reads optional sibling `launch_readiness.json` files into
    attempt records and exposes ready-but-not-launched prerequisite gates
    through `launch_prerequisite_attempts`. The narrower
    `launch_ready_attempts` list excludes `skip_run=true` dry gates, so it only
    contains non-skip rows that could launch after adding the explicit
    full-launch authorization token. A focused regression covers one dry
    prerequisite and one authority-guard prerequisite: both remain preserved in
    `launch_prerequisite_attempts`, but only the non-skip row enters
    `launch_ready_attempts`.
  - The current real index has one launch-prerequisite attempt and zero
    non-skip launch-ready rows. The prerequisite is the refreshed full-mode
    Lane B skip-run artifact
    `lane_b_full_skiprun_refresh_20260816T111056Z`; older ready-ish Lane B full
    artifacts are preserved as failed/diagnostic history because their
    summaries predate the parser-level `active_jobs` evidence now required by
    the run index.
  - If a sibling `launch_readiness.json` sidecar is absent, the run index falls
    back to embedded `summary.json["launch_readiness"]`. A focused regression
    covers an archived summary bundle with embedded launch readiness and no
    sibling sidecar; it is still surfaced under
    `launch_prerequisite_attempts`.
  - Malformed sibling `launch_readiness.json` sidecars are now per-attempt
    failed evidence, not an index-wide crash. A focused regression covers a
    truncated readiness sidecar: the attempt stays in
    `diagnostic_or_failed_attempts` with `launch_readiness_error`, contributes
    zero baseline stats, and is not listed under `launch_ready_attempts`.
  - `launch_ready_attempts` is constrained to Lane A/B full-mode,
    claim-eligible prerequisite attempts. A diagnostic attempt with
    `ready_to_launch=true` and `training_launched=false` remains
    diagnostic/failed and is not surfaced as a full launch-ready prerequisite.
    If a readiness sidecar records `lane` or `mode`, those values must match
    the summary classification; a stale copied sidecar with mismatched `lane`
    is preserved on the diagnostic/failed attempt record but is not surfaced as
    launch-ready. If a readiness sidecar records `run_id` or `attempt_id` at
    top level, or under its nested `classification` object, every present
    value must match the summary classification; focused regressions cover
    copied sidecars with mismatched attempt identity in both shapes and with
    contradictory top-level versus nested identity. New
    `run_speedrun.py` launch-readiness reports write `run_id` and `attempt_id`
    at top level so future sidecars are identity-bound without opening the
    nested classification object.
  - Launch-ready indexing also rejects internally contradictory readiness
    sidecars. A focused regression covers a full Lane B prerequisite summary
    whose sidecar says `ready_to_launch=true` while also recording a
    `blocked_by` data-manifest blocker and `verified_sha=false`; the attempt is
    preserved under `diagnostic_or_failed_attempts` but is excluded from
    `launch_ready_attempts`. The run index now requires no readiness blockers,
    `preflight_ok=true`, `full_mode_gates.verified_sha=true`, and
    `full_mode_gates.nccl_checked=true` before surfacing a row as launch-ready.
  - Launch-ready indexing also requires the full-manifest shape recorded by the
    readiness sidecar. A focused regression covers a sidecar that has
    `ready_to_launch=true`, `verified_sha=true`, and `nccl_checked=true` while
    reporting `manifest_token_budget="smoke"` and `manifest_num_files=1`; the
    attempt is preserved but excluded from `launch_ready_attempts`. The run
    index now requires `manifest_token_budget="900M"`, `manifest_num_files=10`,
    and `manifest_total_bytes=2000010240` for launch-ready rows.
  - Launch-ready indexing also requires the sidecar's explicit full-mode data
    gate state. A focused regression covers a sidecar with `ready_to_launch=true`,
    correct manifest shape, `verified_sha=true`, and `nccl_checked=true`, but
    `data_manifest_checked=false`; the attempt is preserved but excluded from
    `launch_ready_attempts`. The run index now also requires
    `full_mode_gates.full_mode=true` and
    `full_mode_gates.data_manifest_checked=true`.
  - Launch-ready indexing now requires parsed manifest provenance from
    `summary.json`, not just a readiness sidecar. A focused regression covers a
    copied readiness sidecar whose `launch_readiness.data_manifest` points to a
    different manifest from both the result-local pointer and resolved manifest
    target; the attempt is preserved but excluded from `launch_ready_attempts`.
    Another focused regression covers a sidecar-only ready attempt with no
    `data_manifest_summary`; it is preserved but excluded from
    `launch_ready_attempts`. Launch-ready rows require the full 900M
    `data_manifest_summary` shape and pinned data source commit.
  - Launch-ready indexing also requires parsed 8x B200 GPU evidence from the
    summary. A focused regression covers a sidecar-ready full attempt whose
    summary has only two B200 GPU rows; it is preserved but excluded from
    `launch_ready_attempts`. Launch-ready rows now require `gpu_count=8` and
    eight GPU entries whose names include `B200`.
  - Launch-ready indexing also requires parsed rootfs sentinel evidence from
    the summary. A focused regression covers a sidecar-ready full attempt whose
    summary lacks `telemetry.rootfs`; it is preserved but excluded from
    `launch_ready_attempts`. Launch-ready rows now require
    `TORCHTITAN_IN_ROOTFS=1`, `cwd=/workspace/torchtitan`, and the workspace
    sentinel flag.
  - Launch-ready indexing also requires parsed source provenance from the
    summary. A focused regression covers a sidecar-ready full attempt whose
    `summary.source.commit` is stale; it is preserved but excluded from
    `launch_ready_attempts`. Launch-ready rows now require the pinned upstream
    source commit.
  - Launch-ready indexing also requires parsed Lane B patch provenance from the
    summary. A focused regression covers a sidecar-ready Lane B full attempt
    whose summary lacks `variant_patch_classification`; it is preserved but
    excluded from `launch_ready_attempts`. Lane B launch-ready rows now require
    a diff artifact, a non-empty changed-file list, classified patch classes,
    and a first Lane A blocker for each changed file.
  - Launch-ready indexing also requires the operator-facing full-launch token
    marker. A focused regression covers a sidecar with all full-mode gates set
    and `ready_to_launch=true` but no
    `launch_authorization_required_token`; the attempt is preserved but excluded
    from `launch_ready_attempts`. The run index now requires
    `launch_authorization_required_token="launch-full-b200"` for launch-ready
    rows.
  - Launch-ready indexing also requires the explicit launch-authority gate. A
    focused regression covers a sidecar with all full-mode gates set and the
    correct token but `launch_authority_required=false`; the attempt is
    preserved but excluded from `launch_ready_attempts`. The run index now
    requires `launch_authority_required=true` for launch-ready rows.
- Preserve environment and hardware provenance at index level.
  - `summarize.py` carries each summary's top-level `environment`, `gpus`,
    `gpu_count`, `environment_sidecar`, and `hardware_sidecar` into the attempt
    records, so launch-ready and failed-attempt rows can be audited without
    opening every result directory.
- Show that host-side Python is not used for real GPU summary generation; use a
  rootfs-aware wrapper when running the summarizer against real results.
  - `experiments/modded_nanogpt_b200/summarize.sh` re-enters
    `scripts/rootfs/enter_rootfs.sh` before invoking summarizer Python.
  - Latest refresh used `experiments/modded_nanogpt_b200/summarize.sh` against
    `experiments/modded_nanogpt_b200/results/` with
    `--output experiments/modded_nanogpt_b200/results/run_index.json`; the
    index now reports
    `total_attempts=24`, `baseline_stats.count=0`,
    `len(launch_prerequisite_attempts)=1`, and
    `len(launch_ready_attempts)=0`. This refresh exercised the current
    run-index code, including the independent sibling `exit_code.json`
    baseline gate, the skip-run prerequisite split, and the active-job evidence
    gate, against real ignored artifacts. A compact `jq` invariant also reports
    zero launch-ready rows and zero launch-readiness lane/mode mismatches, zero
    `launch_readiness_error` records, zero `summary_error` records, zero
    `metrics_error` records, zero `baseline_evidence_error` records in the
    current artifact set, zero launch-ready rows with present top-level or
    nested `run_id` or
    `attempt_id` identity mismatches, zero launch-ready rows with contradictory
    top-level versus nested identity, zero
    launch-ready rows with blockers, zero launch-ready rows with failed
    preflight, zero launch-ready rows with failed SHA, and zero launch-ready
    rows with failed NCCL, non-full-mode gates, unchecked data manifests,
    missing/incorrect launch-authorization token, missing/false
    launch-authority gate, wrong token budget, wrong shard count, wrong
    total bytes, missing parsed manifest summary, wrong GPU count, non-B200
    GPU names, missing rootfs sentinel evidence, stale source commit, missing
    Lane B patch-classification evidence, missing active-job evidence, failed
    active-job scans, or nonzero active-job counts in the current artifact set.
    Older ready-but-not-launched rows without per-attempt `active_jobs` evidence
    are preserved under diagnostic/failed attempts; the refreshed
    `lane_b_full_skiprun_refresh_20260816T111056Z` row is surfaced as the one
    current skip-run prerequisite. The refreshed index now records top-level
    `launch_readiness_exclusion_stats.count=20`; its `by_phase` entries are
    objects with `count` plus bounded `example_summaries` lists, capped at
    three examples per phase by `summarize.py`. The current real index records
    examples for `active_jobs` (including
    `experiments/modded_nanogpt_b200/results/lane_b_full_patch_class_guard_20260816T012032Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_prelaunch_patch_guard_20260816T012647Z/summary.json`,
    and
    `experiments/modded_nanogpt_b200/results/lane_b_full_rootfs_guard_20260816T031154Z/summary.json`),
    `launch_readiness` (including
    `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_20260815T224326Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_artifacts_20260815T230007Z/summary.json`,
    and
    `experiments/modded_nanogpt_b200/results/wrapper_diag_skip_dcgm_20260815T233425Z/summary.json`),
    and `variant_patch_classification` (including
    `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010335Z/summary.json`,
    `experiments/modded_nanogpt_b200/results/lane_b_full_authority_guard_20260816T010926Z/summary.json`,
    and
    `experiments/modded_nanogpt_b200/results/lane_b_full_gate_20260816T000411Z/summary.json`).
    The phase counts remain 4 `active_jobs`, 7 `launch_readiness`, and
    9 `variant_patch_classification`, plus per-attempt
    `launch_readiness_exclusion` rows, so an operator can see the demotion
    distribution, bounded examples, and first demotion reason from
    `run_index.json`. The refreshed index includes
    `environment`, `gpus`, `gpu_count`,
    `environment_sidecar`, `hardware_sidecar`, `claim_validation`, and
    `telemetry_signs` fields in attempt records, so final reporting can audit
    claim blockers, telemetry bottleneck signs, and launch-readiness demotions
    from the index without reopening every result directory.
    The newest sidecar-backed attempt record is
    `lane_b_full_skiprun_refresh_20260816T111056Z`, with
    `environment_sidecar.kind=preflight_environment`,
    `hardware_sidecar.kind=preflight_gpus`, `gpu_count=8`,
    `claim_validation.successful_b200_reproduction=false`,
    `telemetry_signs.thermal_or_clock_throttling=false`,
    `ready_to_launch=true`, `training_launched=false`, and `skip_run=true`.
  - Static verifier work and the documentation audit update did not run
    launch, GPU, training, preflight, data-prep, download, pip, CUDA, NCCL,
    `torchrun`, full-launch, or artifact-mutating commands. The overall goal
    remains incomplete: no full Lane A/B baseline exists, one skip-run
    prerequisite row exists, no non-skip launch-ready row exists, and full
    launch still requires explicit authorization.
  - Clean-context early active-job evidence capture hardening changed the
    runner contract; the later non-launch Lane B full-mode refresh regenerated
    `run_index.json` with current active-job evidence. Full-mode attempts
    now capture parsed `active_jobs` evidence after successful preflight and
    before either `--skip-run` or missing launch authorization returns.
    Active jobs or active-job scan errors block through the existing
    `active_jobs` blocker before skip-run or launch-authority success can be
    reported. Independent verifier evidence: `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py -q`
    reported `31 passed in 1.08s`,
    `tests/unit_tests/test_modded_nanogpt_b200_summarize.py -q` reported
    `43 passed in 0.10s`,
    `python -m py_compile experiments/modded_nanogpt_b200/run_speedrun.py experiments/modded_nanogpt_b200/summarize.py`
    passed,
    `git diff --check -- experiments/modded_nanogpt_b200/run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py`
    passed, and `python experiments/modded_nanogpt_b200/verify_static.py`
    reported `Static verification passed for 39 file(s).` No launch, wrapper,
    preflight, GPU, data-prep, download, pip, CUDA, NCCL, Triton,
    FlashAttention, `torchrun`, training, artifact-generating summarization, or
    full-launch command was run for this documentation update. The overall
    objective remains incomplete: no full Lane A/B baseline exists, the
    current index has one skip-run launch-prerequisite row and zero non-skip
    launch-ready rows, and full launch still requires the explicit
    authorization token
    `launch-full-b200`.
  - The real ignored index was refreshed again through the approved
    rootfs-aware summarizer:
    `experiments/modded_nanogpt_b200/summarize.sh --results-root experiments/modded_nanogpt_b200/results --output experiments/modded_nanogpt_b200/results/run_index.json`,
    which exited `0`. No launch, GPU, training, preflight, data-prep,
    download, pip, CUDA, NCCL, `torchrun`, or full-launch command was run.
    Historical refreshed facts, now superseded by the
    `lane_b_full_skiprun_refresh_20260816T111056Z` index refresh:
    `total_attempts=23`, `baseline_count=0`, `launch_prerequisite_count=0`,
    `launch_ready_count=0`, `len(stale_or_demoted_artifacts)=20`, and the
    boundedness check `len(stale_or_demoted_artifacts) <= 20` returned true.
    Representative
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
    with `preflight_data_manifest.verified_sha must be true`. The overall
    objective remains incomplete: no full Lane A/B baseline exists, the later
    refreshed index has one skip-run launch-prerequisite row and zero non-skip
    launch-ready rows, and full launch still requires explicit authorization.
