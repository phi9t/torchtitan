# Mini Kimi K3 Launch Completion Audit

Date: 2026-08-21
Last refreshed: 2026-08-22T21:31Z

Objective: proceed to launch the full Mini Kimi-K3 run end to end.

## Completion Criteria

- Full-mode preflight must return `status=ready`.
- `trainable_config` must pass with separate r1 Trainer-smoke evidence from a
  real TorchTitan `run_train.sh` path that completes at least one optimizer
  step.
- `model_fidelity` must pass with strict first-party forward-oracle evidence
  and reviewed launch-backend evidence.
- `real_corpus_manifest` must pass with at least `5_000_000_000` Kimi-tokenized
  tokens, tokenizer identity/fingerprint, completed decontamination evidence,
  source provenance, and verified shard metadata.
- A guarded `run.sh launch` must start the full training stage and write an
  immutable run-attempt outcome.

## Current Evidence

## 2026-08-23T04:20Z GPU-Resume R1 Corpus and Preflight Refresh

- Resumed GPU-eligible work after the user confirmed no visible GPU jobs were
  running. The full guarded launch was not started because the refreshed
  full-mode preflight remains blocked.
- Completed the capped r1 local shard build:
  `experiments/mini_kimi_k3/data/manifest-r1.json` with
  `experiments/mini_kimi_k3/data/tokens-r1`, using the source budgets from
  `experiments/mini_kimi_k3/reports/r1-corpus-plan.json`.
- Refreshed the r1 corpus plan audit:
  `experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json` reports
  `status=ready`, `planned_tokens=5000000000`,
  `available_tokens=5000000000`, and `deficit_tokens=0`.
- Ran the r1 one-step Trainer smoke inside the rootfs against
  `manifest-r1.json` and `tokens-r1`. The report
  `experiments/mini_kimi_k3/results/r1_training_smoke.json` records
  `status=pass`, selected B200 GPU 0 with at least 12000 MiB free, and the
  `run_train.sh` argv points at the r1 manifest and token directory.
- Fixed the full preflight validator to accept Stage 1 input inspection reports
  that aggregate multiple input manifests via `input_manifest.paths`, while
  still requiring a nonempty path list and `input_manifest.sha256`.
- Refreshed `experiments/mini_kimi_k3/results/preflight.json`. Current gate
  state: `rootfs_runner`, `token_shard_loader`, `trainable_config`,
  `model_fidelity`, and `launch_evidence_bundle` pass;
  `real_corpus_manifest` fails only because the manifest still contains
  placeholder decontamination metadata instead of completed/reviewed
  launch-grade decontamination report path and hash.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. It exits
  blocked with missing `full_preflight_ready`, `real_5b_corpus_manifest`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.
- Focused verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1_input or
  prepare_stage1_local" && pytest -q
  tests/unit_tests/test_mini_kimi_k3_preflight.py -k "input_inspection or
  corpus_plan or decontamination"'` completed with `8 passed, 120 deselected`
  and `12 passed, 63 deselected`.

## 2026-08-22T21:31Z Completion Audit Remote Readiness Provenance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, full launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so final completion
  independently rejects remote Stage 1 readiness evidence unless it carries
  `rootfs_marker=1` and `network_mode=networked`. This prevents a stale or
  hand-edited preflight report from satisfying final launch completion with only
  top-level remote-readiness status fields.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_remote_readiness_without_rootfs_marker`.
  The test failed red because completion audit returned `complete` without
  rootfs-marker evidence, then passed after the final-audit predicate required
  rootfs and network provenance.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_remote_readiness_without_rootfs_marker
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_run_gate_without_real_measurement'`
  passed with `4 passed`.
- Shared execution and Mini-K3 launch verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_scaffold_execution_foundation.py
  tests/unit_tests/test_execution_lifecycle_models.py
  tests/unit_tests/test_execution_lifecycle_run_attempt.py
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_launch_runs_train_stage_after_ready_preflight'`
  passed with `39 passed, 1 skipped`.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria: `full_preflight_ready`,
  `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T21:28Z Final Audit Provenance Cross-Checks

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, full launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so final completion
  independently rejects remote Stage 1 readiness evidence unless it carries
  `rootfs_marker=1` and `network_mode=networked`. This prevents a stale or
  hand-edited preflight report from satisfying final launch completion with only
  top-level remote-readiness status fields.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_remote_readiness_without_rootfs_marker`.
  The test failed red because completion audit returned `complete` without
  rootfs-marker evidence, then passed after the final-audit predicate required
  rootfs and network provenance.
- Also hardened measurement lineage for the final audit: `ConditionStatus` can
  now optionally serialize a `stage_invocation_id`; Mini-K3 launch records the
  train stage invocation in the training evaluation; completion audit requires
  that id to match a successful train terminal event from the same attempt.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_remote_readiness_without_rootfs_marker
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_run_gate_without_real_measurement'`
  passed with `4 passed`.
- Completion-audit subset verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k completion_audit'`
  passed with `15 passed, 95 deselected`.
- A broader Mini-K3 owner-suite rerun was attempted:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`. It was interrupted
  before pytest printed a final summary, after reaching 54% with no failure
  trace. The latest complete owner-suite pass before this slice was
  `264 passed, 14 warnings`.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria: `full_preflight_ready`,
  `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T21:12Z Training Measurement Invocation Link

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened the final completion audit so a real training measurement must name
  the exact train `stage_invocation_id` that succeeded in the coordinator event
  stream and was listed in `outcome.json.stage_invocation_ids`. A completed
  outcome plus any successful train event is no longer enough to satisfy
  `training_measurement_real`.
- Extended `torchtitan.experiments.execution.models.ConditionStatus` with an
  optional `stage_invocation_id` field. Existing callers keep the previous JSON
  shape unless they provide the field.
- Updated the Mini-K3 launch path to record the train stage invocation id in
  the training evaluation for both succeeded and failed train-stage outcomes.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_training_measurement_without_stage_invocation`.
  The test failed red because completion audit returned `complete` for a real
  training evaluation that did not identify its train invocation, then passed
  after the audit required the explicit invocation match.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_training_measurement_without_stage_invocation
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_launch_runs_train_stage_after_ready_preflight
  tests/unit_tests/test_scaffold_execution_foundation.py::test_condition_status_can_identify_owning_stage_invocation
  tests/unit_tests/test_scaffold_execution_foundation.py::test_real_zero_measurement_is_real_not_blocked'`
  passed with `6 passed`.
- Completion-audit subset verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k completion_audit'`
  passed with `14 passed, 95 deselected`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `264 passed, 14 warnings`.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria: `full_preflight_ready`,
  `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T16:59Z R1 Smoke Assets/Dump Provenance Guard

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Further hardened `experiments/mini_kimi_k3/preflight.py` so a passing r1
  Trainer smoke report must prove the recorded `run_train.sh` argv used the
  same `--hf_assets_path` recorded in the report and included exactly one
  `--dump_folder`. This prevents a report from claiming the correct assets
  while the command provenance points at a different tokenizer/assets path or
  omits the training-output directory.
- Updated the r1 smoke report test helper and positive evidence assertions to
  include the real smoke producer's `--hf_assets_path` and `--dump_folder`
  command overrides.
- Verified test-first. The new malformed-command cases in
  `test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract`
  first failed because preflight accepted a mismatched `--hf_assets_path` and a
  missing `--dump_folder`, then passed after the validator change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_passes_with_r1_training_smoke_report
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_reports_blocked_r1_training_smoke'`
  passed with `11 passed, 14 warnings`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `176 passed, 14
  warnings`.
- Static checks: `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/preflight.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed file-quality hooks; only `no-commit-to-branch` failed due to branch
  policy.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`; status
  remains `blocked` because full preflight, r1 GPU capacity, r1 optimizer-step
  evidence, the real 5B-token corpus, Stage 1 remote readiness, guarded launch,
  and real training measurement are still missing.

## 2026-08-22T16:40Z R1 Smoke Launcher Structure Guard

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Further hardened `experiments/mini_kimi_k3/preflight.py` so a passing r1
  Trainer smoke report must record a structurally valid command:
  `/usr/bin/env`, then environment assignments, then exactly one
  `./run_train.sh`, then only run arguments. This catches reports that move
  environment assignments after the launcher or include multiple launcher
  tokens while still containing the expected strings somewhere in argv.
- Verified test-first. The new malformed-command cases in
  `test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract`
  first failed because preflight either accepted a duplicate launcher token or
  reported only the older generic command error. After the parser update:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_passes_with_r1_training_smoke_report'`
  passed with `8 passed, 14 warnings`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `174 passed, 14
  warnings`.
- Static checks: `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/preflight.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed file-quality hooks; only `no-commit-to-branch` failed due to branch
  policy.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`; status
  remains `blocked` because full preflight, r1 GPU capacity, r1 optimizer-step
  evidence, the real 5B-token corpus, Stage 1 remote readiness, guarded launch,
  and real training measurement are still missing.

## 2026-08-22T16:21Z R1 Smoke Strict Command Contract Guard

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/preflight.py` so a passing r1 Trainer
  smoke report must prove that the recorded `run_train.sh` argv used the same
  `--dataloader.token_manifest`, `--dataloader.tokens_dir`, and
  `--training.seq_len` values as the preflight request. A report can no longer
  claim matching data in its JSON body while its command provenance points at a
  different manifest, token directory, or sequence length.
- Tightened the same command contract to reject non-canonical rootfs/command
  cwd, duplicate critical argv keys, missing checkpoint disabling,
  mismatched `CUDA_VISIBLE_DEVICES`, mismatched training steps, and unexpected
  local/global batch values.
- Updated the r1 smoke report test helper and positive evidence assertions to
  include the real smoke producer's data, batch, checkpoint, and step command
  overrides.
- Verified test-first. The new parametrized regression
  `test_preflight_trainable_config_rejects_r1_smoke_command_data_mismatch`
  first failed in all three cases because preflight accepted forged command
  provenance. The new
  `test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract`
  first failed in all five cases because preflight accepted bad cwd, duplicate
  step overrides, missing checkpoint disabling, and GPU-selection drift. Both
  tests passed after the validator change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_command_data_mismatch
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_passes_with_r1_training_smoke_report
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_reports_blocked_r1_training_smoke'`
  passed with `10 passed, 14 warnings`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `172 passed, 14
  warnings`.
- Static checks: `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/preflight.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed file-quality hooks; only `no-commit-to-branch` failed due to branch
  policy.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`; status
  remains `blocked` because full preflight, r1 GPU capacity, r1 optimizer-step
  evidence, the real 5B-token corpus, Stage 1 remote readiness, guarded launch,
  and real training measurement are still missing.

## 2026-08-22T15:33Z R1 Smoke Rootfs/Command Provenance Guard

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/preflight.py` so a passing r1 Trainer
  smoke report must prove it came from the TorchTitan rootfs and the expected
  `run_train.sh` path. The `trainable_config` gate now rejects smoke reports
  without `rootfs.marker=1`, a zero-return command, or the required
  `MODULE=mini_kimi_k3`, `CONFIG=mini_kimi_k3_r1_contract`,
  `COMM_MODE=fake_backend`, `NGPU=1`, and `./run_train.sh` argv entries.
- Updated the r1 smoke report test helper to emit the real producer's
  rootfs/command provenance fields by default, so positive preflight fixtures
  match launch evidence more closely.
- Verified test-first. The new regression
  `test_preflight_trainable_config_rejects_r1_smoke_without_rootfs_command`
  first failed because preflight accepted a host-side/incorrect command smoke
  report, then passed after the validator change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_passes_with_r1_training_smoke_report
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_reports_blocked_r1_training_smoke
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_wrong_r1_training_smoke_report
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_without_gpu_evidence
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_without_rootfs_command'`
  passed with `5 passed, 14 warnings`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `164 passed, 14
  warnings`.

## 2026-08-22T15:14Z R1 Smoke GPU-Evidence Preflight Guard

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/preflight.py` so a passing r1 Trainer
  smoke report must also include selected GPU-capacity evidence with
  `gpu_memory.status=pass`. A hand-written or internally inconsistent smoke
  report with skipped/missing GPU evidence can no longer make
  `trainable_config` pass.
- Aligned the preflight test fixture with the real r1 smoke producer schema:
  `r1_training_smoke.py` writes `gpu_memory.status=pass` when a GPU is selected.
- Verified test-first. The new regression
  `test_preflight_trainable_config_rejects_r1_smoke_without_gpu_evidence`
  first failed because preflight accepted a `status=pass` report whose
  `gpu_memory.status` was `skipped`, then passed after the validator change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_passes_with_r1_training_smoke_report
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_reports_blocked_r1_training_smoke
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_wrong_r1_training_smoke_report
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_without_gpu_evidence'`
  passed with `4 passed, 14 warnings`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `163 passed, 14
  warnings`.

## 2026-08-22T14:59Z R1 Smoke GPU-Status Schema Alignment

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Fixed a completion-audit schema mismatch: `r1_training_smoke.py` and
  `r1_smoke_gpu_readiness.py` emit GPU-memory readiness as `status=pass` when a
  GPU is selected, but `_r1_smoke_gpu_capacity_check()` accepted only
  `status=ready`. The audit now accepts the producer's `pass` value, while
  still tolerating `ready` for older fixture/report compatibility.
- Verified test-first. The positive completed-launch fixture was changed to use
  the producer's real `gpu_memory.status=pass`; it failed because the audit
  stayed blocked, then passed after the audit acceptance change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_corrupt_event_stream
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_readme_documents_completion_audit_guidance'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `162 passed, 14
  warnings`.

## 2026-08-22T14:45Z Event-Stream Guidance Documentation

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Updated `experiments/mini_kimi_k3/README.md` and its README coverage test so
  the documented completion-audit guidance fields include
  `event_stream_errors` alongside `event_stream` and `terminal_events`.
- Focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_corrupt_event_stream
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_readme_documents_completion_audit_guidance'`
  passed with `2 passed`.

## 2026-08-22T14:43Z Event-Stream Error Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Tightened `experiments/mini_kimi_k3/completion_audit.py` so blocked
  `training_measurement_real` and `train_stage_succeeded` next-action guidance includes
  `event_stream_errors` when the coordinator JSONL cannot be parsed cleanly.
  This keeps the operator-facing guidance aligned with the fail-closed audit
  evidence instead of reporting only a generic real-measurement or train-stage
  failure.
- Verified test-first. The focused regression first failed with
  `KeyError: 'event_stream_errors'` in
  `test_mini_kimi_k3_completion_audit_rejects_corrupt_event_stream`, then
  passed after threading the parse errors into `_training_measurement_guidance`
  and `_train_stage_guidance`:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_corrupt_event_stream
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `162 passed, 14
  warnings`.

## 2026-08-22T14:17Z Completion-Audit Event-Stream Fail-Closed Gate

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so malformed,
  missing, or non-object coordinator event-stream rows fail closed for
  `training_measurement_real` and `train_stage_succeeded`. A valid-looking
  `outcome.json` plus one good train event can no longer complete the launch
  checklist if the same attempt's event stream is corrupt.
- Verified the behavior test-first. The new regression first failed because a
  corrupt JSONL row was ignored and an otherwise complete-looking attempt
  reported `complete`, then passed after event-stream errors were threaded into
  the training-measurement and train-stage checks:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_corrupt_event_stream
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `162 passed, 14
  warnings`.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json` without
  launching training. It remains blocked with current covered items
  `model_fidelity`, `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`; current missing items remain
  `full_preflight_ready`, `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.
- Updated `experiments/mini_kimi_k3/README.md` to document that missing or
  malformed coordinator event streams fail closed for completion-audit
  train-stage and real-measurement checks.

## 2026-08-22T13:51Z Completion-Audit Schema-Version Gate

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so the
  full-preflight and guarded-launch outcome checks require `schema_version=1`.
  A stale or incompatible preflight/outcome schema can no longer satisfy the
  full-launch completion checklist just because `kind` and status fields look
  ready.
- Verified the behavior test-first. The new regressions first failed because
  otherwise complete-looking reports with `schema_version=0` let the audit
  report `complete`, then passed after the audit validation change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_stale_preflight_schema
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_stale_outcome_schema
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `161 passed, 14
  warnings`.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. It
  remains blocked with current covered items `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`; current missing items remain
  `full_preflight_ready`, `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.
- Updated `experiments/mini_kimi_k3/README.md` to document that completion
  audit requires schema version `1` for both full-preflight and outcome reports.

## 2026-08-22T13:36Z Completion-Audit Run-Gate Measurement Consistency Check

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so
  `training_measurement_real` requires
  `outcome.json.run_gate.has_real_measurement=true` in addition to a completed
  real training evaluation and a matching successful train-stage invocation.
  This prevents stale or manually inconsistent outcome metadata from satisfying
  the full-launch completion checklist.
- Verified the behavior test-first. The new regression first failed because an
  outcome with `evaluations.training.measurement=real` but
  `run_gate.has_real_measurement=false` let the audit report `complete`, then
  passed after the audit validation change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_run_gate_without_real_measurement
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `159 passed, 14
  warnings`.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. It
  remains blocked with current covered items `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`; current missing items remain
  `full_preflight_ready`, `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.
- Updated `experiments/mini_kimi_k3/README.md` to document the
  `run_gate.has_real_measurement` consistency requirement.

## 2026-08-22T13:20Z Completion-Audit Training Measurement Provenance Check

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so
  `training_measurement_real` does not trust `outcome.json.evaluations.training`
  alone. It now also requires a successful train-stage event whose
  `stage_invocation_id` appears in `outcome.json.stage_invocation_ids`.
- Verified the behavior test-first. The existing stale train-event regression
  was extended to assert that `training_measurement_real` fails when the only
  successful train event is not listed by the outcome; it failed before the
  audit change and passed afterward:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_unlisted_train_event
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `158 passed, 14
  warnings`.
- Re-ran the full Mini-K3 unit-test surface inside the rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `236 passed, 14 warnings in 558.85s (0:09:18)`. The warnings were
  the existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. It
  remains blocked with current covered items `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`; current missing items remain
  `full_preflight_ready`, `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.
- Updated `experiments/mini_kimi_k3/README.md` to document the shared
  `stage_invocation_id` requirement for real training-measurement and
  train-stage completion evidence.
- Focused README/regression validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_unlisted_train_event
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_readme_documents_completion_audit_guidance'`
  passed with `2 passed`.
- `git diff --check` passed after the README and ledger edits. Targeted
  pre-commit on `experiments/mini_kimi_k3/completion_audit.py`,
  `experiments/mini_kimi_k3/README.md`,
  `tests/unit_tests/test_mini_kimi_k3_runner.py`, and this ledger passed all
  file-quality hooks; the repository branch-policy hook
  `no-commit-to-branch` failed as expected for the current branch.

## 2026-08-22T12:57Z Completion-Audit Train Event Linkage Check

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so the
  `train_stage_succeeded` check only passes when the successful train-stage
  event's `stage_invocation_id` is listed in `outcome.json.stage_invocation_ids`.
  This prevents a stale or mixed coordinator event stream from satisfying a
  different guarded-attempt outcome.
- Verified the behavior test-first. The new regression first failed because an
  unlisted train `stage_succeeded` event let the audit report `complete`, then
  passed after the audit validation change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_unlisted_train_event
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `157 passed, 14
  warnings`.
- Re-ran the full Mini-K3 unit-test surface inside the rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `236 passed, 14 warnings in 553.28s (0:09:13)`. The warnings were
  the existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- `git diff --check` passed. Targeted pre-commit on
  `experiments/mini_kimi_k3/completion_audit.py`,
  `tests/unit_tests/test_mini_kimi_k3_runner.py`, and this ledger passed all
  file-quality hooks; the repository branch-policy hook
  `no-commit-to-branch` failed as expected for the current branch.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. It
  remains blocked with current covered items `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`; current missing items remain
  `full_preflight_ready`, `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T12:24Z Completion-Audit Outcome Identity Check

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so the
  `guarded_launch_completed` check verifies that the terminal `outcome.json`
  payload belongs to the audited attempt. A stale or copied outcome with a
  different `attempt_id` now fails completion even if it reports
  `execution_outcome=completed`.
- Verified the behavior test-first. The new regression first failed because a
  stale `attempt_id` outcome let the audit report `complete`, then passed after
  the audit validation change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_wrong_attempt_outcome
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `157 passed, 14
  warnings`.
- Re-ran the full Mini-K3 unit-test surface inside the rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `235 passed, 14 warnings in 560.71s (0:09:20)`. The warnings were
  the existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- `git diff --check` passed, and targeted pre-commit passed for
  `experiments/mini_kimi_k3/completion_audit.py`,
  `tests/unit_tests/test_mini_kimi_k3_runner.py`, and this ledger. The
  repository branch-policy hook `no-commit-to-branch` was intentionally
  skipped.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. It
  remains blocked with current covered items `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`; current missing items remain
  `full_preflight_ready`, `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T12:02Z Completion-Audit Launch-Bundle Identity Check

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so completion audit
  independently validates that the preflight `launch_evidence_bundle` evidence
  matches the requested full r1 attempt identity. It now rejects mismatched
  `run_id`, `attempt_id`, `family`, `task`, `lane`, or `model_variant` even if
  the preflight gate says `status=pass`.
- Verified the behavior test-first. The new completion-audit regression first
  failed because a wrong-lane launch bundle let the audit report `complete`,
  then passed after the audit validation change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_wrong_lane_launch_bundle
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader focused validation passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `156 passed, 14
  warnings`.
- Re-ran the full Mini-K3 unit-test surface inside the rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `234 passed, 14 warnings in 559.48s (0:09:19)`. The warnings were
  the existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- `git diff --check` passed, and targeted pre-commit passed for
  `experiments/mini_kimi_k3/completion_audit.py`,
  `tests/unit_tests/test_mini_kimi_k3_runner.py`, and this ledger. The
  repository branch-policy hook `no-commit-to-branch` was intentionally
  skipped.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. It
  remains blocked with current covered items `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`; current missing items remain
  `full_preflight_ready`, `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T11:36Z Evidence Bundle Lane Consistency Gate

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened the launch-evidence preflight gate so an initialized immutable
  attempt bundle must match the requested launch identity before it can satisfy
  `launch_evidence_bundle`. The inspected evidence now carries and validates
  `run_id`, `attempt_id`, `family=mini_kimi_k3`, `task=pretraining`, the launch
  lane, and `model_variant=r1`.
- Verified the behavior test-first. The new wrong-lane regression first failed
  because a `tiny-plumbing` evidence bundle passed a `full` preflight gate, then
  passed after the gate validation change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_rejects_wrong_lane_launch_evidence_bundle
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_accepts_initialized_launch_evidence_bundle
  tests/unit_tests/test_mini_kimi_k3_evidence.py'` passed with `5 passed, 14
  warnings`.
- Re-ran the full Mini-K3 unit-test surface inside the rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `233 passed, 14 warnings in 557.83s (0:09:17)`. The warnings were
  the existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- `git diff --check` passed, and targeted pre-commit passed for
  `experiments/mini_kimi_k3/evidence.py`,
  `experiments/mini_kimi_k3/preflight.py`,
  `tests/unit_tests/test_mini_kimi_k3_evidence.py`,
  `tests/unit_tests/test_mini_kimi_k3_preflight.py`, and this ledger. The
  repository branch-policy hook `no-commit-to-branch` was intentionally
  skipped.
- Refreshed `experiments/mini_kimi_k3/results/preflight.json` with the
  corrected `stage1-source-resolution.json` evidence and the stronger
  launch-bundle identity evidence. Full preflight still exited `21`, as
  expected, because the actual corpus, Stage 1 credentials, and r1 training
  smoke prerequisites remain blocked.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json`. Current
  covered items are `model_fidelity`, `raw_shard_disk_capacity`,
  `stage1_source_resolution`, and `launch_evidence_bundle`; current missing
  items remain `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T11:20Z Launcher Preflight Report Fail-Closed Gate

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Hardened `experiments/mini_kimi_k3/launch.py` so a zero-return preflight
  subprocess is not enough to start training. The launch coordinator now loads
  the derived preflight report and requires `kind=mini_kimi_k3_preflight`, the
  requested mode, and `status=ready` before launching the train stage.
- Verified the behavior test-first. The new regression test first failed
  because `run_launch` returned `0` and ran the fake train stage after a
  blocked preflight report, then passed after the launcher validation change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_launch_blocks_on_stale_preflight_report
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_launch_runs_train_stage_after_ready_preflight'`
  passed with `2 passed`.
- Re-ran the full Mini-K3 unit-test surface inside the rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `232 passed, 14 warnings in 552.01s (0:09:12)`. The warnings were
  the existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- `git diff --check` passed.

## 2026-08-22T11:05Z Full Mini-K3 Suite After Guidance Updates

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Re-ran the full Mini-K3 unit-test surface after the completion-audit CLI,
  source-deficit guidance, README, and wizard updates:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `231 passed, 14 warnings in 551.30s (0:09:11)`. The warnings were
  the existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- Re-ran broad targeted pre-commit across the Mini-K3 scaffold surface:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  SKIP=no-commit-to-branch pre-commit run --files ...'`.
- Result: all actionable hooks passed: trailing whitespace, Python AST, merge
  conflict check, large-file check, EOF fix, license comment insertion, flake8,
  ufmt, pydoclint, codespell, Pyrefly, and lychee. The repository branch-policy
  hook `no-commit-to-branch` was intentionally skipped.

## 2026-08-22T10:53Z Wizard Points To Audit Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Updated `.scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh` so
  the initial inspection stage tells the operator to review the completion
  audit CLI `next actions:` lines and `guidance:` details, especially
  `source_deficits` for corpus acquisition.
- Verified the behavior test-first. The focused wizard test first failed
  because the initial inspection stage did not mention the CLI guidance output,
  then passed after the prompt update:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_unblock_wizard_points_to_completion_audit_cli_guidance'`
  passed with `1 passed`.
- Broader focused verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or readme_documents_completion_audit_guidance or
  unblock_wizard" && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && python -m
  py_compile experiments/mini_kimi_k3/completion_audit.py' && git diff
  --check`.
- Targeted rootfs pre-commit passed with `SKIP=no-commit-to-branch` for the
  audit module, preflight module, Mini-K3 README, runner/preflight tests,
  wizard, and this ledger.

## 2026-08-22T10:49Z Broad Mini-K3 Pre-Commit Check

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Ran targeted pre-commit across the full Mini-K3 scaffold surface, including
  the reusable `torchtitan/experiments/mini_kimi_k3/` package, operational
  `experiments/mini_kimi_k3/` package, all `test_mini_kimi_k3_*.py` tests, the
  operator wizard, and the Mini-K3 scratch env template/ledger.
- Command:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  SKIP=no-commit-to-branch pre-commit run --files ...'`.
- Result: all actionable hooks passed: trailing whitespace, Python AST, merge
  conflict check, large-file check, EOF fix, license comment insertion, flake8,
  ufmt, pydoclint, codespell, Pyrefly, and lychee. The repository branch-policy
  hook `no-commit-to-branch` was intentionally skipped as in prior targeted
  checks.
- This is a broad non-GPU quality gate for the current scaffold, not evidence
  that the full launch is ready.

## 2026-08-22T10:47Z Completion-Audit CLI Docs Sync

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Updated `experiments/mini_kimi_k3/README.md` so the completion-audit section
  documents the human-facing blocked CLI output: `next actions:` and compact
  `guidance:` lines.
- Verified the docs behavior test-first. The focused README test first failed
  because the CLI output strings were undocumented, then passed after updating
  the README:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_readme_documents_completion_audit_guidance'`
  passed with `1 passed`.
- Broader focused verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_readme_documents_completion_audit_guidance &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or readme_documents_completion_audit_guidance" &&
  python -m py_compile experiments/mini_kimi_k3/completion_audit.py' && git
  diff --check`.
- Targeted rootfs pre-commit passed with `SKIP=no-commit-to-branch` for the
  audit module, preflight module, Mini-K3 README, runner/preflight tests, and
  this ledger.

## 2026-08-22T10:44Z Full Mini-K3 Unit-Suite Refresh

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Ran the full Mini-K3 unit-test surface inside the rootfs after the recent
  completion-audit and preflight evidence changes:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `230 passed, 14 warnings in 549.59s (0:09:09)`. The warnings were the
  existing `torch.jit.script_method` deprecation warnings from
  `test_mini_kimi_k3_config_registry.py`.
- This verifies the non-GPU Mini-K3 scaffold/test surface after the latest
  audit-report changes. It does not prove full launch readiness because the live
  completion audit remains blocked on corpus, r1 Trainer-smoke, Stage 1 remote
  readiness, and launch-stage evidence.

## 2026-08-22T10:32Z Corpus Source-Deficit Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Updated full preflight evidence preservation so the embedded
  `r1_corpus_plan_audit` object retains its per-source `sources` map from the
  validated corpus-plan audit report.
- Updated `experiments/mini_kimi_k3/completion_audit.py` so corpus shortfall
  next-action guidance includes `source_deficits` for only the sources with
  positive deficits. Ready sources are omitted from the operator guidance.
- Verified the behavior test-first. The focused completion-audit test first
  failed because `source_deficits` was absent, then passed after preserving the
  source map and deriving source-level deficit guidance:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt'`
  passed with `1 passed`.
- Broader focused verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or readme_documents_completion_audit_guidance"
  tests/unit_tests/test_mini_kimi_k3_preflight.py -k "r1_corpus_plan_audit" &&
  python -m py_compile experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/preflight.py' && git diff --check`.
- Targeted rootfs pre-commit passed with `SKIP=no-commit-to-branch` for the
  audit module, preflight module, Mini-K3 README, runner/preflight tests, and
  this ledger.
- Refreshed full preflight and completion audit only. The completion audit still
  exits `21` with `status=blocked`, but now prints source-level deficits for
  `fineweb-edu`, `web-diverse`, `code-python`, `finemath`, `open-web-math`, and
  `cosmopedia`.

## 2026-08-22T10:25Z Completion-Audit CLI Guidance Details

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Updated `experiments/mini_kimi_k3/completion_audit.py` so each blocked CLI
  next-action line is followed by a compact `guidance:` line when structured
  guidance exists in the report. Lists and nested objects are rendered as stable
  JSON snippets, while scalar fields remain directly readable.
- Verified the behavior test-first. The focused CLI test first failed because
  the stderr output did not include guidance details, then passed after adding
  the formatter:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_cli_prints_missing_summary'`
  passed with `1 passed`.
- Broader focused verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or readme_documents_completion_audit_guidance" &&
  python -m py_compile experiments/mini_kimi_k3/completion_audit.py'`.
- Targeted rootfs pre-commit passed with `SKIP=no-commit-to-branch` for the
  audit module, README, runner tests, and this ledger.
- Refreshed only the completion audit report. It still exits `21` with
  `status=blocked`, but now prints exact guidance values for each missing
  launch blocker, including token deficit, Modal env-file template, r1 smoke
  report, attempt outcome report, and coordinator event stream.

## 2026-08-22T10:21Z Completion-Audit CLI Next Actions

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Updated `experiments/mini_kimi_k3/completion_audit.py` so a blocked CLI run
  prints a `next actions:` section with each missing requirement mapped to its
  operator action, in addition to writing the machine-readable JSON report.
- Verified the behavior test-first. The focused CLI test first failed because
  `next actions:` was not printed, then passed after adding the stderr summary:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_cli_prints_missing_summary'`
  passed with `1 passed`.
- Broader focused verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or readme_documents_completion_audit_guidance" &&
  python -m py_compile experiments/mini_kimi_k3/completion_audit.py'`.
- Refreshed only the completion audit report. It exited `21` as expected and
  printed next actions for the remaining launch blockers. The JSON report
  remains `status=blocked`.
- Targeted rootfs pre-commit passed with `SKIP=no-commit-to-branch` for
  `completion_audit.py`, the Mini-K3 README, the runner tests, and this ledger.

## 2026-08-22T10:16Z Completion-Audit README Sync

- Updated `experiments/mini_kimi_k3/README.md` so the completion-audit
  `next_actions[].guidance` section documents the post-preflight launch fields
  now emitted by the report: `attempt_outcome_report`, `execution_outcome`,
  `measurement`, `event_stream`, and `terminal_events`.
- Verified the docs behavior test-first. The focused README test first failed
  because `attempt_outcome_report` was undocumented, then passed after updating
  the README:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_readme_documents_completion_audit_guidance'`
  passed with `1 passed`.
- No GPU smoke, training, guarded launch, or Modal ingestion was attempted.

## 2026-08-22T10:12Z Post-Preflight Launch Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted.
- Added focused completion-audit guidance for the post-preflight launch
  blockers: `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`. The guidance now points directly at the attempt
  outcome report or coordinator event stream and carries the relevant current
  status fields.
- Verified the behavior test-first. The focused test first failed because
  `guarded_launch_completed` had no `guidance`, then passed after adding the
  launch-stage guidance:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt'`
  passed with `1 passed`.
- Refreshed only the completion audit report:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local
  --attempt-id attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` as
  expected and wrote `status=blocked`.
- The refreshed audit still blocks on full preflight, r1 smoke GPU capacity,
  r1 optimizer-step evidence, the 5B corpus manifest and corpus-plan audit,
  Stage 1 remote readiness, guarded launch completion, real training
  measurement, and train-stage success.

## 2026-08-22T10:06Z Targeted Pre-Commit Hygiene

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Ran targeted `pre-commit run --files ...` over the Mini-K3 source/test/doc
  surface. The first run failed only on actionable lint plus the repository
  `no-commit-to-branch` policy hook.
- Fixed the actionable lint findings:
  - wrapped long `_NEXT_ACTIONS` strings in
    `experiments/mini_kimi_k3/completion_audit.py`;
  - moved `Callable` in `experiments/mini_kimi_k3/corpus_disk_readiness.py`
    from `typing` to `collections.abc`;
  - removed the unused `CommandResult` import from
    `experiments/mini_kimi_k3/stage1_remote_readiness.py`.
- Re-ran the relevant unit/static checks:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1_remote or
  completion_audit or unblock_wizard" && python -m py_compile
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/corpus_disk_readiness.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `21 passed, 72 deselected` and clean static checks.
- Re-ran targeted pre-commit with only `no-commit-to-branch` skipped:
  `SKIP=no-commit-to-branch pre-commit run --files ...` passed all remaining
  hooks: trailing whitespace, Python AST, merge conflicts, large files, EOF,
  license comments, flake8, µfmt, pydoclint, codespell, Pyrefly, and lychee.
- The unskipped targeted pre-commit remains blocked by repository branch policy
  only: `don't commit to branch` failed.

## 2026-08-22T09:58Z Stage 1 Env File Fail-Closed Validation

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `experiments/mini_kimi_k3/stage1_remote_readiness.py` so
  `--env-file` validation fails closed before running DNS or Modal subprocess
  probes when the file is missing required Modal keys or still contains
  placeholder values from `.env.example`.
- Added regression coverage for a copied-but-unedited `.env.example` shape:
  missing `MODAL_PROFILE` plus placeholder `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET` now writes a blocked readiness report with the template
  path and required key list.
- Verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_readiness_blocks_placeholder_env_file &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1_remote or
  completion_audit or unblock_wizard"'` passed with `1 passed`, then `21
  passed, 72 deselected`.
- Refreshed Stage 1 readiness, full preflight, and completion audit only. The
  reports remain `status=blocked`; no remote or GPU work was started. The
  readiness report now stops at `checks.env_file` while the real `.env` is
  absent.
- Static verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd
  /workspace/torchtitan && python -m py_compile
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/completion_audit.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.

## 2026-08-22T09:54Z Wizard Modal Profile Write

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `.scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh` so
  the Modal Credentials stage prompts for the required `MODAL_PROFILE`, defaults
  it to `mini-k3`, and writes it to the ignored `.env` file alongside
  `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
- Verified the change test-first. The focused wizard test first failed because
  the Modal stage did not prompt/write `MODAL_PROFILE`; after the patch it
  passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_unblock_wizard_writes_required_modal_profile'`
  passed with `1 passed`.
- Broader non-GPU runner verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit
  or stage1_remote or unblock_wizard"'` passed with `20 passed, 72
  deselected`.
- Refreshed completion audit only:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id
  attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` and
  wrote `status=blocked`, as expected.
- Static verification: `bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.

## 2026-08-22T09:50Z Full Mini-K3 Unit-Suite Verification

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Ran the full Mini-K3 unit-test surface inside the rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`.
- Result: `228 passed, 14 warnings in 541.70s`.
- This validates the current non-GPU scaffold, gates, reports, model-shape
  helpers, token-shard helpers, and runner logic. It does not cover the blocked
  full-launch requirements: GPU capacity, a real r1 Trainer optimizer step, a
  5B-token corpus manifest, or a completed guarded train stage.

## 2026-08-22T09:39Z README Guidance Contract

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `experiments/mini_kimi_k3/README.md` so the completion-audit section
  documents `next_actions[].guidance` and the currently emitted guidance fields:
  `blocked_preflight_requirements`, `target_tokens`, `available_tokens`,
  `deficit_tokens`, `r1_training_smoke_report`, and `env_file_template`.
- Added a README contract test so future changes to the audit schema keep the
  operator-facing documentation in sync.
- Verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_readme_documents_completion_audit_guidance &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit
  or stage1_remote or unblock_wizard" && python -m py_compile
  experiments/mini_kimi_k3/completion_audit.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `1 passed`, then `19 passed, 72 deselected`, and clean
  static checks.
- The current completion audit remains `status=blocked`. It records structured
  guidance for full preflight blockers, r1 smoke GPU/optimizer-step blockers,
  corpus token deficit blockers, and the Modal env-file blocker.

## 2026-08-22T09:35Z Full-Preflight Blocker Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `experiments/mini_kimi_k3/completion_audit.py` so the blocked
  `full_preflight_ready` next action includes `blocked_preflight_requirements`
  rather than only pointing back to the same blocked preflight report.
- The refreshed completion audit now identifies the current preflight-owned
  blockers as: `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`, and
  `stage1_remote_readiness`.
- Verified the change test-first. The focused completion-audit test first failed
  because `full_preflight_ready` had no `guidance`; after the patch it passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt'`
  passed with `1 passed`.
- Broader non-GPU runner verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit
  or stage1_remote or unblock_wizard"'` passed with `18 passed, 72
  deselected`.
- Refreshed completion audit only:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id
  attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` and
  wrote `status=blocked`, now with a concise preflight-blocker list.
- Static verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd
  /workspace/torchtitan && python -m py_compile
  experiments/mini_kimi_k3/completion_audit.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.

## 2026-08-22T09:31Z R1 Smoke Next-Action Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `experiments/mini_kimi_k3/completion_audit.py` so blocked
  `r1_smoke_gpu_capacity` and `r1_trainer_optimizer_step` next actions carry
  structured guidance from the `trainable_config` gate's
  `r1_training_smoke` evidence.
- The refreshed completion audit now records
  `r1_training_smoke_report=experiments/mini_kimi_k3/results/r1_training_smoke.json`,
  `required_free_mib=12000`, `selected=null`, `status=blocked`, and `steps=0`
  in the appropriate next-action guidance.
- Verified the change test-first. The focused completion-audit test first failed
  because `r1_smoke_gpu_capacity` had no `guidance`; after the patch it passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt'`
  passed with `1 passed`.
- Broader non-GPU runner verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit
  or stage1_remote or unblock_wizard"'` passed with `18 passed, 72
  deselected`.
- Refreshed completion audit only:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id
  attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` and
  wrote `status=blocked`, now with r1 smoke guidance on the GPU-capacity and
  optimizer-step next actions.
- Static verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd
  /workspace/torchtitan && python -m py_compile
  experiments/mini_kimi_k3/completion_audit.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.

## 2026-08-22T09:27Z Corpus Deficit Next-Action Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `experiments/mini_kimi_k3/completion_audit.py` so blocked
  `real_5b_corpus_manifest` and `r1_corpus_plan_audit` next actions carry
  structured token guidance from the r1 corpus-plan audit evidence:
  `target_tokens`, `available_tokens`, and `deficit_tokens`.
- Verified the change test-first. The focused completion-audit test first failed
  because `real_5b_corpus_manifest` had no `guidance`; after the patch it
  passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt'`
  passed with `1 passed`.
- Broader non-GPU runner verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit
  or stage1_remote or unblock_wizard"'` passed with `18 passed, 72
  deselected`.
- Refreshed completion audit only:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id
  attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` and
  wrote `status=blocked`, now with token-deficit guidance on both corpus next
  actions.
- Static verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd
  /workspace/torchtitan && python -m py_compile
  experiments/mini_kimi_k3/completion_audit.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.

## 2026-08-22T09:22Z Completion-Audit Next-Action Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `experiments/mini_kimi_k3/completion_audit.py` so the blocked
  `stage1_remote_readiness` next action carries structured `guidance` copied
  from the preflight env-file evidence: `env_file`,
  `env_file_template`, and `required_keys`.
- Verified the change test-first. The focused completion-audit test first failed
  because the next action had no `guidance`; after the patch it passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt'`
  passed with `1 passed`.
- Broader non-GPU runner verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit
  or stage1_remote or unblock_wizard"'` passed with `18 passed, 72
  deselected`.
- Refreshed completion audit only:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id
  attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` and
  wrote `status=blocked`, now with structured Modal env guidance.
- Static verification: `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd
  /workspace/torchtitan && python -m py_compile
  experiments/mini_kimi_k3/completion_audit.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.

## 2026-08-22T09:18Z Wizard Modal Env Guidance

- Continued under the active no-GPU/no-launch constraint. No GPU smoke,
  training, guarded launch, or Modal ingestion was attempted in this refresh.
- Updated `.scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh` so
  the Modal Credentials stage points operators at the checked-in placeholder
  template `.scratch/mini-kimi-k3-replication/.env.example` and names the
  required keys: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `MODAL_PROFILE`.
- Verified the change test-first. The focused wizard test first failed because
  the wizard did not define or print the env-template guidance; after the
  wizard patch, the focused test passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_unblock_wizard_documents_modal_env_template'`
  passed with `1 passed`.
- Broader wizard-adjacent verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "unblock_wizard or
  stage1_remote_env_template"'` passed with `6 passed, 84 deselected`.
- Static verification: `bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.

## 2026-08-22T06:04Z Local Stage 1 Route Evidence Hardening

- Continued under the active no-GPU/no-launch constraint. No GPU smoke, full
  preflight, full launch, or Modal ingestion was attempted in this refresh.
- Tightened the local Stage 1 corpus route so `local_stage1` can only make
  remote readiness optional when its report is local, readable, hash-matched,
  `kind=mini_kimi_k3_source_provenance`, `schema_version=1`, source-matched,
  and token-count-matched.
- Verified the new guard test-first. The initial red test showed a stale
  `local_stage1.sha256` still allowed `real_corpus_manifest=pass`; after the
  validator change, the focused test passed.
- Focused verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_local_stage1_report_hash_mismatch
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_local_stage1_token_count_mismatch
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_treats_remote_readiness_as_optional_for_local_manifest'`
  passed with `3 passed, 14 warnings`.
- Broader non-GPU preflight verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "local_stage1 or
  source_sidecar or registered_shard_metadata or manifest_with_corpus_metadata
  or remote_readiness"'` passed with `11 passed, 39 deselected, 14 warnings`.
- Static verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/preflight.py' && git diff
  --check` passed.
- Refreshed completion audit only:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local
  --attempt-id attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` and
  wrote `status=blocked`.
- Read-only audit identified the next safe non-GPU readiness gap: wire
  `inspect-stage1-inputs --min-total-tokens 5000000000` reports into full
  preflight and completion-audit as first-class local-input readiness evidence.
  This would validate local input existence, hashes, parseability, and token
  budget before any shard build, GPU work, or Modal ingest.

## 2026-08-22T05:54Z Non-GPU Stop-Point Refresh

- Honored the current operator constraint to stop running the Mini-K3
  experiment and focus only on non-GPU work. No GPU smoke, full preflight,
  full launch, or Modal ingestion was attempted in this refresh.
- Confirmed the imported Stage 1 sidecar hash contract:
  `import-shards` records `source_sidecar_sha256`; full preflight rejects
  missing, mismatched, stale, non-`uint32`, or token-count-inconsistent
  source sidecars.
- Focused sidecar verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_missing_import_source_sidecar
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_stale_import_source_sidecar
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_import_source_sidecar_hash_mismatch
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_import_source_sidecar_without_hash'`
  passed with `4 passed, 14 warnings`.
- Broader non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "source_sidecar or registered_shard_metadata or manifest_with_corpus_metadata
  or remote_readiness" tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "import_shards or stage1 or corpus_plan" && python -m py_compile
  experiments/mini_kimi_k3/import_shards.py experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/env_file.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `34 passed, 96 deselected, 14 warnings`.
- Refreshed completion audit only:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local
  --attempt-id attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json'` exited `21` and
  wrote `status=blocked`, as expected while launch prerequisites are still
  missing.
- Remaining launch blockers are unchanged: full preflight is blocked, r1 GPU
  capacity and optimizer-step evidence are missing, the real 5B-token corpus
  manifest is missing, Stage 1 remote readiness is blocked unless a complete
  local/imported corpus route is provided, and no guarded train stage has
  completed.

Latest verification after adding local Stage 1/decontamination tooling, the r1
corpus-plan deficit audit, preflight/launch forwarding for the Stage 1
source-resolution and remote-readiness reports, and the guarded remote Stage 1
wrapper:

- `experiments/mini_kimi_k3/run.sh plan-r1-corpus --mix-plan experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara/data/mix_plan.json --report experiments/mini_kimi_k3/reports/r1-corpus-plan.json`:
  wrote a 5B-token acquisition plan with six source budgets
- `experiments/mini_kimi_k3/run.sh audit-r1-corpus-plan --manifest experiments/mini_kimi_k3/data/manifest.json --tokens-dir experiments/mini_kimi_k3/data/tokens --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json --report experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json`:
  exit `21`, report `status=blocked`, `available_tokens=4097`,
  `deficit_tokens=4999995903`
- `experiments/mini_kimi_k3/run.sh corpus-disk-readiness --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json --target-path experiments/mini_kimi_k3/data/tokens --report experiments/mini_kimi_k3/reports/corpus-disk-readiness.json --overhead-fraction 0.20`:
  added as a non-spendful readiness check for the future raw `uint32` shard
  filesystem. It checks free bytes against the 5B-token corpus plan plus
  overhead, writes `mini_kimi_k3_corpus_disk_readiness`, and does not create or
  acquire corpus data.
- `experiments/mini_kimi_k3/run.sh launch --results-root experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id attempt-guarded-current-003 --mode full ... --r1-corpus-plan-audit-report experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json --stage1-source-resolution-report experiments/mini_kimi_k3/reports/stage1-source-resolution.json --stage1-remote-readiness-report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json`:
  exit `21`, attempt outcome `blocked`, no train stage launched
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "duplicate_shards or path_escape or noninteger_metadata or inconsistent_plan_total"`:
  `4 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "corpus_plan"`:
  `7 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "r1_corpus_plan_audit"`:
  `2 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "launch_passes_forward_oracle_report or corpus_plan"`:
  `8 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "plans_r1_corpus or writes_stage1 or write_stage1 or inspect_stage1 or inspects_stage1 or input_manifest or stage1 or build_local_shards or import_shards or decontam"`:
  `16 passed`
- `experiments/mini_kimi_k3/run.sh r1-smoke-gpu-readiness --report experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json --min-free-gpu-memory-mib 12000`:
  exit `21`, report `status=blocked`, no selected GPU, best free GPU
  `9745` MiB
- `fuser -v /dev/nvidia*`:
  showed the occupied devices are still held by user-owned `python`,
  `sglang::scheduler`, and `sglang::detoken` processes, including PIDs
  `1958391`, `1959989` through `1959997`, and `1984726` through `1984736`.
  Plain `nvidia-smi` reported high resident memory on all eight B200s but no
  running processes in its process table, so `fuser` is the useful evidence for
  the GPU-memory blocker.
- `ps -p <gpu-holder-pids> -o user=,pid=,ppid=,stat=,etime=,cmd=` plus
  `/proc/<pid>/cwd`:
  showed the primary GPU holder is a GLM-5.2 SGLang server launched from
  `/workspace/monarch`, using `/cache/glm52/venvs/sglang/bin/python -m
  sglang.launch_server ... --tp 8 ... --mem-fraction-static 0.93`, with child
  `sglang::scheduler_TP0` through `sglang::scheduler_TP7` and compile-worker
  processes. This is not a TorchTitan-owned process to terminate without
  explicit operator authorization.
- `experiments/mini_kimi_k3/run.sh r1-smoke-gpu-readiness ...` and
  `experiments/mini_kimi_k3/run.sh r1-training-smoke ...` now include
  `gpu_memory.holder_detail` in their reports when no holder PIDs are visible
  from inside the rootfs process namespace. The current in-rootfs reports say
  `no GPU holder processes were visible from the current process namespace`;
  the host-side `fuser` check above remains the actionable holder evidence.
- `experiments/mini_kimi_k3/run.sh r1-training-smoke --token-manifest experiments/mini_kimi_k3/data/manifest.json --tokens-dir experiments/mini_kimi_k3/data/tokens --report experiments/mini_kimi_k3/results/r1_training_smoke.json --seq-len 4096 --steps 1`:
  exit `21`, report `status=blocked`, error
  `insufficient free GPU memory for r1 Trainer smoke`,
  `optimization.steps=0`, `optimization.max_grad_norm=null`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "r1_smoke_gpu_readiness or r1_training_smoke or launch_blocks or preflight"`:
  `8 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "writes_stage1 or write_stage1 or inspect_stage1 or inspects_stage1 or input_manifest or stage1 or build_local_shards or import_shards or decontam"`:
  `15 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "inspect_stage1 or inspects_stage1"`:
  `3 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1 or input_manifest or inspect_stage1 or inspects_stage1 or build_local_shards or import_shards or decontam"`:
  `13 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "decontam"`:
  `3 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1 or build_local_shards or import_shards or decontam"`:
  `8 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "input_manifest"`:
  `2 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1 or input_manifest or build_local_shards or import_shards or decontam"`:
  `10 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "decontamination_report"`:
  `5 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1 or build_local_shards or import_shards or decontam" tests/unit_tests/test_mini_kimi_k3_preflight.py -k "decontamination or corpus_metadata or source_provenance or token_manifest"`:
  `14 passed`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "stage1_remote_readiness or stage1_source_resolution or r1_corpus_plan_audit"`:
  `4 passed, 37 deselected, 14 warnings`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1_remote_readiness or launch_passes_forward_oracle_report"`:
  `2 passed, 61 deselected`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit or stage1_remote_readiness or launch_passes_forward_oracle_report"`:
  `4 passed, 61 deselected`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_*.py`: `185 passed,
  14 warnings`
- `python -m py_compile experiments/mini_kimi_k3/audit_r1_corpus_plan.py experiments/mini_kimi_k3/preflight.py experiments/mini_kimi_k3/launch.py experiments/mini_kimi_k3/stage1_remote_readiness.py`:
  passed
- `git diff --check`: passed
- `TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan/experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara && python data/probe.py'`:
  passed as an anonymous metadata-only Hugging Face probe and rewrote
  `experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara/data/resolved.json`
  with zero unresolved benchmarks and zero unresolved corpora. The selected
  corpus sources are `HuggingFaceFW/fineweb-edu` sample-100BT,
  `HuggingFaceFW/fineweb` sample-100BT as the `web-diverse` fallback,
  `HuggingFaceTB/finemath` finemath-4plus,
  `open-web-math/open-web-math`, `codeparrot/github-code-clean` Python-all as
  the `code-python` fallback, and `HuggingFaceTB/cosmopedia-v2`.
- `TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && uv tool run --from modal modal --version'`:
  passed with `modal client version: 1.5.4`. This verifies an isolated Modal
  CLI path inside the networked rootfs without modifying the broken repo
  `.venv` or launching remote functions. No `MODAL_*` token variables were
  visible in the rootfs environment; `modal profile current` returned
  `default`.
- `experiments/mini_kimi_k3/run.sh probe-stage1-sources --source-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara --report experiments/mini_kimi_k3/reports/stage1-source-resolution.json --skip-probe`:
  passed and wrote `experiments/mini_kimi_k3/reports/stage1-source-resolution.json`
  with `status=ready`, `num_benchmarks=8`, `num_corpora=6`,
  `unresolved_benchmarks=[]`, and `unresolved_corpora=[]`.
- `TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && ./experiments/mini_kimi_k3/run.sh stage1-remote-readiness --report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json'`:
  exit `21`, report `status=blocked`. `checks.huggingface_dns.status=pass`,
  `checks.modal_cli.status=pass` with Modal client version `1.5.4`, and
  `checks.modal_auth.status=blocked` with detail `Token missing`.
- `TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && ./experiments/mini_kimi_k3/run.sh run-stage1-remote --source-root experiments/mini_kimi_k3/assets/mini-kimi-k3-vizuara --readiness-report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json --step dry_run --report experiments/mini_kimi_k3/reports/stage1-remote-dry-run.json'`:
  exit `21`, report `status=blocked`, detail
  `readiness report must be ready before remote Stage 1`, `step=dry_run`, and
  `argv=[]`. This confirms the wrapper fails closed before invoking Modal while
  the readiness report is blocked.
- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && ./experiments/mini_kimi_k3/run.sh completion-audit --preflight-report experiments/mini_kimi_k3/results/preflight.json --results-root experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id attempt-guarded-current-003 --report experiments/mini_kimi_k3/reports/completion-audit.json'`:
  exit `21`, report `status=blocked`. It fails
  `full_preflight_ready`, `trainable_config_passed`,
  `real_corpus_manifest_passed`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`; it passes
  `model_fidelity_passed` and `launch_evidence_bundle_passed`.
- Fresh 2026-08-22T04:38Z completion-audit update:
  `experiments/mini_kimi_k3/reports/completion-audit.json` now includes a
  machine-readable `next_actions` list derived from missing checklist items.
  The live report remains `status=blocked`; missing requirements are
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.
- Fresh 2026-08-22T04:51Z gate refresh:
  non-spendful gate reports were rerun through rootfs wrappers. GPU readiness
  remains blocked: all eight B200s have less than the required `12000` MiB free
  for the r1 Trainer smoke, with the best device at `9743` MiB free. The host
  `/dev/nvidia*` holder check still points to the `/workspace/monarch`
  GLM-5.2 SGLang service, primary PID `1958391`, plus scheduler, detokenizer,
  and compile-worker children. The corpus audit remains blocked at `4097` of
  `5000000000` required tokens, while disk readiness remains ready with
  `68154888192` free bytes versus `24000000000` required bytes. Stage 1 remote
  readiness remains blocked on missing Modal auth. Full preflight and
  completion audit therefore still exit `21` with `status=blocked`; no guarded
  full launch was attempted.
- Fresh 2026-08-22T04:54Z local-resource search:
  checked likely local corpus/cache locations without launching data prep:
  `${HOME}/datasets`, `${HOME}/hf_home/datasets`,
  repo-local Hugging Face caches, and `experiments/mini_kimi_k3/data`. No
  complete Mini-K3 shard export or source-input set was present. The largest
  repo-local Hugging Face cache is LiveCodeBench Arrow data, which is not the
  r1 Stage 1 corpus mix and does not satisfy the manifest provenance/tokenizer
  contract. No `.scratch/mini-kimi-k3-replication/.env` or Modal config was
  present in the checked repo/home paths, so the Stage 1 remote path remains
  credential-blocked.
- Fresh 2026-08-22T05:09Z route-aware gate hardening:
  full preflight and completion audit now treat `stage1_remote_readiness` as a
  required launch item only when the corpus route still depends on remote Stage
  1 acquisition. A complete local Stage 1 manifest (`local_stage1`) or imported
  first-party shard manifest (`source_sidecar` metadata) can mark remote
  readiness `not_required` instead of failing on missing Modal auth. The live
  manifest does not have local/imported route evidence and has only `4097`
  tokens, so the refreshed preflight still records
  `stage1_remote_readiness.status=blocked`, `trainable_config=fail`, and
  `real_corpus_manifest=fail`; no guarded full launch was attempted.
- Fresh 2026-08-22T05:14Z verification refresh:
  exact route-aware tests passed:
  `pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_treats_remote_readiness_as_optional_for_local_manifest tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness`
  returned `2 passed`. Focused gate tests passed:
  `pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "remote_readiness or manifest_with_corpus_metadata" tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit"`
  returned `3 passed`. `python -m py_compile` for `preflight.py` and
  `completion_audit.py` passed, and `git diff --check` passed. The refreshed
  live full preflight and completion audit still exit blocked because current
  artifacts lack GPU capacity, r1 optimizer-step evidence, the 5B-token corpus,
  and a completed guarded train stage.
- Fresh 2026-08-22T02:11Z refresh:
  `r1-smoke-gpu-readiness`, `stage1-remote-readiness`,
  `audit-r1-corpus-plan`, full `preflight`, `r1-training-smoke`, full
  `preflight` again, and `completion-audit` were rerun through the repo-local
  rootfs wrappers. The final completion audit still exits `21` with
  `status=blocked`.
- Fresh 2026-08-22T02:32Z code/report refresh:
  added tested GPU-holder diagnostics to the r1 smoke memory evidence path.
  Focused rootfs tests passed:
  `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "holder_evidence or finds_holders_from_proc or falls_back_when_fuser_is_missing or r1_smoke_gpu_readiness or r1_training_smoke"`
  returned `11 passed, 62 deselected`, and
  `pytest -q tests/unit_tests/test_mini_kimi_k3_training_smoke.py` returned
  `3 passed`. The regenerated r1 GPU-readiness and training-smoke reports
  still block before training; full preflight and completion audit still exit
  `21`.
- Read-only local artifact and credential search:
  `find experiments/mini_kimi_k3 ...` found only
  `experiments/mini_kimi_k3/data/tokens/fineweb-edu/part-00000.bin` plus the
  current manifest; no local 5B-token shard set, source input manifest, JSONL,
  or Parquet input set was present under the experiment tree. Checks for
  `$HOME/.modal.toml`, `$HOME/.modal/config.toml`, `$HOME/.config/modal.toml`,
  and `$HOME/.config/modal/config.toml` on the host and inside the rootfs all
  returned missing.
- `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1_remote"`:
  `4 passed, 64 deselected`
- `pytest -q tests/unit_tests/test_mini_kimi_k3_*.py`:
  `190 passed, 14 warnings`
- Fresh 2026-08-22T02:44Z broad regression refresh:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `195 passed, 14 warnings in 448.48s`.
- Fresh 2026-08-22T02:47Z gate refresh:
  `r1-smoke-gpu-readiness`, `stage1-remote-readiness`,
  `audit-r1-corpus-plan`, `r1-training-smoke`, full `preflight`, and
  `completion-audit` were rerun through repo-local rootfs wrappers. All launch
  gates still fail closed for the same hard blockers: no GPU has the required
  `12000` MiB free for the r1 Trainer smoke, the local corpus manifest still
  has `4097/5000000000` tokens, Modal auth is still missing, full preflight
  exits `21`, and completion audit exits `21`.
- Fresh 2026-08-22T02:47Z host GPU-holder refresh:
  `fuser -v /dev/nvidia*` still shows the `/workspace/monarch` GLM-5.2 SGLang
  service holding all eight B200 device nodes. The primary process remains
  PID `1958391`, with scheduler PIDs `1959989` through `1959996`,
  detokenizer PID `1959997`, and compile workers `1984726`, `1984727`,
  `1984728`, and `1984732` through `1984736`.
- Fresh 2026-08-22T02:49Z local unblocker search:
  read-only `find` checks under `${HOME}`, `/cache`, and the
  workspace found zero Modal config files in the checked home config locations.
  The only nearby corpus material was the tiny `fineweb_test` Arrow cache,
  FineWeb metadata/lock entries, and existing Mini-K3 fixture assets; no local
  5B-token shard set, source input manifest, JSONL corpus, or Parquet corpus was
  found that could satisfy the full corpus gate without new data acquisition.
- Fresh 2026-08-22T02:56Z Stage 1 remote readiness hardening:
  the readiness report now records redacted Modal auth context. The regenerated
  `experiments/mini_kimi_k3/reports/stage1-remote-readiness.json` still exits
  `21` with `status=blocked`; `MODAL_CONFIG_PATH`, `MODAL_PROFILE`,
  `MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET` are all `unset`, and the checked
  rootfs config paths `/project/home/.modal.toml`,
  `/project/home/.modal/config.toml`, `/project/home/.config/modal.toml`, and
  `/project/home/.config/modal/config.toml` all report `exists=false`.
  Focused rootfs tests passed:
  `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1_remote"`
  returned `4 passed, 69 deselected`, and
  `pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "stage1_remote_readiness"`
  returned `1 passed, 40 deselected, 14 warnings`.
- Fresh 2026-08-22T03:04Z broad regression refresh after the readiness-report
  hardening:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `195 passed, 14 warnings in 453.46s`.
- Fresh 2026-08-22T03:08Z Modal config-path readiness hardening:
  when `MODAL_CONFIG_PATH` is set, `stage1-remote-readiness` now records that
  explicit path in `checks.modal_auth.config_locations` with only existence
  metadata. Focused rootfs tests passed:
  `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1_remote"`
  returned `5 passed, 69 deselected`, and
  `pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "stage1_remote_readiness"`
  returned `1 passed, 40 deselected, 14 warnings`. The regenerated
  Stage 1 remote-readiness, full preflight, and completion-audit reports still
  exit `21`; no Modal credential exists in the current rootfs environment or
  checked config paths.
- Fresh 2026-08-22T03:16Z broad regression refresh after the
  `MODAL_CONFIG_PATH` evidence test:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `196 passed, 14 warnings in 448.05s`.
- Fresh 2026-08-22T03:26Z completion-audit checklist hardening:
  `completion-audit` now emits `prompt_to_artifact_checklist`, mapping the
  launch objective to concrete evidence files for full preflight, r1 Trainer
  optimizer-step evidence, model fidelity, real 5B-token corpus, launch
  evidence bundle, guarded launch completion, real training measurement, and
  train-stage success. The refreshed live report correctly marks only
  `model_fidelity` and `launch_evidence_bundle` as `covered`; all requirements
  that depend on GPU availability, real corpus data, or a completed train stage
  remain `missing`. Focused rootfs tests passed:
  `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit"`
  returned `2 passed, 72 deselected`. The broad Mini-K3 suite passed after this
  change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `196 passed, 14 warnings in 452.40s`.
- Fresh 2026-08-22T03:28Z operator handoff wizard:
  `.scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh` now walks a
  human through the remaining manual unblock sequence: refresh non-spendful
  gates, free/confirm GPU capacity, configure Modal credentials if using the
  remote Stage 1 route, choose/import or ingest the corpus route, rerun full
  preflight, and only then optionally start the guarded launch. The wizard is
  executable, `bash -n` passes, `shellcheck` passes, and it contains no
  non-ASCII characters.
- Fresh 2026-08-22T03:32Z operator handoff wizard correction:
  the wizard's initial inspection stage now refreshes full preflight before
  running completion audit, so the completion report does not read stale gate
  evidence after refreshed GPU, Modal, or corpus checks. `bash -n`,
  `shellcheck`, and `git diff --check` all pass after this correction.
- Fresh 2026-08-22T03:36Z operator handoff wizard secret-handling correction:
  the wizard now defaults `ENV_FILE` to
  `.scratch/mini-kimi-k3-replication/.env` instead of repo-root `.env`, while
  preserving an explicit `ENV_FILE=...` override. `git check-ignore -v` confirms
  that scratch-local env file is ignored by the repo `.gitignore`; `bash -n`,
  `shellcheck`, and `git diff --check` all pass after this correction.
- Fresh 2026-08-22T03:42Z corpus disk-readiness precheck:
  added `experiments/mini_kimi_k3/corpus_disk_readiness.py` and wired
  `experiments/mini_kimi_k3/run.sh corpus-disk-readiness`. The report computes
  `required_raw_bytes = target_tokens * 4`, applies a configurable overhead
  fraction, measures the nearest existing filesystem for the target path, and
  exits `21` unless free space covers the requirement. Focused rootfs tests
  passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "corpus_disk_readiness"'`
  returned `2 passed, 74 deselected`.
  The live command
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && ./experiments/mini_kimi_k3/run.sh corpus-disk-readiness --corpus-plan experiments/mini_kimi_k3/reports/r1-corpus-plan.json --target-path experiments/mini_kimi_k3/data/tokens --report experiments/mini_kimi_k3/reports/corpus-disk-readiness.json --overhead-fraction 0.20'`
  returned `status=ready`, `free_bytes=68658130944`,
  `required_bytes=24000000000`, and `deficit_bytes=0`. This removes the local
  disk-capacity unknown for raw shards, but it does not change the corpus
  deficit: actual 5B-token shards are still missing.
  The broad Mini-K3 regression suite passed after this change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `198 passed, 14 warnings in 454.36s`.
- Fresh 2026-08-22T03:53Z operator wizard disk-readiness wiring:
  `.scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh` now defines
  `CORPUS_DISK_READINESS_REPORT`, refreshes `corpus-disk-readiness` during the
  initial non-spendful gate inspection, and prints the same command after
  corpus import/build. Focused rootfs regression passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "unblock_wizard_refreshes_corpus_disk_readiness"'`
  returned `1 passed, 76 deselected`.
- Fresh 2026-08-22T04:00Z preflight and launch disk-readiness wiring:
  full preflight and guarded launch now accept
  `--corpus-disk-readiness-report` and embed the report under
  `real_corpus_manifest.evidence.corpus_disk_readiness`. Focused rootfs tests
  passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "corpus_disk_readiness" tests/unit_tests/test_mini_kimi_k3_runner.py -k "launch_passes_forward_oracle_report or unblock_wizard_refreshes_corpus_disk_readiness"'`
  returned `2 passed, 117 deselected`.
  Refreshed full preflight with the disk-readiness report still exits `21`;
  the corpus gate now records `corpus_disk_readiness.status=ready`,
  `required_bytes=24000000000`, `free_bytes=68658130944`, and
  `deficit_bytes=0`, but continues to fail because the manifest has only
  `4097/5000000000` tokens and the Stage 1 remote readiness report still has
  `modal_auth=blocked`. Refreshed completion audit still exits `21` with
  `status=blocked`.
  Final focused verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k "corpus_disk_readiness" tests/unit_tests/test_mini_kimi_k3_runner.py -k "corpus_disk_readiness or corpus_plan or launch_passes_forward_oracle_report or unblock_wizard_refreshes_corpus_disk_readiness" && python -m py_compile experiments/mini_kimi_k3/preflight.py experiments/mini_kimi_k3/launch.py experiments/mini_kimi_k3/corpus_disk_readiness.py'`
  plus `bash -n experiments/mini_kimi_k3/run.sh`,
  `bash -n .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh`,
  `shellcheck .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh`,
  and `git diff --check` returned clean. The broad Mini-K3 regression suite
  passed after this integration:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `200 passed, 14 warnings in 459.12s`.
- Fresh 2026-08-22T04:12Z completion-audit disk checklist:
  `experiments/mini_kimi_k3/completion_audit.py` now includes
  `raw_shard_disk_capacity` in `prompt_to_artifact_checklist`. It is covered
  only when the full preflight report contains
  `real_corpus_manifest.evidence.corpus_disk_readiness.status=ready` with
  `deficit_bytes=0`. The refreshed live completion audit still exits `21` with
  `status=blocked`; its checklist now marks `raw_shard_disk_capacity` as
  `covered`, while `full_preflight_ready`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded` remain `missing`.
  Focused rootfs regression and syntax checks passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit" && python -m py_compile experiments/mini_kimi_k3/completion_audit.py experiments/mini_kimi_k3/preflight.py experiments/mini_kimi_k3/launch.py'`
  plus `bash -n experiments/mini_kimi_k3/run.sh`,
  `bash -n .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh`,
  `shellcheck .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh`,
  and `git diff --check` returned clean. The broad Mini-K3 suite passed after
  this completion-audit checklist change:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `200 passed, 14 warnings in 461.57s`.
- Fresh 2026-08-22T04:24Z completion-audit blocker breakdown:
  `experiments/mini_kimi_k3/completion_audit.py` now also includes
  `r1_smoke_gpu_capacity` and `stage1_remote_readiness` in
  `prompt_to_artifact_checklist`, using the nested full-preflight evidence under
  `trainable_config.evidence.r1_training_smoke.gpu_memory` and
  `real_corpus_manifest.evidence.stage1_remote_readiness`. The refreshed live
  completion audit still exits `21` with `status=blocked`; the checklist now
  marks `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `stage1_remote_readiness`, `real_5b_corpus_manifest`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded` as `missing`, while model fidelity, raw shard disk
  capacity, and launch evidence bundle are `covered`.
  Focused rootfs regression and syntax checks passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "completion_audit" && python -m py_compile experiments/mini_kimi_k3/completion_audit.py'`
  plus `bash -n .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh`
  and `git diff --check` returned clean. The broad Mini-K3 suite passed after
  this checklist refinement:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'`
  returned `200 passed, 14 warnings in 454.40s`.
- `.scratch/mini-kimi-k3-replication/stage1-remote-readiness-2026-08-22.md`:
  records the non-ingesting Stage 1 readiness checks. It documents that
  `TORCHTITAN_ROOTFS_NETWORK=networked` is required for Hugging Face metadata
  access, `uv tool run --from modal modal` provides a working isolated Modal
  CLI, and Modal authentication is currently missing in the rootfs.
- Full preflight now accepts
  `--stage1-source-resolution-report experiments/mini_kimi_k3/reports/stage1-source-resolution.json`
  and embeds the validated source-resolution summary under
  `real_corpus_manifest.evidence.stage1_source_resolution`. The latest
  embedded status is `ready` with zero unresolved benchmarks and zero unresolved
  corpora.
- Full preflight now also accepts
  `--stage1-remote-readiness-report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json`
  and embeds the validated remote-readiness summary under
  `real_corpus_manifest.evidence.stage1_remote_readiness`. The latest embedded
  status is `blocked`, with `huggingface_dns=pass`, `modal_cli=pass`, and
  `modal_auth=blocked`.
- `completion-audit` now verifies the objective directly from the full
  preflight report, immutable attempt outcome, and coordinator event stream. It
  blocks unless full preflight is ready, required gates pass, the guarded launch
  completed, the training measurement is real, and a train stage succeeded.

Current full preflight command:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && ./experiments/mini_kimi_k3/run.sh preflight --mode full --token-manifest experiments/mini_kimi_k3/data/manifest.json --tokens-dir experiments/mini_kimi_k3/data/tokens --evidence-results-root experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id attempt-guarded-current-003 --forward-oracle-report .cache/trae-mini-k3-trainer/forward_oracle_r1_candidate_cuda.json --launch-backend-report .cache/trae-mini-k3-trainer/launch_backend_r1_candidate_cuda.json --r1-training-smoke-report experiments/mini_kimi_k3/results/r1_training_smoke.json --r1-corpus-plan-audit-report experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json --stage1-source-resolution-report experiments/mini_kimi_k3/reports/stage1-source-resolution.json --stage1-remote-readiness-report experiments/mini_kimi_k3/reports/stage1-remote-readiness.json'
```

Latest result: exit `21`; report written to
`experiments/mini_kimi_k3/results/preflight.json`, with matching
attempt-local evidence under
`experiments/mini_kimi_k3/results/runs/mini-kimi-k3-r1-local/attempt-guarded-current-003/derived/preflight.json`.

Gate summary from that report:

- `rootfs_runner`: `pass`
- `token_shard_loader`: `pass`
- `trainable_config`: `fail`
- `model_fidelity`: `pass`
- `real_corpus_manifest`: `fail`
- `tiny_training_smoke`: `not_applicable`
- `launch_evidence_bundle`: `pass`

## Requirement Checklist

- Full-mode preflight ready: not met. Current report status is `blocked`.
- r1 Trainer smoke with optimizer-step evidence: not met. Current
  `trainable_config` detail is `insufficient free GPU memory for r1 Trainer
  smoke`. The linked smoke evidence reports `status=blocked`, `steps=0`, and
  `max_grad_norm=null`.
- r1 model fidelity: met for the current launch-backend review scope. Current
  `model_fidelity` is `pass` using the forward-oracle and launch-backend
  reports under `.cache/trae-mini-k3-trainer/`.
- Real 5B-token corpus manifest: not met. Current `real_corpus_manifest` detail
  is `token manifest has fewer tokens than the r1 target (4097 < 5000000000)`.
  The r1 corpus-plan audit also reports `available_tokens=4097` and
  `deficit_tokens=4999995903`; full preflight now embeds that audit under
  `real_corpus_manifest.evidence.r1_corpus_plan_audit`. Per source,
  `fineweb-edu` is short by `2659995903` tokens, while `web-diverse`,
  `code-python`, `finemath`, `open-web-math`, and `cosmopedia` are missing
  entirely. The only registered shard is `fineweb-edu/part-00000.bin` with
  `4097` tokens. Full preflight also embeds
  `real_corpus_manifest.evidence.stage1_remote_readiness.status=blocked`, with
  `huggingface_dns=pass`, `modal_cli=pass`, and `modal_auth=blocked`.
- Guarded full launch train stage: not met. The latest guarded launch attempt
  under
  `experiments/mini_kimi_k3/results/runs/mini-kimi-k3-r1-local/attempt-guarded-current-003/`
  is blocked before training and has no train-stage artifact. Its
  `outcome.json` records `execution_outcome=blocked`,
  `evaluations.preflight.execution_outcome=blocked`,
  `evaluations.preflight.measurement=not_run`, and only one stage invocation:
  the preflight invocation.
- The newest source-resolution and remote-readiness-aware guarded launch attempt
  under
  `experiments/mini_kimi_k3/results/runs/mini-kimi-k3-r1-local/attempt-guarded-current-003/`
  is also blocked before training and has no train-stage artifact. Its derived
  preflight embeds
  `real_corpus_manifest.evidence.stage1_source_resolution.status=ready` and
  `real_corpus_manifest.evidence.stage1_remote_readiness.status=blocked`.

## Current Blockers

1. GPU memory for r1 Trainer smoke.

   Current `r1-smoke-gpu-readiness` shows all B200 devices heavily occupied.
   The best free device still has `9745` MiB free, while the default r1 smoke
   precheck requires `12000` MiB and lower-threshold diagnostic probes have
   already OOMed. Plain `nvidia-smi` reports high resident memory on every B200
   but no running processes in its process table. `fuser -v /dev/nvidia*`
   shows the handles are still held by user-owned `python`, `sglang::scheduler`,
   and `sglang::detoken` processes across all devices. The in-rootfs readiness
   report records that holder PIDs are not visible from the rootfs process
   namespace. The readiness report is
   `experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json`.

2. Real 5B-token corpus.

   The manifest and loader are usable, and local Stage 1 preparation tooling now
   exists for already-local JSONL/Parquet/text inputs. A rootfs-local
   `decontaminate-local` command can filter already-local text against
   explicitly supplied benchmark text files using the first-party 13-gram
   algorithm, then write a scoped decontamination report for
   `prepare-stage1-local`. Full preflight now rejects that local-explicit scope
   for full launch unless a report declares `launch_grade_benchmark_suite`.
   `prepare-stage1-local` also accepts a
   `mini_kimi_k3_source_input_manifest` that lists local source files and
   expected SHA-256 values, so large local source prep can be audited from a
   stable input manifest instead of a long command line. The
   `write-stage1-input-manifest` command creates that manifest from explicit
   local files and refuses to overwrite an existing manifest. The companion
   `inspect-stage1-inputs` command validates that manifest and writes
   per-file/total document and token counts without building shards, which is a
   cheap dry-run gate before spending time on a large local build. It also
   accepts `--min-total-tokens 5000000000` so a local source-input dry run can
   fail closed before shard generation when the input list cannot meet the r1
   token target. `plan-r1-corpus` scales the first-party 55B-token
   `mix_plan.json` to the r1 5B-token launch target and wrote
   `experiments/mini_kimi_k3/reports/r1-corpus-plan.json` with these source
   budgets: `fineweb-edu=2660000000`, `web-diverse=1140000000`,
   `code-python=612500000`, `finemath=262500000`,
   `open-web-math=175000000`, and `cosmopedia=150000000`.
   `audit-r1-corpus-plan` now compares that source-level plan to the current
   manifest, recomputes shard token counts from bytes, rejects stale
   `shard_metadata`, and writes
   `experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json` as the current
   acquisition deficit checklist. The latest audit is blocked with `4097`
   available tokens and `4999995903` missing tokens. A rootfs-local
   Hugging Face metadata probe requires `TORCHTITAN_ROOTFS_NETWORK=networked`;
   the default offline rootfs has an empty `/etc/resolv.conf` and cannot
   resolve `huggingface.co`. With networked mode, `data/probe.py` resolves all
   benchmark and corpus sources, but this only proves metadata/access
   resolution. It does not download data, build the decontamination index,
   tokenize shards, or satisfy the full corpus gate. `modal` is not installed
   in the system rootfs environment, and `uv pip install modal` currently fails
   because repo `.venv/bin/python3` is a stale symlink. `uv tool run --from
   modal modal ...` works in the networked rootfs and is the least-invasive
   Modal CLI path verified so far, but no remote Modal function has been
   started. A repo-local `stage1-remote-readiness` wrapper now writes a stable
   readiness report, and full preflight validates and embeds it. The current
   report is blocked on missing Modal authentication. A repo-local
   `run-stage1-remote` wrapper now allows only reviewed Stage 1 steps and
   requires a ready remote-readiness report before invoking Modal; `ingest_all`
   additionally requires `--allow-spend`. The current dry-run wrapper report is
   blocked with `argv=[]`, proving it did not call Modal while auth is missing.
   A repo-local `probe-stage1-sources` wrapper now writes a stable
   source-resolution report from the first-party probe output, and full
   preflight validates and embeds that report. Future corpus preparation can
   depend on
   `experiments/mini_kimi_k3/reports/stage1-source-resolution.json` and
   `experiments/mini_kimi_k3/reports/stage1-remote-readiness.json` instead of
   the ignored first-party `data/resolved.json` directly.
   This still does not resolve the remote benchmark suite, download corpora, run
   Modal, or create the required 5B-token manifest. The current local corpus is
   still a fixture-sized `4097` tokens. First-party Stage 1 generation is
   Modal/HF-secret based and should not be run without explicit authority and
   credentials.

## Next Concrete Actions

- Re-run `experiments/mini_kimi_k3/run.sh r1-training-smoke` only after at least
  one suitable GPU has enough free memory, or after the r1 smoke memory profile
  is intentionally reduced and reviewed as still representative of the
  Trainer-path gate. The current occupying workload appears to be the
  `/workspace/monarch` GLM-5.2 SGLang server; stopping or reconfiguring it
  requires explicit operator authorization.
- Populate the manifest with real Kimi-tokenized Stage 1 shards totaling at
  least 5B tokens, either by importing a reviewed first-party export with
  `import-shards` or by using `decontaminate-local` plus
  `prepare-stage1-local` on already-local source files and reviewed benchmark
  coverage.
- Configure Modal credentials inside the rootfs if the first-party remote Stage
  1 path is the intended corpus route, then rerun `stage1-remote-readiness`.
  Do not run `ingest_all` without explicit spend authorization.
- Re-run full preflight. Only if it returns `status=ready`, run the guarded
  `launch` command for the full attempt.

## 2026-08-22T05:26Z Non-GPU Progress

- Added optional env-file loading for the non-ingesting remote Stage 1 path.
  `stage1-remote-readiness` and `run-stage1-remote` now accept
  `--env-file .scratch/mini-kimi-k3-replication/.env`, pass the loaded values
  only through the subprocess environment, and record only the env-file path,
  load status, and key names in JSON reports.
- Updated the operator wizard to use `--env-file` when present instead of
  asking the operator to source secrets in the shell. The initial inspection
  stage still runs when the ignored env file is absent.
- Verified the env-file slice without GPU work or remote ingestion:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "stage1_remote" && python -m py_compile
  experiments/mini_kimi_k3/env_file.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/run_stage1_remote.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `7 passed, 82 deselected`.
- Refreshed completion audit only, without launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`.
- Current missing completion requirements are unchanged in substance:
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T05:29Z Secret-Redaction Hardening

- Hardened the env-file path so command output tails redact raw values from
  sensitive environment keys such as `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET`, even if a subprocess echoes the value without a
  `token=` or `secret=` label.
- Verified the redaction behavior with failing-then-passing tests that inject
  raw token values into fake Modal stdout for both `stage1-remote-readiness`
  and `run-stage1-remote`.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "stage1_remote" && python -m py_compile experiments/mini_kimi_k3/env_file.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/run_stage1_remote.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `7 passed, 82 deselected`.
- Executor-seam verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_execution_lifecycle_run_attempt.py
  tests/unit_tests/test_execution_lifecycle_done_scenarios.py'` passed with
  `9 passed`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`.

## 2026-08-22T05:34Z Local Corpus Input Hardening

- Hardened the local Stage 1 corpus path against duplicate input files. The
  `write-stage1-input-manifest`, `inspect-stage1-inputs`, and
  `prepare-stage1-local` paths now reject duplicate resolved file paths instead
  of allowing duplicated token counting toward corpus-readiness evidence.
- Verified the behavior with failing-then-passing tests for duplicate inputs in
  direct manifest creation, manifest inspection, and local shard preparation.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1 or
  corpus_plan" && python -m py_compile experiments/mini_kimi_k3/env_file.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `29 passed, 54 deselected`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Missing requirements are still
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T06:16Z Local Stage 1 Input-Inspection Gate

- Promoted `inspect-stage1-inputs` output to first-class readiness evidence for
  local Stage 1 routes. Full preflight now accepts
  `--stage1-input-inspection-report`, records
  `requires_stage1_input_inspection`, and rejects a local Stage 1 corpus route
  unless the inspection report passes and covers the r1 target token count.
- Wired `launch` to forward `--stage1-input-inspection-report` into full
  preflight before any train stage can run.
- Updated `completion-audit` to include a
  `local_stage1_input_inspection` checklist item and next action. The check is
  covered when local input inspection is not required, and blocking when a local
  Stage 1 route lacks passing inspection evidence.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_requires_input_inspection_for_local_stage1
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_accepts_input_inspection_for_local_stage1
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_failing_stage1_input_inspection_report
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_treats_remote_readiness_as_optional_for_local_manifest
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_launch_runs_train_stage_after_ready_preflight'`
  passed with `5 passed, 14 warnings`.
- Focused completion-audit verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit"'` passed with `3 passed, 80 deselected`.
- No GPU smoke, training, full launch, or Modal ingest was run in this slice.

## 2026-08-22T06:23Z Completion Audit Corpus Evidence Breakdown

- Added separate completion-audit checklist items for `r1_corpus_plan_audit`
  and `stage1_source_resolution`, in addition to the existing real-manifest,
  disk-capacity, local-input-inspection, and remote-readiness checks.
- This keeps the full-launch audit from hiding non-GPU corpus acquisition
  prerequisites behind the broad `real_5b_corpus_manifest` failure. A launch
  remains blocked unless the r1 corpus-plan audit is ready with zero token
  deficit and Stage 1 source resolution is ready with no unresolved benchmark
  or corpus sources.
- Verified the new behavior with a red-green completion-audit test cycle:
  before implementation,
  `pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit"` failed because the checklist lacked
  `r1_corpus_plan_audit`; after implementation it passed with
  `3 passed, 80 deselected`.

## 2026-08-22T06:29Z Operator Command Path Refresh

- Updated the full-launch operator wizard so the refresh-preflight and guarded
  launch command templates forward
  `--stage1-input-inspection-report $STAGE1_INPUT_INSPECTION_REPORT`.
- Added a wizard regression test so the command path cannot silently drift from
  the preflight and launch CLIs again.
- Updated the README preflight synopsis with
  `--stage1-input-inspection-report`.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "local_stage1 or input_inspection or remote_readiness or source_resolution or
  corpus_plan or manifest_with_corpus_metadata"
  tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or unblock_wizard or
  launch_runs_train_stage_after_ready_preflight or stage1_input or stage1 or
  corpus_plan" && python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `44 passed, 93 deselected, 14 warnings`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Covered requirements are `model_fidelity`,
  `raw_shard_disk_capacity`, `local_stage1_input_inspection`,
  `stage1_source_resolution`, and `launch_evidence_bundle`. Missing
  requirements are `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T06:33Z Wizard Non-GPU Inspection Split

- Tightened the operator wizard so Stage 1, "Inspect Current Gates", refreshes
  only non-GPU/non-spendful reports. The `r1-smoke-gpu-readiness` probe now
  appears only in the explicit "GPU Capacity" stage.
- Added a regression test proving the first wizard stage does not contain
  `r1-smoke-gpu-readiness` while the GPU stage still shows the operator command.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "local_stage1 or input_inspection or remote_readiness or source_resolution or
  corpus_plan or manifest_with_corpus_metadata"
  tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or unblock_wizard or
  launch_runs_train_stage_after_ready_preflight or stage1_input or stage1 or
  corpus_plan" && python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `45 passed, 93 deselected, 14 warnings`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Covered requirements are unchanged:
  `model_fidelity`, `raw_shard_disk_capacity`,
  `local_stage1_input_inspection`, `stage1_source_resolution`, and
  `launch_evidence_bundle`. Missing requirements are unchanged:
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T06:38Z Optional Local Input-Inspection Wizard Argument

- Tightened the operator wizard so
  `--stage1-input-inspection-report $STAGE1_INPUT_INSPECTION_REPORT` is
  included in generated preflight and launch commands only after the inspection
  report exists. This avoids making remote or imported corpus routes fail on a
  missing optional local-input report while preserving the explicit local route
  check once the report has been produced.
- Added a regression test for the conditional `stage1_input_inspection_arg`
  helper in `.scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh`.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "local_stage1 or input_inspection or remote_readiness or source_resolution or
  corpus_plan or manifest_with_corpus_metadata"
  tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or unblock_wizard or
  launch_runs_train_stage_after_ready_preflight or stage1_input or stage1 or
  corpus_plan" && python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `46 passed, 93 deselected, 14 warnings`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Covered requirements remain `model_fidelity`,
  `raw_shard_disk_capacity`, `local_stage1_input_inspection`,
  `stage1_source_resolution`, and `launch_evidence_bundle`. Missing
  requirements remain `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T06:50Z Full Preflight Requires Corpus Planning Evidence

- Tightened full-mode preflight so `real_corpus_manifest` cannot pass without
  an r1 corpus-plan audit report and Stage 1 source-resolution report. This
  aligns preflight with the completion-audit checklist and prevents a future
  full launch from greenlighting a corpus manifest whose acquisition plan or
  source resolution was never recorded.
- Added red-green regression tests for missing
  `--r1-corpus-plan-audit-report` and missing
  `--stage1-source-resolution-report` on an otherwise full-token-budget
  manifest fixture.
- Updated the README to describe these two reports as required for full-mode
  preflight, not merely optional embedded evidence.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "local_stage1 or input_inspection or remote_readiness or source_resolution or
  corpus_plan or manifest_with_corpus_metadata"
  tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or unblock_wizard or
  launch_runs_train_stage_after_ready_preflight or stage1_input or stage1 or
  corpus_plan"'` passed with `48 passed, 93 deselected, 14 warnings`.
- Static verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Covered requirements remain `model_fidelity`,
  `raw_shard_disk_capacity`, `local_stage1_input_inspection`,
  `stage1_source_resolution`, and `launch_evidence_bundle`. Missing
  requirements remain `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T05:49Z Imported Sidecar Hash Contract

- Strengthened imported Stage 1 shard provenance so `import-shards` records
  `source_sidecar_sha256` alongside each `source_sidecar` path. Full preflight
  now requires and verifies that hash before accepting imported sidecar
  evidence.
- Added regression tests showing the importer writes sidecar hashes and full
  preflight rejects missing or mismatched sidecar hashes.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "source_sidecar or registered_shard_metadata or manifest_with_corpus_metadata
  or remote_readiness" tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "import_shards or stage1 or corpus_plan" && python -m py_compile
  experiments/mini_kimi_k3/import_shards.py experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/env_file.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `34 passed, 96 deselected, 14 warnings`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`, with the same missing launch-completion requirements.

## 2026-08-22T05:36Z Duplicate Source Input Guard

- Hardened local Stage 1 input accounting so the same resolved local input file
  cannot be listed twice and counted twice. This now applies to
  `write-stage1-input-manifest`, `inspect-stage1-inputs`, and
  `prepare-stage1-local`, including mixed manifest-plus-direct-input paths.
- Verified with failing-then-passing regression tests for duplicate input
  rejection in all three paths.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k "stage1 or
  corpus_plan" && python -m py_compile experiments/mini_kimi_k3/env_file.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `29 passed, 54 deselected`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`, with the same missing launch-completion requirements.

## 2026-08-22T05:41Z Imported Sidecar Validation

- Hardened the imported-corpus route used to make remote Stage 1 readiness
  optional. Full preflight now validates each imported shard's
  `source_sidecar` when present: the sidecar JSON must be readable, declare
  the same source, declare `dtype=uint32`, and match the shard token count.
- Added regression tests showing full preflight rejects missing and stale
  imported sidecars instead of accepting them as enough evidence for a
  local/imported corpus route.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "source_sidecar or registered_shard_metadata or manifest_with_corpus_metadata
  or remote_readiness" tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "stage1 or corpus_plan" && python -m py_compile
  experiments/mini_kimi_k3/env_file.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/preflight.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed with `33 passed, 95 deselected, 14 warnings`.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Missing requirements are still
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T06:55Z Full Preflight Requires Corpus Reports

- Hardened full-mode preflight so `real_corpus_manifest` cannot pass without
  explicit `r1_corpus_plan_audit` and `stage1_source_resolution` report
  evidence. This keeps the full launch gate aligned with the completion-audit
  checklist and prevents a manifest-only route from hiding missing corpus
  planning or source-resolution evidence.
- Added regression coverage for missing corpus-plan and source-resolution
  reports, and updated the local Stage 1 positive fixture to supply both
  reports before expecting the corpus manifest gate to pass.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py -k
  "local_stage1 or input_inspection or remote_readiness or source_resolution or
  corpus_plan or manifest_with_corpus_metadata"
  tests/unit_tests/test_mini_kimi_k3_runner.py -k
  "completion_audit or unblock_wizard or
  launch_runs_train_stage_after_ready_preflight or stage1_input or stage1 or
  corpus_plan"'` passed with `48 passed, 93 deselected, 14 warnings`.
- Static verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Missing requirements are
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`. Covered requirements are `model_fidelity`,
  `raw_shard_disk_capacity`, `local_stage1_input_inspection`,
  `stage1_source_resolution`, and `launch_evidence_bundle`.

## 2026-08-22T07:25Z Full Mini-K3 Unit Sweep

- Repaired legacy full-corpus positive preflight fixtures so they provide the
  mandatory r1 corpus-plan audit and Stage 1 source-resolution reports.
  The production full-mode gate remains strict.
- Broader Mini-K3 verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `222 passed, 14 warnings`.
- Static verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Missing requirements are
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`. Covered requirements are `model_fidelity`,
  `raw_shard_disk_capacity`, `local_stage1_input_inspection`,
  `stage1_source_resolution`, and `launch_evidence_bundle`.

## 2026-08-22T07:49Z Route-Aware Completion Checklist

- Adjusted the completion audit checklist so route-specific gates that are
  explicitly not required remain visible as `not_required`, rather than being
  collapsed into `covered`. This keeps the prompt-to-artifact audit from
  overstating evidence when the active corpus route is remote or local/imported.
- Repaired the full-corpus preflight fixtures that were still missing the
  mandatory r1 corpus-plan audit and Stage 1 source-resolution reports.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `2 passed`.
- Broader Mini-K3 verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `222 passed, 14 warnings`.
- Static verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.
- Refreshed completion audit only, without GPU work or launching training:
  `experiments/mini_kimi_k3/reports/completion-audit.json` remains
  `status=blocked`. Missing requirements are
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`. The remote-route-only local input inspection gate is
  now listed as `not_required`; covered requirements are `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`.

## 2026-08-22T08:05Z Fresh Blocked Stage 1 Readiness

- Hardened `stage1-remote-readiness` so an invalid `--env-file` path writes a
  fresh blocked readiness report instead of exiting before report creation. This
  prevents stale Modal readiness evidence from being reused during launch
  preparation.
- Added a regression test for the missing env-file path. The report now records
  `checks.env_file.status=blocked`, the missing path, and the rootfs marker.
- Non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_readiness_reports_missing_env_file
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader Mini-K3 verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `223 passed, 14 warnings`.
- Static verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check` passed.
- Refreshed Stage 1 remote readiness with
  `.scratch/mini-kimi-k3-replication/.env`; the report is
  `status=blocked` with `checks.env_file.status=blocked` because the env file
  does not exist.
- Refreshed full preflight and completion audit only, without GPU work or
  launching training. `experiments/mini_kimi_k3/reports/completion-audit.json`
  remains `status=blocked`. Missing requirements are `full_preflight_ready`,
  `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`. The remote-route
  local input inspection gate is `not_required`; covered requirements are
  `model_fidelity`, `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`.

## 2026-08-22T08:30Z Env-File Evidence Parser Refresh

- Refreshed non-GPU readiness artifacts only, honoring the pause on Mini-K3
  experiment execution. No GPU smoke, training, guarded launch, or Modal ingest
  was started.
- Static verification passed:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/preflight.py
  experiments/mini_kimi_k3/launch.py
  experiments/mini_kimi_k3/completion_audit.py
  experiments/mini_kimi_k3/stage1_remote_readiness.py
  experiments/mini_kimi_k3/inspect_stage1_inputs.py
  experiments/mini_kimi_k3/prepare_stage1_local.py
  experiments/mini_kimi_k3/write_stage1_input_manifest.py
  torchtitan/experiments/execution/executor.py' && bash -n
  .scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh && git diff
  --check`.
- Refreshed Stage 1 remote readiness with the missing
  `.scratch/mini-kimi-k3-replication/.env`; it exited `21` and wrote
  `experiments/mini_kimi_k3/reports/stage1-remote-readiness.json` with
  `status=blocked` and `checks.env_file.status=blocked`.
- Refreshed `experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json`;
  it remains `status=blocked` with `available_tokens=4097` and
  `deficit_tokens=4999995903`.
- Refreshed `experiments/mini_kimi_k3/reports/corpus-disk-readiness.json`;
  it remains `status=ready` with `66503897088` free bytes against
  `24000000000` required bytes.
- Refreshed full preflight. `experiments/mini_kimi_k3/results/preflight.json`
  remains `status=blocked`, and the real corpus gate now preserves the direct
  env-file blocker:
  `stage1 remote readiness env file is blocked (env file does not exist:
  .scratch/mini-kimi-k3-replication/.env)`.
- Refreshed completion audit. The prompt-to-artifact checklist still reports
  missing `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`; `local_stage1_input_inspection` is `not_required`,
  and `model_fidelity`, `raw_shard_disk_capacity`,
  `stage1_source_resolution`, and `launch_evidence_bundle` are covered.

## 2026-08-22T08:50Z Completion Audit Summary Fields

- Added top-level `missing`, `covered`, and `not_required` arrays to
  `experiments/mini_kimi_k3/reports/completion-audit.json`, derived from the
  prompt-to-artifact checklist. This keeps operator-facing status inspection
  machine-readable without requiring every consumer to re-group checklist rows.
- Verified the change test-first. The focused completion-audit tests first
  failed with missing `result["missing"]`, then passed after adding the
  summary helper:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_accepts_local_corpus_without_remote_readiness'`
  passed with `3 passed`.
- Broader Mini-K3 verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `224 passed, 14 warnings`.
- Refreshed completion audit only, without GPU work or launch. The live report
  remains `status=blocked`; `missing` contains `full_preflight_ready`,
  `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`. `covered`
  contains `model_fidelity`, `raw_shard_disk_capacity`,
  `stage1_source_resolution`, and `launch_evidence_bundle`;
  `not_required` contains `local_stage1_input_inspection`.

## 2026-08-22T09:03Z Completion Criteria In Report

- Added a top-level `success_criteria` list to the completion audit report so
  the launch objective is restated as concrete deliverables in the artifact
  itself. The list matches the prompt-to-artifact checklist order.
- Verified the behavior test-first. The focused tests first failed on missing
  `result["success_criteria"]`, then passed after wiring the field:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_blocked_preflight_and_attempt
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_passes_completed_full_launch'`
  passed with `2 passed`.
- Runner-level non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `87 passed`.
- Refreshed `experiments/mini_kimi_k3/reports/completion-audit.json` only.
  The report remains `status=blocked`; no GPU smoke, training, guarded launch,
  or Modal ingest was started.

## 2026-08-22T09:25Z Completion Audit CLI Missing Summary

- Added `argv` seams to `experiments.mini_kimi_k3.completion_audit` and made
  the blocked CLI path print a compact `missing:` line before the report path.
  This lets operators see the current launch blockers without opening JSON.
- Verified the behavior test-first. The new CLI test first failed because
  `main()` did not accept `argv`, then passed after the CLI seam and missing
  summary were added:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_cli_prints_missing_summary'`
  passed with `1 passed`.
- Runner-level non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `88 passed`.
- Refreshed completion audit only. The CLI now prints:
  `missing: full_preflight_ready, r1_smoke_gpu_capacity,
  r1_trainer_optimizer_step, real_5b_corpus_manifest,
  r1_corpus_plan_audit, stage1_remote_readiness, guarded_launch_completed,
  training_measurement_real, train_stage_succeeded`. The report remains
  `status=blocked`; no GPU smoke, training, guarded launch, or Modal ingest was
  started.

## 2026-08-22T09:38Z Modal Env Template

- Added `.scratch/mini-kimi-k3-replication/.env.example` as a safe local
  template for the missing remote Stage 1 Modal credentials. The template
  contains placeholder `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and
  `MODAL_PROFILE` values only; the real `.env` remains ignored by the root
  `.gitignore`.
- Updated `experiments/mini_kimi_k3/README.md` to point operators at the
  template when using `stage1-remote-readiness --env-file
  .scratch/mini-kimi-k3-replication/.env`.
- Verified the template is parseable by the same env-file loader used by the
  readiness command:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_env_template_is_parseable'`
  passed with `1 passed`.
- Runner-level non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `89 passed`.
- Refreshed completion audit only. It remains `status=blocked` with the same
  missing criteria; no GPU smoke, training, guarded launch, or Modal ingest was
  started.

## 2026-08-22T09:49Z Stage 1 Env-File Guidance In Reports

- Added explicit operator guidance to the blocked Stage 1 remote readiness
  report when `--env-file` points at a missing or invalid file. The
  `checks.env_file` object now includes the safe template path
  `.scratch/mini-kimi-k3-replication/.env.example` and required keys
  `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `MODAL_PROFILE`.
- Verified the behavior test-first. The missing-env-file test first failed
  because `template` and `required_keys` were absent, then passed after adding
  the fields:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_readiness_reports_missing_env_file
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_env_template_is_parseable
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_readiness_loads_modal_env_file'`
  passed with `3 passed`.
- Runner-level non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `89 passed`.
- Refreshed Stage 1 readiness, full preflight, and completion audit only. The
  readiness report remains blocked on the missing real `.env`; the completion
  audit remains `status=blocked` with unchanged missing launch criteria. No GPU
  smoke, training, guarded launch, or Modal ingest was started.

## 2026-08-22T10:01Z Preflight Preserves Env-File Guidance

- Updated full preflight's Stage 1 remote-readiness evidence so
  `real_corpus_manifest.evidence.stage1_remote_readiness.env_file` preserves the
  full env-file check object, including `template` and `required_keys`, instead
  of collapsing it to the string status.
- Verified the behavior test-first. The focused preflight test first failed
  because preflight stored only `"blocked"`, then passed after preserving the
  full object:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_reports_blocked_stage1_remote_env_file'`
  passed with `1 passed, 14 warnings`.
- Preflight-level non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py'` passed with
  `56 passed, 14 warnings`.
- Refreshed Stage 1 readiness, full preflight, and completion audit only. The
  refreshed preflight now embeds the full blocked env-file guidance object:
  `path=.scratch/mini-kimi-k3-replication/.env`,
  `template=.scratch/mini-kimi-k3-replication/.env.example`, and required keys
  `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_PROFILE`. The completion audit
  remains `status=blocked`; no GPU smoke, training, guarded launch, or Modal
  ingest was started.

## 2026-08-22T09:12Z Completion Audit README Sync

- Updated `experiments/mini_kimi_k3/README.md` so the operator-facing
  `completion_audit` section documents the top-level `success_criteria`,
  `missing`, `covered`, and `not_required` arrays.
- Verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python -m py_compile experiments/mini_kimi_k3/completion_audit.py' && git
  diff --check && rg -n "success_criteria|missing|covered|not_required"
  experiments/mini_kimi_k3/README.md
  experiments/mini_kimi_k3/reports/completion-audit.json` passed.

## 2026-08-22T17:33Z R1 Smoke Corpus-Identity Gate

- Stopped short of GPU smoke, guarded launch, full training, or Modal ingest per
  operator instruction; only non-GPU unit, static, and report-audit work was
  performed.
- Hardened full-mode `trainable_config` validation so an r1 training-smoke
  report cannot pass unless preflight was invoked with explicit requested
  `token_manifest` and `tokens_dir` paths. This keeps smoke evidence tied to
  the launch corpus identity instead of accepting a self-contained report in
  isolation.
- Updated preflight fixtures so positive r1 smoke evidence forwards the same
  corpus paths through the smoke report, command provenance, and
  `run_preflight()` call. Command-contract negative cases now also supply the
  requested corpus identity so their intended command mutations remain the
  failing condition.
- Verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_without_requested_corpus_paths
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_passes_with_r1_training_smoke_report'`
  passed with `2 passed, 14 warnings`.
- Focused regression cluster verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_preflight_trainable_config_rejects_r1_smoke_bad_command_contract'`
  passed with `9 passed, 14 warnings`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `255 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files ...` passed all
  file-quality hooks and failed only the expected `no-commit-to-branch` policy
  hook.
- Refreshed completion audit only:
  `experiments/mini_kimi_k3/run.sh completion-audit --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id
  attempt-guarded-current-003 --report
  experiments/mini_kimi_k3/reports/completion-audit.json` exited `21` with
  `status=blocked`. Current blockers remain `full_preflight_ready`,
  `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T17:37Z Non-GPU Launch-Readiness Stop Point

- Refreshed the full preflight report through the rootfs-aware command only:
  `experiments/mini_kimi_k3/run.sh preflight --mode full --report
  experiments/mini_kimi_k3/results/preflight.json --token-manifest
  experiments/mini_kimi_k3/data/manifest.json --tokens-dir
  experiments/mini_kimi_k3/data/tokens --evidence-results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local --attempt-id
  attempt-guarded-current-003 --forward-oracle-report
  .cache/trae-mini-k3-trainer/forward_oracle_r1_candidate_cuda.json
  --launch-backend-report
  .cache/trae-mini-k3-trainer/launch_backend_r1_candidate_cuda.json
  --r1-training-smoke-report
  experiments/mini_kimi_k3/results/r1_training_smoke.json
  --r1-corpus-plan-audit-report
  experiments/mini_kimi_k3/reports/r1-corpus-plan-audit.json
  --corpus-disk-readiness-report
  experiments/mini_kimi_k3/reports/corpus-disk-readiness.json
  --stage1-source-resolution-report
  experiments/mini_kimi_k3/reports/stage1-source-resolution.json
  --stage1-remote-readiness-report
  experiments/mini_kimi_k3/reports/stage1-remote-readiness.json`.
  It exited `21`, wrote `experiments/mini_kimi_k3/results/preflight.json`,
  and correctly reported `Mini Kimi K3 full launch is not ready`.
- Refreshed completion audit against that preflight. It exited `21` with
  `status=blocked`.
- Current covered completion criteria are `model_fidelity`,
  `raw_shard_disk_capacity`, `stage1_source_resolution`, and
  `launch_evidence_bundle`. The local Stage 1 input-inspection criterion is
  `not_required` for the current remote route.
- Current missing criteria remain `full_preflight_ready`,
  `r1_smoke_gpu_capacity`, `r1_trainer_optimizer_step`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.
- Evidence details from refreshed reports: the r1 manifest still contains only
  `4097` tokens against the `5000000000` token target; the corpus-plan audit
  reports a `4999995903` token deficit; Stage 1 source resolution is ready with
  8 benchmarks and 6 corpora resolved; remote readiness is blocked on the
  missing `.scratch/mini-kimi-k3-replication/.env` file with required keys
  `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `MODAL_PROFILE`.
- No GPU smoke, training, guarded launch, full launch, or Modal ingest was
  started.

## 2026-08-22T17:56Z Stage 1 Spend Guard Ordering

- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so spendful
  `ingest_all` requests fail on the missing `--allow-spend` authority before
  parsing an optional env file. This keeps the spend guard as the first
  observable failure, even when an operator supplies a stale or missing
  credential-file path.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_checks_spend_guard_before_env_file`.
  The test failed red because the missing env file was parsed first, then passed
  after moving the spend guard ahead of env-file loading.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_checks_spend_guard_before_env_file
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_requires_ingest_authorization
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_runs_safe_modal_step
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_loads_modal_env_file'`
  passed with `4 passed`.
- Runner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `103 passed`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `256 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T20:36Z Stage 1 Readiness Rootfs Provenance Alignment

- Continued under the active no-GPU/no-launch/no-spend constraint. No GPU
  smoke, training, guarded launch, full launch, or Modal ingestion was
  attempted.
- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so a Stage 1 remote
  step does not trust a readiness artifact unless the artifact records
  `rootfs.marker=1` and `rootfs.network_mode=networked`.
- Hardened `experiments/mini_kimi_k3/preflight.py` with the same rootfs and
  networked-mode validation for Stage 1 remote-readiness reports, and preserved
  `rootfs_marker` in the reported evidence.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_ready_report_without_rootfs_marker`
  and
  `tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_ready_stage1_remote_without_rootfs_marker`.
  Both failed red before the validation patches.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_ready_report_without_rootfs_marker
  tests/unit_tests/test_mini_kimi_k3_runner.py -k stage1_remote'` passed with
  `17 passed, 92 deselected`.
- Focused preflight verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_preserves_ready_stage1_remote_env_file_evidence
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_ready_stage1_remote_without_rootfs_marker
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_reports_blocked_stage1_remote_env_file
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_blocked_stage1_remote_readiness'`
  passed with `4 passed, 14 warnings`.
- Combined runner/preflight verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py'` passed with
  `182 passed, 14 warnings`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `263 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/preflight.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T20:21Z Stage 1 Readiness Rootfs Provenance In Preflight

- Continued under the active no-GPU/no-launch/no-spend constraint. No GPU
  smoke, training, guarded launch, full launch, or Modal ingestion was
  attempted.
- Hardened `experiments/mini_kimi_k3/preflight.py` so full preflight rejects a
  Stage 1 remote-readiness report that claims `status=ready` but does not record
  `rootfs.marker=1` and `rootfs.network_mode=networked`.
- Added
  `tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_ready_stage1_remote_without_rootfs_marker`.
  The test failed red because the report was not rejected for missing rootfs
  marker evidence, then passed after `_stage1_remote_readiness_evidence()`
  validated rootfs provenance.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_ready_stage1_remote_without_rootfs_marker
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_preserves_ready_stage1_remote_env_file_evidence
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_reports_blocked_stage1_remote_env_file
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_blocked_stage1_remote_readiness'`
  passed with `4 passed, 14 warnings`.
- Combined runner/preflight verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py'` passed with
  `182 passed, 14 warnings`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `263 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/preflight.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T19:53Z Stage 1 Readiness Rootfs Provenance

- Continued under the active no-GPU/no-launch/no-spend constraint. No GPU
  smoke, training, guarded launch, full launch, or Modal ingestion was
  attempted.
- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so a Stage 1 remote
  step does not trust a readiness artifact unless the artifact records
  `rootfs.marker=1` and `rootfs.network_mode=networked`. This prevents a
  copied or hand-authored top-level `status=ready` report from authorizing a
  Modal command without matching rootfs provenance.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_ready_report_without_rootfs_marker`.
  The test failed red because a ready report with no rootfs marker reached the
  executor, then passed after `_readiness_evidence()` validated rootfs
  provenance.
- Updated older Stage 1 remote runner fixtures to include the real readiness
  producer's `rootfs.marker=1` field when those tests are intended to exercise
  readiness status, command, or env-file branches rather than rootfs provenance.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_ready_report_without_rootfs_marker
  tests/unit_tests/test_mini_kimi_k3_runner.py -k stage1_remote'` passed with
  `17 passed, 92 deselected`.
- Runner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `108 passed`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `262 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/preflight.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T19:32Z Stage 1 Preflight Env-File Evidence

- Continued under the active no-GPU/no-launch/no-spend constraint. No GPU
  smoke, training, guarded launch, full launch, or Modal ingestion was
  attempted.
- Hardened `experiments/mini_kimi_k3/preflight.py` so full preflight preserves
  top-level `env_file` evidence from a ready Stage 1 remote-readiness report,
  matching the credential provenance now enforced by
  `run_stage1_remote.py`.
- Added
  `tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_preserves_ready_stage1_remote_env_file_evidence`.
  The test failed red with `KeyError: 'env_file'`, then passed after
  `_stage1_remote_readiness_evidence()` propagated top-level env-file evidence.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_preserves_ready_stage1_remote_env_file_evidence
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_reports_blocked_stage1_remote_env_file
  tests/unit_tests/test_mini_kimi_k3_preflight.py::test_full_preflight_rejects_blocked_stage1_remote_readiness'`
  passed with `3 passed, 14 warnings`.
- Preflight-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_preflight.py'` passed with
  `73 passed, 14 warnings`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `261 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  experiments/mini_kimi_k3/preflight.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  tests/unit_tests/test_mini_kimi_k3_preflight.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T19:07Z Stage 1 Required Readiness Checks

- Continued under the active no-GPU/no-launch/no-spend constraint. No GPU
  smoke, training, guarded launch, full launch, or Modal ingestion was
  attempted.
- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so a top-level
  `status=ready` Stage 1 remote-readiness report is not enough by itself. The
  runner now requires `checks` to be an object and requires
  `huggingface_dns`, `modal_cli`, and `modal_auth` to exist with
  `status=pass` before a safe Modal step can execute.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_ready_report_missing_required_check`.
  The test failed red because a forged ready report missing `modal_auth` still
  reached the executor, then passed after `_readiness_evidence()` validated the
  required checks.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_ready_report_missing_required_check
  tests/unit_tests/test_mini_kimi_k3_runner.py -k stage1_remote'` passed with
  `16 passed, 92 deselected`.
- Runner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `107 passed`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `260 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T18:36Z Stage 1 Readiness Env-File Provenance

- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so a ready Stage 1
  remote-readiness report only authorizes a remote step with the same env file
  path that was checked. A stale readiness report for a different credential
  file now writes a blocked `mini_kimi_k3_stage1_remote_step` report before any
  executor call.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_mismatched_readiness_env_file`.
  The test failed red because `dry_run` reached the executor and completed,
  then passed after readiness env-file evidence was propagated and validated.
- Added adjacent coverage in
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_missing_requested_env_file_after_readiness_check`
  for the case where readiness checked a credential file but the guarded
  remote step omits `--env-file` and would otherwise use ambient credentials.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_mismatched_readiness_env_file
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_missing_requested_env_file_after_readiness_check'`
  passed with `2 passed`.
- Stage 1 remote runner verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k stage1_remote'`
  passed with `14 passed, 92 deselected`.
- Runner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `106 passed`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `259 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T18:14Z Stage 1 Env-File Failure Artifact

- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so safe remote Stage
  1 steps write a blocked `mini_kimi_k3_stage1_remote_step` report when an
  optional env file is missing or invalid. This preserves machine-readable
  evidence for operator mistakes instead of failing before artifact creation.
- Preserved the previous spend guard ordering: spendful `ingest_all` still
  blocks on missing `--allow-spend` before env-file parsing.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_writes_report_for_missing_env_file`.
  The test failed red because the env-file parser raised before a report was
  written, then passed after `run_stage1_remote()` caught the env-file
  `ValueError` and wrote blocked report evidence.
- Focused non-GPU verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_writes_report_for_missing_env_file
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_checks_spend_guard_before_env_file
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_requires_ingest_authorization
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_runs_safe_modal_step
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_loads_modal_env_file'`
  passed with `5 passed`.
- Runner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `104 passed`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `257 passed, 14 warnings`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/run_stage1_remote.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all file-quality hooks and failed only the expected
  `no-commit-to-branch` policy hook.
- Refreshed completion audit only. It still exits `21` with `status=blocked`
  and the same missing launch criteria. No GPU smoke, training, guarded launch,
  full launch, or Modal ingest was started.

## 2026-08-22T22:30Z Completion Audit R1 Smoke Provenance

- Continued under the current operator boundary: no GPU smoke, no training, no
  guarded/full launch, and no Modal ingest.
- Hardened `experiments/mini_kimi_k3/completion_audit.py` so
  `trainable_config_passed` requires embedded r1 Trainer-smoke evidence with
  `status=pass`, `steps >= 1`, `rootfs.marker=1`,
  `rootfs.cwd=/workspace/torchtitan`, a successful command return code, command
  cwd `/workspace/torchtitan`, and `./run_train.sh` in the command argv. Missing
  smoke provenance is normalized into visible `rootfs` and `command`
  `status=missing` evidence.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_trainable_gate_without_smoke_provenance`.
  The test failed red because a forged r1-smoke summary with `steps=1` but no
  rootfs or command provenance made the completion audit complete; it passed
  after the stricter predicate.
- Focused completion-audit verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_completion_audit_rejects_trainable_gate_without_smoke_provenance
  tests/unit_tests/test_mini_kimi_k3_runner.py -k completion_audit'` passed
  with `18 passed, 95 deselected`.
- Mini-K3 owner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_*.py'` passed with
  `267 passed, 14 warnings` in `688.06s`.
- Static checks:
  `git diff --check` passed. Targeted `pre-commit run --files
  experiments/mini_kimi_k3/completion_audit.py
  tests/unit_tests/test_mini_kimi_k3_runner.py
  .scratch/mini-kimi-k3-replication/launch-completion-audit-2026-08-21.md`
  passed all content hooks and failed only the expected `no-commit-to-branch`
  policy hook.
- Completion audit refresh:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python experiments/mini_kimi_k3/completion_audit.py --preflight-report
  experiments/mini_kimi_k3/results/preflight.json --results-root
  experiments/mini_kimi_k3/results --run-id mini-kimi-k3-r1-local
  --attempt-id attempt-guarded-current-003'` exited `21` with
  `status=blocked`.
- Current launch blockers remain:
  `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T23:13Z Corpus Plan Summary Consistency Guard

- Continued under the current operator boundary: no GPU smoke, no training, no
  guarded/full launch, and no Modal ingest.
- Hardened `experiments/mini_kimi_k3/audit_r1_corpus_plan.py` so a corpus-plan
  artifact with a `source_tokens` summary must exactly match
  `sources[*].target_tokens`. A mismatched summary can no longer be audited as
  ready just because the detailed source targets add up.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_source_token_mismatch`.
  The test failed red because the audit reported `ready (4/4 tokens, deficit
  0)` despite `source_tokens.fineweb-edu=3` and
  `sources.fineweb-edu.target_tokens=4`; it passed after the stricter plan
  validation.
- Focused rootfs verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_source_token_mismatch
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_runner_audit_r1_corpus_plan_passes_when_manifest_covers_plan
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_runner_audit_r1_corpus_plan_rejects_inconsistent_plan_total
  tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_runner_audits_r1_corpus_plan_deficits'`
  passed with `4 passed`.
- Runner-suite verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `113 passed` in `125.38s`.
- Static check: `git diff --check` passed.
- Refreshed completion audit only. It exited `21` with `status=blocked`;
  remaining blockers are `full_preflight_ready`, `r1_smoke_gpu_capacity`,
  `r1_trainer_optimizer_step`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T23:28Z R1 Trainer Smoke Passed

- Resumed GPU work after operator confirmation that GPU execution was ready.
- Recorded current GPU readiness with:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  experiments/mini_kimi_k3/run.sh r1-smoke-gpu-readiness --report
  experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json
  --min-free-gpu-memory-mib 12000'`; it exited `0` with `status=pass`.
- Ran the bounded one-step r1 Trainer smoke with:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  experiments/mini_kimi_k3/run.sh r1-training-smoke --token-manifest
  experiments/mini_kimi_k3/data/manifest.json --tokens-dir
  experiments/mini_kimi_k3/data/tokens --report
  experiments/mini_kimi_k3/results/r1_training_smoke.json --seq-len 4096
  --steps 1 --local-batch-size 1 --global-batch-size 1
  --min-free-gpu-memory-mib 12000'`; it exited `0`.
- The r1 smoke report records `status=pass`, `rootfs.marker=1`,
  `rootfs.cwd=/workspace/torchtitan`, `CUDA_VISIBLE_DEVICES=0`,
  `command.return_code=0`, `optimization.steps=1`, and
  `optimization.max_grad_norm=0.0031`.
- Refreshed non-training corpus/readiness gates. The r1 corpus-plan audit still
  exits `21` with `4097/5000000000` tokens and a `4999995903` token deficit.
  Stage 1 remote readiness still exits `21` because
  `.scratch/mini-kimi-k3-replication/.env` is missing.
- Refreshed full preflight with the successful r1 smoke evidence. It exits
  `21` with `trainable_config=pass`, `model_fidelity=pass`,
  `launch_evidence_bundle=pass`, and `real_corpus_manifest=fail`.
- Refreshed completion audit. It exits `21` with `status=blocked`; remaining
  blockers are `full_preflight_ready`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T23:51Z Stage 1 Remote Rootfs Check Consumption

- Hardened `experiments/mini_kimi_k3/stage1_remote_readiness.py` so a remote
  readiness report cannot be `ready` unless it includes a passing
  `rootfs_network` check and `TORCHTITAN_ROOTFS_NETWORK=networked`.
- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so it refuses to
  execute even safe Modal steps when a readiness report has a missing or
  blocked `rootfs_network` check. This prevents a stale or forged readiness
  artifact from bypassing the producer-side network gate.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_readiness_blocks_without_networked_rootfs`
  and
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_blocked_rootfs_network_check`.
  Both were verified red before the implementation.
- Verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `115 passed` in `125.88s`.
- Refreshed completion audit only. It exited `21` with `status=blocked`;
  remaining blockers are `full_preflight_ready`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-23T00:11Z Stage 1 Remote Runner Readiness Consumption

- Continued toward the full-launch objective after the r1 Trainer smoke passed.
  No guarded/full launch and no Modal ingestion was started because the full
  preflight still blocks on corpus and remote-readiness prerequisites.
- Confirmed host and scratch Modal credentials were absent without printing
  secrets. Networked rootfs Stage 1 readiness still exits `21` with
  `modal_auth=blocked` and `rootfs_network=pass`.
- Hardened `experiments/mini_kimi_k3/run_stage1_remote.py` so it refuses to
  execute even safe Modal steps if the readiness artifact is missing or failing
  its `rootfs_network` check. This matches the stricter
  `stage1_remote_readiness.py` producer.
- Added
  `tests/unit_tests/test_mini_kimi_k3_runner.py::test_mini_kimi_k3_stage1_remote_rejects_blocked_rootfs_network_check`.
  The test failed red because `run-stage1-remote` completed a fake Modal
  command with `rootfs_network=blocked`; it passed after the readiness consumer
  checked that field.
- Verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `115 passed` in `125.81s`.
- Refreshed full preflight and completion audit. Full preflight exits `21` with
  `trainable_config=pass`, `model_fidelity=pass`,
  `launch_evidence_bundle=pass`, and `real_corpus_manifest=fail`. Completion
  audit exits `21` with remaining blockers `full_preflight_ready`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-22T23:28Z GPU Resume Gate Refresh

- Rechecked live GPU state through the TorchTitan rootfs. All 8 B200s were
  visible with `0` MiB used and about `182632` MiB free per device. A process
  scan found no live Mini-K3, `torchrun`, or Modal process; the only matching
  long-running process was an operator-owned `watch nvidia-smi`.
- Refreshed r1 smoke GPU readiness:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  experiments/mini_kimi_k3/run.sh r1-smoke-gpu-readiness --report
  experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json
  --min-free-gpu-memory-mib 12000'` exited `0` with `status=pass`.
- Refreshed the bounded one-step r1 Trainer smoke:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  experiments/mini_kimi_k3/run.sh r1-training-smoke --token-manifest
  experiments/mini_kimi_k3/data/manifest.json --tokens-dir
  experiments/mini_kimi_k3/data/tokens --report
  experiments/mini_kimi_k3/results/r1_training_smoke.json --seq-len 4096
  --steps 1 --local-batch-size 1 --global-batch-size 1
  --min-free-gpu-memory-mib 12000'` exited `0`.
- The refreshed smoke report records `status=pass`, `rootfs.marker=1`,
  `rootfs.cwd=/workspace/torchtitan`, `CUDA_VISIBLE_DEVICES=0`,
  `command.return_code=0`, `optimization.steps=1`, and
  `optimization.max_grad_norm=0.0032`.
- Refreshed the r1 corpus-plan audit. It still exits `21` with
  `4097/5000000000` available tokens and a `4999995903` token deficit.
- Refreshed Stage 1 remote readiness with `TORCHTITAN_ROOTFS_NETWORK=networked`.
  It still exits `21`: the rootfs network, HuggingFace DNS, and Modal CLI
  checks pass, but Modal auth is blocked because token credentials are absent.
- Searched local workspace paths for an alternate full Mini-K3/Kimi token
  manifest or source corpus. No ready local 5B-token manifest was found. The
  local HuggingFace FineWeb cache is about `9.6M`, which is not a full corpus
  input for this launch.
- Refreshed non-spend prerequisite checks:
  `experiments/mini_kimi_k3/run.sh corpus-disk-readiness --corpus-plan
  experiments/mini_kimi_k3/reports/r1-corpus-plan.json --report
  experiments/mini_kimi_k3/reports/corpus-disk-readiness.json` exited `0` with
  `65843171328/24000000000` bytes free, and
  `experiments/mini_kimi_k3/run.sh probe-stage1-sources --skip-probe --report
  experiments/mini_kimi_k3/reports/stage1-source-resolution.json` exited `0`
  with zero unresolved benchmark or corpus sources.
- Refreshed full preflight using the first-party ladder r1 config path by
  omitting the optional `--oracle-r1-config` override. It exits `21`, with
  `trainable_config=pass`, `model_fidelity=pass`,
  `launch_evidence_bundle=pass`, and `real_corpus_manifest=fail`.
- Refreshed completion audit. It exits `21` with `status=blocked`; remaining
  blockers are `full_preflight_ready`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.
- No guarded/full launch and no Modal ingest was started.

## 2026-08-22T23:36Z Corpus/Auth Blocker Recheck

- Rechecked local corpus options with read-only filesystem searches under
  `${HOME}` and `/data02`. No ready local 5B Mini-K3/Kimi
  token manifest or shard tree was found.
- Checked known dataset caches. The local TorchTitan HuggingFace dataset cache
  is about `9.6M`, and `hf_home/datasets` is empty; neither can satisfy the
  5B-token launch manifest.
- Refreshed disk capacity and source-resolution prerequisites:
  `corpus-disk-readiness` exited `0` with `65843171328/24000000000` bytes free,
  and `probe-stage1-sources --skip-probe` exited `0` with zero unresolved
  sources.
- Attempted the safe Stage 1 remote `probe` through `run-stage1-remote` using
  the current readiness report. It exited `21` before invoking Modal work:
  `experiments/mini_kimi_k3/reports/stage1-remote-probe.json` records
  `status=blocked`, `argv=[]`, and
  `detail="readiness report must be ready (modal_auth=blocked)"`.
- Refreshed completion audit. It exits `21` with `status=blocked`; remaining
  blockers are `full_preflight_ready`, `real_5b_corpus_manifest`,
  `r1_corpus_plan_audit`, `stage1_remote_readiness`,
  `guarded_launch_completed`, `training_measurement_real`, and
  `train_stage_succeeded`.

## 2026-08-22T23:36Z Live Source Probe And Final Gate Recheck

- Ran live Stage 1 source resolution through the networked rootfs. It exited
  `0` with `stage1 source probe: ready (0 unresolved benchmark(s), 0 unresolved
  corpus source(s))`.
- Refreshed the r1 corpus plan and audit. Planning exited `0` and produced the
  5B-token target; the audit exited `21` with `4097/5000000000` tokens and a
  `4999995903` token deficit.
- Refreshed Stage 1 remote readiness again with networked rootfs. It exited
  `21`; Modal auth remains blocked while DNS, Modal CLI, and rootfs network
  checks pass.
- Refreshed full preflight and completion audit. Full preflight exits `21`.
  Completion audit exits `21` with `status=blocked`; remaining blockers are
  `full_preflight_ready`, `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.

## 2026-08-23T00:41Z GPU Resume Refresh

- Resumed GPU-side gate work after the operator reported that the GPUs were
  free. Host `nvidia-smi` showed all 8 NVIDIA B200 devices at `0` MiB used,
  about `182632` MiB free, and `0%` utilization.
- Refreshed the r1 smoke GPU readiness gate through the TorchTitan rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  experiments/mini_kimi_k3/run.sh r1-smoke-gpu-readiness --report
  experiments/mini_kimi_k3/results/r1_smoke_gpu_readiness.json
  --min-free-gpu-memory-mib 12000'`. It exited `0` and wrote `status=pass`.
- Refreshed the bounded one-step r1 Trainer smoke through the TorchTitan
  rootfs:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  experiments/mini_kimi_k3/run.sh r1-training-smoke --token-manifest
  experiments/mini_kimi_k3/data/manifest.json --tokens-dir
  experiments/mini_kimi_k3/data/tokens --report
  experiments/mini_kimi_k3/results/r1_training_smoke.json --seq-len 4096
  --steps 1'`. It exited `0`.
- The refreshed smoke report records `status=pass`, `rootfs.marker=1`,
  `rootfs.cwd=/workspace/torchtitan`, `CUDA_VISIBLE_DEVICES=0`,
  `command.return_code=0`, `optimization.steps=1`, and
  `optimization.max_grad_norm=0.0031`.
- Refreshed the r1 corpus-plan audit. It still exits `21` with
  `4097/5000000000` available tokens and a `4999995903` token deficit across
  the planned six-source r1 corpus.
- Refreshed Stage 1 remote readiness with host-side
  `TORCHTITAN_ROOTFS_NETWORK=networked`. It still exits `21`: rootfs network,
  Hugging Face DNS, and Modal CLI pass, but Modal auth is blocked with
  `Token missing`.
- Refreshed full preflight. It still exits `21` with
  `trainable_config=pass`, `model_fidelity=pass`,
  `launch_evidence_bundle=pass`, and `real_corpus_manifest=fail`.
- Refreshed completion audit. It still exits `21` with `status=blocked`;
  remaining blockers are `full_preflight_ready`,
  `real_5b_corpus_manifest`, `r1_corpus_plan_audit`,
  `stage1_remote_readiness`, `guarded_launch_completed`,
  `training_measurement_real`, and `train_stage_succeeded`.
- No guarded/full launch and no Modal ingest was started.

## 2026-08-23T00:55Z Local Stage 1 Materialization Route

- Added `experiments/mini_kimi_k3/materialize_stage1_local.py`, a rootfs-only
  non-Modal source-materialization command. It reads the resolved Stage 1
  source report and r1 corpus plan, requires `--allow-download` before any
  Hugging Face file download, and writes one
  `mini_kimi_k3_source_input_manifest` per selected source under
  `experiments/mini_kimi_k3/data/stage1-local-inputs`.
- Wired the command into `experiments/mini_kimi_k3/run.sh` as
  `materialize-stage1-local`.
- Documented the command in `experiments/mini_kimi_k3/README.md` and added it
  as an optional corpus route in
  `.scratch/mini-kimi-k3-replication/unblock-full-launch-wizard.sh`.
- The command is not launch evidence by itself. It only prepares local public
  Hugging Face parquet inputs for later decontamination, Kimi tokenization,
  shard registration, corpus-plan audit, and full preflight.
- Verified the red bar first:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  materialize_stage1_local'` failed at collection because
  `experiments.mini_kimi_k3.materialize_stage1_local` did not exist.
- Verified the new focused materializer behavior after implementation:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py -k
  materialize_stage1_local'` passed with `2 passed, 120 deselected`.
- Verified the command fails closed in the real checkout without download
  authority:
  `TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash
  -lc 'cd /workspace/torchtitan &&
  experiments/mini_kimi_k3/run.sh materialize-stage1-local
  --source-resolution-report
  experiments/mini_kimi_k3/reports/stage1-source-resolution.json
  --corpus-plan-report experiments/mini_kimi_k3/reports/r1-corpus-plan.json
  --output-root experiments/mini_kimi_k3/data/stage1-local-inputs --report
  experiments/mini_kimi_k3/reports/stage1-local-materialize.json'`
  exited `21` with `materialize-stage1-local requires --allow-download`.
- Checked public Hugging Face source file availability through the networked
  rootfs without downloading files. The first resolved parquet per source was:
  `code-python` 355 MB, `cosmopedia` 1174 MB, `finemath` 286 MB,
  `fineweb-edu` 2153 MB, `open-web-math` 236 MB, and `web-diverse` 2148 MB.
  A six-source pilot would therefore download roughly 6 GB before
  tokenization.
- Verified the full Mini-K3 runner suite:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  pytest -q tests/unit_tests/test_mini_kimi_k3_runner.py'` passed with
  `123 passed` in `125.95s`.
- Refreshed the authoritative launch gates after the new report existed. The
  r1 corpus-plan audit still exits `21` with `4097/5000000000` tokens and a
  `4999995903` token deficit. Stage 1 remote readiness still exits `21` with
  `modal_auth=blocked` and `Token missing`. Full preflight still exits `21`;
  completion audit still exits `21` with `status=blocked`.
- No guarded/full launch, no full local materialization, and no Modal ingest
  was started.
