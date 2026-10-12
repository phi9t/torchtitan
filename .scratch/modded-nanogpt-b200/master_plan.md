# Modded NanoGPT B200 Master Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or the repo-local clean-context
> subagent workflow before executing implementation, audit, verification, or
> launch tasks from this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: active master plan
Last updated: 2026-08-20

**Goal:** Carry the TorchTitan-local `modded-nanogpt` B200 program through one
sequential execution-control ladder from the current hardened but failed
full-attempt state to auditable small-scale evidence: first repair and verify
the current blocker, then establish the first B200-compatible local setup trial
on exactly two visible B200 GPUs, then stop for review before any broader GPU
allocation, repeatability campaign, or reproduction claim.

**Architecture:** Keep the benchmark as a repo-local research program under
`experiments/modded_nanogpt_b200/`, with generated sources, data, caches, and
results ignored. The host shell orchestrates only; every substantive Python,
package, data, CUDA, NCCL, FlashAttention, Triton, training, parsing, and
summarization command runs through the bwrap rootfs. Launch permission is
schema-driven: source, data, runtime, hardware, optimized kernels, active-job
safety, authorization, command env, and summary evidence must be verified before
training can count.

**Tech Stack:** TorchTitan repo-local shell wrappers, bwrap rootfs, Python 3.12,
PyTorch `2.13.0+cu132`, CUDA runtime `13.2`, Triton `3.7.1`, NCCL, FA3 via
`kernels`, FA2 via `flash-attn==2.8.3.post1`, uv for Python deps, mise for
non-Python tools, JSON schema-governed attempt artifacts, pytest unit tests,
rootfs-aware preflight and summarization wrappers.

**Specs:**

- `.scratch/modded-nanogpt-b200/spec.md`
- `.scratch/modded-nanogpt-b200/rootfs_runtime_env_spec.md`
- `.scratch/modded-nanogpt-b200/completion_audit.md`
- `.scratch/modded-nanogpt-b200/execution_prompt.md`
- `experiments/modded_nanogpt_b200/preflight_checklist.md`

## Global Constraints

- Direct user instructions, `AGENTS.md`, `CONSTITUTION.md`, and
  `docs/agents/agentic-engineering.md` take precedence.
- Full B200 training launch is not authorized by this plan. A non-skip full
  launch requires the trusted user request itself to contain
  `launch-full-b200`; only then may wrappers pass the lower-level
  `--launch-authorization=launch-full-b200` argument.
- Do not commit, push, open a pull request, merge, clean, delete, reset, stash,
  or rewrite existing generated results without explicit authorization.
- Host shell may create ignored directories, set wrapper environment variables,
  call repo-local shell wrappers, inspect git status, and inspect generated
  logs.
- Python, package installation, dataset code, CUDA, FlashAttention, Triton,
  NCCL, `torchrun`, training, validation, log parsing, telemetry parsing, and
  summarization for real attempts must run inside `scripts/rootfs/enter_rootfs.sh`.
- Do not run upstream `requirements.txt`; it pins or can pull an incompatible
  Torch stack. Install only curated non-Torch direct deps inside rootfs.
- Generated source snapshots, data shards, logs, telemetry, compiler caches,
  result bundles, and run indexes stay out of git.
- Faithful-upstream source, currently represented by legacy `lane=A`, must
  remain the pinned upstream commit with no source edits.
- B200-compatible source differences, currently represented by legacy `lane=B`,
  must be saved as `variant_patch.diff` and classified with the first
  faithful-upstream blockers before preflight or launch.
- A run is not successful unless final validation is reached. Missing final
  validation means `val_loss`, `train_time`, and `step_avg` remain null or
  absent.
- `ready_to_launch=true` is not a launched run. Skip-run dry gates are
  prerequisite evidence only.
- Baseline statistics require non-skip `training_launched=true`, training exit
  code `0`, final validation metrics, rootfs sentinel evidence, the declared
  GPU allocation, NCCL over the declared world size, SHA-verified 900M FineWeb
  data, source provenance, and conservative claim classification.
- The active NanoGPT launch policy is a two-GPU small-scale trial. Future
  non-diagnostic attempts must expose exactly two B200 GPUs to the rootfs and
  launch with a matching two-rank distributed world. This leaves at least one
  GPU available to other users on shared hosts.
- Run NanoGPT attempts sequentially only. A new attempt may start only after the
  previous attempt has finished, artifacts have been summarized, and the
  active-job scan proves no concurrent NanoGPT, `torchrun`, or data-prep work
  is running on the target GPUs.

---

## Claim Classification and Legacy Lane Fields

This plan is now one sequential execution-control ladder. The former Lane A,
Lane B, and Lane C framing is no longer a set of primary execution tracks or an
execution order. Existing JSON, CLI, schema, issue, and result-artifact fields
that use `lane: A|B|C` remain for compatibility and must not be removed in this
documentation-only edit.

Treat lanes as claim labels and legacy artifact fields:

- `lane=A` denotes the legacy faithful-upstream artifact class.
- `lane=B` denotes the legacy B200-compatible local setup artifact class.
- `lane=C` denotes the legacy ablation artifact class.

Approved claim labels for future artifacts and summaries are:

- Faithful upstream reproduction
- B200-compatible local setup
- B200 local variant
- Diagnostic
- Ablation

Current v1 artifact schemas and parser/runner code still emit legacy
`claim_label` strings for compatibility. Interpret `B200 compatibility
patchset` and `B200 systems-only` as the legacy labels for the
`B200-compatible local setup` category, `B200 ML variant` as the legacy label
for `B200 local variant`, `B200 upstream reproduction` as the legacy label for
`Faithful upstream reproduction`, and lowercase `diagnostic` as the legacy label
for `Diagnostic`. Do not rename existing artifact fields or checked-in schema
examples until a separate compatibility migration is planned.

Historical paths such as `lane_b_full_20260816T181747Z`, CLI examples that pass
`--lane`, and schema examples that require `classification.lane` are retained
where they describe existing artifacts or current compatibility contracts. They
do not authorize treating lanes as parallel execution tracks.

---

## Current State

Current resolved slices:

- Issue `01`: source checkout and B200 preflight are resolved.
- Issue `02`: FineWeb data preparation and manifest are resolved.
- Issue `03`: reproduction wrapper and log parser are resolved.
- Issue `05`: harness summary and run index are resolved.
- Issue `10`: experiment schema foundation is resolved.
  `experiments/modded_nanogpt_b200/experiment_config.py` loads typed
  standard-library JSON specs, uses `experiment_kind` as the canonical field,
  maps legacy `lane` values for existing artifacts, validates 1/2/4/8 GPU
  arms, classifies supported versus record-only knobs, and expands advisory
  observability profiles.
- Issue `11`: experiment plan materialization is resolved.
  The schema layer materializes run IDs, attempt IDs, result directories,
  legacy lane fields, claim labels, claim eligibility, visible GPU env,
  `torchrun` world size, and preflight expected GPU counts.
- Issue `12`: experiment matrix runner is resolved for dry-run and explicitly
  authorized mocked sequential execution paths. `run_experiment_matrix.py`
  writes plan/report artifacts, defaults its CLI to dry-run, requires
  `--execute` for non-dry arm execution, executes arms sequentially through
  `run_speedrun.run_attempt`, stops after the first failed arm, refreshes the
  run index after non-dry execution, records pending arms as skipped instead of
  launching them after a blocker, and preserves per-arm observability
  summaries. The checked-in
  `experiments/modded_nanogpt_b200/configs/gpu_ladder_prerequisite.json`
  now points at the current full manifest refresh and selected FA2/Triton
  prerequisite tuple rather than the stale torch-MLP fallback.
- Issue `13`: observability profiles and RSI report are resolved as advisory
  matrix evidence. ByteRobust/Mycroft/Argus/Eroica-style profiles classify
  feature status, produce semantic timeline or diagnostic recommendations
  without launching probes, and collect matrix-level RSI evidence without
  granting recovery, retry, or launch authority.
- Issue `14`: diagnostic performance probe ladder is resolved. The selected
  FA2/Triton tuple passed static wrapper/preflight timing, import/construction
  timing, one-GPU synthetic CUDA microstep plus Triton backend smoke, two-rank
  NCCL plus two-GPU microstep prerequisite, watchdog heartbeat calibration, and
  observability overhead control. Torch MLP fallback remains a diagnostic-only
  B200 kernel blocker and must not be used for the next full launch.
- Issue `08`: optimized kernel certification is resolved for the selected
  FA2/Triton tuple. The rootfs-generated report at
  `experiments/modded_nanogpt_b200/results/issue08_kernel_cert_fa2_triton_20260819T083000Z/runtime/optimized_kernel_report.json`
  records `launch_eligible=true`, `blockers=[]`, and digest
  `a789b17eef84c62d386f4f395d32ec0c0924bc261fcd450c07668d17d3882969`.
  The selected rows pass for FA2 attention, Triton MLP, source-local
  Triton/DC kernels, FP8 `torch._scaled_mm`, TorchInductor cache, Triton tensor
  descriptor, and two-rank NCCL. Unselected FA3, FA4, flex, torch SDPA, torch
  MLP fallback, and installed-but-unselected `flashinfer` remain visible
  nonblocking rows.
- Master-plan Task `09`: the non-launch prerequisite refresh is resolved for
  the current two-GPU Lane B FA2/Triton launch gate. This does not resolve
  issue `09`, which remains blocked until an authorized non-skip trial can run.
  The latest rootfs-generated skip-run artifact at
  `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z/summary.json`
  records `claim_eligible=true`, `ready_to_launch=true`,
  `training_launched=false`, `skip_run=true`, `blocked_by=[]`, 2x B200 GPU
  evidence, clear active-job evidence, matching command-environment and runtime
  verification digests, embedded
  `runtime_verification.training_launch_allowed=true`, and launch-eligible
  optimized-kernel evidence. The older IPv4 NCCL row remains historical proof
  of the rendezvous repair, but the strict runtime-env row is the current
  handoff prerequisite.
- Runtime component design exists in `rootfs_runtime_env_spec.md`, including
  rootfs layout, uv, mise, schemas, verifiers, PyTorch/nanoGPT components, and
  optimized-kernel report design.

Current blocked slices:

- Issue `06`: the first B200-compatible local setup baseline is blocked by
  the missing trusted-message `launch-full-b200` token. The older authorized
  launched legacy `lane=B` attempt reached `torchrun` and failed because
  training produced no output for the 600-second no-output watchdog window
  during model compile and kernel warmup. The diagnostic probe ladder has since
  identified the selected FA2/Triton tuple and current non-launch prerequisite,
  so the next step is an authorized fresh full attempt rather than another
  equivalent diagnostic.
- Issue `09`: the first two-GPU trial did not launch. The current non-launch
  prerequisite refresh selected visible GPUs `0,1`, passed the full-mode
  FA2/Triton skip-run gate, and refreshed the run index, but no non-skip
  training attempt began because the trusted user request did not contain
  `launch-full-b200`. The next iteration must start from a
  fresh result directory after rechecking idle GPUs and active jobs.
- Issue `04`: ablation work is blocked by an accepted faithful-upstream or
  B200-compatible local setup baseline. The active Lane B path to that baseline
  is the trusted-authorized, non-skip, two-GPU FA2/Triton full attempt.
- Faithful-upstream full reproduction, currently represented by legacy `lane=A`,
  is blocked by FA3 B200 kernel support in the current runtime.
- The earlier production target of 10 successful full jobs is superseded by the
  two-GPU sequential trial policy. No repeatability campaign should start until
  a two-GPU trial succeeds and the user explicitly authorizes broader use.

Current evidence to cite before rerunning anything:

- Latest full manifest:
  `experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`
- Latest non-launch B200-compatible prerequisite, using legacy `lane=B`:
  `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z/`
- Latest launched B200-compatible attempt, using legacy `lane=B`:
  `experiments/modded_nanogpt_b200/results/lane_b_full_20260818T014833Z_attempt_001/`
- Latest two-GPU prelaunch-only trial directory:
  `experiments/modded_nanogpt_b200/results/lane_b_two_gpu_trial_20260818T041445Z_attempt_001/`
- Current run index:
  `experiments/modded_nanogpt_b200/results/run_index.json`
- Current completion audit:
  `.scratch/modded-nanogpt-b200/completion_audit.md`
- Latest issue `14` selected-tuple diagnostics:
  `experiments/modded_nanogpt_b200/results/issue14_diag_microstep_1gpu_triton_20260819T073926Z_attempt_001/`
  and
  `experiments/modded_nanogpt_b200/results/issue14_diag_microstep_2gpu_triton_20260819T073926Z_attempt_001/`

Current numeric state from current artifacts:

- `baseline_stats.count=0`
- `total_attempts=63`
- current run-index `launch_prerequisite_attempts=4`, preserving historical
  prerequisite rows
- one current strict launch-prerequisite artifact:
  `lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z`
- `launch_ready_attempts=0`
- The current prerequisite rows are skip-run dry gates, not production evidence.
- The launched attempt has `training_launched=true`, `skip_run=false`,
  `preflight_ok=true`, and training exit code `124`; it is not baseline
  evidence.
- The concrete launched-attempt blocker is
  `training produced no output for 600 seconds`.
- RSI-control foundation verification:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_experiment_config.py tests/unit_tests/test_modded_nanogpt_b200_run_experiment_matrix.py && python3 -m py_compile experiments/modded_nanogpt_b200/experiment_config.py experiments/modded_nanogpt_b200/run_experiment_matrix.py'`
  passed with `14 passed in 0.17s` and compile success. A rootfs wrapper
  dry-run of `run_experiment_matrix.sh` against a temporary copy of the
  checked-in config wrote plan and matrix report artifacts without launching
  training; the report recorded four planned arms, advisory RSI evidence,
  `g2_visible_devices=0,1`, and `g2_claim_label=B200 compatibility patchset`.
  Focused regression coverage also proves non-dry matrix execution stops after
  the first failed arm, preserves that first failing exit code, marks later
  arms `status=skipped` with `skip_reason=previous_arm_failed`, and reports
  skipped arms separately from missing-summary warnings in advisory RSI
  evidence.
  The earlier rootfs guard/schema/matrix check
  `python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_cli_guard.py tests/unit_tests/test_modded_nanogpt_b200_experiment_config.py tests/unit_tests/test_modded_nanogpt_b200_run_experiment_matrix.py`
  passed with `29 passed in 0.86s`, including direct host fail-closed coverage
  for `run_experiment_matrix.py` and CLI coverage proving `parse_args([])`
  stays dry-run while `parse_args(["--execute"])` is the only non-dry path.
  A wrapper invocation of `run_experiment_matrix.sh --config <temp config>`
  without `--dry-run` or `--execute` also wrote planned-only artifacts with
  `dry_run=true` and did not launch training.
  `python3 experiments/modded_nanogpt_b200/verify_static.py --json` listed the
  changed matrix runner and matrix tests among static candidates, and
  `python3 experiments/modded_nanogpt_b200/verify_static.py` passed with
  `Static verification passed for 36 file(s).` The later completion-audit
  bundle supersedes this as the current broad non-launch verification:
  `410 passed, 2 skipped`, the focused tracker guard reports
  `49 passed`, explicit-file Pyrefly
  over the 31-file static Python surface reports `0 errors`, and static
  verification reports `Static verification passed for 114 file(s).` A later
  rootfs all-files pre-commit run passed with only the protected-branch hook
  skipped: `SKIP=no-commit-to-branch pre-commit run --all-files`.

## Sequential Milestone Ladder

Execute milestones in this order unless a direct user instruction changes the
goal. A later milestone does not begin until the earlier launch-governing gates
that feed it are green or explicitly classified as diagnostic-only.

1. **Evidence Baseline:** keep tracker/specs coherent, preserve the failed
   launch facts, and make schema/static checks the first gate.
2. **Hermetic Local Runtime:** land rootfs runtime, uv Python runtime, mise tool
   runtime, and schema files without weakening the rootfs-only boundary.
3. **Source and Data Visibility:** repair and verify the `DATA_PATH` / manifest
   shard-root contract before any further full launch.
4. **Kernel and Distributed Stack:** harden PyTorch/nanoGPT component evidence,
   NCCL, and the optimized-kernel certification matrix. A full launch is valid
   only when the selected attention/MLP/supporting tuple is `supported` and
   passes build plus smoke checks. Unsupported unselected FA3, FA4, Triton, and
   fallback paths must still be probed or classified with structured evidence,
   but they do not block a known-good selected tuple.
5. **No-Output Performance Probe Ladder:** resolved by issue `14`. Preserve its
   diagnostic artifacts as non-baseline evidence and use only the selected
   FA2/Triton tuple for the next full attempt.
6. **Launch Gate:** reconcile runtime verifier, launcher, readiness,
   summarizer, active-job scan, authorization, two-GPU allocation, and
   non-launch prerequisite evidence.
7. **First Two-GPU Trial:** run the next authorized B200-compatible attempt only
   after the data-path gate and launch gate pass, with exactly two visible B200
   GPUs and no concurrent NanoGPT attempts, then preserve or repair the first
   real blocker.
8. **Repeatability:** do not run a production campaign by default. After one
   two-GPU trial succeeds, run only sequential follow-up attempts that the user
   explicitly authorizes, refreshing the index after each and stopping on the
   first new blocker.
9. **Optional Claim Refinement and Ablations:** revisit faithful-upstream FA3
   only when a plausible B200 `sm_100` path exists, and run ablations only after
   a baseline exists.

## Artifact Map

Tracked or intended tracked control files:

- `.scratch/modded-nanogpt-b200/master_plan.md`: this navigation plan.
- `.scratch/modded-nanogpt-b200/spec.md`: canonical harness spec.
- `.scratch/modded-nanogpt-b200/rootfs_runtime_env_spec.md`: production rootfs,
  schema, package, filesystem, and optimized-kernel verifier spec.
- `.scratch/modded-nanogpt-b200/completion_audit.md`: current evidence audit and
  blocked-state summary.
- `.scratch/modded-nanogpt-b200/execution_prompt.md`: launch-session operating
  prompt.
- `.scratch/modded-nanogpt-b200/issues/*.md`: ticket history and current
  blockers.
- `experiments/modded_nanogpt_b200/*.py`: source, data, preflight, runner,
  parser, summarizer, diagnostics, active-job scan, and static verifier code.
- `experiments/modded_nanogpt_b200/*.sh`: rootfs-aware entrypoints.
- `experiments/modded_nanogpt_b200/preflight_checklist.md`: operator launch
  checklist.
- `tests/unit_tests/test_modded_nanogpt_b200_*.py`: focused unit coverage.
- `scripts/rootfs/*.sh`: shared TorchTitan rootfs build/entry support.

Generated and ignored state:

- `experiments/modded_nanogpt_b200/sources/`: pinned upstream and
  B200-compatible source snapshots.
- `experiments/modded_nanogpt_b200/data/`: reusable experiment data.
- `experiments/modded_nanogpt_b200/results/`: attempts, manifests, telemetry,
  compiler caches, summaries, and run indexes.
- `.cache/torchtitan-rootfs/modded_nanogpt_b200/`: target runtime state root.
- `scripts/rootfs/rootfs/` or future `scripts/rootfs/store/`: rootfs image
  state.

## Task 1: Tracker and Spec Coherence

**Files:**

- Modify: `.scratch/modded-nanogpt-b200/master_plan.md`
- Modify as needed: `.scratch/modded-nanogpt-b200/completion_audit.md`
- Modify as needed: `.scratch/modded-nanogpt-b200/spec.md`
- Modify as needed: `.scratch/modded-nanogpt-b200/rootfs_runtime_env_spec.md`
- Modify as needed: `.scratch/modded-nanogpt-b200/issues/*.md`

**Interfaces:**

- Consumes: current specs, issue docs, result artifacts, and run index.
- Produces: a coherent tracker where future agents can identify authority,
  current blocker, next gate, and completed slices without replaying history.

- [x] **Step 1: Read authoritative context**

  Run read-only inspection:

  ```bash
  sed -n '1,220p' CONSTITUTION.md
  sed -n '1,220p' docs/agents/agentic-engineering.md
  sed -n '1,220p' AGENTS.md
  sed -n '1,260p' .scratch/modded-nanogpt-b200/spec.md
  sed -n '1,220p' .scratch/modded-nanogpt-b200/rootfs_runtime_env_spec.md
  sed -n '1,220p' .scratch/modded-nanogpt-b200/completion_audit.md
  ```

  Expected: no command mutates files; the current blocker and authority boundary
  are clear.

- [x] **Step 2: Update the tracker only when state changes**

  Edit `completion_audit.md` only after new evidence exists. Do not refresh
  merely to restate the same launch-authorization blocker.

- [x] **Step 3: Run static text checks**

  Run:

  ```bash
  python3 experiments/modded_nanogpt_b200/verify_static.py
  python3 - <<'PY'
  from pathlib import Path

  roots = [Path(".scratch/modded-nanogpt-b200"), Path("experiments/modded_nanogpt_b200")]
  skip_dirs = {"results", "sources", "data", "__pycache__"}
  needle = "\\" + "n+"
  bad = []
  for root in roots:
      for path in root.rglob("*"):
          if not path.is_file():
              continue
          if any(part in skip_dirs for part in path.parts):
              continue
          if needle in path.read_text(errors="ignore"):
              bad.append(str(path))
  if bad:
      raise SystemExit("\\n".join(bad))
  PY
  ```

  Expected: static verifier passes; the Python scan prints no malformed patch
  artifacts. `rg` is not required inside the rootfs.

  Evidence: rootfs static verifier reported `Static verification passed for
  20 file(s).` The original `rg` command was not portable because `rg` is not
  installed in the rootfs, and a `grep` fallback self-matched this plan's own
  instruction line. The Python scan above builds the pattern at runtime, avoids generated trees, and
  self-matching command text.

## Task 2: Rootfs Runtime Store and Canonical Environment

**Files:**

- Create: `scripts/rootfs/runtime_env.sh`
- Modify: `scripts/rootfs/build_rootfs.sh`
- Modify: `scripts/rootfs/enter_rootfs.sh`
- Modify: `scripts/rootfs/rootfs_target.sh`
- Create tests as appropriate under `tests/unit_tests/`

**Interfaces:**

- Consumes: `rootfs_runtime_env_spec.md` mount, env, state-root, and network
  mode contract.
- Produces: a single rootfs runtime contract with canonical paths, read-only
  rootfs support, writable state binds, `/dev/shm`, temp/scratch/cache floors,
  and networked/offline modes.

- [x] **Step 1: Add a failing unit seam for canonical env generation**

  Expected env keys:

  ```text
  TORCHTITAN_IN_ROOTFS
  TORCHTITAN_ROOTFS_ENV
  TORCHTITAN_ROOTFS_STORE_ID
  TORCHTITAN_ROOTFS_HOST_STATE
  HOME
  XDG_CACHE_HOME
  UV_CACHE_DIR
  PIP_CACHE_DIR
  MISE_DATA_DIR
  MISE_CACHE_DIR
  MISE_CONFIG_DIR
  PYTHON
  TMPDIR
  HF_HOME
  HF_HUB_CACHE
  TORCH_HOME
  CUDA_HOME
  CUDA_PATH
  ```

  Expected red: the current entrypoint still uses `HOME=/root` and lacks the
  central runtime-state contract.

  Evidence: `tests/unit_tests/test_rootfs_bwrap_plan.py::test_enter_rootfs_plan_exports_canonical_runtime_environment`
  failed red on missing `TORCHTITAN_ROOTFS_ENV` and old `HOME=/root`, then
  passed after the launcher emitted the canonical `/project` runtime
  environment.

- [x] **Step 2: Implement `runtime_env.sh`**

  It must compute paths from the repo root and optional environment overrides,
  reject paths outside the repo or runtime state root, and expose shell
  functions for env export and bwrap bind construction.

  Evidence: `scripts/rootfs/runtime_env.sh` now centralizes canonical env
  values, repo-local runtime state root validation, state directory creation,
  `/project/*` bwrap bind construction, and bwrap env export. Focused tests in
  `tests/unit_tests/test_rootfs_runtime_env_shell.py` cover canonical JSON
  emission and rejection of host state roots outside the repo `.cache` tree.
  `scripts/rootfs/enter_rootfs.sh` now sources the helper instead of owning a
  second copy of the state-bind and environment contract.

- [x] **Step 3: Add managed rootfs store support**

  `build_rootfs.sh` must stage rootfs content, pre-create bind targets, write an
  ownership marker and `manifest.json`, and atomically activate a selected
  store entry. Legacy `scripts/rootfs/rootfs` remains diagnostic unless it has a
  schema-valid legacy manifest with `mutable_rootfs_allowed=false`.

  Evidence: `scripts/rootfs/build_rootfs.sh` now supports `--store DIR` to
  publish a staged rootfs under `DIR/content/<store_id>` and update
  `DIR/selected.json`; `scripts/rootfs/enter_rootfs.sh` supports
  `--rootfs-store DIR` and resolves the selected entry before plan emission.
  `scripts/rootfs/rootfs_target.sh` now validates ownership markers and
  `build_manifest.json` records with `mutable_rootfs_allowed=false` for both
  selected store entries and manifest-required legacy rootfs trees. Focused
  red/green coverage:
  `python3 -m pytest -q tests/unit_tests/test_rootfs_bwrap_plan.py::test_enter_rootfs_resolves_selected_store_root tests/unit_tests/test_rootfs_bwrap_plan.py::test_enter_rootfs_rejects_legacy_rootfs_without_manifest tests/unit_tests/test_rootfs_bwrap_plan.py::test_enter_rootfs_accepts_nonmutable_legacy_manifest tests/unit_tests/test_rootfs_build_store_shell.py tests/unit_tests/test_execution_rootfs_selection_shell.py`
  passed with `11 passed in 5.51s`.

- [x] **Step 4: Add network mode**

  `enter_rootfs.sh` must support:

  ```text
  TORCHTITAN_ROOTFS_NETWORK=networked|offline
  ```

  Full launches use offline mode unless explicitly diagnostic.

  Evidence: `scripts/rootfs/enter_rootfs.sh` now defaults
  `TORCHTITAN_ROOTFS_NETWORK` to `offline`, omits `--share-net` and resolver
  binds in offline plans, and accepts `networked` only as an explicit opt-in.
  Invalid values fail before plan emission. `scripts/rootfs/verify_runtime_env.py`
  validates only `offline` and `networked`. Focused red/green coverage:
  `python3 -m pytest -q tests/unit_tests/test_rootfs_bwrap_plan.py::test_enter_rootfs_emits_plan_without_running_bwrap tests/unit_tests/test_rootfs_bwrap_plan.py::test_enter_rootfs_can_emit_networked_plan tests/unit_tests/test_rootfs_bwrap_plan.py::test_enter_rootfs_rejects_invalid_network_mode`
  passed with `3 passed in 3.58s`.

- [x] **Step 5: Verify without GPU launch**

  Run host-side static/unit checks only:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_rootfs_runtime_env.py
  bash -n scripts/rootfs/runtime_env.sh scripts/rootfs/build_rootfs.sh scripts/rootfs/enter_rootfs.sh
  ```

  Evidence: `python3 -m pytest -q tests/unit_tests/test_rootfs_runtime_env_shell.py tests/unit_tests/test_rootfs_bwrap_plan.py`
  passed with `10 passed in 8.81s` for the initial environment and network
  slice. After managed-store integration,
  `python3 -m pytest -q tests/unit_tests/test_rootfs_runtime_env_shell.py tests/unit_tests/test_rootfs_bwrap_plan.py tests/unit_tests/test_rootfs_build_store_shell.py tests/unit_tests/test_execution_rootfs_selection_shell.py tests/unit_tests/test_execution_rootfs_identity.py`
  passed with `36 passed in 14.31s`. Latest refresh of the same shared-rootfs
  surface passed with `36 passed in 14.53s`. `bash -n
  scripts/rootfs/runtime_env.sh scripts/rootfs/enter_rootfs.sh
  scripts/rootfs/build_rootfs.sh scripts/rootfs/rootfs_target.sh`, `python3 -m
  py_compile scripts/rootfs/verify_runtime_env.py`, and `git diff --check`
  all exited successfully.

## Task 3: uv Python Runtime

**Files:**

- Create: `experiments/modded_nanogpt_b200/runtime/pyproject.toml`
- Create: `experiments/modded_nanogpt_b200/runtime/requirements.direct.txt`
- Create: `experiments/modded_nanogpt_b200/runtime/requirements.lock`
- Create: `experiments/modded_nanogpt_b200/runtime/sync_python_env.sh`
- Modify: `experiments/modded_nanogpt_b200/preflight.py`
- Modify: `experiments/modded_nanogpt_b200/run_preflight.sh`
- Modify: `experiments/modded_nanogpt_b200/run_speedrun.py`
- Add tests under `tests/unit_tests/test_modded_nanogpt_b200_runtime_*.py`

**Interfaces:**

- Consumes: rootfs canonical env and existing preflight runtime checks.
- Produces: `/project/venvs/b200-runtime/bin/python` plus a structured
  `python_env_report.json`.

- [x] **Step 1: Add red tests for dependency policy**

  Assert the runtime dep inputs exclude `torch`, `triton`, and CUDA runtime
  replacement packages, while including the direct non-Torch deps:

  ```text
  numpy
  tqdm
  huggingface-hub
  datasets
  tiktoken
  typing-extensions
  setuptools
  kernels
  flash-attn==2.8.3.post1
  ```

  Evidence: `tests/unit_tests/test_modded_nanogpt_b200_runtime_python.py`
  covers runtime dependency inputs, excludes rootfs-owned Torch/Triton/CUDA
  replacement packages, and requires the direct non-Torch dependency set above.
  The test failed red before `experiments/modded_nanogpt_b200/runtime/` existed
  and passed after adding the runtime files.

- [x] **Step 2: Implement sync wrapper**

  The wrapper re-enters rootfs and runs:

  ```bash
  uv venv --python "$(command -v python)" --system-site-packages /project/venvs/b200-runtime
  uv pip install --python /project/venvs/b200-runtime/bin/python --no-deps --require-hashes -r experiments/modded_nanogpt_b200/runtime/requirements.lock
  ```

  Normal sync uses the populated hash lock. The `requirements.direct.txt`
  fallback is reserved for explicit networked setup diagnostics or lock
  regeneration; it is not a full-launch path.

  Evidence: `experiments/modded_nanogpt_b200/runtime/sync_python_env.sh`
  re-enters `scripts/rootfs/enter_rootfs.sh` when invoked from the host, creates
  `/project/venvs/b200-runtime` with `uv venv --system-site-packages`, uses
  `uv pip install --no-deps --require-hashes` when `requirements.lock` is
  populated, and refuses the direct requirements fallback unless
  `TORCHTITAN_ROOTFS_NETWORK=networked`. The checked-in lock is now populated
  with hashes for the direct non-Torch runtime dependencies only. A full
  dependency lock was rejected because it pulled forbidden Torch/Triton/CUDA
  replacement wheels; the sync contract intentionally installs this direct
  lock with `--no-deps --require-hashes` over the rootfs-owned Torch stack.

- [x] **Step 3: Gate active Python**

  Full preflight and full launch must use `/project/venvs/b200-runtime/bin/python`
  once the sync wrapper exists. Host-side direct Python remains blocked by
  `cli_guard.py`.

  Evidence: `experiments/modded_nanogpt_b200/rootfs_guard.sh` now exposes
  `select_modded_nanogpt_python`, and `run_preflight.sh` plus
  `run_speedrun.sh` exec that selected interpreter. When the managed venv is
  present it is preferred; otherwise wrappers keep the existing rootfs Python
  fallback. `preflight.py` itself did not need a code change because the
  interpreter boundary is owned by the rootfs shell wrappers. Host-side direct
  Python remains covered by `tests/unit_tests/test_modded_nanogpt_b200_cli_guard.py`.

- [x] **Step 4: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py
  python3 experiments/modded_nanogpt_b200/verify_static.py
  ```

  Evidence: `python3 -m pytest -q
  tests/unit_tests/test_modded_nanogpt_b200_runtime_python.py
  tests/unit_tests/test_modded_nanogpt_b200_preflight.py
  tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py` passed with
  `67 passed in 1.33s`. `bash -n
  experiments/modded_nanogpt_b200/runtime/sync_python_env.sh
  experiments/modded_nanogpt_b200/rootfs_guard.sh
  experiments/modded_nanogpt_b200/run_preflight.sh
  experiments/modded_nanogpt_b200/run_speedrun.sh`,
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan &&
  python3 experiments/modded_nanogpt_b200/verify_static.py'`, and
  `git diff --check` all passed; static verification reported `Static
  verification passed for 25 file(s).`

## Task 4: mise Tool Runtime

**Files:**

- Create: `experiments/modded_nanogpt_b200/runtime/mise.toml`
- Create: `experiments/modded_nanogpt_b200/runtime/sync_tools.sh`
- Modify: `scripts/rootfs/build_rootfs.sh`
- Add tests under `tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py`

**Interfaces:**

- Consumes: rootfs canonical env and runtime state root.
- Produces: `/project/mise` tool state and `tool_env_report.json`.

- [x] **Step 1: Add red tests for tool path ownership**

  Assert `MISE_DATA_DIR`, `MISE_CACHE_DIR`, and `MISE_CONFIG_DIR` resolve under
  `/project` or the checked-in runtime directory, never under host home.

  Evidence: `tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py` asserts
  the default tool state paths in `sync_tools.sh` stay under `/project/mise`
  and the checked-in runtime directory, with no `$HOME/.local` or `$HOME/.cache`
  ownership.

- [x] **Step 2: Install mise in the rootfs build**

  Pin mise by version in the rootfs manifest. Do not curl-install mise during a
  training attempt.

  Evidence: `scripts/rootfs/build_rootfs.sh` now defines `MISE_VERSION`, installs
  `/usr/local/bin/mise` during rootfs image build, and records `mise_version` in
  `build_manifest.json`. Runtime sync does not curl-install tools during an
  attempt; it requires `mise` to already exist in the rootfs.

- [x] **Step 3: Add tool sync wrapper**

  Wrapper commands:

  ```bash
  mise trust experiments/modded_nanogpt_b200/runtime/mise.toml
  mise install --yes --cd experiments/modded_nanogpt_b200/runtime
  mise exec --cd experiments/modded_nanogpt_b200/runtime -- shellcheck --version
  ```

  Evidence: `experiments/modded_nanogpt_b200/runtime/mise.toml` declares
  `shellcheck = "0.10.0"`, and `sync_tools.sh` re-enters the rootfs from host,
  runs `mise trust`, `mise install --yes`, and a `mise exec ... shellcheck
  --version` smoke, then writes `tool_env_report.json`.

- [x] **Step 4: Verify**

  Run:

  ```bash
  bash -n experiments/modded_nanogpt_b200/runtime/sync_tools.sh
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py
  ```

  Evidence: `python3 -m pytest -q
  tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py` passed with
  `4 passed in 0.05s`. Broader runtime/rootfs coverage,
  `python3 -m pytest -q
  tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py
  tests/unit_tests/test_modded_nanogpt_b200_runtime_python.py
  tests/unit_tests/test_rootfs_build_store_shell.py
  tests/unit_tests/test_rootfs_bwrap_plan.py`, passed with `23 passed in
  14.29s`. `bash -n experiments/modded_nanogpt_b200/runtime/sync_tools.sh
  experiments/modded_nanogpt_b200/runtime/sync_python_env.sh
  scripts/rootfs/build_rootfs.sh`, rootfs-contained static verification, and
  `git diff --check` all passed; static verification reported `Static
  verification passed for 27 file(s).`

## Task 5: Schema Files and Validation Helpers

**Files:**

- Create: `experiments/modded_nanogpt_b200/runtime/schemas/*.schema.json`
- Create: `experiments/modded_nanogpt_b200/runtime/schema_validation.py`
- Modify: `experiments/modded_nanogpt_b200/preflight.py`
- Modify: `experiments/modded_nanogpt_b200/run_speedrun.py`
- Modify: `experiments/modded_nanogpt_b200/parse_log.py`
- Modify: `experiments/modded_nanogpt_b200/summarize.py`
- Add tests under `tests/unit_tests/test_modded_nanogpt_b200_schemas.py`

**Interfaces:**

- Consumes: artifact shapes defined in `spec.md` and `rootfs_runtime_env_spec.md`.
- Produces: current-version schema validation for every launch-governing input
  and report.

- [x] **Step 1: Add red tests for required schemas**

  Required schema files:

  ```text
  rootfs_manifest.schema.json
  runtime_env.schema.json
  python_env_report.schema.json
  tool_env_report.schema.json
  filesystem_report.schema.json
  launch_prerequisites.schema.json
  runtime_verification.schema.json
  attempt.schema.json
  preflight_report.schema.json
  launch_readiness.schema.json
  command_env.schema.json
  command_argv.schema.json
  nanogpt_component_manifest.schema.json
  optimized_kernel_report.schema.json
  summary.schema.json
  ```

  Tests must prove missing required fields fail and unknown top-level fields are
  rejected for launch-gating reports.

- [x] **Step 2: Implement standard-library bootstrap validation**

  The first implementation may validate required keys, types, schema version,
  and top-level unknown fields without a third-party `jsonschema` dependency.

- [x] **Step 3: Wire writers to schemas**

  Existing writers should call validation before atomic writes for new
  current-version artifacts.

- [x] **Step 4: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_schemas.py
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py
  ```

  Evidence:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_schemas.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_runtime_python.py tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py'`
  passed with `64 passed in 1.47s`.

  Evidence:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py'`
  passed with `75 passed in 0.28s`.

## Task 6: Runtime Verifier

**Files:**

- Create: `scripts/rootfs/verify_runtime_env.py`
- Create: `experiments/modded_nanogpt_b200/runtime/verify_runtime.py`
- Modify: `experiments/modded_nanogpt_b200/run_preflight.sh`
- Modify: `experiments/modded_nanogpt_b200/run_speedrun.py`
- Add tests under `tests/unit_tests/test_modded_nanogpt_b200_runtime_verifier.py`

**Interfaces:**

- Consumes: rootfs manifests, canonical env, Python/tool reports, filesystem
  state, preflight reports, launch prerequisites, command env/argv, and
  authorization state.
- Produces: attempt-local `runtime_verification.json`, `command.env.json`,
  `command.argv.json`, and launch prerequisite reports.

- [x] **Step 1: Add red tests for acyclic verifier flow**

  A wrapper must be unable to write `ready_to_launch=true` unless
  `runtime_verification.ok=true`, current schema versions are present, and
  `training_launch_allowed=true`.

  Evidence: `tests/unit_tests/test_modded_nanogpt_b200_runtime_verifier.py`
  covers successful verifier output and partial failure output. Existing
  `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py` coverage proves
  authorized full launches fail closed before telemetry and `torchrun` when the
  runtime verification sidecar is missing, malformed, `ok=false`, stale-digest,
  or lacks `training_launch_allowed=true`.

- [x] **Step 2: Implement repo-generic verifier**

  `scripts/rootfs/verify_runtime_env.py` checks rootfs sentinel, cwd, store ID,
  mount targets, network mode, canonical env, device/driver visibility,
  filesystem capacity, Python path, and Torch/CUDA/Triton drift.

  Evidence: `scripts/rootfs/verify_runtime_env.py` validates the emitted bwrap
  plan's schema, cwd, rootfs store ID, mount targets, offline/networked mode,
  canonical `/project` environment, managed Python path, and visible-device
  export. Device/driver visibility, filesystem capacity, and live
  Torch/CUDA/Triton drift remain delegated to preflight and optimized-kernel
  reports rather than this static bwrap-plan verifier.

- [x] **Step 3: Implement experiment verifier**

  `experiments/modded_nanogpt_b200/runtime/verify_runtime.py` joins Python,
  tool, filesystem, source, data, hardware, NCCL, backend, active-job,
  authorization, command env, and preflight evidence.

  Evidence: `experiments/modded_nanogpt_b200/runtime/verify_runtime.py` now
  validates attempt-local `command.env.json`, preserves the command-environment
  digest, writes `runtime/runtime_verification.json`, certifies
  `training_launch_allowed`, and fails closed on malformed command-env records
  or explicit noncanonical runtime fields. `run_speedrun.py` writes
  schema-governed `command.argv.json` alongside the legacy `command.argv` text
  sidecar and requires a current verifier report before training launch.
  Broader joins across Python/tool/filesystem/source/data/hardware/NCCL/backend
  evidence remain represented through preflight, launch-readiness, and
  optimized-kernel sidecars; they are not duplicated inside the bootstrap
  verifier.

- [x] **Step 4: Preserve partial failure reports**

  Every failure includes `phase`, `message`, `expected`, `actual`, and
  `artifact_path` when applicable, and writes partial verifier output before
  returning exit `21`.

  Evidence: partial verifier failures write `runtime/runtime_verification.json`
  with `ok=false`, `training_launch_allowed=false`, a stable
  `command_env_digest`, and blocker entries containing `phase`, `message`,
  `expected`, `actual`, and `artifact_path`.

- [x] **Step 5: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_runtime_verifier.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py
  ```

  Evidence:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_schemas.py tests/unit_tests/test_modded_nanogpt_b200_runtime_verifier.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py'`
  passed with `57 passed in 1.30s`.

  Evidence:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile experiments/modded_nanogpt_b200/runtime/verify_runtime.py experiments/modded_nanogpt_b200/run_speedrun.py && python3 experiments/modded_nanogpt_b200/verify_static.py'`
  passed, with static verification reporting `Static verification passed for
  31 file(s).`

## Task 7: PyTorch/nanoGPT Component and Optimized-Kernel Verifier

**Files:**

- Create: `experiments/modded_nanogpt_b200/runtime/nanogpt_components.py`
- Modify: `experiments/modded_nanogpt_b200/preflight.py`
- Modify: `experiments/modded_nanogpt_b200/runtime/verify_runtime.py`
- Add tests under `tests/unit_tests/test_modded_nanogpt_b200_components.py`

**Interfaces:**

- Consumes: selected source tree, legacy lane artifact field, attention backend,
  MLP backend, command env, and rootfs Python runtime.
- Produces: `nanogpt_component_manifest.json` and
  `optimized_kernel_report.json`.

**Tracker:** Implement this task through Issue `08`, which sharpens the
optimized-kernel verifier into a certified matrix gate before launch-readiness
integration.

- [x] **Step 1: Add red source-inventory tests**

  Tests must prove the manifest distinguishes:

  - selected `train_gpt.py`;
  - unselected `train_gpt_medium.py`;
  - source-local `triton_kernels.py`;
  - source-local `dc_triton_kernels.py` when imported;
  - `data/cached_fineweb10B.py`;
  - `data/fineweb.py`;
  - current absence of `flashinfer`, `flash_infer`, and `flash-infer`.

  Evidence: Issue `08` completed this task through
  `experiments/modded_nanogpt_b200/optimized_kernel_certifier.py` and
  `tests/unit_tests/test_modded_nanogpt_b200_optimized_kernel_certifier.py`.
  The current implementation records selected and unselected attention/MLP
  rows, source-local Triton/DC rows, and forbidden `flashinfer` visibility in
  `optimized_kernel_report.json`. The final artifact is the optimized-kernel
  report, not a separate `nanogpt_component_manifest.json`; this is the
  accepted Issue `08` implementation seam.

- [x] **Step 2: Add selected component rules**

  Required full-mode components:

  ```text
  torch
  torch_cuda
  torch_scaled_mm_fp8
  torch_compile_inductor
  torch_distributed_nccl
  triton
  triton_tensor_descriptor
  source_triton_kernels
  source_dc_triton_kernels
  attention_fa3
  attention_fa2
  attention_flex
  attention_fa4
  mlp_triton
  mlp_torch
  tiktoken
  numpy
  datasets
  huggingface_hub
  kernels
  flash_attn
  flashinfer
  ```

  Unselected components still appear with `selected_for_launch=false`;
  forbidden components appear with `forbidden_absent=true`.

  Evidence: `optimized_kernel_certifier.py` emits rows for Torch/CUDA/Triton,
  FP8 `torch._scaled_mm`, TorchInductor cache, NCCL, FA2/FA3/FA4/flex/SDPA
  attention, Triton/Torch MLP, source-local Triton/DC kernels, and forbidden
  `flashinfer`. Selected rows gate `launch_eligible`; unselected unsupported
  rows remain visible but nonblocking.

- [x] **Step 3: Implement FA3/FA2/Flex/FlashInfer rules**

  FA3 requires the selected entrypoint's declared `kernels.get_kernel` target
  and varlen BF16 smoke. For current full `train_gpt.py`, that target is
  `kernels-community/flash-attn3` at version `1`. FA2 requires
  `flash-attn==2.8.3.post1`, source-build/setup evidence, `kernels`
  importability when the source imports `get_kernel`, CUDA synth packages,
  `libcudart` bridge, and varlen BF16 smoke. Flex is diagnostic-only.
  FlashInfer is forbidden unless a future backend verifier is added. FA4 must
  be represented as a named attention backend even when the current runtime has
  no provider; absence is recorded as `support_status=unsupported` with
  evidence, not omitted from the matrix.

  Evidence: FA2, FA3, FA4, flex, torch SDPA, and flashinfer behaviors are
  represented in the certified matrix. The selected FA2/Triton artifact at
  `experiments/modded_nanogpt_b200/results/issue08_kernel_cert_fa2_triton_20260819T083000Z/runtime/optimized_kernel_report.json`
  records `launch_eligible=true`, `blockers=[]`, and digest
  `a789b17eef84c62d386f4f395d32ec0c0924bc261fcd450c07668d17d3882969`.

- [x] **Step 4: Implement source-local Triton rules**

  MLP Triton smoke uses `FusedLinearReLUSquareFunction.apply` with
  `x=[2,16,768]`, `w1=[3072,768]`, `w2=[3072,768]`, backward, and CUDA
  synchronization. DC Triton smoke calls
  `dc_attention_postonly_nodd_correction_add_base_triton` with `B=1`, small
  `T`, `H=6`, `D=128`, `window<=128`, and `seq_lens=[0,T]`.

  Evidence: `optimized_kernel_certifier.py` contains source-local Triton and
  DC Triton probes, and the selected FA2/Triton certification report records
  selected-row success for both source-local kernel classes.

- [x] **Step 5: Verify**

  Unit tests may mock CUDA-heavy smokes; real CUDA smokes must run only through
  rootfs wrappers:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_components.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py
  ```

  Evidence:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_optimized_kernel_certifier.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py'`
  passed with `67 passed in 7.45s`.

## Task 8A: Data Path and Manifest Shard-Root Repair

**Files:**

- Modify: `experiments/modded_nanogpt_b200/run_speedrun.py`
- Modify: `experiments/modded_nanogpt_b200/preflight.py`
- Modify as needed: `experiments/modded_nanogpt_b200/preflight_checklist.md`
- Test: `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py`
- Test: `tests/unit_tests/test_modded_nanogpt_b200_preflight.py`

**Interfaces:**

- Consumes: full manifest files whose shards live under
  `<source>/data/fineweb10B/*.bin`, upstream `train_gpt.py` logic that joins
  `DATA_PATH` with `data/fineweb10B`, and the launched failure artifact
  `lane_b_full_20260816T181747Z`.
- Produces: a pre-launch verifier that blocks doubled `data/data/fineweb10B`
  lookups before telemetry or `torchrun`, plus tests proving absolute and
  repo-relative manifests derive the correct source-visible `DATA_PATH`.

- [x] **Step 1: Add the red regression**

  Test the exact launched failure:

  ```python
  def test_full_launch_rejects_source_visible_doubled_data_path_before_torchrun(tmp_path):
      # Manifest shards under <source>/data/fineweb10B must produce DATA_PATH=<source>.
      # DATA_PATH=<source>/data is invalid because upstream looks for
      # DATA_PATH/data/fineweb10B/fineweb_val_*.bin.
      ...
  ```

  Expected red: current or stale launcher logic reaches the training command or
  computes `DATA_PATH=<source>/data`.

  Evidence: `tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py`
  includes `test_source_data_path_blocker_catches_doubled_data_root` and
  `test_full_launch_rejects_bad_source_visible_data_path_before_torchrun`,
  covering the doubled `data/data/fineweb10B` failure before telemetry or
  `torchrun`.

- [x] **Step 2: Implement the source-visible data path blocker**

  The non-skip full launch path must check, before telemetry and `torchrun`:

  ```text
  ${DATA_PATH}/data/fineweb10B/fineweb_train_*.bin
  ${DATA_PATH}/data/fineweb10B/fineweb_val_*.bin
  ```

  Failure writes `blocker.phase=data_path`, `exit_code.phase=data_path`, and
  `launch_readiness.training_launched=false`.

  Evidence: `run_speedrun.py` implements `_source_data_path_blocker`, calls it
  before optimized-kernel certification, runtime verification, telemetry, and
  `torchrun`, and writes `blocker.phase=data_path`, `exit_code.phase=data_path`,
  `launch_readiness.training_launched=false`, and a parsed summary on failure.

- [x] **Step 3: Verify manifest-derived roots**

  Add tests for both absolute and repo-relative manifest shard paths:

  ```text
  <source>/data/fineweb10B/fineweb_train_000000.bin -> DATA_PATH=<source>
  experiments/.../source/data/fineweb10B/fineweb_train_000000.bin -> DATA_PATH=/workspace/torchtitan/experiments/.../source
  ```

  Evidence: `test_data_path_comes_from_absolute_manifest_shard_root` and
  `test_data_path_comes_from_repo_relative_manifest_shard_root` cover absolute
  and repo-relative manifest shard roots.

- [x] **Step 4: Update operator checklist**

  Checklist wording must say `DATA_PATH` is the root containing `data/`, not the
  `data/` directory itself, and that a doubled `data/data/fineweb10B` lookup is
  a no-go condition.

  Evidence: `experiments/modded_nanogpt_b200/preflight_checklist.md` now states
  that runner-derived `DATA_PATH` is the upstream-compatible root containing
  `data/`, and explicitly lists `data/data/fineweb10B` as a no-go condition.

- [x] **Step 5: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py
  python3 experiments/modded_nanogpt_b200/verify_static.py
  ```

  Expected: focused tests pass; no rootfs/GPU/full launch command is required
  for this repair.

  Evidence:
  `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_optimized_kernel_certifier.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py'`
  passed with `67 passed in 7.45s`.

## Task 8B: Launcher and Launch-Readiness Integration

**Precondition:** Task 8B starts only after Task 8A proves the current
manifest-derived `DATA_PATH` makes upstream `train_gpt.py` see both train and
validation shards. Task 8A now covers that precondition with focused
manifest-root tests and a doubled `data/data/fineweb10B` no-go regression; do
not run another full launch if a material data-path input changes without
rerunning those checks.

**Files:**

- Modify: `experiments/modded_nanogpt_b200/run_speedrun.py`
- Modify: `experiments/modded_nanogpt_b200/run_speedrun.sh`
- Modify: `experiments/modded_nanogpt_b200/parse_log.py`
- Modify: `experiments/modded_nanogpt_b200/summarize.py`
- Add tests under existing runner/parser/summarizer test files.

**Interfaces:**

- Consumes: runtime verifier, optimized-kernel report, active-job scan, preflight
  report, data manifest, source status, and authorization.
- Produces: final `launch_readiness.json` and a training subprocess environment
  whose digest matches the verified command env.

- [x] **Step 1: Add red tests for stale/missing verifier reports**

  Full launch must stop before telemetry and `torchrun` if any current-version
  report is missing, stale for `run_id`/`attempt_id`, schema-invalid, or
  `ok=false`.

  2026-08-19: Complete for the runtime verification sidecar. A parametrized
  red/green regression covers missing, malformed, `ok=false`, and stale-digest
  `runtime/runtime_verification.json` cases. Authorized full launches now stop
  before telemetry and `torchrun` with `blocker.phase=runtime_verification`.

- [x] **Step 2: Add env digest matching**

  `command.env.json`, `runtime_verification.json`, and
  `launch_readiness.json` must agree on the effective training environment
  digest. The legacy `command.env` remains a human-readable redacted sidecar.

  2026-08-19: Complete for runner-produced attempts. `run_speedrun.py` now
  writes redacted structured `command.env.json`, attempt-local
  `runtime/runtime_verification.json`, and matching `command_env_digest` fields
  in `launch_readiness.json`. The digest is over the redacted command
  environment, preserving the existing secret-redaction boundary.

- [x] **Step 3: Preserve existing guards**

  Keep result-directory reuse, result-local cache dirs, data-path visibility,
  active-job scan, authorization, variant patch classification, SHA, NCCL, and
  known-stall gates fail-closed.

  2026-08-19: The full runner suite passed after adding the env-digest and
  runtime-verification gates, preserving the existing guard behavior.

- [x] **Step 4: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py
  ```

  2026-08-19 rootfs verification:

  ```text
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py
  -> 48 passed in 1.22s

  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py
  -> 122 passed in 1.33s

  python3 -m py_compile experiments/modded_nanogpt_b200/run_speedrun.py
  -> passed

  python3 experiments/modded_nanogpt_b200/verify_static.py
  -> Static verification passed for 16 file(s).
  ```

## Task 9: Non-Launch Prerequisite Refresh

**Files:**

- Generated only under `experiments/modded_nanogpt_b200/results/`
- Update as evidence changes: `.scratch/modded-nanogpt-b200/completion_audit.md`

**Interfaces:**

- Consumes: updated runtime and launch verifiers.
- Produces: a fresh full-mode skip-run or authority-guard prerequisite artifact,
  plus refreshed `run_index.json`.

- [x] **Step 1: Run active-job scan**

  Host command:

  ```bash
  experiments/modded_nanogpt_b200/check_active_jobs.sh --active-jobs-output experiments/modded_nanogpt_b200/results/<run_id>/active_jobs.json
  ```

  Expected: wrapper re-enters rootfs; report has `ok=true` and
  `active_job_count=0`.

  Evidence: `lane_b_full_skiprun_ipv4_nccl_refresh_20260819T105321Z` preserved a
  clean prelaunch active-job scan and the attempt-local scan in `summary.json`
  records `ok=true`, `active_job_count=0`, and `active_jobs=[]`.

- [x] **Step 2: Run full-mode skip-run gate**

  Use the latest full manifest and the B200-compatible FA2/Triton
  configuration. Keep `--lane B` because current CLI/schema artifacts still
  require the legacy compatibility field:

  ```bash
  experiments/modded_nanogpt_b200/run_speedrun.sh \
    --mode full \
    --lane B \
    --run-id <run_id> \
    --attempt-id <attempt_id> \
    --arm B0 \
    --source experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa \
    --data-manifest experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json \
    --result-dir experiments/modded_nanogpt_b200/results/<run_id> \
    --attention-backend fa2 \
    --mlp-backend triton \
    --verify-sha \
    --skip-run
  ```

  Expected: `preflight_ok=true`, `training_launched=false`, and
  `skip_run=true`.

  Evidence: the fixed rootfs gate exited `0` and wrote
  `launch_readiness.json` with `ready_to_launch=true`,
  `training_launched=false`, `skip_run=true`, `blocked_by=[]`,
  `classification.claim_eligible=true`, matching `command_env_digest` and
  `runtime_verification.command_env_digest`, 2x B200 GPU evidence, and
  launch-eligible optimized-kernel evidence.

- [x] **Step 3: Refresh index**

  ```bash
  experiments/modded_nanogpt_b200/summarize.sh \
    --results-root experiments/modded_nanogpt_b200/results \
    --output experiments/modded_nanogpt_b200/results/run_index.json
  ```

  Expected: the index surfaces prerequisite evidence but still reports zero
  baseline attempts.

  Historical evidence: refreshed `experiments/modded_nanogpt_b200/results/run_index.json`
  reports `total_attempts=62`, `baseline_stats.count=0`,
  `len(launch_prerequisite_attempts)=3`, and
  `len(launch_ready_attempts)=0`. The latest prerequisite row points to
  `lane_b_full_skiprun_ipv4_nccl_refresh_20260819T105321Z/summary.json`.

  Later strict runtime-env handoff refresh supersedes that row as current
  prerequisite evidence. A fresh rebuilt run index reports
  `total_attempts=63`, `baseline_stats.count=0`,
  `len(launch_prerequisite_attempts)=1`, and
  `len(launch_ready_attempts)=0`; the latest current prerequisite row points to
  `lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z/summary.json`.

## Task 10: First Authorized B200-Compatible Full Baseline

**Files:**

- Generated only under `experiments/modded_nanogpt_b200/results/<run_id>/`
- Update as evidence changes: `.scratch/modded-nanogpt-b200/completion_audit.md`

**Interfaces:**

- Consumes: latest prerequisite evidence and a trusted user request containing
  `launch-full-b200`.
- Produces: the first non-skip B200-compatible local setup full attempt, either
  successful final validation evidence or a preserved blocker.

- [ ] **Step 1: Confirm authority and no duplicate blocker**

  Proceed only when the trusted user message itself contains
  `launch-full-b200`. If no material input changed and that token is absent,
  stop at the authorization boundary.

- [ ] **Step 2: Launch after trusted-request authority**

  ```bash
  experiments/modded_nanogpt_b200/run_speedrun.sh \
    --mode full \
    --lane B \
    --run-id <run_id> \
    --attempt-id <attempt_id> \
    --arm B0 \
    --source experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa \
    --data-manifest experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json \
    --result-dir experiments/modded_nanogpt_b200/results/<run_id> \
    --attention-backend fa2 \
    --mlp-backend triton \
    --verify-sha \
    --launch-authorization=launch-full-b200
  ```

  Expected: if gates pass, `training_launched=true` appears immediately before
  `torchrun`; telemetry is started; the attempt is summarized even on failure.
  This command template is valid only after Step 1 confirms that the trusted
  user message contains `launch-full-b200`.
  If using the no-argument convenience wrapper
  `experiments/modded_nanogpt_b200/launch_nanogpt_2gpu_full_rootfs.sh` instead
  of calling `run_speedrun.sh` directly, set
  `MODDED_NANOGPT_2GPU_FULL_LAUNCH_AUTHORIZATION=launch-full-b200` only after
  the trusted user request includes `launch-full-b200`; without that marker the
  wrapper exits before active-job scan or `run_speedrun.sh`.

- [ ] **Step 3: Babysit the run**

  Watch `run.log`, telemetry sidecars, no-output watchdog state, and stop
  snapshots. Do not detach and forget a long full job.

- [ ] **Step 4: Classify outcome**

  Successful baseline requires final validation, `val_loss <= 3.28`,
  upstream `train_time`, training exit code `0`, conservative
  B200-compatible local setup claim classification, and all baseline evidence
  gates. Any failure records the first blocker phase and stops additional
  equivalent launches until repaired.

## Task 11: Real-Run Repair Loop

**Files:**

- Modify only the narrow failing harness/runtime/source files indicated by
  evidence.
- Add focused tests for every repair.
- Update completion audit after verified evidence.

**Interfaces:**

- Consumes: first failing full-run artifact.
- Produces: one repaired gate or one classified no-go condition.

- [ ] **Step 1: Identify the first failing phase**

  Prefer structured artifacts in this order:

  ```text
  blocker.json
  exit_code.json
  launch_readiness.json
  runtime/runtime_verification.json
  preflight_report.json
  summary.json
  analysis.md
  run.log
  telemetry/stop_snapshot.json
  ```

- [ ] **Step 2: Write a focused red test or diagnostic**

  Use unit tests for parser/summary/launcher/schema bugs. Use rootfs diagnostic
  wrappers for package, CUDA, FA, Triton, NCCL, or source-runtime bugs.

- [ ] **Step 3: Fix one cause**

  Preserve claim classification and the legacy lane artifact field. Edits to
  faithful-upstream source are not allowed. B200-compatible source edits require
  a new variant patch diff and classification.

- [ ] **Step 4: Re-run the exact failing gate**

  Do not jump directly to another full launch if a cheaper preflight or
  diagnostic command proves the repair.

## Task 12: Sequential Two-GPU Trial Repeatability

**Files:**

- Generated only under `experiments/modded_nanogpt_b200/results/`
- Update as evidence changes: `.scratch/modded-nanogpt-b200/completion_audit.md`

**Interfaces:**

- Consumes: one successful B200-compatible two-GPU trial.
- Produces: sequential two-GPU follow-up evidence authorized one attempt at a
  time, or a preserved blocker that stops further attempts.

- [ ] **Step 1: Freeze the trial configuration**

  Record claim label, legacy lane field, arm, source digest, data manifest
  digest, runtime store ID, Python env digest, tool env digest,
  optimized-kernel report contract, command env digest, and launch command
  template. The template must expose exactly two B200 GPUs to the rootfs and
  use a matching two-rank distributed launch.

- [ ] **Step 2: Run attempts one at a time**

  Each attempt gets a unique `run_id` and `attempt_id`. Run active-job scan
  before each launch. Do not reuse result directories. Do not launch a second
  NanoGPT attempt until the previous attempt has stopped and its summary has
  been refreshed.

- [ ] **Step 3: Stop on first new blocker**

  If any attempt fails a prerequisite, training, final validation, telemetry, or
  summary gate, stop and classify. Do not spend additional GPU budget on a
  deterministic repeat without a new user authorization.

- [ ] **Step 4: Refresh run index after each attempt**

  Use the rootfs-aware summarizer. Baseline count must come from
  `run_index.json`, not manual counting.

- [ ] **Step 5: Final trial acceptance**

  The trial is complete when the latest authorized two-GPU row has non-skip
  `training_launched=true`, exit code `0`, final validation, `val_loss <= 3.28`,
  and all rootfs/data/NCCL/two-GPU/source evidence. A ten-run or eight-GPU
  campaign is out of scope unless separately authorized.

## Task 13: Optional Faithful-Upstream FA3 Revisit

**Files:**

- Modify only docs/checklists unless a new FA3 runtime path is available.
- Do not modify faithful-upstream source.

**Interfaces:**

- Consumes: a new `kernels-community/flash-attn3` or equivalent FA3 artifact
  with B200 `sm_100` support.
- Produces: either a faithful-upstream attempt using the legacy `lane=A`
  artifact class or an updated FA3 blocker.

- [ ] **Step 1: Verify material input changed**

  A faithful-upstream revisit requires a new FA3 package/artifact, new rootfs
  stack, or new B200-compatible FA3 evidence. Do not rerun the known failed FA3
  path just to reconfirm it.

- [ ] **Step 2: Run FA3 preflight only**

  Use full-mode preflight and optimized-kernel report. Training remains
  unauthorized unless explicit launch authority is present.

- [ ] **Step 3: Preserve faithful-upstream cleanliness**

  `git status --short` inside the upstream source must be empty before and after
  any faithful-upstream attempt.

## Task 14: Optional Ablation Matrix

**Files:**

- Modify: `.scratch/modded-nanogpt-b200/issues/04-b200-ablation-matrix.md`
- Generated only under `experiments/modded_nanogpt_b200/results/`

**Interfaces:**

- Consumes: accepted faithful-upstream or B200-compatible baseline artifact.
- Produces: one-variable-at-a-time B200 ablation results.

- [ ] **Step 1: Declare baseline arm**

  Use the faithful-upstream arm only if that artifact class has a successful
  baseline. Otherwise use the nearest source-equivalent B200-compatible
  baseline.

- [ ] **Step 2: Define one variable per arm**

  Allowed examples: FA3 versus FA2 when FA3 works, FP8 diagnostic changes,
  NCCL env changes, cold versus warm compile cache, runtime stack comparison.

- [ ] **Step 3: Require matched evidence**

  Every ablation records source, patch diff, data manifest, runtime env,
  hardware, final validation loss, upstream `train_time`, shell wall-clock, and
  memory.

- [ ] **Step 4: Reject uncontrolled comparisons**

  If source, data, model, optimizer, schedule, validation count, or runtime stack
  differs outside the declared variable, label the arm diagnostic or split it
  into a separate ablation.

## Task 15: Documentation, Review, and Handoff

**Files:**

- Modify: `.scratch/modded-nanogpt-b200/completion_audit.md`
- Modify as needed: `experiments/modded_nanogpt_b200/preflight_checklist.md`
- Modify as needed: `.scratch/modded-nanogpt-b200/issues/*.md`

**Interfaces:**

- Consumes: latest implementation and run evidence.
- Produces: concise handoff state for the next agent or human operator.

- [x] **Step 1: Request independent review**

  Use clean-context review for any implementation slice that changes launch
  authority, schema, rootfs, dependency, source provenance, data manifest,
  parser, summary, or baseline inclusion behavior.

  Issue `15` is documentation-only. It does not change launch authority,
  schema, rootfs, dependency, source provenance, data manifest, parser, summary,
  or baseline inclusion behavior, so independent code review is not required
  for this slice.

- [x] **Step 2: Run focused verification**

  Minimum non-GPU verification before handoff:

  ```bash
  python3 experiments/modded_nanogpt_b200/verify_static.py
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py
  ```

  Add schema/runtime/component tests once those files exist.

  2026-08-19 rootfs verification:

  ```text
  python3 experiments/modded_nanogpt_b200/verify_static.py
  -> Static verification passed for 16 file(s).

  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py tests/unit_tests/test_modded_nanogpt_b200_optimized_kernel_certifier.py tests/unit_tests/test_modded_nanogpt_b200_performance_probe.py
  -> 144 passed in 4.54s
  ```

  Current final non-launch verification:

  ```text
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python experiments/modded_nanogpt_b200/verify_static.py && pytest -q tests/unit_tests/test_modded_nanogpt_b200_*.py tests/unit_tests/test_execution_rootfs_selection_shell.py tests/unit_tests/test_rootfs_bwrap_plan.py tests/unit_tests/test_rootfs_build_store_shell.py tests/unit_tests/test_rootfs_runtime_env_shell.py'
  -> Static verification passed for 114 file(s).
  -> 410 passed, 2 skipped

  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_modded_nanogpt_b200_tracker.py'
  -> 49 passed
  ```

  Latest continuation audits:

  ```text
  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_modded_nanogpt_b200_optimized_kernel_certifier.py tests/unit_tests/test_modded_nanogpt_b200_performance_probe.py'
  -> 13 passed in 3.03s

  In-memory summarize.build_index(Path("experiments/modded_nanogpt_b200/results"))
  -> total_attempts=63
  -> baseline_stats.count=0
  -> launch_prerequisite_attempts.count=4
  -> launch_ready_attempts.count=0

  validate_attempt_dir(lane_b_full_skiprun_runtime_env_refresh_20260819T131652Z)
  -> ok=true, sidecar_count=24, failed_sidecar_count=0

  ./experiments/modded_nanogpt_b200/check_active_jobs.sh
  -> ok=true, active_job_count=0, ignored_match_count=0

  scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_modded_nanogpt_b200_tracker.py'
  -> 49 passed
  ```

  Explicit-file Pyrefly over the 31-file static Python surface reported `0 errors`, with the known `/workspace/pytorch` search-path
  warning. `bash -n` over NanoGPT/rootfs
  shell wrappers, JSON/TOML parsing for touched configs/schemas, text/lock
  hygiene for runtime dependency files inside rootfs, and `git diff --check`
  also exited `0`. A later rootfs all-files pre-commit run passed with only the
  protected-branch hook skipped:
  `SKIP=no-commit-to-branch pre-commit run --all-files`.
  No preflight, skip-run refresh, summarizer refresh, GPU probe, matrix
  execution, generated artifact cleanup, staging, commit, or full launch was
  run.

- [x] **Step 3: Update current-state docs**

  Completion audit must state:

  - newest relevant result path;
  - current `baseline_stats.count`;
  - current prerequisite and launch-ready counts;
  - whether training launched;
  - first blocker phase;
  - exact next action and required authority.

- [x] **Step 4: Preserve dirty-tree boundaries**

  Do not stage, commit, clean, reset, or delete generated artifacts unless the
  user explicitly authorizes that action.

  No git staging, commit, cleanup, reset, generated artifact mutation, or
  training launch was performed for issue `15`.

## Stop Conditions

Stop and report instead of continuing when any condition holds:

- the trusted user request does not contain `launch-full-b200` and the next
  meaningful action is a non-skip full launch;
- active-job scan reports another real target job or scan failure;
- a full-mode verifier report is missing, stale, schema-invalid, or `ok=false`;
- data manifest lacks full 900M shape, pinned source commit, or SHA evidence;
- selected optimized-kernel report does not prove the selected attention/MLP/DC
  Triton path;
- rootfs sentinel or canonical env evidence is missing;
- source provenance or B200-compatible patch classification fails;
- final validation is absent or metrics exceed the claim threshold;
- a deterministic runtime blocker appears in a launched attempt.

## Current Next Action

Issue tickets `10`-`15` and `08` are complete for the current non-launch
foundation. Master-plan Tasks `10`-`14` remain launch- or baseline-dependent.
The RSI-control layer now has typed schema loading, plan materialization,
sequential matrix-runner dry/mocked execution coverage, advisory matrix-level
RSI evidence, a dry-run-by-default matrix CLI with explicit `--execute` gating,
diagnostic performance-probe evidence, a launch-eligible optimized-kernel
certification report, and a current handoff record. The next full launch still
requires the trusted user request itself to contain `launch-full-b200`. With
that authority, run Task 10 using the B200-compatible claim class, legacy
`lane=B`, arm `B0`, FA2 attention, Triton MLP, full manifest
`experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`,
and the lower-level `--launch-authorization=launch-full-b200` marker.

Unchecked task blocker map:

- Task 10 is blocked by the missing trusted `launch-full-b200` request.
- Task 11 is blocked until Task 10 produces a real full-run artifact.
- Task 12 is blocked until one B200-compatible two-GPU trial succeeds.
- Task 13 is blocked until a material FA3/B200 kernel input changes.
- Task 14 is blocked until an accepted faithful-upstream or B200-compatible
  baseline artifact exists.

Latest completion-audit decision:

- The prompt-to-artifact audit found no remaining unblocked non-launch
  implementation or handoff issue to repair.
- Issues `04`, `06`, and `09` remain the only blocked issue tickets; each
  blocker traces to the missing trusted launch authority or missing accepted
  non-skip baseline.
- Do not mark the active thread goal complete until an authorized non-skip
  two-GPU Lane B full attempt is launched, stopped, parsed, summarized,
  classified, and accepted as a baseline, or until the user explicitly
  redefines completion short of that baseline.
