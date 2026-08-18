# Modded NanoGPT B200 Master Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or the repo-local clean-context
> subagent workflow before executing implementation, audit, verification, or
> launch tasks from this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: active master plan
Last updated: 2026-08-18

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
  launch requires explicit `--launch-authorization=launch-full-b200` in the
  trusted user request stream.
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
- Runtime component design exists in `rootfs_runtime_env_spec.md`, including
  rootfs layout, uv, mise, schemas, verifiers, PyTorch/nanoGPT components, and
  optimized-kernel report design.

Current blocked slices:

- Issue `06`: the first B200-compatible local setup baseline is blocked by a
  verified launched full-attempt failure. The latest authorized launched legacy
  `lane=B` attempt reached `torchrun` and failed because training produced no
  output for the 600-second no-output watchdog window during model compile and
  kernel warmup.
- Issue `08`: optimized kernel certification is not yet first-class. The
  current harness has one-off FA2, FA3, Triton, torch, NCCL, and component
  smokes, but no single certified matrix report that proves the selected launch
  tuple is supported/pass while preserving structured nonblocking evidence for
  known-unsupported unselected FA3, FA4, and diagnostic fallback paths.
- Issue `09`: the first two-GPU trial did not launch. It selected idle GPUs
  `0,1` and wrote a clean prelaunch active-job scan, but no non-skip training
  attempt began because the request did not include the explicit
  `launch-full-b200` authorization token. The next iteration must start from a
  fresh result directory after rechecking idle GPUs and active jobs.
- Issue `04`: ablation work is blocked by an accepted faithful-upstream or
  B200-compatible local setup baseline.
- Faithful-upstream full reproduction, currently represented by legacy `lane=A`,
  is blocked by FA3 B200 kernel support in the current runtime.
- The earlier production target of 10 successful full jobs is superseded by the
  two-GPU sequential trial policy. No repeatability campaign should start until
  a two-GPU trial succeeds and the user explicitly authorizes broader use.

Current evidence to cite before rerunning anything:

- Latest full manifest:
  `experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`
- Latest non-launch B200-compatible prerequisite, using legacy `lane=B`:
  `experiments/modded_nanogpt_b200/results/lane_b_full_skiprun_refresh_20260816T111056Z/`
- Latest launched B200-compatible attempt, using legacy `lane=B`:
  `experiments/modded_nanogpt_b200/results/lane_b_full_20260818T014833Z_attempt_001/`
- Latest two-GPU prelaunch-only trial directory:
  `experiments/modded_nanogpt_b200/results/lane_b_two_gpu_trial_20260818T041445Z_attempt_001/`
- Current run index:
  `experiments/modded_nanogpt_b200/results/run_index.json`
- Current completion audit:
  `.scratch/modded-nanogpt-b200/completion_audit.md`

Current numeric state from current artifacts:

- `baseline_stats.count=0`
- `total_attempts=26`
- `launch_prerequisite_attempts=1` before the launched failure is excluded from
  baseline stats
- `launch_ready_attempts=0`
- The one prerequisite row is a skip-run dry gate, not production evidence.
- The launched attempt has `training_launched=true`, `skip_run=false`,
  `preflight_ok=true`, and training exit code `124`; it is not baseline
  evidence.
- The concrete launched-attempt blocker is
  `training produced no output for 600 seconds`.

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
5. **Launch Gate:** reconcile runtime verifier, launcher, readiness,
   summarizer, active-job scan, authorization, two-GPU allocation, and
   non-launch prerequisite evidence.
6. **First Two-GPU Trial:** run the next authorized B200-compatible attempt only
   after the data-path gate and launch gate pass, with exactly two visible B200
   GPUs and no concurrent NanoGPT attempts, then preserve or repair the first
   real blocker.
7. **Repeatability:** do not run a production campaign by default. After one
   two-GPU trial succeeds, run only sequential follow-up attempts that the user
   explicitly authorizes, refreshing the index after each and stopping on the
   first new blocker.
8. **Optional Claim Refinement and Ablations:** revisit faithful-upstream FA3
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

- [ ] **Step 1: Read authoritative context**

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

- [ ] **Step 2: Update the tracker only when state changes**

  Edit `completion_audit.md` only after new evidence exists. Do not refresh
  merely to restate the same launch-authorization blocker.

- [ ] **Step 3: Run static text checks**

  Run:

  ```bash
  python3 experiments/modded_nanogpt_b200/verify_static.py
  rg -n -F '\\n+' .scratch/modded-nanogpt-b200 experiments/modded_nanogpt_b200
  ```

  Expected: static verifier passes; `rg` prints no malformed patch artifacts.

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

- [ ] **Step 1: Add a failing unit seam for canonical env generation**

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

- [ ] **Step 2: Implement `runtime_env.sh`**

  It must compute paths from the repo root and optional environment overrides,
  reject paths outside the repo or runtime state root, and expose shell
  functions for env export and bwrap bind construction.

- [ ] **Step 3: Add managed rootfs store support**

  `build_rootfs.sh` must stage rootfs content, pre-create bind targets, write an
  ownership marker and `manifest.json`, and atomically activate a selected
  store entry. Legacy `scripts/rootfs/rootfs` remains diagnostic unless it has a
  schema-valid legacy manifest with `mutable_rootfs_allowed=false`.

- [ ] **Step 4: Add network mode**

  `enter_rootfs.sh` must support:

  ```text
  TORCHTITAN_ROOTFS_NETWORK=networked|offline
  ```

  Full launches use offline mode unless explicitly diagnostic.

- [ ] **Step 5: Verify without GPU launch**

  Run host-side static/unit checks only:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_rootfs_runtime_env.py
  bash -n scripts/rootfs/runtime_env.sh scripts/rootfs/build_rootfs.sh scripts/rootfs/enter_rootfs.sh
  ```

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

- [ ] **Step 1: Add red tests for dependency policy**

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

- [ ] **Step 2: Implement sync wrapper**

  The wrapper re-enters rootfs and runs:

  ```bash
  uv venv --python "$(command -v python)" --system-site-packages /project/venvs/b200-runtime
  uv pip install --python /project/venvs/b200-runtime/bin/python --no-deps --require-hashes -r experiments/modded_nanogpt_b200/runtime/requirements.lock
  ```

  If `requirements.lock` is not populated yet, allow the documented temporary
  `requirements.direct.txt` fallback only in networked setup mode.

- [ ] **Step 3: Gate active Python**

  Full preflight and full launch must use `/project/venvs/b200-runtime/bin/python`
  once the sync wrapper exists. Host-side direct Python remains blocked by
  `cli_guard.py`.

- [ ] **Step 4: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py
  python3 experiments/modded_nanogpt_b200/verify_static.py
  ```

## Task 4: mise Tool Runtime

**Files:**

- Create: `experiments/modded_nanogpt_b200/runtime/mise.toml`
- Create: `experiments/modded_nanogpt_b200/runtime/sync_tools.sh`
- Modify: `scripts/rootfs/build_rootfs.sh`
- Add tests under `tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py`

**Interfaces:**

- Consumes: rootfs canonical env and runtime state root.
- Produces: `/project/mise` tool state and `tool_env_report.json`.

- [ ] **Step 1: Add red tests for tool path ownership**

  Assert `MISE_DATA_DIR`, `MISE_CACHE_DIR`, and `MISE_CONFIG_DIR` resolve under
  `/project` or the checked-in runtime directory, never under host home.

- [ ] **Step 2: Install mise in the rootfs build**

  Pin mise by version in the rootfs manifest. Do not curl-install mise during a
  training attempt.

- [ ] **Step 3: Add tool sync wrapper**

  Wrapper commands:

  ```bash
  mise trust experiments/modded_nanogpt_b200/runtime/mise.toml
  mise install --yes --cd experiments/modded_nanogpt_b200/runtime
  mise exec --cd experiments/modded_nanogpt_b200/runtime -- shellcheck --version
  ```

- [ ] **Step 4: Verify**

  Run:

  ```bash
  bash -n experiments/modded_nanogpt_b200/runtime/sync_tools.sh
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_runtime_tools.py
  ```

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

- [ ] **Step 1: Add red tests for required schemas**

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

- [ ] **Step 2: Implement standard-library bootstrap validation**

  The first implementation may validate required keys, types, schema version,
  and top-level unknown fields without a third-party `jsonschema` dependency.

- [ ] **Step 3: Wire writers to schemas**

  Existing writers should call validation before atomic writes for new
  current-version artifacts.

- [ ] **Step 4: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_schemas.py
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py
  ```

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

- [ ] **Step 1: Add red tests for acyclic verifier flow**

  A wrapper must be unable to write `ready_to_launch=true` unless
  `runtime_verification.ok=true`, current schema versions are present, and
  `training_launch_allowed=true`.

- [ ] **Step 2: Implement repo-generic verifier**

  `scripts/rootfs/verify_runtime_env.py` checks rootfs sentinel, cwd, store ID,
  mount targets, network mode, canonical env, device/driver visibility,
  filesystem capacity, Python path, and Torch/CUDA/Triton drift.

- [ ] **Step 3: Implement experiment verifier**

  `experiments/modded_nanogpt_b200/runtime/verify_runtime.py` joins Python,
  tool, filesystem, source, data, hardware, NCCL, backend, active-job,
  authorization, command env, and preflight evidence.

- [ ] **Step 4: Preserve partial failure reports**

  Every failure includes `phase`, `message`, `expected`, `actual`, and
  `artifact_path` when applicable, and writes partial verifier output before
  returning exit `21`.

- [ ] **Step 5: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_runtime_verifier.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py
  ```

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

- [ ] **Step 1: Add red source-inventory tests**

  Tests must prove the manifest distinguishes:

  - selected `train_gpt.py`;
  - unselected `train_gpt_medium.py`;
  - source-local `triton_kernels.py`;
  - source-local `dc_triton_kernels.py` when imported;
  - `data/cached_fineweb10B.py`;
  - `data/fineweb.py`;
  - current absence of `flashinfer`, `flash_infer`, and `flash-infer`.

- [ ] **Step 2: Add selected component rules**

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

- [ ] **Step 3: Implement FA3/FA2/Flex/FlashInfer rules**

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

- [ ] **Step 4: Implement source-local Triton rules**

  MLP Triton smoke uses `FusedLinearReLUSquareFunction.apply` with
  `x=[2,16,768]`, `w1=[3072,768]`, `w2=[3072,768]`, backward, and CUDA
  synchronization. DC Triton smoke calls
  `dc_attention_postonly_nodd_correction_add_base_triton` with `B=1`, small
  `T`, `H=6`, `D=128`, `window<=128`, and `seq_lens=[0,T]`.

- [ ] **Step 5: Verify**

  Unit tests may mock CUDA-heavy smokes; real CUDA smokes must run only through
  rootfs wrappers:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_components.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py
  ```

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

- [ ] **Step 1: Add the red regression**

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

- [ ] **Step 2: Implement the source-visible data path blocker**

  The non-skip full launch path must check, before telemetry and `torchrun`:

  ```text
  ${DATA_PATH}/data/fineweb10B/fineweb_train_*.bin
  ${DATA_PATH}/data/fineweb10B/fineweb_val_*.bin
  ```

  Failure writes `blocker.phase=data_path`, `exit_code.phase=data_path`, and
  `launch_readiness.training_launched=false`.

- [ ] **Step 3: Verify manifest-derived roots**

  Add tests for both absolute and repo-relative manifest shard paths:

  ```text
  <source>/data/fineweb10B/fineweb_train_000000.bin -> DATA_PATH=<source>
  experiments/.../source/data/fineweb10B/fineweb_train_000000.bin -> DATA_PATH=/workspace/torchtitan/experiments/.../source
  ```

- [ ] **Step 4: Update operator checklist**

  Checklist wording must say `DATA_PATH` is the root containing `data/`, not the
  `data/` directory itself, and that a doubled `data/data/fineweb10B` lookup is
  a no-go condition.

- [ ] **Step 5: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py
  python3 experiments/modded_nanogpt_b200/verify_static.py
  ```

  Expected: focused tests pass; no rootfs/GPU/full launch command is required
  for this repair.

## Task 8B: Launcher and Launch-Readiness Integration

**Precondition:** Task 8B starts only after Task 8A proves the current
manifest-derived `DATA_PATH` makes upstream `train_gpt.py` see both train and
validation shards. Do not run another full launch while the doubled
`data/data/fineweb10B` failure remains unverified.

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

- [ ] **Step 1: Add red tests for stale/missing verifier reports**

  Full launch must stop before telemetry and `torchrun` if any current-version
  report is missing, stale for `run_id`/`attempt_id`, schema-invalid, or
  `ok=false`.

- [ ] **Step 2: Add env digest matching**

  `command.env.json`, `runtime_verification.json`, and
  `launch_readiness.json` must agree on the effective training environment
  digest. The legacy `command.env` remains a human-readable redacted sidecar.

- [ ] **Step 3: Preserve existing guards**

  Keep result-directory reuse, result-local cache dirs, data-path visibility,
  active-job scan, authorization, variant patch classification, SHA, NCCL, and
  known-stall gates fail-closed.

- [ ] **Step 4: Verify**

  Run:

  ```bash
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py
  ```

## Task 9: Non-Launch Prerequisite Refresh

**Files:**

- Generated only under `experiments/modded_nanogpt_b200/results/`
- Update as evidence changes: `.scratch/modded-nanogpt-b200/completion_audit.md`

**Interfaces:**

- Consumes: updated runtime and launch verifiers.
- Produces: a fresh full-mode skip-run or authority-guard prerequisite artifact,
  plus refreshed `run_index.json`.

- [ ] **Step 1: Run active-job scan**

  Host command:

  ```bash
  experiments/modded_nanogpt_b200/check_active_jobs.sh --active-jobs-output experiments/modded_nanogpt_b200/results/<run_id>/active_jobs.json
  ```

  Expected: wrapper re-enters rootfs; report has `ok=true` and
  `active_job_count=0`.

- [ ] **Step 2: Run full-mode skip-run gate**

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

- [ ] **Step 3: Refresh index**

  ```bash
  experiments/modded_nanogpt_b200/summarize.sh \
    --results-root experiments/modded_nanogpt_b200/results \
    --output experiments/modded_nanogpt_b200/results/run_index.json
  ```

  Expected: the index surfaces prerequisite evidence but still reports zero
  baseline attempts.

## Task 10: First Authorized B200-Compatible Full Baseline

**Files:**

- Generated only under `experiments/modded_nanogpt_b200/results/<run_id>/`
- Update as evidence changes: `.scratch/modded-nanogpt-b200/completion_audit.md`

**Interfaces:**

- Consumes: latest prerequisite evidence and explicit user launch authority.
- Produces: the first non-skip B200-compatible local setup full attempt, either
  successful final validation evidence or a preserved blocker.

- [ ] **Step 1: Confirm authority and no duplicate blocker**

  Proceed only when the trusted user message explicitly authorizes a full B200
  launch or supplies the launch token. If no material input changed and no token
  is present, stop at the authorization boundary.

- [ ] **Step 2: Launch with explicit token**

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

- [ ] **Step 1: Request independent review**

  Use clean-context review for any implementation slice that changes launch
  authority, schema, rootfs, dependency, source provenance, data manifest,
  parser, summary, or baseline inclusion behavior.

- [ ] **Step 2: Run focused verification**

  Minimum non-GPU verification before handoff:

  ```bash
  python3 experiments/modded_nanogpt_b200/verify_static.py
  python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py
  ```

  Add schema/runtime/component tests once those files exist.

- [ ] **Step 3: Update current-state docs**

  Completion audit must state:

  - newest relevant result path;
  - current `baseline_stats.count`;
  - current prerequisite and launch-ready counts;
  - whether training launched;
  - first blocker phase;
  - exact next action and required authority.

- [ ] **Step 4: Preserve dirty-tree boundaries**

  Do not stage, commit, clean, reset, or delete generated artifacts unless the
  user explicitly authorizes that action.

## Stop Conditions

Stop and report instead of continuing when any condition holds:

- explicit full-launch authorization is absent and the next meaningful action is
  a non-skip full launch;
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

Immediate next action is Task 8A: verify and, if needed, repair the
manifest-derived `DATA_PATH` contract that caused
`lane_b_full_20260816T181747Z` to fail with a doubled
`data/data/fineweb10B` lookup. This is non-launch work and does not require GPU
budget.

After Task 8A passes and prerequisite evidence is refreshed, the next full
launch still requires explicit full-launch authorization. With that authority,
run Task 10 using the B200-compatible claim class, legacy `lane=B`, arm `B0`,
FA2 attention, Triton MLP, full manifest
`experiments/modded_nanogpt_b200/results/full_manifest_refresh_20260816T111021Z/data_manifest.json`,
and `--launch-authorization=launch-full-b200`.
