# Bwrap Rootfs Runtime Environment Spec

Status: implemented-through-non-launch-foundation

## Intent

Define the production runtime environment for B200 training programs that run
through TorchTitan's bwrap rootfs. This spec covers the host directory layout,
the in-rootfs project layout, package management, writable state, cache policy,
scratch and shared-memory sizing, lifecycle commands, and verification gates.

The immediate consumer is `experiments/modded_nanogpt_b200`, but the contract is
intended to be reusable by other repo-local GPU research programs that need the
same rootfs boundary.

## Authority and Scope

Authority order for this effort:

1. `.scratch/modded-nanogpt-b200/spec.md`
2. This file
3. `experiments/modded_nanogpt_b200/preflight_checklist.md`
4. `scripts/rootfs/enter_rootfs.sh` and `scripts/rootfs/build_rootfs.sh`

This spec does not authorize a full B200 training launch, a commit, a push, a
pull request, or destructive cleanup of existing rootfs/result/cache trees.

## Design Goals

- Every substantive Python, package-management, CUDA, NCCL, data, training,
  parsing, and summarization command runs after `TORCHTITAN_IN_ROOTFS=1`.
- Host shell remains an orchestrator only: it may create ignored directories,
  set wrapper environment variables, call repo-local wrappers, and inspect
  generated artifacts.
- The rootfs runtime is reproducible from checked-in scripts plus explicit
  package lock artifacts. No operator should need to remember an out-of-band
  install command.
- Mutable state is split by owner: immutable base rootfs, package-manager
  caches, persistent project virtualenv/tool installs, reusable data, reusable
  model caches, and per-attempt evidence.
- Full training can run offline after a networked preparation phase has
  populated package wheels, tool installs, source snapshots, data manifests,
  and HuggingFace assets.
- Attempts record enough environment evidence to distinguish an actual training
  regression from a rootfs, package, cache, data, or driver drift.
- Runtime state is governed by checked-in schemas. Launch wrappers trust only
  verifier reports written from those schemas, never ad hoc environment
  inspection or operator notes.

## Current State Summary

The current `scripts/rootfs/enter_rootfs.sh`:

- binds `scripts/rootfs/rootfs` as `/`;
- mounts the checkout read-write at `/workspace/torchtitan`;
- creates tmpfs `/tmp`;
- provides `/dev`, `/proc`, NVIDIA devices, host NVIDIA driver libraries, and
  optional `nvidia-smi`;
- read-only binds host `/etc/resolv.conf` and `/etc/hosts`;
- unshares all namespaces while sharing host network;
- sets `PATH`, `CUDA_HOME`, `CUDA_PATH`, `LD_LIBRARY_PATH`,
  `TORCHTITAN_IN_ROOTFS=1`, `HOME=/root`, and `NVIDIA_VISIBLE_DEVICES`.

The current modded-nanogpt wrappers:

- re-enter `scripts/rootfs/enter_rootfs.sh` before Python work;
- guard on `TORCHTITAN_IN_ROOTFS=1`, cwd `/workspace/torchtitan`, and the
  rootfs sentinel file;
- place HuggingFace cache under `/workspace/torchtitan/.cache/huggingface`;
- place full-attempt TorchInductor and Triton caches under the attempt result
  directory;
- place preflight/diagnostic compiler caches under shared ignored result-cache
  directories;
- rely on rootfs-installed Torch `2.13.0+cu132`, CUDA runtime `13.2`, and
  Triton `3.7.1`.

Closed non-launch foundation gaps:

- `scripts/rootfs/runtime_env.sh` centrally owns the rootfs runtime environment
  policy, writable project state, cache roots, and offline/networked mode.
- `enter_rootfs.sh`, `build_rootfs.sh`, and `rootfs_target.sh` use the shared
  runtime policy and managed rootfs store helpers.
- `experiments/modded_nanogpt_b200/runtime/` defines the experiment Python and
  tool sync contracts, schema files, and runtime verifier.
- Package and tool sync wrappers re-enter the rootfs, reject forged rootfs
  sentinels, and write project-owned state under `/project`.
- Full launch readiness is gated on schema-validated runtime verification and
  canonical command-environment evidence.

Closed networked setup gap:

- `requirements.lock` is now populated by a networked rootfs setup phase using
  `uv pip compile --no-deps --generate-hashes`. The lock is direct-only by
  design: Torch, Triton, CUDA, and NVIDIA runtime packages remain owned by the
  rootfs stack and are forbidden in the experiment runtime lock. Offline sync
  still fails closed if a future edit removes hash-locked package entries.

## Configuration Model

The runtime has four configuration scopes. A full launch is valid only when all
four scopes agree.

1. **Declared runtime config.** Human-authored, portable files under
   `experiments/modded_nanogpt_b200/runtime/`. These files describe logical
   roots, package contracts, tool contracts, verifier profiles, capacity
   floors, network policy, source/data requirements, and backend selectors. They
   may use logical refs such as `repo://`, `state://`, `result://`,
   `data://`, `rootfs://`, `cache://`, and `attempt://`. They must not contain
   workstation-specific absolute host paths.
2. **Local environment resolver.** Machine-local YAML that maps logical refs to
   concrete host paths and host policies. This is the only configuration layer
   that may contain absolute host paths. Tracked examples are templates only;
   operator-filled resolver files stay ignored.
3. **Materialized run config.** Machine-written JSON for one `run_id` and
   `attempt_id`. It resolves every logical ref, records argv arrays and
   environment maps, names the selected rootfs, captures package/tool/runtime
   digests, and declares the exact sandbox projection expected for this
   attempt. It is written under the attempt result directory before preflight.
4. **Emitted bwrap plan.** The rootfs entrypoint resolves the materialized run
   config into the actual bwrap argv, mount list, environment list, cwd, device
   projection, network mode, and inner command. The emitted plan is a Contract
   Artifact. Launch is invalid if this plan differs from the materialized run
   config.

All command values at schema boundaries are structured argv arrays. Shell
strings are allowed only inside wrapper implementations after the schema-owned
argv has already been validated.

The intended file layout is:

```text
experiments/modded_nanogpt_b200/runtime/
  declared-runtime.yaml              # portable logical roots and policies
  local-environment.example.yaml     # tracked template, no real host paths
  profiles/
    diagnostic.yaml
    full-b200-fa2-triton.yaml
    full-b200-fa3-triton.yaml
    full-b200-flex-diagnostic.yaml
  env_contract.json
  pyproject.toml
  uv.lock
  requirements.direct.txt
  requirements.lock
  mise.toml
  sync_python_env.sh
  sync_tools.sh
  materialize_runtime.py
  verify_runtime.py
  schemas/
    declared_runtime.schema.json
    local_environment.schema.json
    materialized_runtime.schema.json
    bwrap_plan.schema.json
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

Attempt-local materialization and verifier artifacts live under:

```text
experiments/modded_nanogpt_b200/results/<run_id>/runtime/
  materialized_runtime.json
  bwrap_plan.json
  rootfs_manifest.resolved.json
  runtime_env_report.json
  python_env_report.json
  tool_env_report.json
  filesystem_report.json
  nanogpt_component_manifest.json
  optimized_kernel_report.json
  launch_prerequisites.json
  runtime_verification.json
  command.env.json
  command.argv.json
```

The host wrapper may pass only two categories of input into materialization:

- selected declared profile path and CLI classification fields;
- local resolver path, normally via
  `MODDED_NANOGPT_LOCAL_ENV=experiments/modded_nanogpt_b200/runtime/local-environment.local.yaml`.

Everything else used for launch must come from the materialized config and the
verified emitted bwrap plan.

Example local environment template:

```yaml
schema_version: 1
roots:
  repo: <absolute-path-to-torchtitan-checkout>
  state: <absolute-path-to-ignored-runtime-state-root>
  data: <absolute-path-to-experiments/modded_nanogpt_b200/data>
  results: <absolute-path-to-experiments/modded_nanogpt_b200/results>
rootfs:
  torchtitan-default: <absolute-path-to-selected-bwrap-rootfs>
network:
  default_prepare_mode: networked
  default_full_mode: offline
devices:
  expected_gpu_count: 8
  expected_gpu_name_regex: "NVIDIA B200"
```

The resolver rejects paths that are symlinks into unexpected filesystems unless
the declared profile explicitly allows that filesystem class. It also rejects a
portable declared config that embeds an absolute host path.

## Host Capability Survey

Before build, sync, preflight, or launch, a host survey records the facts that
affect whether the bwrap runtime can be trusted. The survey is read-only except
for writing its own report and may run from the host because it is checking host
capability, not executing training logic.

Required survey fields:

- `bwrap`: executable path, version, user-namespace availability, support for
  `--ro-bind`, `--bind`, `--tmpfs`, `--dev-bind`, `--proc`, `--unshare-all`,
  and network-sharing behavior.
- `rootfs`: candidate rootfs paths, selected rootfs ref, ownership marker,
  filesystem type, free bytes, free inodes, executable-bit behavior, and
  whether bind targets required by this spec already exist.
- `repo`: checkout path, expected `/workspace/torchtitan` projection, current
  git commit, dirty/untracked state summary for control files, and whether the
  checkout path is local storage.
- `state`: runtime state root path, owner UID/GID, mode, filesystem type, free
  bytes, free inodes, executable-bit behavior, file-lock behavior, and whether
  the root is on local storage.
- `data`: data root path, manifest path, manifest total bytes, free bytes,
  free inodes, read throughput smoke where cheap, and SHA verification status.
- `results`: result root path, ability to create an attempt directory
  atomically, free bytes, free inodes, fsync behavior where available, and
  stale-run collision status.
- `scratch` and `tmp`: candidate host paths, local-storage proof, free bytes,
  free inodes, executable-bit behavior, file-lock behavior, and cleanup policy.
- `/dev/shm`: host-visible size, in-rootfs projected size if already known,
  fallback host-backed shm path if used, local-storage proof, and whether the
  fallback is a network filesystem.
- `gpu`: visible NVIDIA devices, `/dev/nvidia*` inventory, `nvidia-smi` path,
  driver version, CUDA driver version, GPU names, PCI bus IDs, UUIDs, MIG state,
  total B200 capacity, and whether the active profile exposes exactly the
  declared GPU allocation. The current NanoGPT profile declares a two-GPU
  small-scale trial.
- `driver_libs`: resolved host `libcuda.so*`, `libnvidia-*.so*`, and any other
  driver libraries injected into rootfs; record path, realpath, size, mtime,
  and SHA256 when cheap enough.
- `network`: whether preparation mode can reach package indexes and
  HuggingFace endpoints, and whether offline mode can omit host networking.
- `build_tools`: Docker availability only for rootfs build/export, not for
  training; tar, rsync, git, and checksum tool paths used by the rootfs builder.

The report is `host_capability_report.json`. It is not enough for launch by
itself; it is consumed by materialization and by `runtime_verification.json`.
Full launch fails if a required host fact is missing, stale for the current
attempt, or inconsistent with the materialized runtime.

## Host Directory Contract

All generated paths below are host paths relative to the TorchTitan checkout
unless an absolute path is explicitly configured.

### Checked-In Control Plane

These files are source and should be tracked:

```text
scripts/rootfs/
  build_rootfs.sh
  enter_rootfs.sh
  rootfs_target.sh
  runtime_env.sh                 # shared env and mount policy
  verify_runtime_env.py          # structured runtime verifier

experiments/modded_nanogpt_b200/
  *.sh                           # rootfs re-entry wrappers
  *.py                           # harness, preflight, parser, summarizer
  runtime/
    pyproject.toml               # experiment direct deps, no torch
    requirements.direct.txt      # allowlisted direct installs
    requirements.lock            # hash-pinned direct install input
    mise.toml                    # non-Python tool declarations
    schema_validation.py         # lightweight schema validator
    sync_python_env.sh           # rootfs-aware Python env sync
    sync_tools.sh                # rootfs-aware tool env sync
    verify_runtime.py            # launch-governing runtime verifier
    schemas/
      attempt.schema.json
      command_argv.schema.json
      command_env.schema.json
      filesystem_report.schema.json
      launch_prerequisites.schema.json
      launch_readiness.schema.json
      nanogpt_component_manifest.schema.json
      optimized_kernel_report.schema.json
      preflight_report.schema.json
      python_env_report.schema.json
      rootfs_manifest.schema.json
      runtime_env.schema.json
      runtime_verification.schema.json
      summary.schema.json
      tool_env_report.schema.json
```

The `runtime/pyproject.toml` must list only experiment-owned direct Python
dependencies. It must not depend on or pin `torch`, `triton`, CUDA wheels, or
packages that replace the B200-capable rootfs stack.

### Rootfs Image and Store

Current compatibility path:

```text
scripts/rootfs/rootfs/
```

Target managed-store path:

```text
scripts/rootfs/store/
  selected.json
  content/
    <store_id>/
      .torchtitan-rootfs-owner
      manifest.json
      bin/
      usr/
      opt/
```

`store_id` should be content-addressed or manifest-addressed, for example:

```text
pytorch-2.13.0-cu132-cudnn9-uv-<manifest-sha12>
```

The selected content directory is the only rootfs mounted at `/`. The builder
must stage a new content entry, write the ownership marker and manifest, verify
it, then atomically update `selected.json`. It must not delete arbitrary paths.

The rootfs image must pre-create every bind target used by the read-only
production mount contract. At minimum the image contains:

```text
/workspace/torchtitan
/project/home
/project/xdg-cache
/project/uv-cache
/project/pip-cache
/project/mise
/project/venvs
/project/wheels
/project/downloads
/project/scratch
/project/tmp
/project/shm
/project/logs
/run/torchtitan/attempt
/run/torchtitan/nvidia-host
```

This is required because bwrap cannot create missing bind targets inside a
read-only rootfs.

### Runtime State Root

Use one ignored runtime state root per experiment:

```text
.cache/torchtitan-rootfs/modded_nanogpt_b200/
  home/
  xdg-cache/
  uv-cache/
  pip-cache/
  mise/
  venvs/
    b200-runtime/
  wheels/
  downloads/
  scratch/
  tmp/
  shm/
  logs/
```

This state is reusable but not authoritative. It may be deleted after no jobs
are running, then regenerated from the tracked runtime files and manifests.

### Persistent Experiment Data

Use the existing ignored experiment directories:

```text
experiments/modded_nanogpt_b200/
  sources/
  data/
  results/
```

Required ownership:

- `sources/`: fetched upstream and B200-compatible source snapshots; dirty
  source checks must run before launch.
- `data/`: reusable local datasets and manifests; full FineWeb manifests must
  include shard paths, byte sizes, SHA256s, source commit, token budget, and
  freshness status.
- `results/`: immutable run-attempt evidence, logs, telemetry, launch
  readiness, per-attempt compiler caches, and summaries.

Full attempts should continue to use result-local compiler caches:

```text
experiments/modded_nanogpt_b200/results/<run_id>/
  torchinductor_cache/
  triton_cache/
  tmp/
  shm/
```

Preflight and diagnostic commands may use shared caches only when their artifact
labels say they are not full-attempt evidence.

## Schema Contract

All launch-governing runtime inputs and verifier outputs must have explicit,
checked-in schemas under:

```text
experiments/modded_nanogpt_b200/runtime/schemas/
```

Schemas are versioned with a top-level integer `schema_version`. A verifier may
accept older versions only when the compatibility behavior is implemented in
code and covered by tests. Full launch must use the current schema version for
every runtime-governing artifact.

Required schema files:

- `declared_runtime.schema.json`: describes portable runtime policy. Required
  fields: `schema_version`, `profiles`, `logical_roots`, `network_modes`,
  `capacity_floors`, `package_contracts`, `tool_contracts`,
  `mount_projection`, `device_policy`, and `verifier_profiles`.
- `local_environment.schema.json`: describes machine-local resolver input.
  Required fields: `schema_version`, `roots`, `rootfs`, `network`, `devices`,
  and `storage_policy`. This schema is the only one that allows absolute host
  paths by default.
- `materialized_runtime.schema.json`: describes one run's resolved launch
  contract. Required fields: `schema_version`, `run_id`, `attempt_id`,
  `declared_profile`, `local_environment`, `resolved_roots`, `rootfs_ref`,
  `network_mode`, `mounts`, `canonical_env`, `inner_argv`, `capacity_floors`,
  `package_contracts`, `tool_contracts`, `device_policy`, `nvidia_driver_policy`,
  `expected_reports`, and `digest`.
- `bwrap_plan.schema.json`: describes the rootfs entrypoint's emitted plan.
  Required fields: `schema_version`, `run_id`, `attempt_id`, `entrypoint`,
  `rootfs`, `cwd`, `network_mode`, `mounts`, `devices`, `driver_libraries`,
  `environment`, `inner_argv`, `bwrap_argv`, `redactions`, and `digest`.
- `rootfs_manifest.schema.json`: describes the selected rootfs content. Required
  fields: `schema_version`, `store_id`, `base_image`, `base_image_digest`,
  `builder_script_commit`, `created_at_utc`, `python_version`, `torch_version`,
  `cuda_runtime_version`, `triton_version`, `uv_version`, `mise_version`,
  `cuda_synth_packages`, `apt_packages`, `precreated_bind_targets`,
  `ownership_marker`, and `content_digest`.
- `runtime_env.schema.json`: describes the intended bwrap environment and mount
  plan. Required fields: `schema_version`, `environment_class`, `network_mode`,
  `declared_runtime`, `local_environment`, `materialized_runtime`,
  `bwrap_plan`, `rootfs_mount`, `repo_mount`, `state_root`, `canonical_paths`,
  `canonical_env`, `mount_projection`, `device_policy`,
  `nvidia_driver_policy`, `cache_policy`, `scratch_policy`, and
  `offline_policy`.
- `python_env_report.schema.json`: describes uv-managed Python state. Required
  fields: `schema_version`, `python_executable`, `python_prefix`, `uv_version`,
  `uv_cache_dir`, `pip_cache_dir`, `requirements_input`, `requirements_digest`,
  `installed_distributions`, `torch_version`, `cuda_runtime_version`,
  `triton_version`, `direct_imports`, and `drift_ok`.
- `tool_env_report.schema.json`: describes mise-managed tool state. Required
  fields: `schema_version`, `mise_version`, `mise_config`, `mise_config_digest`,
  `mise_data_dir`, `mise_cache_dir`, `tools`, and `tool_paths_ok`.
- `filesystem_report.schema.json`: describes writable runtime filesystems.
  Required fields: `schema_version`, `paths`, `capacity_floors`, `statvfs`,
  `local_storage_checks`, `read_write_checks`, and `capacity_ok`.
- `launch_prerequisites.schema.json`: describes non-runtime prerequisites owned
  by the benchmark harness. Required fields: `schema_version`,
  `classification`, `source`, `data_manifest`, `hardware`, `nccl`, `backend`,
  `active_jobs`, `authorization`, `known_stall_policy`,
  `prerequisites_satisfied`, and `training_launch_allowed`.
- `runtime_verification.schema.json`: joins every verifier result. Required
  fields: `schema_version`, `classification`, `run_id`, `attempt_id`,
  `environment_class`, `checked_at_utc`, `host_capability`,
  `rootfs_manifest`, `runtime_env`, `materialized_runtime`, `bwrap_plan`,
  `python_env`, `tool_env`, `filesystem`, `launch_prerequisites`, `ok`,
  `blockers`, and `report_digests`.
- `attempt.schema.json`: describes immutable attempt metadata. Required fields:
  `schema_version`, `classification`, `command`, `environment`,
  `created_at_utc`, and `artifact_paths`. It must not allow raw authorization
  tokens.
- `preflight_report.schema.json`: describes the existing preflight output.
  Required fields: `schema_version`, `classification`, `environment`, `checks`,
  `ok`, and `blockers`.
- `launch_readiness.schema.json`: describes the final launch gate. Required
  fields: `schema_version`, `classification`, `run_id`, `attempt_id`,
  `prerequisites_satisfied`, `authorization_ok`, `skip_run`,
  `training_launch_allowed`, `ready_to_launch`, `runtime_verification`,
  `effective_env_digest`, `command_env_digest`, `blocked_by`, and
  `training_launched`.
- `command_env.schema.json`: describes the sanitized environment passed to the
  training command. Required fields: `schema_version`, `run_id`, `attempt_id`,
  `env`, `redactions`, and `digest`.
- `command_argv.schema.json`: describes the sanitized command argv. Required
  fields: `schema_version`, `run_id`, `attempt_id`, `argv`, `redactions`, and
  `digest`.
- `nanogpt_component_manifest.schema.json`: describes the PyTorch/nanoGPT
  runtime component graph selected for an attempt. Required fields:
  `schema_version`, `source_tree`, `source_commit`, `source_digest`,
  `train_entrypoint`, `kernel_source_files`, `python_imports`,
  `attention_backend`, `mlp_backend`, `torch_compile_policy`, `fp8_policy`,
  `distributed_backend`, `data_components`, `required_components`,
  `optional_components`, and `forbidden_components`.
- `optimized_kernel_report.schema.json`: describes optimized kernel library
  verification. Required fields: `schema_version`, `run_id`, `attempt_id`,
  `selected_components`, `component_reports`, `attention_backend_ok`,
  `mlp_backend_ok`, `torch_compile_ok`, `fp8_ok`, `distributed_ok`,
  `data_pipeline_ok`, `forbidden_components_absent`, `ok`, and `blockers`.
- `summary.schema.json`: describes summarized attempt outcome. Required fields:
  `schema_version`, `classification`, `runtime_verification`,
  `launch_readiness`, `metrics`, `full_attempt_blocker`, and
  `included_in_baseline_stats`.

Every schema must set `additionalProperties` deliberately. For launch-gating
reports, unknown top-level fields are rejected unless the schema explicitly
marks an extension object for non-gating diagnostics.

## Verifier Contract

The verifier flow is acyclic:

1. Host capability survey records host-level facts and candidate roots.
2. Runtime materialization validates declared runtime config, local resolver
   config, rootfs selection, logical-root resolution, profile selection,
   command argv, canonical env, capacity floors, and expected reports.
3. The rootfs entrypoint emits `bwrap_plan.json`; bootstrap runtime verifiers
   validate rootfs, emitted plan, runtime env, Python env, tool env, filesystem
   state, and command schema before preflight.
4. Preflight consumes those bootstrap reports, validates its own
   `preflight_report.json` against schema, and writes harness prerequisite
   evidence.
5. The final runtime verifier joins bootstrap reports, preflight output,
   launch prerequisites, sanitized command env/argv, and authorization state
   into `runtime_verification.json`.
6. Launch readiness consumes `runtime_verification.json`. `torchrun` can start
   only after launch readiness is schema-valid and allows training.

Every launch path must run verifiers before training, including dry launch and
missing-authorization paths. Use these booleans consistently:

- `prerequisites_satisfied`: source, data, runtime, hardware, NCCL, backend,
  active-job, and known-stall gates pass, independent of whether the operator
  supplied launch authorization.
- `authorization_ok`: the required full-launch authorization is present, or the
  selected mode does not require it.
- `training_launch_allowed`: the runtime verifier found no blockers for the
  selected command environment and the caller requested launch-permission
  validation. The runner must still honor `skip_run=true` by not launching
  training.
- `ready_to_launch`: reserved for the final launch-readiness artifact. It means
  the launch-governing gates passed for the attempt envelope. A skip-run
  artifact may record `ready_to_launch=true` as prerequisite evidence, but it
  still records `training_launched=false` and remains excluded from baseline
  stats and non-skip launch-ready rows.

A launch wrapper may write `ready_to_launch=true` only after the runtime
verifier has written a current-version `runtime_verification.json` with
`ok=true` and `training_launch_allowed=true`.

The verifier entrypoints are:

```text
scripts/rootfs/verify_runtime_env.py
experiments/modded_nanogpt_b200/runtime/verify_runtime.py
experiments/modded_nanogpt_b200/runtime/materialize_runtime.py
```

`scripts/rootfs/verify_runtime_env.py` owns repo-generic rootfs checks:

- emitted bwrap plan schema validation and digesting;
- rootfs sentinel, cwd, and entrypoint path;
- selected rootfs store ID and manifest schema validation;
- bwrap network mode and offline environment;
- canonical env vars and path ownership;
- mount target existence and read/write expectations;
- `/dev/shm`, `/project/tmp`, `/project/scratch`, cache, data, and result
  filesystem capacity floors;
- NVIDIA device and host-driver library visibility;
- active Python path and rootfs-provided Torch/CUDA/Triton version contract.

`experiments/modded_nanogpt_b200/runtime/verify_runtime.py` owns
experiment-specific checks:

- declared runtime and local environment schema validation;
- logical-root resolution and materialized runtime digest validation;
- uv-managed Python environment and direct dependency imports;
- mise-managed tool configuration and resolved executables;
- PyTorch/nanoGPT component manifest validation;
- optimized kernel library verification for the selected source/backend pair;
- source provenance and variant patch classification;
- data manifest schema, SHA verification, and `DATA_PATH` visibility;
- active-job scan freshness;
- full-mode hardware, NCCL, backend, known-stall, and authorization gates;
- separate prerequisite, authorization, skip-run, and final launch permission
  booleans;
- consistency between `classification`, `attempt.json`,
  `preflight_report.json`, `launch_readiness.json`, and
  `runtime_verification.json`;
- equality between the effective command environment digest and the
  `runtime_env_report`/`command.env.json` digests that launch readiness records.

Verifier outputs must be written atomically under the attempt result directory:

```text
<result_dir>/runtime/
  host_capability_report.json
  materialized_runtime.json
  bwrap_plan.json
  rootfs_manifest.resolved.json
  runtime_env_report.json
  python_env_report.json
  tool_env_report.json
  filesystem_report.json
  nanogpt_component_manifest.json
  optimized_kernel_report.json
  launch_prerequisites.json
  runtime_verification.json
  command.env.json
  command.argv.json
```

The same reports may also be copied to `/project/logs` for reusable environment
setup commands, but the attempt-local copies are the launch authority.

`bwrap_plan.json` must be emitted by `scripts/rootfs/enter_rootfs.sh` before any
payload process starts. The implemented emit-plan control is:

```text
TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY=1
TORCHTITAN_ROOTFS_MATERIALIZED_CONFIG=<result_dir>/runtime/materialized_runtime.json
```

In emit-plan mode the entrypoint resolves the selected rootfs, mount list,
environment, cwd, device projection, driver-library projection, network mode,
and inner argv, writes the plan to stdout or `TORCHTITAN_ROOTFS_PLAN_OUTPUT`,
and exits without executing the payload. Normal execution must use the same
materialized config and must produce the same plan digest immediately before
launch. A changed digest blocks full mode.

Plan validation checks:

- rootfs path and store ID match `materialized_runtime.json`;
- every mount target, source, mode, and mutability flag matches the declared
  mount projection;
- cwd is `/workspace/torchtitan`;
- `TORCHTITAN_IN_ROOTFS=1` and all canonical env values match exactly;
- `PYTHON`, `HOME`, `TMPDIR`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`,
  `MISE_DATA_DIR`, `MISE_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, compiler paths,
  and cache dirs point to governed sandbox paths;
- `DATA_PATH` is a root containing `data/fineweb10B/`, not the `data/`
  directory itself;
- offline mode omits host network sharing and sets offline environment values;
- networked mode is limited to setup and diagnostic phases unless the attempt
  is explicitly classified as diagnostic;
- NVIDIA devices and driver libraries are exactly the surveyed set for full
  mode;
- inner argv equals the schema-owned command argv; and
- the redacted bwrap argv contains no raw authorization token or secret.

All verifiers return distinct phases:

- `host_capability`
- `materialize_runtime`
- `bwrap_plan`
- `rootfs_schema`
- `runtime_env`
- `python_env`
- `tool_env`
- `filesystem`
- `source`
- `data_manifest`
- `hardware`
- `nccl`
- `backend`
- `nanogpt_components`
- `optimized_kernels`
- `active_jobs`
- `authorization`

Failures must include `phase`, `message`, `expected`, `actual`, and
`artifact_path` when applicable. Full launch stops on the first failing phase
only after writing the partial verifier report and a launch-readiness blocker.

The verifier implementation should prefer Python standard-library validation
for the first pass so the schema gate does not depend on a package that might be
missing before uv sync. If JSON Schema validation is added later, `jsonschema`
must be installed through the uv-managed runtime and the bootstrap verifier must
still be able to fail closed before that environment exists.

## PyTorch NanoGPT Component Contract

The training source is not a generic PyTorch script. It is a compact nanoGPT
program with a small set of optimized libraries and source-local kernels that
must be treated as launch-governing components.

The component manifest must be derived from the selected source tree before
preflight. It records the exact files and import surfaces that the source uses:

```text
train_gpt.py
train_gpt_medium.py
triton_kernels.py
dc_triton_kernels.py
data/cached_fineweb10B.py
data/fineweb.py
evals/hellaswag.py
```

For each file, the manifest records path, SHA256, source role, and whether it is
used by the selected full attempt. A source tree with extra optimized-kernel
files must either classify them in the manifest or fail full-mode verification.
Files that exist but are not selected, such as `train_gpt_medium.py` during the
900M `train_gpt.py` speedrun, are recorded as `used_by_selected_attempt=false`
with their own import surfaces. They do not inherit the selected entrypoint's
backend assumptions.

The manifest is not hand-authored. The verifier derives it by inspecting the
selected source tree, the lane, and the launcher-selected backend environment.
It records:

- import surfaces discovered from the source files;
- package names and import names for runtime libraries;
- source-local kernel module names and resolved module file paths;
- selected attention and MLP backend selectors;
- package provenance, including uv runtime, rootfs base, source tree, or
  setup-wrapper output;
- required, optional, diagnostic-only, and forbidden component status;
- smoke name, expected device capability, expected dtype, expected output shape,
  and cache-directory policy for every optimized component.

Current source facts that must be represented:

- selected `train_gpt.py` uses PyTorch, CUDA, Triton, TorchDynamo,
  TorchInductor, `torch.distributed`, `kernels.get_kernel`, source-local
  `triton_kernels`, source-local `dc_triton_kernels`, NumPy, and tiktoken;
- selected upstream `train_gpt.py` attention is FA3 via
  `kernels-community/flash-attn3`, version `1`;
- `train_gpt_medium.py`, when selected by a future attempt, has its own inline
  optimized-kernel surface and currently resolves FA3 through
  `varunneal/flash-attention-3`; it must receive a separate component manifest
  and verifier profile before it can support a full claim;
- the B200 compatibility source adds
  `MODDED_NANOGPT_ATTN_BACKEND={fa3,fa2,flex}` and imports PyTorch
  FlexAttention helpers for the flex path;
- the B200 compatibility source adds
  `MODDED_NANOGPT_MLP_BACKEND={triton,torch}`;
- the inspected source trees do not import `flashinfer`, `flash_infer`, or a
  package named `flash-infer`.

### Required Core Components

These components are required for every full attempt:

- `torch`: importable from the active uv-managed Python, exactly the pinned
  rootfs version for full mode, with CUDA available.
- `torch.cuda`: sees exactly the declared active allocation for full trial mode,
  currently two B200 devices, reports capability `(10, 0)` or newer for each
  visible device, and can allocate BF16 and FP8 tensors on device 0.
- `torch._scaled_mm`: present and passes a small FP8 -> BF16 smoke with
  `use_fast_accum` both true and false where supported.
- `torch.compile` and TorchInductor: enabled for full mode, writes only under
  the attempt-local `TORCHINDUCTOR_CACHE_DIR`, and performs a tiny compile smoke
  before launch readiness.
- `triton`: importable, exactly the pinned rootfs version for full mode, writes
  only under the attempt-local `TRITON_CACHE_DIR`, and can compile a tiny
  `@triton.jit` kernel on B200.
- `triton.tools.tensor_descriptor.TensorDescriptor`: importable because the
  source-local MLP and softcap kernels use it.
- `torch.distributed` with NCCL: all-reduce smoke must pass over the declared
  launch world size before full trial mode; the current NanoGPT trial requires
  two ranks.
- `kernels`: importable for selected source trees that import
  `from kernels import get_kernel` at module load time. This source-import
  requirement is separate from selecting the FA3 artifact.
- `numpy`, `tiktoken`, `datasets`, `huggingface_hub`, `tqdm`,
  `typing_extensions`, and `setuptools`: importable from the active runtime
  Python when the selected source or data path needs them.

### Source-Local Triton Kernels

The source-local Triton modules are launch-governing:

- `triton_kernels.py`: symmetric matrix kernels, fused MLP, fused softcapped
  cross entropy, FP8 quantization, transpose copy/add, and scaled-mm helper
  paths.
- `dc_triton_kernels.py`: dynamic-correction attention post-processing kernels.

The verifier must import both modules from the selected source directory, not
from `PYTHONPATH` fallback locations. It records module file paths and digests.
If a selected source tree does not contain one of these modules, the verifier
must prove the selected entrypoint does not import it before treating the module
as absent but allowed.

Full-mode Triton MLP verification requires:

- `MODDED_NANOGPT_MLP_BACKEND=triton`;
- import `triton_kernels`;
- run `FusedLinearReLUSquareFunction.apply` on BF16 CUDA tensors shaped like the
  existing smoke (`x=[2, 16, 768]`, `w1=[3072, 768]`, `w2=[3072, 768]`);
- run backward, synchronize CUDA, and record output shape, dtype, cache dir, and
  failure class.

The PyTorch MLP fallback remains diagnostic-only until a full-mode verifier
explicitly approves it. A passing local PyTorch MLP smoke is not sufficient for
baseline inclusion.

Full-mode dynamic-correction Triton verification requires:

- import `dc_triton_kernels` from the selected source directory when the
  selected entrypoint imports it;
- verify that the selected entrypoint calls
  `dc_attention_postonly_nodd_correction_add_base_triton` or classify the module
  as import-only;
- run a bounded forward/backward smoke through
  `dc_attention_postonly_nodd_correction_add_base_triton` on BF16 CUDA tensors
  with the source-supported shape family: `B=1`, small `T`, `H=6`, `D=128`,
  `window<=128`, `seq_lens=[0,T]`, base output shape `[T,6,128]` or
  `[1,T,6,128]`, and post-only DC weights compatible with the helper;
- synchronize CUDA and record output shape, dtype, gradient availability, cache
  dir, and failure class.

An import-only `dc_triton_kernels` check is not enough for full mode when the
selected training graph uses dynamic-correction attention.

### Attention Backends

The selected attention backend is controlled by `MODDED_NANOGPT_ATTN_BACKEND`
and must match lane classification.

`fa3`:

- Required for Lane A.
- Requires `kernels` package import.
- Requires the selected entrypoint's declared `get_kernel(...)` target to
  resolve. For current full `train_gpt.py` this is
  `get_kernel("kernels-community/flash-attn3", version=1)`.
- Requires `flash_attn_interface.flash_attn_varlen_func` to run the varlen
  sliding-window BF16 smoke used by current preflight.
- On B200, verifier must record whether the loaded kernel artifact contains an
  `sm_100`/Blackwell-compatible image when that can be inspected. If image
  inspection is unavailable, the CUDA smoke result is the authority.

`fa2`:

- Current Lane B full candidate.
- Requires `flash-attn==2.8.3.post1`, built or installed through
  `experiments/modded_nanogpt_b200/setup_flash_attention.sh`.
- Requires `kernels` package importability when the selected source tree imports
  `get_kernel` at module load time, even though FA2 does not resolve the FA3
  artifact.
- Requires CUDA synth build packages `nvidia-cuda-nvcc`, `nvidia-cuda-crt`,
  `nvidia-cuda-cccl`, and `nvidia-nvvm` at the pinned CUDA wheel version.
- Requires `/opt/cuda-synth/bin/nvcc`, `libcudart.so.13`, and the unversioned
  `libcudart.so` bridge.
- Requires `from flash_attn.flash_attn_interface import flash_attn_varlen_func`
  and the same varlen/window BF16 smoke used by setup/preflight.

`flex`:

- Uses `torch.nn.attention.flex_attention.create_block_mask` and
  `flex_attention`.
- Currently diagnostic-only for full B200 claims unless a future ticket
  promotes it. The verifier may run a smoke, but full-mode launch readiness must
  reject `MODDED_NANOGPT_ATTN_BACKEND=flex`.

`sdpa`:

- Not a current source-selected backend in the inspected source trees.
- If a future source adds `torch.nn.functional.scaled_dot_product_attention` or
  an `sdpa` backend, it must be added to the component manifest, get its own
  schema entry, and pass a varlen-equivalent smoke before use in full mode.

`fa4`:

- Not a current source-selected backend in the inspected source trees.
- It is a named certification-matrix backend because future Blackwell attention
  work may use FA4 or FA4-like providers.
- The verifier must probe for an explicit provider, package, or source import
  only when one is declared by the runtime profile or source manifest. In the
  current runtime, the expected result is `support_status=unsupported` with
  evidence that no FA4 provider is selected.
- Unsupported unselected FA4 evidence does not block a full launch whose
  selected attention backend is supported and passes its smoke.

`flashinfer` / `flash-infer`:

- Not imported by the inspected source trees.
- It is a forbidden component for current full attempts. If future source code
  imports `flashinfer` or `flash_infer`, or a package named `flashinfer` /
  `flash-infer` is selected in the runtime manifest, full launch must fail until
  a new backend-specific verifier is added.

### Torch Compile and FP8 Policy

The source uses `torch.compile` for model, optimizer/math helpers, and custom
op implementations. Full mode must reject global settings that silently disable
or materially change compilation unless the lane explicitly classifies them.

Verifier requirements:

- `TORCHDYNAMO_DISABLE` is absent or false for full mode.
- No extra `torch._inductor.config` or non-upstream `torch.compile` tuning flags
  are enabled for Lane A.
- `TORCHINDUCTOR_CACHE_DIR` is attempt-local.
- A small `torch.compile(dynamic=False)` smoke runs on CUDA and writes only to
  the allowed cache.
- FP8 tensor creation works for `torch.float8_e4m3fn` and `torch.float8_e5m2`
  where the source uses them.
- `torch._scaled_mm` supports the source's FP8 call shapes well enough for the
  existing primitive smoke and MLP smoke.

### Data and Tokenizer Components

The verifier must treat data-side Python packages as runtime components because
the training source imports tokenizer and data code inside the timed program or
data-preparation path:

- `tiktoken`: `get_encoding("gpt2")` succeeds inside rootfs and uses a writable
  cache path governed by the runtime env.
- `numpy`: importable and usable for shard loading helpers.
- `datasets` and `huggingface_hub`: importable for data preparation and cache
  under `HF_HOME`/`HF_HUB_CACHE`.
- `tqdm`: importable for `data/fineweb.py` progress reporting.
- `setuptools` and `typing_extensions`: present when required by package build
  metadata or runtime imports.

Full launch requires `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`,
`TMPDIR`, and package-manager caches to match `runtime_env_report.json`.

### Optimized Kernel Verification Report

`optimized_kernel_report.json` is attempt-local and schema-governed. It records
one object per component. Each component report includes:

- `name`, `kind`, `role`, `requested`, `selected_for_launch`,
  `required_for_selected_launch`, and `diagnostic_only`;
- `support_status`: one of `supported`, `unsupported`, `unknown`, or
  `forbidden`;
- `build_status`: one of `pass`, `fail`, `not_required`, or `not_run`;
- `smoke_status`: one of `pass`, `fail`, `not_required`, or `not_run`;
- `launch_eligible`: true only when the component is selected for launch,
  supported, and its required build and smoke checks pass, or when the component
  is an unselected non-required dependency whose documented status permits it;
- `failure_class`, `failure_detail`, `stdout_tail`, and `artifact_paths` for
  every failed, unsupported, unknown, or forbidden result;
- `import_name`, `package_name`, `expected_version`, and `actual_version`;
- `source`, `source_path`, `source_digest`, and setup-wrapper provenance when
  the component comes from the selected source tree or a build wrapper;
- `selector_env`, `expected_selector_value`, and `actual_selector_value` for
  backend-controlled components;
- `cache_dirs` and `cache_policy_ok`;
- `device` and, when inspectable, native artifact `sm_arches`;
- `smoke` with deterministic shape, dtype, synchronization, and failure class;
- `forbidden_absent` for components that must not appear in the current source
  or runtime;
- `ok` and structured `blockers`.

Example component report:

```json
{
  "name": "flash-attn",
  "kind": "attention",
  "role": "attention_backend",
  "requested": true,
  "selected_for_launch": true,
  "required_for_selected_launch": true,
  "diagnostic_only": false,
  "support_status": "supported",
  "build_status": "pass",
  "smoke_status": "pass",
  "launch_eligible": true,
  "failure_class": null,
  "failure_detail": null,
  "import_name": "flash_attn.flash_attn_interface",
  "package_name": "flash-attn",
  "expected_version": "2.8.3.post1",
  "actual_version": "2.8.3.post1",
  "source": "uv-runtime|rootfs|source-tree",
  "source_path": null,
  "source_digest": null,
  "selector_env": "MODDED_NANOGPT_ATTN_BACKEND",
  "expected_selector_value": "fa2",
  "actual_selector_value": "fa2",
  "cache_dirs": {
    "torchinductor": "...",
    "triton": "..."
  },
  "cache_policy_ok": true,
  "device": {
    "name": "NVIDIA B200",
    "capability": [10, 0]
  },
  "sm_arches": ["sm_100"],
  "smoke": {
    "name": "varlen_window_bf16",
    "ok": true,
    "output_shape": [256, 6, 128],
    "output_dtype": "torch.bfloat16",
    "failure_class": null
  },
  "stdout_tail": null,
  "artifact_paths": [],
  "forbidden_absent": null,
  "ok": true,
  "blockers": []
}
```

Required component names for current source trees:

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
attention_fa4
attention_flex
attention_torch_sdpa
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

Components that are not selected still appear in the report as
`selected_for_launch=false`. Unsupported or unknown unselected components, such
as current FA3 or FA4 provider paths on B200, must preserve hard evidence in the
report but do not block full launch unless the selected source or runtime
profile requests them. Forbidden components, such as `flashinfer` in the
current source trees, must be recorded with `selected_for_launch=false` and
`forbidden_absent=true`. The forbidden-component check must search source
imports, runtime dependency manifests, installed distributions, and backend
selector values. A package may be installed in the reusable uv environment only
if it is not selected and does not appear in the selected source manifest; a
selected or imported forbidden package blocks full launch until this spec adds a
backend verifier for it.

Full launch requires:

- `optimized_kernel_report.ok=true`;
- selected attention backend report has `support_status=supported`,
  `build_status=pass` when a build is required, `smoke_status=pass`, and
  `launch_eligible=true`;
- selected MLP backend report has `support_status=supported`,
  `build_status=pass` when a build is required, `smoke_status=pass`, and
  `launch_eligible=true`;
- all required core components `ok=true`;
- every forbidden component absent;
- diagnostic-only components unselected;
- selected component selector env values match `command.env.json`;
- report digests included in `runtime_verification.json` and
  `launch_readiness.json`.

The report-level `ok=true` is a certified-matrix decision, not an all-green
matrix decision. It means all selected launch-governing components are
launch-eligible, every required core prerequisite passed, forbidden components
are absent, and every unsupported or unknown unselected component has structured
evidence. Known-unsupported unselected FA3, FA4, Triton variants, or fallback
paths must not be hidden, but they also must not force the selected FA2/Triton
or future certified tuple to fail solely because they are present in the matrix.

## In-Rootfs Layout

The runtime must present these canonical paths:

```text
/
  workspace/
    torchtitan/                  # checkout, cwd
  project/
    home/                        # mounted from runtime state root
    xdg-cache/
    uv-cache/
    pip-cache/
    mise/
    venvs/
    wheels/
    downloads/
    scratch/
    tmp/
    shm/
    logs/
  run/
    torchtitan/
      attempt/                   # optional bind to current result dir
      nvidia-host/               # optional future host-driver injection path
```

Canonical environment:

```text
TORCHTITAN_IN_ROOTFS=1
TORCHTITAN_ROOTFS_ENV=modded_nanogpt_b200
TORCHTITAN_ROOTFS_STORE_ID=<store_id-or-legacy-rootfs>
TORCHTITAN_ROOTFS_HOST_STATE=<host runtime state root>
TORCHTITAN_ROOTFS_PROJECT=/workspace/torchtitan
TORCHTITAN_ROOTFS_LOG_DIR=/project/logs
HOME=/project/home
XDG_CACHE_HOME=/project/xdg-cache
UV_CACHE_DIR=/project/uv-cache
PIP_CACHE_DIR=/project/pip-cache
UV_FIND_LINKS=/project/wheels
MISE_DATA_DIR=/project/mise/data
MISE_CACHE_DIR=/project/mise/cache
MISE_CONFIG_DIR=/workspace/torchtitan/experiments/modded_nanogpt_b200/runtime
PYTHON=/project/venvs/b200-runtime/bin/python
PATH=/project/venvs/b200-runtime/bin:/project/mise/data/shims:/opt/cuda-synth/bin:/usr/local/bin:/usr/bin:/bin
TMPDIR=/project/tmp
TEMP=/project/tmp
TMP=/project/tmp
HF_HOME=/workspace/torchtitan/.cache/huggingface
HF_HUB_CACHE=/workspace/torchtitan/.cache/huggingface/hub
TORCH_HOME=/workspace/torchtitan/.cache/torch
MPLCONFIGDIR=/project/xdg-cache/matplotlib
CUDA_HOME=/opt/cuda-synth
CUDA_PATH=/opt/cuda-synth
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/cuda-synth/lib64
CC=/usr/bin/gcc
CXX=/usr/bin/g++
```

For B200 extension builds:

```text
TORCH_CUDA_ARCH_LIST=10.0
MAX_JOBS=8
```

For a full modded-nanogpt attempt, the launcher additionally owns:

```text
DATA_PATH=<root containing data/fineweb10B/>
TORCHINDUCTOR_CACHE_DIR=<result_dir>/torchinductor_cache
TRITON_CACHE_DIR=<result_dir>/triton_cache
MODDED_NANOGPT_ATTN_BACKEND=<fa2|fa3|flex>
MODDED_NANOGPT_MLP_BACKEND=<triton|torch>
MODDED_NANOGPT_CE_COMPUTE_CAPABILITY=100  # Lane B only
```

## Mount Contract

### Mount Projection Table

The declared runtime config owns the projection table. The emitted bwrap plan
must contain one object per row with `host_source`, `sandbox_target`, `mode`,
`owner`, `capacity_floor`, and `verifier_phase`.

| Host source | Sandbox target | Mode | Owner | Full-mode floor | Verifier phase |
| --- | --- | --- | --- | --- | --- |
| `<selected-rootfs>` | `/` | ro-bind target | rootfs builder | rootfs manifest present | `rootfs_schema` |
| `procfs` | `/proc` | proc | bwrap | n/a | `bwrap_plan` |
| host `/dev` plus explicit NVIDIA devices | `/dev` | dev/dev-bind | rootfs entrypoint | declared allocation; two B200 devices for current NanoGPT trial | `hardware` |
| `<checkout>` | `/workspace/torchtitan` | bind, rw for current compatibility; target ro-bind after source mutation is eliminated | TorchTitan checkout | source provenance clean | `source` |
| `<state>/home` | `/project/home` | bind rw | runtime state | 1 GiB | `filesystem` |
| `<state>/xdg-cache` | `/project/xdg-cache` | bind rw | runtime state | 4 GiB | `filesystem` |
| `<state>/uv-cache` | `/project/uv-cache` | bind rw | uv | 16 GiB | `python_env` |
| `<state>/pip-cache` | `/project/pip-cache` | bind rw | uv/build backends | 16 GiB | `python_env` |
| `<state>/mise` | `/project/mise` | bind rw | mise | 8 GiB | `tool_env` |
| `<state>/venvs` | `/project/venvs` | bind rw | uv | 32 GiB | `python_env` |
| `<state>/wheels` | `/project/wheels` | bind rw in prep, ro-bind in full when locked | uv wheelhouse | enough for lock | `python_env` |
| `<state>/downloads` | `/project/downloads` | bind rw in prep, ro-bind or absent in full | setup wrappers | enough for declared downloads | `filesystem` |
| `<state>/scratch` | `/project/scratch` | bind rw | compilers and setup | 128 GiB | `filesystem` |
| `<state>/tmp` | `/project/tmp` | bind rw | runtime temp | 32 GiB | `filesystem` |
| `<state>/logs` | `/project/logs` | bind rw | runtime setup logs | 4 GiB | `runtime_env` |
| tmpfs or `<state>/shm` | `/dev/shm` | tmpfs or bind rw | distributed runtime | 64 GiB | `filesystem` |
| `<result_dir>` | `/run/torchtitan/attempt` | bind rw | current attempt | 128 GiB | `launch_prerequisites` |
| `<data_root>` | `/workspace/torchtitan/experiments/modded_nanogpt_b200/data` | bind rw for prep, ro-bind for full after manifest lock | experiment data | manifest bytes plus 1 GiB | `data_manifest` |
| host `/etc/resolv.conf` | `/etc/resolv.conf` | ro-bind in networked mode only | host resolver | present when networked | `runtime_env` |
| host `/etc/hosts` | `/etc/hosts` | ro-bind | host resolver | present | `runtime_env` |
| host driver libs | original library paths or `/run/torchtitan/nvidia-host` | ro-bind | host driver | surveyed libs present | `hardware` |
| host `nvidia-smi` | `/usr/bin/nvidia-smi` or original path | ro-bind | host driver | present for full mode | `hardware` |

The checkout is currently writable because the existing harness writes ignored
artifacts below the repo. The target contract is to move all attempt writes to
`/run/torchtitan/attempt`, `/project/*`, or declared ignored data/result roots
so the checkout can become read-only for full attempts. Until that migration is
implemented, `runtime_verification.json` must record `repo_mount_mutability` and
full summaries must not imply that the source tree was immutable.

Every row has two proofs:

- materialization proof: the host source is resolved from the local environment
  or the current attempt and matches the declared runtime profile;
- projection proof: `enter_rootfs.sh` emits the same row in `bwrap_plan.json`
  and an in-rootfs stat/read/write probe verifies mode and capacity.

### Base Mounts

The production bwrap entrypoint should mount:

```text
--ro-bind <selected-rootfs> /
--proc /proc
--dev /dev
--tmpfs /tmp
--bind <checkout> /workspace/torchtitan
--bind <state>/home /project/home
--bind <state>/xdg-cache /project/xdg-cache
--bind <state>/uv-cache /project/uv-cache
--bind <state>/pip-cache /project/pip-cache
--bind <state>/mise /project/mise
--bind <state>/venvs /project/venvs
--bind <state>/wheels /project/wheels
--bind <state>/downloads /project/downloads
--bind <state>/scratch /project/scratch
--bind <state>/tmp /project/tmp
--bind <state>/logs /project/logs
--tmpfs /dev/shm
```

The legacy entrypoint may continue to use `--bind <rootfs> /` while rootfs
bootstrapping remains mutable, but full training should target a read-only
rootfs plus explicit writable binds.

Full mode fails closed unless the mounted rootfs has a schema-valid
`rootfs_manifest` report. The legacy `scripts/rootfs/rootfs` path may be used
for full mode only after a verifier has produced a current-version legacy
manifest with equivalent fields and `mutable_rootfs_allowed=false`. Otherwise
legacy rootfs launches are diagnostic-only and cannot enter baseline
statistics.

### `/dev/shm`

Training jobs need a predictable shared-memory surface. The default should be a
tmpfs at `/dev/shm` sized to at least 64 GiB when the host permits it:

```text
--tmpfs /dev/shm
```

If bwrap does not expose a size option in the deployed version, the preflight
must measure `/dev/shm` with `statvfs` from inside rootfs and fail full mode
when available bytes are below the configured floor. A host-backed fallback may
be used only when it is explicit:

```text
--bind <state>/shm /dev/shm
```

The fallback must be per-host local storage, not a network filesystem.

### Scratch and Temp

`/project/tmp` and `/project/scratch` must be local, writable, and large enough
for extension builds and compiler artifacts. Minimum full-mode floors:

- `/project/tmp`: 32 GiB free
- `/project/scratch`: 128 GiB free
- attempt result filesystem: 128 GiB free
- dataset filesystem: at least manifest `total_bytes` plus 1 GiB free. For the
  current full 900M FineWeb manifest under the upstream `data/fineweb10B/` path,
  this means at least 4 GiB free.

Wrapper code should set `TMPDIR`, `TEMP`, and `TMP` to `/project/tmp` before any
Python, uv, pip, compiler, FlashAttention, Triton, or TorchInductor work.

### Network Phases

Two rootfs modes are required:

- `networked`: allows package resolution, source fetch, data download, and
  HuggingFace asset population.
- `offline`: no network access; required for full training attempts unless the
  attempt artifact explicitly records a diagnostic exception.

The current entrypoint uses `--share-net`. The production entrypoint should add
a mode switch:

```text
TORCHTITAN_ROOTFS_NETWORK=networked|offline
```

`offline` should omit `--share-net` and must set:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Full-mode launch readiness must record the selected network mode.

Network policy by phase:

| Phase | Default network mode | Allowed writes | Full-evidence status |
| --- | --- | --- | --- |
| host capability survey | host ambient | survey report only | prerequisite evidence |
| rootfs build/select | networked when building, offline when selecting | rootfs store | prerequisite evidence |
| uv wheel download/sync | networked for wheel population, offline for locked sync verification | `/project/wheels`, `/project/uv-cache`, venv | prerequisite evidence |
| mise install | networked for first install, offline for resolved tool verification | `/project/mise` | prerequisite evidence |
| source fetch | networked | `sources/` | prerequisite evidence |
| data preparation | networked unless all data is already local | `data/`, manifests | prerequisite evidence |
| diagnostic preflight | networked or offline, recorded | diagnostic only when networked |
| full preflight | offline | attempt runtime reports | launch gate |
| full training | offline | attempt result dir and allowed caches | production evidence candidate |
| summarization | offline | summary/index artifacts | evidence processing |

Full training with networked mode is not production evidence. It may run only as
a diagnostic attempt and must be excluded from baseline statistics.

## Package Management

### Python: uv

Python dependency management for experiment code must use uv inside the rootfs.

Allowed package operations:

```bash
experiments/modded_nanogpt_b200/runtime/sync_python_env.sh
```

That wrapper should re-enter rootfs, set the canonical environment, then run:

```bash
uv venv --python "$(command -v python)" --system-site-packages /project/venvs/b200-runtime
uv pip install \
  --python /project/venvs/b200-runtime/bin/python \
  --no-deps \
  --require-hashes \
  -r experiments/modded_nanogpt_b200/runtime/requirements.lock
```

For an explicit networked setup refresh, regenerate the direct-only lock with:

```bash
TORCHTITAN_ROOTFS_NETWORK=networked scripts/rootfs/enter_rootfs.sh -- bash -lc \
  'cd /workspace/torchtitan && uv pip compile --no-deps --generate-hashes \
    --python-platform x86_64-manylinux_2_39 --python-version 3.12 \
    --output-file experiments/modded_nanogpt_b200/runtime/requirements.lock \
    experiments/modded_nanogpt_b200/runtime/requirements.direct.txt'
```

The fallback direct install path exists only for networked setup diagnostics;
full-mode offline sync uses the hash-locked file.

The direct dependency set must include the modded-nanogpt non-Torch runtime
packages:

```text
numpy
tqdm
huggingface-hub
datasets
tiktoken
typing-extensions
setuptools
kernels          # only for Lane A or FA3 preflight
flash-attn==2.8.3.post1  # only for Lane B FA2 setup
```

Rules:

- Do not run upstream `requirements.txt`; it pins `torch==2.10`.
- Do not run `uv pip sync` against a lock that includes `torch`, `triton`, or
  CUDA packages; sync semantics are too destructive for the rootfs-provided
  B200 stack unless the lock is explicitly curated for this environment.
- Do not install a dependency that replaces Torch, Triton, CUDA runtime, or the
  rootfs compiler stack in the full-mode environment.
- uv cache must live in `/project/uv-cache`.
- pip cache, when used by uv or package build backends, must live in
  `/project/pip-cache`.
- wrapper commands must execute `${PYTHON}` or prepend
  `/project/venvs/b200-runtime/bin` to `PATH` before invoking bare `python`.
- The active Python for preflight and launch must be
  `/project/venvs/b200-runtime/bin/python` after the sync task exists.
- Full preflight must import-check the direct dependencies and reject Torch,
  CUDA, or Triton version drift.

The uv contract has two modes:

- **prepare mode:** may resolve packages from the network and populate
  `/project/wheels` plus `/project/uv-cache`; writes `python-sync.json` with
  resolved versions, wheel filenames, hashes, and whether any build happened.
- **locked mode:** may install only from `/project/wheels` or a hash-pinned
  lock; used by full preflight and full launch verification; fails if a wheel is
  missing, if a build would contact the network, or if installing would replace
  Torch, Triton, CUDA packages, or CUDA synth packages.

`flash-attn==2.8.3.post1` is special because the FA2 candidate may require a
local build against the B200 rootfs toolchain. Its setup wrapper must run in
prepare mode, record CUDA synth package versions and `TORCH_CUDA_ARCH_LIST`, and
then leave a locked wheel or installed distribution that locked mode can verify
without rebuilding during full launch.

### Non-Python Tools: mise

Non-Python tool management must use mise inside the rootfs. No host-installed
tool should become part of a full attempt unless it is explicitly injected and
recorded as host-driver or host-orchestrator evidence.

Proposed tracked config:

```toml
# experiments/modded_nanogpt_b200/runtime/mise.toml
[tools]
node = "22.18.0"
shellcheck = "0.10.0"
shfmt = "3.10.0"
```

Tool state must live under:

```text
MISE_DATA_DIR=/project/mise/data
MISE_CACHE_DIR=/project/mise/cache
MISE_CONFIG_DIR=/workspace/torchtitan/experiments/modded_nanogpt_b200/runtime
```

The setup wrapper should be:

```bash
experiments/modded_nanogpt_b200/runtime/sync_tools.sh
```

It must run inside rootfs and execute:

```bash
mise trust experiments/modded_nanogpt_b200/runtime/mise.toml
mise install --yes --cd experiments/modded_nanogpt_b200/runtime
mise exec --cd experiments/modded_nanogpt_b200/runtime -- shellcheck --version
```

If mise itself is not present in the rootfs, the rootfs builder must install it
from a pinned release and record the version in the rootfs manifest. The setup
wrapper must not curl-install mise into `HOME` during a training attempt.

The mise contract also has two modes:

- **prepare mode:** may download tool archives into `/project/mise/cache` and
  install them under `/project/mise/data` after `mise.toml` is trusted.
- **locked mode:** may only verify already-installed tool versions and shim
  paths. Full preflight uses locked mode. A missing shellcheck, shfmt, node, or
  future declared tool is a tool-env blocker, not a reason to install during a
  full attempt.

### Rootfs Base Packages

The rootfs base image should provide only system-level runtime and build tools:

- bash, coreutils, git, curl, ca-certificates, rsync;
- gcc, g++, clang, llvm, libclang, pkg-config;
- cmake, ninja;
- CUDA synth toolchain under `/opt/cuda-synth`;
- uv;
- mise;
- Python and the B200-capable Torch stack from the base image.

Python experiment dependencies belong in `/project/venvs/b200-runtime`, not in
the rootfs base image, unless they are needed to build the rootfs itself.

## Lifecycle

### 1. Survey Host and Resolve Local Environment

Host orchestration command:

```bash
experiments/modded_nanogpt_b200/runtime/materialize_runtime.py \
  survey-host \
  --local-environment experiments/modded_nanogpt_b200/runtime/local-environment.local.yaml \
  --output experiments/modded_nanogpt_b200/results/<run_id>/runtime/host_capability_report.json
```

This command records host capability and validates
`local_environment.schema.json`. It does not launch training or run package
installs. The local environment file may be created from the tracked example,
but the filled file remains ignored because it contains absolute host paths.

### 2. Build or Select Rootfs

Host orchestration command:

```bash
scripts/rootfs/build_rootfs.sh --dest scripts/rootfs/rootfs
```

This legacy command is sufficient only for the current compatibility rootfs. It
is not production-compliant until the builder also pins the base image by
digest, pins uv and mise, pre-creates the read-only bind targets, writes the
manifest below, and verifies the exported runtime.

Target command after managed-store support exists:

```bash
scripts/rootfs/build_rootfs.sh --store scripts/rootfs/store --activate
```

Required output:

- rootfs ownership marker;
- rootfs manifest with base image digest, installed apt packages, uv version,
  mise version, Python version, Torch version, CUDA runtime, Triton version,
  CUDA synth package versions, build time, and builder script commit;
- selected rootfs record;
- schema-valid `rootfs_manifest.schema.json` report for the selected rootfs.

### 3. Sync Python Dependencies

Host orchestration command:

```bash
experiments/modded_nanogpt_b200/runtime/sync_python_env.sh
```

This command re-enters rootfs and writes:

```text
.cache/torchtitan-rootfs/modded_nanogpt_b200/venvs/b200-runtime/
.cache/torchtitan-rootfs/modded_nanogpt_b200/logs/python-sync.json
```

The sync report must include:

- uv version;
- Python executable and prefix;
- installed distributions and versions;
- proof that Torch, CUDA, and Triton versions still match the full-mode
  contract after sync;
- schema validation against `python_env_report.schema.json`.

### 4. Sync Non-Python Tools

Host orchestration command:

```bash
experiments/modded_nanogpt_b200/runtime/sync_tools.sh
```

This command re-enters rootfs and writes:

```text
.cache/torchtitan-rootfs/modded_nanogpt_b200/mise/
.cache/torchtitan-rootfs/modded_nanogpt_b200/logs/tool-sync.json
```

The sync report must include each tool name, requested version, resolved
version, executable path, and checksum when mise exposes one. It must validate
against `tool_env_report.schema.json`.

### 5. Prepare Sources and Data

Source and data wrappers continue to be rootfs-aware:

```bash
experiments/modded_nanogpt_b200/fetch_upstream.sh
experiments/modded_nanogpt_b200/prepare_data.sh --mode full --verify-sha ...
```

Networked mode is allowed here. The output manifests are the handoff to offline
training.

### 6. Materialize Attempt Runtime

Host orchestration command:

```bash
experiments/modded_nanogpt_b200/runtime/materialize_runtime.py \
  materialize \
  --profile experiments/modded_nanogpt_b200/runtime/profiles/full-b200-fa2-triton.yaml \
  --local-environment experiments/modded_nanogpt_b200/runtime/local-environment.local.yaml \
  --run-id <run_id> \
  --attempt-id <attempt_id> \
  --result-dir experiments/modded_nanogpt_b200/results/<run_id> \
  --output experiments/modded_nanogpt_b200/results/<run_id>/runtime/materialized_runtime.json
```

Materialization resolves logical paths, assigns canonical env values, records
network mode, names package/tool reports, names source/data manifests, and
writes the expected inner argv. It must fail before preflight if any absolute
path appears in the declared profile, if any resolved path escapes its declared
root, if the selected result directory collides with stale attempt content, or
if the selected profile cannot support the requested claim label.

### 7. Emit and Verify Bwrap Plan

Host orchestration command:

```bash
TORCHTITAN_ROOTFS_EMIT_PLAN_ONLY=1 \
TORCHTITAN_ROOTFS_MATERIALIZED_CONFIG=experiments/modded_nanogpt_b200/results/<run_id>/runtime/materialized_runtime.json \
TORCHTITAN_ROOTFS_PLAN_OUTPUT=experiments/modded_nanogpt_b200/results/<run_id>/runtime/bwrap_plan.json \
scripts/rootfs/enter_rootfs.sh -- true
```

The payload is not executed. The emitted plan is validated against
`bwrap_plan.schema.json` and compared with `materialized_runtime.json`. Full
mode fails closed if the plan is missing, schema-invalid, or has a different
digest immediately before launch.

### 8. Preflight

Full preflight must run in the same rootfs mode and package environment that
will launch training:

```bash
experiments/modded_nanogpt_b200/run_preflight.sh --mode full ...
```

Required checks:

- rootfs sentinel: env marker, cwd, sentinel path;
- schema validation for every launch-governing artifact;
- selected rootfs ID and rootfs manifest;
- active Python is the uv-managed runtime Python;
- uv, pip, and mise cache dirs are under `/project`;
- direct dependency imports pass;
- Torch, CUDA runtime, and Triton versions match the pinned contract;
- exactly the declared active allocation is visible inside rootfs; for the
  current NanoGPT trial this is two B200 GPUs;
- `/dev/shm`, `/project/tmp`, `/project/scratch`, data filesystem, result
  filesystem capacity floors pass;
- NCCL is available and no diagnostic skip is active;
- source commit/provenance is valid;
- full data manifest has the expected token budget, shard count, byte count,
  source commit, and SHA verification;
- `DATA_PATH` exposes both `data/fineweb10B/fineweb_train_*.bin` and
  `data/fineweb10B/fineweb_val_*.bin` from the source process cwd;
- offline mode is selected for full training, or the artifact labels the run as
  diagnostic;
- `runtime_verification.json` exists, uses the current schema version, and has
  `ok=true`;
- `training_launch_allowed=true` for any path that starts `torchrun`.

### 9. Full Attempt

Full launch remains:

```bash
experiments/modded_nanogpt_b200/run_speedrun.sh --mode full ...
```

This wrapper is the only supported full-attempt launch entrypoint. Direct
`torchrun`, direct `python train_gpt.py`, upstream `sources/.../run.sh`, or an
interactive rootfs shell that starts training manually are invalid full attempts
and cannot be counted as launch evidence. If an alternate entrypoint is added,
it must call the same schema verifiers and write the same attempt-local reports
before it can start training.

The launcher must record:

- rootfs store ID or legacy rootfs marker;
- rootfs manifest path and digest;
- network mode;
- Python executable, prefix, and installed package report digest;
- mise tool report digest;
- all canonical cache, temp, scratch, home, data, and result paths;
- `/dev/shm`, `/project/tmp`, `/project/scratch`, data, and result filesystem
  capacity snapshots;
- host NVIDIA driver library provenance and `nvidia-smi` output;
- runtime verifier report digests;
- the effective environment digest used by the training subprocess, matched
  against `runtime_verification.json` and `command.env.json`;
- `command.env.json` with redacted environment metadata and no secret or
  authorization-token values;
- optional legacy text `command.env` compatibility output with the same
  redaction boundary, which must not replace `command.env.json`.

The launcher must refuse to start `torchrun` when any current-version verifier
report is missing, schema-invalid, stale for the current `run_id`/`attempt_id`,
reports `ok=false`, or reports `training_launch_allowed=false`.

### 10. Summarization

Parsing and summarization continue to run inside rootfs. A summary is invalid
for full baseline statistics unless it includes the rootfs runtime evidence
listed above.

## Verification Gates for Implementation

Unit tests should cover:

- schemas reject missing required fields and unknown top-level fields;
- schema-version drift is blocked for full launch;
- canonical env generation rejects paths outside the runtime state root;
- generated bwrap args include rootfs, checkout, state, cache, scratch, temp,
  and `/dev/shm` mounts;
- offline mode omits network sharing and sets offline environment variables;
- full mode rejects `HOME=/root` and missing `TMPDIR`;
- uv environment wrapper refuses upstream `requirements.txt`;
- direct dependency set excludes Torch and Triton;
- mise config path and data/cache dirs are under `/project`;
- preflight reports capacity failures with distinct phases;
- launch readiness cannot go green without `runtime_verification.ok=true`;
- missing authorization keeps a non-skip full attempt out of
  `launch_ready_attempts`; explicit `skip_run=true` may still produce
  `ready_to_launch=true` prerequisite evidence, but it remains excluded from
  baseline stats and non-skip launch-ready rows;
- run summaries reject full attempts missing rootfs runtime evidence.

Integration checks should run through the rootfs wrapper:

```bash
experiments/modded_nanogpt_b200/run_preflight.sh --mode diagnostic --skip-gpu
experiments/modded_nanogpt_b200/runtime/sync_python_env.sh --dry-run
experiments/modded_nanogpt_b200/runtime/sync_tools.sh --dry-run
```

Full GPU checks remain authority-gated by the existing B200 launch policy.

## Implementation Ticket State

The recommended ticket split below is implemented for the non-launch
foundation, with the remaining package-lock population and any full-launch
runtime repairs deferred to the authority-gated launch path:

1. Add `scripts/rootfs/runtime_env.sh` and rootfs state-root mounts. Done.
2. Add managed rootfs store support to `build_rootfs.sh` and
   `rootfs_target.sh`, including `--store`, `--activate`, selected-rootfs
   resolution, rootfs manifests, pinned uv/mise installs, and pre-created
   read-only bind targets. Done for the tested rootfs store surface.
3. Add checked-in schema files under
   `experiments/modded_nanogpt_b200/runtime/schemas/` plus schema-version tests.
   Done.
4. Add runtime verifier for env, mount, cache, scratch, `/dev/shm`, and network
   mode evidence. Done for the launch-governing runtime and command-env
   evidence used by the active harness.
5. Add uv-managed experiment Python environment files and sync wrapper. Done;
   the direct-only runtime lock is hash-populated, and offline mode rejects any
   future placeholder or hashless lock before invoking uv.
6. Add mise to the rootfs build and experiment tool sync wrapper. Done for the
   project-owned tool environment contract and `shellcheck=0.10.0` pin.
7. Wire `run_preflight.sh` and `run_speedrun.py` to the canonical runtime env
   and `${PYTHON}`. Done.
8. Gate launch readiness and `torchrun` on current-version verifier reports.
   Done.
9. Extend preflight and summarization evidence requirements. Done for the
   active two-GPU RSI foundation prerequisite and run-index surfacing.
10. Add offline rootfs mode and require it for full attempts. Done at the
   wrapper/policy level; the networked setup phase has populated the runtime
   lock required by offline package sync.

Fresh continuation verification recorded in
`.scratch/modded-nanogpt-b200/completion_audit.md` reported the combined
NanoGPT/rootfs non-launch owner suite as `410 passed, 2 skipped`.
The focused tracker guard now reports `49 passed`. Focused
optimized-kernel and diagnostic performance-probe tests reported
`13 passed in 3.03s`. The current strict prerequisite artifact validation
reported `ok=true`, 24 sidecars, and zero failed sidecars; active-job scan
reported `ok=true`, `active_job_count=0`, and `ignored_match_count=0`. The
current in-memory run index still reports `total_attempts=63`,
`baseline_stats.count=0`, `launch_prerequisite_attempts.count=4`, and
`launch_ready_attempts.count=0`. The latest strict runtime-env prerequisite row
remains the current validated handoff artifact; older prerequisite rows are
historical skip-run dry gates. Prior explicit-file Pyrefly over the
31-file static Python surface reported `0 errors`, text/lock hygiene for
runtime dependency files passed, and static verification reported
`Static verification passed for 114 file(s)`. A later rootfs all-files
pre-commit run passed with only the protected-branch hook skipped:
`SKIP=no-commit-to-branch pre-commit run --all-files`.

## Non-Goals

- Replacing the current launch authorization gate.
- Moving modded-nanogpt logic into TorchTitan core.
- Making Docker an equivalent training backend.
- Solving multi-node or scheduler integration.
- Cleaning existing generated result, data, cache, or rootfs directories.
