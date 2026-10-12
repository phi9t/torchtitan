# Experiment Execution Design

Date: 2026-08-20
Status: draft design

## Purpose

Fortify the shared execution substrate for repo-local research programs. This
substrate should make it easy to run experiments inside the required rootfs,
record immutable attempt evidence, attach artifacts, classify outcomes, and
reuse preflight behavior across programs without copying shell and JSON logic.

The intended home is `torchtitan/experiments/execution/`, with thin wrappers in
top-level `experiments/<program>/`.

## Current State

The existing execution layer defines useful primitives:

- `RunDeclaration`: immutable scientific intent with a normalized digest.
- `Attempt`: one operational execution of a declaration.
- `StageSpec`: one declared stage such as doctor, preflight, prepare, train,
  evaluate, ingest, or report.
- `StageEvent`: one start event and exactly one terminal event per invocation.
- `ArtifactRef`: provenance and freshness metadata.
- `AttemptOutcome`: terminal attempt summary.
- preflight and rootfs helpers under `torchtitan/experiments/execution/`.

Several programs already implement overlapping lifecycle behavior:

- NanoGPT B200 has rootfs re-entry, runtime verification, active-job scans,
  launch readiness, schema validation, parsing, and summaries.
- Countdown has rootfs runners, preflight, staged artifacts, split registries,
  training, export, and evaluation reports.
- Scaffold-to-policy has rootfs-managed benchmark runners, runtime doctor,
  external harness probes, report inputs, and result manifests.
- Qwen3 FineWeb has a smaller offline run/sweep harness.
- RL batch-invariance has rootfs launch and summary logic for trainer/generator
  parity evidence.

## Production Target

The execution substrate should expose one small interface:

```text
declare run -> begin attempt -> run stages -> attach artifacts -> finish attempt
```

Callers should not manually implement:

- rootfs entry checks;
- run and attempt directory layout;
- command argv and environment capture;
- start and terminal stage events;
- atomic JSON writes;
- freshness and reuse classification;
- terminal outcome derivation;
- report-input attachment.

Program-specific code should provide:

- declaration fields;
- stage list;
- adapter choice;
- domain-specific validators;
- claim-classification rules;
- report templates.

## Module Interfaces

### Run Declaration Store

Responsibilities:

- normalize declaration content;
- compute declaration digest;
- reject run ID reuse with different content;
- allow explicit resume when digest matches.

Interface:

```text
record_declaration(results_root, declaration) -> declaration_record
load_declaration(results_root, run_id) -> declaration_record
```

### Attempt Store

Responsibilities:

- create immutable attempt directories;
- link parent attempts;
- write `manifest.json`;
- preserve append-only event streams;
- write terminal `outcome.json` exactly once.

Interface:

```text
begin_attempt(declaration, attempt_id, parent=None) -> RunAttempt
attach_attempt(run_id, attempt_id) -> RunAttempt
finish_attempt(report_input, evaluations, outcome) -> AttemptOutcome
```

### Stage Runner

Responsibilities:

- run commands through an adapter;
- record start and terminal events;
- capture return code, failure type, timestamps, and selected sidecar files;
- prevent multiple terminal events for one invocation.

Interface:

```text
run_stage(StageSpec, extra=None, extra_files=None) -> StageEvent
```

### Rootfs Adapter

Responsibilities:

- decide whether the current process is inside the trusted rootfs;
- re-enter with a stable cwd, environment, and command argv;
- emit a plan-only artifact for audit;
- fail closed when a rootfs-required stage cannot prove the boundary;
- report a `RootfsIdentity` record rich enough for downstream claim validation,
  not just a boolean sentinel.

Interface:

```text
rootfs_required(script_rel, argv) -> exec_or_continue
rootfs_identity() -> RootfsIdentity
emit_rootfs_plan(command) -> plan_json
```

`RootfsIdentity` is part of the evidence contract. The minimum schema is:

- `inside_rootfs`: derived from `TORCHTITAN_IN_ROOTFS=1`;
- `workspace_cwd`: expected to be `/workspace/torchtitan` for claim-bearing
  experiment work;
- `workspace_sentinel`: path and digest or stat metadata for
  `scripts/rootfs/enter_rootfs.sh` as seen from inside the rootfs;
- `python_executable`: `sys.executable` for Python stages;
- `python_prefix` and `python_version`: enough to distinguish host Python from
  the rootfs environment;
- `rootfs_store_id` or `rootfs_manifest_digest`: stable identity for the mounted
  rootfs when the provisioning layer exposes it;
- `entry_script`: relative wrapper path used for re-entry;
- `original_argv` and `effective_argv`: host request and in-rootfs command;
- `cwd_before_entry` and `cwd_after_entry`;
- `env_allowlist` and `env_redaction_policy`: names retained, values redacted,
  and values hashed.

If a field cannot be collected, the adapter records `uncollected` with a reason.
It must not silently downgrade to the older environment-only check.

### Artifact Registry

Responsibilities:

- register produced, reused, resumed, imported, and external artifacts;
- record freshness independently from work status;
- include digests where byte identity matters;
- preserve source paths for state-estimator ingestion;
- define the serialization and retention policy for command outputs and
  sidecars.

Interface:

```text
attach_artifact(kind, path, work_status, freshness, digest=None, metadata=None)
```

Artifact references serialize as JSON objects with:

- `kind`: domain label such as dataset_manifest, stdout_tail, checkpoint,
  profile, report_input, summary, or external_measurement;
- `uri`: repo-local path, external URI, or logical imported location;
- `work_status`: produced, reused, resumed, imported, or external;
- `freshness`: fresh, reused, stale, unknown, or not_applicable;
- `digest`: algorithm and value when byte identity affects the claim;
- `size_bytes` and `mtime_ns` for local files;
- `producer_stage_id`;
- `schema_version` for structured artifacts;
- `redaction`: none, value_redacted, tokenized, or digest_only;
- `metadata`: domain-specific extensions.

Stage stdout and stderr are artifacts, not implicit console history. The runner
keeps bounded inline tails in the stage event, stores full streams or declared
tails as sidecar files according to the stage policy, redacts environment
secrets before serialization, and records digests for every sidecar used in a
claim. A report may cite only registered artifacts.

## Schema Policy

Execution schemas should be versioned at the shared layer when they describe
shared concepts:

- run declaration;
- attempt manifest;
- stage event;
- artifact reference;
- condition status;
- attempt outcome;
- rootfs identity;
- command argv/env capture.

Program schemas may extend these with domain fields:

- NanoGPT lane, arm, backend, source patch, and speedrun summary;
- Countdown split, verifier, scaffold subset, and pass@k;
- scaffold benchmark identity, harness version, and task-level reports;
- Qwen mesh, dataset, checkpoint, and promotion proof.

Shared schema validation should be a library call. Program wrappers may expose
CLI entrypoints around it.

## Error Handling

- Invalid declaration reuse is a hard error.
- Missing rootfs identity for a rootfs-required stage is a hard error.
- Missing optional evidence is represented as `uncollected`, not fabricated.
- A blocked stage is different from a failed stage; launch authorization,
  active-job policy, missing assets, and unsupported hardware should be blocked
  when no attempted workload ran.
- A resume after a terminal outcome creates a new attempt with parent linkage.
- A summary cannot promote an attempt if required artifact references are
  missing, stale, or inconsistent.

## Verification Ladder

- Unit: model validation, digest stability, atomic writes, terminal-event rules,
  rootfs plan emission, and artifact freshness logic.
- Shell: host invocation re-enters rootfs; rootfs invocation continues without
  recursion; wrappers preserve argv.
- Integration: one fake stage produces a complete attempt bundle with manifest,
  process events, artifacts, report input, and outcome.
- Program adoption: NanoGPT, Countdown, scaffold-to-policy, and Qwen runners use
  the shared interface for at least one stage each.
- Regression: old program-specific artifacts can still be summarized or are
  rejected with a clear migration error.

## Migration Plan

1. Freeze the current execution dataclasses and add schema compatibility tests.
2. Move rootfs identity and plan emission into one shared helper.
3. Convert NanoGPT B200 wrappers to call the shared rootfs adapter.
4. Convert NanoGPT attempt and summary files to attach shared artifact refs.
5. Convert Countdown and scaffold rootfs runners one stage at a time.
6. Convert Qwen FineWeb run/sweep artifacts to the shared declaration/attempt
   shape.
7. Add state-estimator ingestion over shared attempt bundles.

## Open Design Decisions

- Whether shell wrappers should stay per program or be generated from stage
  declarations.
- Whether JSON Schemas should be authoritative with Python dataclasses generated
  from them, or whether Python dataclasses should remain authoritative with
  schema snapshots.
- Which migration promise to make for historical result directories.
