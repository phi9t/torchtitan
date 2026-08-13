# Core Run Evidence

Core `torchtitan.train` records a versioned, repo-local evidence bundle for each
process in a training run attempt. The bundle gives native TorchTitan artifacts
a shared run and attempt identity without copying or converting those artifacts.
It is enabled by default.

This document describes the implemented schema version 1 contract. The broader
research architecture in the linked observability documents is a roadmap, not a
description of v1.

## Configuration and layout

The output root defaults to `./outputs`. Run evidence is configured on the core
trainer config:

```text
--dump_folder=./outputs
--run-evidence.enable
--run-evidence.folder=run_evidence
```

`run_evidence.folder` must be one safe relative path segment. The default layout
is therefore:

```text
dump_folder/run_evidence/run_id/attempt_id/
  manifest.json
  indexes/artifacts.<process_id>.jsonl
  processes/<process_id>/outcome.json
```

For core training, `process_id` is
`trainer.core.global_rank_<rank-padded-to-six-digits>`. Each process owns its
index and outcome file. Processes in the same attempt share the manifest.

Disable the recorder with `--run-evidence.no-enable`. This disables the bundle
and makes the artifact, event-context, distributed-binding, and phase-binding
facades no-ops. It does not disable structured logging, TensorBoard, profiling,
checkpoints, or Flight Recorder themselves; those producers continue according
to their own configuration but are not indexed by run evidence.

`dump_folder` must be a local filesystem path while run evidence is enabled.
Remote dump roots such as `s3://...` are rejected. An individual indexed
artifact may still use a URI.

## Run and attempt identity

`run_id` identifies the logical training experiment. `attempt_id` identifies
one launch or elastic restart. Identity is resolved independently for each ID
in this order:

1. `TORCHTITAN_RUN_ID` or `TORCHTITAN_ATTEMPT_ID`, respectively.
2. A nonempty `TORCHELASTIC_RUN_ID` other than the literal value `none`.
3. A generated UUID for a single-process run only.

For `WORLD_SIZE != 1`, both identities must resolve from launcher-provided
values; otherwise startup fails. `run_train.sh` and `multinode_trainer.slurm`
preserve nonempty `TORCHTITAN_*` values and generate an independent UUID for
each missing value. `run_train.sh` also uses the run ID as the torchrun
rendezvous ID. Launchers pass the attempt base unchanged.

The recorder appends `-restart-<TORCHELASTIC_RESTART_COUNT>` to the resolved
attempt base when the restart count is greater than zero. Count zero has no
suffix. For example, attempt base `launch-17` with restart count `2` becomes
`launch-17-restart-2`. Run, attempt, role, and actor identifiers accept only
letters, digits, `.`, `_`, and `-`, must begin with a letter or digit, and are
limited to 128 characters.

Do not reuse the same attempt and process identity. The process index is opened
exclusively, and a collision fails startup. A separately launched retry should
receive a new attempt base; an elastic restart receives the restart suffix.

## Manifest

`manifest.json` is created once and is immutable. Later processes may join the
attempt only when all required manifest fields match exactly:

- `schema_version` (currently `1`), `run_id`, and `attempt_id`;
- `config.normalized`, the complete normalized job config, and
  `config.sha256`, the SHA-256 of its canonical JSON;
- `source.revision` and `source.dirty` from Git;
- `command`, formed by joining the process `sys.argv` values with spaces; and
- `runtime.python` and `runtime.torch`.

The manifest does not contain an environment allowlist or a dump of the full
environment. That is not a secrecy guarantee: the complete normalized config
and joined command can contain tokens, credentials, signed URLs, or other
sensitive values supplied through config or CLI arguments. Review both before
sharing a bundle. Artifact contents, artifact metadata, structured messages,
paths, host names, PIDs, and truncated exception messages may also be sensitive.

If the Git queries fail, v1 writes `source.revision="unknown"` and
`source.dirty=null`. The null value means collection was unsuccessful; it is
not proof of either a clean or dirty checkout.

## Record schemas and correlation

The record types deliberately have different envelopes. Do not assume every
record has process identity or both clocks.

### Artifact index rows

Each line of `indexes/artifacts.<process_id>.jsonl` is an artifact transition.
It includes:

- `schema_version=1`, `evidence_schema_version=1`, and
  `record_type="artifact"`;
- `run_id`, `attempt_id`, and process fields: `process_id`, `role`, `actor_id`,
  `host_name`, `pid`, `global_rank`, `local_rank`, and `world_size`;
- `artifact_id`, `producer`, `kind`, `relation`, `state`, `path`, `path_type`,
  and JSON `metadata`;
- process-local `artifact_seq`, `wall_time_ns`, and `monotonic_ns`; and
- optional `step` and current nested `phase`.

`artifact_id` is deterministic over schema version, run, attempt, producer,
kind, relation, and normalized path. It intentionally does not include process
identity, so separate ranks can calculate the same identity for the same native
artifact description. `artifact_seq` orders transitions in one process index;
it is not a global sequence.

Artifact paths have one of three types:

- `dump_relative`: a local path within `dump_folder`, stored relative to it;
- `external_absolute`: a local path outside `dump_folder`, stored as a resolved
  absolute path; or
- `uri`: a string beginning with a URI scheme such as `s3://`, retained as
  supplied.

### Structured event rows

When the core entrypoint's recorder is active, structured JSONL rows retain the
logger's existing fields and add `evidence_schema_version=1`, `run_id`,
`attempt_id`, the process fields above, process-local `event_seq`,
`wall_time_ns`, and `monotonic_ns`. After `Trainer` constructs its distributed
meshes, later events also receive device, parallelism-degree, and available
`mesh_axis_<axis>_{rank,size}` fields. Events produced before that binding do
not have those distributed fields.

The structured logger separately assigns `seq_id` in each formatter instance.
`seq_id` is a JSONL-handler sequence, while `event_seq` is assigned by the
process-local evidence recorder before handlers run. With custom or multiple
handlers, their `seq_id` values are not a substitute for `event_seq`.

Structured events do not carry `artifact_id`. To locate their artifact:

1. Match the event's `run_id`, `attempt_id`, and `process_id` to the process
   artifact index.
2. Find the `producer="structured_logger"`,
   `kind="torchtitan.structured_events"` row whose normalized `path` names the
   JSONL file being read.
3. Use `event_seq` for recorder-local event order and `seq_id` for order within
   that particular formatted JSONL stream.

### Process outcomes

`processes/<process_id>/outcome.json` is an immutable per-process terminal
record. It includes `schema_version=1`, `evidence_schema_version=1`,
`record_type="process_outcome"`, run/attempt and process identity,
`outcome` (`succeeded`, `failed`, or `interrupted`), and
`elapsed_monotonic_ns`. Failed and interrupted outcomes add `exception_type`
and an `exception_message` truncated to 512 characters.

An outcome uses elapsed monotonic time. It does not contain `wall_time_ns`, an
absolute `monotonic_ns`, `event_seq`, or `artifact_seq`. There is no top-level
multi-process outcome in v1; inspect every expected process outcome separately.

## Artifact lifecycle and native kinds

The bundle as a whole is not immutable. The manifest and each process outcome
are immutable, artifact indexes append a row for every transition, and native
files or directories can evolve while their producer is active.

The states are:

- `declared`: the producer intends to create or update the artifact;
- `complete`: the producer completed its native close, export, save, or load;
- `failed`: the producer operation failed; and
- `retired`: a declared or completed artifact is no longer active.

Valid transitions are `declared -> complete`, `declared -> failed`,
`declared -> retired`, and `complete -> retired`. A producer may directly
record a completed artifact. Completing a local artifact requires its indexed
path to exist. V1 records lifecycle state but does not enforce retention or
delete artifacts.

Core producers currently register these native kinds:

| Kind | Producer and lifecycle |
| --- | --- |
| `torchtitan.structured_events` | The default structured JSONL handler declares its file at open and completes it only after the native file handler closes successfully. |
| `tensorboard.event_stream` | TensorBoard declares its event directory at writer construction and completes or fails it when the writer closes. |
| `pytorch.profiler.trace` | The profiler declares each gzip Chrome trace before export and completes or fails it after export and any post-processor. |
| `pytorch.cuda.memory_snapshot` | The memory profiler declares each pickle before serialization and completes or fails it after writing. |
| `pytorch.flight_recorder.dump` | Distributed setup declares the configured native dump prefix when the Flight Recorder buffer is enabled. It remains `declared`; a row is not evidence that a timeout occurred or a dump file completed. |
| `torchtitan.checkpoint` | Checkpoint save outputs and load inputs declare before native DCP or Hugging Face work, then complete or fail. Async output completion occurs only after the save future resolves. |

Checkpoint rows describe a native checkpoint path and storage operation. V1
does not create a semantic checkpoint manifest, content digests, parent lineage,
state inventory, data-position proof, or restore-equivalence result.

## Write and failure behavior

There is no central collector. Every process writes its own index, flushes each
appended row synchronously, and publishes its own outcome. The flush is a Python
file flush, not an `fsync` durability guarantee. Structured logging handlers
also run synchronously in the caller, and custom handlers can perform blocking
I/O. No v1 overhead claim follows from this implementation.

Invalid configuration, missing multi-process identity, a manifest mismatch,
index collision, invalid artifact transition, or an evidence write failure
normally fails the process rather than silently dropping evidence. During an
already active training exception, an artifact append `EvidenceWriteError`, an
outcome write failure, or an index-close failure is logged where possible so
the original exception remains primary. Producer failure paths similarly make
a best effort to record `failed` without replacing the producer's original
error. A process can therefore terminate without a final artifact transition or
outcome when storage itself fails; treat missing terminal records as incomplete
evidence, not success.

## Adoption scope and deferred milestones

Only core `torchtitan.train` installs `RunEvidence`, before it initializes the
structured logger. Constructing `Trainer` or an artifact-producing component
directly remains a no-op for evidence unless the caller installs a recorder.
Online RL, Forge, and repo-local offline research programs have not adopted the
recorder.

TorchFT configurations that launch through core `torchtitan.train` inherit the
core recorder. Their full persistent checkpoint uses the inherited checkpoint
artifact lifecycle. TorchFT's private per-replica dataloader checkpoint is not
indexed.

The following remain later milestones and must not be inferred from a v1
bundle:

- top-level multi-process outcome aggregation and typed incident records;
- external tool or hardware ingestion, including DCGM, EUD, py-spy, NCCL RAS,
  `nccl-tests`, SuperBench, and Nsight;
- anomaly triggers, capture tiers and policy, budgets, and retention
  enforcement;
- completed Flight Recorder dump discovery and offline Flight Recorder joins;
- semantic checkpoint manifests, digests, validation, and lineage;
- offline analyzers, derived timelines, and dashboards;
- automatic retry or recovery and fleet quarantine; and
- measured Tier-0 throughput, memory, host CPU, artifact-volume, detection, or
  correctness proof.

A `local_tensor` launch is useful for checking entrypoint and evidence bootstrap
plumbing, but the entrypoint returns before constructing `Trainer`. It is not
distributed-training, numerical, scale, convergence, overhead, correctness, or
reliability evidence. See [Debugging](debugging.md) for its exact scope.
