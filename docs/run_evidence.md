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
absolute `monotonic_ns`, `event_seq`, or `artifact_seq`. Inspect every expected
process outcome separately, or reduce them with the post-hoc aggregator below.

### Incident records

A row with `record_type="incident"` is an append-only record of a diagnosable
training failure or degraded-progress event. It is written to the same
`indexes/artifacts.<process_id>.jsonl` through the artifact append path, but it
is not an artifact transition and carries no `artifact_id` or `artifact_seq`.

Each incident row inherits the full structured-event envelope (`run_id`,
`attempt_id`, process fields, `event_seq`, both clocks, distributed context when
bound, and the current nested `phase`) and adds `schema_version=1`,
`record_type="incident"`, and:

- `incident_class`: one of the nine v1 fault-suite classes (`collective_hang`,
  `rank_death`, `compute_straggler`, `dataloader_straggler`, `nonfinite_loss`,
  `checkpoint_corruption`, `checkpoint_interruption`, `inconsistent_rank_config`,
  `hardware_event`). V1 wires only the `nonfinite_loss` emitter; the other eight
  are the classification enum only, not proven fault injections.
- `capture_state`: `normal`, `suspected`, `capturing`, `continue`, or
  `abort_and_preserve`.
- `policy`: `capture_before_abort`, `continue_bounded_warning`, or `abort_fatal`.
  These describe intended disposition only; v1 does not retry, recover, or
  quarantine.
- `summary` and always-present JSON `metadata`.

Optional progress-envelope fields are recorded only when supplied:
`detected_locus`, `attribution_confidence`, `step`, `last_operation`,
`useful_work_preserved`, and `terminal_disposition`. An uncollected optional
field is omitted, exactly like the optional `step`/`phase`. An
unknown-but-observed value is present with an explicit sentinel
(`detected_locus="unknown"`,
`attribution_confidence="collective_timeout_insufficient_evidence"`), mirroring
the `source.dirty=null` semantics. A recorded incident does not by itself prove
that recovery was attempted or that the run continued.

### Attempt outcome aggregation

`python -m torchtitan.observability.aggregate_outcome <attempt_dir>` reduces all
`processes/*/outcome.json` files into one immutable `aggregate_outcome.json`
sibling to `manifest.json`. It runs post-hoc, off the training path, with no
cross-rank barrier, so it preserves the no-central-collector contract.

The record has `record_type="attempt_outcome_aggregate"`, `run_id`/`attempt_id`
from the manifest, `expected_world_size`, `process_outcome_count`,
`missing_process_count`, the per-process `outcomes`, `first_failure` (the lowest
failed `global_rank`), `consistent_world_size`, and `reduced_wall_time_ns` (the
longest observed elapsed time). `aggregate_outcome` is `failed` if any process
failed, `interrupted` if none failed and any was interrupted, `succeeded` only
when every expected process succeeded with a consistent positive world size, and
`incomplete` otherwise. A missing process outcome or an inconsistent world size
is `incomplete`, never success. A second write raises unless `--force` replaces
it atomically.

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

- fault-injection certification of the other eight incident classes, incident
  emitters beyond the non-finite loss seam, and recovery, ETTR, or membership
  and restore-equivalence evidence;
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

### Producer roadmap for estimator core hooks

This roadmap is documentation only. It does not change schema version 1, does
not imply that these producers exist today, and does not authorize core code
changes, a central collector, optional dependencies, or training-control-flow
changes. Child tickets must implement each producer at its owning component
boundary and preserve the current v1 bundle contract.

The proposed fields below are the minimum useful core hooks for the
training-state estimator. Each field should be emitted only when the owning
component already has the source event or artifact in hand; otherwise the
offline analyzer should mark the evidence family absent instead of inventing
values.

- Topology fields:
  - `topology_epoch`: owner `torchtitan.distributed` mesh setup; Tier 0; low
    overhead risk when emitted once per mesh or topology change; source kind
    distributed binding or membership snapshot; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus artifact/event
    inspection; fallback absent means all topology-derived observations carry
    `quality=uncollected`; child ticket needed.
  - `device_uuid`: owner distributed device binding; Tier 0; low overhead risk
    when read once after device assignment; source kind distributed binding
    event; validation command `NGPU=2 ./run_train.sh` on CUDA hardware plus
    event inspection; fallback use `device_index` only and mark physical-device
    joins as incomplete; child ticket needed.
  - `mesh_axis_<axis>_{rank,size}`: owner distributed mesh setup; Tier 0; low
    overhead risk because current structured events already carry available
    axis ranks and sizes after binding; source kind structured event context;
    validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus structured-event
    inspection; fallback retain process-only joins and skip mesh graph edges;
    no child ticket needed for current axes unless new mesh axes are added.
  - `topology_source`: owner distributed setup or launcher metadata adapter;
    Tier 0 for launcher/static metadata, Tier 1 for external topology dumps;
    medium overhead risk if it shells out or queries hardware at startup;
    source kind topology snapshot artifact; validation command
    `NGPU=2 ./run_train.sh` with the source enabled plus artifact inspection;
    fallback topology source is `unknown` and physical-link confidence is
    unavailable; child ticket needed.

- Microbatch fields:
  - `step`: owner core `Trainer` train-step loop; Tier 0; low overhead risk
    because step identity is already tracked for logs and checkpoint cadence;
    source kind structured event context; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus structured-event
    inspection; fallback align by event order only and mark transaction
    observations under-specified; child ticket needed only for missing
    step-bound events.
  - `microbatch`: owner pipeline schedule or train-step loop; Tier 0 when PP is
    disabled or when emitted at optimizer-step granularity, Tier 1 for
    per-microbatch timing; medium overhead risk if emitted per microbatch on
    every rank; source kind structured train-step or schedule event; validation
    command `NGPU=2 ./run_train.sh` with a PP config plus event inspection;
    fallback aggregate only at optimizer-step scope and skip microbatch
    dependency edges; child ticket needed.
  - `microbatch_count`: owner pipeline schedule or train-step loop; Tier 0; low
    overhead risk when emitted once per optimizer step; source kind structured
    train-step event; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus event inspection;
    fallback infer nothing from config and mark microbatch completeness unknown;
    child ticket needed.
  - `gradient_accumulation_index`: owner train-step loop; Tier 0; low overhead
    risk when emitted once per local accumulation step; source kind structured
    train-step event; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus event inspection;
    fallback do not distinguish local accumulation work from optimizer-step
    work; child ticket needed.

- Collective fields:
  - `process_group_id`: owner distributed process-group creation or collective
    wrapper; Tier 0 for stable creation snapshots, Tier 1 for per-collective
    events; medium overhead risk until PyTorch exposes stable IDs for every
    group; source kind process-group membership snapshot or collective event;
    validation command `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus
    membership/event inspection; fallback join collectives only by rank, phase,
    and time window; child ticket needed.
  - `process_group_epoch`: owner distributed process-group lifecycle; Tier 0;
    low overhead risk when incremented only on group creation, destruction, or
    elastic membership change; source kind process-group membership snapshot;
    validation command `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus
    snapshot inspection; fallback transaction-risk output reports membership
    epoch unavailable; child ticket needed.
  - `collective_name`: owner collective wrapper, Flight Recorder join, or
    profiler summary adapter; Tier 1; medium overhead risk for per-collective
    logging, low if derived offline from existing traces; source kind
    collective event or trace summary; validation command
    `NGPU=2 ./run_train.sh` with profiler/Flight Recorder enabled plus summary
    inspection; fallback collective residuals remain phase-level only; child
    ticket needed.
  - `collective_seq`: owner collective wrapper or trace summary adapter; Tier 1;
    medium overhead risk for online per-collective counters; source kind
    collective event or trace summary; validation command
    `NGPU=2 ./run_train.sh` with profiler/Flight Recorder enabled plus summary
    inspection; fallback cannot separate arrival skew from repeated same-name
    collective calls; child ticket needed.

- Checkpoint fields:
  - `checkpoint_id`: owner checkpoint manager; Tier 0; low overhead risk when
    derived from the native checkpoint path already indexed by v1; source kind
    checkpoint artifact lifecycle row; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` with checkpointing enabled
    plus artifact-index inspection; fallback use artifact path as an opaque
    checkpoint reference; child ticket needed for semantic identity beyond
    current artifact IDs.
  - `checkpoint_operation`: owner checkpoint manager; Tier 0; low overhead risk
    because save/load direction is already known at declaration; source kind
    checkpoint artifact lifecycle metadata; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` with checkpointing enabled
    plus artifact-index inspection; fallback infer only from artifact relation
    and mark operation confidence low; child ticket needed.
  - `checkpoint_parent_id`: owner checkpoint manager; Tier 0; low overhead risk
    when recorded once per save after successful load or previous save; source
    kind semantic checkpoint manifest; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` resume from a prior
    checkpoint plus manifest inspection; fallback lineage graph starts a new
    unknown-parent root; child ticket needed.
  - `checkpoint_state_inventory`: owner checkpoint manager; Tier 1; medium
    overhead risk if it enumerates shards or state entries during a hot path;
    source kind semantic checkpoint manifest; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` checkpoint save/load smoke
    plus manifest inspection; fallback report checkpoint presence without
    restore-equivalence or shard-completeness claims; child ticket needed.

- Data cursor fields:
  - `data_cursor_id`: owner dataloader or dataset state component; Tier 0; low
    overhead risk when emitted at checkpoint or step boundary only; source kind
    data-cursor snapshot or checkpoint metadata; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` with checkpointing enabled
    plus metadata inspection; fallback transaction reasoning reports data
    position unavailable; child ticket needed.
  - `data_cursor_position`: owner dataloader or dataset state component; Tier 0
    at checkpoint boundary, Tier 1 at regular step intervals; medium overhead
    risk if serialized frequently or for large sampler state; source kind
    data-cursor snapshot; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` checkpoint/resume smoke plus
    snapshot inspection; fallback cannot prove replay position or sample
    freshness; child ticket needed.
  - `rng_state_id`: owner checkpoint manager with dataloader and RNG state
    providers; Tier 0 at checkpoint boundary; medium overhead risk if content
    digests are computed synchronously; source kind checkpoint semantic
    manifest; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` checkpoint/resume smoke plus
    manifest inspection; fallback checkpoint lineage cannot claim deterministic
    resume equivalence; child ticket needed.

- Replica and process-group epoch fields:
  - `replica_id`: owner TorchFT or replica manager, absent for non-replicated
    core runs; Tier 0; low overhead risk when copied from replica setup
    metadata; source kind replica membership snapshot; validation command
    TorchFT smoke with run evidence enabled plus snapshot inspection; fallback
    treat the process as a single unnamed replica; child ticket needed.
  - `replica_epoch`: owner TorchFT or replica manager; Tier 0; low overhead risk
    when emitted only on replica membership or role changes; source kind
    replica membership snapshot; validation command TorchFT failover or restart
    smoke plus snapshot inspection; fallback transaction-risk output reports
    replica epoch unavailable; child ticket needed.
  - `replica_role`: owner TorchFT or replica manager; Tier 0; low overhead risk
    when emitted with replica membership; source kind replica membership
    snapshot; validation command TorchFT smoke plus snapshot inspection;
    fallback do not infer primary/standby or per-replica checkpoint ownership;
    child ticket needed.

- Transaction fields:
  - `transaction_id`: owner train-step loop or checkpoint manager; Tier 0; low
    overhead risk when assigned once per optimizer step and reused by related
    checkpoint events; source kind structured train-step event and checkpoint
    manifest; validation command `COMM_MODE=fake_backend NGPU=2 ./run_train.sh`
    plus event/artifact inspection; fallback transaction observations remain
    advisory and unjoined; child ticket needed.
  - `transaction_state`: owner train-step loop; Tier 0; low overhead risk for
    coarse `started`, `optimizer_committed`, and `checkpointed` states; source
    kind structured train-step event; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus event inspection;
    fallback cannot distinguish speculative work from committed optimizer
    state; child ticket needed.
  - `optimizer_step_committed`: owner optimizer/train-step loop; Tier 0; low
    overhead risk when emitted once after successful optimizer and scheduler
    updates; source kind structured train-step event; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` plus event inspection;
    fallback transaction-risk output cannot claim whether the step committed;
    child ticket needed.
  - `latest_recoverable_checkpoint_id`: owner checkpoint manager; Tier 0; low
    overhead risk when updated after checkpoint completion; source kind
    checkpoint artifact lifecycle row or semantic manifest; validation command
    `COMM_MODE=fake_backend NGPU=2 ./run_train.sh` checkpoint save smoke plus
    artifact/manifest inspection; fallback recovery horizon remains unknown;
    child ticket needed.

- Incident fields:
  - `incident_id`: owner run-evidence incident producer at the component that
    observes the fault; Tier 0 for fatal correctness and process outcomes,
    Tier 2 for anomaly-triggered diagnostics; low overhead risk for one record
    per incident; source kind typed incident record; validation command focused
    failure injection or existing failing config plus incident-record
    inspection; fallback derive only coarse incidents from process outcomes and
    artifact failures; child ticket needed.
  - `incident_type`: owner observing component; Tier 0; low overhead risk for
    enum classification; source kind typed incident record; validation command
    focused failure injection plus incident-record inspection; fallback classify
    as `unknown_failure` from terminal outcome only; child ticket needed.
  - `incident_severity`: owner observing component; Tier 0; low overhead risk
    for bounded enum values; source kind typed incident record; validation
    command focused failure injection plus incident-record inspection; fallback
    analyzer uses conservative unknown severity; child ticket needed.
  - `incident_window_ns`: owner incident producer or offline analyzer; Tier 2
    if anomaly-triggered, Tier 0 only for fixed terminal-failure windows; low
    overhead risk when storing two timestamps; source kind typed incident
    record or analyzer report; validation command focused failure injection
    plus report inspection; fallback analyzer uses configured default windows;
    child ticket needed.
  - `incident_source_artifact_id`: owner incident producer; Tier 0 for
    artifact-failure incidents, Tier 2 for diagnostic artifacts; low overhead
    risk when referencing existing artifact rows; source kind typed incident
    record; validation command artifact-failure injection plus index and
    incident inspection; fallback incident is not joined to native artifacts;
    child ticket needed.

A `local_tensor` launch is useful for checking entrypoint and evidence bootstrap
plumbing, but the entrypoint returns before constructing `Trainer`. It is not
distributed-training, numerical, scale, convergence, overhead, correctness, or
reliability evidence. See [Debugging](debugging.md) for its exact scope.
