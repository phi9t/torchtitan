# Temporal durable execution for scaffold-to-policy

Date: 2026-08-12
Status: primary-source design note; no implementation is implied

## Conclusion

Use Temporal as a host-side, durable control plane around coarse experiment
stages. Keep TorchTitan checkpoints, generated datasets, rollouts, evaluation
outputs, manifests, and the repo-local run-attempt bundle as the canonical
scientific and recovery record. A Temporal Workflow should contain only
deterministic orchestration over small immutable references; Activities should
launch existing rootfs-managed commands for generation, SFT, export,
evaluation, and Monarch RL.

This boundary is required by both systems. Temporal replays Workflow code and
requires it to emit the same command sequence; filesystem access, subprocesses,
CUDA inspection, model execution, and other external interactions therefore
belong in Activities. Temporal's Python sandbox helps catch nondeterminism but
is explicitly not complete isolation. [Temporal Workflow determinism](https://docs.temporal.io/workflow-definition#deterministic-constraints),
[Python SDK sandbox](https://docs.temporal.io/develop/python/best-practices/python-sdk-sandbox)

The proposed ownership is:

```text
host Temporal server and worker
  -> deterministic campaign / experiment Workflow
     -> coarse, idempotent Activity
        -> scripts/rootfs/enter_rootfs.sh
           -> generation | run_train.sh SFT | export | evaluation
           -> python -m torchtitan.experiments.rl.train
        -> immutable local artifacts and stage receipt
```

Temporal must remain an optional experiment dependency. It must not enter the
PyTorch-native core Trainer or turn online RL into a core Trainer mode. This
matches the repository's explicit separation of core training, online RL, and
repo-local research programs in
[`torchtitan/experiments/README.md`](../../torchtitan/experiments/README.md) and
the approved repo-local evidence contract in the
[training research vehicle design](../../docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md#run-attempt-evidence-model).

## Local server and process boundary

For a single-machine research installation, run a version-pinned Temporal CLI
development server on the host with an explicit persistent database file, for
example:

```bash
temporal server start-dev \
  --ip 127.0.0.1 \
  --db-filename /absolute/path/to/temporal.sqlite
```

Without `--db-filename`, Workflow Executions are lost when the server process
dies. The CLI also warns that `start-dev` is not for production and skips some
HTTP security checks, so it should be loopback-only and treated as local
research infrastructure, not a shared or exposed service. [Temporal CLI
`server start-dev`](https://docs.temporal.io/cli/command-reference/server#start-dev)

Create the Temporal state directory outside the checkout, rootfs, result trees,
and shared temporary paths. Use an owner-only directory (`0700`), a restrictive
umask, and `0600` for the database, WAL, backup, and manifest files. Resolve the
directory canonically and refuse symlinks, unexpected file types, or ownership
by another user before server startup, backup, restore, or cleanup. Record the
Temporal CLI/server binary version and digest, Python SDK and environment-lock
digest, SQLite library/tool version used for backup and integrity checks,
Worker revision, namespace and retention settings, and state-schema version.
Loopback binding is the only access control promised by this development
topology; shared or remote access requires a production service design.

Run the server and Worker outside bwrap. The existing rootfs launcher `exec`s
bwrap, uses `--die-with-parent`, mounts the rootfs and checkout read-write,
shares the host network, injects NVIDIA devices/libraries, and forwards
`CUDA_VISIBLE_DEVICES`. Docker socket access is an explicit opt-in for external
harnesses. [Rootfs entry source](../../scripts/rootfs/enter_rootfs.sh#L62-L135)
An Activity should acquire a local GPU/resource lease, set the visible devices,
and then invoke this launcher; code inside the rootfs does not need the Temporal
SDK or Temporal server credentials. Acquisition or export stages that require
provider credentials use the opaque, invocation-scoped capability contract
below; no credential becomes part of Workflow state.

Rootfs construction must be a separate serialized preflight, not an implicit
side effect of a retried GPU Activity. Today the entry script automatically
builds a missing rootfs, while the build uses mutable image/package references
such as a tagged base image and `uv:latest`. More urgently, the builder accepts
an arbitrary `--dest`, later recursively removes it, and the entry script can
forward a missing arbitrary `--rootfs` into that path.
[Entry-time build](../../scripts/rootfs/enter_rootfs.sh#L64-L68),
[rootfs build inputs](../../scripts/rootfs/build_rootfs.sh#L10-L13),
[mutable build steps](../../scripts/rootfs/build_rootfs.sh#L75-L111),
[destructive export path](../../scripts/rootfs/build_rootfs.sh#L113-L125)

Before any automatic build remains enabled, both commands must use one
fail-closed resolver: canonicalize the existing parent and basename; accept
only exact managed destinations below a dedicated allowlisted store; use
`lstat`-equivalent checks to reject symlinked components, mount points,
unexpected file types, and unrelated existing directories; and explicitly
refuse empty paths, `/`, home, repository/workspace/store roots, and every
out-of-store path. Replacement requires a valid ownership marker and matching
manifest. A missing custom rootfs must fail rather than trigger a build.

Export into a uniquely owned staging directory under that store, validate the
manifest and content digest, expected structure, executable `bin/bash`, and a
minimal bwrap launch, then publish an immutable content-addressed rootfs.
Activation atomically replaces a small fsynced selection record; it never
deletes the active rootfs. Retain the prior selection until a post-activation
launch passes, and restore it on failure. Each launch holds a lease on its
selected digest. Garbage collection may remove only an enumerated, owned,
unselected, unleased digest after process and mount checks show no live user;
it is never part of Activity retry.

Before durable execution is enabled, record the selected rootfs/build digest
and validate it against the stage declaration. Temporal history is not evidence
that two Activities ran in the same environment.

Package, repository, and image acquisition must likewise be a separate
`network_acquisition` Activity. It consumes hashed locks and exact commits,
downloads into quarantine, verifies every input, builds and smokes a
content-addressed virtual-environment/source/image bundle, and atomically
publishes it. Scientific generation, training, evaluation, and harness
Activities mount that bundle and the base rootfs read-only under an offline or
validated local-network profile. They must not install or upgrade packages,
clone/fetch/checkout Git repositories, pull images, or mutate shared runtime
state.

This is a required migration, not a future optimization:
`run_bigcodebench_hard_public_vllm_smoke.sh` installs packages into the shared
rootfs; `run_external_harness_preflight_smoke.sh` creates and updates runtime
virtual environments; `run_terminal_bench_oracle_probe.sh` installs packages
and clones/fetches/checks out Terminal-Bench; the tau2 execution and mock-score
probes do the equivalent for tau2; and the ARC-AGI-2 public smoke clones its
repository during the run. These scripts remain plumbing probes until their
networked steps become acquisition receipts and their measured consumers pass
with sealed inputs, denied outbound acquisition, and an unchanged environment
digest.

Use capability-specific Activity Task Queues named `cpu`, `gpu1`, `gpu8`,
`coding-sandbox`, `harbor`, and `tau2` for routing to capable Workers. Task
Queues persist Activity Tasks and Workers poll only when they have capacity,
but a queue is not by itself an exclusive GPU allocator. A filesystem-backed
lease keyed by GPU UUID or another local admission mechanism is still required
before subprocess launch. [Temporal Task Queues](https://docs.temporal.io/task-queue#what-is-a-task-queue)

### SQLite backup and restore protocol

The development server's database is an operational recovery index, not the
scientific record, but its backups still need transactionally consistent
handling. Make a backup only after either stopping the server cleanly and
confirming the database is quiescent or invoking SQLite's online backup API
against the live database. Copying the main file while a WAL may contain
committed state is not a backup protocol. Write into a new owner-only staging
directory, run `PRAGMA integrity_check`, record the source-state identity and
all runtime/schema versions plus the backup digest, then atomically publish the
backup manifest. [SQLite online backup API](https://www.sqlite.org/backup.html),
[SQLite integrity check](https://www.sqlite.org/pragma.html#pragma_integrity_check)

Restore never overwrites the current state in place. Materialize the backup in
a new isolated `0700` directory, verify ownership, `0600` files, digest,
version/schema compatibility, and database integrity, then start the pinned
server on an isolated port and namespace. Replay representative and restored
open histories against the intended Worker code. Reconcile every Workflow
reference with the canonical repo-local run, attempt, stage, checkpoint, and
artifact receipts and hashes. Missing, extra, mutable, or contradictory local
truth fails the restore. Only then atomically select the restored state;
preserve the previous state for rollback. Backup retention and deletion use a
separate explicit policy over enumerated, validated backup directories.

Because Event History and SQLite backups contain Activity inputs, results,
heartbeat details, failure messages, Memos, and Search Attributes, no secret
value, hash, credential pathname, credential-bearing command, or raw provider
error may enter any of them. Workflows receive only an opaque declared
capability ID. The host Activity resolves credentials to an invocation-scoped
file or descriptor immediately before subprocess launch, scrubs exception and
subprocess text before raising a classified Temporal error, and keeps any
restricted raw diagnostic only in the canonical local incident bundle.

## Workflow and Activity granularity

Use two Workflow types:

- `ExperimentRunWorkflow` owns one frozen scientific declaration and advances
  its stages.
- `ResearchCampaignWorkflow` starts experiment child Workflows or waits at
  explicit promotion gates.

Use a deterministic, project-and-namespace-scoped Workflow ID derived from the
workflow kind, `run_id` or campaign ID, and declaration digest. The client start
policy rejects an already-running ID and returns an existing compatible
execution only after its declaration reference validates; a terminal Workflow
ID is not silently reused for a new operational attempt. A deliberate new
attempt is represented inside the compatible run Workflow or started under an
explicit generation suffix according to the frozen ID-reuse policy. Client
start retries therefore cannot create parallel run Workflows. Workflow IDs
remain operational correlation values, never scientific identity.

Workflow state contains identifiers, declaration digests, stage states, and
artifact receipts only. Do not import TorchTitan, inspect the filesystem, read
environment variables, call subprocesses, enumerate CUDA, or perform hashing
inside Workflow code. Those actions are replay-unsafe external interactions.
The default sandbox should remain enabled; pass through only deterministic,
side-effect-free schema modules. [Workflow determinism](https://docs.temporal.io/workflow-definition#deterministic-constraints),
[sandbox passthrough constraints](https://docs.temporal.io/develop/python/best-practices/python-sdk-sandbox#passthrough-modules)

An Activity should produce one coarse, recoverable artifact boundary, not one
prompt, rollout, batch, optimizer step, or test case. Suitable Activities are:

- split/preflight validation;
- one resumable generation or evaluation shard;
- verified dataset materialization;
- one complete `torchrun` SFT job;
- one adapter/full-policy export and validation;
- one evaluation condition or official harness job;
- one complete deterministic or asynchronous Monarch RL job; and
- report-input validation/finalization.

Temporal records Activity arguments and results in Workflow Event History, and
each Activity adds scheduling, start, and terminal events. The official guidance
therefore recommends balancing atomic retry boundaries against history growth.
[Python Activity execution](https://docs.temporal.io/develop/python/activities/execution#start-an-activity-execution),
[Activity atomicity guidance](https://docs.temporal.io/develop/python/best-practices/error-handling#design-activities-to-be-atomic)
The current modular transfer runner already exposes the natural boundaries:
generation, dataset construction, base evaluation, `torchrun` SFT, LoRA export,
adapter evaluation, and report input.
[Current staged runner](run_modular_sequences_transfer_smoke.sh#L80-L199)

Core SFT must continue through `run_train.sh -> torchrun -> torchtitan.train`.
[Core launcher](../../run_train.sh#L36-L46) Online RL must remain one direct
`python -m torchtitan.experiments.rl.train` process whose controller provisions
the trainer and generator meshes; it must not be wrapped in `torchrun`.
[RL entrypoint](../../torchtitan/experiments/rl/train.py#L281-L325)

## At-least-once delivery and local idempotence

Temporal Activities have at-least-once execution semantics. A Worker can finish
an Activity and crash before acknowledging it, after which Temporal may run the
Activity again. The official recommendation is to make Activities idempotent
and use a key stable across retries. [Temporal Activity idempotence](https://docs.temporal.io/develop/python/best-practices/error-handling#make-activities-idempotent)

For this repository, use a stable logical stage key such as:

```text
run_id / attempt_id / stage_id / shard_id / declaration_digest
```

and record Temporal's Workflow ID, Workflow Run ID, Activity ID, and Activity
attempt only as correlation fields. A repo-local `attempt_id` spans one
top-level execution of the run's stage sequence. A genuine stage relaunch while
that attempt remains active gets a new `stage_invocation_id`; it does not create
a new run attempt. An operator/campaign retry after a terminal attempt creates
a new `attempt_id` and therefore a new stage key. Activity delivery retries
within one attempt share the key. Cross-attempt reuse is an explicit,
provenance-bearing consume operation, never same-key reconciliation. A
redelivery that validates and returns an already-complete receipt creates no
new invocation and records `delivery_disposition=reconciled_no_launch`. It does
not rewrite the original artifact's `work_status`; bytes produced by the
original invocation remain `produced`. Reserve `work_status=reused` for a new
stage or attempt that explicitly consumes verified preexisting bytes.
Temporal IDs must not replace the scientific `run_id`, operational
`attempt_id`, checkpoint lineage, or artifact identity.

Before every launch, the Activity must reconcile the stage key against local
truth:

1. If a same-key terminal receipt exists, verify the attempt, declaration
   digest, output hashes, provenance, and task-specific validation, then return
   that immutable receipt without launching and set only the orthogonal
   delivery disposition.
2. If a live execution record exists, observe it only through a durable local
   supervisor or official external job API whose stable job identity, ownership,
   status, and terminal receipt can be verified. A replacement Worker cannot
   generally wait on, reap, or recover the exit status of an arbitrary
   non-child PID. Without such a supervisor, retain the lease, mark the state
   ambiguous/blocked, and require reconciliation; never start a duplicate GPU
   job.
3. If a validated resumable checkpoint or committed shard ledger exists, make
   the explicit resume decision at the appropriate scope: a new stage
   invocation inside an active attempt, or a linked new attempt after the prior
   attempt is terminal.
4. If partial output or process state is ambiguous, emit a blocked/failed
   attempt and require review. Do not guess that a retry is safe.
5. Otherwise reserve the attempt if this is its first stage, allocate a new
   stage invocation, append `stage_started`, and launch.

Outputs should be written under invocation-specific staging paths and promoted
atomically only after validation. The terminal stage receipt should include the
declaration digest, exact command/profile, input and output hashes, checkpoint
parent, canonical work status (`produced`, `reused`, `resumed`, `imported`, or
`external`), freshness (`verified_new`, `verified_preexisting`, or `unknown`),
and terminal outcome. Temporal Activity completion is then an index to that
receipt, not the receipt itself.

This is also why the current manifest setup cannot be reused unchanged:
`scaffold_setup_run_manifest` truncates the run JSONL, although stage rows are
subsequently appended. A Temporal redelivery could therefore erase earlier
evidence unless run/attempt creation becomes immutable and append-only first.
[Current truncation and append behavior](run_common.sh#L27-L34),
[current stage append](run_common.sh#L98-L101)

## Long-running Activities, timeouts, heartbeats, and cancellation

Every Activity must set Start-To-Close or Schedule-To-Close. Start-To-Close
limits one attempt; Schedule-To-Close limits the total logical Activity across
retries. Schedule-To-Start measures queue wait and is non-retryable; for expected
GPU backlogs, monitor queue latency instead of using a short value that turns
normal admission delay into failure. [Temporal Activity timeout semantics](https://docs.temporal.io/encyclopedia/detecting-activity-failures)

Generation, SFT, evaluation, and RL Activities should set:

- Start-To-Close longer than the largest legitimate single launch;
- a stage-specific Schedule-To-Close covering that stage's queue and bounded
  execution/retry allowance, not the whole campaign budget;
- a short Heartbeat Timeout relative to acceptable lost time; and
- an explicit bounded Retry Policy rather than Temporal's default unlimited
  Activity attempts.

The campaign separately enforces a deadline and persistent compute ledger, so
several Activities cannot each consume the campaign-wide allowance.

Long Activities should heartbeat semantic progress: stage and phase, subprocess
identity, committed shard count, optimizer/policy step, last validated
checkpoint, and last-progress time. Heartbeat details can be supplied to the
next retry, but they are throttled, may not reach the server before a Worker
crash, and are not visible to the running Workflow as a progress stream.
Therefore they are hints; the committed local ledger/checkpoint remains truth.
[Activity heartbeats, throttling, and progress recovery](https://docs.temporal.io/encyclopedia/detecting-activity-failures#activity-heartbeat)

Cancellation is cooperative. A non-local Activity receives cancellation only
through heartbeats. Start every long subprocess Activity with
`ActivityCancellationType.WAIT_CANCELLATION_COMPLETED`, a heartbeat timeout,
and a bounded cleanup allowance; Workflow code awaits the Activity handle.
Cleanup completion means a verified local terminal receipt, not merely delivery
of the cancellation request.

The host Activity adapter must append cancellation intent, invoke the
adapter-specific cleanup handle, request graceful termination, and wait through
the checkpoint/evidence grace interval. If work remains, it verifies ownership
before using its cgroup/process-group kill path and any Docker or official-job
cleanup API, then waits/reaps and proves there are no owned descendants,
containers, GPU contexts, or leases. Only then may it publish the interrupted
outcome, release the lease, and re-raise cancellation. Docker/Harbor resources
are daemon-owned and cannot be inferred gone from subprocess exit alone.
Temporal does not perform this cleanup. Avoid Workflow termination except for a
stuck Workflow because termination gives Workflow code no cleanup opportunity.
[Python cancellation semantics](https://docs.temporal.io/develop/python/workflows/cancellation)

The existing bwrap `--die-with-parent` setting is useful defense in depth, and
the RL entrypoint performs best-effort close on `KeyboardInterrupt` or
`asyncio.CancelledError`, but neither proves that every descendant and GPU
resource is gone. [bwrap process policy](../../scripts/rootfs/enter_rootfs.sh#L73-L81),
[RL graceful close](../../torchtitan/experiments/rl/train.py#L301-L322)
Cancellation and Worker-death tests must inspect the complete process tree and
GPU lease before allowing a retry. A replacement worker may signal an orphan
only when the durable supervisor or an equally strong ownership record proves
the exact process group/cgroup belongs to that stage; a PID match alone is not
sufficient.

## Retry policy by stage

Activities retry by default with exponential backoff and unlimited attempts;
Workflow Executions do not retry by default. Permanent failures can be raised
as `ApplicationError(non_retryable=True)` or named in
`non_retryable_error_types`. [Temporal Retry Policy defaults](https://docs.temporal.io/encyclopedia/retry-policies#default-behavior),
[Python non-retryable failures](https://docs.temporal.io/develop/python/best-practices/error-handling#mark-specific-errors-as-non-retryable)

That Workflow-Execution rule does not make arbitrary Python exceptions in
Workflow code terminal: ordinary Workflow Task failures are retried and can
leave an execution stuck. Validate declarations before starting, convert
expected permanent domain failures to typed `ApplicationError` outcomes, keep
unexpected Workflow-code exceptions visible, and alert/test when a Workflow
Task repeatedly fails without advancing history.

Use the following repository policy:

| Failure/stage | Temporal policy | Required local decision |
| --- | --- | --- |
| Invalid declaration, split leakage, verifier implementation/crash, canonical-control or semantic-preflight failure, incompatible checkpoint, deterministic config error | Non-retryable | Preserve failure evidence; change inputs in a new scientific/operational attempt. A candidate rejected by a working verifier is real score evidence, not an Activity error. |
| Asset/network acquisition | Small bounded retry with backoff | Verify content digest before publication. |
| Generation/evaluation shard | Bounded retry | Reuse committed rows; resume only missing identities; reject duplicate/conflicting rows. |
| SFT | Initially one automatic attempt | A later launch may resume only from a validated complete DCP checkpoint with matching config, data position, and full train-state lineage. |
| Export/report construction | Bounded retry | Recompute into staging and atomically publish only if parent/output hashes agree. |
| Official external harness | Bounded polling retry, not blind job resubmission | Reattach through the harness job/trial ID; ingest its official result artifact. |
| Deterministic or async RL | No automatic relaunch initially | Require quiescent cleanup and explicit review of checkpoint and prompt-group state. |

The RL restriction is material, not conservative wording. The controller saves
trainer state, but its own source says resume currently restores only model,
optimizer, and policy version; active-slot rollouts and dataset stream position
are not restored. [Current RL resume limitation](../../torchtitan/experiments/rl/controller.py#L655-L664)
The generator is deliberately not an independent checkpoint owner, because it
pulls trainer weights through TorchStore.
[RL checkpoint ownership](../../torchtitan/experiments/rl/controller.py#L340-L350)
Automatic Temporal retry of RL would therefore risk replaying prompts or
changing asynchronous policy-age history. Enable it only after a controller
checkpoint proves buffer, in-flight work, dataset, trainer, generator-sync, and
policy-version recovery at a declared quiescent boundary.

Temporal does not supersede TorchTitan checkpoint semantics. For SFT, the
resume receipt must point to a checkpoint that contains the required model,
optimizer, scheduler, RNG, dataloader, and train-step state and has passed the
repository's restore-equivalence gate. A Heartbeat claiming `step=N` is not a
checkpoint commit.

## Payload and Event History limits

Do not send datasets, rollouts, logs, checkpoints, model weights, or complete
evaluation results through Temporal. Python Activity parameters and results are
recorded in Event History. Heartbeats, failure details, Memos, and Search
Attributes are history or service data too, so the same no-secret and compact
reference policy applies to all of them. Sanitize an Activity's exception and
subprocess summary before raising it; never assume that an exception is a local
log line. Documented limits are 2 MB for a single payload and 4 MB for a gRPC
message/History transaction. Large histories also slow Worker recovery.
[Python Activity payload limits](https://docs.temporal.io/develop/python/activities/basics#develop-activity-parameters)

The documented self-hosted defaults warn at 10 MB or 10,240 History events and
error at 50 MB or 51,200 events; payloads warn at 256 KB and error at 2 MB.
These server values are configurable, but the design should stay comfortably
below the defaults rather than depending on a local override.
[Self-hosted defaults](https://docs.temporal.io/self-hosted-guide/defaults)

Pass only compact immutable references and summaries, for example:

```json
{
  "run_id": "...",
  "stage_id": "sft_raw",
  "declaration_digest": "sha256:...",
  "receipt_path": ".../outcome.json",
  "receipt_digest": "sha256:...",
  "checkpoint_id": ".../step-1200",
  "status": "completed"
}
```

The local filesystem is effectively the large-object store for this
single-machine design. References must be immutable and content-verified;
passing a mutable pathname merely moves the race outside Temporal. Temporal's
own data-handling design likewise treats large-payload external storage as an
optional offload layer while keeping references in History.
[Python data handling](https://docs.temporal.io/develop/python/data-handling),
[Temporal external storage](https://docs.temporal.io/external-storage)

Use Continue-As-New at campaign phase boundaries or when
`workflow.info().is_continue_as_new_suggested()` is true. Continue-As-New keeps
the Workflow ID, creates a new Temporal Run ID, and starts a fresh Event
History. Pass only compact stage/artifact references as the new input, and wait
for Signal/Update handlers to finish first.
[Python Continue-As-New](https://docs.temporal.io/develop/python/workflows/continue-as-new)
The project's `run_id` and `attempt_id` remain unchanged unless the scientific
or operational execution actually changes; a Temporal Run ID change is only an
orchestration-chain event.

## Testing, replay, and deployment versioning

The minimum proof ladder is:

1. Unit-test declaration, attempt-scoped idempotency keys, immutable producer
   status, delivery disposition, and receipt reconciliation without Temporal.
2. Use `ActivityEnvironment` to test heartbeats, cancellation, non-retryable
   classification, and retry resume details.
3. Use `WorkflowEnvironment` with mock Activities to test deterministic
   Workflow-ID conflict/reuse behavior, gate ordering, bounded retries,
   `WAIT_CANCELLATION_COMPLETED`, repeated Workflow Task failure alerts, and
   Continue-As-New; use time skipping for timers/backoff.
4. Run host integration tests against a local server and fake subprocess,
   including server restart, Worker death before Activity acknowledgement,
   duplicate delivery, cancellation that waits for verified process/cgroup and
   Docker cleanup, timeout, corrupt receipt, and ambiguous live process.
5. Exercise both a quiesced and online SQLite backup, restrictive permissions,
   integrity and digest failure, an isolated restore, Workflow replay, canonical
   receipt reconciliation, activation, and rollback without overwriting the
   prior state.
6. Run rootfs CPU integration for hostile destination refusal, staged rootfs
   validation, atomic selection and rollback, per-digest launch leases, sealed
   environment consumption, signal propagation, and atomic artifact
   publication. Then run one-GPU generation/SFT checks, and only then multi-GPU
   SFT/RL fault injection.

The Python SDK provides `ActivityEnvironment`, local/time-skipping Workflow
environments, mocked Activities, and a `Replayer`. Temporal recommends replaying
a representative set of open and closed histories in CI and failing on any
nondeterminism. [Python SDK testing and replay](https://docs.temporal.io/develop/python/best-practices/testing-suite)

Workflow code changes can make old histories nondeterministic even when the new
code is otherwise correct. Before deploying orchestration changes, export
representative histories into compact test fixtures, run replay, and use Worker
Versioning or Python `workflow.patched()` for command-sequence changes. Record
the Worker code revision and Temporal SDK/CLI/server versions in each local
attempt bundle. [Python Workflow versioning](https://docs.temporal.io/develop/python/workflows/versioning)

## Adoption gates

Temporal becomes the default staged executor only after all of these pass:

- persistent local-server restart plus quiesced/online backup, integrity,
  isolated restore, replay, receipt reconciliation, activation, and rollback
  drills under the restrictive state-file policy;
- deterministic Workflow replay against captured histories;
- Activity acknowledgement-loss and duplicate-delivery proof;
- append-only local run/attempt/stage receipts with atomic artifact publication;
- safe rootfs destination refusal, content-addressed activation/rollback, live
  launch leases, and immutable offline environment proof;
- bwrap process-tree cancellation and GPU-lease release proof;
- interrupted-versus-uninterrupted SFT checkpoint equivalence;
- payload/history budget and Continue-As-New test;
- a scanner-backed proof that payloads, heartbeat details, exceptions,
  histories, worker logs, and backups contain no secret material; and
- explicit evidence that W&B/TensorBoard/Temporal remain projections or
  orchestration indexes, while the repo-local bundle and validated artifacts
  remain canonical.

Until the RL controller's joint recovery gap is closed, Temporal may durably
record, cancel, and surface an RL failure, but it must not automatically relaunch
that RL stage.
