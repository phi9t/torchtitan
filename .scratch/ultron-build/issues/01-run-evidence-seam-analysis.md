# TorchTitan run evidence seam analysis

Type: research
Status: resolved
Blocked by: -

## Requirement

Prove the first TorchTitan integration seam for Ultron run evidence and update
commit records. This is seam analysis only unless the human separately
authorizes TorchTitan code edits.

Answer:

- where run identity and topology epoch should originate;
- where trainer step, microbatch, process-group generation, and active replica
  set are available or can be attached;
- where PREPARED/COMMITTED-style records could be emitted;
- how checkpoint generation and run evidence artifacts would link to the ledger;
- which fields cannot be obtained in TorchTitan and must be supplied by an
  external controller, NCCL adapter, or lower-layer adapter.

## Exclusions

Do not edit TorchTitan code, commit, push, create a pull request, run large GPU
jobs, run live/destructive fault injection, or change optimizer, checkpoint, or
distributed process-group semantics unless the human explicitly authorizes that
repo/action.

## Verification Evidence

- Cite current TorchTitan file:line evidence for every proposed seam.
- Classify each hot-path claim as `static-seam-only`, `fixture-proven`, or
  `performance-proven`.
- `performance-proven` requires a measured command and result; otherwise state
  that hot-path overhead is not performance-proven.
- State whether each proposed emission point performs synchronous IO, dynamic
  allocation, blocking communication, or cross-process coordination on the
  training hot path.
- State whether the next TorchTitan ticket should be schema plumbing, ledger
  emission, or checkpoint/run-evidence linking.

## Answer

### Verdict

TorchTitan has a plausible Phase 1 Ultron seam, but this ticket proves it only
as `static-seam-only`. No performance measurement was run, so hot-path overhead
is not performance-proven.

The lowest-risk next TorchTitan ticket is **schema plumbing**: map existing
TorchTitan run evidence and trainer/checkpoint fields into the Ultron identity,
ledger, and evidence schemas behind a dry-run or fixture-only path. Do not start
with optimizer semantics changes.

### Run identity and topology epoch

Run identity should originate from `RunEvidence`, not a new parallel identity
system:

- `RunEvidence.__enter__` resolves `run_id` and `attempt_id`, publishes a
  manifest, and opens a per-process artifact index
  (`torchtitan/observability/run_evidence.py:114`).
- `_resolve_identity` uses `TORCHTITAN_RUN_ID`, `TORCHTITAN_ATTEMPT_ID`,
  `TORCHELASTIC_RUN_ID`, and `TORCHELASTIC_RESTART_COUNT`; multi-process runs
  require explicit run and attempt ids
  (`torchtitan/observability/run_evidence.py:208`).
- The manifest already records schema version, run id, attempt id, normalized
  config hash, source state, command, and runtime versions
  (`torchtitan/observability/run_evidence.py:264`).
- `Trainer.__init__` initializes distributed state and then binds distributed
  evidence context immediately after building `ParallelDims`
  (`torchtitan/trainer.py:260`).

Topology epoch is not present as an explicit TorchTitan field today. For Phase 1,
use the run attempt plus static distributed context as `topology_epoch` for
non-elastic fixtures. A real topology epoch must come from an external
controller or a future TorchTitan membership/process-group generation hook.

### Trainer step, microbatch, process group generation, and active replica set

Available in TorchTitan today:

- The trainer owns `self.step` and `self.ntokens_seen`, initialized before
  checkpoint loading (`torchtitan/trainer.py:518`).
- The main loop increments `self.step`, sets structured logger step context, and
  wraps one step in a trace span (`torchtitan/trainer.py:960`).
- `train_step` owns the optimizer-step work, including gradient accumulation,
  forward/backward, optimizer step, scheduler step, metrics reductions, and
  invalid-loss failure (`torchtitan/trainer.py:808`).
- `RunEvidence.bind_distributed` captures parallel dimensions, world size,
  device type/index/UUID, and mesh-axis rank/size when mesh introspection exists
  (`torchtitan/observability/run_evidence.py:164`).
- `RunEvidence._open_process_index` captures process id, role, actor id, host,
  pid, global rank, local rank, and world size
  (`torchtitan/observability/run_evidence.py:307`).

Not available as first-class TorchTitan fields:

- Active replica set.
- Process-group generation.
- Explicit topology epoch.
- NCCL collective sequence or communicator state.

Those should be supplied by an external controller and NCCL/lower-layer adapter,
then joined to TorchTitan run evidence through run/attempt/rank/step identity.

### PREPARED and COMMITTED seams

Static candidate seams:

- `PREPARED(step, replica_epoch)` can be emitted after all gradient accumulation
  forward/backward work has completed and before optimizer mutation, around the
  start of the `optim` span. The relevant block begins at
  `torchtitan/trainer.py:868`.
- A safer first dry-run approximation is to classify `PREPARED` immediately
  before `self.optimizers.step()` after `maybe_wait_for_staging()` returns
  (`torchtitan/trainer.py:876`).
- `COMMITTED(step, replica_epoch)` can be emitted after `self.optimizers.step()`
  and `self.lr_schedulers.step()` complete (`torchtitan/trainer.py:877`).
- The metric reduction/logging block after optimizer step is not the commit
  boundary; it is evidence and health reporting (`torchtitan/trainer.py:880`).

Hot-path classification:

- These are `static-seam-only` claims. Emitting records here could add
  synchronous IO or dynamic allocation if implemented naively. Any future code
  must use an existing buffered/dry-run run-evidence path or prove overhead with
  measurement. Hot-path overhead is not performance-proven.

### Checkpoint generation and ledger link

TorchTitan checkpointing already has useful artifact lifecycle evidence:

- `Trainer.train` loads checkpoint state before training and captures the loaded
  step for relative-step accounting (`torchtitan/trainer.py:947`).
- `Trainer.train` calls `self.checkpointer.save(self.step, last_step=...)` after
  `train_step` returns (`torchtitan/trainer.py:973`).
- `CheckpointManager.save` creates a checkpoint id and metadata before writing
  or staging checkpoint state (`torchtitan/components/checkpoint.py:747`).
- It declares a checkpoint artifact with relation, step, and metadata before the
  write/stage path (`torchtitan/components/checkpoint.py:755`).
- Successful synchronous saves finish the artifact as `COMPLETE`; async saves
  retain pending artifact state for later completion
  (`torchtitan/components/checkpoint.py:841`).
- `_declare_checkpoint_artifact` and `_finish_checkpoint_artifact` both call
  `record_artifact` with producer `checkpoint`, kind
  `torchtitan.checkpoint`, path, state, relation, step, and metadata
  (`torchtitan/components/checkpoint.py:1128`,
  `torchtitan/components/checkpoint.py:1146`).
- `Trainer.state_dict` includes `step` and `ntokens_seen`; loading restores both
  (`torchtitan/trainer.py:1007`).

The Ultron ledger should link `DURABLY_CHECKPOINTED` to the checkpoint artifact
id and checkpoint id/generation from this lifecycle, not infer durable commit
from tensor writes alone.

### Fields TorchTitan cannot supply alone

- NCCL collective sequence, communicator id, async error, RAS, and progress
  lifecycle: supplied by the NCCL adapter.
- Physical fabric identity beyond host/device/GPU UUID: supplied by controller,
  scheduler, DCGM, NCCL, or fleet inventory.
- Active replica set and replica epoch under elastic membership: supplied by an
  external controller or future TorchTitan/TorchFT membership layer.
- Process-group generation: supplied by distributed membership/process-group
  management, not current core trainer fields.
- Topology epoch under dynamic or elastic topology: supplied by controller.

### Next ticket

Next TorchTitan work should be **schema plumbing** in a separately authorized
implementation ticket: produce fixture or dry-run records from existing
`RunEvidence`, trainer step, and checkpoint artifact data, with no optimizer,
checkpoint, or process-group semantic changes.
