# Robust Training Reliability

Robust training is the discipline of preserving useful progress under failure.
A training run is robust when it either continues from a verified-equivalent
state or fails with enough runtime control evidence to attribute and contain
the fault.

For large distributed training, the non-negotiable run failure modes are lost
progress, degraded or unstable progress, and wrong progress. Networking
failures dominate lost progress. Stragglers and bad machines dominate degraded
or unstable progress: stragglers are the observed symptom, while bad machines
are one important cause. Silent correctness corruption dominates wrong progress.

TorchTitan should own the training-semantics evidence, degraded-progress
evidence, and recovery-validation hooks for these modes. The surrounding
training platform should own fleet action such as retry policy, replacement,
host quarantine, and health scoring.

## Priorities

### Networking failures

Networking failures turn distributed training into a liveness problem. One rank
that stops making collective progress can stall the whole step, and the first
rank to report the timeout is not necessarily the rank that caused it.

The reliability goal is not merely to time out. It is to produce enough context
to decide whether to wait, abort, retry, or escalate to fleet-level diagnosis.
TorchTitan-owned evidence should include process-group context, mesh context,
rank and host identity, step, training phase, and the last known distributed
operation when available.

Networking failures should be split into two layers:

- Data-plane liveness: collectives, process groups, NCCL progress, rank stalls.
- Control-plane liveness: rendezvous, scheduler state, rank membership, artifact
  coordination.

TorchTitan should primarily own data-plane evidence and expose enough context
for the platform to correlate control-plane state.

### Stragglers

Stragglers turn distributed training into a degraded-progress problem. A
straggler can be a trainer rank, data loader, device, host, or pipeline stage
that repeatedly delays step progress relative to peers without fully stopping
progress.

Stragglers should be treated as an observed symptom, not a root cause. A slow
trainer rank may point to device, collective, checkpoint, compile, or host
issues. A slow data loader may point to shard skew, decode or tokenization cost,
object-store latency, local disk behavior, worker imbalance, or host-to-device
handoff. Treating every straggler as a bad machine risks quarantining healthy
hosts for data or workload imbalance.

Straggler evidence should identify the observed locus:

- Trainer rank.
- Data loader.
- Pipeline stage.
- Checkpoint path.
- Unknown.

It should also include phase attribution:

- Data fetch, decode, tokenize, or batch construction.
- Host-to-device transfer.
- Forward.
- Backward.
- Optimizer.
- Collective wait.
- Checkpoint save or load.
- Evaluation.
- Compile or cache warmup.
- Unknown.

A slow component becomes a reliability concern when the skew is repeated,
material, and phase-attributed. One-off variance is performance noise; persistent
peer-relative skew threatens useful progress, cost predictability, and recovery
decisions.

TorchTitan should own straggler evidence and phase attribution. Remediation
depends on the attributed cause: the platform may retry or quarantine a host,
the data layer may rebalance shards or workers, and model or parallelism code
may need to address workload imbalance.

### Bad machines

Bad machines are hosts whose local hardware or host-local environment causes
abnormal failure, slowdown, or corruption risk. At scale, rare host issues
become normal operating conditions.

The thesis should distinguish hard failures from soft failures. Hard failures
crash, timeout, or produce explicit health events. Soft failures create
stragglers, intermittent hangs, retry loops, ECC storms, link flaps, local
filesystem failures, or unexplained throughput cliffs.

A machine should not be declared bad from a single vague slowdown or from an
unattributed straggler. It requires either hard health evidence or repeatable
correlated degradation. TorchTitan should emit machine-attribution evidence
when available. The platform should retry on a fresh allocation and quarantine
machines on repeated or high-confidence faults.

### Silent correctness corruption

Silent correctness corruption is the most dangerous class because training can
continue while becoming invalid. It includes invalid model state, optimizer
state, data stream, numerical values, checkpoint lineage, or unintended math
changes. It does not include every ordinary convergence regression; the key
property is unintended invalid state or unintended math, data, or lineage drift.

Correctness checks must be first-class runtime and validation machinery, not
only postmortem debugging. The minimum sentinel set is:

- Checkpoint manifest integrity.
- State lineage metadata.
- NaN/Inf, loss, and grad_norm sentinels.
- Restore equivalence tests.

Dataset shard checksums are important, but the owning layer depends on how the
data pipeline is deployed. TorchTitan should preserve enough sample or token
position metadata to detect skipped or duplicated progress after restart when
the data loader supports it.

## Progress Evidence Envelope

Every diagnosable training failure or degraded-progress event should carry a
progress evidence envelope. The envelope is the minimum context another system
or human needs to attribute, contain, or continue monitoring the event.

TorchTitan-owned fields:

- Run id and attempt id when available.
- Global step.
- Observed locus: trainer rank, data loader, pipeline stage, checkpoint path,
  or unknown.
- Training phase: data fetch, decode, tokenize, batch construction,
  host-to-device transfer, forward, backward, optimizer, collective wait,
  checkpoint, eval, compile or cache warmup, or unknown.
- Rank, local rank, and device id.
- Mesh axes and process-group context.
- Last known collective or checkpoint operation when available.
- Peer-relative timing and persistence when reporting a straggler.
- Normalized training config or config digest.
- Checkpoint id and parent checkpoint id when applicable.
- Model, tokenizer, and data identifiers when available.
- Data shard, worker, sample, or token position when available.

Platform-joined fields:

- Host id and allocation id.
- Scheduler job id.
- Host health events.
- NIC, GPU, filesystem, and kernel or driver health signals.
- Retry and quarantine history.

First-fault attribution should be best effort and should include confidence.
Useful classifications include:

- Observed local fault.
- Suspected peer fault.
- Collective timeout with insufficient evidence.
- Platform-confirmed host fault.

The goal is to avoid turning distributed ambiguity into false certainty.

## Recovery Standard

Detection is not the robustness boundary. Recovery is. After a failure, the
system must be able to answer whether the run resumed the intended training
trajectory, skipped or duplicated data, lost optimizer state, changed RNG state,
corrupted metrics, or silently forked the experiment.

Production recovery should provide semantic continuity: the resumed run has not
lost, duplicated, or silently changed training state, even when bitwise identity
is not promised.

Validation should include deterministic restore proof. In a small deterministic
configuration, an interrupted-and-restored run should match the uninterrupted
run for loss and grad_norm. Production-scale jobs should not promise bitwise
identity unless deterministic mode and full-stack constraints are active.

Lineage attached to training state should include:

- Source version.
- Normalized config.
- Model, tokenizer, and data identifiers.
- Checkpoint id and parent checkpoint id.
- Global step.
- Token or sample position when available.
- World size and mesh shape.
- Library versions and relevant precision, compile, and distributed settings.
- Whether model, optimizer, scheduler, RNG, and dataloader state are present.

## Proof Standard

Robustness features should not be considered done merely because a code path
exists. They need proof that the intended failure semantics hold.

Required proof classes:

- Unit proof for state capture, metadata, and validation helpers.
- Deterministic restore proof for checkpoint and recovery behavior.
- Injected failure proof for failure handling behavior.
- Scale proof for distributed communication changes.
- Operational proof only when claiming platform integration.

The initial failure-injection matrix should cover:

- Kill one rank during a training step.
- Hang one rank before a collective.
- Corrupt a checkpoint shard or manifest.
- Restore after interruption and compare deterministic continuation.
- Introduce rank-local config mismatch.
- Inject a trainer-rank straggler and verify phase-attributed evidence.
- Inject a data-loader straggler and verify data-side locus attribution.

Dataset shard corruption should be added when the data pipeline contract is in
scope for the feature being validated.

## Non-Goals

TorchTitan should not implement a fleet scheduler.

TorchTitan should not quarantine machines directly.

TorchTitan should not promise survival under arbitrary corruption.

TorchTitan should not hide failures with speculative trainer-level retries.
Retries without attribution can convert a diagnosable failure into silent state
ambiguity.

TorchTitan should not treat every straggler as a bad machine. Trainer-rank and
data-loader stragglers require phase attribution before remediation.

TorchTitan should not claim bitwise production reproducibility unless
deterministic mode and the required full-stack constraints are active.

## Roadmap

### 1. Progress envelope and lineage metadata

Define the training-semantics fields TorchTitan emits on failure and checkpoint
events, and the degraded-progress fields it emits for stragglers. Done means
events include rank, host-joinable identity, observed locus, step, phase,
distributed context, peer-relative timing when applicable, and lineage fields,
with tests for envelope construction.

### 2. Checkpoint integrity and restore equivalence

Validate checkpoint manifests and record parentage, step, config, and state
presence. Done means deterministic interrupted-vs-uninterrupted runs match loss
and grad_norm in a small validation configuration.

### 3. Distributed liveness and first-fault attribution

Surface process-group, mesh, rank, and operation context for distributed stalls
and timeouts. Done means injected rank kill and rank hang tests produce a
progress evidence envelope with best-effort attribution.

### 4. Straggler attribution

Surface phase-attributed timing for trainer-rank and data-loader stragglers.
Done means injected trainer-side and data-side stragglers produce distinct
evidence, and neither path implies bad-machine remediation without supporting
host evidence.

### 5. Failure injection harness

Build repeatable tests for rank death, rank hangs, trainer-rank stragglers,
data-loader stragglers, checkpoint corruption, restore equivalence, and
rank-local config mismatch. Done means each failure mode has an expected outcome
and cannot silently pass.

### 6. Platform handoff contract

Define which TorchTitan evidence fields the surrounding platform consumes for
retry, replacement, and quarantine policy. Done means the contract is documented
and an operational integration test proves the platform can act on TorchTitan
evidence without TorchTitan owning fleet policy.
