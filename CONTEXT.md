# Robust Training

This context defines the reliability language for large distributed training
runs built with TorchTitan and the surrounding training platform.

## Language

**Training Run**:
A logical training experiment that can span one or more launch, restart, or
recovery attempts while retaining one lineage identity.
_Avoid_: Job, process, attempt

**Run Attempt**:
One launch or elastic restart of a training run, with its own immutable manifest
and process outcomes plus append-only artifact indexes.
_Avoid_: Run, retry folder

**Evidence Artifact**:
A native log, trace, snapshot, metric stream, checkpoint, or diagnostic result
whose provenance is recorded in a run-attempt artifact index.
_Avoid_: Copied output, attachment

**Robust Training Run**:
A training run that either continues from a verified-equivalent state after a
failure or fails with enough evidence to attribute and contain the fault.
_Avoid_: Reliable job, failure-proof training

**Run Failure Mode**:
A class of failure outcome that explains how a training run stops being useful:
lost progress, degraded or unstable progress, or wrong progress.
_Avoid_: Run-death mode, root cause, bug category

**Bad Machine**:
A host whose local hardware or host-local environment causes abnormal failure,
slowdown, or corruption risk during distributed training.
_Avoid_: Broken node, flaky box

**Straggler**:
A trainer rank, data loader, device, host, or pipeline stage that repeatedly
delays distributed training progress relative to peers without fully stopping
progress.
_Avoid_: Slow node, bad machine

**Silent Correctness Corruption**:
A failure where training continues and appears healthy while model state,
optimizer state, data stream, numerical values, or checkpoint lineage is invalid.
_Avoid_: Silent data corruption, bad data

**Recovery Equivalence**:
The property that an interrupted-and-restored training run preserves the
intended training trajectory, including model, optimizer, scheduler, RNG, and
dataloader state.
_Avoid_: Successful restart, checkpoint loads

**Runtime Control Evidence**:
Observable training evidence that can drive a decision to wait, abort, retry,
quarantine, restore, or declare a run invalid.
_Avoid_: Logs, observability

**Progress Evidence Envelope**:
The minimum context attached to a training failure or degraded-progress event
so another system or human can attribute, contain, or continue monitoring it.
_Avoid_: Metrics, trace

**Failure Evidence Envelope**:
The minimum context attached to a training failure so another system or human
can attribute it: rank, host, device, step, phase, distributed context, and
lineage.
_Avoid_: Error log, crash report

**First-Fault Attribution**:
A best-effort classification of where a distributed failure appears to have
originated, including the confidence of that classification.
_Avoid_: Root cause, guilty rank

**Typed Incident Record**:
An append-only run-evidence row that carries the process event envelope plus a
classified fault class, capture state, disposition policy, and best-effort
first-fault attribution for one diagnosable failure or degraded-progress event.
It records a diagnosis; it is not proof that recovery was attempted.
_Avoid_: Error log entry, alert, exception record

**Attempt Outcome Aggregate**:
An immutable post-hoc reduction of every per-process outcome in one attempt into
a single attempt-level result, where a missing process outcome or inconsistent
world size is incomplete rather than success.
_Avoid_: Job status, central collector, cluster health

**Semantic Continuity**:
The production recovery standard that a resumed run has not lost, duplicated,
or silently changed training state, even when bitwise identity is not promised.
_Avoid_: Approximate restart, resumed successfully
