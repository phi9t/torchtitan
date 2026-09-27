# 26 — Turn the reserved incident classes into proven detections

**What to build:** Deterministic injectors and typed detections for the v1 fault
classes that `IncidentClass` reserves but nothing emits, starting with
inconsistent per-rank config and collective hang.

**Blocked by:** none. **Blocks:** 08 — promotion boundary (its detection-latency
criterion).

**Status:** faults 1 and 3 landed and mutation-tested; their multi-rank
injected runs and faults 2, 4, 5 outstanding

## What is proven so far, precisely

Landed for fault 1 (inconsistent per-rank config):

- `_validate_shared_manifest` now names the differing fields, reports config
  divergence by the two digests the manifest already carries, and emits a typed
  `INCONSISTENT_RANK_CONFIG` incident with `ABORT_FATAL`, `ABORT_AND_PRESERVE`
  and `TRAINER_RANK`, attributed to the rank that observed the divergence.
- `attribution_confidence` is deliberately omitted: a rank sees only that it
  disagrees with whichever rank published first, so which side is wrong is not
  determined. Guessing it would be an overclaim.
- The comparison's scope is pinned as `SHARED_MANIFEST_FIELDS` and the
  non-detection is a test, not a comment: divergence outside those fields --
  per-rank environment, host libraries, device state -- is invisible here.
- Three mutations confirmed to kill the tests: removing the recording call,
  reverting to the field-less message, and emitting through the module-level
  `record_incident` facade. The third matters most. `__enter__` publishes the
  manifest *before* it opens the process index and *before* it installs the
  recorder as active, so the facade would find no active recorder and drop the
  incident silently. That is detection with no record, and it passes any test
  that only checks the raised error.

**Not proven, and not claimed:** this is verified with two `RunEvidence`
instances holding genuinely different configs, which exercises the real detector
on the real artifact. It is not a full-stack multi-rank injected run, because no
injector exists to make one rank of a live job run a different config -- the
config system has no env-override path. Until that injector exists, the evidence
covers the detector, not the job.


## Why this ticket exists rather than 04-09

The formalization proves properties of a healthy run. Nothing yet shows it
*detects* a fault, because no fault has ever been injected. I first assigned
this work to tickets 04-09, which was wrong: those are the contract and
refinement chain (04 promotes the versioned trace contract, 05 and 06 refine
single-rank and distributed semantics, 07 composes the bridge, 08 measures the
promotion boundary) plus 09, the Kineto joined key. Only 08 carries a fault
criterion, and the "controlled mutations" in 05/06/07 are trace mutations, which
already ship.

I also recorded that the repository contained no fault injection whatsoever.
That came from grepping `tests/` and `torchtitan/components/` and generalising
to the repository. It is false, and the corrected picture is the starting point
below.

## What already exists

- `torchtitan/observability/run_evidence.py:65-81` defines all nine v1 classes,
  and says in its own docstring that v1 wires one real emitter and the other
  eight are "schema-reserved, not proven fault injections".
- `record_incident` (`run_evidence.py:748`) is a complete keyword-only emission
  API: class, capture state, policy, summary, locus, confidence, step, last
  operation, work-preserved, terminal disposition, metadata. It is a no-op when
  no recorder is installed and it deliberately downgrades its own write failure
  to a log while another exception is in flight.
- Exactly one production caller: `torchtitan/trainer.py:936`, `NONFINITE_LOSS`.
  It fires only on steps where loss is logged.
- `docs/adr/0007-incident-records-and-post-hoc-outcome-aggregation.md:40-50`
  records this boundary deliberately: the schema landed first so that injection
  could produce meaningful evidence.
- `state_estimator/` carries mode inference and probe recommendations, with
  `CALIBRATION_STATE = "heuristic_uncalibrated"` and an
  `InjectedFaultEvaluationCase` whose only construction site is a unit test.

What does not exist is any runtime injector: no `os.kill`, no delay hook in the
trainer or dataloader, no checkpoint-shard corruption, and no manifest-integrity
validator in `torchtitan/components/checkpoint.py`.

## The detector for fault 1 already exists and is unclassified

`_validate_shared_manifest` (`run_evidence.py:363-376`) compares seven fields of
the shared run manifest -- including `config`, which carries both the normalized
job config and its sha256 -- and raises `EvidenceContractError` when any differ.

Every rank that loses the race to create the manifest performs this comparison,
so a divergent rank is always caught by somebody: if the divergent rank writes
first, every other rank raises against it. Detection does not depend on which
rank wins.

Three things are wrong with it as evidence:

1. It emits no incident, so a real inconsistent-config abort produces no
   `INCONSISTENT_RANK_CONFIG` record and the aggregator cannot classify it.
2. Its message -- "existing run evidence manifest does not match" -- names no
   field, no rank and no digest. This is the same defect as the gate's bare
   "dirty source identity changed", which cost a GPU run to diagnose.
3. Nothing tests the divergent case, so the check's own failure path is
   unproven.

## Shape

Per fault, four artifacts. The fourth matters as much as the first three.

1. A deterministic injector, off by default.
2. The evidence it leaves in the bundle.
3. The check that fires, **named in the result token**, with correct
   attribution -- not a different check firing incidentally.
4. A recorded non-detection wherever the evidence genuinely cannot see the
   fault, rather than a weakened check that appears to cover it. The
   no-`Partial` check is the precedent: it inspects parameters, not gradients,
   because the trace records no gradient placement, and it says so (ticket 23).

### Fault 1 -- inconsistent per-rank config

Smallest and highest-value, because the detector exists.

- Make `_validate_shared_manifest` name the differing fields, and for `config`
  the differing digests, before raising.
- Emit `INCONSISTENT_RANK_CONFIG` with
  `detected_locus=TRAINER_RANK`, `policy=ABORT_FATAL`,
  `capture_state=ABORT_AND_PRESERVE`, and metadata naming the fields.
- Injector: perturb one rank's config after parsing, off by default, selected by
  rank so the divergence is deterministic.
- Non-detection to record explicitly: fields excluded from the comparison cannot
  be detected this way. State which, and prove a divergence in an excluded field
  is *not* reported, so the check's scope is measured rather than assumed.

### Fault 3 -- checkpoint manifest integrity: LANDED

`docs/robust_training_reliability.md:113-118` names this as the first of four
minimum correctness sentinels and it had no implementation. It now runs in
`dcp_load`'s plain-DCP branch immediately before `dcp.load`, so the defect is
named while nothing has been loaded and the abort is attributed to the
checkpoint path rather than to whatever the reader raises from a worker thread.

Four defects detected, each with its own result token: `MANIFEST_MISSING`,
`MANIFEST_UNREADABLE`, `SHARD_MISSING`, `SHARD_TRUNCATED`. The incident is
`CHECKPOINT_CORRUPTION` / `ABORT_FATAL` / `CHECKPOINT_PATH`, and unlike fault 1
it does carry `attribution_confidence=OBSERVED_LOCAL_FAULT`, which is justified:
this process read the manifest and stat-ed the files itself, inferring nothing
from a peer.

Truncation detection needed a measurement to establish. `ChunkStorageMetadata`
offsets are tensor coordinates, not file bytes; the byte ranges live in
`Metadata.storage_data`, which for `FileSystemWriter` maps `MetadataIndex` to
`_StorageInfo(relative_path, offset, length, ...)`. Confirmed against a real
`dcp.save`, where the shard's size equalled the last recorded offset plus its
length exactly. `storage_data` is typed `Any` and writer-private, so the fields
are read by duck typing and an unrecognised layout skips shard checks with a
warning rather than reporting a defect it did not observe.

Cost, measured inside the rootfs: 0.066 ms median for one shard, 2.794 ms for
512. It is O(num_shards) stat pairs paid once per load, so it does not touch
steady-state throughput.

Non-detections recorded as tests rather than comments: bit flips inside shard
data are invisible because DCP writes no per-shard checksum; the HF safetensors
path is not covered and warns once per job that it is not; a backend that
refuses a size probe skips truncation detection for that file with a warning,
proven against a genuinely truncated shard; a shard longer than required is not
a defect because trailing bytes are never read.

One discovery worth recording: eight pre-existing tests in `test_checkpoint.py`
were fixtured with a zero-byte touched `.metadata` -- which is exactly the
`MANIFEST_UNREADABLE` defect -- and began failing the moment the sentinel was
wired. They were repaired with a hand-built real `Metadata` pickle and matching
shard, not by weakening the sentinel. The fixtures had been describing a corrupt
checkpoint all along and nothing noticed.

### Fault 2 -- collective hang

The model and the refinement bridge already reason about this, so detection can
be traced to a named invariant rather than to a crash. Requires a bounded wait,
because a detector that hangs alongside the job is not a detector.

### Faults 3-5

Checkpoint corruption including manifest integrity, rank death, and compute plus
dataloader stragglers. Rank death and the hardware drills come last: they need
infrastructure rather than formal work.

## Acceptance

- Each landed fault has an injected run and a healthy run, with the token naming
  which check fired.
- A passing gate is explicitly **not** sufficient: it shows the properties hold
  on the run that was made.
- Every new guard ships with the mutation that kills it, registered rather than
  described.
- Detection latency reported per fault, separately from throughput.

## Abandon criterion

If a fault cannot be injected deterministically, stop and record why. A flaky
injector produces flaky evidence, which is worse than none.
