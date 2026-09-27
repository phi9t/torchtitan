# 26 — Turn the reserved incident classes into proven detections

**What to build:** Deterministic injectors and typed detections for the v1 fault
classes that `IncidentClass` reserves but nothing emits, starting with
inconsistent per-rank config and collective hang.

**Blocked by:** none. **Blocks:** 08 — promotion boundary (its detection-latency
criterion).

**Status:** all five faults landed with multi-rank injected runs and
mutation-tested. Trainer wiring and NCCL coverage outstanding.

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

**Multi-process injected run, added later.** The single-process test constructs
two recorders sequentially, which never exercises the mechanism the detector
actually relies on: an exclusive-create race on the shared manifest. There is
now a four-process concurrent run where one rank holds a divergent config.

Which rank wins the race is not deterministic and the test does not pretend
otherwise. What is deterministic is the invariant: with two distinct configs
present, somebody always disagrees with the publisher. If the divergent rank
wins, the other three detect; otherwise the divergent rank detects alone. So the
detection count is 1 or 3, never 0, and every detector must leave a typed
incident naming the config field. Asserting the invariant rather than a winner
is what makes a racing test trustworthy instead of flaky.

The mutation that removes the recording is pinned against this test too, and
surfaces sharply: without it the detecting rank's index file is never created at
all.

**What is still not claimed.** This is four real processes racing on a real
filesystem, not a fault injected into a live `Trainer` step on GPUs. No injector
exists to make one rank of a real training job hold a different config, because
the config system has no env-override path. The evidence now covers the detector
under genuine concurrency; it still does not cover a running training job.


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

**Multi-rank injected run, added later.** Four processes with a real gloo group
perform one `dcp.save`, rank 0 truncates a shard, and every rank then validates.
All four detect, which is the property that matters: the sentinel reads the
whole manifest, so detection must not depend on which rank wrote the damaged
shard. Gloo on CPU is deliberate -- manifest integrity has nothing to do with
the accelerator, and a CPU-only test runs anywhere. Under the mutation that
neuters the truncation comparison, every rank reports MISSED, meaning each one
loaded a damaged checkpoint.

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

### Faults 2 and 4 -- collective hang and rank death: LANDED

My own blocker claim for fault 2 was wrong, and the correction is the useful
part. I wrote that detection is timeout-based and therefore not deterministic.
That conflates latency with outcome: a rank that never issues a collective
always causes its peers to time out. The latency varies; the outcome does not.
That makes it injectable.

Measured on four ranks with a four-second Gloo timeout, the two faults are
distinguishable, which was not obvious beforehand:

| injection | Gloo signature |
|---|---|
| rank stays alive, never issues | `Timed out waiting 4000ms for recv/send operation` |
| rank exits, closing its sockets | `Connection closed by peer`, `Read error` |

Both arrive as a plain `RuntimeError` from the same collective, so the class has
to come from the message. `torchtitan/observability/distributed_faults.py`
classifies them and records the matched signature in the incident metadata, so a
reader can see what the classification rested on rather than trusting it.

`FaultConfidence.COLLECTIVE_TIMEOUT_INSUFFICIENT_EVIDENCE` already existed for
exactly this ambiguity -- a timeout says a peer did not arrive, not which peer
or why -- and that is what a hang records. Rank death records
`SUSPECTED_PEER_FAULT`.

Injected for real, four processes each time: one run where rank 1 sleeps through
the collective, one where it exits. Every survivor records the right class, and
neither run produces the other's class.

**The scope limit, which is narrow and must not be forgotten.** The signatures
are **Gloo's**. NCCL reports timeouts and aborts through a different path with
different text and is **not covered**. An unrecognized error is returned
unclassified rather than guessed, and that is tested with a NCCL watchdog
message among the cases -- because over-classification is the dangerous
direction here: a wrong class sends a reader to the wrong subsystem, which is
worse than an honest "unclassified". The mutation that makes the table match
everything is caught by precisely that case.

Nothing is wired into the trainer. This is a classifier plus proof that it works
on real injected faults; deciding where the core catches a failed collective is
a separate change.

### Fault 5 -- stragglers: LANDED

The blocker I recorded was real but narrower than it looked. Straggler detection
does need per-step durations, and those are not in the evidence bundle. But the
detector does not have to read them from the bundle: a caller can hand them
over, and in a distributed job every rank can learn every rank's duration with
one `all_gather`. So the same boundary used for faults 2 and 4 applies -- build
the detector, prove it on a real injected fault, and leave the trainer wiring as
its own change.

`detect_step_stragglers` compares each rank against the **median**, and that is
the whole design rather than a detail. A slow rank drags a mean toward itself
and can hide under any threshold, which is exactly the case the check exists
for. The mutation swapping median for mean is pinned against the real
four-process injected run, where it reports an empty result: a genuine
twentyfold straggler goes completely undetected. A synthetic case catches it
too, but the injected run is the one that makes the point.

Three deliberate refusals, each tested rather than commented:

- `slowdown_threshold` is required, not defaulted. What counts as a straggler is
  a property of the job, and a default here would be a number this module
  invented and then everyone quietly relied on. A threshold at or below 1.0 is
  refused, since it would flag the median itself.
- Fewer than three ranks returns nothing by construction. With two samples every
  median lies between them, so "the slow one" is not separable from "the fast
  one" without an external baseline this function does not have.
- The incident class is the caller's to supply, because the duration's meaning
  is invisible from the number: the same comparison distinguishes a compute
  straggler from a dataloader one depending on which phase was timed. The two
  get different attribution loci, and passing any other class is a programmer
  error.

A straggler is a performance fault, so it records `CONTINUE_BOUNDED_WARNING` and
`useful_work_preserved=True` -- the step produced the right answer, late. That
is the one fault here that should not abort.

Injected for real: four processes, a fixed 1.0 s delay on one rank against 0.05
s elsewhere, durations gathered by `all_gather`, and every rank independently
names the same straggler. Unanimity matters: ranks disagreeing would make the
detection useless for attribution.

**Still not wired.** Nothing calls this from the trainer, and per-step durations
still are not in the bundle, so a post-hoc detector over sealed evidence remains
future work. What exists is a detector proven against a real injected fault.

### What the evidence gap for fault 5 still is

Investigated rather than assumed. None of the three is a small change, and each
is blocked on something that does not exist yet:

**Fault 2, collective hang.** Detection is inherently timeout-based, which puts
it straight against this ticket's abandon criterion: a fixed seed does not make
a wall-clock threshold deterministic. It is still the most valuable of the
three, because the protocol model and the refinement bridge already reason about
exactly this hazard, so a detection could be traced to a named invariant rather
than to a crash. What it needs is a bounded wait -- a detector that blocks
alongside the job is not a detector -- which means integrating with the NCCL
watchdog rather than adding a wait of our own.

**Fault 4, rank death.** Needs process supervision to kill a rank and observe
the survivors' view. That is launcher-level infrastructure, not formal work,
which is why this ticket put it last.

**Fault 5, stragglers.** The blocker is evidence, not detection logic. Straggler
detection is a cross-rank comparison of per-step durations, and those durations
are not in the bundle:

- `RunEvidence` records `monotonic_ns` per event and `elapsed_monotonic_ns` per
  process outcome, but no per-step duration.
- `MetricsProcessor` does compute `time_metrics/end_to_end(s)`,
  `time_data_loading` and `data_loading_times`, but it averages them over
  `log_freq` and emits them as TensorBoard scalars.
- `metrics.py` does import `record_artifact`, which is easy to misread as the
  timing reaching the bundle. It does not: the artifact records the TensorBoard
  log *file*, not the values.

So a straggler detector joined by rank and step needs per-step timing added to
the evidence bundle first, as Tier 0 always-on structured metrics, under the 1%
median throughput budget. Post-hoc detection over the bundle would then be
deterministic and CPU-only, which is the attractive part --
`aggregate_outcome.py` already exists for exactly that shape of analysis. The
detector is the easy half; the evidence is the work.

### Faults 3-5

Checkpoint corruption including manifest integrity, rank death, and compute plus
dataloader stragglers. Rank death and the hardware drills come last: they need
infrastructure rather than formal work.

## Blocked: the full gate cannot run while GPU 0 is wedged

Recorded so the next attempt does not mistake this for a regression. GPU 0 on
this host fails a bare `torch.ones(4, device="cuda:0")` with
`cudaErrorDevicesUnavailable`, while GPUs 2, 4, 5 and 6 pass the identical
probe. Compute mode is `Default`, persistence is on, ECC uncorrected counts are
zero, and `nvidia-smi` reports the device idle at ~130 MiB, so nothing in the
usual places shows it.

That blocks the nine-stage gate, because the device list is a pinned contract in
three places -- `run_scout_b.sh:102`, `run_scout_a.sh:96` and `scout_b.py:3553`
-- and the Scout A regression stage requires `CUDA_VISIBLE_DEVICES=0`
specifically. With devices overridden to 2,4,5,6 the gate reaches five stages,
passing `source_manifest`, `focused_pytest`, `cuda_pytest` (a real four-rank
Qwen3 step), `owning_pytest` and `artifact_sync`, then stops at
`scout_a_regression` on the device assertion.

**The assertion should not be relaxed to get a green gate.** Device identity is
part of the evidence identity chain, and weakening a pinned contract to make a
run pass is the failure mode this whole program exists to prevent. The gate
passed in full earlier in the session on the commit before the trainer wiring,
so the blocker is the hardware, not the change. Re-run it once the device
recovers.

This also corrects an earlier diagnosis in this ticket: the ten checkpoint
sentinel tests that failed with the same error were not hitting general
contention, they were hitting this one device.

## Follow-up found while adding the multi-rank runs

The ten single-process checkpoint sentinel tests need a free GPU, and they do
not need to. They call a real `dcp.save` through `build_real_dcp_checkpoint`,
and DCP reaches for the accelerator whenever one is visible, so under GPU
contention all ten fail with `CUDA error: CUDA-capable device(s) is/are busy or
unavailable`. Observed on a loaded box: 11 failed, 334 passed. With
`CUDA_VISIBLE_DEVICES=""` the same suite is 57 passed.

This matters beyond the inconvenience. A test that fails when somebody else is
using a GPU gets marked flaky and then skipped, and a skipped guard checks
nothing -- the same failure mode this program keeps finding in other forms.
Manifest integrity is pure filesystem work and has no accelerator dependency at
all, which is why the four-rank test added here pins its children to CPU
explicitly.

Fix worth making: pin the sentinel tests CPU-only rather than relying on a free
device. Not done here because changing how ten tests invoke DCP is its own
change with its own risk, and it should not ride along with a fault-injection
commit.

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
