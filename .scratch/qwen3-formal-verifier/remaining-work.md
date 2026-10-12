# Remaining work for the Qwen3 formal verification vehicle

State as of `eb1dba3c8`, pushed to `origin/ultron/mainline`. Every claim below
is measured at that commit, with the command or location that establishes it.
Items are sized as **S** (hours), **M** (a day), **L** (multi-day), and each
names what would make it verified rather than merely done.

## Measured starting state

Six of the nine `IncidentClass` values now emit from production code. The four
direct call sites:

| class | emitter |
|---|---|
| `NONFINITE_LOSS` | `trainer.py:940` |
| `COMPUTE_STRAGGLER` | `trainer.py:995` |
| `INCONSISTENT_RANK_CONFIG` | `run_evidence.py:426` |
| `CHECKPOINT_CORRUPTION` | `checkpoint.py:196` |

`COLLECTIVE_HANG` and `RANK_DEATH` emit indirectly through
`record_collective_failure`, called from the train loop's exception handler.
They pass the class as a variable, so a grep for the literal does not find them
-- worth knowing before concluding they are unwired.

Three do not emit at all, and they are items 1, 2 and 3 below.

Verification currently standing: 33 mutations all killing; the nine-stage gate
green at `20f6dabc8`; a live four-GPU run recording real straggler incidents; a
TLC mesh ladder to eight ranks.

## 1. Wire `DATALOADER_STRAGGLER` -- S

`detect_step_stragglers` already supports it: the class is a parameter and the
attribution locus branches on it (`distributed_faults.py:160,194`). Nothing in
production passes it, because the trainer times only the whole `train_step`
(`trainer.py`, around the `step_started_ns` measurement). So a dataloader stall
is currently reported as a compute straggler, against `TRAINER_RANK`, which
sends a reader to the wrong subsystem.

`MetricsProcessor` already accumulates `data_loading_times`
(`metrics.py:366,407`), so the duration exists; it needs to reach the seam. The
cheap shape is to time the batch fetch inside `train_step` and run a second
comparison on the same logging-step cadence, reusing the existing all_gather
rather than adding another.

**Verified when:** an injected dataloader delay on one rank is reported as
`dataloader_straggler` with `detected_locus=dataloader`, and an injected compute
delay is still reported as `compute_straggler`. The pair matters -- one test
alone cannot show the two are distinguished.

## 2. Emit `CHECKPOINT_INTERRUPTION` -- M

No emitter. The class is for a checkpoint that was interrupted mid-write rather
than corrupted after it, and the two want different dispositions: an interrupted
write is a retry candidate, a corrupt one is not.

The seam is the save path in `torchtitan/components/checkpoint.py`, not the load
path where the integrity sentinel lives. Distinguishing the two needs a
marker written before and cleared after a save, so a manifest present without a
completed marker is an interruption.

**Verified when:** a save killed mid-write is detected on the next load as
`checkpoint_interruption` rather than `checkpoint_corruption`, and a genuinely
truncated shard is still the latter. Deterministic injection is feasible -- kill
between the two writes -- which is why this is M rather than L.

## 3. `HARDWARE_EVENT` and the DCGM/EUD drills -- L

No emitter, and no DCGM integration beyond the label. This is the one v1 row
that needs infrastructure rather than a detector: `dcgmi` field collection, a
threshold policy, and stopped-job EUD per the tiering in CLAUDE.md. Treat as its
own campaign, not an increment on this work.

**Abandon criterion:** if DCGM cannot be read without root inside the rootfs,
stop and record the boundary rather than adding a privileged path to core.

## 4. NCCL fault signatures -- M

`distributed_faults.py` classifies **Gloo** text only, and says so. NCCL reports
timeouts and aborts through a different path, so a real NCCL hang currently
returns unclassified -- correct behaviour, but it means the two most valuable
classes do not fire on the backend production actually uses.

The work is measurement, not design: inject a hang and a rank death on NCCL,
record the verbatim signatures, and extend the table with the backend recorded
in the metadata so a reader can tell which table matched. Needs real GPUs.

**Verified when:** the same two injections used for Gloo produce the same two
classes on NCCL, and an unrecognized NCCL error still returns unclassified. Keep
the Gloo tests -- they are the CPU-runnable half.

## 5. Per-step durations in the evidence bundle -- M

Live detection works, but post-hoc analysis over a sealed bundle does not,
because per-step durations are not recorded. `RunEvidence` has `monotonic_ns`
per event and `elapsed_monotonic_ns` per process, and `MetricsProcessor`
averages its timings over `log_freq` into TensorBoard scalars. Note `metrics.py`
does call `record_artifact`, which is easy to misread as the timing reaching the
bundle: it registers the TensorBoard *file*, not the values.

This is a schema addition -- a metric record type, or a periodic incident-free
row -- and must hold the Tier 0 budget of 1% median throughput.

**Verified when:** `aggregate_outcome.py` can identify the straggler from a
sealed bundle alone, with no live process, and the throughput delta against the
same config without it is inside 1% over at least ten steps.

## 6. Finish the mesh parameterization -- M, and one L

Ticket 28 holds the detail. Ordered by value:

- **Replace the hand-written `Ranks2x2` quadruple with generated definitions**
  (S). Safe now: the derived instance provably explores the same state graph,
  38321 distinct. This also retires `run_tlc_device_mesh_model.sh:189-200`'s awk
  scrape of those four names, which silently reports the wrong bound if they are
  renamed.
- **Make the exporter record `batch` and `loss`** (M). `_runtime_topology`
  already calls `get_all_one_dimensional_meshes()` and discards all but `fsdp`
  and `tp`, via a hardcoded two-tuple and a hardcoded six-element coordinate.
  This one moves the canonical trace: `--update-artifacts`, a `RAW_SCHEMA`
  decision, and two gate runs.
- **`MeshTopology.tla`** hardcodes 2x2 independently (L, or drop it).
- **A third non-trivial axis** stays out until the `dp_replicate` mesh-name
  convention is *observed*. Inventing one is the unmeasured assumption this work
  exists to remove.

## 7. The contract chain, tickets 04-09 -- L

Restated against what now exists, since the originals predate the protocol
model, the Lean proof and the bridge:

- **04** promote the versioned trace contract. Still needed, and its premise
  widened: `single_rank.py:37` pins `raw.v0` while `device_mesh.py` pins
  `raw.v1`, an un-contracted fork.
- **05, 06, 07** largely subsumed by the bridge and the fidelity differential.
  Residues only: single-rank Lean independence, and Lean-side transition
  establishment.
- **08** wholly outstanding, and it owns the detection-latency criterion that
  the fault work should now be able to satisfy.
- **09** is the genuine missing half. The positional zip still ships and is
  cited in the code as awaiting it (`device_mesh.py:152`). A joined key removes
  the last place where correspondence is assumed from ordering.
- **24** the four-rank unit fixture is still not a trace `DeviceMeshValid`
  accepts, which is why end-to-end checker runs against it are not repeatable.

## 8. Verification debt -- S, blocked

- **Re-run the full nine-stage gate on HEAD.** Blocked: GPU 0 is wedged, and the
  device list is a pinned contract in three places. Ten commits after
  `20f6dabc8`, including the core trainer and evidence changes, are backed by
  unit tests, 33 mutations and a live run but not by the gate. Do not relax the
  device assertion to get a pass.
- **Pin the checkpoint sentinel tests to CPU.** Ten of them need a free GPU and
  have no reason to; under contention they fail and a flaky guard gets skipped.
- **Tier 0 is 12:13**, of which one Bazel target is 293 s. If it needs to come
  down, that target or the fidelity instance count is the lever -- not the
  mutation manifest, whose cost scales with how much is actually guarded.

## Suggested order

1, 6-first-bullet, 8-second-bullet first: all small, all remove a wrong or
missing attribution. Then 4 and 5, which together turn the fault suite from
"detects live" into "detectable after the fact on the backend production uses".
Then 2. Then 7 as its own effort, with 09 before 04. Leave 3 and the third mesh
axis until something forces them.
