# Distributed Training Observability: A Bounded Paper Closure

Date: 2026-08-12

## Executive summary

The two seed papers describe complementary systems, not competing monitoring
stacks:

- ByteRobust is an incident-observability and recovery control plane. It
  combines low-cost continuous signals, on-demand rank/process inspection,
  stop-time diagnostics, rollback/replay, and failure-domain-aware
  checkpointing. Its central operational choice is to isolate a useful suspect
  set quickly rather than wait for a perfectly precise root cause.
- Fault-Tolerant HSDP (FT-HSDP) is a recovery data plane. It makes the HSDP
  replica the failure unit, records enough membership and step state to make a
  consistent commit/retry decision, catches up a recovering replica from peer
  state, and explicitly evaluates the numerical effect of changing replica
  participation.

Together, they imply that TorchTitan should make a repo-local run bundle the
source of truth and use a tiered capture policy:

1. Keep cheap, structured progress, health, lineage, and collective history on
   for every run.
2. Trigger bounded rank/process snapshots and short timeline captures when
   progress, parity, or peer-relative timing becomes anomalous.
3. Stop training before running active hardware, fabric, or kernel replay
   diagnostics.

This is not a recommendation to put fleet management in TorchTitan. The repo
should own training-semantic evidence and reproducible artifacts; a launcher or
platform can join host health, choose retries, or quarantine hardware. That
boundary matches the existing reliability ADR.

## Method and bounds

### Inclusion criteria

The paper closure includes:

- both seed papers;
- a paper directly cited by either seed when it materially defines an
  observability mechanism, failure model, recovery mechanism, or evaluation
  baseline used to reason about distributed model training; and
- a paper directly citing a seed when it supplies a materially different
  correctness or recovery baseline.

Only primary sources were used: the papers, official proceedings pages,
official author/project pages, and official source or product documentation.
Tool documentation is inventoried separately and does not consume the paper
cap.

### Exclusion criteria

The closure excludes generic model, optimizer, parallelism, scheduling, and
networking papers unless a seed uses them to define a failure or observation
contract. It also excludes secondary summaries; papers that merely mention a
seed; systems without an observable mechanism or relevant evaluation
baseline; and sources whose operational details are not publicly inspectable.

For example, pipeline scheduling, generic HSDP/FSDP, and all-reduce algorithm
papers explain the training substrate but not the observability contract.
General cloud gray-failure and storage papers are useful background but add no
training-specific mechanism beyond the included sources. Newer citing work was
excluded when it only proposed another recovery layout; ReCoVer was retained
because it challenges the numerical-work invariant directly.

### Stop conditions

- Citation depth: at most one hop backward or forward from either seed.
- Source cap: at most 12 papers, including the seeds.
- Second-hop exception: permitted only if a seed directly adopted an
  instrumentation or diagnostic method whose first-hop source did not specify
  it. No exception was needed.
- Search stopped when the 12-source cap was reached and every retained source
  mapped to at least one of: kernel/module attribution, distributed liveness,
  job/host health, recovery, checkpoint lineage, correctness, or evaluation.
- Temporal cutoff: 2026-08-12.

## Paper closure (12 sources)

| Source | Relationship and reason for inclusion |
| --- | --- |
| [Robust LLM Training Infrastructure at ByteDance (ByteRobust)](https://arxiv.org/abs/2509.16293) | Seed; defines a layered incident-observability, diagnosis, isolation, and recovery system. |
| [Training LLMs with Fault Tolerant HSDP on 100,000 GPUs](https://arxiv.org/abs/2602.00277) | Seed; defines replica-level recovery, consistency, catch-up, and correctness evaluation. |
| [MegaScale](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng) | Cited by both seeds; supplies the deep-stack monitoring and large-scale stability baseline, including heartbeats and RDMA traffic. |
| [Characterization of Large Language Model Development in the Datacenter](https://www.usenix.org/conference/nsdi24/presentation/hu) | Cited by ByteRobust; supplies a production failure taxonomy and log-driven diagnosis baseline. |
| [Minder](https://www.usenix.org/conference/nsdi25/presentation/deng) | Cited by FT-HSDP; defines low-latency, peer-relative faulty-machine detection from runtime monitoring patterns. |
| [Evolution of Aegis](https://www.usenix.org/conference/nsdi25/presentation/dong) | Cited by ByteRobust; defines collective-library instrumentation, pre-delivery checks, and diagnosis of hangs and degradation. |
| [SuperBench](https://www.usenix.org/conference/atc24/presentation/xiong) | Cited and used as a stop-time diagnostic baseline by ByteRobust; defines proactive component and workload validation. |
| [ByteCheckpoint](https://www.usenix.org/conference/nsdi25/presentation/wan-borui) | Cited by ByteRobust; defines checkpoint instrumentation, parallelism-independent state, and performance analysis. |
| [Gemini](https://www.amazon.science/publications/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints) | Cited by both seeds; defines in-memory checkpoint placement, traffic scheduling, and recovery-cost baselines. |
| [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/abs/2410.21680) | Cited by ByteRobust; supplies a failure taxonomy, effective-training-time model, and cluster-scale baseline. |
| [Understanding Silent Data Corruption in LLM Training](https://aclanthology.org/2025.acl-long.996/) | Cited by ByteRobust; supplies an LLM-specific SDC failure model and targeted detection/mitigation baseline. |
| [ReCoVer](https://arxiv.org/abs/2605.11215) | Directly cites FT-HSDP; retained as the one forward-citation because it makes constant microbatches per iteration a recovery invariant and compares fault-tolerant HSDP against an unfailed trajectory. |

## Direct findings from the closure

### ByteRobust: observe broadly, diagnose hierarchically

ByteRobust separates explicit failures (clear logs, exit codes, or hardware
events), implicit failures (hangs, performance degradation, anomalous training
trajectories), and human-triggered restarts. The same symptom is not a root
cause: a hang, illegal memory access, or NaN can originate in infrastructure or
user code. This argues against alert schemas that label a host faulty merely
because a rank was slow or first to time out.

Its always-on monitor uses second-level, non-workload GPU/host/network queries,
training series such as loss, gradient norm, and MFU, logs and exit codes, and
CUDA/RDMA/host/storage events. A performance decline or missing RDMA traffic
triggers deeper inspection; high-confidence hardware events can bypass it.
When the job is suspended, diagnostics proceed from targeted GPU and network
tests to code rollback and deterministic replay. The seed reports that most
incidents resolve with simple eviction, reattempt, or rollback, and only a
small tail needs dual-phase replay. This supports an escalation ladder rather
than permanent deep profiling. See the [ByteRobust paper](https://arxiv.org/abs/2509.16293).

For an implicit hang or fail-slow event, ByteRobust uses py-spy and PyTorch
Flight Recorder to inspect the process tree and stack/collective state. It
clusters stacks across machines, treats dominant groups as the healthy
baseline, maps outliers onto TP/PP/DP fault domains, and may evict the shared
parallel group. This is deliberately coarse attribution. It is useful even
when the precise failed sender/receiver cannot be identified.

Its correctness fallback is workload-shaped replay: a deterministic reference
model, fixed weights/input, and fixed TP/PP/DP or EP/PP/DP arrangement produces
rank-comparable results. Dual horizontal/vertical groupings reduce the suspect
set while retaining relevant communication. ByteRobust also warns that active
diagnostics have side effects: its deployment observed EUD altering a prior
frequency lock, and the paper reports imperfect EUD recall for SDC. An active
diagnostic result must therefore record tool/driver versions, before/after
device settings, target serials, and the exact stopped-job window; it is not a
transparent metric.

ByteRobust evaluates a full incident lifecycle: detection, localization,
failover, recomputation, per-mechanism incident resolution, checkpoint blocking
time, restart time, MFU, and cumulative plus sliding-window effective training
time ratio (ETTR). A profiler that only reports faster kernels cannot establish
that a months-long training job is more productive.

### FT-HSDP: recovery state is observable state

FT-HSDP reports that hardware dominated interruptions in a 32K-GPU workload,
with HBM, PCIe, NCCL watchdog, faulty compute, kernel, SSD, network, and SDC
categories all material. It also describes a continuous root-cause analyzer
that samples collective telemetry every 5-20 seconds, builds a wait-for graph
first across collectives and then ranks, and distinguishes ranks that never
joined a collective from those that joined but stopped making progress. For
fail-slow behavior it compares rank aggregate collective completion times. The
paper reports less than 1 MB of host-DRAM telemetry per GPU and no observed
training impact for that internal implementation, but that result should not be
assumed for a different recorder. See the [FT-HSDP paper](https://arxiv.org/abs/2602.00277).

The recovery protocol makes several new facts observable per step:

- replica membership and health epoch;
- next step proposed by each replica;
- whether every rank in a replica finished gradient exchange;
- whether the replica committed the optimizer step or discarded gradients and
  retried;
- whether a replica is healthy, behind, fetching state, sending a zero
  gradient, or rejoined;
- the checkpoint/source replica used for catch-up; and
- the number of healthy replicas, actual batch/tokens processed, and any
  learning-rate intervention.

Without these fields, loss discontinuity after recovery cannot be separated
from ordinary model noise. FT-HSDP also keeps per-replica dataloader progress
each step and requires every batch to be trained exactly once. That makes data
position part of checkpoint lineage, not merely loader debugging information.

Its failure-free throughput, failure-detection/reconfiguration/retrain stall,
catch-up stall, peer checkpoint fetch, FTAR bandwidth, loss, and downstream
evaluation are separate measurements. At 98K GPUs it exposes cold-path costs
that a steady-state trace hides, including initialization, consensus,
reconfiguration, checkpoint load, and the first recovered step. CPU-scale
emulation caught control-plane bugs before the GPU allocation existed, but the
paper explicitly notes that such emulation does not validate GPU code.

### What the one-hop sources add

- MegaScale shows why a common timeline must join application phase,
  communication, data loading, memory, and hardware/network series. A decline
  in all of them is correlation, not attribution; deep-stack evidence is needed
  to distinguish the upstream bottleneck. [MegaScale](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng)
- Minder treats each job's healthy peers as its baseline and learns temporal
  multivariate patterns rather than applying one global utilization threshold.
  This supports peer-relative and persistence-aware straggler evidence.
  [Minder](https://www.usenix.org/conference/nsdi25/presentation/deng)
- Aegis moves high-value observation into the collective library so runtime
  state can be captured without customer-code changes, and separates
  pre-delivery validation from in-job localization. [Aegis](https://www.usenix.org/conference/nsdi25/presentation/dong)
- SuperBench validates both components and representative AI workloads, then
  chooses a bounded benchmark subset based on diagnostic value and cost. It is
  a stop-time/preflight suite, not an always-on profiler.
  [SuperBench paper](https://www.usenix.org/conference/atc24/presentation/xiong)
- ByteCheckpoint and Gemini make save placement and transfer interference
  visible. State availability across the expected failure domain, not just a
  successful save call, determines recoverability.
  [ByteCheckpoint](https://www.usenix.org/conference/nsdi25/presentation/wan-borui),
  [Gemini](https://www.amazon.science/publications/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints)
- The cluster characterization and reliability studies show that the objective
  needs both job-level incident distributions and useful-work ratios; a single
  large-job MTBF is insufficient for a mixed research environment.
  [Datacenter characterization](https://www.usenix.org/conference/nsdi24/presentation/hu),
  [reliability study](https://arxiv.org/abs/2410.21680)
- The LLM SDC study reinforces that NaN/Inf checks alone cannot cover wrong but
  finite execution, and that reproducibility varies with fault and thermal
  conditions. [LLM SDC paper](https://aclanthology.org/2025.acl-long.996/)
- ReCoVer exposes a real semantic alternative to FT-HSDP: preserving a constant
  number of microbatches per iteration rather than varying work with live
  replicas. Whichever policy is researched, the run bundle must record actual
  per-step work and the intended invariant; `step` alone is ambiguous.
  [ReCoVer](https://arxiv.org/abs/2605.11215)

## Design inferences for TorchTitan

Everything in this section is an inference from the sources and repository
inspection, not a claim that the papers prescribe TorchTitan's implementation.

### One repo-local evidence bundle

Use one immutable directory per run attempt with a manifest and stable internal
layout. At minimum it should contain:

- normalized config and digest, source revision/dirty state, command,
  environment and library/driver/tool versions;
- `run_id`, `attempt_id`, `host_name`, `pid`, global/local/mesh rank, device
  UUID, model actor/role, and process-group/mesh membership;
- structured events and scalars with wall and monotonic clocks, step,
  microbatch, training phase, and a per-process sequence number;
- profiler, memory, Flight Recorder, py-spy, DCGM, NCCL, and deep-dive artifacts
  indexed rather than copied into ad hoc locations;
- checkpoint manifest, parent checkpoint, state-presence list, data position,
  save/load/staging timestamps, and validation status; and
- terminal outcome plus detection, diagnosis, recovery, lost-work, and
  useful-work intervals.

Every exported dashboard should be reproducible from that bundle. Repo-local
does not mean one giant trace or that every rank records every expensive tool.
It means every artifact has a local identity and provenance before optional
export.

### Correlation keys are the deep feature

Tool collection without common keys will recreate ByteRobust's original manual
investigation problem. The minimum join key is:

```text
run_id / attempt_id / role / host / pid / global_rank / mesh_rank / device_uuid
process_group_id / mesh_axes / step / microbatch / phase / event_seq / clock
checkpoint_id / parent_checkpoint_id / data_position / membership_epoch
```

CUDA device index is not sufficient across hosts or after remapping. A
communicator hash is not sufficient across processes without job/attempt
identity. Wall clock enables cross-process joins; monotonic clock and sequence
number preserve local order when clocks skew.

### Layer-specific questions

#### Kernel

Record kernel name, launch/correlation id, CUDA stream, device UUID, enclosing
operator/module/phase, shapes/dtypes where deliberately enabled, compile graph
identity, and step/microbatch. This answers whether a regression is launch
overhead, idle gaps, occupancy/resource pressure, memory traffic, or a changed
kernel choice. It does not by itself identify a slow peer or hardware defect.

#### Torch module and graph

Use stable `record_function`/NVTX ranges around model blocks, MoE routing,
attention, loss, optimizer, checkpoint staging, and collectives. Preserve eager
module FQNs plus compiled graph/code identity. Otherwise a fused/compiled trace
can be kernel-rich but model-poor. Shapes and stacks are triggered fields due to
trace volume and overhead.

#### Distributed runtime

Keep a bounded collective history with process-group identity, mesh axes,
collective type, sequence, tensor size/dtype, start/completion, timeout/error,
and rank participation. Snapshot repeated NCCL RAS or rank stacks before abort
when possible. Join the result to phase ranges and host/device identity. This
supports wait-for and peer-outlier analysis without asserting that the first
timed-out rank is faulty.

#### Job and host

Capture step/MFU/tokens and per-phase time alongside temperature, power,
clocks/throttling, ECC/Xid, PCIe/NVLink, utilization, and process accounting.
Record cold-start, compilation, checkpoint, restart, and first-step time as
separate intervals. ETTR should be derived from explicit useful/unproductive
intervals and reported cumulatively and over a window.

#### Checkpoint and lineage

Treat model, optimizer, scheduler, RNG, dataloader, and train-step state as an
atomic semantic checkpoint even if storage is sharded. Record content
inventory/digests, save completion versus staging completion, parentage,
parallel mesh, data position, and restore validation. Test recoverability
against the actual failure domain; a readable shard on the failed host is not a
backup.

#### Correctness

Always record loss, grad norm, valid tokens, nonfinite sentinels, configuration
and checkpoint lineage. Trigger deterministic replay or tensor/checkpoint
comparison for wrong-progress suspicion. For recovery research, record actual
work per step, membership, skipped/replayed data, and optimizer-commit outcome.
A finite loss or matching rounded stdout is not proof of correct execution.

## Bounded tool inventory

Origin labels: **preferred** means explicitly requested; **closure** means a
seed or included one-hop source uses it or exposes the gap it fills; **both**
means both. Costs are qualitative unless the primary documentation states a
number. The inventory is intentionally bounded to tools that answer a distinct
question.

| Tool (origin) | Scope and activation | Cost / artifact | Required correlation and questions answered |
| --- | --- | --- | --- |
| TorchTitan structured JSONL (**closure**, repo substrate) | Always-on by default through `debug.enable_structured_logging`; per-rank spans, instants, and scalars. | Small per-event CPU/file I/O; per-rank JSONL plus derived Chrome/Perfetto Gantt. Current handler is synchronous file logging, so overhead must be measured rather than assumed. | Already has host, pid, global/local rank, source, step, timestamps, caller, and sequence. Add run/attempt, role/mesh rank, device UUID, PG/mesh, microbatch, phase, membership, and lineage. Answers training-phase and cross-rank skew questions. |
| [PyTorch Profiler/Kineto](https://docs.pytorch.org/docs/stable/profiler.html) and [`record_function`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html) (**preferred**) | Scheduled CPU/CUDA operator and kernel capture; instrument stable module/phase ranges. TorchTitan already exports selected cycles. | Active windows can perturb execution; shapes, stacks, memory, and long/many-rank windows increase cost. Compressed Chrome trace per rank. | Join run/attempt/rank/device/step/microbatch and stable range/graph identity. Answers operator-to-kernel attribution, launch gaps, CPU/GPU overlap, shapes, and kernel selection. |
| [py-spy](https://github.com/benfred/py-spy) (**both**) | External sampling or instantaneous `dump` against live PIDs; `--subprocesses` covers loader/checkpoint children. | Project describes very low overhead because sampling is out of process; higher rate/native/locals increase cost and attach may require ptrace/root. SVG, speedscope/raw samples, or text stacks. | Record host, PID/process tree, thread ID/name, rank/role, step/phase, and capture time. Answers where Python workers or subprocesses are blocked or consuming CPU; cannot see GPU kernel progress. |
| [PyTorch Flight Recorder](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html) (**both**) | In-memory circular collective history enabled before process-group init; dump on timeout or demand. TorchTitan already configures buffer, dump-on-timeout, and per-rank paths. | Bounded host memory; C++ stacks and CUDA timing add CPU/event cost. Binary per-rank dumps plus `torchfrtrace` report/JSON. | Join run/attempt, rank, PG id/name, mesh axes, collective sequence/type/size, step/phase and dump trigger. Answers mismatched, missing, unfinished, or slow collectives and last-known rank progress. |
| [NVIDIA DCGM telemetry/job statistics](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/process-and-job-statistics.html) (**both**) | Node host-engine watches for GPU/NVSwitch telemetry and explicit scheduler job windows. Use low-rate continuous gauges/counters; start before work. | Low-overhead sampled fields; shorter periods/long retention consume more host-engine work/memory. Export sampled records and verbose job summary because DCGM is not durable storage. Profiling metrics can conflict with Nsight tools. | Join caller job id to run/attempt, allocation, host, GPU UUID/NVML device and PID. Answers thermal/power/clock throttling, utilization imbalance, ECC/Xid, PCIe/NVLink, energy and conflicting processes; not source/kernel attribution. |
| [DCGM EUD](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-eud.html) (**both**) | Stop-time/preflight active compute, memory, transfer, and HSIO diagnostics on supported bare-metal/passthrough systems. Pause telemetry and training. | Destructive-to-performance active workload: roughly under 5 minutes for the level-3 profile and about 20 minutes for level 4; can contend with telemetry/driver and has imperfect SDC coverage in ByteRobust. Plain log plus MLE/decoded JSON. | Record run incident, host, GPU serial/UUID, driver/EUD version, suite, start/end, result and before/after clock/settings. Answers whether a targeted device fails active subsystem tests, not whether user code caused a symptom. |
| [NCCL RAS and debug logging](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html) (**closure**) | RAS is enabled by default in recent NCCL and queried on demand; targeted `NCCL_DEBUG_SUBSYS` logs are a launch-time escalation. | RAS is designed as a lightweight TCP side network; queries may take seconds for large/unresponsive jobs. INFO/TRACE logs can be voluminous and debug settings should not remain permanently set. Text status and per-rank debug logs. | Join run/attempt, node/PID/GPU, NCCL communicator primary+secondary hashes, rank, PG/mesh and step/phase. Answers dead/unresponsive processes, communicator membership and collective-count outliers, init/topology/transport choices. |
| [NVIDIA `nccl-tests`](https://github.com/NVIDIA/nccl-tests) (**closure**) | Stop-time/preflight single- or multi-node collective correctness and bandwidth tests, chosen to match the suspect path and datatype/size. | Active exclusive GPU/network workload; correctness iterations can be slow at scale. Text or JSON with alg/bus bandwidth, latency and errors. | Record allocation/topology, hosts, UUIDs, NCCL/CUDA/driver version, operation, size/dtype, ranks and suspect incident. Answers whether the fabric/collective path reproduces correctness or throughput faults outside the model. |
| [Microsoft SuperBench](https://microsoft.github.io/superbenchmark/docs/introduction/) (**closure**) | Pre-delivery or stop-time component microbenchmarks plus representative model workloads, selected by suspected subsystem. | Active and potentially long; container/deployment effects must be recorded. Structured benchmark results, system inventory, and analysis report. | Join benchmark run to training incident, exact hosts/devices/topology and image/tool versions. Answers whether a node is a workload-relative gray failure and provides fleet/ladder baselines. Keep orchestration outside core Trainer. |
| [PyTorch CUDA memory snapshots](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) (**closure**) | Bounded allocation-history ring and triggered/periodic/OOM snapshots; already wrapped by TorchTitan with Python stacks. | Official docs estimate about 2 us per Python trace; entries are several KB and snapshots can reach GBs, so bound history. Pickled snapshot viewed locally with memory_viz. Only sees PyTorch allocator memory, not all NCCL/external allocations. | Join run/rank/device/step/phase and snapshot trigger; annotate module/graph where possible. Answers allocator fragmentation, allocation lifetime, OOM request/free state, and leak regressions. |
| [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/) (**closure**) | Triggered short system timeline for CUDA, NVTX, OS runtime, scheduling, NCCL and selected NIC/DCGM metrics; use capture ranges/rank sampling. | Nontrivial tracing/sampling volume and overhead; bound duration and traced ranks. DCGM profiling metrics may need pausing. Self-contained `.nsys-rep`, optionally SQLite/CSV/JSONL. | Use NVTX domains/ranges carrying stable phase/module/step identifiers; record rank/PID/host/device and clocks. Answers CPU scheduling, CUDA launch, stream overlap, collective, kernel, and host/device idle-gap questions on one timeline. |
| [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) (**closure**) | Deep dive on a filtered kernel/NVTX range after a timeline identifies it. | High and behavior-changing: counter limits require replay; memory save/restore, cache/clock control, launch serialization and software patching can alter execution. Never use as production throughput evidence. `.ncu-rep`, CSV/pages via CLI. | Record exact binary/source revision, kernel id/name, launch parameters, device, clocks/cache/replay settings, input shape/dtype, step/phase. Answers occupancy, scheduler, tensor-core, memory-pipeline and instruction-level bottlenecks. |
| TorchTitan deterministic loss/checkpoint comparison (**closure**, repo substrate) | `--debug.seed=42 --debug.deterministic`, seed checkpoints, `scripts/loss_compare.py`, and focused restore/fault-injection runs. | Deliberately slower and not production representative; generates TensorBoard/full-precision comparison files and checkpoint evidence. | Join model/config/data/checkpoint digests, mesh, precision, RNG/data position and exact work per step. Answers whether a non-computation/recovery change preserved promised loss and grad-norm identity. |

### Capture tiers

**Tier 0: every run.** Run manifest; structured JSONL; loss/grad norm/tokens;
per-phase step timing; checkpoint lineage; bounded Flight Recorder; low-rate DCGM
health/job counters; stdout/stderr and terminal outcome. This tier must have a
measured and enforced overhead/volume budget.

**Tier 1: scheduled samples.** Short PyTorch Profiler windows on selected ranks,
bounded memory history/snapshots, and richer peer-relative phase metrics. Use
the same scheduled steps in comparative 2/4/8-GPU validation runs.

**Tier 2: anomaly-triggered.** On stalled progress, repeated phase skew,
nonfinite values, MFU decline, or timeout precursor: dump Flight Recorder;
query NCCL RAS; capture repeated py-spy process-tree stacks; retain a bounded
tail of structured/DCGM evidence; optionally capture a short Nsight Systems
range if the process still progresses. Capture before abort where safe.

**Tier 3: stopped-job/deep dive.** Targeted NCCL tests, DCGM/EUD, SuperBench,
deterministic workload-shaped replay, Nsight Systems rerun, and filtered Nsight
Compute. These results must be attached to the original incident but marked as
new executions, never presented as direct recordings of the failed run.

## Current repository mapping and gaps

Repository inspection was read-only. Relevant current surfaces are:

- `torchtitan/observability/structured_logger/`: per-rank JSONL spans/scalars
  with host, pid, global/local rank, source, caller, step, wall timestamps, and
  local sequence; a Gantt/Chrome trace generator exists.
- `torchtitan/trainer.py`: spans for initialization, batch fetch, forward/backward,
  optimizer, distributed metric collection, step, checkpoint, and profiling;
  loss, grad norm, tokens, MFU, throughput, data loading, and allocator peaks
  flow through the metrics components.
- `torchtitan/tools/profiler.py`: scheduled CPU/CUDA Kineto traces and bounded
  Python-stack memory snapshots, including OOM snapshots.
- `torchtitan/distributed/utils.py` and `CommConfig`: Flight Recorder is on by
  default with per-rank timeout dumps, plus initialization/training process-group
  timeouts.
- `torchtitan/components/checkpoint.py`: DCP model/optimizer/scheduler/dataloader/
  train-state save/load, asynchronous staging options, and structured save/load
  spans.
- `docs/robust_training_reliability.md` and ADR 0001 already place semantic
  evidence and recovery validation in TorchTitan while leaving quarantine and
  replacement to the platform.

### Material gaps

1. **No canonical run/attempt manifest.** Structured files contain useful local
   identity but no run id, attempt id, source revision, normalized config digest,
   device UUID, allocation, or artifact index. Separate metrics, traces,
   snapshots, Flight Recorder dumps, configs, and checkpoints are not one
   immutable evidence graph.
2. **Rank identity is ambiguous for multi-actor work.** The structured logger
   itself has TODOs for mesh rank and duplicate sources. Role, actor instance,
   mesh axes, PG id, and globally unique rank/process identity are needed before
   core training and RL evidence can share tooling.
3. **Phase spans stop above modules/kernels.** Core phases are visible, but
   stable model-block/operator/NVTX correlation and compiled graph identity are
   not a common contract. Deep kernel capture cannot yet be joined cleanly back
   to model semantics.
4. **Distributed evidence is post-timeout, not a live joined view.** Flight
   Recorder is a strong primitive, but there is no repo-local analyzer that joins
   it with mesh/phase/host identity, repeated stack snapshots, NCCL RAS, or
   peer-relative timing. The present docs overstate that timeout dumps are
   automatically generated without documenting version/env and analysis
   artifacts precisely.
5. **No hardware/job evidence ingestion.** DCGM/NCCL logs and diagnostic results
   have no local adapter/schema. The repo correctly does not quarantine hosts,
   but it still needs a place to index platform-produced evidence.
6. **Metrics are dashboard-oriented and rank-reduced.** TensorBoard/W&B receive
   useful training series, while per-rank distributions and structured anomaly
   events needed for straggler/fault attribution are incomplete. W&B must remain
   an optional export rather than lineage storage.
7. **Checkpoint validity is storage-level.** Discovery checks DCP/HF metadata,
   but the documented progress envelope's manifest integrity, parent lineage,
   state-presence/digests, data position, restore equivalence, and failure-domain
   availability are not one enforced contract.
8. **Correctness detection is logging-coupled.** The trainer's nonfinite-loss
   crash occurs only on metric-logging steps (and has a TODO). There is no
   always-on finite grad/model sentinel or production SDC guarantee; deterministic
   loss comparison remains an offline proof tool.
9. **No incident lifecycle or ETTR derivation.** The repo records useful spans
   but not a typed detection -> diagnosis -> recovery -> validation incident,
   lost/recomputed work, recovery membership, or cumulative/windowed useful-work
   ratio.
10. **No capture policy/overhead contract.** Tools exist independently, but
    rank sampling, trace windows, trigger rules, retention, artifact budgets,
    tool conflicts, and perturbation labels are not normalized across the
    2/4/8-GPU ladder.

## Recommended research-foundation order

1. Define and test the versioned run-attempt manifest, correlation keys,
   artifact index, and typed incident/progress envelope. Make current structured
   logs, profiler traces, memory snapshots, Flight Recorder dumps, metrics, and
   checkpoints register into it without changing their data formats first.
2. Establish the 2/4/8-GPU Qwen3 0.6B/1.7B validation ladder with matched
   steps/tokens/configs and an observability perturbation suite. Report artifact
   volume and throughput impact for each tier before certifying it as default.
3. Add stable phase/module/collective correlation and offline local analyzers:
   per-rank phase skew, Flight Recorder/mesh join, incident timeline, checkpoint
   lineage graph, and cumulative/windowed ETTR.
4. Add launcher-owned adapters for DCGM/job evidence, py-spy, NCCL RAS/logs,
   and stopped-job diagnostics. Keep fleet action outside TorchTitan, but index
   every result in the run bundle.
5. Add fault-injection and deterministic recovery gates before researching a
   new recovery protocol: rank kill, pre-collective hang, trainer/data straggler,
   corrupted checkpoint/manifest, interrupted restore equivalence, and wrong
   per-rank config.
6. Only then promote Qwen3 8B and 30B-A3B MoE performance baselines. MoE adds
   expert/routing imbalance and much larger checkpoint state, so a platform
   validated only on dense small models is not certified for it.

This sequence produces useful evidence early without committing TorchTitan to
ByteRobust's private controller, FT-HSDP's particular recovery semantics, or a
remote observability service.
