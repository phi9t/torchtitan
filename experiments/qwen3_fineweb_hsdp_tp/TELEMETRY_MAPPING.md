# ByteRobust monitoring and telemetry machinery, mapped onto torchtitan

This is a companion to [`ARTICLE.md`](ARTICLE.md). That article traced the
HSDP+TP mechanics of the experiment in this folder. This one narrows the lens
to a single question drawn from arXiv:2509.16293v4 ("Robust LLM Training
Infrastructure at ByteDance", ByteRobust, SOSP '25): **what telemetry,
observability, and diagnostics machinery does ByteRobust depend on, and where
do the equivalent hooks live in torchtitan?**

The method is a bounded "transitive closure of references" pass: parse the
paper's bibliography, tag each entry by role, then follow the
telemetry-tagged subset (plus the adjacent fault-tolerance and checkpointing
work it leans on) down to the specific signal each contributes. Only that
subset is deep-dived. The result is then mapped, signal by signal, to a
concrete torchtitan `file:line` hook.

A framing carried over from the prior concept-level mapping: ByteRobust is a
**cluster-manager / platform** system, whereas torchtitan is a
**single-job training framework**. The two halves overlap only where a
platform-level signal has to be produced *inside* the training process
(NCCL flight records, process-group timeouts, NaN surfacing, per-step
metrics). Anything that reads GPU hardware counters, clusters stack traces
across thousands of machines, or reschedules pods is platform-half and has no
in-repo analog by design -- torchtitan's Core Principle 1 keeps that machinery
in upstream/platform layers, not in the training loop.

Line numbers are against the tree at the time of writing; if the source has
moved, search for the quoted symbol.

## Contents

1. [Scope and method](#1-scope-and-method)
2. [Full bibliography, classified](#2-full-bibliography-classified)
3. [The telemetry subset, deep-dived](#3-the-telemetry-subset-deep-dived)
4. [ByteRobust telemetry -> torchtitan mapping](#4-byterobust-telemetry---torchtitan-mapping)
5. [Gap summary](#5-gap-summary)

---

## 1. Scope and method

ByteRobust's own telemetry story is stated across four places in the paper:

- **Sec. 1 (Introduction)** names the failure taxonomy (CUDA error, NaN, job
  hang, MFU decline, SDC) and the signals prior systems use: log/exit-code
  analysis, timeouts, and MegaScale's plummeting-RDMA-traffic indicator. It
  states ByteRobust's philosophy of "lightweight real-time detection with
  hierarchical stop-time diagnostics" plus "data-driven clustering of runtime
  stack-traces" for isolation.
- **Sec. 2 (Background)** quantifies the taxonomy in Table 1 over 778,135 jobs
  (Job Hang 9.9%, MFU Decline 0.8%, NaN 0.3%, etc.) and describes the
  bitwise-consistent rollback-and-replay used to verify engineering changes.
- **Sec. 9 (Experiences and Limitations)** is the densest telemetry section:
  it names **EUD** (NVIDIA's diagnostic tool, ~70% SDC recall), the
  **MiniGPT verification suite**, and **dual-phase replay**, and recounts EUD
  itself causing an MFU regression by lifting a GPU frequency lock.
- **Sec. 10 (Related Work)** cites the named machinery: MegaScale (heartbeats +
  RDMA metrics), Hu et al. (LLM log agent), SuperBench (DL benchmarking to
  locate faulty GPUs), the SDC literature (Hochschild, Dixit), and the
  checkpointing lineage (Check-N-Run, CheckFreq, Gemini, ByteCheckpoint).

"Telemetry" here means any signal used to *detect* a fault, *diagnose* it, or
*verify* a fix -- as distinct from the recovery machinery (warm standby, hot
update, checkpoint backup) that acts once a fault is known. The classification
below tags the whole bibliography, but only the `Telemetry/Diagnostics` rows
and the `Fault-tolerance/Recovery` rows that produce a detection signal are
deep-dived in Section 3.

Every torchtitan `file:line` cited in Section 4 was re-read against the current
tree before writing.

---

## 2. Full bibliography, classified

Each reference is tagged with one primary role:

- `Telemetry/Diagnostics` -- produces or consumes a health/fault signal.
- `Fault-tolerance/Recovery` -- elastic scheduling, failover, resilience.
- `Checkpointing` -- state persistence and restore.
- `Parallelism/Systems` -- distributed training mechanics and infra.
- `Model/LLM` -- models, model reports, capabilities.
- `Other` -- datasets, benchmarks, storage/network background, tooling.

The paper carries ~90 numbered references (`bib.bib1`..`bib.bib99`). The
telemetry-relevant and structurally-important entries are listed explicitly;
the large `Model/LLM` and generic-`Other` tail is summarized in aggregate rows
to keep the table legible, since only the first three roles are deep-dived.

| Ref (author, year) | Work | Role |
| --- | --- | --- |
| Ben Frederickson 2019 | py-spy (sampling stack-trace profiler) | Telemetry/Diagnostics |
| NVIDIA 2024a | EUD GPU diagnostic tool | Telemetry/Diagnostics |
| Dong et al. 2024 | C4: communication-driven training efficiency | Telemetry/Diagnostics |
| Dong et al. 2025 | Evolution of Aegis: fault diagnosis for AI training | Telemetry/Diagnostics |
| Xiong et al. 2024 | SuperBench: DL benchmarking to locate faulty GPUs | Telemetry/Diagnostics |
| Hochschild et al. 2021 | Cores that don't count (SDC) | Telemetry/Diagnostics |
| Dixit et al. 2021 | Silent Data Corruptions at Scale | Telemetry/Diagnostics |
| Wang et al. 2023b | SDC characterization | Telemetry/Diagnostics |
| Hu et al. 2024 | LLM log agent + rule heuristics for stop-time diagnosis | Telemetry/Diagnostics |
| He et al. 2024 | Fail-stop diagnosis / recovery overhead study | Telemetry/Diagnostics |
| Huang et al. 2017 | Gray failure | Telemetry/Diagnostics |
| Gunawi et al. 2018; Lu et al. 2023; Zhang et al. 2024 | Fail-slow / gray-failure in storage | Telemetry/Diagnostics |
| Tan et al. 2019 | Gray failure in datacenter networks | Telemetry/Diagnostics |
| Ganatra et al. 2023 | Gray failure in cloud services | Telemetry/Diagnostics |
| Sima et al. 2022 | Ekko: fail-slow avoidance in parameter servers | Telemetry/Diagnostics |
| Jiang et al. 2024 | MegaScale (heartbeats + RDMA metrics + stop-time checks) | Fault-tolerance/Recovery |
| Kokolis et al. 2024 | Requeue / rescheduling on failure | Fault-tolerance/Recovery |
| Barham et al. 2022 | Pathways: async distributed dataflow (reschedule) | Fault-tolerance/Recovery |
| Thorpe et al. 2023 | Bamboo: redundant compute on spot instances | Fault-tolerance/Recovery |
| Jang et al. 2023 | Oobleck: pipeline-template resilience | Fault-tolerance/Recovery |
| Duan et al. 2024 | Parcae: proactive parallelism reconfiguration | Fault-tolerance/Recovery |
| Athlur et al. 2022 | Varuna: low-cost elastic training | Fault-tolerance/Recovery |
| Li et al. 2023; Gandhi et al. 2024; Wagenlander et al. 2024 | Elastic / resilient training | Fault-tolerance/Recovery |
| Xiao et al. 2018; Peng et al. 2018 | GPU cluster managers (Gandiva / Optimus) | Fault-tolerance/Recovery |
| Weng et al. 2022; Jeon et al. 2019; Gao et al. 2024; Zhang et al. 2020 | DL job / failure characterization studies | Fault-tolerance/Recovery |
| Mohan et al. 2021 | CheckFreq: pipelined checkpointing | Checkpointing |
| Eisenman et al. 2022 | Check-N-Run: differential + quantized checkpoints | Checkpointing |
| Wang et al. 2023a | Gemini: in-memory checkpoints with inter-machine backup | Checkpointing |
| Wan et al. 2025 | ByteCheckpoint: parallelism-agnostic checkpoints | Checkpointing |
| Zhong et al. 2024 | Async checkpointing | Checkpointing |
| Shoeybi et al. 2019 | Megatron-LM (TP; also baseline blocking save) | Parallelism/Systems |
| Narayanan et al. 2019, 2021 | PipeDream / 3D-parallel Megatron | Parallelism/Systems |
| Huang et al. 2019 | GPipe | Parallelism/Systems |
| Qi et al. 2024 | Zero-bubble pipeline | Parallelism/Systems |
| Jacobs et al. 2023; Liu et al. 2023 | Sequence parallelism / ring attention | Parallelism/Systems |
| Li et al. 2020; Zhao et al. 2023 | PyTorch DDP / FSDP | Parallelism/Systems |
| Rajbhandari et al. 2020 | ZeRO | Parallelism/Systems |
| Lepikhin et al. 2020; Shazeer et al. 2017 | GShard / MoE | Parallelism/Systems |
| Chen et al. 2016 | Gradient checkpointing | Parallelism/Systems |
| Ren et al. 2021 | CPU offloading (ZeRO-Offload) | Parallelism/Systems |
| Chang et al. 2024; Chen et al. 2024 | FLUX / Centauri comm-compute overlap | Parallelism/Systems |
| Ge et al. 2025 | Hybrid Data Parallelism (HDP) | Parallelism/Systems |
| Kingma 2014 | Adam optimizer | Parallelism/Systems |
| Foundation 2022 | HDFS | Other |
| Achiam 2023; Dubey 2024; Team 2024/2025; Brown 2020; Chowdhery 2023; Liu 2024; Yang 2025; Roziere 2023; ByteDance Seed 2025a/b; Guo 2025; Zhao 2025; Dosovitskiy 2021 | LLM / model reports and capabilities | Model/LLM |
| OpenAI 2022/2024; Anthropic 2023; Github 2022; Gallotta 2024; xAI 2024; BigScience 2022 | LLM products / deployments / chronicles | Model/LLM |
| Wei 2021; Wang 2022; Sheng 2024; Ma 2025 | Lifecycle management / training-system background | Other |

Roles present: `Telemetry/Diagnostics` and `Fault-tolerance/Recovery`
dominate the systems half of the bibliography, which matches a paper whose
contribution *is* the detect-diagnose-recover loop. The `Model/LLM` tail is
scene-setting (why the jobs are large and long-running) and is not deep-dived.

---

## 3. The telemetry subset, deep-dived

Each mechanism below lists the **signal** it produces, **how ByteRobust
consumes it** (real-time detection, stop-time diagnosis, or SDC/fix
verification), and the **reference** it traces to.

### 3.1 py-spy runtime stack sampling -> data-driven stack clustering

- Signal: per-process Python call stacks, sampled without instrumenting the
  target (py-spy, Ben Frederickson 2019).
- Consumption: when lightweight real-time detection and stop-time checks fall
  short, ByteRobust samples runtime stack traces across ranks and applies
  **data-driven clustering** to isolate suspect machines at a fault domain
  (a parallel group). The insight is that healthy ranks in the same collective
  cluster into a common stack, and stragglers/hung ranks fall into an outlier
  cluster -- so a hang that emits no error still produces a separable signal.
  This is real-time-adjacent detection feeding isolation, not root-causing.
- Reference: py-spy (Ben Frederickson 2019).

### 3.2 NVIDIA EUD / DCGM hardware diagnostics

- Signal: GPU hardware health from NVIDIA's EUD diagnostic tool (and the DCGM
  family of counters it builds on): ECC/memory errors, clock/thermal state,
  and SDC-relevant checks.
- Consumption: stop-time diagnosis and SDC screening. Sec. 9 reports EUD
  reaching only **~70% SDC recall** in their environment, and gives a
  cautionary case where EUD *itself* lifted a GPU frequency lock and caused an
  MFU regression -- telemetry as a fault source. This is the direct motivation
  for the MiniGPT suite (3.7) as a supplement.
- Reference: NVIDIA 2024a (EUD).

### 3.3 RDMA traffic monitoring + periodic heartbeats (MegaScale)

- Signal: RDMA network traffic volume per rank, plus periodic liveness
  heartbeats.
- Consumption: real-time detection. MegaScale (Jiang et al. 2024) uses
  **plummeting RDMA traffic** as an implicit-failure indicator (a rank that
  stops communicating is likely hung or slow) and heartbeats as a liveness
  tripwire. The paper is explicit that this signal detects but does not
  *isolate* -- MegaScale "still requires manual investigation" to pin the root
  cause, which is the gap ByteRobust's stack clustering closes.
- Reference: Jiang et al. 2024 (MegaScale).

### 3.4 C4: communication-driven telemetry

- Signal: collective-communication behavior as an efficiency and health
  signal (slow/errored collectives, imbalance).
- Consumption: detection of communication-path faults and slowdowns; C4
  frames the network as the observable that drives both efficiency tuning and
  fault localization in large parallel training.
- Reference: Dong et al. 2024 (C4).

### 3.5 Aegis: fault diagnosis service

- Signal: aggregated diagnostic signals across an AI training service in
  production (the diagnosis layer that consumes the above).
- Consumption: stop-time diagnosis at platform scope -- a productionized
  fault-diagnosis service, i.e. the kind of system ByteRobust is compared
  against and coexists with.
- Reference: Dong et al. 2025 (Evolution of Aegis).

### 3.6 SuperBench: stress-test benchmarking

- Signal: synthetic benchmark scores (GPU/network micro-benchmarks) used to
  flag underperforming machines.
- Consumption: stop-time diagnosis. SuperBench (Xiong et al. 2024) runs DL
  benchmarking to locate faulty GPU machines. ByteRobust uses selective stress
  testing as its *baseline* comparison (Table 6) and argues its runtime-signal
  approach is faster than benchmark-driven localization.
- Reference: Xiong et al. 2024 (SuperBench).

### 3.7 SDC literature + MiniGPT verification and dual-phase replay

- Signal: silent data corruption -- computations that are wrong without
  erroring, surfacing as NaN loss, gradient anomalies, or loss spikes.
- Consumption: SDC/fix verification. The SDC literature (Hochschild et al.
  2021 "Cores that don't count"; Dixit et al. 2021 "Silent Data Corruptions at
  Scale") establishes that at scale a fraction of cores silently miscompute.
  ByteRobust's own answer, since EUD only reaches ~70% recall, is the
  **MiniGPT verification suite**: deterministic workloads for intra-machine
  validation, plus **dual-phase replay** for inter-machine fault reproduction.
  The bitwise-consistent rollback-and-replay described in Sec. 2 (roll back a
  few steps, rerun, confirm the loss curve overlaps) is the same idea applied
  to verifying that an engineering change did not perturb numerics.
- References: Hochschild et al. 2021; Dixit et al. 2021; ByteRobust Sec. 2, 9.

### 3.8 Hu et al. log agent + gray-failure lineage

- Signal: stdout/stderr logs and exit codes.
- Consumption: stop-time diagnosis. Hu et al. 2024 layer an LLM-based log
  agent with rule heuristics over log data; the gray-failure lineage (Huang
  et al. 2017 and the storage/network/cloud follow-ons) frames the class of
  faults that emit degraded-but-not-dead signals. ByteRobust's stance is that
  log-only diagnosis misses implicit failures that runtime state would catch.
- References: Hu et al. 2024; Huang et al. 2017.

---

## 4. ByteRobust telemetry -> torchtitan mapping

Status legend:

- **Implemented** -- torchtitan produces this exact signal end to end.
- **Primitive present** -- torchtitan emits the raw signal but not the
  higher-level consumption (clustering, hardware polling, cross-job
  aggregation).
- **Platform-scope (no analog)** -- lives in the cluster manager by design;
  no in-repo hook, consistent with keeping non-training machinery upstream.

| ByteRobust signal / tool (Sec. 3) | torchtitan hook (`file:line`) | Status |
| --- | --- | --- |
| py-spy stack sampling -> stack clustering (3.1) | NCCL FlightRecorder dump-on-timeout, `torchtitan/distributed/utils.py:485-499`; buffer/dir knobs `trace_buf_size` / `save_traces_folder` at `torchtitan/config/configs.py:286-293` | Primitive present (dumps per-rank collective traces on timeout; no cross-rank clustering) |
| RDMA traffic + heartbeat hang tripwire (3.3) | Process-group timeouts `set_pg_timeouts`, `torchtitan/distributed/utils.py:529`; `init_timeout_seconds` / `train_timeout_seconds` at `torchtitan/config/configs.py:277-284`; torchft heartbeat/quorum in `torchtitan/experiments/torchft/manager.py` (lighthouse in `experiments/torchft/README.md:30-33`) | Primitive present (timeout trips on a hang; torchft adds heartbeat/quorum, but no RDMA-traffic metric) |
| C4 communication telemetry (3.4) | FlightRecorder collective trace buffer, `torchtitan/distributed/utils.py:492-499` | Primitive present (records collectives for post-hoc inspection; not a live comm-health monitor) |
| NaN / gradient-anomaly / SDC surfacing (3.2, 3.7) | `torch.autograd.set_detect_anomaly(...)` at `torchtitan/distributed/utils.py:215-223`; grad-norm computed and returned in `clip_grad_norm_` at `torchtitan/distributed/utils.py:560-638` and logged as `grad_norm` in `torchtitan/components/metrics.py:500` | Primitive present (anomaly stacks + grad_norm surfaced; no SDC recall guarantee, no hardware screening) |
| MFU-decline / throughput telemetry (3.3, Table 1) | Metrics pipeline in `torchtitan/components/metrics.py`: `mfu` / `tps` computed at `metrics.py:476-489`, logged with `loss`, `grad_norm`, and `memory/*` at `metrics.py:500-511` to TensorBoard/W&B | Implemented (per-step MFU/tps/loss/grad_norm/memory emitted; decline detection is left to the consumer) |
| Bitwise rollback-and-replay verification (2, 3.7) | `--debug.seed` + `--debug.deterministic` (see `torchtitan/distributed/utils.py:225` onward for seed handling) and `scripts/loss_compare.py` for loss/grad_norm comparison | Implemented (same-config runs are bitwise-reproducible and diffable, the framework half of dual-phase replay) |
| GPU memory pressure (Table 1: CPU/GPU OOM) | `DeviceMemoryMonitor` in `torchtitan/components/metrics.py:38-92`; peak/retry/OOM stats logged at `metrics.py:506-511` | Implemented (per-device memory + alloc-retry + OOM counters emitted) |
| EUD / DCGM hardware diagnostics (3.2) | -- | Platform-scope (no analog): GPU hardware health polling is outside the training process |
| SuperBench stress-test benchmarking (3.6) | -- | Platform-scope (no analog): whole-cluster stress testing is a scheduler task |
| Aegis fault-diagnosis service (3.5) | -- | Platform-scope (no analog): cross-job diagnosis service |
| Log agent / exit-code analysis (3.8) | -- | Platform-scope (no analog): torchtitan logs to stdout, but log aggregation/agenting is external |
| Data-driven stack clustering across machines (3.1) | -- | Platform-scope (no analog): FlightRecorder dumps the input, clustering is a platform job |
| MiniGPT deterministic verification suite (3.7) | -- | Platform-scope (no analog): a standalone verification workload, not a training-loop hook |

---

## 5. Gap summary

**Telemetry ByteRobust has that torchtitan does not.** All of it is
platform-half by construction:

- **Hardware health polling** (EUD/DCGM): ECC/clock/thermal/SDC screening.
  torchtitan reads no GPU hardware counters; it only sees the numerical
  fallout (NaN, grad-norm anomaly) after the fact.
- **Cross-rank stack-trace clustering** (py-spy driven): torchtitan's
  FlightRecorder produces the per-rank collective traces that such clustering
  would consume, but the clustering, fault-domain assignment, and over-eviction
  decision are all platform logic.
- **RDMA-traffic monitoring** (MegaScale-style): torchtitan detects a hang via
  process-group timeout, but has no continuous network-traffic metric to catch
  a slowdown before it becomes a timeout.
- **Dual-phase replay and the MiniGPT suite**: torchtitan has the *ingredients*
  for the framework half (bitwise-deterministic runs, `loss_compare.py`), but
  the orchestrated intra-/inter-machine replay campaign is external.
- **Cross-job diagnosis and stress testing** (Aegis, SuperBench): entirely
  scheduler/service scope.

**torchtitan primitives that could feed such a system.** Where ByteRobust
needs a signal *from inside the training process*, torchtitan already emits a
usable primitive:

- FlightRecorder dumps (`utils.py:485-499`) are exactly the runtime stack /
  collective-state input a clustering-based isolator wants.
- Process-group timeouts (`utils.py:529`) are the hang tripwire; torchft's
  `Manager` (`experiments/torchft/manager.py`) adds heartbeat + quorum on top,
  the closest in-repo analog to MegaScale's liveness layer.
- `set_detect_anomaly` (`utils.py:215-223`) plus the returned `grad_norm`
  (`metrics.py:500`) surface the NaN/gradient-anomaly signals that are the
  observable end of SDC.
- The metrics pipeline (`components/metrics.py`) already emits MFU, tps, loss,
  grad_norm, and memory per step -- the raw series an MFU-decline detector or a
  loss-spike watcher would subscribe to.
- Bitwise determinism (`--debug.seed` / `--debug.deterministic`) plus
  `scripts/loss_compare.py` give the numerically-exact replay that a
  verification suite is built around.

The consistent shape of the gap: torchtitan produces the **in-process
signals** (traces, timeouts, NaN, per-step metrics, reproducible loss) and
stops there; ByteRobust's contribution is the **platform layer** that
aggregates, clusters, screens hardware, and acts on those signals across
thousands of machines. That division is intentional -- torchtitan's charter
keeps cluster-management machinery out of the training loop -- so the
"missing" telemetry is missing on purpose, and the mapping is best read as
which torchtitan primitive a ByteRobust-style platform would tap for each
signal.
