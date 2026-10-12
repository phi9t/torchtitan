# LLM training reliability survey: per-paper deep dive

Companion to [`REFERENCE_CLOSURE.md`](REFERENCE_CLOSURE.md) (the depth-2
reference closure) and [`TELEMETRY_MAPPING.md`](TELEMETRY_MAPPING.md) (the
ByteRobust-telemetry-to-torchtitan mapping). Where the closure doc classifies
*what each paper cites*, this doc digs into *what each paper does*: the problem,
the core mechanisms, the headline numbers, the telemetry it collects, and the
limitations the authors themselves state.

Coverage: 13 of the 14 seed papers. Paper 7 (Intel, "Fine-grained Automated
Failure Management for Extreme-Scale GPU Accelerated Systems," SC '25,
DOI 10.1145/3712285.3759883) is skipped -- its ACM full text is paywalled, so a
grounded deep dive is not possible without fabricating detail.

Every figure below is drawn from the actual paper text (arXiv PDFs and USENIX
proceedings PDFs fetched and parsed locally); numbers are the authors' own.
Two arXiv-ID notes: the survey listed ARGUS as `2606.20374` and ResiHP as
`2605.*`; these future-dated 2026 IDs are what the resolver returns and are the
correct records (ARGUS `2606.20374v2` is the Tencent tracing paper; the similar
`2506.20374` is an unrelated astrophysics paper).

## Contents

1. [ByteRobust (ByteDance, SOSP '25)](#1-byterobust)
2. [MegaScale (ByteDance, NSDI '24)](#2-megascale)
3. [ARGUS (Tencent, APSys '26)](#3-argus)
4. [EROICA (Alibaba, NSDI '26)](#4-eroica)
5. [Mycroft (ByteDance, SOSP '25)](#5-mycroft)
6. [Revisiting Reliability (Meta, HPCA '25)](#6-revisiting-reliability)
8. [A Story of Two GPUs (UIUC/NCSA, SC '25)](#8-a-story-of-two-gpus)
9. [Minder (ByteDance, NSDI '25)](#9-minder)
10. [Aegis (Alibaba, NSDI '25)](#10-aegis)
11. [GREYHOUND (HKUST/Alibaba, ATC '25)](#11-greyhound)
12. [ResiHP (Fudan/Shanghai AI Lab, arXiv)](#12-resihp)
13. [R2CCL (Univ. of Maryland, arXiv)](#13-r2ccl)
14. [Llama 3 (Meta, 2024)](#14-llama-3)
15. [Cross-cutting synthesis](#15-cross-cutting-synthesis)

---

## 1. ByteRobust

**Robust LLM Training Infrastructure at ByteDance.** SOSP '25. arXiv:2509.16293.

**Thesis.** A GPU-infrastructure manager that prioritizes rapid, coarse-grained
fault *isolation* over precise root-cause localization, reaching **97% ETTR** on
a three-month, 9,600-GPU job.

**Problem.** At tens of thousands of GPUs, failures (CUDA errors, NaN, hangs) are
inevitable and conventional fail-stop -> log-analysis -> reschedule -> reload-
remote-checkpoint recovery costs hours to days. The hard cases are *implicit*
failures (hangs, MFU decline, SDC) that emit no clear signal, too few spares to
swap at scale, and continuous human/code evolution over multi-month runs.

**Key mechanisms.**
- **Automated fault-tolerance state machine** (Fig. 5): second-level real-time
  checks (NIC down, DCGM status, PCIe bandwidth, memory row remapping, GPU
  temperature, Xid) -> hierarchical stop-time checks (EUD, intra-machine
  all-to-all, inter-machine all-gather) -> in-place reattempt -> code rollback
  -> dual-phase replay.
- **Dual-Phase Replay** (Algorithm 1): dimension-aware group testing for hard
  faults such as SDC. Holds TP/PP fixed, varies only DP; partitions machines
  horizontally (by `x_i / m`) then vertically (by `x_i mod n`), replays each,
  and intersects the two faulty groups. With `m = k*PP_size`, `n = DP_size/k`
  and `m <= n`, the suspect set has cardinality 1.
- **Data-driven over-eviction**: for silent hangs/MFU decline, an on-demand
  tracer (py-spy + flight-recorder) captures per-process stack traces; a
  three-step pipeline parses process trees, clusters stacks by string match
  (dominant cluster = healthy, minority = outliers), maps outliers to a shared
  parallel group, and over-evicts that whole group. Fail-slow repeats the
  aggregation every 10 s and flags the group with the highest cumulative flag
  count over 5 rounds.
- **In-place lazy hot-update**: code/data changes applied without tearing down
  pods; non-critical updates deferred and merged into the next inevitable
  recovery, or forced after 24 h.
- **Warm standby** sized to the P99 of a binomial failure distribution (e.g. 4
  backups per 1,024 instances), and **over-eviction-aware checkpointing**:
  every-step in-memory checkpoint with dual CPU buffers, async D2H on a
  dedicated stream, and cross-parallel-group backup.

**Headline numbers.** 97% cumulative ETTR on 9,600 GPUs for 3 months, unproductive
time <= 50 min; MFU up 1.25x (dense) / 1.58x (MoE) via hot-updated code; on
16,384 GPUs warm standby up to 10.87x and hot update up to 11.04x faster than
requeue, within 5.19% of the oracle; every-step checkpoint <0.9% overhead,
cutting blocking time 99.69% vs Megatron save. Across 19 large jobs, real-time
eviction handled 32.52%, reattempt 22.70%, rollback 9.20%, and only 1.23%
needed dual-phase replay. Census of 778,135 jobs: CUDA error 36.1%, job hang
9.9%, code/data adjustment 17.3%, NaN 0.3%. NVIDIA EUD reaches only **70% SDC
recall** in their environment.

**Telemetry.** wandb loss/grad-norm/MFU (5x jump or NaN = faulty); stdout/stderr
+ exit codes; CUDA/RDMA/host/storage events (zero RDMA traffic in 10 min = hang);
second-level system inspections; on-demand full process-tree stack traces.

**Stated limits.** GPU diagnostic tools lag hardware (EUD once caused GPU
downclocking); intentional PP-group over-eviction yields 6-7 false positives per
job; SDC remains hard, non-deterministic, with overhead growing at scale.

**Systems introduced.** ByteRobust (Robust Controller ~20k LoC Go, Robust Agent
~5k LoC Python, Runtime Analyzer ~12k LoC Go, CKPT manager ~3k LoC Python);
Dual-Phase Replay; MiniGPT verification suite.

---

## 2. MegaScale

**Scaling LLM Training to More Than 10,000 GPUs.** ByteDance, NSDI '24.
arXiv:2402.15627.

**Thesis.** Two principles -- algorithm-system co-design and in-depth
observability -- yield **55.2% MFU on 175B over 12,288 GPUs, 1.34x over
Megatron-LM**.

**Problem.** At >10,000-GPU single-job scale: high MFU despite communication,
operator, data-pipeline and memory overhead; and stability over weeks where
failures and stragglers are the norm and one straggler slows the whole job.

**Key mechanisms.**
- **Algorithmic co-design**: parallel transformer block (`y = x + MLP(LN(x)) +
  Attn(LN(x))`); sliding-window attention (O(s*w) vs O(s*s)); LAMB scaling batch
  4x, cutting interleaved-pipeline bubbles 87.5% (`(4vp-1)/m` -> `(vp-1)/4m`).
- **3D-parallel communication overlap**: DP all-gather prefetched at iteration
  start with priority-ordered launches; PP send/recv decoupled across warm-up/
  steady/cool-down; TP/SP all-gather and reduce-scatter fused into FFN Linears
  with chunked GEMM pipelined against communication.
- **Efficient operators / data pipeline**: FlashAttention-2, fused LayerNorm and
  GeLU; two-layer tree data loader removing redundant per-worker loads.
- **Fast comm-group init**: replaces PyTorch single-threaded TCPStore with async
  Redis and lowers the global barrier from O(n^2) to O(n) -- 2,048-GPU init
  1047s -> 361s -> under 5s; under 30s at >10,000 GPUs.
- **Network tuning**: custom three-tier fat-tree fabric on Tomahawk 4, ECMP-conflict reduction,
  Swift RTT + DCQCN/ECN, tuned NCCL retransmit + NIC `adap_retrans`.
- **Robust framework + observability**: driver/executor/Kubernetes with periodic
  heartbeats (IP, Pod, hardware, process status, logs, RDMA metrics); lightweight
  self-checks; two-stage checkpoint; CUDA-event-timer with heat-map + 3D timeline.

**Headline numbers.** 55.2% MFU on 175B/12,288 GPUs (1.34x Megatron-LM; +14% at
largest scale); 256-GPU ablation 47.7% -> 65.3% MFU (+17.6%); 530B weak-scaling up
to +6.1% MFU over Megatron-LM; >10,000-GPU production run for weeks, repaired
>100 times, >90% faults auto-detected, diagnostics <10 min, catch-up within
15 min, >90% effective training time.

**Telemetry.** Heartbeats (executor state, filtered stdout/stderr, RDMA metrics
-- a plummet flags implicit anomalies, cessation auto-triggers recovery);
millisecond-precision monitoring (second-level for ECN/PFC/QoS/link-flap, ms for
congestion and DP/PP transfer limits); CUDA-event traces via file -> Kafka ->
analytical DB for heat-map / 3D-view.

**Stated limits.** Probabilistic hardware anomalies escape self-checks;
stragglers from irregular GC and some critical-path PyTorch ops; MFU decline from
reduce-scatter time skew growing with steps; link flapping needs larger NCCL
timeouts; proactive fault tolerance assumes predictable failures.

**Systems introduced.** MegaScale (on Megatron-LM 285068c8); robust training
framework; CUDA-event-timer + 3D visualization; open-sourced under veScale.

---

## 3. ARGUS

**Production-Scale Tracing and Performance Diagnosis for over 10,000-GPU
Clusters.** Tencent, APSys '26. arXiv:2606.20374.

**Thesis.** A low-overhead (<2%), always-on, fine-grained tracer that decomposes
observability into three signals, compresses raw kernel traces **~3,700x** via
online KDE clustering, and runs a **5-level** progressive fail-slow diagnosis
narrowing tens of thousands of ranks to single-digit suspects.

**Problem.** Fail-slow faults in synchronous training drag the whole job without
errors (motivating case: a 4,096-GPU job wasting ~23,758 GPU-hours). Existing
tools are either always-on but coarse ("cannot answer which kernel slowed down")
or fine-grained profilers that are phase-level, trigger-only, or 5-30%+ overhead.
Trace volume is the wall: 10^4-10^5 kernel events/min/GPU = 6-60 GB/min cluster.

**Key mechanisms.**
- **Three trace types along the call hierarchy**: (1) streaming **py-spy** CPU
  call stacks (no hooks, reads process memory); (2) framework semantics via
  **CUDA Events** on the correct stream (forward/backward/optimizer/comm);
  (3) GPU kernel activity via **CUPTI Activity API** injected through
  `CUDA_INJECTION64_PATH` (`libcupti_injector.so`).
- **CUPTI overhead control**: decoupled control/collection/async-export paths;
  selective process injection (skips compilation/launcher/child processes);
  pre-allocated fixed-size buffer pool off the hot path; bounded queues with
  explicit backpressure drop.
- **Compression ~3,700x**: log-transform kernel durations, build a Gaussian-KDE
  density (Scott's rule bandwidth `h = 1.06*sigma*n^(-1/5)`), split at local
  minima, and store each `(kernel, stream, rank)` window as a few
  `(count, p50, p99)` triples -- 10 MB -> 2.7 KB per rank per step.
- **5-level diagnosis** (L1-L3 auto and parallel, L4/L5 on-demand): L1
  iteration-time anomaly (sliding-window ratio gate + full-scan change-point);
  L2 phase-level straggler attribution (coefficient of variation + z-score,
  parallelism-group-aware routing); L3 kernel-distribution comparison
  (reconstruct per-rank CDFs from the triples as a log-normal mixture, compare
  by **Wasserstein-1 / Earth Mover's Distance**, flag by IQR fence); L4 Perfetto
  critical-path; L5 CPU-stack host-stall localization.

**Headline numbers.** Total overhead <2% (CUPTI ~1-2%) vs PyTorch Profiler
20-44% (then OOM) and nsys failing always-on; memory +2 GB (8-GPU) / +10 GB
(32-GPU), constant over time; per-rank-per-step data 10.6 MB raw -> 18.7 KB
summaries vs nsys 63.56 MB / Profiler 48.76 MB; ~2.7 GB/min upload at 10,000
GPUs; deployed 10,000+ GPUs for >6 months. Case magnitudes: self_attention
2,252 ms vs 6-16 ms normal (>150x); intra-group W1 17-23k vs inter-group up to
2.73M.

**Telemetry.** Per-kernel name/launch/duration/stream; framework phase durations;
streaming Python stacks; per-rank iteration series; compressed (count, p50, p99)
kernel stats. Fault taxonomy spans compute-HW, comm, host-side, framework/config.

**Stated limits.** L4/L5 still need engineer expertise (exploring LLM-agent
automation); compression is lossy; backpressure can drop data; masking effects
(natural VLM variation, grad-sync alignment) can defeat L1-L3; overhead measured
at 8/32 GPUs while 10,000-GPU claims come from deployment; inference serving is
future work.

**Systems introduced.** ARGUS; FT-Client UI; NDTimeline-style semantics package;
built on py-spy, CUPTI, CUDA Events, Vector, Perfetto, Grafana, Prometheus.

---

## 4. EROICA

**Online Performance Troubleshooting for Large-scale Model Training.** Alibaba
Cloud + UIUC, NSDI '26. arXiv:2506.08528.

**Thesis.** Profile on-demand and summarize each function's runtime behavior into
a tiny 3-value pattern, giving profiler-grade detail at cluster-wide coverage and
localizing root causes with minimal production impact.

**Problem.** Performance (not crash) issues where online monitors are coarse
(root-cause only ~29.6%) and offline profilers produce ~100 MB/worker/sec
(~1 TB/s at 10,000 GPUs), untenable online and unusable without reproduction
(7.4% never reproduce). 9-month, ~100,000-GPU issue mix: 44.4% hardware, 48.2%
software, 7.4% unknown.

**Key mechanisms.**
- **Runtime behavior patterns**: each function `f` on worker `w` becomes
  `P_{f,w} = (beta, mu, sigma)` -- beta = fraction of profiling window on the
  critical path, mu = mean hardware-resource utilization, sigma = its std. All
  timestamp-independent, so cross-host comparison needs **no clock sync** (NTP's
  ~10 ms error would break event-timestamp comparison).
- **Critical-path extraction** with priority (GPU compute > memory > collective
  > Python); Algorithm 1 binary-searches the longest subinterval capturing
  >= 0.8x total utilization with <= g consecutive zero samples.
- **On-demand trigger**: wraps `dataloader.next()`/`optimizer.step()` at import;
  degradation when the recent N=50 iteration mean exceeds recent-shortest by >5%,
  or elapsed >= 5x average (blockage). Globally synchronized 20 s window via
  rank-0 iteration-ID broadcast over TCP.
- **Localization**: distance-from-expectation `D_{f,w}` (min Manhattan distance
  to an expected range, e.g. Python beta in [0,0.01]) and differential distance
  `Delta_{f,w}` (how many of N sampled workers differ by >= delta=0.4); abnormal
  iff `beta>0.01 and (D>0 or Delta > M_f + k*MAD_f)`, k=5. DBSCAN/HDBSCAN/GMM/
  mean-shift were rejected (noise separation, too many hyperparameters).
- **Engineering**: direct Kineto dump (-33% data-gen time), `cuptiFinalize()` to
  clear residual hooks, Kubernetes `emptyDir` + privileged container for 10 kHz
  hardware sampling.

**Headline numbers.** 1.5 years on ~100,000 GPUs; diagnosed 78/80 hard issues
others missed = **97.5%**, throughput +20-100% (largest job 6,144 GPUs); a
3,400-GPU job in 3 min, a 1,000,000-GPU job end-to-end in 7 min; pattern data
30 KB/worker vs ~3 GB raw (10^5x smaller). Case: 3,400 H800 iteration 10.5s ->
8.5s, +34% throughput.

**Telemetry.** Torch Profiler events + nsys hardware sampling at **10 kHz** (GPU
freq/SM, DRAM, NVLink, PCIe, GPU-NIC); ~3 GB raw per worker per session,
compressed to 30 KB patterns.

**Stated limits.** Cannot auto-root-cause all issues (complex user Python still
needs inspection); cannot see problems outside the task; a workload fully
overlapping comm and compute could hide stalls; 2/80 failed; AI auto-fix
(Cursor/Sonnet 4.5) correct only sometimes.

**Systems introduced.** EROICA (~7k LoC Python); integrates Cursor for AI
auto-fix; compares against DCGM, Dynolog, MegaScale, NCCL Profiler, Nsight,
Torch Profiler.

---

## 5. Mycroft

**Tracing Dependencies in Collective Communication Towards Reliable LLM
Training.** ByteDance + CUHK + Harvard, SOSP '25. arXiv:2509.03018.

**Thesis.** A lightweight distributed tracer that opens the CCL black box by
tracing collective-communication (Coll-level) internal states and exploiting
control/data dependencies to find root causes.

**Problem.** CCLs (NCCL) are black boxes; on silent timeouts, fail-slows, or
degradation, operators cannot see internal state. Op-/kernel-/RDMA-level tools
lack fine-grained, distributed, real-time observability. Motivation: Meta
OPT-175B saw >105 restarts, ~1.25 incidents/day, 61,000 GPU-hours impacted.

**Key mechanisms.**
- **Coll-level tracing** at Flow-level (per QP/network flow, since NCCL channels
  are flow-granular) and Chunk-level (per-path chunk progress, chunks ~a few MB,
  by aggregating state not per-chunk logging).
- **Instruments NCCL proxy threads** with two tracepoints: a completion log (on
  CollOp finish: timestamps, bytes, NIC info, metadata) and a periodic (~100 ms)
  real-time state log reporting accumulated progress across devices.
- **G/T/D three-stage progress** per chunk: **GPU_ready (G)**,
  **RDMA_transmitted (T)**, **RDMA_done (D)** -- distinguishing "GPU has not
  produced data" vs "not submitted to RDMA" vs "submitted but CQE incomplete."
- **Data collection**: fixed-size circular buffer in host shared memory (512 MB/
  host), written on NCCL internal streams (near-zero overhead); a read-only agent
  uploads via Kafka.
- **Real-time trigger** (Algorithm 1): samples >= 1 rank per DP group, capped at
  10. Failure: a rank has a real-time log but no completion log in a window.
  Straggler: throughput halves or CollOp interval doubles (straggler threshold
  1 s late).
- **Dependency-driven RCA** (Algorithm 2): builds a global distributed state
  machine, then AcquireGroupLastLog -> CheckMinOp -> CheckRCTable (or
  CheckMinData) to categorize the fault.

**Headline numbers.** Detects anomalies within 15 s in 90% of cases; root cause
within 20 s in 60%; deployed >6 months at ByteDance; testbed 32 A100, scales to
tens of thousands of GPUs; NCCL 2.21.5, 1,100 LoC C++, <10 tracepoints; 512 MB/
host fixed trace. Context: anomalies propagate cluster-wide in a few hundred ms;
NPKit caused a two-thirds bus-bandwidth drop; XPUTimer traces exceed 100 MB/
iteration/GPU.

**Telemetry.** NCCL proxy completion + periodic real-time state logs; metadata
(IP, comm_id, Gid, GPU_id, Channel_id, QP_id); operation (timestamps, op_name,
op_seq, msg_size); chunk (stuck_time, total_chunks, GPU_ready, RDMA_transmitted,
RDMA_done).

**Stated limits.** Observability confined to collective communication (no
end-to-end training view); cannot capture application intent (intentional
overlap/parallelism), so distinguishing anomaly from benign variation is hard;
deep hardware/software root causes still need other tools; direct NCCL-source
tracepoints may not port to newer NCCL/CCLs; manual thresholds.

**Systems introduced.** Mycroft; integrates py-spy and PyTorch Flight Recorder;
compared against Kineto, Chakra, Mystique, GREYHOUND, Nsight, NPKit, NVRx,
XPUTimer, Aegis.

---

## 6. Revisiting Reliability

**Revisiting Reliability in Large-Scale Machine Learning Research Clusters.**
Meta FAIR, HPCA '25. arXiv:2410.21680.

**Thesis.** An 11-month, 4-million-job, 150M+ A100-GPU-hour study of two Meta
research clusters showing large jobs suffer most from failures but small jobs
dominate by count; contributes a failure taxonomy, MTTF projections, and an
analytical ETTR estimator.

**Problem.** Understanding and mitigating job-failure impact across scales in
general-purpose (not LLM-only) multi-tenant clusters, and quantifying reliability
needed to reach ~12,000-GPU scale.

**Key mechanisms.**
- **Failure taxonomy** via differential diagnosis over failure domains (user code
  / system software / hardware infra).
- **Health-check-based reliability** every 5 min; high-severity checks (GPU
  inaccessible, NVLink error, uncorrectable ECC, failed row-remaps, PCI/IB link
  errors) immediately remove and reschedule; calibrated so <1% of successful jobs
  see a failed check.
- **Analytical E[ETTR] estimator** as a function of `N_nodes`, failure rate
  `r_f`, checkpoint interval, write time, restart overhead, queue time;
  MTTF = `(N_nodes*r_f)^-1`; Daly-Young optimal interval
  `Delta_cp* = sqrt(2*w_cp/(N_nodes*r_f))`; accurate to ~5% vs Monte Carlo.
- **Lemon-node detection**: 7 signals (excl_jobid_count, xid_cnt, tickets,
  out_count, multi/single_node_node_fails, single_node_node_failure_rate)
  thresholded from a 28-day CDF.
- **Adaptive Routing (AR)** and SHIELD for fabric resilience.

**Headline numbers.** RSC-1: 16k GPUs, `r_f`=6.50 failures/1000 node-days; RSC-2:
8k GPUs, `r_f`=2.34. **MTTF: 8-GPU job = 47.7 days; 1024-GPU = 7.9 hours;**
projected 16,384-GPU = 1.8 h, 131,072-GPU = 0.23 h. Infra failures hit 0.2% of
jobs but 18.7% of GPU runtime. >90% of jobs use <1 server but <10% of GPU time.
Lemon detection: 40 faulty nodes, >85% accuracy, cutting 512+-GPU job failures
from **14% to 4%** and improving large-job completion >30%. Largest jobs reach
ETTR ~0.85-0.9; to hit 0.9 at 12,000 GPUs, `r_f` must drop 6.50 -> ~1 or
checkpoint write to ~10 s. Without AR resilience, 50-75% bandwidth loss in
bring-up. One 1024-GPU job NODE_FAILed 35 times -> 548 preemptions.

**Telemetry.** Slurm statuses, node health checks (XID, mounts, services), GPU
swap rates, IB link errors, per-GPU-hour attributed failure rates, lemon-node
signals. Does *not* focus on SDC.

**Stated limits.** Correlation not causation from health checks; cannot track
productive-runtime components at scale (treats restart/write overhead as free
params, assumes 5 min each); ETTR is a conservative underestimate; assumes
classical (not async) checkpoint; SPMD/BSP focus; no SDC analysis.

**Systems introduced.** RSC-1/RSC-2; lemon-node pipeline; uses Slurm, submitit,
SHIELD, Adaptive Routing, PyTorch Flight Recorder.

---

## 8. A Story of Two GPUs

**Characterizing the Resilience of Hopper H100 and Ampere A100 GPUs.**
UIUC/NCSA + IBM + Nokia Bell Labs, SC '25. arXiv:2503.11901.

**Thesis.** A 2.5-year, 11.7M-GPU-hour field study of NCSA Delta finds H100 GPU
*memory* resilience markedly worse than A100 (**3.2x lower per-GPU MTBE** for
uncorrectable ECC) despite better overall hardware resilience, and GPU errors
almost always kill jobs because application recovery is ineffective.

**Problem.** First field characterization comparing A100 (Ampere) vs H100
(Hopper, in GH200) GPU error/failure resilience and job impact, filling gaps in
prior studies limited to older or cluster-level data.

**Key mechanisms.** Delta = 1,056 GPUs (448 A100 40GB + 608 H100 96GB in 152
4-way GH200 nodes). Data pipeline: RegEx over NVIDIA XID syslog messages + Slurm
DB + DCGM (1-min). **Error Coalescing** (Algorithm 1) collapses identical
same-GPU errors within Delta_t=5 s. **Error Propagation** computes
`P(e2|e1) = #e2/#e1` for e2 within 5 s (intra-GPU or same-node inter-GPU).
Metrics: system/node/per-GPU/per-GB MTBE (capacity-normalized). Availability +
discrete-event simulation for overprovisioning.

**Headline numbers.** H100 memory 3.2x lower per-GPU MTBE for uncorrectable ECC
(88,768 h vs A100 283,271 h); per-GB MTBE ~8.5M h (HBM3) vs ~11.3M h (HBM2e) =
24% lower. Row-remapping mitigates 92% of H100 uncorrectable ECC but spare rows
capped at 512 (not scaled with capacity) -- 8 row-remap failures on H100 vs 0 on
A100. A100 HW errors: 8,863 MMU, 3,857 GSP, 1,922 NVLink, 77 PMU SPI; H100: 3
GSP, 0 NVLink, 0 PMU SPI (GH200 CPU-GPU integration). System MTBE A100 1.4 h /
H100 1.9 h; availability ~99.4% / 99.3% ("two nines"); one DBE example took 19 h
recovery. Overprovisioning: a 608-GPU 1-month job at 2.2 h recovery needs 5% (31)
extra GPUs for 99.9% availability (>$1M/month); 5-min recovery drops it to 2%.

**Telemetry.** XID logs (syslog RegEx), Slurm DB, DCGM (1-min). Utilization A100
51% / H100 41%; temps 40C / 37C.

**Stated limits.** Studies errors not faults; logging inconsistencies (0.18-0.49%
node lockups); no job-script access (name/module heuristics for ML/non-ML); H100
only in GH200 form; H100 window (146 days) much shorter than A100 (895 days).

**Systems introduced.** Delta (study subject); open-sourced coalescing +
propagation + impact pipeline and simulator (Zenodo 15287639).

---

## 9. Minder

**Faulty Machine Detection for Large-scale Distributed Model Training.**
ByteDance + Tsinghua + Northeastern + Harvard, NSDI '25. arXiv:2411.01791.

**Thesis.** Detect the faulty machine at runtime by exploiting inter-machine
metric *similarity* and fault *continuity*, with per-metric LSTM-VAE denoising
and Z-score/decision-tree metric prioritization -- reacting in **3.6 s** with
**0.904 precision**.

**Problem.** Manual faulty-machine diagnosis is slow (>30 min avg, up to days).
Faults average ~2/day per task; one machine fault cascades (NCCL timeout) to halt
thousands. Loss e.g. $650-$1700 per 40-min incident on a 128-machine V100 task.

**Key mechanisms.**
- **Machine-level similarity**: 3D parallelism balances load, so machines show
  similar per-second metric fluctuations; the machine with the largest summed
  distance from others is the suspect (unsupervised, fault-type-agnostic --
  supervised learning rejected because "normal" is task-dependent).
- **Continuity**: true faults persist (most >5 min); a candidate must recur
  across consecutive windows past a **4-minute** threshold to filter jitter.
- **Per-metric LSTM-VAE denoising**: one VAE per metric (hidden 4, latent 8,
  1 LSTM layer, window 8), reconstruction MSE <0.0001; metrics kept separate to
  avoid mutual interference.
- **Metric prioritization**: per-machine Z-score per metric, max across machines,
  fed to a decision tree ranking sensitivity (top: PFC, CPU usage, GPU duty/
  power/tensor activity, NVLink bandwidth).
- **Online detection**: denoise -> pairwise Euclidean distance of embeddings ->
  sum -> normalize -> threshold -> continuity check -> next metric.

**Headline numbers.** 3.6 s average alert (>99% / 500x faster than manual);
precision 0.904 / recall 0.883 / F1 0.893 vs Mahalanobis 0.788/0.767/0.777;
deployed >1 year, 150 fault instances over 9 months, 4 to >1500 machines (up to
10,000 Ampere GPUs). Fault mix: ECC 25.7%, CUDA exec 15%, GPU exec 10%, PCIe
downgrading 8.6%; individual-machine faults 99% of cases.

**Telemetry.** Out-of-band per-second host counters: compute (CPU usage, GPU duty
cycle, power, graphics/tensor engine activity), communication (PFC Tx rate,
throughput, NVLink bandwidth, RDMA), storage (memory, disk/HDFS). Runs on 1
dedicated machine, called every 8 min over 15-min windows.

**Stated limits.** Second-level granularity misses fast-propagating faults
(concurrent switch-reboot undetectable, needs ms-level not widely deployed); AOC
errors partly missed (no optical counters); machine-level only (in-machine root
cause needs human analysis); labels imperfect.

**Systems introduced.** Minder; compared to Mahalanobis baseline; complements
SuperBench, DCGM, EUD, R-Pingmesh, HostPing, Collie.

---

## 10. Aegis

**Evolution of Aegis: Fault Diagnosis for AI Model Training Service in
Production.** Alibaba Cloud, NSDI '25.

**Thesis.** A production fault-diagnosis system for a *public* training cloud that
localizes failures and degradation at runtime without modifying customer code,
primarily by customizing the CCL.

**Problem.** A single-point fault cascades into a whole-task crash, hiding the
culprit among all hosts' error reports. SuperBench is offline-only (init tens of
minutes); MegaScale needs instrumenting customer model code, which public
customers reject on confidentiality grounds.

**Key mechanisms.**
- **Phase-1 basic error diagnosis**: combines training/system/NIC/switch logs;
  `CriticalError()` isolates hard faults (double-bit ECC XID 48, XID 94/95, link
  down); `DistError()` handles "connection reset by peer" -- if only 2 hosts show
  it, isolate directly; `RootDiag()` clusters by common source/destination GPU.
- **Parallelized topology-aware offline diagnosis**: per-host self-checks in
  parallel, then multi-host reference training, split by Pod/ToR-group so tasks
  do not contend on shared links.
- **Phase-2 procedure-aware runtime diagnosis (core)**: customizes CCL (a
  swappable plugin, no customer code change) to record per-operator/per-GPU
  **CL (collective launch count), WR (work-request count), WC (work-completion
  count)**. Compute failure: faulty GPU has `CL < CL of group`; comm failure:
  `WR != WC` flags the culprit, then NetDiag on src/dst.
- **Performance-degradation diagnosis**: Z-score outlier over 20+ metrics
  sustained over 10 min; plus CCL per-op duration TD (compute degradation if
  `TD < 0.8*TC`) and throughput N (comm degradation if `N > 1.5*mean`).
- **Check Before Delivery (CBD)**: pre-delivery parallel checklist, full <10 min,
  light <1 min.

**Headline numbers.** >97% reduction in diagnosis idle time; 84.6% fewer restarts;
71% less performance degradation; Phase-1 cut idle 71%, Phase-2 a further 91%;
runtime diagnosis ratio 77% -> ~100%; CBD intercepts 1-2% problematic hosts.
Scale grew >40x over 16 months; 100-230 critical failures/week; 73% of tasks fail
in first 10 min. Failure mix: 45.6% GPU-related; A100 MTTF ~400 days, H100 ~200
days; 71% of distributed failures turned out unrelated to the network.

**Telemetry.** Training logs, dmesg, CCL logs, NIC driver logs, switch syslog,
RDMA Pingmesh, in-band per-hop coloring, and custom-CCL CL/WR/WC/TD/N.

**Stated limits.** CCL info localizes but does not root-cause (offline, out of
scope); must maintain a custom CCL per released CCL version; offline diagnosis
wastes GPU time; parallel offline scheme can miss Tier-2/3 switch faults (a real
case of silent >1KB-packet loss was invisible to 64B probes).

**Systems introduced.** Aegis (Phase-1/2), CBD; contrasted with SuperBench and
MegaScale.

---

## 11. GREYHOUND

**Hunting Fail-Slows in Hybrid-Parallel Training at Scale.** HKUST + Alibaba,
USENIX ATC '25.

**Thesis.** A framework-agnostic system detecting compute and communication
fail-slows at runtime with **>99% accuracy** and mitigating them via a
ski-rental-inspired multi-level mechanism, without human intervention.

**Problem.** Fail-slows (CPU contention, thermal throttling, congestion) slow
synchronous training but are hard to detect: per-iteration sync drops all GPUs'
utilization at once, and shared links make telemetry ambiguous. Treating them as
fail-stops via checkpoint-restart is overkill (GPT2-100B dump ~100 min > mean
fail-slow duration).

**Key mechanisms.**
- **DETECT, three phases, master-worker**: non-intrusive `LD_PRELOAD` shim
  intercepting only top-level NCCL/CUDA interfaces (works with NCCL/ACCL/MSCCL),
  inter/intra-node via Redis and shared memory.
- **Tracking**: autocorrelation function finds the iteration period (first lag
  with ACF >= 0.95); **Bayesian Online Change-point Detection (BOCD)** flags slow
  iterations (>10% change), plus a verification step (BOCD+V) discarding <10% as
  jitter.
- **Profiling**: CUDA events time each comm group; cross-group comparison flags a
  group exceeding the median by >10%.
- **Validation**: lightweight suspension by trapping hooked NCCL calls in a wait
  loop (no checkpoint/restart, reuses CUDA context); compute validation runs GEMM
  monopolizing SMs; comm validation decomposes ring/tree into O(1) non-overlapping
  P2P passes (2 for even-rank ring) instead of O(N^2).
- **MITIGATE, ski-rental multi-level**: S1 Ignore, S2 adjust micro-batch
  distribution (quadratic program via cvxpy), S3 adjust parallelism topology
  (reassign congested links from heavy DP to light PP groups), S4 checkpoint-
  restart; escalates when accumulated slowdown equals the next strategy's cost.

**Headline numbers.** Detection 99.8% (498/499); BOCD+V 100% accuracy / 0%
false-positive rate / 0% false-negative rate (compute), 99.1% / 0% / 2.3%
(comm); mitigation reduces compute-fail-slow
slowdown up to 1.59x, comm up to 1.23x; end-to-end +1.58x on 256 H800. Tracking
overhead avg 0.39% (max 1.1%); BOCD detects in 2-3 iterations (<5 s).
Characterization: 10,000+ GPUs, 4,000 nodes; compute fail-slows <2% frequency
(~10 min), comm fail-slows 40% frequency (~24 min); at >= 512 GPUs, 16/27 jobs
had fail-slows (72 min mean, 1.34x JCT delay). ~5.5k LoC DETECT + ~1.5k LoC
MITIGATE (Megatron-LM plugin).

**Telemetry.** NCCL call types + timestamps (LD_PRELOAD), inferred iteration
time, per-group CUDA-event times, SM utilization, CNP counts, GPU temperature
(NVML), CPU satisfaction rate.

**Stated limits.** Cannot detect fail-slows that appear only under specific
compute-comm co-execution; mitigation needs framework support (Megatron-LM
plugin); TP adaptation ineffective (single-node); no room to mitigate if all DP
groups degrade.

**Systems introduced.** GREYHOUND (DETECT + MITIGATE).

---

## 12. ResiHP

**Taming LLM Training Failures with Dynamic Hybrid Parallelism.** Fudan +
Shanghai AI Lab + HKUST, arXiv:2605.06374.

**Thesis.** A resilient system that (a) filters sequence-length-induced iteration
fluctuations for accurate fail-slow/fail-stop detection and (b) progressively
adapts TP, PP, and DP jointly to mitigate the device-performance skew both
failure types create.

**Problem.** Prior systems ignore sequence-length variability (with packing,
attention cost scales with sum(l_i^2), so healthy times fluctuate and trigger
false alarms) and adapt only one parallelism dimension, conservatively excluding
a whole TP group even when one device fails.

**Key mechanisms.**
- **Fail-stop detection**: hierarchical two-level heartbeat (intra-node local
  monitor + inter-node coordinator) plus a TCP side-channel for socket
  disconnection; scales with node count not device count.
- **Workload-aware fail-slow detection**: Greyhound-style change-point +
  validation, but adds a **micro-batch time predictor** `T_MB ~ alpha*N +
  beta*sum(l_i^2)` and a DAG-based **iteration-time predictor** (F/B/W chunks,
  critical-path length); validation triggers only if observed exceeds predicted
  healthy by **>25%**.
- **TP selective device exclusion**: excludes only failed/degraded devices, keeps
  power-of-two TP degrees, and picks the subgroup maximizing `k * min p_i`.
- **PP layer repartition**: moves layers off the straggling stage (e.g.
  (4,4,4) -> (5,2,5)).
- **DP progress-aware workload migration**: constrained makespan minimization;
  online heuristic migrates a pending micro-batch from the slowest to fastest DP
  group when the progress gap exceeds delta or on fail-stop eviction.
- **P2P optimization + state recovery**: under heterogeneous TP degrees, sender
  scatters into N chunks, sends one copy over InfiniBand, receiver reconstructs
  via NVLink all-gather; optimizer state copied from surviving replicas.

**Headline numbers.** Throughput 1.04-4.39x over SOTA on 256 GPUs; detection
~99.4% (>99.0% fail-slow, 99.6% fail-stop); cost-model mean absolute percentage
error is 1.19-1.58% for MTP and 2.81-5.06% for ITP; false alarms 0-0.3 vs Greyhound 3.7-8.7; false-alarm overhead 34-49
ms vs Greyhound 1.24-1.72 s. Fail-stop 1.22-1.82x over ReCycle, 1.07-1.51x over
Oobleck. Failure-amplification (LLaMA2-13B, TP4/DP2/PP4): one degraded GPU delays
+16 at DP; ResiHP cuts idle-time amplification from 25.43x to 11.14x at DP. ~9k
LoC Python; testbed 32 nodes x 8 A100, 200 Gbps HDR IB.

**Telemetry.** Heartbeat liveness + local progress, iteration-time series,
per-device normalized throughput, per-stage progress counters, packed sequence
lengths, memory footprint.

**Stated limits.** Covers only fail-stop and *persistent* fail-slow (no signature
= out of scope); layer repartition contributes less; communicator reconstruction
is framework-specific; strengthened ReCycle gives negligible benefit in mixed
failures.

**Systems introduced.** ResiHP (Detector + Scheduler), MTP, ITP, progress-aware
migration. Baselines: Greyhound, Adaptra, ReCycle, Oobleck.

---

## 13. R2CCL

**Reliable and Resilient Collective Communication Library for LLM Training and
Serving.** Univ. of Maryland, arXiv:2512.25059.

**Thesis.** A drop-in NCCL/RCCL extension that keeps an in-flight collective
running through inter-node NIC/link failures by hot-repairing to backup RDMA
connections, then rebalancing over surviving heterogeneous links -- avoiding
checkpoint restarts.

**Problem.** Network faults cause up to half of failed jobs and 10-15% of wasted
GPU-hours; a mid-collective NIC fault aborts the job because NCCL assumes a fixed
graph. Checkpoint recovery is heavy -- median 68 min (~12.7% of job duration).

**Key mechanisms.**
- **Bilateral failure awareness**: out-of-band notification over the bootstrap
  network; either endpoint alerts its peer immediately -- detection minutes ->
  milliseconds.
- **Precise fault localization**: dedicated probe QP pools + three-point
  triangulation (zero-byte RDMA Writes from both endpoints + auxiliary NIC) to
  distinguish local-NIC vs remote-NIC vs link failure.
- **Live migration (hot repair)**: GPU-NIC multi-registration (each GPU buffer
  pre-registered with all NICs at init, ordered by PCIe distance) + DMA-buffer
  rollback (sender rewinds to first uncompleted chunk, receiver resets to last
  confirmed chunk, retransmits over the next backup NIC).
- **R2CCL-Balance** (single failure, all collectives except throughput-AllReduce):
  redistributes the failed NIC's share proportional to available bandwidth,
  PXN- and NUMA-aware, NCCL algorithms unchanged.
- **R2CCL-AllReduce** (throughput): decomposes into a global AllReduce (throttled
  by the degraded node) + a partial AllReduce (excluding it) run concurrently,
  cutting the bottleneck from ~2D toward ~1.75D; ring when lost-bandwidth
  fraction X<1/3, R2CCL-AllReduce when 1/3<=X<1.
- **Multi-failure**: topology-aware logical re-ranking inserting bridge nodes;
  recursive R2CCL-AllReduce peeling off faster sub-rings.

**Headline numbers.** Under a single NIC failure (12.5% bandwidth loss): <1%
training / <3% inference overhead; beats AdapCC 12.18x (training) and DejaVu 47x
(inference). GPT-3 2.7B DP=16: R2CCL-AllReduce 0.71% overhead; two simultaneous
failures 1.24%. Simulation: 10 concurrent failures across 512 GPUs -> only 4.3%
overhead. 175B/1024-GPU: ~54x less failure-induced time vs AdapCC. ~3k LoC C++
drop-in on NCCL 2.23.4; testbed 2 nodes x 8 H100 + 8 ConnectX-7 400Gbps.

**Telemetry.** RDMA CQ/QP errors, zero-byte probe outcomes, OOB notifications,
per-NIC health, per-collective data volume + NIC bandwidth, alpha-beta model
params.

**Stated limits (explicit scope table).** Out of scope: NVLink/NVSwitch faults,
switch outages, network partitions, GPU/OS/process crashes, cross-rail
miswiring. Partial: link flapping/CRC only if escalating to a transport failure;
PCIe/degraded GPUDirect only if other NICs remain. Requires the process to stay
alive with >= 1 healthy inter-node NIC.

**Systems introduced.** R2CCL (HotRepair, Balance, AllReduce, recursive AllReduce,
bridge re-ranking). Baselines: AdapCC (training), DejaVu (inference), SimAI.

---

## 14. Llama 3

**The Llama 3 Herd of Models.** Meta AI, 2024. arXiv:2407.21783.

**Thesis.** A full technical report on Llama 3 (up to 405B); the reliability
content is a small but concrete slice: at 16,384-GPU scale over a 54-day
snapshot, 466 job interruptions occurred, 78% of unexpected ones hardware or
suspected-hardware, GPU issues 58.7%.

**Problem (infra slice).** Sustaining a 405B pretraining run over a 16,384-GPU
cluster where interruptions are frequent and diagnosis must be fast.

**Key mechanisms (infra slice).**
- **NCCLX flight recorder** (Meta's in-house NCCL): records collective metadata,
  stack info, comm events, and timings, auto-exported on watchdog/heartbeat
  timeout; on-demand config change deepens tracing without a job restart.
- **Straggler screening**: tools prioritize the most suspicious slow comm in a
  process group, checking only a few top suspects.
- **Automation**: the vast majority of interruptions handled automatically, only
  a few needing human intervention.
- Fun facts: 1-2% diurnal throughput swing (midday heat); synchronous waits
  cause tens-of-megawatts power swings near grid limits; 405B chose no MoE/RL for
  model+system stability.

**Headline numbers.** 16,384 GPUs; 54-day snapshot; 466 interruptions; 78% of
unexpected interruptions hardware/suspected-hardware; GPU issues 58.7%.

**Telemetry.** NCCLX flight recorder (collective metadata + stacks + events +
timings), watchdog/heartbeat timeouts, straggler comm ranking.

**Stated limits.** The report notes model and system stability are equally
critical (the stated reason 405B avoided MoE/RL); the reliability treatment is
brief and does not detail mechanisms deeply.

**Systems introduced.** NCCLX (Meta's NCCL enhancement) + PyTorch flight
recorder integration.

---

## 15. Cross-cutting synthesis

Reading the 13 together, a few axes organize the space.

**Detection substrate.** Three families of signal recur:
- *Framework-semantics / iteration time* (MegaScale heat-map, ARGUS L1/L2,
  GREYHOUND ACF+BOCD, ResiHP ITP, Minder per-second metrics) -- cheap, always-on,
  localizes to a machine or group.
- *CCL-internal state* (Mycroft G/T/D chunk progress, Aegis CL/WR/WC,
  R2CCL RDMA CQ/QP) -- opens the collective black box to separate compute-not-
  ready from RDMA-not-done.
- *Kernel / hardware sampling* (ARGUS CUPTI + py-spy, EROICA 10 kHz nsys, Story
  of Two GPUs XID+DCGM) -- profiler-grade, historically too heavy for always-on,
  now made viable by compression (ARGUS ~3,700x, EROICA 10^5x pattern reduction).

**Detect vs diagnose vs recover.** ByteRobust, Aegis, and MegaScale are
end-to-end platforms (detect + isolate + recover). ARGUS, EROICA, Mycroft, Minder
are diagnosis-focused. GREYHOUND and ResiHP pair detection with in-place
*mitigation* (micro-batch/topology reshaping) rather than restart. R2CCL is pure
in-flight *recovery* at the CCL layer. Story of Two GPUs and Revisiting
Reliability are measurement studies that inform the others' assumptions
(MTTF scaling, per-GPU error rates, lemon nodes).

**The fail-slow consensus.** Multiple independent groups (GREYHOUND, ResiHP,
EROICA, ARGUS) converge on the same insight: fail-slows are persistent (do not
self-heal), so *on-demand* deep profiling triggered after a lightweight detector
is both sufficient and affordable. They differ on the comparison primitive:
cross-rank behavior patterns (EROICA), Wasserstein distance on kernel
distributions (ARGUS), cross-group CUDA-event timing (GREYHOUND), embedding
distance (Minder).

**What this maps to in torchtitan.** As detailed in
[`TELEMETRY_MAPPING.md`](TELEMETRY_MAPPING.md), torchtitan implements the
primitives these systems build on -- NCCL FlightRecorder dump-on-timeout,
process-group timeouts, anomaly/NaN detection, the MFU/loss/grad-norm metrics
pipeline, torchft quorum, and bitwise loss_compare -- but not the platform-half
machinery (hardware health polling, stack-trace clustering, dual-phase replay,
kernel-distribution comparison). The papers are, in effect, a catalog of what a
production controller wraps around a torchtitan-like trainer.
