# LLM training reliability survey: cross-paper fact synthesis

Companion to [`PAPER_DEEP_DIVE.md`](PAPER_DEEP_DIVE.md) (per-paper deep dives),
[`REFERENCE_CLOSURE.md`](REFERENCE_CLOSURE.md) (depth-2 reference closure), and
[`TELEMETRY_MAPPING.md`](TELEMETRY_MAPPING.md) (ByteRobust-telemetry-to-torchtitan
mapping). This doc extracts every concrete fact from the 13 deep-dived papers,
categorizes them, and distills three views the papers agree on:

1. [Categorized facts](#1-categorized-facts) -- every grounded number/mechanism,
   binned by topic.
2. [Common failure modes and challenges](#2-common-failure-modes-and-challenges).
3. [The common distributed-training setup](#3-the-common-distributed-training-setup)
   -- model, framework/software, GPU hardware, networking.

Papers referenced by short name: ByteRobust (1), MegaScale (2), ARGUS (3),
EROICA (4), Mycroft (5), Revisiting Reliability (6), A Story of Two GPUs (8),
Minder (9), Aegis (10), GREYHOUND (11), ResiHP (12), R2CCL (13), Llama 3 (14).
Paper 7 (Intel) is excluded (paywalled). Every number is the authors' own.

---

## 1. Categorized facts

### 1.1 Scale, fleet, and study size

| Fact | Source |
| --- | --- |
| 97% ETTR on 9,600 GPUs for 3 months; unproductive time <= 50 min | ByteRobust |
| Census of 778,135 jobs | ByteRobust |
| 55.2% MFU on 175B over 12,288 GPUs | MegaScale |
| Deployed on >10,000 GPUs, repaired >100 times, run for weeks | MegaScale |
| Always-on tracing on 10,000+ GPUs for >6 months | ARGUS |
| Motivating case: 4,096-GPU job wasting ~23,758 GPU-hours | ARGUS |
| 1.5 years on ~100,000 GPUs; largest job 6,144 GPUs | EROICA |
| Deployed >6 months at ByteDance; scales to tens of thousands of GPUs | Mycroft |
| 11 months, 4M jobs, 150M+ A100-GPU-hours, two clusters (16k + 8k GPUs) | Revisiting |
| 2.5 years, 11.7M GPU-hours, 1,056 GPUs (448 A100 + 608 H100) on NCSA Delta | Story |
| Deployed >1 year; 150 fault instances over 9 months; 4 to >1500 machines | Minder |
| Scale grew >40x over 16 months; 100-230 critical failures/week | Aegis |
| 10,000+ GPUs, 4,000 nodes characterization | GREYHOUND |
| 16,384 GPUs, 54-day snapshot | Llama 3 |

### 1.2 ETTR, MFU, and effective-time results

| Fact | Source |
| --- | --- |
| 97% cumulative ETTR; MFU up 1.25x (dense) / 1.58x (MoE) via hot-updated code | ByteRobust |
| Warm standby up to 10.87x, hot update up to 11.04x faster than requeue on 16,384 GPUs; within 5.19% of oracle | ByteRobust |
| 55.2% MFU (1.34x Megatron-LM); 256-GPU ablation 47.7% -> 65.3% MFU | MegaScale |
| >90% effective training time; >90% faults auto-detected; diagnostics <10 min; catch-up within 15 min | MegaScale |
| Throughput +20-100% (largest job 6,144 GPUs); 78/80 hard issues diagnosed (97.5%) | EROICA |
| Largest jobs reach ETTR ~0.85-0.9 | Revisiting |
| To hit 0.9 ETTR at 12,000 GPUs, r_f must drop 6.50 -> ~1 or checkpoint write to ~10 s | Revisiting |
| >97% reduction in diagnosis idle time; 84.6% fewer restarts; 71% less degradation | Aegis |
| End-to-end +1.58x on 256 H800 | GREYHOUND |
| Throughput 1.04-4.39x over SOTA on 256 GPUs | ResiHP |

### 1.3 Detection and diagnosis latency / accuracy

| Fact | Source |
| --- | --- |
| Real-time checks at second-level; fail-slow re-aggregates every 10 s over 5 rounds | ByteRobust |
| >90% faults auto-detected; diagnostics <10 min | MegaScale |
| 5-level diagnosis narrows tens of thousands of ranks to single-digit suspects | ARGUS |
| Anomalies detected within 15 s in 90% of cases; root cause within 20 s in 60% | Mycroft |
| Anomalies propagate cluster-wide in a few hundred ms | Mycroft |
| Alert in 3.6 s average (>99% / 500x faster than manual >30 min); precision 0.904 / recall 0.883 / F1 0.893 | Minder |
| Detection 99.8% (498/499); BOCD detects in 2-3 iterations (<5 s); overhead avg 0.39% (max 1.1%) | GREYHOUND |
| BOCD+V 100% accuracy / 0% false-positive / 0% false-negative (compute); 99.1%/0%/2.3% (comm) | GREYHOUND |
| Detection ~99.4% (>99.0% fail-slow, 99.6% fail-stop); false alarms 0-0.3 vs Greyhound 3.7-8.7 | ResiHP |
| Bilateral awareness cuts detection from minutes to milliseconds | R2CCL |

### 1.4 Failure taxonomy and incident-mix statistics

| Fact | Source |
| --- | --- |
| Job mix: CUDA error 36.1%, code/data adjustment 17.3%, job hang 9.9%, NaN 0.3% | ByteRobust |
| Across 19 jobs: real-time eviction 32.52%, reattempt 22.70%, rollback 9.20%, dual-phase replay 1.23% | ByteRobust |
| NVIDIA EUD reaches only 70% SDC recall | ByteRobust |
| 9-month ~100,000-GPU mix: 44.4% hardware, 48.2% software, 7.4% unknown | EROICA |
| Online monitors root-cause only ~29.6%; 7.4% of issues never reproduce | EROICA |
| Infra failures hit 0.2% of jobs but 18.7% of GPU runtime | Revisiting |
| One 1024-GPU job NODE_FAILed 35 times -> 548 preemptions | Revisiting |
| Fault mix: ECC 25.7%, CUDA exec 15%, GPU exec 10%, PCIe downgrading 8.6%; individual-machine 99% | Minder |
| 45.6% GPU-related failures; 73% of tasks fail in first 10 min; 71% of "distributed" failures unrelated to network | Aegis |
| Meta OPT-175B: >105 restarts, ~1.25 incidents/day, 61,000 GPU-hours impacted | Mycroft (cites) |
| 466 interruptions in 54 days; 78% of unexpected ones hardware/suspected-hardware; GPU issues 58.7% | Llama 3 |
| Network faults cause up to half of failed jobs and 10-15% of wasted GPU-hours | R2CCL |

### 1.5 Fail-slow characterization

| Fact | Source |
| --- | --- |
| Compute fail-slows <2% frequency (~10 min); comm fail-slows 40% frequency (~24 min) | GREYHOUND |
| At >= 512 GPUs, 16/27 jobs had fail-slows (72 min mean, 1.34x JCT delay) | GREYHOUND |
| Fail-slows are persistent (do not self-heal), so post-onset profiling is timely | EROICA, GREYHOUND, ResiHP, ARGUS |
| Case: self_attention 2,252 ms vs 6-16 ms normal (>150x) | ARGUS |
| Case: 3,400 H800 iteration 10.5s -> 8.5s after fix (+34% throughput) | EROICA |
| One straggler slows the whole synchronous job | MegaScale, GREYHOUND, Minder |

### 1.6 Hardware error / reliability measurements

| Fact | Source |
| --- | --- |
| H100 memory 3.2x lower per-GPU MTBE for uncorrectable ECC (88,768 h vs A100 283,271 h) | Story |
| Per-GB MTBE ~8.5M h (HBM3) vs ~11.3M h (HBM2e) = 24% lower | Story |
| Row-remapping mitigates 92% of H100 uncorrectable ECC; spare rows capped at 512 (not capacity-scaled) | Story |
| 8 row-remap failures on H100 vs 0 on A100 | Story |
| A100 HW errors: 8,863 MMU, 3,857 GSP, 1,922 NVLink, 77 PMU SPI; H100: 3 GSP, 0 NVLink, 0 PMU SPI | Story |
| System MTBE A100 1.4 h / H100 1.9 h; availability ~99.4% / 99.3% | Story |
| A100 MTTF ~400 days, H100 ~200 days | Aegis |
| MTTF: 8-GPU job = 47.7 days; 1024-GPU = 7.9 h; 16,384-GPU = 1.8 h; 131,072-GPU = 0.23 h | Revisiting |
| RSC-1 r_f = 6.50 failures/1000 node-days; RSC-2 = 2.34 | Revisiting |
| Lemon detection: 40 faulty nodes, >85% accuracy, 512+-GPU failures 14% -> 4% | Revisiting |
| One DBE example took 19 h recovery; utilization A100 51% / H100 41%; temps 40C / 37C | Story |

### 1.7 Compression, overhead, and data-volume engineering

| Fact | Source |
| --- | --- |
| Compression ~3,700x via log-transform + Gaussian-KDE clustering, storing (count, p50, p99) | ARGUS |
| Total overhead <2% (CUPTI ~1-2%) vs PyTorch Profiler 20-44% then OOM | ARGUS |
| Per-rank-per-step 10.6 MB raw -> 18.7 KB summaries; ~2.7 GB/min upload at 10,000 GPUs; +2-10 GB memory | ARGUS |
| Trace volume wall: 10^4-10^5 kernel events/min/GPU = 6-60 GB/min cluster | ARGUS |
| Pattern data 30 KB/worker vs ~3 GB raw (10^5x smaller) | EROICA |
| Offline profilers produce ~100 MB/worker/sec (~1 TB/s at 10,000 GPUs) | EROICA |
| Fixed-size 512 MB/host circular buffer, near-zero overhead, 1,100 LoC C++, <10 tracepoints | Mycroft |
| XPUTimer traces exceed 100 MB/iteration/GPU; NPKit caused two-thirds bus-bandwidth drop | Mycroft (cites) |
| Every-step checkpoint <0.9% overhead, cutting blocking time 99.69% vs Megatron save | ByteRobust |
| Tracking overhead avg 0.39% (max 1.1%) | GREYHOUND |
| Single NIC failure (12.5% bandwidth loss): <1% training / <3% inference overhead | R2CCL |

### 1.8 Codebase size and testbeds

| Fact | Source |
| --- | --- |
| Robust Controller ~20k LoC Go, Agent ~5k Python, Runtime Analyzer ~12k Go, CKPT mgr ~3k Python | ByteRobust |
| Built on Megatron-LM 285068c8; open-sourced under veScale | MegaScale |
| EROICA ~7k LoC Python; integrates Cursor for AI auto-fix | EROICA |
| Mycroft NCCL 2.21.5, 1,100 LoC C++; testbed 32 A100 | Mycroft |
| GREYHOUND ~5.5k LoC DETECT + ~1.5k LoC MITIGATE (Megatron-LM plugin); testbed 256 H800 | GREYHOUND |
| ResiHP ~9k LoC Python; testbed 32 nodes x 8 A100, 200 Gbps HDR IB | ResiHP |
| R2CCL ~3k LoC C++ drop-in on NCCL 2.23.4; testbed 2 nodes x 8 H100 + 8 ConnectX-7 400Gbps | R2CCL |

### 1.9 Recovery and mitigation mechanisms

| Mechanism | Source |
| --- | --- |
| Direct eviction / reattempt / code rollback / dual-phase replay state machine | ByteRobust |
| Warm standby sized to P99 binomial (e.g. 4 backups per 1,024); in-place lazy hot-update | ByteRobust |
| Every-step in-memory checkpoint, dual CPU buffers, async D2H, cross-group backup | ByteRobust |
| Two-stage (host then remote) checkpoint | MegaScale |
| Ski-rental multi-level: Ignore -> micro-batch rebalance -> topology reshape -> checkpoint-restart | GREYHOUND |
| Progressive TP/PP/DP adaptation; selective device exclusion; layer repartition; workload migration | ResiHP |
| Live migration to backup RDMA connection + DMA-buffer rollback; R2CCL-Balance / -AllReduce | R2CCL |
| Health-check-based removal every 5 min; lemon-node isolation; Adaptive Routing + SHIELD | Revisiting |
| Check Before Delivery pre-flight (<10 min full, <1 min light) | Aegis |

---

## 2. Common failure modes and challenges

Binning every failure the papers name yields four recurring modes plus a set of
cross-cutting challenges every system wrestles with.

### 2.1 The four failure modes

**A. Fail-stop (hard crash / hang).** A process, GPU, NIC, or link dies and the
synchronous collective blocks the whole job. Signatures: CUDA errors (36.1% of
jobs, ByteRobust), NCCL timeout cascades (Minder: one machine halts thousands),
double-bit ECC / XID 48 / XID 94-95 / link-down (Aegis CriticalError), zero RDMA
traffic for 10 min (ByteRobust hang rule). Detected cheaply by heartbeats,
exit codes, and timeout tripwires. Recovery = evict + reschedule + reload
checkpoint. This is the well-understood case; the papers optimize its *latency*
(warm standby, in-place update, host-side checkpoints).

**B. Fail-slow / straggler (still running, just slower).** The hard case that
five independent groups (GREYHOUND, EROICA, ARGUS, ResiHP, Minder) built systems
around. Causes: CPU contention, thermal throttling (Llama 3: 1-2% diurnal swing),
network congestion, irregular GC (MegaScale), reduce-scatter time skew growing
with steps (MegaScale). Hard because per-iteration sync drops *all* GPUs'
utilization at once (GREYHOUND) and shared links make telemetry ambiguous.
Frequency is non-trivial: comm fail-slows hit 40% of the time (~24 min each) and
at >= 512 GPUs 16/27 jobs saw one (GREYHOUND). Universal insight: fail-slows are
*persistent*, so a lightweight always-on detector plus on-demand deep profiling
is both sufficient and affordable.

**C. Silent Data Corruption (SDC) / implicit faults.** No error, no XID, wrong
math. NVIDIA EUD catches only 70% (ByteRobust); Revisiting Reliability explicitly
de-prioritizes SDC ("rarely triggers user complaints"). ByteRobust's dual-phase
replay (dimension-aware group testing over DP while holding TP/PP) is the only
surveyed mechanism that isolates SDC, and it handled just 1.23% of events at high
cost. Surfaced indirectly via loss/grad-norm anomalies (5x jump or NaN).

**D. Network faults (NIC / link / switch / QP).** Up to half of failed jobs and
10-15% of wasted GPU-hours (R2CCL). Modes: NIC dropout, PCIe downgrade, link
flapping (needs larger NCCL timeouts, MegaScale), CRC errors, QP errors, ToR
faults, silent >1KB-packet loss invisible to 64B probes (Aegis). R2CCL keeps the
collective alive through a mid-flight NIC failure; Aegis/Minder localize; but
NVLink/NVSwitch faults and network partitions remain out of scope for R2CCL.

### 2.2 Cross-cutting challenges

| Challenge | How it shows up |
| --- | --- |
| **Fault propagation** | A single-point fault cascades to a whole-job crash within a few hundred ms (Mycroft), hiding the culprit among all hosts' error reports (Aegis, Minder). |
| **Root cause vs localization** | Most systems localize the faulty machine/group but cannot root-cause (Minder, Aegis CCL info, Mycroft). Full RCA still needs human experts (ARGUS L4/L5, EROICA complex Python). |
| **Telemetry data-volume wall** | Fine-grained traces are 6-60 GB/min/cluster (ARGUS) or ~1 TB/s (EROICA); the enabling trick is compression (3,700x ARGUS, 10^5x EROICA) or fixed circular buffers (Mycroft 512 MB/host). |
| **CCL is a black box** | On silent timeout/fail-slow, operators cannot see NCCL internal state; opening it needs custom tracepoints (Mycroft G/T/D) or a custom CCL (Aegis CL/WR/WC), which then must be re-maintained per NCCL version. |
| **Distinguishing anomaly from intent** | Intentional comm/compute overlap and natural workload variation (VLM, packed sequence lengths) defeat naive thresholds (Mycroft, ARGUS masking, ResiHP sequence-length false alarms). |
| **Diagnosis tools lag hardware** | EUD once caused GPU downclocking; offline profilers are unusable without reproduction and 7.4% never reproduce (EROICA); SuperBench init takes tens of minutes (Aegis). |
| **Scale-driven overheads** | torch.dist init is O(n^2) (MegaScale fixed to O(n)); centralized coordinators bottleneck; MTTF collapses from 47.7 days (8 GPU) to 7.9 h (1024 GPU) to 0.23 h (131,072 GPU) (Revisiting). |
| **Recovery cost dominates** | Checkpoint restart is heavy -- GPT2-100B dump ~100 min (GREYHOUND), median 68 min = 12.7% of job (R2CCL) -- motivating in-place mitigation over restart. |
| **Too few spares** | At scale there are not enough spares to swap (ByteRobust), forcing over-eviction and warm-standby ratios (1/64 to 1/256). |
| **Human/code evolution** | Multi-month runs see continuous code/data changes (17.3% of ByteRobust events), needing lazy hot-update to avoid full relaunch. |

---

## 3. The common distributed-training setup

Aggregating what every paper assumes or reports, the "typical" large-scale LLM
training stack is remarkably consistent across ByteDance, Alibaba, Tencent, Meta,
and academia.

### 3.1 Model layer

- **Dense and MoE transformers**, 100B-405B parameters (175B MegaScale, 405B
  Llama 3, 100B+ in ByteRobust/EROICA cases). Llama 3 405B deliberately avoided
  MoE/RL for model+system stability.
- **Architectural efficiency knobs** (MegaScale): parallel transformer block
  (`y = x + MLP(LN(x)) + Attn(LN(x))`), sliding-window attention (O(s*w) vs
  O(s*s)), LAMB to scale batch 4x.
- **Kernels**: FlashAttention-2, fused LayerNorm, fused GeLU (MegaScale).
- **Sequence packing**: attention cost scales with sum(l_i^2), so per-iteration
  time fluctuates with packed sequence lengths -- a source of false fail-slow
  alarms that ResiHP models explicitly (`T_MB ~ alpha*N + beta*sum(l_i^2)`).

### 3.2 Parallelism / framework layer

- **3D (hybrid) parallelism**: Tensor (TP) x Pipeline (PP) x Data (DP), plus
  Sequence Parallel and Expert Parallel for MoE. This is the universal assumption
  (MegaScale, ByteRobust dual-phase replay holds TP/PP and varies DP, GREYHOUND,
  ResiHP, Minder "3D parallelism balances load").
- **Communication overlap**: DP all-gather prefetch, interleaved 1F1B pipeline
  (bubble `(4vp-1)/m` -> `(vp-1)/4m`), TP/SP all-gather + reduce-scatter fused
  into FFN Linears with chunked GEMM (MegaScale).
- **Base frameworks**: Megatron-LM (MegaScale fork 285068c8, GREYHOUND plugin,
  ResiHP), PyTorch + NCCL/NCCLX (Llama 3, Mycroft, all). torchtitan is the same
  family of PyTorch-native SPMD trainer.
- **ZeRO / sharding**: ZeRO-2 with first-layer prefetch (MegaScale); FSDP-style
  sharding is the torchtitan analog.
- **Comm-group init**: PyTorch TCPStore is O(n^2) and slow (1047s at 2,048 GPUs);
  replaced with async Redis + O(n) barrier (MegaScale).
- **Checkpointing**: hierarchical GPU -> host -> SSD/peer, async D2H on a
  dedicated stream, every-step in-memory + periodic remote (ByteRobust, MegaScale
  two-stage). Daly-Young optimal interval `sqrt(2*w_cp/(N_nodes*r_f))` (Revisiting).
- **Schedulers**: Kubernetes (MegaScale driver/executor, EROICA emptyDir), Slurm
  + submitit (Revisiting, Story of Two GPUs).

### 3.3 GPU hardware layer

- **GPUs**: NVIDIA A100 (Ampere) and H100 (Hopper), incl. GH200 (Story of Two
  GPUs); H800 (GREYHOUND, EROICA), V100 (Minder legacy). 8 GPUs/node is standard;
  GH200 uses 4-way nodes.
- **Intra-node interconnect**: NVLink / NVSwitch (NVLink-error is a tracked fault
  in Minder/Aegis/Story; R2CCL explicitly excludes NVLink/NVSwitch faults).
- **Memory**: HBM2e (A100) / HBM3 (H100); uncorrectable ECC is the dominant
  memory fault, mitigated by row-remapping (512 spare rows, capacity-independent)
  and page retirement (Story of Two GPUs).
- **Health signals**: XID syslog codes, DCGM (1-min polling), PCIe bandwidth,
  memory row-remap counts, GPU temperature, GPU duty cycle / power / tensor
  activity (Minder, Story, ByteRobust, Revisiting).
- **Diagnostics**: NVIDIA EUD (70% SDC recall), SuperBench stress tests (offline),
  intra/inter-node all-to-all and all-gather probes (ByteRobust, Aegis, Minder).

### 3.4 Networking layer

- **Fabric**: RDMA over Converged Ethernet (RoCE) or InfiniBand; multi-tier
  fat-tree (Clos-style) on Broadcom Tomahawk 4 (MegaScale); 200 Gbps HDR IB
  (ResiHP), 400 Gbps ConnectX-7 (R2CCL).
- **CCL**: NCCL (2.21.5 Mycroft, 2.23.4 R2CCL), NCCLX (Meta), RCCL/ACCL/MSCCL
  (GREYHOUND framework-agnostic). The proxy threads and QP/flow/chunk internals
  are what Mycroft and Aegis instrument.
- **Congestion control**: Swift RTT + DCQCN/ECN, PFC, tuned NCCL retransmit and
  NIC `adap_retrans` (MegaScale). PFC Tx rate is Minder's most fault-sensitive
  metric.
- **Routing / resilience**: Adaptive Routing (AR) + SHIELD (Revisiting: without
  AR, 50-75% bandwidth loss in bring-up); topology-aware scheduling (MegaScale);
  PXN- and NUMA-aware rebalancing (R2CCL).
- **Network telemetry**: RDMA metrics via heartbeats, in-band per-hop coloring,
  RDMA Pingmesh / R-Pingmesh, packet-level telemetry, three-point QP
  triangulation (MegaScale, Aegis, R2CCL, Minder).
- **Failure surface**: NIC dropout, link flap/CRC, QP errors, ToR/switch faults,
  silent large-packet loss. Fault localization must separate local-NIC vs
  remote-NIC vs link (R2CCL) and compute-not-ready vs RDMA-not-done (Mycroft
  G/T/D).

### 3.5 The stack at a glance

```
Model      : 100B-405B dense/MoE transformer; FA2, fused LN/GeLU; packed seqs
             |
Framework  : Megatron-LM / PyTorch(+NCCL); 3D parallelism (TP x PP x DP) + SP/EP;
             ZeRO/FSDP sharding; comm-overlap; hierarchical async checkpoint;
             K8s / Slurm scheduling
             |
GPU HW     : 8x A100/H100 per node; NVLink/NVSwitch intra-node; HBM2e/HBM3 + ECC
             row-remap; XID/DCGM/PCIe/temp health signals; EUD/SuperBench probes
             |
Network    : RoCE/InfiniBand multi-tier fat-tree; NCCL/NCCLX CCL over RDMA;
             DCQCN/ECN/PFC congestion control; Adaptive Routing; Pingmesh telemetry
```

Every reliability system in the survey attaches to one or more layers of this
stack: measurement studies (Story, Revisiting) characterize the GPU-HW and
network layers; CCL tracers (Mycroft, Aegis, R2CCL) sit at the networking layer;
fail-slow detectors (GREYHOUND, EROICA, ARGUS, Minder, ResiHP) read framework
and GPU-HW signals; and end-to-end platforms (ByteRobust, MegaScale) wrap all
four. As [`TELEMETRY_MAPPING.md`](TELEMETRY_MAPPING.md) shows, torchtitan
implements the framework-layer primitives (FlightRecorder, PG timeouts, NaN/
anomaly detection, MFU/loss/grad-norm metrics, torchft quorum) that these
production controllers build their detection and recovery machinery on top of.
