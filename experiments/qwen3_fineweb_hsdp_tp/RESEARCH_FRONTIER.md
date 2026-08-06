# Applied research frontier: next-gen robust large-scale training

An evergreen guide for building the next generation of robust, large-scale LLM
training infrastructure. It is grounded in a 13-paper reliability survey
(ByteRobust, MegaScale, ARGUS, EROICA, Mycroft, Revisiting Reliability, A Story
of Two GPUs, Minder, Aegis, GREYHOUND, ResiHP, R2CCL, Llama 3) and its depth-2
reference closure, and it is written to outlast any single paper: it states the
durable problems, the design principles the field has converged on, the open
frontiers, and a torchtitan-anchored roadmap.

Source docs (read these for the receipts):
- [`PAPER_DEEP_DIVE.md`](PAPER_DEEP_DIVE.md) -- per-paper mechanisms and numbers.
- [`FACTS_SYNTHESIS.md`](FACTS_SYNTHESIS.md) -- categorized facts, failure modes,
  common stack.
- [`REFERENCE_CLOSURE.md`](REFERENCE_CLOSURE.md) -- what the field cites and how
  the ideas descend.
- [`TELEMETRY_MAPPING.md`](TELEMETRY_MAPPING.md) -- which primitives torchtitan
  already implements.

This doc is opinionated on purpose. Where the evidence supports a bet, it makes
the bet. Update it as new evidence arrives; the section structure is meant to be
stable, the contents are meant to churn.

## Contents

1. [The durable problem statement](#1-the-durable-problem-statement)
2. [What the field has already settled](#2-what-the-field-has-already-settled)
3. [The open frontiers](#3-the-open-frontiers)
4. [Design principles for a next-gen system](#4-design-principles-for-a-next-gen-system)
5. [Reference architecture](#5-reference-architecture)
6. [torchtitan-anchored roadmap](#6-torchtitan-anchored-roadmap)
7. [How to keep this doc evergreen](#7-how-to-keep-this-doc-evergreen)

---

## 1. The durable problem statement

The north-star metric is **ETTR** (Effective Training Time Ratio): the fraction
of wall-clock a large job spends making forward progress. Everything below is in
service of driving ETTR toward 1.0 at ever-larger scale. Three facts make this
hard and will not change with the next hardware generation:

- **Failure is the steady state, not the exception.** MTTF collapses with scale:
  47.7 days at 8 GPUs, 7.9 hours at 1,024, a projected 0.23 hours at 131,072
  (Revisiting Reliability). Any system that treats a failure as a rare event is
  designing for the wrong regime.
- **Synchronous training turns a local fault into a global stall.** One straggler
  or one dead NIC blocks the entire collective within a few hundred milliseconds
  (Mycroft, MegaScale, GREYHOUND). Blast radius, not fault count, is the enemy.
- **Recovery cost dominates the loss budget.** Checkpoint restart is heavy --
  ~100 min for a GPT2-100B dump (GREYHOUND), a median 68 min = 12.7% of job
  duration (R2CCL). The expensive part is almost never detecting the fault; it
  is the round-trip back to productive work.

The corollary that the whole field now accepts: **minimize total unproductive
time over the job lifecycle**, rather than maximize root-cause precision. Fast,
coarse isolation plus cheap recovery beats slow, exact diagnosis (ByteRobust's
97% ETTR was won this way).

## 2. What the field has already settled

Do not re-litigate these. They are the load-bearing consensus across the survey;
a next-gen system should treat them as the substrate it builds on.

- **The training stack is standardized.** 3D parallelism (TP x PP x DP, plus
  SP/EP for MoE) on Megatron-LM / PyTorch + NCCL, 8x A100/H100 per node,
  NVLink/NVSwitch intra-node, RoCE/InfiniBand multi-tier fat-tree inter-node.
  Megatron-LM is cited by 10 of 13 papers; ZeRO/GPipe/FSDP/PipeDream by 5-6 each.
  Reliability is a layer wrapped around a standard trainer, not a bespoke stack.
- **Checkpointing is a solved primitive.** In-memory / hierarchical checkpoints
  (GEMINI, cited by 6 of 13; CheckFreq by 5) plus Young-Daly optimal-interval
  theory give every-step snapshots at <0.9% overhead (ByteRobust). The mechanism
  is mature; only the recovery round-trip remains expensive.
- **Fail-slow is persistent, so on-demand deep profiling is the right pattern.**
  Five independent groups (GREYHOUND, EROICA, ARGUS, ResiHP, Minder) converged on
  the same two-tier design: a cheap always-on detector (iteration-time / metric
  similarity) that arms an expensive short-window profiler only after onset. This
  is settled methodology.
- **Compression makes always-on fine-grained telemetry viable.** The data-volume
  wall (6-60 GB/min/cluster raw) is beaten by aggressive summarization -- ARGUS
  ~3,700x via log-transform + KDE clustering, EROICA ~10^5x behavior patterns,
  Mycroft a fixed 512 MB/host ring buffer. Always-on kernel-grade tracing at <2%
  overhead is now demonstrated, not aspirational.
- **CCL internal state is observable.** The NCCL black box can be opened with a
  handful of tracepoints (Mycroft G/T/D chunk progress) or a swappable custom CCL
  (Aegis CL/WR/WC), separating "compute not ready" from "RDMA not done."

## 3. The open frontiers

Ordered by expected leverage. Each states the gap, the evidence it is real, and
a concrete direction.

### F1. Silent Data Corruption (highest leverage, least served)
**Gap.** SDC produces wrong math with no error, no XID, no signal. NVIDIA EUD
catches only 70% (ByteRobust); the only surveyed isolation mechanism is
ByteRobust's dual-phase replay, which handled 1.23% of events at high cost;
Revisiting Reliability explicitly de-prioritizes SDC. The closure has almost no
prior art (Dixit'21, Hochschild'21, Dr.DNA'24 and little else).
**Why it matters more over time.** As fail-stop and fail-slow get automated away,
SDC becomes the residual that silently corrupts weeks of training.
**Directions.** Cheap continuous numeric invariants (loss/grad-norm distribution
monitoring beyond the crude 5x/NaN trip); redundant/replica cross-checks on a
sampled fraction of ranks; dimension-aware group testing generalized beyond
ByteRobust's DP-only replay; activation-distribution anomaly detection (Dr.DNA
lineage) run online rather than offline.

### F2. Cross-comparable health, not per-paper point metrics
**Gap.** Every paper reports ETTR/MFU/MTTF/overhead on different scales,
hardware, and definitions, with no normalization. There is no shared health
score that composes hardware, network, and framework signals into one
comparable, capacity-normalized number.
**Evidence.** Story of Two GPUs shows per-GPU vs per-GB MTBE flip conclusions
(H100 looks 3.2x worse per-GPU but only 24% worse per-GB); Revisiting's lemon-node
score is 7 ad-hoc signals thresholded from a CDF.
**Directions.** A capacity- and generation-normalized reliability score;
per-failure-type recurrence distributions instead of one global blacklist
threshold; a health state machine richer than healthy/unhealthy (the survey's own
critique: first-strike-repaired, under-monitoring, second-strike-suspect,
awaiting-replacement).

### F3. In-place mitigation over restart (elasticity without the round-trip)
**Gap.** The heavy cost is the recovery round-trip. GREYHOUND (ski-rental
micro-batch/topology reshaping) and ResiHP (progressive TP/PP/DP adaptation,
non-uniform TP) show in-place mitigation works, but both are framework-specific
plugins with sharp limits (single-node TP, no room if all DP groups degrade).
**Directions.** First-class elastic parallelism in the trainer (not a plugin):
non-uniform TP/PP degrees, live micro-batch rebalancing, layer repartition, all
driven by a cost model that compares mitigation vs restart online. R2CCL shows
the CCL layer can also self-heal (hot-repair to a backup RDMA connection) without
touching the process -- push that idea up into the collective scheduler.

### F4. Root cause, not just localization
**Gap.** Most systems localize the faulty machine/group but stop there (Minder,
Aegis CCL info, Mycroft). Full root-cause still needs human experts (ARGUS L4/L5,
EROICA complex-Python cases). The 44.4%-hardware / 48.2%-software split (EROICA)
means half of all issues are software the current tools cannot attribute.
**Directions.** Close the loop from localization to attribution to a durable
knowledge base (the survey's own gap #1: "lacks fault attribution, repeatedly
steps on the same rake"); LLM-agent-assisted diagnosis over the compressed traces
(ARGUS and EROICA both flag this as future work); a cross-job meta-database so a
recurring signature is diagnosed once, not every time.

### F5. Unified observability plane
**Gap.** The signals exist but live in silos: framework-semantics/iteration time,
CCL-internal G/T/D, kernel/hardware sampling, network telemetry. The survey's own
finding: "Lagrange, Malphite, Onenet do not interoperate; debugging means
switching tools." No single system fuses all three detection substrates.
**Directions.** One trace model spanning py-spy CPU stacks + CUDA-event framework
phases + CUPTI kernels + CCL chunk progress + RDMA counters, correlated by a
common iteration ID (EROICA's globally-synchronized window is the seed). This is
integration engineering more than research, but it is where day-to-day toil lives.

### F6. Detecting faults that only appear under real co-execution
**Gap.** Explicitly stated limits in GREYHOUND and EROICA: a fault that manifests
only when compute and communication overlap in production can be invisible to
isolated GEMM/P2P validation probes. ResiHP's sequence-length-induced fluctuation
is a related false-alarm source.
**Directions.** Validation that replays the actual overlapped schedule, not
synthetic microbenchmarks; workload-aware baselines (ResiHP's T_MB ~ alpha*N +
beta*sum(l_i^2) predictor) so natural variation is not mistaken for a fault.

## 4. Design principles for a next-gen system

Distilled from what worked across the survey. These are the invariants a new
system should be architected around.

1. **Minimize unproductive time, not maximize precision.** Prefer fast coarse
   isolation (even slight over-eviction) plus cheap recovery. ByteRobust's
   real-time eviction handled 32.5% of events; exact replay only 1.23%.
2. **Two-tier detection.** Cheap always-on detector arms an expensive on-demand
   profiler. Never run profiler-grade telemetry unconditionally; never rely on
   coarse metrics alone for diagnosis.
3. **Compress at the source.** Summarize to distributions/patterns on the node
   before upload. Raw traces do not leave the host. Budget: single-digit MB per
   rank per step, low-single-digit percent overhead.
4. **Timestamp-independent comparison.** Cross-rank behavior comparison should not
   need clock sync (EROICA: NTP's ~10 ms error breaks event-timestamp diffing).
   Compare distributions and critical-path shares, not absolute times.
5. **Parallelism-group-aware everything.** Route comparisons within the correct
   group (attention across DP-equivalent ranks, EP compute within EP groups, DP
   reduce-scatter within DP groups). A raw all-rank comparison is meaningless
   under 3D parallelism.
6. **Recover in place first, restart last.** Escalate: ignore -> micro-batch
   rebalance -> topology reshape -> CCL hot-repair -> checkpoint restart, gated by
   an online cost model (ski-rental).
7. **Non-intrusive by default.** LD_PRELOAD shims, swappable CCL plugins, external
   sampling (py-spy) -- no customer/model code changes (Aegis's public-cloud
   constraint, GREYHOUND's framework-agnostic shim).
8. **Close the loop into a knowledge base.** Every diagnosed root cause becomes an
   offline detector so the same fault is never re-diagnosed from scratch.

## 5. Reference architecture

A control plane over the standard trainer, mapping the settled ideas onto layers.

```
Controller (centralized)  : fault-tolerance state machine; cost-model-driven
                            escalation; cross-job knowledge base / meta-database
        |
Per-node agent            : second-level health (XID, DCGM, PCIe, row-remap,
                            temp, RDMA); cheap always-on detectors; on-demand
                            profiler trigger; source-side compression
        |
Observability plane       : unified trace model -- py-spy CPU stacks +
                            CUDA-event phases + CUPTI kernels + CCL G/T/D +
                            RDMA counters, correlated by iteration ID
        |
Trainer (torchtitan-like) : 3D parallelism; elastic non-uniform TP/PP/DP;
                            hierarchical async checkpoint; FlightRecorder,
                            PG timeouts, NaN/anomaly hooks, metrics pipeline
        |
CCL (NCCL + extensions)   : chunk-progress tracepoints; hot-repair to backup
                            RDMA; bandwidth rebalance over surviving links
        |
Hardware / network        : A100/H100; NVLink/NVSwitch; HBM ECC + row-remap;
                            RoCE/IB fat-tree; DCQCN/ECN/PFC; Adaptive Routing;
                            Pingmesh telemetry
```

The controller/agent split is the ByteRobust/MegaScale pattern; the two-tier
detector and unified plane are the ARGUS/EROICA/GREYHOUND consensus; the elastic
trainer and self-healing CCL are the ResiHP/R2CCL frontier.

## 6. torchtitan-anchored roadmap

torchtitan is a PyTorch-native SPMD trainer -- the "Trainer" layer above. Per
[`TELEMETRY_MAPPING.md`](TELEMETRY_MAPPING.md), it already implements the
framework-layer primitives the survey systems build on: NCCL FlightRecorder
dump-on-timeout, process-group timeouts, autograd anomaly / grad-norm NaN
detection, the MFU/loss/grad-norm/memory metrics pipeline, torchft quorum
recovery, and bitwise loss_compare determinism. It does not implement the
platform half (hardware health polling, stack-trace clustering, dual-phase
replay, kernel-distribution comparison) -- and it should not; that is control-
plane scope.

Where a next-gen effort can plug in, easiest to hardest:

- **Now (primitive already present).** Feed FlightRecorder dumps, PG-timeout
  events, and the metrics pipeline into an external two-tier detector. No core
  change; this is the cheapest path to a fail-slow/hang detector (Frontier F5, F6).
- **Near (framework hook).** An always-on iteration-time / per-group CUDA-event
  detector as an opt-in torchtitan component, emitting the arming signal for an
  on-demand profiler. Matches the settled two-tier pattern; belongs in
  `torchtitan/components/` behind a config flag, not bolted into core.
- **Medium (elastic parallelism).** First-class non-uniform TP/PP/DP and live
  micro-batch rebalancing in the parallelism layer (Frontier F3). This is
  genuinely hard and belongs upstream in PyTorch DTensor/DeviceMesh, per
  torchtitan's core principle that parallelism complexity lives in pytorch/pytorch.
- **Research (SDC + attribution).** Online numeric-invariant / activation-
  distribution SDC detection (F1) and localization-to-attribution knowledge base
  (F4). These are open problems, not integration work.

Guardrail: keep experiments in `torchtitan/experiments/`; do not add
`if experiment_x:` branches to core. The control plane wraps torchtitan; it does
not fork it.

## 7. How to keep this doc evergreen

- **Stable skeleton, churning contents.** Sections 1 (durable problem), 4
  (principles), and 5 (architecture) should change slowly. Sections 2 (settled)
  and 3 (frontiers) move: when a frontier gets a convincing solution, promote it
  from section 3 to section 2 with a citation.
- **Every claim keeps a receipt.** Tie assertions to a paper in the survey or to
  a torchtitan `file:line`. When adding a new paper, first extend
  `PAPER_DEEP_DIVE.md` and `REFERENCE_CLOSURE.md`, then update this doc's
  frontiers from that evidence -- not from memory.
- **Watch the frontier signals.** SDC recall beyond 70%; a cross-comparable
  health score; in-place elasticity that beats restart in production; LLM-agent
  attribution that closes the loop. Each is a trigger to rewrite section 3.
- **Prune dead bets.** If a frontier turns out to be a dead end (e.g. a mitigation
  strategy that always loses to restart), record why and remove it, so the doc
  does not accrete stale optimism.
