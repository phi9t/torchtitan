# Robotics-Inspired State Estimation for Large Training Jobs

Date: 2026-08-18
Status: Research design note

## Summary

A large training job can be treated as a partially observed dynamical system
whose "body" is a multiplex graph of GPUs, hosts, links, process groups, model
shards, data assignments, and control services.

The useful robotics analogy is not "apply one Kalman filter." The useful
formulation is:

```text
analytical execution and hardware model
  + hybrid factor-graph smoother
  + learned graph-temporal observation and residual models
  + change-point and fault-mode inference
  + active diagnostic probing
  + hard safety and transaction invariants
```

The estimator should produce a belief state:

```text
b_t(x, z, F) = p(x_t, z_t, F_t | Y_{<=t})
```

where `x` is continuous health and capacity state, `z` is discrete operating or
fault mode, and `F` is the causal fault variable.

## Why Training Is Harder Than Ordinary Localization

Training jobs have:

- latent state distributed over thousands to tens of thousands of components;
- continuous states such as temperature, bandwidth, service rate, and queue
  depth;
- discrete modes such as healthy, degraded, desynchronized, failed, and
  recovering;
- observation clocks ranging from microsecond kernel events to multi-minute
  thermal trends;
- changing topology as ranks, replicas, process groups, sequences, or experts
  are reassigned;
- transactional state, where a speculative optimizer step may or may not be
  allowed to commit;
- endogenous observations, since GPU utilization depends on workload assignment
  and on whether the rank is waiting for another rank; and
- active probes such as canary collectives, replay on another GPU, rerouting, or
  precision changes.

The estimator therefore needs hierarchy, asynchronous smoothing, topology
awareness, robust likelihoods, and hard invariants.

## Multiplex System Graph

Represent the job as a time-varying graph:

```text
G_t = (V, E_phys, E_logical_t, E_control_t)
```

The physical graph includes GPU-to-HBM, GPU-to-NVLink, GPU-to-PCIe, GPU-to-NIC,
NIC-to-switch, host-to-storage, and host-to-power or cooling relationships. It
changes relatively slowly.

The logical graph includes DP, FSDP, HSDP, TP, PP, CP, and EP process groups;
microbatch dependencies; sequence-to-rank assignments; and expert ownership.
It may change at topology epochs and, for dynamic context or MoE workloads, at
batch boundaries.

The control graph includes scheduler ownership, supervision links, heartbeat
links, replica membership, checkpoint ownership, and recovery coordinators.

Keeping the graphs distinct matters. A logical TP neighbor may share an
NVSwitch, while an outer fault-tolerant replica peer may live in a different
failure domain.

## Latent State

For each GPU or compute resource, maintain a compact continuous state:

```text
x_gpu[v] = {
  tc_service_rate,
  cuda_service_rate,
  hbm_bandwidth,
  thermal_state,
  power_or_throttle_state,
  memory_pressure,
  intrinsic_health
}
```

Service rates and bandwidths should usually be represented in normalized or log
space because multiplicative degradation becomes additive.

For each network edge or path:

```text
x_net[e] = {
  effective_bandwidth,
  base_latency,
  queue_state,
  retransmission_or_loss_propensity,
  path_health
}
```

For each rank or process:

```text
x_rank[r] = {
  liveness_and_progress,
  cpu_scheduling_delay,
  dataloader_delay,
  allocator_pressure,
  runtime_queue_state
}
```

At the job level:

```text
x_job[k] = {
  training_phase,
  synchronization_or_pipeline_state,
  workload_composition,
  checkpoint_and_transaction_freshness
}
```

Discrete modes should include per-resource states such as healthy, transient,
degraded, failed, and recovering, plus causal fault families:

```text
nominal
compute_degradation
network_degradation
host_or_data_stall
collective_desynchronization
memory_fault
numerical_corruption
storage_fault
control_plane_fault
planned_intervention
```

Planned intervention must be first class. A new kernel, longer context, changed
expert routing, or intentional checkpoint rollback may legitimately shift the
sensor distribution.

Fault durations are not memoryless. Link flaps, thermal degradation, software
bugs, and recovery windows have different duration distributions. A hidden
semi-Markov model, interacting multiple model estimator, or
Rao-Blackwellized particle filter is more appropriate than a plain HMM when
duration matters.

## Transactional State

Training has state that ordinary robotics estimators rarely represent:

```text
x_txn = {
  committed_step,
  speculative_step,
  replica_epoch,
  process_group_epoch,
  latest_recoverable_checkpoint,
  data_cursor
}
```

The estimator must know whether an anomaly threatens only performance or may
invalidate committed model state. TorchFT makes this distinction explicit with
`should_commit()` after backward and before the optimizer step; every rank in a
replica receives the same commit decision.

The learned system must not decide whether optimizer state, checkpoint state,
or data cursor state is valid. Those remain explicit safety invariants.

## Multi-Timescale Dynamics

Use at least three timescales:

- fast state: kernel launch/completion, stream dependencies, collective chunk
  progress, RDMA work requests, queue depth, NIC completion, and clock
  fluctuation;
- step-level state: forward/backward duration, pipeline bubbles, per-rank
  arrival skew, tokens per second, checkpoint staging, and optimizer duration;
- slow state: thermal baseline, recurrent ECC, device degradation, persistent
  congestion, storage wear, and software-version effects.

A useful hierarchical decomposition is:

```text
x[v,t] =
  x_global[t]
  + x_site[site(v),t]
  + x_rack[rack(v),t]
  + x_host[host(v),t]
  + x_residual[v,t]
```

This distinguishes one bad GPU, one bad host, rack cooling, fabric congestion,
and global software regressions.

## Execution Is Max-Plus

Synchronization creates max operations. For operation `o` in step `k`:

```text
start_time[o,k] = max(end_time[p,k] for p in predecessors(o))
end_time[o,k] = start_time[o,k] + duration[o,k]
duration[o,k] = D_o(work[o,k], resource_state[o,k], mode[k]) + noise
```

For compute kernels, a roofline-inspired approximation is:

```text
duration ~= max(FLOPs / compute_rate, bytes / HBM_bandwidth)
            + launch_delay
            + queue_delay
```

For a ring all-reduce of `M` bytes over `n` ranks:

```text
T_ring ~= 2 * (n - 1) * latency
          + 2 * (n - 1) / n * M / effective_bandwidth
```

Observed collective duration is more accurately:

```text
T_collective ~= arrival_skew + network_progress_time
arrival_skew = max(rank_arrival_time) - min(rank_arrival_time)
```

This is the core ambiguity: a long collective can be caused by a network
slowdown or by one rank arriving late after slow computation or data loading.

## Observation Model

Each sensor observation should retain:

```text
event_time
ingestion_time
sensor_id
entity_id
logical_role
physical_identity
topology_epoch
process_epoch
step_or_microbatch
measurement
quality_flag
sampling_policy
```

Scalar and vector measurements include temperature, clock rate, PCIe replay
count, MFU, step duration, tokens per expert, and queue depth. Likelihoods may
be Gaussian, Student-t, lognormal, or count-based depending on the metric.

Event observations include Xid, process exit, link flap, row remapping, and
watchdog timeout. These fit marked point-process models where event intensity
depends on state and mode.

Categorical observations include collective state, process state, checkpoint
status, and DCGM health result.

High-volume traces should not be fused globally as raw events. Retain local
rings and feed the online estimator compressed summaries such as:

- kernel-duration histograms;
- kernel-count sketches;
- phase distributions;
- frequent call-stack signatures;
- collective progress summaries;
- quantiles and tail statistics.

Freeze the full local ring during incidents.

## Estimation Target

The online system estimates:

```text
p(x_t, z_t, F_t | Y_<=t, U_<t, G_<=t, W_<=t)
```

For diagnosis, filtering is insufficient because late observations and symptom
propagation matter. Use a fixed-lag smoother:

```text
p(x_[t-L:t], z_[t-L:t], F_[t-L:t] | Y_<=t)
```

Outputs should include:

- continuous state estimate and covariance;
- discrete mode probabilities;
- root-cause posterior;
- probability of failure within a horizon;
- probability the current step is unsafe to commit;
- expected performance under continued operation;
- predicted effects of candidate interventions;
- uncertainty and observability warnings.

## Observability and Identifiability

Some states are not distinguishable from available sensors. For a locally
linearized model, an information matrix such as `sum(H_i^T R_i^-1 H_i)` exposes
poorly observable state directions through small eigenvalues.

Example ambiguity:

```text
rank arrived late
```

and:

```text
rank arrived normally, network transfer stalled
```

can look identical from step time alone. Flight Recorder, collective launch
counters, flow-level tracing, CPU stack snapshots, and preceding kernel traces
make the hypotheses separable.

A useful operational output is:

```text
fault_posterior:
  host_or_data_stall: 0.46
  gpu_compute_stall: 0.37
  network_stall: 0.14
  other: 0.03

observability_warning:
  no gpu-start timestamp from rank 317
  no flow-level trace for channel 4

best_next_probe:
  capture GPU event state and run one local compute canary
```

## Classical Design: Hybrid Incremental Factor-Graph Smoother

The classical baseline should be a sparse, distributed factor graph with local
filters and discrete-mode inference, not a monolithic EKF.

At semantic anchor points:

```text
p(X,Z | Y,U) proportional_to
  p(x_0,z_0)
  * product_k p(z_k | z_{k-1})
  * product_k p(x_k | x_{k-1}, z_k, u_k)
  * product_i p(y_i | x_tau_i, z_tau_i)
```

Hard and soft factors encode:

- topology constraints;
- synchronization constraints;
- transaction constraints;
- checkpoint lineage constraints;
- process-group and membership-epoch constraints.

Create variables at phase boundaries, collective launches, training-step
boundaries, checkpoint transitions, topology changes, and anomaly-triggered
timestamps. Do not create state nodes every millisecond for every GPU.

Each host or device can run a local robust EKF, UKF, information filter, or
simpler robust estimator to summarize compute capacity, HBM capacity, thermal
state, memory pressure, local network service, and sensor health.

For Gaussian summaries, information form is convenient:

```text
Lambda = P^-1
eta = P^-1 * x_hat
sensor contribution:
  Delta_Lambda = H^T R^-1 H
  Delta_eta = H^T R^-1 y
```

Avoid double-counting correlated estimates. If cross-correlation is unknown,
use conservative fusion such as covariance intersection.

Telemetry is heavy-tailed, so robust likelihoods are required. Student-t or
contaminated Gaussian models prevent a single delayed log, JIT window, or
sensor glitch from dominating the posterior.

Sensors themselves have health modes:

```text
healthy
delayed
biased
stuck
missing
```

Sensor-health state avoids blaming a GPU because one DCGM field, trace agent,
or clock source is faulty.

## Change Detection and Root Cause

Use innovation residuals and normalized innovation squared for single-window
surprises. Use CUSUM or likelihood-ratio accumulation for weak persistent
degradation such as thermal throttling or bandwidth decay.

For local root-cause inversion around a nominal point:

```text
residual ~= fault_to_symptom_jacobian * sparse_fault_vector + noise
```

Estimate a sparse fault vector under topology constraints. This yields an
explainable root-cause ranking:

- a failed NIC affects flows through that NIC;
- a slow PP stage affects downstream waits;
- a software desynchronization affects process-group sequence;
- a rack event affects colocated hosts.

## Active Diagnosis

The estimator should recommend probes by expected information gain minus cost
and risk:

```text
best_probe = argmax_a I(F; Y^a | Y) - lambda * cost(a) - gamma * risk(a)
```

Candidate probes:

- replay same microbatch on another GPU to distinguish compute fault from
  workload skew;
- replay same data under different placement to distinguish data fault from
  hardware fault;
- run local compute canary and isolated collective to separate network fault
  from late arrival;
- change rail or network plane to separate NIC from path or switch fault;
- replay in BF16 or FP32 accumulation to distinguish FP8 instability from
  hardware SDC;
- run prior code on the same allocation to separate software regression from
  hardware;
- query an independent sensor to separate sensor fault from resource fault.

The best probe is the cheapest safe action that most reduces posterior
ambiguity. It is not necessarily the deepest profiler.

## Learned Components

The production target should not be purely neural. Learned pieces should start
as observation and residual models inside a classical scaffold.

Useful learned components:

- graph-temporal probabilistic state-space model conditioned on known physical,
  logical, and control graphs;
- continuous-time encoders for irregular observations, such as Neural CDE-style
  models;
- multiplex graph message passing over physical, logical, and control edges;
- modality-specific encoders for scalar telemetry, event logs, kernel traces,
  communication traces, and framework semantics;
- prediction heads for continuous state, fault mode, root cause, failure
  hazard, forecasts, and action effects;
- KalmanNet-style learned gains or learned residual dynamics;
- learned covariance and likelihood factors.

Do not ask the model to memorize raw kernel duration. Condition on expected
work and topology and learn normalized residuals:

```text
residual = (observed_duration - analytic_expected_duration) / expected_sigma
```

Learned factors should not override hard invariants for transaction safety,
checkpoint validity, process-group membership, or numerical-corruption policy.

## Production Architecture

Deploy five layers:

1. Local reflex estimators for GPUs, hosts, NICs, and ranks. These run robust
   scalar filters, heartbeats, CUSUM, local event rings, and sensor-health
   estimation.
2. Parallel-group estimators for TP, PP, CP, EP, FSDP, and HSDP groups. These
   track expected work, arrival skew, collective sequence, group critical path,
   membership, and rank equivalence.
3. Job-level fixed-lag factor graph for the last several minutes or last
   several hundred steps. It inserts delayed observations at event time,
   fuses physical and logical graphs, infers fault modes, and estimates
   checkpoint and transaction risk.
4. Learned residual and observation system that provides trace embeddings,
   adaptive likelihoods, residual dynamics, adaptive covariance, root-cause
   priors, and failure hazard.
5. Active diagnosis and decision support that computes posterior causes,
   information gain of probes, counterfactual outcomes, and uncertainty-aware
   recommendations.

Hard invariants sit outside learned systems.

## Worked Example: Slow All-Reduce

When a DP all-reduce becomes 2.8x slower, hypotheses include:

- late compute arrival;
- host or data stall;
- NIC degradation;
- switch or path congestion;
- collective desynchronization;
- normal workload skew.

If Flight Recorder shows rank 127 scheduled 610 ms later, launch counters show
late launch but normal completion after launch, flow traces show expected
bandwidth once launched, ARGUS-like traces show extra dataloader wait on rank
127, and DCGM shows no hardware anomaly, the posterior should favor host/data
stall rather than network fault.

The recommended action is not replacing the GPU or NIC. It may be to invalidate
the speculative step if required, restart the dataloader worker, preserve the
allocation, capture host CPU/I/O evidence, and escalate only if recurrence
crosses threshold.

## Worked Example: Network Versus Upstream GPU Fault

If all ranks enter a collective at similar times but one flow stalls, and:

- Flight Recorder shows the same operation, shape, dtype, and sequence on all
  ranks and all operations GPU-started;
- communication tracing shows source data ready but RDMA transmit not
  advancing;
- NIC telemetry shows increasing link errors on one port; and
- GPU kernels and clocks are normal;

then the posterior should strongly favor NIC or port fault.

Actions can include recommendation to fence the replica or process group,
switch rail or network plane where supported, rebuild communicator under a new
process-group epoch, quarantine the port or host, then validate recovery.

## Training Learned Estimators

True latent health state is rarely labeled. Use five data sources:

- nominal self-supervision: masked sensor reconstruction, next-event
  prediction, phase-duration forecasting, cross-modal prediction, and
  graph-neighbor prediction;
- controlled fault injection: GPU clock cap, CPU sleep, dataloader pause,
  packet loss, bandwidth throttling, QP failure, missing collective, mismatched
  shape, storage delay, corrupted FP8 scale, rank crash, stale checkpoint
  writer;
- digital-twin simulation with operation DAG, topology, queues, synchronization,
  fault propagation, checkpoint state, and membership changes;
- historical incident labels, using intervention outcome as stronger evidence
  than initial ticket labels; and
- intervention records: pre-action posterior, action, post-action state,
  recovery, and recurrence.

Evaluation must separately measure anomaly, fault mode, root cause, and action
success calibration.

## Evaluation Protocol

State-estimation metrics:

- RMSE where injected ground truth exists;
- credible-interval coverage;
- state negative log likelihood.

Detection metrics:

- false alerts per 1,000 GPU-hours;
- time to detection;
- detection probability by severity;
- missed low-and-slow degradation;
- alert persistence.

Localization metrics:

- root-cause top-1 and top-k;
- correct failure-domain rate;
- physical/logical scope intersection over union;
- causal-chain accuracy.

Calibration metrics:

- Brier score;
- expected calibration error;
- reliability diagrams;
- credible-interval coverage.

System utility:

```text
validated_committed_tokens / allocated_gpu_seconds
```

also tracking healthy GPU-seconds unnecessarily interrupted, false eviction
rate, operator minutes, state lost per incident, and SDC escape rate.

## Development Sequence

1. Analytical and data substrate: unified event identity, topology graph,
   phase and transaction markers, fixed-lag telemetry store, local robust
   filters, and analytical compute/collective duration models.
2. Classical factor-graph estimator: asynchronous measurement factors,
   device/link/process latent state, IMM or semi-Markov modes, robust
   likelihoods, sparse root-cause inference, and active diagnostic selection.
3. Learned observation factors: encoders for kernel trace, call stack,
   communication-flow trace, hardware event sequences, and model-semantic
   metrics.
4. Learned residual dynamics and covariance: learned gains, learned `Q/R`, and
   workload-conditioned predictions.
5. Counterfactual action model trained on injections, recoveries, replays,
   resource replacements, and topology changes.
6. Bounded automation only after transaction state is known, actions are
   reversible, posterior confidence is calibrated, safety invariants are
   satisfied, and post-action validation is defined.

## References

- Interacting Multiple Model Filter and Smoother:
  https://www.mdpi.com/1424-8220/21/12/4164
- TorchFT manager and commit semantics:
  https://meta-pytorch.org/torchft/manager.html
- PyTorch Flight Recorder:
  https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/
- ARGUS:
  https://arxiv.org/abs/2606.20374
- iSAM2 and factor-graph smoothing:
  https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html
- Robust Bayesian filtering with Student-t distributions:
  https://arxiv.org/abs/1703.02428
- Neural Controlled Differential Equations:
  https://papers.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html
- Graph Deviation Network:
  https://aaai.org/papers/04027-graph-neural-network-based-anomaly-detection-in-multivariate-time-series/
- KalmanNet:
  https://weizmann.elsevierpure.com/en/publications/kalmannet-neural-network-aided-kalman-filtering-for-partially-kno/
- Mycroft:
  https://arxiv.org/abs/2509.03018
- iSAM2 paper:
  https://journals.sagepub.com/doi/10.1177/0278364911430419
