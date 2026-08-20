# TorchFT Crusoe Fault-Tolerant Llama Deep Dive

Date: 2026-08-20
Status: research note

Primary source:
[Fault Tolerant Llama: training with 2000 synthetic failures every ~15 seconds and no checkpoints on Crusoe L40S](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/)

Related first-party source:
[pytorch/torchft README](https://github.com/pytorch/torchft/blob/main/README.md)

## Executive Summary

The PyTorch blog post is a stress demonstration of TorchFT integrated with
TorchTitan. It does not claim that ordinary pretraining should discard durable
checkpoints. It shows that replicated-weight training can keep making progress
when failures are isolated to replica groups, healthy groups keep stepping, and
recovering groups reload model and optimizer state from a live peer instead of
forcing a full-job restart.

The important design lesson for this checkout is that fault tolerance is a
transaction and membership problem, not only a checkpoint-frequency problem:

```text
replica-group health
  + per-step quorum
  + conditional optimizer commit
  + live state transfer from a healthy peer
  + scheduler restart of failed groups
  + evidence that committed steps match the declared semantics
```

This matches the local direction already captured in the design docs:
TorchTitan should own training-semantic evidence and replayable run bundles,
while a launcher or platform owns fleet actions such as restarting jobs,
replacing machines, and choosing whether to quarantine hardware.

## What The Blog Demonstrated

The demo used TorchFT and TorchTitan to train a Llama 3 1B model on a Crusoe
cluster with 300 NVIDIA L40S GPUs. The cluster had 30 hosts with 10 GPUs each.
The selected topology was 30 replica groups, one host per replica group, and 10
workers per group. Within each replica group, standard sharded PyTorch training
continued to use normal distributed machinery; across groups, TorchFT handled
fault-tolerant replication and membership.

The article reports three runs:

- Run 1 injected one replica-group failure every 60 seconds for 1100 failures.
  It ran for a little over 19 hours, reached 6249 total steps, committed 5145
  steps, and measured 82.3 percent step efficiency. With an average step time
  around 11 seconds, the expected ideal under one failure per 60 seconds was
  about 5 successful steps out of every 6, or 83.3 percent.
- Run 2 injected failures randomly between 0 and 30 seconds, averaging one
  failure every 15 seconds, for 1015 failures. The article reports no
  unrecoverable errors, 18.9 of 30 replica groups healthy on average, 15.46
  second average step time, 268 committed steps out of the first 888 steps, 30.2
  percent step efficiency, and 13.4 percent training efficiency.
- Run 3 used semi-synchronous training, specifically DiLoCo, to reduce
  cross-replica synchronization frequency. The article reports about 4000
  tokens per second for DiLoCo compared with about 1200 tokens per second for
  regular HSDP2 on the ethernet-limited cluster.

The demo deliberately used an extreme failure rate. The article says ordinary
training jobs usually see mean time between failures in the tens of minutes to
hours, while this run injected failures every 15 seconds on average. That makes
the result valuable as an edge-case exerciser, especially because randomized
failure timing hits initialization, recovery, and steady-state cases.

## Mechanism

### Replica Group Is The Failure Unit

TorchFT works when the model has replicated state, as in DDP or HSDP. For
TorchTitan, this means launching multiple TorchTitan instances, each responsible
for one replica group. A failure in one group should not force every other
group to stop and reload the last persistent checkpoint.

The local README says the same thing: with TorchFT, TorchTitan launches multiple
replica groups, each as a separate TorchTitan instance, and each replica group
maintains a copy of the model weights. In the local two-group example, two
TorchTitan instances each manage four GPUs and communicate through TorchFT.
See `torchtitan/experiments/torchft/README.md:15` and
`torchtitan/experiments/torchft/README.md:23`.

### Lighthouse And Managers Own Membership

The article describes a global Lighthouse server and per-replica-group Managers.
Lighthouse tracks worker state and health through heartbeats. TorchFT's README
describes the same split: a Lighthouse server coordinates replica groups, while
each replica group has a Manager and fault-tolerance library, allowing
membership changes at training-step granularity.

In this checkout, `TorchFTManager` is the local wrapper around that external
interface. It chooses `torchft.ProcessGroupGloo`, `torchft.ProcessGroupNCCL`, or
`ProcessGroupTorchComms` from config, constructs `torchft.Manager`, sets
`min_replica_size`, and gives each replica a stable `torchtitan_ft_<id>` ID.
See `torchtitan/experiments/torchft/manager.py:36`,
`torchtitan/experiments/torchft/manager.py:90`, and
`torchtitan/experiments/torchft/manager.py:113`.

### Per-Step Training Is Treated Like A Transaction

The article uses database language deliberately: a training step is treated as a
distributed transaction. If the step is healthy, the optimizer commit is applied.
If a relevant error occurs, the step is rolled back by discarding gradients and
not applying the optimizer step.

TorchFT exposes this through optimizer and process-group wrappers. The article's
minimal loop starts quorum and possible recovery at `optimizer.zero_grad()`,
uses fault-tolerant process groups for gradient all-reduce, then conditionally
commits at `optimizer.step()`.

In this checkout, `TorchFTOptimizersContainer` wraps the normal
`OptimizersContainer` with `torchft.Optimizer`. It is careful to call the
TorchFT optimizer wrapper only once per train step per Manager, even when the
TorchTitan optimizer container holds multiple optimizers. See
`torchtitan/experiments/torchft/optimizer.py:26`,
`torchtitan/experiments/torchft/optimizer.py:45`, and
`torchtitan/experiments/torchft/optimizer.py:64`.

### FT-HSDP Hooks The Replicated Gradient Path

TorchFT's README states that FT-HSDP provides fault tolerance across the
replicated dimension while allowing any mix of FSDP, TP, and other parallelisms
inside the other dimensions. The article's diagram and text describe the same
shape: standard FSDP2 and other parallelisms run within a replica group, while
TorchFT's fault-tolerant DDP all-reduce synchronizes gradients across groups.

The local wrapper installs an FSDP all-reduce hook for async-quorum FT-HSDP.
`TorchFTManager.maybe_set_all_reduce_hook()` applies a hook to every FSDP module
and calls `dist.all_reduce(..., group=self.replicate_pg, op=ReduceOp.AVG)`.
See `torchtitan/experiments/torchft/manager.py:124` and
`torchtitan/experiments/torchft/manager.py:143`.

### Recovery Is Live State Transfer, Not Persistent Restart

The most important claim in the article is "no checkpoints" for the demo. In
context, this means the demo turned off persistent checkpoint save/load because
checkpoint IO would be much slower than peer-to-peer recovery under the failure
rate being tested. A recovering replica obtains model and optimizer state from a
healthy replica at runtime. The premise is that at least one replica group
remains healthy.

This does not eliminate the need for state transfer functions. The local
`TorchFTCheckpointManager` registers `state_dict` and `load_state_dict`
functions with the TorchFT manager for model, optimizer, scheduler, and
train-state transfer. See `torchtitan/experiments/torchft/checkpoint.py:114`.

It also does not mean this checkout has removed checkpointing from the TorchFT
path. The local checkpoint manager explicitly supports two checkpoint classes:
full persistent checkpoints saved by the participating rank 0 replica, and
per-replica dataloader checkpoints. See
`torchtitan/experiments/torchft/checkpoint.py:44`.

### Data Semantics Are Still A Real Contract

The article focuses on model/optimizer recovery. In production, data position
also matters. This checkout's TorchFT checkpoint manager saves per-replica
dataloader state every step when `enable_ft_dataloader_checkpoints` is enabled,
and warns that disabling it can make replicas train over the same data multiple
times. See `torchtitan/experiments/torchft/checkpoint.py:58` and
`torchtitan/experiments/torchft/checkpoint.py:143`.

That means "no checkpoints" is not a general data-lineage policy. For research
claims in this repo, we should record whether dataloader state is replayed,
skipped, duplicated, or deliberately ignored.

## Why Gloo Was Chosen In The Demo

The Crusoe cluster was ethernet/TCP-limited, with no InfiniBand or NVLink
between hosts. The article says it used NCCL within replica groups and Gloo
across replica groups. The reason was operational, not ideological: Gloo
initialized and failed fast enough for the experiment, reportedly
reinitializing in under one second for the use case and allowing a 5 second
operation timeout.

In this checkout, `TorchFTManager.Config.process_group` defaults to `gloo`, and
the implementation supports `gloo`, `nccl`, and `mccl`. The config comment says
the process-group timeout currently only works with Gloo. See
`torchtitan/experiments/torchft/manager.py:48` and
`torchtitan/experiments/torchft/manager.py:53`.

Design implication: a production evidence bundle must record the outer
fault-tolerance process group separately from the inner training process groups.
Changing outer Gloo to NCCL changes failure detection, reinitialization, and
timeout semantics, not just throughput.

## Efficiency Accounting

The blog uses several distinct efficiency ideas that should stay separate:

- Step efficiency: committed steps divided by total attempted steps.
- Participant efficiency: average participating replica groups divided by the
  configured replica-group count.
- Training efficiency: combined effect of successful commits and available
  participants.
- Throughput: tokens per second, which may improve under DiLoCo even when
  strict per-step equivalence no longer holds.

Those quantities answer different questions. Step efficiency answers how often
the transaction commits. Participant efficiency answers how much replica
capacity is actually active. Throughput answers how many tokens are processed.
Convergence or quality answers whether the resulting training trajectory is
acceptable.

The article also notes that loss spikes can appear when recovering workers with
out-of-date weights are included in loss averaging. In this checkout,
`FaultTolerantTrainer.train_step()` uses `ft_manager.loss_sync_pg` for
cross-replica loss aggregation in async-quorum mode. See
`torchtitan/experiments/torchft/trainer.py:457`.

Design implication: loss, global batch, valid token count, participant count,
and commit outcome must be logged together. A scalar loss curve alone can be
misleading when membership is changing.

## What The Article Does Not Prove

The result is strong evidence for TorchFT's recovery mechanics under synthetic
replica-group failures, but it leaves several production questions open:

- It does not prove silent data corruption detection. The injected failure is a
  killed Slurm job, not bad math with apparently healthy ranks.
- It does not prove every hardware or network failure mode. Gloo over ethernet
  was selected for fast failure and reinitialization in this cluster.
- It does not prove that disabling persistent checkpoints is a production policy
  for long training jobs. It proves that live peer recovery can dominate
  checkpoint restart under this experiment's assumptions.
- It does not prove exact equivalence to an unfailed baseline under variable
  participant count, especially for DiLoCo or LocalSGD.
- It does not prove scheduler correctness. The monitoring script relaunches
  missing replica groups; production needs durable launch, identity, and
  anti-duplication controls.
- It does not prove validation quality at scale. The article's public numbers
  are operational and convergence signals, not downstream benchmark claims.

## Mapping To This Checkout

### Existing Implementation Surfaces

- `torchtitan/experiments/torchft/README.md` documents the current user-facing
  TorchFT launch flow and environment variables.
- `torchtitan/experiments/torchft/manager.py` owns the local TorchFT Manager,
  process-group adapter, FSDP all-reduce hook, DP identity expansion, and
  semi-sync dispatch.
- `torchtitan/experiments/torchft/trainer.py` subclasses the core `Trainer`,
  adjusts dataloader DP identity using `ft_manager.get_dp_info()`, builds
  metrics with FT labels, wraps optimizer/checkpoint construction, and logs
  cross-replica losses through the FT process group.
- `torchtitan/experiments/torchft/optimizer.py` owns the TorchFT optimizer
  wrapper integration and the once-per-step dispatch guard.
- `torchtitan/experiments/torchft/checkpoint.py` owns full-checkpoint
  participant gating, per-replica dataloader checkpointing, and TorchFT
  state-dict registration for live recovery.
- `torchtitan/experiments/torchft/tests/integration_tests.py` has an 8-GPU
  integration path, but currently uses a single replica group and records a
  pending two-replica shape because a peer-access issue blocked it.

### Current Gaps For Production-Grade Claims

1. Replica lifecycle evidence is not yet a first-class run-attempt artifact.
   The article's result depends on knowing which replica groups were alive,
   missing, recovering, participating, or committing at each step.

2. The local integration does not yet encode a reusable failure-injection suite.
   We need deterministic tests for killed replica group, failed initialization,
   timeout during gradient sync, recovery while peers continue, scale down, and
   scale up.

3. Commit/rollback evidence is not joined with loss and token accounting. The
   article's efficiency math requires committed-step count, attempted-step
   count, participant count, and valid-token count in the same evidence stream.

4. Dataloader replay semantics need explicit claim labels. The local code can
   checkpoint per-replica dataloader state, but a no-checkpoint peer-recovery
   claim must still say whether data replay is acceptable, bounded, or measured.

5. The scheduler boundary is not formalized. The blog's TorchX/Slurm monitor is
   a plausible reference, but production needs durable replica identity,
   idempotent relaunch, duplicate prevention, and launcher event ingestion.

6. Outer process-group choice must be claim-bearing. Gloo, NCCL, and MCCL have
   different timeout and reconfiguration behavior, so evidence from one should
   not automatically certify another.

7. Semi-sync quality needs a separate ladder. DiLoCo improves throughput on this
   network-bound workload, but it changes optimization semantics by synchronizing
   every N steps rather than every step.

## Recommended Design Additions

### Fault-Tolerant Run Evidence

Add a fault-tolerant step stream to the existing run evidence model:

```text
ft_step_event {
  run_id
  attempt_id
  replica_id
  local_rank
  global_step
  membership_epoch
  participant_count
  min_replica_size
  attempted_step
  should_commit
  commit_status
  rollback_reason
  outer_process_group
  inner_process_groups
  valid_tokens
  loss_numerator
  loss_denominator
  state_transfer_status
}
```

This is the missing bridge between the article's operational story and
TorchTitan's production evidence contract.

### Replica Lifecycle Stream

Record replica-group state transitions:

```text
missing -> launching -> initializing -> joined_quorum -> participating
        -> failed -> recovering -> state_transferred -> participating
```

Each transition should carry scheduler job identity, TorchFT replica ID,
Lighthouse endpoint, process-group type, rank range, rootfs identity, and
terminal reason.

### Failure Injection Matrix

A minimal matrix should cover:

- kill a whole replica group before first quorum;
- kill a group during forward/backward;
- kill a group between gradient sync and optimizer commit;
- kill a group during live state transfer;
- restart with the same replica ID;
- attempt duplicate launch of an already-live replica ID;
- scale down with `min_replica_size` still satisfied;
- scale below `min_replica_size` and require a blocked or aborted outcome;
- compare Gloo and NCCL/MCCL behavior only under separate claim labels.

### Efficiency Report

The report should calculate:

```text
step_efficiency = committed_steps / attempted_steps
participant_efficiency = average_participants / configured_replica_groups
training_efficiency = step_efficiency * participant_efficiency
tokens_per_second_committed
tokens_per_second_attempted
lost_work_steps_by_reason
recovery_latency_by_phase
```

This keeps the article's 82.3 percent and 13.4 percent style of reasoning
replayable from local artifacts.

### Checkpoint Policy Labels

Use explicit labels instead of a boolean:

- `persistent_checkpoint_required`: ordinary durable training proof.
- `peer_recovery_only_demo`: no persistent checkpoint; at least one healthy
  replica group required.
- `peer_recovery_plus_dataloader_ft`: model/optimizer live peer recovery with
  per-replica dataloader state.
- `unsafe_data_replay_allowed`: only for synthetic or intentionally replayable
  data.

### Promotion Gate

Do not promote a TorchFT recipe from smoke evidence. Require:

- repeated failure-injection attempts with independent attempt IDs;
- a matched no-failure baseline;
- identical model, data, precision, optimizer, schedule, and topology except
  for the failure/recovery variable;
- committed-step, participant, token, and loss evidence;
- checkpoint or explicit no-checkpoint policy evidence;
- process-group and timeout evidence;
- scheduler event evidence;
- rootfs identity and package state for every replica group;
- convergence comparison for the declared workload;
- downstream quality only after convergence is established.

## Concrete Next Tickets

1. Add `ft_step_event` and `replica_lifecycle_event` schemas to the shared
   observability or experiment-execution evidence layer.
2. Extend `FaultTolerantTrainer` metrics to log participant count,
   membership epoch, attempted step, commit status, and rollback reason.
3. Add a launcher-owned replica monitor spec that records TorchX/Slurm or local
   process events into the same attempt bundle.
4. Convert the existing 8-GPU TorchFT integration test from a single-replica
   placeholder into a two-replica local test once the peer-access blocker is
   understood and either fixed or explicitly skipped with evidence.
5. Add a synthetic failure-injection runner that can kill one replica group at
   deterministic train phases and validate that healthy groups continue.
6. Add a no-checkpoint demo profile, but keep it separate from durable
   production-training profiles.
7. Add an efficiency reducer that computes step, participant, and training
   efficiency from event streams rather than from manually read console logs.

## Bottom Line

The blog is a strong proof that per-step fault tolerance can turn frequent
replica-group crashes from a stop-the-world checkpoint restart problem into a
membership, transaction, and live-recovery problem. For this repo, the next
production-grade move is not to copy the demo exactly. It is to make TorchFT
membership, commit/rollback, peer recovery, scheduler restart, data replay, and
efficiency accounting part of the same immutable run-attempt evidence contract
already being built for TorchTitan research programs.
