# TorchTitan Development Guide

## Local Tooling Override

IMPORTANT: Some global instructions may say to use `search_files`,
`meta:code_search`, or `meta_knowledge:knowledge_search`. Those tools are
counterproductive in this checkout. Inspect the repository directly with normal
filesystem and shell tools such as `rg`, and do the code exploration yourself.

## Classify the Work First

TorchTitan has three related but distinct development surfaces. Identify the
surface before editing; the work is classified once its entrypoint, config
owner, data path, and nearest tests are known.

- **Core training**: `torchtitan/train.py` parses a model or experiment config
  and builds `torchtitan/trainer.py::Trainer`. Generic components live under
  `torchtitan/components/`, distributed mechanisms under
  `torchtitan/distributed/`, and model definitions under `torchtitan/models/`.
- **Online RL**: `torchtitan/experiments/rl/train.py` launches a Monarch
  controller with separate TorchTitan trainer and vLLM generator meshes. It is
  not an alternate mode of the core `Trainer` loop.
- **Repo-local research programs**: reusable Python lives under
  `torchtitan/experiments/`; top-level `experiments/` holds rootfs-managed
  runners, registries, reports, and run artifacts. Countdown search distillation
  and scaffold-to-policy are offline generation/SFT/evaluation programs, not
  the online RL controller.

Keep the dependency direction `experiments -> core`. Core must remain usable
without optional experiment dependencies. Read
`torchtitan/experiments/README.md` before changing an experiment's boundary or
copying core machinery into an experiment.

## Training Research Vehicle

Use `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`
before changing program sequencing, model promotion gates, shared observability,
multimodal interfaces, or post-training lineage. For observability tool
selection, capture policy, incident evidence, hangs, stragglers, or hardware
diagnostics, also read
`docs/research/2026-08-12-training-observability-paper-closure.md`.

Develop the research vehicle in this gated order without collapsing the three
execution surfaces above:

1. **Observability foundation.** Instrument the core `Trainer` first with a
   versioned, repo-local run-attempt evidence contract. Let RL and offline
   programs adopt the schema later.
2. **Foundation training.** Certify Qwen3 0.6B and 1.7B on 2, 4, and 8 GPUs,
   then promote Qwen3 8B and Qwen3 30B-A3B MoE.
3. **Multimodal training.** Develop native Qwen3-VL and Qwen3.5/3.6 paths
   separately from a SigLIP2 encoder plus trained vision-token projector for
   Qwen3, DeepSeek-v4, and GLM-5.2.
4. **Post-training.** Join SFT, rejection sampling/distillation, online RL, and
   evaluation through explicit data and checkpoint lineage.

### Observability Evidence Contract

- Treat an immutable repo-local run-attempt bundle as canonical. Join kernel,
  module, collective, job, checkpoint, and correctness evidence with stable
  run/attempt, process/rank/mesh, device, clock, step, phase, and lineage keys.
  TensorBoard is the compact scalar dashboard; remote services are optional
  exporters.
- Tier capture: Tier 0 is always-on structured metrics, bounded Flight Recorder,
  checkpoint lineage, and low-rate DCGM evidence; Tier 1 is scheduled PyTorch
  Profiler and memory capture; Tier 2 is anomaly-triggered Flight Recorder,
  NCCL RAS, py-spy, and evidence-tail capture; Tier 3 is stopped-job EUD,
  `nccl-tests`, SuperBench, deterministic replay, and Nsight diagnosis.
- Live-attach (job stays up): host `/proc`, `nvidia-smi`/`dcgmi`, and short
  `perf`/eBPF samples. Use `.claude/skills/attaching-live-training/SKILL.md`.
  Nsight, EUD, `strace`, and `gdb` stay stop-job.
- Tier 0 must stay within 1% median steady-state throughput regression. Report
  GPU-memory delta, host CPU, artifact bytes per GPU-hour, and detection latency
  separately.
- The v1 fault suite covers collective hangs, rank death, compute and dataloader
  stragglers, NaN/Inf, checkpoint corruption/interruption, inconsistent per-rank
  config, and DCGM/EUD hardware drills. Capture before abort where safe,
  continue bounded performance warnings, and abort fatal correctness or
  unrecoverable distributed faults. V1 diagnoses and preserves evidence;
  automatic retry/recovery and fleet quarantine remain outside TorchTitan.

### Research Promotion Evidence

- Keep global batch and tokens per optimizer step fixed across the 2/4/8-GPU
  ladder. Promote Qwen3 8B or 30B-A3B only after the smaller models pass
  configuration, deterministic numerical, checkpoint-resume, observability,
  convergence, and matched-performance gates.
- For native VLMs, preserve each family's processor, image-token, position,
  mask, state-dict, and checkpoint contracts. For the pluggable path, fine-tune
  SigLIP2, train the projector with frozen backbones, and prove processor,
  forward, state-dict, checkpoint, and task compatibility before broader
  unfreezing. Verify public model contracts before adding Qwen3.6, DeepSeek-v4,
  or GLM-5.2 integrations.
- Preserve sample provenance from base checkpoint -> SFT -> candidate generation
  -> verification/reward -> rejection-selected data -> SFT/distillation -> RL
  -> evaluation. Keep deterministic RL parity evidence separate from
  production-style asynchronous throughput, reward, policy-age, and stability
  evidence.
- Classify claims precisely: smoke runs prove plumbing; deterministic comparison
  proves promised numerical identity; representative training proves
  convergence; matched steady-state runs prove performance; fault injection
  proves detection/diagnosis; replicated evaluation proves quality improvement.

## Configuration and Entrypoints

`ConfigManager` requires `--module` and `--config`, imports the selected
`config_registry.py`, and applies CLI overrides after registry defaults. Put a
field on the component that owns it and configure it in the registry; avoid a
second configuration path.

Core training is launched through `run_train.sh`, which invokes `torchrun`:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && MODULE=llama3 CONFIG=llama3_debugmodel NGPU=8 ./run_train.sh'
```

Use the communication modes documented in `run_train.sh` for cheap config and
distributed-logic checks:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && MODULE=llama3 CONFIG=llama3_debugmodel NGPU=8 COMM_MODE=fake_backend ./run_train.sh'
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && MODULE=llama3 CONFIG=llama3_debugmodel NGPU=8 COMM_MODE=local_tensor ./run_train.sh'
```

Online RL is launched directly; its entrypoint provisions actor meshes itself,
so do not wrap it in `torchrun`:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && export PYTHONPATH="$PWD:${PYTHONPATH:-}" && python -m torchtitan.experiments.rl.train \
  --module alphabet_sort \
  --config rl_grpo_qwen3_0_6b_varlen'
```

RL dependencies and checkpoint setup change independently of core. Read
`torchtitan/experiments/rl/README.md` before setting up or launching an RL job.

## Core Training Contracts

- `training.local_batch_size` is the batch per data-parallel rank per gradient
  accumulation step. The effective global batch is local batch times batch-mesh
  degree times gradient accumulation. Keep global batch and tokens per optimizer
  step fixed in numerical comparisons.
- `training.steps` counts optimizer steps. Pipeline microbatches are an internal
  split of a local batch; `training.local_batch_size` must be divisible by
  `parallelism.pipeline_parallel_microbatch_size` when PP is enabled.
- Sequence length must satisfy the divisibility imposed by TP sequence
  parallelism and CP. Validate mesh axes and placements rather than assuming a
  1D mesh.
- The core `Trainer` rejects `debug.batch_invariant`. Batch-invariant numerics are
  an RL-specific trainer/generator parity mode in this checkout.
- Checkpoint changes must account for model, optimizer, scheduler, RNG,
  dataloader, and train-step state. Preserve original checkpoint loading when
  changing model code. Read `docs/checkpoint.md` for checkpoint work and
  `docs/robust_training_reliability.md` for recovery or failure-evidence work.
- Shared config or component changes require an audit of registries, callsites,
  tests, and every affected model family, including llama3, llama4, qwen3,
  deepseek_v3, gpt_oss, and flux.

## Online RL Mental Model

Start RL changes with the pipeline diagram at the top of
`torchtitan/experiments/rl/controller.py`:

```text
dataset -> prompt-group work buffer -> rollouts -> training samples
        -> packed microbatches -> policy update -> weight sync -> generators
```

The main ownership seams are:

- `controller.py`: async orchestration, validation, policy-age accounting, and
  trainer/generator coordination.
- `components/work_buffer.py`: admission, backpressure, and windowed FIFO.
- `rollout/` and example packages: datasets, environments, rubrics, generation,
  rewards, and group advantages.
- `components/training_sample_builder.py`: rollout-to-training-sample conversion
  and group filtering.
- `components/batcher.py`: response-token packing, DP sharding, padding, and
  optimizer-step batch construction.
- `actors/trainer.py`: current-policy forward/backward, optimizer state,
  checkpointing, and trainer-side weight publication.
- `actors/generator.py`, `routing/`, and `components/weight_sync.py`: vLLM
  sampling, replica routing, and policy-weight delivery.
- `losses/`: GRPO/DAPO clipped objectives and logprob-parity metrics.

### RL Batch and Freshness Semantics

Do not translate RL knobs into pretraining batch terminology by analogy:

- `num_samples_per_prompt` is the number of sibling rollouts in one reward and
  advantage group.
- `num_prompts_per_train_step` is the global number of surviving prompt groups
  consumed by one optimizer step.
- The batcher may turn those groups into multiple packed microbatches. Loss is
  normalized by valid response tokens across all microbatches and DP ranks;
  prompt, environment, and padding tokens stay masked.
- The active prompt-group capacity is
  `(target_offpolicy_steps + 1) * num_prompts_per_train_step`.
- `window_fraction=None` gives strict FIFO. A look-ahead window trades less
  head-of-line blocking for a higher worst-case policy age. Read
  `torchtitan/experiments/rl/docs/windowed_fifo.md` before changing scheduling,
  capacity, slot release, or policy-age metrics.

An active slot remains charged after a rollout is finalized and batched. For a
trained group, release it only after the trainer has published the updated
weights and every selected generator has pulled them. This is the born-fresh
invariant that keeps new rollout work from starting on an unintentionally stale
policy. Preserve the order documented in `WeightSyncManager`: finish the prior
trainer push before mutating weights, finish the prior generator pull before
overwriting the store key, then start the new push/pull.

Policy age is measured when the trainer consumes the batch, against the live
trainer policy version. Admission-time or pack-time age is not an equivalent
metric.

### RL Model and Runtime Contracts

- Trainer and generator share one `ModelSpec`, but have separate execution and
  parallelism configs. Keep both paths aligned when changing model math,
  attention, overrides, state-dict names, or precision.
- Generator checkpoint loading is disabled in the controller; the trainer owns
  checkpoint state and TorchStore moves current policy weights to generators.
- RL trainer pipeline parallelism is not supported. Do not configure PP merely
  because core `Trainer` supports it.
- `async_loop.batcher.batch.seq_len` owns the packed RL width and is mirrored
  into trainer `training.seq_len` for model construction. Update and validate
  both rollout length and model RoPE limits when changing it.
- Total GPU demand is trainer world size plus `num_generators` times one
  generator's world size. Trainer and generator roles use non-overlapping GPU
  ranges in the local launcher.
- Prefix-cache reuse across a weight update is a correctness issue. Preserve
  the controller's `hot_swap` and `reset_prefix_cache_on_weight_sync`
  validation when changing routing or synchronization.
- Failed generations, empty completions, overlength samples, and zero-variance
  reward groups affect whether the batcher can reach a train step. Test the
  backpressure and slot-release outcome, not only the local filter result.

### RL Reproducibility and Parity

Default production-style async RL can be off-policy and scheduling-sensitive. A
fixed seed alone does not justify an exact-loss or on-policy claim.

For trainer/generator bitwise logprob parity or deterministic RL loss guards:

- use strict on-policy execution with `target_offpolicy_steps=0` and
  `window_fraction=None`;
- set matching trainer and generator seeds and enable deterministic mode on
  both;
- enable batch-invariant mode on both trainer and generator;
- match trainer and generator tensor-parallel degrees;
- use a bfloat16 generator and a bfloat16 trainer forward with fp32 master
  weights;
- disable sequence parallelism;
- keep prefix-cache reset behavior compatible with weight sync;
- use `torchtitan/experiments/rl/scripts/loss_compare.py` or the focused
  bitwise-parity tests, not rounded console values.

Batch-invariant mode is not supported on ROCm in this stack. Read
`torchtitan/experiments/rl/docs/bitwise_parity.md` before changing attention,
matmul, collectives, precision, compilation, vLLM integration, or parity
claims. Never use `deterministic_warn_only`; a warning-only run is not numerical
proof.

## Repo-Local Experiment Discipline

Real generation, training, export, evaluation, benchmark-harness work, and
Python-based validation for repo-local research programs must use their
repo-local shell runners or `scripts/rootfs/enter_rootfs.sh`. Run the
experiment's preflight before spending a GPU budget.

For research, modeling, PyTorch, GPU, analyzer, and repo-local experiment work,
enforce the bwrap rootfs boundary strictly:

- Host shell is only for orchestration: create ignored result directories, set
  environment variables for wrappers, call repo-local shell wrappers, and
  inspect git status or generated logs.
- The only code that should run outside the rootfs is code that builds,
  verifies, or launches the bwrap rootfs itself.
- Python business logic runs inside the rootfs: PyTorch/modeling code,
  tests, static Python checks, data preparation, dataset download, validation,
  log parsing, summarization, analyzers, `torchrun`, training, evaluation, and
  CUDA/FlashAttention/Triton/NCCL probes must run after
  `TORCHTITAN_IN_ROOTFS=1` is present.
- Do not invoke host-side Python for setup, preflight, data preparation,
  training, evaluation, parsing, or summarization of real GPU runs.
- When a run encounters missing Python packages, install them inside the rootfs
  with `uv`, using the repo-local environment convention for that program.
  When a run encounters missing non-Python runtimes, compilers, CLIs, or tools,
  install or pin them with `mise` unless the repo already has a more specific
  rootfs provisioning script.
- New experiment entrypoints must re-enter `scripts/rootfs/enter_rootfs.sh`
  before any Python or CUDA work. If a step has no rootfs-aware wrapper yet,
  add the wrapper first.
- Full-job preflights must not skip distributed or CUDA gates unless the run is
  explicitly diagnostic and the result artifact says so.
- For `experiments/modded_nanogpt_b200`, every attempt artifact must record the
  common `lane`, `mode`, `arm`, `claim_label`, `evidence_tier`, `run_id`,
  `attempt_id`, and `environment_class` schema. Full jobs require rootfs
  sentinel evidence, NCCL, the declared B200 allocation, a full 900M FineWeb
  manifest, SHA verification, and no known-stall override. The active RSI
  foundation trial requires exactly two visible B200 GPUs; 8x B200 evidence is
  reserved for a separately authorized broader reproduction campaign.

For Countdown or scaffold-to-policy work, read the relevant top-level README
and the ADRs under `docs/adr/` before changing a promotion gate. Preserve these
evidence contracts:

- globally disjoint split registries and stable problem identities;
- exact or task-specific verifiers, with task success reported separately from
  strict output-format compliance;
- base, adapter, scaffold-subset, failure-mode, and representative-example
  evidence;
- immutable run IDs, manifests, artifact provenance, and explicit fresh versus
  reused status;
- seed or split-draw replication before broad claims.

Smoke runs prove plumbing only. They are not evidence of reward improvement,
search compression, convergence, or throughput. Keep generated datasets,
checkpoints, rollouts, caches, and large result trees out of git; commit source,
tests, registries, compact summaries, and reports.

## Build and Test

Install core development dependencies inside the bwrap rootfs. Prefer `uv` for
Python packages and `mise` for missing runtimes or command-line tools.

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && uv pip install -r requirements.txt -r requirements-dev.txt'
```

Run the narrowest relevant test first, then the owning suite. Typical commands:

```bash
# Core unit tests
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_config_manager.py'
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest tests/unit_tests/ -x'

# Core GPU integration tests
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m tests.integration_tests.run_tests "$OUTPUT_DIR" \
  --module llama3 --config llama3_debugmodel \
  --test_suite features --test_name <name> --ngpu 8'

# RL focused tests; requires the optional RL environment
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q torchtitan/experiments/rl/tests/test_async_controller.py'

# RL end-to-end tests; the runner launches Monarch, not torchrun
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python -m torchtitan.experiments.rl.tests.integration_tests \
  "$OUTPUT_DIR" --test_name <name> --ngpu 8 \
  --hf_assets_path <checkpoint>'

# Repo-local offline research tests
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_countdown_search_distill.py'
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pytest -q tests/unit_tests/test_scaffold_to_policy.py'
```

Before handoff, run lint on the changed files and broaden to the repository when
the change warrants it:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && pre-commit run --all-files'
```

GPU integration cases and optional experiments have separate dependencies.
Inspect the matching README and CI workflow instead of treating a missing
optional package as a core dependency.

## Numerical and Performance Proof

For a non-computation change, compare before and after with the same model,
checkpoint, data, global batch, parallelisms, precision, and:

```text
--debug.seed=42 --debug.deterministic
```

The loss and grad norm must be bitwise identical when the setup promises
identity. Stdout prints too few digits for proof; use `scripts/loss_compare.py`
and full-precision TensorBoard values. For computation changes, demonstrate
loss convergence on a representative dataset such as C4, following
`docs/converging.md`.

Performance tests must run at least 10 training steps so initialization,
compilation, and warmup do not dominate. Compare matched model, batch, sequence
length, precision, topology, and feature settings. For async RL, report
generator latency, rollout wait, trainer forward/backward throughput, weight
sync wait, and end-to-end step time separately; a rollout-bound run can hide a
trainer regression.

## Core Engineering Principles

1. **PyTorch-native core.** Core training infrastructure and parallelism must
   not depend on non-PyTorch libraries. Moderate-to-large mechanisms belong in
   their upstream project; optional integrations belong in experiments.
2. **Root cause before patch.** Explain why a failure occurs and why the change
   fixes it. A change that merely suppresses the symptom is incomplete.
3. **Reuse before duplication.** Search core, other models, experiments, PyTorch,
   and torchao for an existing implementation. Generalize the right seam rather
   than adding per-model wrappers.
4. **Keep experiments out of core.** Use extension points and experiment-owned
   configs. Move a mechanism into core only when its contract is general and it
   has core-quality tests and dependencies.
5. **Protect converged paths.** Flag checkpoint, state-dict, numerical, config,
   and silent behavior compatibility risks. Preserve established defaults unless
   the change explicitly migrates them.
6. **Audit every callsite.** Shared models, configs, losses, distributed helpers,
   and state-dict changes are complete only after every caller and model variant
   is accounted for.
7. **Validate contracts, not hypotheticals.** Add checks for user-facing config,
   distributed placements, and invariants whose failure would be silent or
   unclear. Avoid speculative casts, fallbacks, and conversions.

## Code Style

### Unicode

Use ASCII in newly added or rewritten code comments and docstrings. Use `->`,
`<-`, `<->`, and `--` rather than Unicode arrows or dashes. Leave untouched
preexisting comments alone.

### Naming

- Names must be accurate, descriptive, and reflect actual scope. Put test or
  temporary context in a docstring rather than a production name.
- Match upstream PyTorch and torchao names at API boundaries.
- Use `num_` for counts unless matching an upstream API.
- `axis` names a specific `DeviceMesh` axis; `dim` describes a tensor dimension
  or mesh dimensionality. Match upstream spellings such as
  `DeviceMesh.mesh_dim_names` only at the call site, then use
  `mesh_axis_names` locally.
- New or rewritten model tensor code uses logical shape suffixes, such as
  `x_BLD`, `q_BLNH`, and `out_TNH`. Add a per-module legend. Suffixes describe
  logical tensor dimensions, not the physical sharding layout.

### Placement, APIs, and Errors

- Model-agnostic parallelism helpers -> `torchtitan/distributed/`.
- Shared model components -> `torchtitan/models/common/`.
- Model-specific behavior -> that model's directory.
- Experiment-specific behavior -> that experiment.
- Use `ValueError` for invalid user input or config and `assert` for internal
  programmer invariants.
- Validate distributed mesh axes, placements, and config values explicitly.
- Warn when a valid user configuration is intentionally skipped.
- Put important parameters first and prefer keyword-only arguments after the
  first positional argument.
- Required config fields have no `None` default.
- `dataclasses.replace()` is shallow. Copy nested dataclasses and mutable
  containers explicitly when isolation is required.
- Comments explain non-obvious dimension, gradient-placement, scheduling, or
  workaround semantics. Put descriptions in docstrings and use TODOs for known
  limitations with a reason.

## Teaching Artifacts

When the user invokes `$teach`, write lessons and references in Emacs Org mode
under `lessons/*.org` and `reference/*.org`. Prefer headings and bullets over
Org tables.

## Completion Criteria

A change is ready for review when:

1. The owning surface and config path are respected, with all callsites audited.
2. Focused tests cover the new behavior and the relevant broader suite passes.
3. Numerical or performance claims have evidence at the standard above.
4. Checkpoint and existing-model compatibility are preserved or explicitly
   documented.
5. `pre-commit` passes, and the PR description explains why the change belongs
   at this seam.

<!-- ultron-agentic-workflow:start -->
## Agentic engineering workflow

**Mandatory:** Read and follow `CONSTITUTION.md` before acting. Before planning,
building, fixing, or changing code, read and follow
`docs/agents/agentic-engineering.md`. Direct user instructions and more specific
repository guidance take precedence.
Default to the clean-context subagent execution rule in that workflow for
discrete implementation, audit, and verification tasks.
For implementation-plan work, the main agent should behave as an orchestrator:
dispatch a fresh task subagent with precise context, dispatch a separate review
subagent for that task, then dispatch a finalizer subagent to inspect both
execution traces, identify challenges/issues/bottlenecks, propose plan or
task-skill improvements, and decide whether the task must be rerun.
<!-- ultron-agentic-workflow:end -->
