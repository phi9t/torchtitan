# Qwen Certification Design

Date: 2026-08-20
Status: draft design

## Purpose

Fortify the Qwen-related work into a repeatable certification ladder for
foundation training and downstream policy experiments. The goal is to prevent
successful launches, smoke tests, calibration probes, and replicated quality
claims from being treated as the same kind of evidence.

## Current State

Qwen work appears in several places:

- `experiments/qwen3_fineweb_hsdp_tp/`: hermetic local Qwen3 debug-model
  training on a FineWeb subset with HSDP plus TP and introspection scripts.
- `torchtitan/models/qwen3/`: model config and registry surface used by core
  training.
- `experiments/countdown_search_distill/`: Qwen3-1.7B LoRA SFT and evaluation
  path for Countdown search distillation.
- `experiments/scaffold_to_policy/`: Qwen3-1.7B/vLLM evaluation and policy
  experiments over reasoning, coding, and agentic benchmarks.
- `torchtitan/experiments/rl/`: online RL recipes using Qwen-family models in a
  Monarch-controlled trainer/generator architecture.

The validated local training pattern uses the normal TorchTitan config path,
local data registration, a rootfs-managed runtime, and an 8-GPU mesh such as
`dp_replicate=2`, `dp_shard=2`, `tp=2`.

## Certification Ladder

### C0: Static Contract

Evidence:

- model config resolves through `ConfigManager`;
- dataset resolves through the `DATASETS` registry or a documented local data
  adapter;
- mesh dimensions multiply to world size;
- sequence length satisfies TP, CP, and PP constraints;
- checkpoint policy is explicit;
- rootfs and package environment are recorded for real runs;
- dataset acquisition, import, tokenization, manifest generation, and validation
  are declared as rootfs-required stages for claim-bearing runs.

Claim supported: the configuration is well formed.

### C0.5: Data Acquisition And Import

Evidence:

- download or import command executed inside `scripts/rootfs/enter_rootfs.sh`;
- dataset source, revision, filters, shard list, and local root are recorded;
- generated manifests include digests or documented external identities;
- tokenizer and preprocessing versions are pinned in the attempt evidence;
- reused local data is registered as reused or imported with freshness and
  digest metadata.

Claim supported: the declared data source is available to the certification
run through a reproducible rootfs-managed path. Host-side helpers such as a
direct `python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py` invocation
are developer conveniences only; they are deprecated for claim-bearing Qwen
certification evidence unless wrapped by the shared rootfs execution substrate.

### C1: Smoke

Evidence:

- tiny run launches;
- model builds;
- dataloader produces batches;
- forward/backward/optimizer step executes;
- artifacts and logs are written.

Claim supported: plumbing works. No convergence or quality claim.

### C2: Numerical Contract

Evidence:

- deterministic seed and deterministic mode are configured where the comparison
  promises identity;
- global batch and tokens per optimizer step are held fixed;
- loss and grad norm are compared with full-precision artifacts, not rounded
  stdout;
- expected non-identity cases are explicitly labeled, for example different
  distributed settings that change RNG offsets or accumulation order.

Claim supported: the promised numerical relationship holds.

### C3: Checkpoint Contract

Evidence:

- model, optimizer, scheduler, RNG, dataloader, and train-step state are saved;
- resume reproduces the expected continuation behavior;
- checkpoint lineage is present in attempt evidence;
- model state-dict compatibility is tested for the affected family.

Claim supported: training can be interrupted and resumed at the declared level.

### C4: Convergence

Evidence:

- representative data path;
- enough steps to observe the declared convergence signal;
- matched config, batch, precision, and schedule against the baseline;
- TensorBoard or structured metrics preserved.

Claim supported: the model trains as intended on the declared workload.

### C5: Matched Performance

Evidence:

- at least 10 measured steps after initialization and warmup;
- matched model, batch, sequence length, precision, topology, and feature flags;
- step time, MFU, memory, data loading, and compile/warmup are separated;
- rootfs and host hardware identity are recorded.

Claim supported: the run meets or misses a performance target under matched
conditions.

### C6: Downstream Quality

Evidence:

- frozen split identities;
- verifier versions;
- base and candidate policy provenance;
- prompt or chat-template contract;
- paired comparisons and replication when the claim covers more than one fixed
  run.

Claim supported: the declared downstream policy improved under the declared
scope.

## Production Target

Qwen certification should be a module with this interface:

```text
declare_qwen_certification(model, data, mesh, checkpoint, claim)
  -> run or import evidence
  -> validate certification ladder
  -> emit promotion evaluation
```

It should not be a collection of one-off README instructions. The implementation
can still be split across core config registries, experiment runners, and
report scripts, but the proof vocabulary must be shared.

## Module Interfaces

### QwenRunSpec

Owns:

- model family and size;
- tokenizer and checkpoint source;
- data manifest;
- sequence length;
- batch and gradient accumulation;
- parallelism mesh;
- precision and compile flags;
- checkpoint policy;
- claim type.

Hardening requirements:

- validate mesh math before launch;
- record global batch and tokens per optimizer step;
- make local dataset roots explicit and content-addressed when possible;
- require rootfs-managed data acquisition or import evidence before real
  certification stages consume a dataset;
- reject missing checkpoint policy when checkpoint evidence is required.

### QwenEvidenceBundle

Owns:

- config snapshot;
- dataset manifest;
- checkpoint lineage;
- training logs and metrics;
- loss comparison artifacts;
- performance timing;
- observability artifacts;
- terminal outcome.

Hardening requirements:

- use shared attempt and artifact references;
- preserve raw evidence so summaries can be regenerated;
- distinguish missing, uncollected, failed, and not-applicable evidence.

### PromotionEvaluator

Owns:

- required ladder level for a requested claim;
- evidence completeness checks;
- hold/reject/promote recommendation;
- human-readable missing-evidence report.

Hardening requirements:

- never promote from smoke evidence alone;
- treat incompatible mesh, batch, data, precision, or checkpoint settings as a
  hold unless the claim explicitly permits the difference;
- make model-size promotion one-way through the ladder: 0.6B and 1.7B before
  8B, then 30B-A3B MoE.
- emit an advisory evaluation record; human review or a separately designed
  control plane owns any final promotion action.

## Relation To Other Directions

- Foundation training supplies the core evidence hooks and config path.
- Experiment execution supplies rootfs, attempt, stage, and artifact interfaces.
- Countdown and scaffold-to-policy consume certified Qwen checkpoints and produce
  downstream quality evidence.
- Online RL consumes Qwen policies only after deterministic trainer/generator
  parity and checkpoint lineage are established.
- State estimator consumes Qwen run-attempt evidence for diagnosis and probe
  recommendations, not for automatic promotion.

## Verification Ladder

- Unit: config validation, dataset registration, mesh math, promotion evaluator,
  and evidence completeness.
- CPU/static: instantiate configs without launching GPU work.
- Fake backend/local tensor: validate distributed config and control flow.
- GPU smoke: one or two steps to prove plumbing only.
- Numerical: deterministic comparison using full-precision artifacts.
- Checkpoint: save/resume and lineage validation.
- Convergence: representative data training.
- Performance: matched steady-state comparisons.
- Quality: replicated downstream evaluation where the claim scope requires it.

## Open Design Decisions

- Whether Qwen certification should be represented as a generic experiment
  lifecycle declaration or a Qwen-specific wrapper over it.
- Which exact artifacts define the minimum C2 numerical proof for each Qwen
  model size.
- Whether local FineWeb subset manifests should move into a shared dataset
  manifest registry.
- How to represent MoE-specific routing and expert-parallel evidence without
  overloading dense-model fields.
