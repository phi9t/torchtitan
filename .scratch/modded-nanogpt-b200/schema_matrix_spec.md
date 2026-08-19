# Modded NanoGPT B200 Schematized Experiment Matrix

Status: ready-for-implementation

## Intent

Extend the existing B200 NanoGPT harness from a fixed two-GPU prerequisite
launcher into a schematized experiment-control layer that an outer RSI loop can
use safely. The control layer must accept typed experiment specifications,
validate every candidate before launch, materialize deterministic run plans,
execute configurable GPU-count arms, and return normalized evidence for model
training improvement loops.

The existing single-run wrappers remain valid. The new layer sits above them and
does not weaken rootfs, preflight, active-job, source, data, or claim gates.

## Naming

New schema fields use `experiment_kind`, not `lane`.

Allowed `experiment_kind` values:

- `upstream_reproduction`: faithful pinned upstream attempt, legacy `lane=A`.
- `b200_compatibility`: minimal B200 compatibility patchset, legacy `lane=B`.
- `optimization_ablation`: intentional systems/model/training changes, legacy
  `lane=C`.
- `diagnostic`: targeted failure isolation, not a training-quality claim.
- `prerequisite`: plumbing or readiness run, such as the 1/2/4 GPU ladder.

Existing artifacts that use `lane` must remain readable. New artifacts may carry
legacy `lane` as a compatibility field, but the canonical schema field is
`experiment_kind`.

## Architecture

Add a deep module under `experiments/modded_nanogpt_b200/` whose external
interface is:

```python
spec = load_experiment_spec(path)
plan = materialize_experiment_plan(spec)
write_plan_artifacts(plan, result_dir)
```

The implementation behind that interface owns defaults, canonical JSON,
validation, derived run IDs, supported/record-only/rejected knob
classification, preflight command generation, launch command generation,
observability policy, and claim policy.

The outer RSI loop must not construct shell commands directly. It proposes a
schema candidate and consumes structured validation, preflight, runtime,
metrics, blocker, and observability evidence.

## Schema Requirements

Every spec includes:

- `schema_version`
- `experiment_id`
- `description`
- `source`
- `data_manifest`
- `result_root`
- `execution_policy`
- `defaults`
- `arms`

Every arm includes:

- identity: `name`, `experiment_kind`, `legacy_lane`, `arm`, `tags`;
- hardware: `num_gpus`, optional `gpu_ids`, `expected_gpu_name`;
- backend: `attention_backend`, `mlp_backend`, `compile_policy`,
  `ce_compute_capability`;
- model family: `model_family`, `source_variant`, `model_config`;
- training schedule: `training_schedule`, stage batch sizes, validation batch
  size, scheduled and extension iteration counts, evaluation cadence;
- batch policy: `batch_policy`, global token batches, gradient accumulation
  policy, divisibility rules;
- parallelism: `world_size`, future tensor/data/pipeline fields, and support
  status for each;
- optimizer: optimizer family and tunables as schema-known fields;
- data: train files, validation files, manifest, SHA policy;
- preflight: required checks, diagnostic skips, timeout budget;
- observability: capture profile and trigger policy;
- claim policy: prerequisite, diagnostic, compatibility, or baseline gates;
- safety: max runtime, no-output timeout, active-job policy, rootfs-required.

Schema fields whose underlying source hook does not exist yet must be accepted
only as `record_only` or rejected with an explicit validation finding. A run
must never silently claim a setting was applied when it was only recorded.

## First Supported Knobs

The first implementation supports applying:

- `experiment_kind`
- `num_gpus` in `[1, 2, 4, 8]`
- `gpu_ids`, defaulting to the first `num_gpus` device indexes
- `attention_backend`
- `mlp_backend`
- `compile_policy`
- `ce_compute_capability`
- `mode`
- `verify_sha`
- `source`
- `data_manifest`
- `result_root`
- observability profile

The first implementation records but does not apply:

- model shape overrides;
- optimizer overrides;
- non-world-size parallelism fields;
- training-stage batch overrides;
- validation batch overrides.

Later tickets may move batch schedule and validation batch from `record_only`
to `supported` after source-side hooks and preflight divisibility checks exist.

## Validation Requirements

Static validation must reject:

- unknown schema version;
- duplicate arm names;
- `num_gpus` not in `[1, 2, 4, 8]`;
- `8 % num_gpus != 0`;
- `gpu_ids` length not equal to `num_gpus`;
- duplicate `gpu_ids`;
- unsupported enum values;
- full training arms without rootfs-required execution policy;
- full training arms with diagnostic-only skips;
- unsupported fields marked as `supported`.

Materialization must derive:

- `run_id`
- `attempt_id`
- canonical `legacy_lane`
- canonical `claim_label`
- `claim_eligible`
- `CUDA_VISIBLE_DEVICES`
- `NVIDIA_VISIBLE_DEVICES`
- `torchrun --nproc_per_node`
- preflight `--expected-gpus`
- result directory
- observability artifacts

## Observability Requirements

Observability is first-class in the schema and plan. It must be modeled as an
advisory capture profile, not as an afterthought.

Profiles:

- `tier0`: always-on run-attempt evidence, rootfs sentinel, command/env,
  active-job scan, preflight checks, structured log parsing, process/disk/GPU
  watcher summaries, DCGM availability, and artifact sizes.
- `byterobust`: tier0 plus peer-relative progress, no-output progress probes,
  stop snapshots, NCCL evidence, source/data/checkpoint lineage where present,
  and post-run blocker classification.
- `mycroft`: advisory semantic timeline extraction over run/preflight/telemetry
  evidence, centered on incidents or completion.
- `argus`: advisory active diagnostic planner that recommends but does not
  launch probes such as deeper NCCL, EUD, replay, or profiler capture.
- `eroica`: evaluation/RSI reporting profile that compares arms, records
  observability warnings, and separates detection, localization, recommendation,
  and training-quality evidence.

These names are schema labels for local observability profiles. They do not
grant recovery, restart, quarantine, eviction, or learned safety authority.

Every arm summary must record:

- requested observability profile;
- supported observability features;
- unavailable observability features with reasons;
- trigger decisions;
- advisory findings;
- whether evidence was missing, stale, speculative, or observed.

## Execution Requirements

The matrix executor runs arms sequentially by default. Parallel execution is out
of scope until the scheduler can prove disjoint `gpu_ids`, active-job isolation,
and independent result directories.

Every arm must pass preflight before training. A failed preflight writes a
structured arm result and does not start `torchrun`.

All Python, PyTorch, CUDA, NCCL, parsing, summarization, and analyzer execution
must run inside `scripts/rootfs/enter_rootfs.sh`.

## Claim Boundaries

`upstream_reproduction` and `b200_compatibility` can become claim-eligible only
when the arm is 8-GPU, full mode, rootfs-verified, source/data valid, preflight
valid, final validation reached, and target gates pass.

`prerequisite`, `diagnostic`, and early `optimization_ablation` arms are
non-claimable unless a later spec explicitly defines a claim and evidence tier.

The outer RSI loop may optimize over non-claimable arms, but reports must label
them as advisory training evidence, not benchmark success.

## Initial Verification Commands

Focused tests:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_experiment_config.py'
```

Owning harness tests:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/test_modded_nanogpt_b200_launch_nanogpt_rootfs.py tests/unit_tests/test_modded_nanogpt_b200_run_speedrun.py tests/unit_tests/test_modded_nanogpt_b200_preflight.py tests/unit_tests/test_modded_nanogpt_b200_parse_log.py tests/unit_tests/test_modded_nanogpt_b200_summarize.py'
```

Static Python check:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile experiments/modded_nanogpt_b200/*.py'
```
