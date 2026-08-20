# NanoGPT B200 Speedrun Design

Date: 2026-08-20
Status: draft design

## Purpose

Fortify the `modded-nanogpt` B200 speedrun harness so that every attempted run
has auditable source, data, runtime, hardware, launch, parser, and summary
evidence, and so that no artifact can overstate what the run proves.

This is a repo-local research program. It does not change TorchTitan core
training semantics.

## Current State

The implementation lives in `experiments/modded_nanogpt_b200/`. The canonical
tracker and implementation spec live under `.scratch/modded-nanogpt-b200/`.

The current harness has:

- source fetching and pin verification for `kellerjordan/modded-nanogpt`;
- rootfs re-entry wrappers for source, data, preflight, setup, launch, parse,
  summary, matrix, optimized-kernel, and performance-probe commands;
- runtime dependency manifests under `experiments/modded_nanogpt_b200/runtime/`;
- JSON Schemas for attempt, preflight, runtime, summary, command, and kernel
  reports;
- active-job scans and sequential launch hygiene;
- Lane A/B/C classification;
- a two-GPU full-run convenience launcher gated by explicit authorization;
- optimized-kernel certification and diagnostic performance probes;
- unit tests for config parsing, preflight, launch, parsing, summary, schemas,
  runtime verification, tracking, and probes.

The current validated state is non-launch foundation. It supports a launch-ready
prerequisite artifact, not a successful reproduction claim.

## Lane Model

### Lane A: Faithful Upstream Reproduction

Lane A preserves the upstream source at the pinned commit. Wrappers may set
environment, cache, rootfs, compiler, logging, timing, and metadata. They may
not edit upstream source, switch attention backends, change data streams,
change validation token counts, or alter model/optimizer/schedule/precision
semantics.

Lane A can support a B200 upstream reproduction claim only when:

- source is clean before and after the run;
- full preflight passes;
- full data manifest and SHA verification pass;
- training reaches final validation;
- final validation loss is at or below the declared target;
- upstream `train_time` and shell wall-clock are both reported.

### Lane B: Minimal B200 Compatibility Patchset

Lane B starts from the pinned upstream commit and applies the smallest local
patch set needed to run on B200. Every source difference must be preserved in
`variant_patch.diff` and classified as `environment`, `hardware-detection`,
`kernel-compat`, `timing-harness`, or `ML-affecting`.

Lane B can support a B200 compatibility claim only when every patch records the
first Lane A blocker it addresses. If any patch changes model, optimizer,
schedule, train data, validation data, validation count, or banned compile
policy, the claim label becomes a B200 ML variant rather than a source-equivalent
compatibility patchset.

### Lane C: Optimization Ablations

Lane C is blocked until a baseline exists. It changes one variable per arm and
must keep source, data, runtime, GPU allocation, validation, timing, and memory
evidence comparable. Lane C is never the place to diagnose a no-output launch
or repair basic runtime drift.

## Production Target

The harness should become a set of deep modules around a small runner
interface:

```text
load experiment spec
  -> materialize arm plan
  -> verify rootfs/runtime/source/data/host readiness
  -> execute or skip one attempt
  -> parse logs
  -> classify claim
  -> write summary and run index
```

Callers should not manually know the sidecar filenames, allow-list rules,
rootfs sentinel details, or claim-classification edge cases.

## Module Interfaces

### ExperimentSpec and ExperimentArmPlan

Owns:

- declared arms;
- supported knobs;
- GPU counts and IDs;
- world size;
- observability profile;
- rootfs requirement;
- derived launch environment;
- legacy lane mapping.

Hardening requirements:

- reject unknown knobs unless explicitly marked record-only;
- reject inconsistent `num_gpus`, `gpu_ids`, and `world_size`;
- keep semantic `experiment_kind` separate from legacy lane strings;
- make unsupported arm requests fail before any source or data mutation.

### SourceVariant

Owns:

- upstream repository URL and pinned commit;
- source checkout location;
- clean-source checks;
- Lane B patch materialization;
- patch classification and first-blocker metadata.

Hardening requirements:

- make Lane A source mutability impossible through the public interface;
- record a digest or status snapshot before and after launch;
- validate `variant_patch.diff` before a Lane B summary can be claim eligible.

### RuntimeEnvironment

Owns:

- rootfs identity;
- Python environment identity;
- tool versions;
- CUDA, NCCL, B200, and package checks;
- launch authorization presence, without echoing supplied tokens into immutable
  attempt metadata.

Hardening requirements:

- use the shared rootfs adapter from `torchtitan/experiments/execution/`;
- emit one runtime verification artifact per attempt;
- fail closed when host-side Python is about to do real work;
- distinguish missing optional tools from failed required tools.

### LaunchReadiness

Owns:

- active-job policy;
- visible GPU count and names;
- NCCL over declared world size;
- data manifest and SHA verification;
- lane-specific source validity;
- known-stall override policy;
- authorization gate.

Hardening requirements:

- no long GPU job runs unless the trusted user request provided
  `launch-full-b200`;
- blocked attempts produce structured artifacts and do not enter baseline stats;
- prelaunch sidecar filenames are accepted by the same allow-list used for
  attempt reuse.

### ClaimClassifier

Owns:

- `claim_eligible`;
- `claim_label`;
- baseline inclusion;
- invalidation reasons;
- diagnostic/prerequisite/smoke/full distinctions.

Hardening requirements:

- classify from artifacts, not caller intent;
- reject contradictory artifacts;
- make every reproduction or compatibility claim replayable from sidecars;
- never infer success from exit code alone.

### SummaryBuilder

Owns:

- `summary.json`;
- markdown summary;
- run index update;
- baseline statistics;
- latest-pointer files.

Hardening requirements:

- separate upstream `train_time` from shell wall-clock;
- keep full validation loss distinct from partial or absent validation;
- refuse to summarize a diagnostic as a baseline;
- preserve failed and blocked attempts as first-class evidence.

## Rootfs Contract

Every substantive step runs inside `scripts/rootfs/enter_rootfs.sh`. Shell
wrappers may be launched from the host, but must re-enter before Python, CUDA,
NCCL, package, parser, summarizer, data, or training work.

The harness must prove rootfs identity using both environment and filesystem
sentinels. The minimum trusted state is:

- `TORCHTITAN_IN_ROOTFS=1`;
- current working directory is `/workspace/torchtitan`;
- `/workspace/torchtitan/scripts/rootfs/enter_rootfs.sh` exists.

## Known Runtime Facts

- The active full mode is a two-GPU B200 trial unless an eight-GPU campaign is
  separately authorized.
- Lane A FA3 has failed on B200 because the available kernel image does not
  support the device.
- Lane B FA2 `flash-attn==2.8.3.post1` source-builds in the rootfs and passes
  the B200 smoke.
- Triton MLP work has known sm100 failure history and a bounded diagnostic path.
- PyTorch MLP fallback has local smoke evidence but not full-job warmup proof.

These are inputs and blockers. They are not reproduction evidence.

## Verification Ladder

- Static: `verify_static.py` validates wrapper shapes, schema references, and
  required files.
- Unit: `tests/unit_tests/test_modded_nanogpt_b200_*.py` and relevant rootfs
  shell tests cover config, schema, runtime, launch, parse, summary, probes, and
  tracker behavior.
- Runtime prerequisite: rootfs verification, source verification, data manifest,
  NCCL, active-job scan, and launch readiness produce sidecars with `skip_run`.
- Diagnostic: performance probes answer no-output, compile, warmup, NCCL, and
  observability-overhead questions without claim eligibility.
- Full launch: two visible B200 GPUs, authorization token from trusted request,
  no active jobs, SHA-verified full data, lane-valid source, final validation,
  and conservative summary.
- Production: a machine-checkable matrix of at least 10 successful non-skip full
  B200 attempts before calling the workflow productionized. Every counted
  attempt must have an independent attempt ID and the same declared lane, arm,
  backend, source patch digest, data manifest digest, rootfs identity, visible
  two-GPU B200 allocation class, final validation target, claim label, evidence
  tier, and no known-stall override. Any intentional variation starts a separate
  production matrix rather than being mixed into the count.

## Migration Targets

1. Move generic rootfs and command-capture logic to shared experiment execution.
2. Replace duplicated per-script rootfs boilerplate with a single generated or
   sourced wrapper contract.
3. Promote shared schemas for command, rootfs, runtime, attempt, and artifact
   references.
4. Keep lane rules, speedrun parsing, and upstream patch provenance local.
5. Add replay tests that build claim classification from a fixture attempt
   directory without running any training.

## Out Of Scope

- Running a full B200 launch without `launch-full-b200`.
- Calling a skip-run, smoke, prerequisite, or diagnostic a reproduction.
- Changing TorchTitan core trainer behavior.
- Pushing results or source externally.
