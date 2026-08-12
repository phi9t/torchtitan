# Runtime, Durable Execution, And Preflight Roadmap

Status: approved design. The current runtime doctor and `run_common.sh` are
useful prototypes, but they do not yet satisfy this contract.

This roadmap defines the execution foundation for scaffold-to-policy,
Countdown, and later repo-local research programs. It covers immutable run and
attempt identity, bwrap rootfs execution, composable runtime doctors,
benchmark-semantic preflight, Temporal durability, tracing, artifact lineage,
report integrity, and failure injection.

It is an adoption plan within Phase 4 of the repository-wide training research
vehicle. The Phase 1 core `Trainer` observability foundation and its canonical
run-attempt schema must land first, followed by the required foundation-model
and multimodal phase gates. This roadmap extends that schema with
experiment-stage, artifact, verifier, harness, and promotion evidence; it does
not define a parallel observability authority. See
[`docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`](../../docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md).

It does not replace core TorchTitan training, the online RL controller, task
verifiers, or external benchmark harnesses. The training program is specified
in `training_research_plan.md`; benchmark programs are specified in
`reasoning_research_plan.md` and `coding_research_plan.md`.

## 1. Why The Current Foundation Must Change

The repository already has important pieces:

- `scripts/rootfs/build_rootfs.sh` builds and exports a Docker-derived rootfs;
- `scripts/rootfs/enter_rootfs.sh` enters it with bwrap, mounts the checkout,
  exposes NVIDIA devices and host driver libraries, and optionally exposes
  Docker for Harbor;
- `experiments/scaffold_to_policy/run_common.sh` handles rootfs re-entry,
  environment setup, stage timing, JSONL rows, and optional stage-status files;
- `run_runtime_doctor.sh` checks imports, model assets, CUDA, vLLM memory, cache
  paths, and optional executables;
- shared report helpers hash and label artifacts;
- several runners already record command stages and blocker report inputs;
- core SFT and online RL already have structured logging and metric backends.

These are not yet a durable evidence system.

Known gaps to close include:

- `scaffold_setup_run_manifest` truncates the manifest for an existing
  `RUN_ID`; a nested doctor or retry can erase prior rows;
- run identity does not distinguish a scientific declaration from an
  operational attempt;
- artifact freshness is inferred partly from path or payload occurrence of a
  run ID rather than producer evidence;
- stage-status sidecars are optional and are not a complete lifecycle;
- scaffold report builders do not consistently validate the stage journal;
- latest report selection is lexical and can select a failed candidate;
- runtime metadata is optional in paths that make real model claims;
- the current doctor has one broad requirement set, so optional packages and
  harnesses cannot compose cleanly;
- several runners do not invoke the shared doctor or stage recorder;
- current report and roadmap terminology conflates lane, runtime profile,
  execution mode, score validity, failure, and promotion;
- rootfs build inputs and runtime mutation are not sufficiently pinned for a
  reproducible scientific identity;
- `build_rootfs.sh --dest` currently accepts an arbitrary path and later removes
  that path recursively, while `enter_rootfs.sh --rootfs` can pass a missing
  custom path into the builder; neither path has an allowlist, ownership marker,
  symlink-safe target check, or recoverable activation protocol;
- some measured-stage wrappers still install packages or clone, fetch, and
  check out repositories in writable result trees or the shared rootfs, so a
  rerun can silently change its executable environment;
- persistent Temporal SQLite backup and restore do not yet have a permissions,
  consistency, integrity, compatibility, or receipt-reconciliation contract;
- staged runs have no durable controller across host, worker, or process
  restarts.

The solution is a deep experiment-only lifecycle module, a thin shell facade,
and an optional host-side Temporal adapter.

## 2. Design Boundary

Place the shared implementation under:

```text
torchtitan/experiments/execution/
  lifecycle/
  rootfs/
  preflight/
  reporting/
  temporal/
```

Top-level shell runners remain under `experiments/`. Core cannot import this
package.

The module hides:

- run and attempt reservation;
- append-only event publication;
- atomic outcome and artifact commits;
- process and signal handling;
- doctor profile composition;
- artifact hashing and lineage;
- before/after provenance checks;
- report enrichment and validation;
- optional Temporal workflow/activity integration;
- optional TensorBoard/W&B indexing.

Its identity and event envelope reuses the certified core fields for
run/attempt, process/rank/mesh, device, clock, step, phase, checkpoint, and
lineage. Experiment-only fields extend that envelope. Schema versioning and
conversion are explicit when the lifecycle needs a field not yet present in
core.

It deliberately does not own:

- a general DAG language or scheduler;
- model, optimizer, or RL loop configuration;
- scientific task ordering inside a run family;
- benchmark scoring semantics;
- GPU fleet allocation or host quarantine;
- remote artifact storage;
- a replacement metrics system.

Task-specific sequencing stays close to the task runner. The shared module
provides lifecycle and evidence depth without centralizing every experiment.

## 3. Identity Model

### 3.1 Run declaration

A `run_id` identifies one immutable scientific declaration. The declaration
includes:

- family and program;
- task and scientific lane;
- hypothesis and predecessor gates;
- benchmark, split, and evaluator identities;
- model, tokenizer, adapter, and checkpoint inputs;
- scaffold, prompt, renderer, tools, and sampling;
- training or RL config identity;
- exact comparison cell and seed/draw;
- planned stages and expected artifact classes;
- compute and stopping budget;
- source and rootfs requirements.

The normalized declaration receives a digest. Reusing a `run_id` with a
different digest is an error.

### 3.2 Attempt

An `attempt_id` identifies one top-level operational execution of the frozen
run declaration. It begins before the first coordinator event and ends with
exactly one attempt outcome after its stage sequence completes, blocks, fails,
or is interrupted. An operator/campaign retry or recovery after that terminal
outcome creates a new attempt and links its parent; it never truncates or
reopens the old attempt.

A `stage_invocation_id` identifies one actual subprocess, external job, or
in-process stage launch within an attempt. Temporal Activity delivery attempts
and reconciliation do not create a new `attempt_id`. A genuine relaunch within
the still-active attempt gets a new `stage_invocation_id`; a redelivery that
only validates a terminal receipt creates neither fresh work nor a new stage
invocation. This distinction prevents a coarse Activity retry from fragmenting
one run execution into arbitrary attempt bundles.

Work committed by an interrupted attempt may be resumed only in a new linked
attempt, and only when:

- its declaration digest is identical;
- the stage explicitly supports resume;
- the input and last committed artifact lineage validates;
- no conflicting live process or GPU lease exists;
- the resume action and parent attempt are recorded before execution.

Temporal workflow, workflow-run, and Activity IDs are correlation values. They
are not scientific identity.

### 3.3 Attempt bundle

```text
<results-root>/runs/<run-id>/<attempt-id>/
  manifest.json
  processes/<process-id>/events.jsonl
  artifacts/<producer>/...
  indexes/artifacts.<process-id>.jsonl
  checkpoints/<checkpoint-id>.json
  incidents/<incident-id>.json
  outcome.json
  derived/report_input.json
  derived/...
```

This is the upstream core run-attempt layout, not an experiment fork.
`manifest.json` includes or immutably references the normalized scientific
declaration. The lifecycle coordinator is one process with its own event stream
and artifact index. Torchrun ranks, Monarch actors, vLLM workers, sandbox
executors, and harness processes emit or register their own process evidence;
they do not contend on a single global JSONL file. Wall clock supports joins,
while monotonic clock and process-local sequence preserve ordering.

Terminal files are immutable. Raw process records and artifact indexes are
append-only. Derived reports and views may be regenerated. Partial and
interrupted evidence is retained. Legacy artifacts are imported or reused with
`freshness=unknown`; they are not retroactively assigned `verified_new`.

## 4. Lifecycle Interface

The internal Python interface is typed and small.

```python
RunAttempt.create(declaration, attempt_id, results_root)
RunAttempt.run_stage(stage_spec)
RunAttempt.finish(canonical_report_input, evaluations, attempt_outcome)
```

`finish` validates per-condition measurement and promotion states in
`evaluations`, derives the run-gate summary, and commits the independent
attempt outcome. It does not accept one run-wide measurement/promotion pair.

The shell facade exposes the migration-compatible equivalent:

```text
python -m torchtitan.experiments.execution begin ...
python -m torchtitan.experiments.execution stage ... -- COMMAND...
python -m torchtitan.experiments.execution finish ...
```

`run_common.sh` delegates to this facade. Existing task runners may continue to
sequence shell commands while gaining the typed lifecycle.

Do not require every shell workflow to become a Python recipe before it can
migrate. A future family-level Python callable may use the same internal API,
but it is not a prerequisite or a second lifecycle.

## 5. Stage Protocol

A stage declaration contains:

- unique stage ID and human name;
- stage kind;
- execution adapter and profile;
- exact argv and working directory;
- dependencies;
- declared inputs and outputs;
- timeout and cancellation grace period;
- retry classification;
- resume capability;
- expected gate or validator;
- resource lease requirements.

Stage kinds include:

```text
doctor
preflight
acquire
prepare
generate
verify
train
checkpoint
export
evaluate
ingest
report
```

Execution adapters include:

```text
host_test
rootfs_cpu
rootfs_vllm
rootfs_torchrun_sft
rootfs_monarch_rl
coding_sandbox
external_harness
```

`coding_sandbox` is a host-launched, minimal bwrap executor distinct from the
broad trusted experiment rootfs. It receives only one task, evaluator shim,
candidate, pinned dependency root, limits, and an empty scratch directory. It
has no network, GPU, Docker socket, checkout, model assets, caches, or secrets,
and uses process-group plus cgroup cleanup. The coding research plan owns its
full security and acceptance contract.

Each stage appends `stage_started` before launch and one terminal event when
possible:

- `stage_succeeded`;
- `stage_blocked`;
- `stage_failed`;
- `stage_interrupted`.

An unmatched start remains crash evidence. Terminal events contain command,
stage invocation ID, clocks, duration, return code, rootfs identity, resource
lease, input/output references, log paths, and normalized failure information.

Stage names are unique within an attempt. Only one process may write a given
process-local event stream or artifact index. The lifecycle coordinator owns
outer stage transitions; nested runners attach to inherited run/attempt and
process identities, register child processes, and may not initialize or write
the coordinator stream.

## 6. Artifact And Checkpoint Lineage

Each artifact reference contains:

- stable artifact ID and class;
- path, size, digest, and media/schema type;
- existence and validation result;
- work status: `produced`, `reused`, `resumed`, `imported`, or `external`;
- freshness: `verified_new`, `verified_preexisting`, or `unknown`;
- producer run, attempt, and stage;
- source/config/data/model/checkpoint parents;
- before and after observations;
- atomic completion receipt;
- external job or harness identity when applicable.

`work_status` describes how the stage relates to the artifact; `freshness`
describes what provenance proves about the bytes. `produced` normally pairs
with `verified_new`; `reused` pairs with `verified_preexisting`; a `resumed`
stage may commit a `verified_new` terminal artifact from a preexisting parent.
Imported or external evidence may remain `unknown` when no reliable before/after
observation exists. `fresh` is not a work-status value. A run ID in the path is
not freshness proof.

Checkpoint references additionally record semantic state inventory, step,
tokens, topology, parent checkpoint, and commit status. DCP, PEFT adapters, and
merged Hugging Face policies are distinct artifact classes.

## 7. One Canonical Report Input

Do not add a separate `scaffold_to_policy_result_manifest`. Existing task
report builders remain the scientific schema owner. The lifecycle layer adds a
validated `execution` section to the canonical report input.

Minimum top-level sections are:

```text
schema_version
run
evaluation
attempt_outcome
artifacts
checks
metrics
promotion
```

Task-specific sections such as verifier, scaffold, preflight, training,
failure analysis, and examples remain available.

`evaluation` is a collection keyed by `evaluation_id` and condition/cell/split,
not one run-wide score object. Each entry carries its own execution,
measurement, and promotion statuses. This permits, for example, a valid dev
measurement and a blocked later split in the same attempt without discarding or
overstating either result. `attempt_outcome` is derived separately from all
stage terminal events and does not overwrite evaluation status.

### 7.1 Independent status dimensions

```text
execution_outcome: completed | blocked | failed | interrupted
measurement: real | fixture | smoke | invalid | not_run
promotion: promote | hold | reject | not_evaluated
```

These values occur per evaluation condition. Attempt-level execution is a
separate derived summary with links to every stage and condition.

Rules:

- a valid score of zero is `measurement=real`;
- a valid regression is `measurement=real`, normally `promotion=reject`;
- a runtime, access, verifier, or integrity blocker has no fabricated score;
- process exit zero does not imply task success or promotion;
- an external harness score is ingested from its official artifact;
- score tables include only valid measurements unless the table explicitly
  analyzes plumbing or blockers.

### 7.2 Complete evaluation identity

An `evaluation_id` digest covers:

- benchmark release, task manifest, split, and exclusions;
- evaluator/verifier code and configuration;
- model, tokenizer, checkpoint, and adapter;
- prompt, renderer, scaffold, tools, and agent loop;
- sampling, seeds, context, action, and retry budgets;
- harness/provider and runtime versions;
- rootfs and dependency identity;
- timeout, network, sandbox, and capability policy.

Changing any field creates a new evaluation condition.

### 7.3 Latest selection

Each declaration names a stable logical result slot such as
`family/task/benchmark-split/model-role/prompt-condition` and an explicit
supersession parent when replacing a prior condition. An index considers only
terminal report inputs whose identity and checks validate, partitions them by
logical slot and evaluation identity or declared equivalence class, and orders
within that partition by completion time plus immutable attempt identity. It
never compares or supersedes unlike prompts, models, datasets, evaluators, or
harnesses merely because they ran later. It reports both latest valid and
latest attempt for each slot so a recent failure cannot silently replace or
hide the last valid evidence.

## 8. Rootfs Reproducibility

### 8.1 Build identity

The current builder uses a base tag, `uv:latest`, current apt repositories, and
unpinned pip packages. The hardened builder must produce a build manifest with:

- base image reference and resolved digest;
- pinned `uv` image digest;
- apt snapshot or locked package/version inventory;
- Python/wheel lock and hashes;
- CUDA toolkit package versions;
- Dockerfile/build input digest;
- resulting image ID and exported-rootfs identity;
- build time, host tool versions, and source revision.

Scientific reports reference the manifest digest, not only
`torchtitan-rootfs:local`.

### 8.2 Safe build target and activation

The current destructive path is a release blocker, not merely a
reproducibility issue. `build_rootfs.sh` accepts any `--dest`, exports into a
sibling temporary directory, then executes `rm -rf "$DEST"`. When
`enter_rootfs.sh` sees that a caller-supplied `--rootfs` lacks `bin/bash`, it
automatically invokes that builder with the same path. A typo, symlink, broad
directory, or unrelated existing directory can therefore become a deletion
target.

Before any automatic build is retained, the launcher and builder must share one
target resolver with this fail-closed contract:

- canonicalize the existing parent and requested basename before Docker or any
  filesystem mutation; accept only an exact destination in a static managed
  registry. The F0 compatibility allowlist may contain only the canonical
  legacy target, and F3 narrows managed builds to content entries beneath a
  dedicated rootfs-store directory;
- use `lstat`-equivalent checks on every deletion or activation endpoint;
  refuse a symlink target, a symlinked parent component, a non-directory parent,
  a mount point, or an unexpected file type rather than traversing it;
- explicitly refuse empty paths, `/`, the home directory, repository or
  workspace roots, the store root itself, and any destination outside the
  allowlist, even if a caller requests it through `--dest` or `--rootfs`;
- replace an existing directory only when its exact path carries a valid
  builder ownership marker whose store ID and manifest digest agree with the
  registry; an unrelated or ambiguous existing target is never removed;
- allow implicit construction only for the canonical default or a registered
  store ID. A missing arbitrary custom `--rootfs` is an error, not an invitation
  to build;
- create cleanup targets with a trusted unique name inside the validated store,
  record their ownership before extraction, and clean only that exact
  non-symlink staging directory. No unchecked variable, glob, or path prefix is
  a deletion authority.

Build and activation are separate transactions. Export into an
invocation-scoped staging directory, then validate its ownership marker,
manifest digest, expected rootfs structure, executable `bin/bash`, and a
minimal bwrap smoke before it can become selectable. Publish it as an immutable
content-addressed directory. Activation atomically replaces a small fsynced
selection record that names the validated directory; it does not recursively
replace the active directory. Keep the previously selected digest until a
post-activation launch check passes. A failed check or interrupted activation
restores the prior selection record, and an invalid staged tree is quarantined
for inspected cleanup. If the platform cannot provide the required atomic
same-filesystem publication, fail rather than fall back to delete-and-move.

The launcher resolves the selection record to a canonical real directory,
revalidates its marker and digest, acquires a per-digest launch lease, and passes
that directory to bwrap. The lease remains held until the complete process tree
exits. Rollback therefore changes only the selection used by future launches;
it does not invalidate a rootfs already bound by a live job. Garbage collection
is a later explicit operation over enumerated, unselected, owned store entries
that have no lease or refcount and whose process and mount checks prove no live
user. It is never part of build, activation, launcher startup, or Activity
retry.

### 8.3 Runtime mount policy

The target scientific profile uses:

- read-only base rootfs;
- read-only source checkout where practical;
- explicit writable attempt data, results, checkpoint, cache, and temporary
  mounts;
- `/proc`, `/dev`, and required NVIDIA devices;
- read-only host driver libraries and `nvidia-smi`;
- declared environment allowlist;
- no implicit host home, credentials, or arbitrary filesystem paths.

Nested writable mounts under a read-only source tree may preserve current
repo-relative artifact paths during migration. Set `PYTHONDONTWRITEBYTECODE=1`
and place compiler caches in declared writable paths.

### 8.4 Capabilities

Capabilities are explicit and recorded:

```text
network_none
local_distributed_network
network_acquisition
host_network
gpu
docker_socket
host_repo_mirror
external_task_images
```

Docker access remains restricted to declared Harbor/Terminal-Bench stages. A
Docker socket is effectively host control and cannot appear in a normal model
or coding-executor profile.

`network_none` exposes no network transport to the workload.
`local_distributed_network` permits only the loopback, Unix-socket, and
explicitly required single-node rendezvous/NCCL traffic while denying DNS and
external egress. It is distinct from `host_network`, which shares the host
network namespace and remains a measured compatibility escape hatch during
migration. The exact namespace/firewall implementation and allowed interfaces
are part of runtime identity.

Separate asset acquisition from scientific execution. The long-term default
for generation, training, and evaluation is `network_none` or
`local_distributed_network` after assets validate. Before enforcing this
default, prove torchrun rendezvous/collectives, Monarch, vLLM, sandbox-broker
IPC, and each harness under the intended namespace and egress policy. A run is
not offline merely because it made no observed download while sharing the host
network.

### 8.5 Immutable environment acquisition

Package and source acquisition is a distinct, network-enabled preparation
stage. It resolves only declaration-pinned inputs, downloads into quarantine,
verifies hashes and revisions, builds a content-addressed environment bundle,
runs an import or harness smoke, and atomically publishes the sealed bundle.
The bundle manifest includes the lockfile digest, every wheel or source archive
hash, exact VCS commit and clean-tree status, build-tool versions, external
image digests, and the final bundle digest. Mutable tags, branches, Git URLs
without an independently verified commit, and unhashed package resolution are
not scientific inputs.

Generation, training, evaluation, and official-harness science stages receive
the selected environment, source snapshots, wheelhouse, task images, and data
as read-only mounts under `network_none` or the validated
`local_distributed_network` profile. They may write only declared attempt
outputs and caches. They must not run `pip install`, upgrade a package, invoke
`git clone`, `git fetch`, or `git checkout`, pull an image, or mutate the base
rootfs, a shared virtual environment, or a shared source checkout. Preflight
hashes the sealed inputs immediately before launch and terminal publication
checks that those inputs did not change.

The current mutation inventory is explicit:

- `run_bigcodebench_hard_public_vllm_smoke.sh` installs BigCodeBench
  dependencies into the shared rootfs;
- `run_external_harness_preflight_smoke.sh` creates and updates Harbor,
  Terminal-Bench, and tau2 virtual environments at runtime;
- `run_terminal_bench_oracle_probe.sh` installs a virtual environment and
  clones, fetches, and checks out Terminal-Bench;
- `run_tau2_execution_probe.sh` and `run_tau2_mock_score_smoke.sh` install tau2
  and clone, fetch, and check out its repository;
- `run_arc_agi2_public_vllm_smoke.sh` clones the ARC-AGI repository during the
  run.

These remain plumbing probes until migrated. Their network and mutation steps
move to declared acquisition Activities; the compatibility runners then consume
only the resulting read-only bundle. Migration is complete only when a guard
that denies package-manager and VCS mutation during measured stages passes for
every listed runner and two executions of one declaration report the same
environment digest.

### 8.6 Runtime identity

Every rootfs attempt records:

- bwrap and launcher source versions;
- rootfs build manifest digest;
- exact mounts and capabilities;
- host kernel, driver, and injected NVIDIA libraries;
- visible GPU UUID, model, memory, and allocation;
- torch, CUDA, vLLM, Triton, transformers, datasets, and relevant experiment
  package versions;
- cache paths and offline/online policy;
- effective environment allowlist.

### 8.7 Secrets and outbound projections

Credentials are capabilities, not ordinary environment metadata. A stage
declaration records only the allowlisted secret name, purpose, provider, and
delivery mechanism--never its value, prefix, length, low-entropy hash, or a
command line containing it.

- Prefer invocation-scoped read-only credential files or inherited file
  descriptors mounted only into acquisition/export stages that require them.
- Do not place HF, W&B, cloud, provider, SSH-agent, or benchmark credentials in
  the base rootfs, source tree, general cache, declaration, event stream, argv,
  stdout/stderr, traceback, or Temporal payload/history.
- Redact configured names and provider-specific token patterns before log or
  event publication. Preserve a redaction count and scanner version, not the
  matched bytes.
- Scan artifact manifests, text logs, structured events, reports, and export
  bundles before terminal publication. A detection blocks external export and
  creates restricted incident evidence.
- Strip the secret capability before launching model inference, SFT/RL,
  candidate sandboxes, or unrelated harness stages.

TensorBoard is local and still receives only declared scalar/tag fields. W&B
and every other remote exporter use a versioned field and artifact allowlist;
raw prompts, generations, source, checkpoints, environment dumps, commands,
logs, and credential-bearing paths are denied by default. An exporter reads a
validated local derived view, never arbitrary attempt directories. Export
failure or redaction blockage cannot change local scientific status.

## 9. Profile And Clause Model

A scientific lane is not an execution profile.

Scientific lanes:

```text
reasoning
coding
agentic
```

Composable execution profiles:

| Profile | Purpose |
| --- | --- |
| `host_static` | docs, lint, parser, and unit tests without rootfs |
| `rootfs_cpu` | rootfs identity and CPU-only repo execution |
| `vllm_1gpu` | local model generation/evaluation |
| `sft_1gpu` | small core TorchTitan SFT |
| `sft_8gpu` | representative distributed SFT |
| `rl_deterministic` | strict trainer/generator parity and on-policy RL |
| `rl_async` | production-style async RL |
| `coding_exec` | untrusted candidate execution boundary |
| `harbor` | Harbor, Docker, task images, and official verifier |
| `tau2` | tau2 data, state reset, simulator, and evaluator |

A run composes profiles needed by its stages. Optional packages affect only
profiles that declare them.

The `coding_exec` profile must never be implemented as the existing trusted
rootfs plus `subprocess.run`. Its doctor proves that the independent minimal
bwrap root, namespace isolation, cgroup/rlimit enforcement, output bounds, and
descendant cleanup are available on the host before a candidate is launched.

Doctor clauses use `pass`, `fail`, or `skip`. A profile is `ready` only when
all required clauses pass. Do not overload `selected`, which currently means
different things in split registries, preflights, and reports. During schema
migration, readers accept legacy `selected`; new writers use explicit terms.

## 10. Generic Doctor Clauses

The generic doctor proves execution prerequisites, not benchmark semantics.

Clause groups:

1. **attempt**: valid declaration, writable attempt bundle, inherited attempt
   consistency;
2. **rootfs**: active bwrap boundary, build manifest, mounts, capabilities;
3. **source**: expected revision and no undeclared source drift;
4. **packages**: profile-specific imports and executables;
5. **assets**: model/tokenizer/config/safetensors presence and hashes;
6. **GPU**: count, UUIDs, allocation, memory query, and optional headroom;
7. **distributed**: world-size arithmetic, ports/network policy, mesh
   requirements;
8. **cache**: declared writable cache and offline policy;
9. **external**: harness executable, checkout, image, or simulator only when
   required;
10. **output**: atomic write and rename support on artifact paths.

The doctor writes one artifact and never initializes the attempt journal. A
failed required clause yields `execution_outcome=blocked`, a stable blocker
code, and no score.

## 11. Benchmark-Semantic Preflight

Runtime readiness is necessary but insufficient. A score is valid only after
task-owned semantic preflight.

### 11.1 Shared data checks

- source release and raw-cache digest;
- normalized stable problem identities;
- global exact split non-overlap;
- task-specific near-duplicate audit;
- frozen task manifest and exclusions;
- canonical answer/solution acceptance;
- evaluator/verifier identity;
- prompt, tool, sampling, and retry policy;
- evaluation-unit and aggregation semantics.

### 11.2 Reasoning

- parser and canonical-solution fixtures;
- answer-form coverage and explicit exclusions;
- randomized and balanced multiple-choice positions;
- format success separate from task success;
- ARC pack/unpack and exact-grid validation;
- no-tool conditions remain no-tool.

### 11.3 Coding

- official task/harness revision;
- canonical solution passes in the pinned environment;
- compiler/runtime and dependency identity;
- public versus hidden test policy;
- candidate extraction and entry-point contract;
- network, filesystem, process, CPU, memory, output, and timeout limits;
- cleanup after fork, signal, or timeout;
- explicit statement that a subprocess alone is not a security sandbox.

### 11.4 Harbor/Terminal-Bench

- Harbor and task-repo revisions;
- task manifest and image digests;
- Docker/Compose/provider identity and host-path mapping;
- official verifier availability;
- oracle/canonical and no-op controls;
- result and per-trial artifact ingestion.

### 11.5 tau2

- repo/package and task/domain pins;
- policy files, tool schema, user simulator, and evaluator;
- deterministic clean state reset;
- max steps/errors, seeds, retries, concurrency, and timeout;
- upstream `results.json` and trajectory ingestion.

## 12. Local Temporal Adapter

Temporal is an optional experiment dependency installed outside the rootfs.
See `temporal_durable_execution_research.md` for the primary-source rationale
and SDK constraints.

### 12.1 Processes and storage

- Run a version-pinned local Temporal development server with an explicit
  persistent database file outside temporary/rootfs paths.
- Put the state directory outside the checkout, rootfs, result trees, and shared
  temporary paths; create it owner-only (`0700` directory and `0600` database,
  WAL, backup, and manifest files) under a restrictive umask, and reject
  symlinks or files owned by another user.
- Run the Python worker on the host and connect to the local namespace.
- Activities launch `scripts/rootfs/enter_rootfs.sh -- <runner>`.
- Rootfs workloads do not import the Temporal SDK or contact the server.
- Pin workflow and Activity code versions and replay retained histories before
  incompatible worker changes.
- Record the Temporal CLI/server binary version and digest, Python SDK and lock
  digest, SQLite library/tool version used for backup and integrity checks,
  worker code revision, namespace and retention settings, and state/backup
  schema version in the operational manifest.

The local development server is useful for local durability but is not a
production service. Bind it only to loopback and do not rely on `start-dev` as
an authenticated multi-user service. A host that requires remote or shared
access needs a production deployment design outside this roadmap.

Backups are consistent database operations, not file copies. Either stop the
server cleanly and verify that SQLite has quiesced, or use SQLite's online
backup API against the live database. Never copy only the main database while
WAL state may be outstanding. Write the backup to a new restrictive staging
directory, run a full SQLite integrity check, record its digest plus all
versions above, then atomically publish the backup manifest. Failed or
unverified backups are quarantined and cannot satisfy the durability gate.

Restore is non-destructive: stop or isolate the normal server, materialize the
backup in a new owner-only state directory, verify digest, manifest
compatibility, file ownership/modes, and SQLite integrity, then start a pinned
server on an isolated port and namespace. Replay retained Workflow histories
against the intended Worker code and reconcile every referenced run, attempt,
stage receipt, and artifact digest with the canonical repo-local bundles.
Missing, extra, mutable, or contradictory receipts fail the drill. Only after
replay and reconciliation pass may an atomic state selector activate the
restored directory; the prior state remains available for rollback. Retention
and backup deletion are separate explicit policies over enumerated backups.

The SQLite state and its backups are operationally sensitive even though they
are not the evidence authority. Secret values, hashes, credential files, and
credential-bearing commands must never enter Workflow arguments, results,
Memos, Search Attributes, heartbeat details, Activity exceptions, Event
History, or worker logs. Workflows carry only an opaque declared secret
capability ID. The host Activity resolves that capability to an
invocation-scoped file or descriptor immediately before launch, scrubs
subprocess exceptions before raising a classified Temporal error, and keeps any
restricted raw diagnostic only in the local incident bundle.

### 12.2 Workflows

Two workflow types are sufficient:

- `ExperimentRunWorkflow`: one immutable run declaration and the operational
  attempts needed to reach its terminal run outcome;
- `ResearchCampaignWorkflow`: ordered or parallel child runs plus promotion
  waits and compute-tier authorization.

Workflow IDs are deterministic within the configured project and Temporal
namespace from workflow kind, run/campaign ID, and declaration digest. Client
retries reject an incompatible or concurrently running ID and may return an
existing compatible handle only after declaration validation. Terminal ID
reuse follows one frozen policy with an explicit generation suffix for a new
execution; it is never an implicit way to create a new attempt. This prevents
parallel duplicate run Workflows while keeping Workflow IDs correlation-only.

Workflow code carries only small immutable declarations and artifact
references. It performs no filesystem reads, subprocess launch, clock/random
access, CUDA inspection, or TorchTitan import. Those belong in Activities.

Use Continue-As-New only at campaign or major phase boundaries when history
growth warrants it. It does not create a new scientific run.

### 12.3 Activities and at-least-once safety

Activities align with artifact commits. Stable identity is
`run_id / attempt_id / stage_id / shard_id / declaration_digest`. Delivery
retries inside one attempt share that key; a linked new attempt receives a new
key. Cross-attempt artifact reuse is an explicit consuming stage with recorded
provenance, never same-key reconciliation. Because an Activity may run more
than once:

- acquire a local resource/path lease before launch;
- write only to attempt/invocation-scoped temporary outputs;
- publish after validation via atomic rename or an equivalent commit marker;
- return a same-attempt prior receipt only after identity and hash validation;
  preserve its original work status and record the redelivery separately as
  `delivery_disposition=reconciled_no_launch`;
- observe or reattach only through a durable local supervisor or official
  external job API with verified ownership, status, and terminal receipt;
- never infer completion from path existence;
- never retry a clean benchmark zero or policy failure;
- never treat an arbitrary recorded PID as attachable and never automatically
  retry ambiguous training or RL state.

Every long Activity has a stage-specific Start-To-Close,
Schedule-To-Close, heartbeat, and bounded retry policy. Schedule-To-Close
includes that Activity's queue time and may not be copied from the campaign's
total budget. The campaign separately enforces its absolute deadline and
persistent compute ledger across all Activities.

### 12.4 Heartbeats and cancellation

Long Activities heartbeat stage, process identity, semantic progress, and last
committed checkpoint. Retry revalidates local state; heartbeat details are
hints, not evidence. Start them with
`ActivityCancellationType.WAIT_CANCELLATION_COMPLETED`, a heartbeat timeout,
and a bounded cleanup allowance; the Workflow awaits the Activity handle until
its verified local terminal receipt exists.

Cancellation follows the complete adapter-owned resource scope:

1. append cancellation intent;
2. request graceful termination through the adapter-specific cleanup handle;
3. allow a bounded checkpoint-safe evidence interval where supported;
4. if still live, revalidate ownership, then use the owned cgroup/process-group
   kill path and Docker or official-job cleanup API as applicable;
5. wait/reap and verify that no owned descendants, daemon-owned containers,
   GPU contexts, or resource leases remain;
6. publish terminal evidence; and
7. release the lease and acknowledge cancellation.

`bwrap --die-with-parent` remains enabled. Cancellation behavior is tested for
the shell, torchrun children, Monarch actors, vLLM workers, and external
harness descendants. Docker/Harbor resources are daemon-owned and require
explicit API cleanup and verification; subprocess-group exit is insufficient.

### 12.5 Retry policy

Retry only classified transient failures, such as a short-lived service or
download error in an acquisition Activity. Use bounded policies.

Non-retryable examples include:

- invalid declaration or config;
- split, verifier implementation/crash, canonical-control, or semantic-preflight
  failure;
- corrupt or ambiguous checkpoint;
- deterministic parity failure;
- nonfinite training state;
- repeated setting-dependent OOM;
- artifact identity mismatch;
- policy-age or weight-sync violation.

A wrong answer, benchmark-defined task failure/timeout, reward zero, or real
score regression is not an Activity error. The Activity completes successfully
with `measurement=real`; the condition's promotion is normally `reject` or
`hold`. The workflow does not rerun it because the declared measurement is
terminal, not because Temporal received a non-retryable failure.

Workflow-level retries are not used as a substitute for Activity-specific
recovery. Workflow Executions do not retry by default, but ordinary Workflow
Task exceptions do retry and can wedge without advancing history. Validate
declarations before start, map expected permanent domain failures to typed
terminal application outcomes, and alert/test repeated Workflow Task failures.

## 13. Resource Leases

Temporal task queues classify work but do not allocate hardware by themselves.
The local worker uses explicit leases whose records enter the attempt bundle.

```text
cpu
gpu1
gpu8
coding-sandbox
harbor-docker
tau2
```

Lease validation resolves exact device UUIDs and rejects overlapping live
leases. A stale lease may be recovered only after process and device checks
prove no owner remains. Do not delete or override an ambiguous lease.

Routine policy:

- CPU activities may run concurrently within fixed worker limits;
- one-GPU jobs run only on explicitly assigned devices;
- routine 8-GPU evidence is serialized;
- Harbor Docker access is serialized initially;
- benchmark simulators use task-specific concurrency limits;
- no automatic fallback to more GPUs, a different topology, or a different
  model when a lease is unavailable.

## 14. Tracing And Experiment Tracking

Local evidence is authoritative. The lifecycle indexes rather than replaces
existing loggers.

Common correlation fields:

```text
run_id attempt_id workflow_id activity_id stage_id stage_invocation_id
family task lane arm split seed
process rank local_rank mesh device_uuid
checkpoint_id parent_checkpoint_id policy_version
```

The attempt journal captures outer spans. Core SFT structured JSONL captures
training spans and scalars. RL typed metrics and structured traces capture
controller, generator, trainer, buffer, batching, and sync behavior. External
harness raw artifacts remain intact.

Required indexed locations include:

- stage stdout/stderr;
- structured JSONL;
- full-precision scalar files;
- TensorBoard event directories;
- checkpoint events;
- upstream harness job/trial files;
- canonical report input.

TensorBoard is the default local scalar dashboard. W&B is an optional one-way
mirror whose project/run reference is recorded. Exporter failure cannot mutate
or invalidate local evidence. No remote service stores the only copy of a
metric, checkpoint identity, or promotion decision.

All projections follow the secret and outbound-allowlist contract in Section
8.7. Correlation and environment capture record allowlisted variable names and
non-secret values only.

## 15. Failure Taxonomy

Normalize owner and retryability without collapsing measurement status.

| Class | Examples | Default retry |
| --- | --- | --- |
| declaration | invalid config, identity drift, unsupported profile | no |
| orchestration | worker loss, transient server connection | bounded if idempotent |
| rootfs | build, mount, permission, capability, signal propagation | after explicit fix |
| resource | lease conflict, missing GPU, insufficient memory | no silent retry |
| asset | missing or hash-mismatched model/data/image | after explicit acquisition/fix |
| model execution | load, generation, SFT, RL process failure | only from proven recovery point |
| checkpoint | partial, corrupt, incompatible, ambiguous lineage | no |
| verifier/evaluator | crash, canonical failure, semantic mismatch | no score; fix and rescore |
| harness | task load, reset, provider, official result failure | according to owner and idempotence |
| policy/task | wrong answer/action, benchmark timeout, reward zero | successful real measurement |
| evidence | missing log, output mutation, report mismatch | no promotion |
| scientific | regression, inconclusive interval, guardrail failure | real measurement; hold/reject |

Stable blocker codes may refine these classes. `score_regression` belongs to
scientific promotion, not the non-score failure table.

## 16. Exit Semantics

The shell facade uses stable process meanings:

- `0`: operational completion, including a scientifically valid hold/reject;
- `2`: invalid declaration, arguments, profile, or capability;
- `10`: preflight blocker with canonical blocker report;
- `20`: execution failure with preserved attempt evidence;
- `30`: evidence, artifact, or report-integrity failure;
- `128+signal`: signal termination where the shell can preserve it.

An external harness may return zero while trials fail. Final execution and
measurement status comes from validated official artifacts, not process exit
alone.

## 17. Validation And Failure-Injection Matrix

### 17.1 Unit and host integration

- declaration normalization and digest stability;
- duplicate run and attempt rejection;
- append-only start/terminal events;
- crash recovery with unmatched start;
- artifact before/after and hash validation;
- atomic publish and collision behavior;
- legacy import and schema migration;
- explicit status-dimension validation;
- latest valid/latest attempt indexing;
- doctor profile composition and optional clauses;
- rootfs destination canonicalization, allowlist, ownership-marker, exact file
  type, symlink, broad-target, and selection-record validation;
- environment-bundle lock, digest, immutability, and measured-stage mutation
  denial;
- fake command executor, clock, and temporary filesystem;
- Temporal workflow tests and history replay;
- deterministic Workflow-ID start conflict/reuse and repeated Workflow Task
  failure behavior;
- duplicate Activity delivery, attempt-key separation, immutable producer work
  status, and acknowledgement-loss delivery disposition;
- cancellation completion waiting, complete process/cgroup/Docker cleanup, and
  retry classification.

Host unit and static tests do not require rootfs, model assets, CUDA, or
Temporal server availability.

### 17.2 Rootfs CPU integration

- pinned build-manifest validation;
- attempts to use `/`, the home/repository/workspace/store root, an out-of-store
  path, symlinked parent or target, mount point, wrong file type, and unrelated
  existing directory as `--dest` or missing `--rootfs` fail before Docker,
  cleanup, or any target mutation;
- staged export validation, immutable content-addressed publication, atomic
  selection, launch-check rollback, interrupted activation recovery, and
  cleanup limited to an exact owned staging directory;
- per-digest launch leases keep an old selected rootfs alive across activation;
  garbage collection refuses selected, leased, mounted, or live-process-used
  digests and deletes only one enumerated owned entry;
- read-only base and declared writable mounts;
- environment allowlist;
- offline and acquisition network profiles;
- an acquisition stage publishes a locked environment bundle, while its
  measured consumer cannot invoke package/VCS acquisition or mutate the
  bundle, shared rootfs, or source snapshot;
- command argv, working directory, exit, and signal propagation;
- source and host-path mapping;
- output atomicity;
- Docker capability absent by default.

### 17.3 GPU integration

- exact GPU UUID allocation and lease collision;
- vLLM one-GPU load/generation and memory headroom;
- short one-GPU SFT checkpoint/resume/export;
- representative eight-GPU SFT numerics;
- deterministic RL trainer/generator parity;
- async policy-age, capacity, and slot-release invariants;
- process-tree cancellation for torchrun, Monarch, and vLLM.

### 17.4 External harness integration

- Harbor oracle and no-op controls;
- Docker host-path and image identity;
- official job/trial result ingestion;
- tau2 data check, clean reset, scripted/no-op controls, and results ingestion;
- benchmark failure versus infrastructure failure classification.

### 17.5 Fault injection

- worker death before launch, during work, and after output commit;
- Temporal server restart with persistent database;
- Activity completion with lost acknowledgement;
- client start retry and conflicting Workflow ID;
- repeated Workflow Task exception without history progress;
- subprocess hang and heartbeat timeout;
- SIGTERM and forced cancellation, including a daemon-owned Harbor container;
- incomplete or corrupt artifact/checkpoint;
- output mutation after commit;
- duplicate GPU/path lease request;
- rootfs start failure;
- rootfs export validation failure, activation interruption, corrupt selection
  record, and failed post-activation launch with verified rollback;
- network acquisition interruption;
- package or VCS mutation attempted from an offline measured stage;
- live-server backup interruption, corrupt SQLite backup, incompatible restore
  manifest, replay failure, and receipt-reconciliation mismatch;
- external job already running or already completed;
- report build failure after a valid score.

Every injected failure has an expected event sequence, retry decision, artifact
state, and report outcome. A failure test cannot pass merely because the
process exited nonzero.

## 18. Migration Inventory

### 18.1 Already using `run_common.sh`

Current shared-lifecycle users include the AIME, ARC-AGI-2,
BigCodeBench-Hard, GPQA, LiveCodeBench, and MMLU-Pro model runners plus GPQA
access preflight. They still require migration from the prototype JSONL writer
to the typed lifecycle and profile doctor.

### 18.2 Direct wrappers to migrate

Audit and migrate the existing runners for:

- arithmetic words fixture and vLLM;
- GSM fixture and GSM8K;
- MATH;
- HumanEval and MBPP;
- modular fixture, calibration, and transfer;
- external-harness dry-run and preflight;
- tau2 score and execution probes;
- Terminal-Bench/Harbor probes.

Countdown runners use similar concepts and should migrate after one scaffold
reference runner proves compatibility.

Keep existing script names as thin compatibility entrypoints. Do not rename
scientific conditions or alter defaults during lifecycle migration.

### 18.3 Runtime-mutation migration

Treat the scripts listed in Section 8.5 as an explicit burn-down list. First
extract their package, source, and image setup into digest-verifying acquisition
Activities. Then make each existing script accept only a declared sealed bundle
and remove its runtime installer, clone, fetch, checkout, or image-pull path.
Finally run the compatibility entrypoint with outbound network denied and the
rootfs, bundle, and source snapshot mounted read-only.

The migration exit is mechanical: repository search finds no package-manager or
VCS mutation reachable from a measured stage; profile tests deny those
executables in science contexts; every environment receipt resolves to a
content-addressed locked bundle; and repeat executions preserve the declared
environment digest. A smoke label does not exempt a run whose score is used as
scientific evidence.

## 19. Implementation Waves

### Wave F0: Truth-preserving fixes

- Add regression tests reproducing manifest truncation.
- Make attempt creation exclusive and append-only.
- Define explicit status dimensions and legacy translations.
- Fix latest indexing to separate latest valid from latest attempted.
- Require report identity validation.
- Correct roadmap references to actual runners.
- Reject report claims whose pass@k exceeds the actual rollout budget, and
  require an explicit shift contract before an `ood` label drives promotion.
- Reproduce the arbitrary-`--dest` deletion path in a non-destructive fixture,
  then add the shared canonical target resolver, allowlist, exact type/symlink
  checks, ownership requirement, and broad-target refusals before automatic
  rootfs construction can remain enabled.
- Disable implicit missing-rootfs construction during this interim wave; every
  missing default or custom `--rootfs` fails closed until F3 activation and
  rollback are proven.

Done when nested doctor, retry, and same-run-ID tests cannot erase evidence and
no blocker is encoded as a score, and hostile destination fixtures cannot cause
Docker invocation, cleanup, deletion, or mutation outside an owned staging
directory.

Status: landed. The append-only manifest and distinct attempt markers live in
`run_common.sh:scaffold_setup_run_manifest`; explicit status dimensions and
legacy translation in `torchtitan/experiments/scaffold_to_policy/execution_status.py`;
latest-valid versus latest-attempt indexing plus report-identity and
pass@k-budget validation in `report_artifacts.py`; and the shared fail-closed
destination resolver, ownership marker, and disabled implicit construction in
`scripts/rootfs/rootfs_target.sh`, `build_rootfs.sh`, and `enter_rootfs.sh`.
Regression coverage is in `tests/unit_tests/test_scaffold_execution_foundation.py`.

### Wave F1: Typed lifecycle

- Implement declaration, attempt, stage, artifact, outcome, and report models.
- Implement atomic JSON/JSONL writers and validators.
- Add command executor and report-adapter seams.
- Provide `begin`, `stage`, and `finish` CLI operations.
- Delegate `run_common.sh` to the CLI.

Done when a fake multi-stage run, a blocked run, a failed run, and an
interrupted attempt followed by a linked resume attempt all produce validated
immutable bundles.

### Wave F2: Profile doctor and semantic preflight

- Split generic doctor clauses from task preflight.
- Implement `rootfs_cpu`, `vllm_1gpu`, and representative reasoning profiles.
- Add `run_preflight.sh` as a query over profiles, not a manifest initializer.
- Attach doctor and preflight artifacts to canonical reports.

Done when optional package absence affects only its declaring profile and host
unit tests remain ungated.

### Wave F3: Rootfs identity and capability hardening

- Pin build inputs and emit the build manifest.
- Replace delete-and-move export with an allowlisted immutable
  content-addressed store, staged structural/digest/smoke validation, atomic
  selection, retained prior selection, and tested rollback.
- Add read-only base/source modes and explicit writable mounts.
- Add environment and capability policy.
- Move the Section 8.5 mutation inventory into locked content-addressed
  environment/source/image acquisition and prohibit package-manager, VCS, and
  shared-rootfs mutation in measured execution.
- Prove GPU/distributed behavior before changing defaults.

Done when two builds from the same locked inputs have an explainable identity,
unsafe destinations and interrupted activation cannot damage the prior rootfs,
rollback and live-launch retention are proven, every listed runner consumes a
read-only locked environment, runtime mutation is detected, and normal profiles
cannot access Docker or undeclared host paths.

### Wave F4: Temporal durability

- Add optional dependency and local server/worker commands.
- Implement run and campaign workflows.
- Wrap lifecycle stages in coarse Activities.
- Add task queues and local resource leases.
- Implement heartbeats, cancellation, idempotency, and bounded retry policy.
- Add replay/version tests and the restrictive-permission, versioned online or
  quiesced SQLite backup, integrity, isolated restore, rollback, and local
  receipt-reconciliation drill.
- Prove that payload, heartbeat, exception, history, log, and backup scans
  contain no secret material.

Done when server restart, worker restart, lost acknowledgement, cancellation,
duplicate delivery, and a corrupt or interrupted backup/restore preserve one
unambiguous local artifact outcome without overwriting the prior Temporal state
or exposing secrets.

### Wave F5: Runner migration

- Migrate one synthetic reasoning transfer end to end.
- Migrate Countdown.
- Migrate public reasoning and coding runners.
- Migrate SFT export/evaluation lineage.
- Migrate deterministic then async RL.
- Migrate Harbor and tau2.

Done when no serious runner writes the prototype manifest directly and all
canonical reports validate attempt lineage.

### Wave F6: Experiment views and operations

- Index structured SFT, RL, task, and lifecycle evidence into the upstream
  certified run-attempt schema.
- Add local campaign/status summaries.
- Add TensorBoard views and optional W&B export.
- Add stale-workflow, stale-lease, and no-progress alerts.
- Document backup, replay, recovery, and incident handling.

This wave does not postpone observability until after runner migration. Core
Tier 0 evidence, clock semantics, checkpoint lineage, capture escalation, and
the 1% overhead gate are prerequisites. F6 supplies experiment-specific joins,
views, mirrors, and operations after the lifecycle migration is stable.

Done when an operator can distinguish queued, running, progressing, blocked,
failed, interrupted, completed, measured, and promoted states without opening
raw logs.

## 20. Foundation Promotion Gate

The infrastructure is ready for expensive replicated research only when:

- attempts are immutable and append-only;
- every serious run has a complete evaluation identity;
- rootfs inputs and runtime capabilities are recorded and validated;
- doctor profiles are composable and task preflight preserves semantics;
- artifact freshness and checkpoint lineage are evidence-based;
- the canonical report is singular and validates stage evidence;
- Temporal restart/retry/cancellation behavior is proven locally;
- GPU leases prevent duplicate local work;
- SFT resume and export gates pass;
- RL Activities remain non-retrying until joint recovery passes;
- focused unit, shell, rootfs, GPU, harness, and fault-injection suites pass;
- timestamped reports clearly separate new evidence from imported history.
