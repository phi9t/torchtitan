# Core Run Evidence Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every core TorchTitan training process a shared run/attempt identity, a versioned repo-local manifest, additive structured-event correlation, process-local terminal evidence, and an append-only index of existing native artifacts.

**Architecture:** Add one deep `RunEvidence` module with a process-global facade because core TorchTitan already runs one `Trainer` and one structured logger per process. The facade installs a local recorder only in `torchtitan.train`; shared profiler, metrics, checkpoint, and distributed modules call a no-op-safe `record_artifact(...)` interface, so Forge, TorchFT, online RL, and direct unit-test construction do not acquire a new runtime dependency or constructor argument. Native artifact formats and compatibility paths remain unchanged and are referenced from `dump_folder/run_evidence/<run_id>/<attempt_id>/`.

**Tech Stack:** Python standard library, existing PyTorch runtime metadata, JSON/JSONL, `contextvars`, `pytest`, TorchTitan `Configurable` dataclasses.

## Global Constraints

- This plan implements only the first observability milestone from `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md` and `docs/research/2026-08-12-training-observability-paper-closure.md`.
- Keep the dependency direction `experiments -> core`; do not import optional experiment or diagnostic packages into core.
- Keep current structured-log, profiler, memory-snapshot, Flight Recorder, TensorBoard, and checkpoint data formats and compatibility paths.
- Do not add a synchronous cross-rank collector, remote service, directory scan, content hash, or remote call to the training hot path.
- Tier 0 remains enabled by default and must later pass the approved no-more-than-1% steady-state throughput gate; this milestone adds unit/static proof, not the 2/4/8-GPU performance certification.
- `run_id` identifies a logical training run; `attempt_id` identifies one launch or elastic restart. Every worker receives both from launcher-owned environment variables before its first structured event.
- Raw manifest, artifact-index, and process-outcome records are immutable or append-only. Derived analysis is out of scope.
- A Tier-0 initialization or synchronous append failure is explicit and fatal; evidence errors must not silently disappear or replace an already-active training exception during cleanup.
- New comments and docstrings use ASCII.
- Do not add capture triggers, incident classification, automatic recovery, quarantine, DCGM, py-spy, NCCL RAS queries, Nsight, module/kernel ranges, or RL adoption in this plan.

---

### Task 1: Deep Run Evidence Module

**Files:**
- Create: `torchtitan/observability/run_evidence.py`
- Create: `tests/unit_tests/observability/test_run_evidence.py`

**Interfaces:**
- Consumes: `Configurable`, process environment, normalized `Trainer.Config.to_dict()` output, and local filesystem paths.
- Produces:

```text
class RunEvidenceError(RuntimeError)
class EvidenceContractError(RunEvidenceError)
class EvidenceCollisionError(RunEvidenceError)
class EvidenceWriteError(RunEvidenceError)

class RunEvidence(Configurable)
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config)
        enable: bool = True
        folder: str = "run_evidence"

    RunEvidence.__init__(
        self,
        config: Config,
        *,
        dump_folder: str,
        job_config: Mapping[str, Any],
        role: str,
        actor_id: str,
    ) -> None

    RunEvidence.__enter__(self) -> RunEvidence
    RunEvidence.__exit__(self, exc_type, exc_value, traceback) -> bool
    RunEvidence.bind_distributed(
        self,
        parallel_dims: ParallelDims,
        device: torch.device,
    ) -> None

class ArtifactState(str, enum.Enum)
    DECLARED = "declared"
    COMPLETE = "complete"
    FAILED = "failed"
    RETIRED = "retired"

class ArtifactRelation(str, enum.Enum)
    INPUT = "input"
    OUTPUT = "output"

record_artifact(
    *,
    producer: str,
    kind: str,
    path: str | os.PathLike[str],
    state: ArtifactState = ArtifactState.COMPLETE,
    relation: ArtifactRelation = ArtifactRelation.OUTPUT,
    artifact_id: str | None = None,
    step: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str | None

event_context() -> dict[str, Any]
bind_phase(phase: str) -> AbstractContextManager[None]
```

- `record_artifact(...)` returns `None` when no enabled `RunEvidence` is installed; shared producers need no feature branch.
- `event_context()` returns an empty dict when disabled and otherwise returns a fresh flat correlation mapping with a strictly increasing structured-event sequence.
- An append failure raises `EvidenceWriteError` when no other exception is active. If called while propagating another exception, it logs the evidence failure and preserves the original exception.
- Internal helpers and concrete recorder classes stay private to the module.

- [ ] **Step 1: Write failing configuration and identity tests**

Add tests that pin the default config, reject an empty/absolute/traversing evidence folder, reject a remote `dump_folder` while evidence is enabled, auto-generate IDs only for a single-process launch, accept launcher-provided `TORCHTITAN_RUN_ID` and `TORCHTITAN_ATTEMPT_ID`, append `TORCHELASTIC_RESTART_COUNT` to the attempt, and reject a multi-process launch without shared IDs.

Define these local test helpers at the top of the file; later examples in this task use them:

```python
@pytest.fixture
def launcher_identity(monkeypatch):
    identity = SimpleNamespace(run_id="research-run", attempt_id="launch-17")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", identity.run_id)
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", identity.attempt_id)
    monkeypatch.delenv("TORCHELASTIC_RESTART_COUNT", raising=False)
    return identity


def build_evidence(tmp_path):
    return RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 3}},
        role="trainer",
        actor_id="core",
    )


def read_artifact_rows(tmp_path, identity):
    index = next(
        (
            tmp_path
            / "run_evidence"
            / identity.run_id
            / identity.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    return [json.loads(line) for line in index.read_text().splitlines()]
```

```python
def test_multi_process_run_requires_launcher_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("TORCHTITAN_RUN_ID", raising=False)
    monkeypatch.delenv("TORCHTITAN_ATTEMPT_ID", raising=False)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")

    evidence = RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 1}},
        role="trainer",
        actor_id="core",
    )

    with pytest.raises(ValueError, match="TORCHTITAN_RUN_ID"):
        evidence.__enter__()
```

- [ ] **Step 2: Run identity tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/observability/test_run_evidence.py \
  -k 'config or identity or multi_process'
```

Expected: collection fails because `torchtitan.observability.run_evidence` does not exist.

- [ ] **Step 3: Implement config validation and launcher identity resolution**

Implement these exact precedence and validation rules:

```text
run_id:
  TORCHTITAN_RUN_ID
  -> non-empty TORCHELASTIC_RUN_ID other than "none"
  -> UUID only when WORLD_SIZE == 1

attempt base:
  TORCHTITAN_ATTEMPT_ID
  -> non-empty TORCHELASTIC_RUN_ID other than "none"
  -> UUID only when WORLD_SIZE == 1

attempt_id:
  <attempt base>-restart-<TORCHELASTIC_RESTART_COUNT> when restart count > 0
  <attempt base> otherwise
```

Accept only identifiers matching `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. `Config.folder` must be one relative path segment and default to `run_evidence`.

- [ ] **Step 4: Write failing manifest and collision tests**

Cover canonical sorted JSON, SHA-256 config digest, schema version `1`, normalized config, source revision/dirty fields, command/runtime metadata, shared manifest verification, and deterministic process-index collision detection.

```python
def test_manifest_and_process_index_are_created(tmp_path, launcher_identity):
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 3}},
        role="trainer",
        actor_id="core",
    ):
        pass

    attempt = (
        tmp_path
        / "run_evidence"
        / launcher_identity.run_id
        / launcher_identity.attempt_id
    )
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["config"]["normalized"] == {"training": {"steps": 3}}
    assert len(manifest["config"]["sha256"]) == 64
    assert list((attempt / "indexes").glob("artifacts.*.jsonl"))
```

- [ ] **Step 5: Run manifest tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/observability/test_run_evidence.py \
  -k 'manifest or collision or digest'
```

Expected: FAIL because the manifest and index writer are not implemented.

- [ ] **Step 6: Implement immutable manifest publication and process index ownership**

Write canonical JSON with `sort_keys=True`, compact separators for hashing, and a trailing newline for files. Publish `manifest.json` with exclusive creation; on `FileExistsError`, retry reading boundedly to handle a concurrent writer, then require equality of schema, identity, normalized config digest, source state, command, and runtime versions. Create exactly one index file per deterministic core process identity:

For example, ranks 0 and 1 own `trainer.core.global_rank_000000` and
`trainer.core.global_rank_000001` respectively.

Open `indexes/artifacts.<process_id>.jsonl` with exclusive creation so reusing an attempt ID for a fresh process fails before training. Use global rank from the environment; include host, PID, local rank, and world size in records rather than in the shared manifest.

- [ ] **Step 7: Write failing artifact lifecycle and correlation tests**

Test no-op behavior before installation, deterministic artifact IDs, valid transitions (`declared -> complete|failed|retired`, `complete -> retired`), rejected backward/duplicate transitions, path normalization relative to `dump_folder`, preservation of external absolute paths/URIs, JSON-serializable metadata validation, local-path existence for `COMPLETE`, wall/monotonic clocks, process-local artifact sequence, phase nesting, distributed context binding, fatal append errors on an otherwise healthy path, and preservation of an already-active exception when append also fails.

```python
def test_artifact_index_is_append_only(tmp_path, launcher_identity):
    with build_evidence(tmp_path):
        trace_path = tmp_path / "profiling" / "trace.json.gz"
        artifact_id = record_artifact(
            producer="pytorch_profiler",
            kind="pytorch.profiler.trace",
            path=trace_path,
            state=ArtifactState.DECLARED,
            step=4,
        )
        trace_path.parent.mkdir(parents=True)
        trace_path.write_bytes(b"compressed-trace")
        record_artifact(
            producer="pytorch_profiler",
            kind="pytorch.profiler.trace",
            path=trace_path,
            state=ArtifactState.COMPLETE,
            artifact_id=artifact_id,
            step=4,
        )

    rows = read_artifact_rows(tmp_path, launcher_identity)
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert rows[0]["artifact_id"] == rows[1]["artifact_id"]
```

- [ ] **Step 8: Run lifecycle tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/observability/test_run_evidence.py \
  -k 'artifact or event_context or phase or distributed'
```

Expected: FAIL because the facade, lifecycle validation, and correlation context are incomplete.

- [ ] **Step 9: Implement append-only artifacts, correlation context, and process outcomes**

Use one process-local lock and one append-only file descriptor. Derive artifact IDs from schema version, run ID, attempt ID, producer, kind, relation, and normalized path so all ranks observing one distributed checkpoint use the same artifact identity. Require completed local files/directories to exist; URI completion remains producer-asserted. Each index row contains:

```json
{
  "schema_version": 1,
  "record_type": "artifact",
  "artifact_id": "a73ba08d8f6a49b18f7378e15e2f3fa7",
  "producer": "pytorch_profiler",
  "kind": "pytorch.profiler.trace",
  "relation": "output",
  "state": "complete",
  "path": "profiling/trace.json.gz",
  "path_type": "dump_relative",
  "wall_time_ns": 1786543200123456789,
  "monotonic_ns": 908765432101234,
  "artifact_seq": 0,
  "run_id": "research-qwen3-small",
  "attempt_id": "launch-17",
  "process_id": "trainer.core.global_rank_000000",
  "role": "trainer",
  "actor_id": "core",
  "host_name": "trainer-node-0",
  "pid": 41237,
  "global_rank": 0,
  "local_rank": 0,
  "world_size": 1,
  "step": 4,
  "phase": "profiling",
  "metadata": {}
}
```

Omit fields that were not collected; use explicit `null` only after a collection attempt returned an unknown value. On context exit, atomically create `processes/<process_id>/outcome.json` with `succeeded`, `failed`, or `interrupted`, elapsed monotonic time, and bounded exception type/message. Preserve an active training exception if outcome writing also fails; raise the evidence error when no earlier exception exists.

- [ ] **Step 10: Run the complete module test file**

Run:

```bash
.venv/bin/pytest -q tests/unit_tests/observability/test_run_evidence.py
```

Expected: PASS.

- [ ] **Step 11: Commit the deep module**

```bash
git add \
  torchtitan/observability/run_evidence.py \
  tests/unit_tests/observability/test_run_evidence.py
git commit -m "Add core run evidence recorder"
```

### Task 2: Core Bootstrap and Structured Correlation

**Files:**
- Modify: `torchtitan/trainer.py`
- Modify: `torchtitan/train.py`
- Modify: `torchtitan/observability/structured_logger/structured_logging.py`
- Modify: `torchtitan/observability/structured_logger/jsonl_handler.py`
- Modify: `torchtitan/observability/structured_logger/__init__.py`
- Modify: `tests/unit_tests/observability/test_structured_logging.py`
- Create: `tests/unit_tests/test_train_run_evidence.py`
- Modify: `tests/unit_tests/test_config_manager.py`

**Interfaces:**
- Consumes: `RunEvidence.Config`, `RunEvidence` context manager, `event_context()`, `bind_phase()`, and `record_artifact(...)` from Task 1.
- Produces: `Trainer.Config.run_evidence`, core entrypoint lifecycle, additive fields on every structured JSONL record, and `close_structured_logger()`.

- [ ] **Step 1: Write failing config and entrypoint lifecycle tests**

Pin that every `Trainer.Config` gets enabled evidence by default, the field serializes through `to_dict()`, CLI overrides can disable it/change its folder, and `train.main()` writes a success outcome on the `local_tensor` early-return path and a failed outcome when config build raises.

Keep helpers local to `test_train_run_evidence.py`: construct the real debug config through `ConfigManager.parse_args(["--module", "llama3", "--config", "llama3_debugmodel", "--comm.mode=local_tensor"])`, and patch only `ConfigManager`, logger setup, and trainer construction when exercising `train.main()`.

```python
def test_trainer_config_owns_run_evidence():
    config = make_trainer_config()
    assert config.run_evidence.enable is True
    assert config.run_evidence.folder == "run_evidence"
    assert config.to_dict()["run_evidence"]["enable"] is True
```

- [ ] **Step 2: Run focused config/entrypoint tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_config_manager.py \
  tests/unit_tests/test_train_run_evidence.py
```

Expected: FAIL because `Trainer.Config.run_evidence` and entrypoint lifecycle wiring do not exist.

- [ ] **Step 3: Add the owner config and wrap core training in the evidence lifecycle**

Add:

```python
run_evidence: RunEvidence.Config = field(default_factory=RunEvidence.Config)
```

Build it after config parsing and before structured logger initialization:

```python
with config.run_evidence.build(
    dump_folder=config.dump_folder,
    job_config=config.to_dict(),
    role="trainer",
    actor_id="core",
):
    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )
    try:
        trainer = config.build()
        trainer.train()
    finally:
        sl.close_structured_logger()
```

Retain the current `local_tensor`, seed-checkpoint, trainer cleanup, and process-group
teardown branches inside this lifecycle; the abbreviated body above shows ownership
and ordering rather than replacing those branches.

Do not install evidence in online RL, Forge, TorchFT, or offline-program entrypoints in this milestone.

- [ ] **Step 4: Write failing structured correlation and close tests**

Extend the existing fixture to reset both structured logging and active evidence. Assert legacy fields remain unchanged and enabled evidence adds `evidence_schema_version`, run/attempt/process/role/actor identity, `wall_time_ns`, `monotonic_ns`, `event_seq`, and the current nested phase. Assert device UUID and mesh-axis ranks appear only after `bind_distributed(...)`. Assert `close_structured_logger()` closes/removes handlers, is idempotent, and allows test reinitialization.

Add a file-local launcher identity fixture with the same fixed environment values from Task 1 and a file-local `read_only_structured_row(tmp_path)` helper. Do not import test helpers across test modules.

```python
def test_formatter_adds_run_evidence_without_changing_legacy_fields(
    tmp_path, structured_logger_fixture, launcher_identity
):
    with build_evidence(tmp_path):
        init_structured_logger(rank=0, source="training", output_dir=str(tmp_path))
        log_trace_instant("training_start")
        row = read_only_structured_row(tmp_path)
        close_structured_logger()

    assert row["source"] == "training"
    assert row["global_rank"] == 0
    assert row["run_id"] == launcher_identity.run_id
    assert row["attempt_id"] == launcher_identity.attempt_id
    assert row["event_seq"] == 0
```

- [ ] **Step 5: Run structured tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/observability/test_structured_logging.py \
  -k 'evidence or phase or close'
```

Expected: FAIL because structured records are not enriched and the close interface is missing.

- [ ] **Step 6: Enrich each LogRecord once and bind span phase context**

Have `event_extra(...)` attach one private evidence mapping from `event_context()` to the `LogRecord`; merge it flat in `TraceJsonlFormatter`. This keeps every handler for one record on the same `event_seq`. In `log_trace_span`, enter `bind_phase(base_name)` before the start event and reset it after error/end emission, including nested and asynchronous spans. Preserve the existing `torch.compiler.is_compiling()` no-op behavior.

Add `close_structured_logger()` to close/remove every handler and reset the logger lifecycle flags without changing the default handler factory interface.

- [ ] **Step 7: Register the native structured log lifecycle**

In `TraceJsonlHandler.__init__`, register its existing path as:

```python
record_artifact(
    producer="structured_logger",
    kind="torchtitan.structured_events",
    path=filepath,
    state=ArtifactState.DECLARED,
    metadata={"format": "jsonl", "source": source},
)
```

On handler close, append `COMPLETE` with the same artifact ID before the evidence context exits. If no evidence is active, both calls remain no-ops. If close runs while another exception is propagating, preserve that original exception according to the central `record_artifact(...)` contract.

- [ ] **Step 8: Bind device and mesh evidence immediately after mesh construction**

In `Trainer.__init__`, call the active evidence recorder after `self.init_distributed()` returns. Record the device UUID using the existing PyTorch device-properties interface and axis-local ranks/sizes from public `ParallelDims` mesh accessors. Do not reach into `_global_meshes` or invent a one-dimensional mesh assumption.

- [ ] **Step 9: Run the owning focused suite**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/observability/test_structured_logging.py \
  tests/unit_tests/observability/test_run_evidence.py \
  tests/unit_tests/test_train_run_evidence.py \
  tests/unit_tests/test_config_manager.py
```

Expected: PASS.

- [ ] **Step 10: Commit core bootstrap and correlation**

```bash
git add \
  torchtitan/trainer.py \
  torchtitan/train.py \
  torchtitan/observability/structured_logger/structured_logging.py \
  torchtitan/observability/structured_logger/jsonl_handler.py \
  torchtitan/observability/structured_logger/__init__.py \
  tests/unit_tests/observability/test_structured_logging.py \
  tests/unit_tests/test_train_run_evidence.py \
  tests/unit_tests/test_config_manager.py
git commit -m "Correlate core structured training evidence"
```

### Task 3: Profiler, Memory, Metrics, and Flight Recorder Indexing

**Files:**
- Modify: `torchtitan/tools/profiler.py`
- Modify: `torchtitan/components/metrics.py`
- Modify: `torchtitan/distributed/utils.py`
- Modify: `tests/unit_tests/test_profiler.py`
- Create: `tests/unit_tests/test_metrics_run_evidence.py`
- Create: `tests/unit_tests/test_distributed_run_evidence.py`

**Interfaces:**
- Consumes: no-op-safe `record_artifact(...)`, `ArtifactState`, and existing native producer paths.
- Produces: lifecycle rows for Kineto traces, CUDA memory snapshots, TensorBoard event directories, and declared Flight Recorder dump prefixes.

- [ ] **Step 1: Write failing profiler and memory registration tests**

Mock native export/snapshot writers but use a real temporary evidence bundle. Assert `DECLARED -> COMPLETE` around a successful export and `DECLARED -> FAILED` when export or post-processing raises. Pin exact producer/kind/format metadata and training step.

Each new Task 3 test file defines its own `active_evidence` fixture with fixed launcher environment values and a local `artifact_rows(...)` reader. Do not add a repository-wide `conftest.py`. Extract a private `Profiler._export_trace(prof, output_file, post_processor)` helper so the trace lifecycle is directly testable without constructing a CUDA profiler schedule.

```python
def test_trace_handler_registers_native_trace(tmp_path, active_evidence):
    profiler = Profiler(Profiler.Config(enable_profiling=True))
    fake_profiler = SimpleNamespace(
        step_num=7,
        export_chrome_trace=lambda path: Path(path).write_bytes(b"trace"),
    )
    output_file = tmp_path / "profiling" / "traces" / "rank0_trace.json.gz"
    profiler._export_trace(
        fake_profiler,
        output_file=str(output_file),
        post_processor=None,
    )
    rows = artifact_rows(active_evidence, kind="pytorch.profiler.trace")
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert rows[-1]["step"] == 7
```

- [ ] **Step 2: Run profiler tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q tests/unit_tests/test_profiler.py -k evidence
```

Expected: FAIL because profiler outputs are not registered.

- [ ] **Step 3: Register Kineto and memory snapshot lifecycles**

Declare immediately before the native write, append `FAILED` in `except`, preserve and re-raise the original producer exception even if evidence append also fails, and append `COMPLETE` only after the file is closed and any configured trace post-processor succeeds. Use:

```text
producer=pytorch_profiler, kind=pytorch.profiler.trace, format=chrome_trace_json_gzip
producer=pytorch_memory, kind=pytorch.cuda.memory_snapshot, format=python_pickle_v4
```

Do not move or copy files.

- [ ] **Step 4: Write failing TensorBoard lifecycle tests**

Construct `TensorBoardLogger` directly in a temporary evidence session. Assert it declares `tensorboard.event_stream` at initialization, completes it once on close, and remains unchanged/no-op without active evidence. Do not index W&B as canonical lineage.

- [ ] **Step 5: Run TensorBoard tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q tests/unit_tests/test_metrics_run_evidence.py
```

Expected: FAIL because TensorBoard does not register its native log directory.

- [ ] **Step 6: Register TensorBoard event-directory lifecycle**

Store the returned artifact ID on `TensorBoardLogger`, declare after `SummaryWriter` successfully opens, and complete after `writer.close()`. Preserve the existing constructor and metric-logging interface, including preservation of an active writer/training exception if completion evidence also fails.

- [ ] **Step 7: Extract and test Flight Recorder environment configuration**

Extract the existing environment setup from `init_distributed(...)` into a private helper that accepts `CommConfig` and `base_folder`. Test that a positive buffer sets the same three environment variables and declares one artifact with `path_semantics="prefix"`; a zero buffer declares nothing. Preserve the current `TORCH_NCCL_ASYNC_ERROR_HANDLING=3` behavior.

```python
def test_flight_recorder_prefix_is_declared(tmp_path, active_evidence, monkeypatch):
    config = CommConfig(
        trace_buf_size=32,
        save_traces_folder="comm_traces",
        save_traces_file_prefix="rank_",
    )
    _configure_flight_recorder(config, str(tmp_path))
    rows = artifact_rows(active_evidence, kind="pytorch.flight_recorder.dump")
    assert rows[0]["state"] == "declared"
    assert rows[0]["metadata"]["path_semantics"] == "prefix"
```

- [ ] **Step 8: Run Task 3 tests**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_profiler.py \
  tests/unit_tests/test_metrics_run_evidence.py \
  tests/unit_tests/test_distributed_run_evidence.py
```

Expected: PASS.

- [ ] **Step 9: Commit native diagnostic indexing**

```bash
git add \
  torchtitan/tools/profiler.py \
  torchtitan/components/metrics.py \
  torchtitan/distributed/utils.py \
  tests/unit_tests/test_profiler.py \
  tests/unit_tests/test_metrics_run_evidence.py \
  tests/unit_tests/test_distributed_run_evidence.py
git commit -m "Index native training diagnostic artifacts"
```

### Task 4: Checkpoint Input and Output Lifecycle Evidence

**Files:**
- Modify: `torchtitan/components/checkpoint.py`
- Modify: `tests/unit_tests/test_checkpoint.py`
- Verify: `torchtitan/experiments/torchft/checkpoint.py`
- Verify: `torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py`

**Interfaces:**
- Consumes: no-op-safe artifact lifecycle from Task 1 and existing `CheckpointManager` sync/async completion seams.
- Produces: per-rank observations of checkpoint input/output declaration, success, and failure without changing DCP/HF data or checkpoint discovery.

- [ ] **Step 1: Write failing synchronous save/load evidence tests**

Activate a temporary evidence session around the existing fake DCP save/load fixtures. Pin:

Define the evidence fixture and JSONL reader inside `test_checkpoint.py`; preserve the file's existing `unittest.TestCase` style by installing/closing evidence in a dedicated context per evidence test rather than adding a global pytest fixture.

```text
producer=checkpoint
kind=torchtitan.checkpoint
relation=output for save
relation=input for load
format=torch_distributed_checkpoint or huggingface_safetensors
metadata includes step, model_only, and async_mode where applicable
```

Assert successful synchronous work appends `DECLARED -> COMPLETE`, while a native DCP exception appends `DECLARED -> FAILED` and remains the raised exception.

- [ ] **Step 2: Run synchronous checkpoint tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_checkpoint.py \
  -k 'evidence and (save or load)'
```

Expected: FAIL because checkpoint paths are not registered.

- [ ] **Step 3: Add private checkpoint artifact helpers**

Keep the public `CheckpointManager` interface unchanged. Add private helpers that declare and finish an artifact around a known `checkpoint_id`; preserve the original DCP/HF exception if evidence cleanup also fails. Each rank records its own observation with the same attempt/path-derived artifact ID.

- [ ] **Step 4: Write failing async completion tests**

For `async` and `async_with_pinned_mem`, assert save dispatch leaves the artifact `DECLARED`; `maybe_wait_for_saving()` appends `COMPLETE` only after `future.result()` succeeds and appends `FAILED` before re-raising a future error. `maybe_wait_for_staging()` must not claim durable completion.

```python
def test_async_checkpoint_completes_only_after_upload_wait(manager, active_evidence):
    manager.save(curr_step=5)
    assert artifact_states(active_evidence, "torchtitan.checkpoint") == ["declared"]
    manager.maybe_wait_for_saving()
    assert artifact_states(active_evidence, "torchtitan.checkpoint") == [
        "declared",
        "complete",
    ]
```

- [ ] **Step 5: Run async tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_checkpoint.py \
  -k 'evidence and async'
```

Expected: FAIL because durable completion is not connected to the pending save future.

- [ ] **Step 6: Implement pending artifact completion at the existing wait seam**

Store at most one pending `(artifact_id, checkpoint_id, step, metadata)` alongside the existing single `save_future`. Complete/fail it inside `maybe_wait_for_saving()` after `future.result()`. Last-step saves remain synchronous and complete before return. A process killed before completion deliberately leaves a declared artifact for postmortem analysis.

- [ ] **Step 7: Audit inherited and direct callers**

Run the base checkpoint suite and TorchFT checkpoint suite with no active core evidence. Confirm the no-op facade preserves Forge, RL trainer, Flux, TorchFT, and direct unit-test constructors. Do not instrument TorchFT's private per-replica dataloader checkpoint path in this core-first milestone.

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_checkpoint.py \
  torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py
```

Expected: PASS.

- [ ] **Step 8: Commit checkpoint lifecycle evidence**

```bash
git add \
  torchtitan/components/checkpoint.py \
  tests/unit_tests/test_checkpoint.py
git commit -m "Record checkpoint artifact lifecycles"
```

### Task 5: Launcher-Owned Shared Identity

**Files:**
- Modify: `run_train.sh`
- Modify: `multinode_trainer.slurm`
- Create: `tests/unit_tests/test_run_evidence_launchers.py`

**Interfaces:**
- Consumes: `TORCHTITAN_RUN_ID` and `TORCHTITAN_ATTEMPT_ID` identity contract from Task 1.
- Produces: one shared run ID and attempt base before `torchrun`, with user/platform overrides preserved.

- [ ] **Step 1: Write failing launcher contract tests**

Parse the scripts as text and assert both export the two variables before invoking `torchrun`, preserve pre-set values, and use `TORCHTITAN_RUN_ID` as the rendezvous ID. Pin that the local communication-mode path receives the same exported IDs.

- [ ] **Step 2: Run launcher tests and verify the red state**

Run:

```bash
.venv/bin/pytest -q tests/unit_tests/test_run_evidence_launchers.py
```

Expected: FAIL because launchers do not export evidence identity.

- [ ] **Step 3: Generate IDs once in each launcher before worker creation**

Use the launcher Python interpreter and standard-library UUIDs:

```bash
TORCHTITAN_RUN_ID=${TORCHTITAN_RUN_ID:-"$(python3 -c 'import uuid; print(uuid.uuid4())')"}
TORCHTITAN_ATTEMPT_ID=${TORCHTITAN_ATTEMPT_ID:-"$(python3 -c 'import uuid; print(uuid.uuid4())')"}
export TORCHTITAN_RUN_ID TORCHTITAN_ATTEMPT_ID
```

Pass `--rdzv_id "${TORCHTITAN_RUN_ID}"` in both launchers. Do not regenerate inside `srun`, a worker, or an elastic restart. The recorder adds the elastic restart suffix from `TORCHELASTIC_RESTART_COUNT`.

- [ ] **Step 4: Run launcher tests and shell syntax checks**

Run:

```bash
.venv/bin/pytest -q tests/unit_tests/test_run_evidence_launchers.py
bash -n run_train.sh
bash -n multinode_trainer.slurm
```

Expected: PASS.

- [ ] **Step 5: Commit launcher identity**

```bash
git add \
  run_train.sh \
  multinode_trainer.slurm \
  tests/unit_tests/test_run_evidence_launchers.py
git commit -m "Assign shared run evidence identity at launch"
```

### Task 6: Domain Language, Operator Documentation, and Milestone Verification

**Files:**
- Modify: `CONTEXT.md`
- Create: `docs/run_evidence.md`
- Modify: `torchtitan/observability/structured_logger/README.md`
- Modify: `docs/debugging.md`

**Interfaces:**
- Consumes: final schema, paths, environment variables, artifact kinds, and limitations from Tasks 1-5.
- Produces: canonical terminology and operator-facing instructions for finding and interpreting the bundle.

- [ ] **Step 1: Add canonical domain terms**

Add concise glossary entries:

```markdown
**Training Run**:
A logical training experiment that can span one or more launch, restart, or
recovery attempts while retaining one lineage identity.
_Avoid_: Job, process, attempt

**Run Attempt**:
One launch or elastic restart of a training run, with its own immutable evidence
and terminal process outcomes.
_Avoid_: Run, retry folder

**Evidence Artifact**:
A native log, trace, snapshot, metric stream, checkpoint, or diagnostic result
whose provenance is recorded in a run-attempt artifact index.
_Avoid_: Copied output, attachment
```

- [ ] **Step 2: Document the v1 run-attempt bundle and identity contract**

`docs/run_evidence.md` must include:

```text
dump_folder/run_evidence/run_id/attempt_id/
  manifest.json
  indexes/artifacts.<process_id>.jsonl
  processes/<process_id>/outcome.json
```

Document schema versioning, config/source capture, launcher precedence, elastic restart suffixes, correlation fields, path types, artifact lifecycle states, native artifact kinds, the disabled baseline, and how to join a structured event to an artifact row. Explicitly state that top-level multi-process outcome aggregation, incident records, external tools, triggers, retention enforcement, checkpoint semantic manifests, and offline analyzers are later milestones.

- [ ] **Step 3: Update structured logging and debugging references**

Explain that core `torchtitan.train` installs run evidence before the logger, while other surfaces remain no-op until separately adopted. Preserve existing structured-logger examples and add the new close/lifecycle behavior and correlation fields.

- [ ] **Step 4: Run focused and owning suites**

Run:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/observability/test_run_evidence.py \
  tests/unit_tests/observability/test_structured_logging.py \
  tests/unit_tests/test_train_run_evidence.py \
  tests/unit_tests/test_profiler.py \
  tests/unit_tests/test_metrics_run_evidence.py \
  tests/unit_tests/test_distributed_run_evidence.py \
  tests/unit_tests/test_checkpoint.py \
  tests/unit_tests/test_config_manager.py \
  tests/unit_tests/test_run_evidence_launchers.py \
  torchtitan/experiments/torchft/tests/test_torchft_checkpoint.py
```

Expected: PASS.

- [ ] **Step 5: Run a repo-local bootstrap smoke without spending a GPU training budget**

Run into a fresh gitignored output directory:

```bash
MODULE=llama3 \
CONFIG=llama3_debugmodel \
NGPU=2 \
COMM_MODE=local_tensor \
./run_train.sh --dump_folder=./outputs/run_evidence_local_tensor_smoke
```

Verify one manifest, one process artifact index, a declared/completed structured JSONL artifact, and a successful process outcome. This smoke proves launch/bootstrap plumbing only; it is not Tier-0 overhead, distributed scale, correctness, convergence, or reliability evidence.

- [ ] **Step 6: Run changed-file lint and static checks**

Run:

```bash
pre-commit run --files \
  CONTEXT.md \
  docs/run_evidence.md \
  docs/debugging.md \
  torchtitan/observability/structured_logger/README.md \
  torchtitan/observability/run_evidence.py \
  torchtitan/observability/structured_logger/structured_logging.py \
  torchtitan/observability/structured_logger/jsonl_handler.py \
  torchtitan/observability/structured_logger/__init__.py \
  torchtitan/train.py \
  torchtitan/trainer.py \
  torchtitan/tools/profiler.py \
  torchtitan/components/metrics.py \
  torchtitan/components/checkpoint.py \
  torchtitan/distributed/utils.py \
  run_train.sh \
  multinode_trainer.slurm \
  tests/unit_tests/observability/test_run_evidence.py \
  tests/unit_tests/observability/test_structured_logging.py \
  tests/unit_tests/test_train_run_evidence.py \
  tests/unit_tests/test_profiler.py \
  tests/unit_tests/test_metrics_run_evidence.py \
  tests/unit_tests/test_distributed_run_evidence.py \
  tests/unit_tests/test_checkpoint.py \
  tests/unit_tests/test_config_manager.py \
  tests/unit_tests/test_run_evidence_launchers.py
```

Expected: PASS. If the repository-wide Pyrefly baseline still reports the two previously documented unrelated diagnostics, record them separately and require zero diagnostics in changed files.

- [ ] **Step 7: Review manifest/index samples for secret leakage and schema consistency**

Inspect the smoke bundle. Confirm the manifest captures only an allowlist of launcher/runtime fields; it must not dump the complete environment. Confirm every row uses schema version `1`, the same run/attempt IDs, one process ID, monotonic and wall clocks, and normalized paths.

- [ ] **Step 8: Commit docs and final verification fixes**

```bash
git add \
  CONTEXT.md \
  docs/run_evidence.md \
  docs/debugging.md \
  torchtitan/observability/structured_logger/README.md
git commit -m "Document core run evidence contract"
```

### Task 7: Independent Review and Readiness Gate

**Files:**
- Review: every file changed by Tasks 1-6
- Compare against: `docs/superpowers/specs/2026-08-12-training-research-vehicle-design.md`
- Compare against: `docs/research/2026-08-12-training-observability-paper-closure.md`

**Interfaces:**
- Consumes: the completed implementation and verification evidence.
- Produces: standards review, spec review, and an explicit decision on whether the milestone is ready for the later 2/4/8-GPU perturbation certification.

- [ ] **Step 1: Request two-axis code review**

Use the repo's `code-review` skill from the implementation base commit. Require reviewers to check:

```text
Standards axis:
- core/experiment dependency direction
- config ownership and callsite audit
- structured-logging compatibility
- checkpoint/TorchFT compatibility
- no hidden hot-path collection or optional dependency
- exception preservation and append-only I/O correctness

Spec axis:
- shared identity before first event
- immutable versioned manifest
- process-local correlation and outcome
- native artifact index coverage
- disabled/no-op compatibility
- explicit deferral of triggers, tools, incidents, and scale certification
```

- [ ] **Step 2: Resolve every Critical or Important finding test-first**

For each accepted finding, add or tighten a failing focused test, reproduce the failure, implement the smallest root-cause fix, rerun the focused and owning suites, and ask the same reviewer to verify the fix. Do not accept speculative refactors outside this milestone.

- [ ] **Step 3: Run verification-before-completion**

Re-run Task 6 Steps 4-7 from the final HEAD. Capture exact pass counts, skipped optional/GPU tests, and any pre-existing unrelated repository diagnostics. Do not claim the 1% overhead gate or 2/4/8-GPU certification from this milestone.

- [ ] **Step 4: Prepare the next milestone boundary**

Record that the next independently designed milestone is the 2/4/8-GPU Qwen3 0.6B/1.7B Tier-0 perturbation and artifact-volume certification. Do not implement it in this branch.
