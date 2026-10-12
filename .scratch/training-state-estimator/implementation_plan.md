# Training State Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. The main agent is an orchestrator, not the task implementer. Each task gets fresh, precise subagent context, an independent review subagent, and a finalizer subagent that audits both traces before the orchestrator marks the task complete.

**Goal:** Build an offline, evidence-bundle-driven state-estimator analyzer for TorchTitan training jobs, with concrete data collection, classical modeling, probe recommendation, learned-extension, and evaluation paths.

**Architecture:** The first implementation is an offline analyzer under `torchtitan/observability/state_estimator/`. It reads immutable `run_evidence` bundles, writes deterministic derived artifacts under `derived/state_estimator/`, produces heuristic belief summaries and diagnosis reports, and defines optional learned-model extension interfaces without adding ML dependencies to the core analyzer.

**Tech Stack:** Python standard library, dataclasses, JSON/JSONL, pathlib, pytest, existing TorchTitan `run_evidence` bundle layout. No GPU, distributed launch, neural model library, or optional diagnostic tool is required for the initial analyzer. All Python execution for implementation, tests, static checks, and CLI usage must run inside the TorchTitan bwrap rootfs.

**Spec:** `.scratch/training-state-estimator/spec.md`

## Global Constraints

- The analyzer is offline and read-only over raw run-attempt evidence bundles.
- Raw evidence files must not be modified; derived artifacts are written under `derived/state_estimator/`.
- Estimator output is advisory and may not change optimizer commit, checkpoint validity, membership, or recovery behavior.
- Learned-model interfaces are optional and disabled by default.
- The core analyzer must import without `torch`, graph libraries, neural ODE/CDE packages, transformers, or GPU libraries.
- Missing optional evidence produces quality findings, not hard analyzer failure.
- Missing required manifest or invalid core identity fails graph construction with a clear error.
- Heuristic scores must be labeled as uncalibrated until evaluation artifacts prove calibration.
- Every implementation task uses rootfs-executed, host-scope tests first.
- All Python commands must be invoked through the repo-local bwrap wrapper:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && <python command>'
```

- The only code that may run outside rootfs is code that builds, verifies, or
  launches the bwrap rootfs itself.
- Business logic runs inside rootfs: PyTorch/modeling code, analyzer code,
  tests, static Python checks, data preparation, validation, log parsing,
  summarization, and CLI entrypoints.
- If a Python dependency is missing, add it inside rootfs with `uv` using the
  repo-local environment convention for the program. If a non-Python runtime,
  compiler, CLI, or tool is missing, add or pin it with `mise` unless a more
  specific rootfs provisioning script already owns it.

Do not run bare host-side `python`, `python3`, `pytest`, `pip`, or `py_compile`
commands for this effort. Host shell may orchestrate the rootfs wrapper and may
run non-Python file-inspection commands such as `rg`, `sed`, `git diff`, and
`git status`.

When command examples below show `python3 ...`, treat them as payloads inside
the rootfs wrapper unless the example explicitly includes the full wrapper.
Bare host-side Python execution is outside the plan and does not satisfy
verification.

---

## File Structure

Create a focused package:

```text
torchtitan/observability/state_estimator/
  __init__.py
  schema.py
  bundle.py
  fixtures.py
  graph.py
  estimators.py
  timeline.py
  probes.py
  learned.py
  evaluation.py
  cli.py
```

Responsibilities:

- `schema.py`: dataclasses, constants, JSON helpers, enums, quality finding model.
- `bundle.py`: raw run-evidence bundle reader and validation helpers.
- `fixtures.py`: test-only-style fixture builder usable by tests and docs examples. It must not import pytest.
- `graph.py`: derived evidence graph builder and artifact writer.
- `estimators.py`: deterministic phase, liveness, skew, and evidence-quality estimators.
- `timeline.py`: fixed-lag incident timeline reconstruction.
- `probes.py`: recommendation-only active diagnostic planner.
- `learned.py`: optional learned-likelihood extension schemas and null loader.
- `evaluation.py`: labeled fixture evaluation manifest, runner, and report writer.
- `cli.py`: one offline entrypoint that runs graph, estimators, timeline, probes, and optional evaluation output.

Tests:

```text
tests/unit_tests/observability/state_estimator/
  test_schema.py
  test_fixtures.py
  test_bundle.py
  test_graph.py
  test_estimators.py
  test_timeline.py
  test_probes.py
  test_learned.py
  test_evaluation.py
  test_cli.py
```

Documentation and tracker:

```text
.scratch/training-state-estimator/
  implementation_plan.md
  schema.md
  issues/*.md
```

---

### Task 1: Schema and Vocabulary Contract

**Files:**
- Create: `torchtitan/observability/state_estimator/__init__.py`
- Create: `torchtitan/observability/state_estimator/schema.py`
- Create: `tests/unit_tests/observability/state_estimator/test_schema.py`
- Create: `.scratch/training-state-estimator/schema.md`

**Interfaces:**
- Produces:
  - `EvidenceState = Literal["present", "missing", "unknown", "uncollected", "malformed", "contradictory"]`
  - `CalibrationState = Literal["heuristic", "calibrated", "not_applicable"]`
  - `QualityFinding`
  - `EntityRef`
  - `Observation`
  - `DerivedPaths`
  - `canonical_json(data: Mapping[str, Any]) -> str`
  - `write_json_atomic(path: Path, data: Mapping[str, Any]) -> None`
  - `read_json_object(path: Path) -> dict[str, Any]`

- Consumes: existing run-evidence schema described in `docs/run_evidence.md`.

- Later tasks rely on these exact field names:
  - `schema_version`
  - `kind`
  - `severity`
  - `evidence_state`
  - `message`
  - `source_path`
  - `entity`
  - `metadata`

- [ ] **Step 1: Write the failing schema tests**

Create `tests/unit_tests/observability/state_estimator/test_schema.py`:

```python
from __future__ import annotations

import json

from torchtitan.observability.state_estimator.schema import (
    DerivedPaths,
    EntityRef,
    QualityFinding,
    canonical_json,
    read_json_object,
    write_json_atomic,
)


def test_quality_finding_serializes_stable_fields(tmp_path):
    finding = QualityFinding(
        kind="missing_optional_artifact",
        severity="warning",
        evidence_state="missing",
        message="structured event stream was not present",
        source_path="indexes/artifacts.trainer.core.global_rank_000000.jsonl",
        entity=EntityRef(kind="process", id="trainer.core.global_rank_000000"),
        metadata={"producer": "structured_logger"},
    )

    assert finding.to_json() == {
        "schema_version": 1,
        "kind": "missing_optional_artifact",
        "severity": "warning",
        "evidence_state": "missing",
        "message": "structured event stream was not present",
        "source_path": "indexes/artifacts.trainer.core.global_rank_000000.jsonl",
        "entity": {"kind": "process", "id": "trainer.core.global_rank_000000"},
        "metadata": {"producer": "structured_logger"},
    }


def test_atomic_json_write_uses_canonical_json(tmp_path):
    output = tmp_path / "nested" / "artifact.json"

    write_json_atomic(output, {"b": 2, "a": 1})

    assert output.read_text() == '{"a":1,"b":2}\n'
    assert read_json_object(output) == {"a": 1, "b": 2}
    assert not output.with_suffix(".json.tmp").exists()


def test_derived_paths_are_under_state_estimator_directory(tmp_path):
    paths = DerivedPaths.from_attempt_path(tmp_path)

    assert paths.root == tmp_path / "derived" / "state_estimator"
    assert paths.evidence_graph == paths.root / "evidence_graph.json"
    assert paths.entities == paths.root / "entities.json"
    assert paths.timeline == paths.root / "timeline.jsonl"
    assert paths.quality == paths.root / "quality.json"
    assert paths.belief_summary == paths.root / "belief_summary.json"
    assert paths.diagnosis == paths.root / "diagnosis.md"
```

- [ ] **Step 2: Run the schema tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_schema.py
```

Expected: import failure for `torchtitan.observability.state_estimator.schema`.

- [ ] **Step 3: Implement the schema module**

Create `torchtitan/observability/state_estimator/__init__.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
```

Create `torchtitan/observability/state_estimator/schema.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared schemas for offline training state-estimator artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, Mapping


SCHEMA_VERSION = 1

EvidenceState = Literal[
    "present",
    "missing",
    "unknown",
    "uncollected",
    "malformed",
    "contradictory",
]
Severity = Literal["info", "warning", "error"]
CalibrationState = Literal["heuristic", "calibrated", "not_applicable"]


@dataclass(frozen=True, slots=True)
class EntityRef:
    kind: str
    id: str

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}


@dataclass(frozen=True, slots=True)
class QualityFinding:
    kind: str
    severity: Severity
    evidence_state: EvidenceState
    message: str
    source_path: str | None = None
    entity: EntityRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "severity": self.severity,
            "evidence_state": self.evidence_state,
            "message": self.message,
            "source_path": self.source_path,
            "entity": self.entity.to_json() if self.entity else None,
            "metadata": dict(self.metadata),
        }
        return data


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    kind: str
    entity: EntityRef
    event_time_ns: int | None
    ingestion_time_ns: int | None
    source_path: str
    quality: EvidenceState
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "kind": self.kind,
            "entity": self.entity.to_json(),
            "event_time_ns": self.event_time_ns,
            "ingestion_time_ns": self.ingestion_time_ns,
            "source_path": self.source_path,
            "quality": self.quality,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class DerivedPaths:
    root: Path
    evidence_graph: Path
    entities: Path
    timeline: Path
    quality: Path
    belief_summary: Path
    diagnosis: Path
    evaluation_summary: Path
    evaluation_report: Path

    @classmethod
    def from_attempt_path(cls, attempt_path: Path) -> "DerivedPaths":
        root = attempt_path / "derived" / "state_estimator"
        return cls(
            root=root,
            evidence_graph=root / "evidence_graph.json",
            entities=root / "entities.json",
            timeline=root / "timeline.jsonl",
            quality=root / "quality.json",
            belief_summary=root / "belief_summary.json",
            diagnosis=root / "diagnosis.md",
            evaluation_summary=root / "evaluation_summary.json",
            evaluation_report=root / "evaluation_report.md",
        )


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json(data) + "\n")
    tmp.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
```

- [ ] **Step 4: Write the schema vocabulary doc**

Create `.scratch/training-state-estimator/schema.md` with this structure:

```markdown
# Training State Estimator Schema

Status: implementation contract

## Raw Versus Derived Evidence

Raw evidence is the immutable run-attempt bundle described in
`docs/run_evidence.md`. Derived evidence is regenerated analyzer output under
`derived/state_estimator/`.

## Evidence State Values

- `present`: the field or artifact exists and parsed successfully.
- `missing`: the expected field or artifact was absent.
- `unknown`: the analyzer cannot infer the value from available records.
- `uncollected`: the producing tool was not enabled or did not claim to collect the signal.
- `malformed`: a record or artifact was present but failed schema validation.
- `contradictory`: two immutable records disagree about the same identity or event.

## Derived Artifacts

- `evidence_graph.json`: entity and relationship graph.
- `entities.json`: normalized entity index.
- `timeline.jsonl`: ordered observation and transition timeline.
- `quality.json`: missing, malformed, unknown, uncollected, and contradictory evidence findings.
- `belief_summary.json`: advisory estimator output.
- `diagnosis.md`: human-readable diagnosis and probe recommendations.
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_schema.py
git diff --check -- torchtitan/observability/state_estimator tests/unit_tests/observability/state_estimator .scratch/training-state-estimator/schema.md
```

Commit:

```bash
git add torchtitan/observability/state_estimator/__init__.py \
  torchtitan/observability/state_estimator/schema.py \
  tests/unit_tests/observability/state_estimator/test_schema.py \
  .scratch/training-state-estimator/schema.md
git commit -m "Add training state estimator schemas" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 2: Synthetic Run-Evidence Fixture Builder

**Files:**
- Create: `torchtitan/observability/state_estimator/fixtures.py`
- Create: `tests/unit_tests/observability/state_estimator/test_fixtures.py`

**Interfaces:**
- Consumes: `schema.canonical_json`.
- Produces:
  - `ProcessFixture`
  - `EvidenceBundleFixture`
  - `build_minimal_evidence_bundle(root: Path, *, run_id: str, attempt_id: str, processes: Sequence[ProcessFixture]) -> Path`
  - `write_structured_events(attempt_path: Path, process_id: str, rows: Sequence[Mapping[str, Any]]) -> Path`
  - `write_artifact_index(attempt_path: Path, process_id: str, rows: Sequence[Mapping[str, Any]]) -> Path`

- Later tasks use these fixtures to test graph, estimation, timeline, probe, and evaluation behavior.

- [ ] **Step 1: Write failing fixture tests**

Create `tests/unit_tests/observability/state_estimator/test_fixtures.py`:

```python
from __future__ import annotations

import json

from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)


def test_fixture_builds_single_process_success_bundle(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                outcome="succeeded",
            )
        ],
    )

    manifest = json.loads((attempt_path / "manifest.json").read_text())
    outcome = json.loads(
        (
            attempt_path
            / "processes"
            / "trainer.core.global_rank_000000"
            / "outcome.json"
        ).read_text()
    )
    indexes = list((attempt_path / "indexes").glob("artifacts.*.jsonl"))

    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "fixture-run"
    assert manifest["attempt_id"] == "attempt-001"
    assert outcome["record_type"] == "process_outcome"
    assert outcome["outcome"] == "succeeded"
    assert len(indexes) == 1


def test_fixture_builds_two_rank_phase_events(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=2,
                phases=[("train/step", 1, 1_000, 2_000)],
            ),
            ProcessFixture(
                process_id="trainer.core.global_rank_000001",
                global_rank=1,
                local_rank=1,
                world_size=2,
                phases=[("train/step", 1, 1_500, 3_000)],
            ),
        ],
    )

    event_paths = sorted((attempt_path / "structured_logs").glob("*.jsonl"))
    assert len(event_paths) == 2
    rank_one_rows = [json.loads(line) for line in event_paths[1].read_text().splitlines()]
    assert rank_one_rows[0]["phase"] == "train/step"
    assert rank_one_rows[0]["step"] == 1
    assert rank_one_rows[0]["global_rank"] == 1
```

- [ ] **Step 2: Run fixture tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_fixtures.py
```

Expected: import failure for `fixtures`.

- [ ] **Step 3: Implement fixture builder**

Create `torchtitan/observability/state_estimator/fixtures.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small host-side run-evidence fixtures for state-estimator tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from torchtitan.observability.state_estimator.schema import canonical_json


@dataclass(frozen=True, slots=True)
class ProcessFixture:
    process_id: str
    global_rank: int
    local_rank: int
    world_size: int
    role: str = "trainer"
    actor_id: str = "core"
    host_name: str = "fixture-host"
    pid: int = 1000
    outcome: str = "succeeded"
    phases: Sequence[tuple[str, int, int, int]] = field(default_factory=tuple)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(data) + "\n")


def _process_context(
    *, run_id: str, attempt_id: str, process: ProcessFixture
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "process_id": process.process_id,
        "role": process.role,
        "actor_id": process.actor_id,
        "host_name": process.host_name,
        "pid": process.pid,
        "global_rank": process.global_rank,
        "local_rank": process.local_rank,
        "world_size": process.world_size,
    }


def write_structured_events(
    attempt_path: Path,
    process_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = attempt_path / "structured_logs" / f"{process_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows))
    return path


def write_artifact_index(
    attempt_path: Path,
    process_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = attempt_path / "indexes" / f"artifacts.{process_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows))
    return path


def build_minimal_evidence_bundle(
    root: Path,
    *,
    run_id: str,
    attempt_id: str,
    processes: Sequence[ProcessFixture],
) -> Path:
    attempt_path = root / "run_evidence" / run_id / attempt_id
    attempt_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        attempt_path / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "config": {"normalized": {"training": {"steps": 1}}, "sha256": "fixture"},
            "source": {"revision": "fixture", "dirty": False},
            "command": "fixture",
            "runtime": {"python": "fixture", "torch": "fixture"},
        },
    )
    for process in processes:
        context = _process_context(run_id=run_id, attempt_id=attempt_id, process=process)
        rows = []
        for event_seq, (phase, step, wall_time_ns, monotonic_ns) in enumerate(process.phases):
            rows.append(
                {
                    "evidence_schema_version": 1,
                    "event_seq": event_seq,
                    "wall_time_ns": wall_time_ns,
                    "monotonic_ns": monotonic_ns,
                    "phase": phase,
                    "step": step,
                    "message": "phase_marker",
                    **context,
                }
            )
        event_path = write_structured_events(attempt_path, process.process_id, rows)
        write_artifact_index(
            attempt_path,
            process.process_id,
            [
                {
                    "schema_version": 1,
                    "evidence_schema_version": 1,
                    "record_type": "artifact",
                    "artifact_id": f"artifact-{process.process_id}",
                    "producer": "structured_logger",
                    "kind": "torchtitan.structured_events",
                    "relation": "output",
                    "state": "complete",
                    "path": str(event_path.relative_to(root)),
                    "path_type": "external_absolute",
                    "wall_time_ns": 1,
                    "monotonic_ns": 1,
                    "artifact_seq": 0,
                    "metadata": {},
                    **context,
                }
            ],
        )
        outcome = {
            "schema_version": 1,
            "evidence_schema_version": 1,
            "record_type": "process_outcome",
            "outcome": process.outcome,
            "elapsed_monotonic_ns": 1000,
            **context,
        }
        if process.outcome == "failed":
            outcome["exception_type"] = "RuntimeError"
            outcome["exception_message"] = "fixture failure"
        _write_json(attempt_path / "processes" / process.process_id / "outcome.json", outcome)
    return attempt_path
```

- [ ] **Step 4: Run fixture tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_fixtures.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/fixtures.py \
  tests/unit_tests/observability/state_estimator/test_fixtures.py
git commit -m "Add training state estimator fixtures" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 3: Bundle Reader and Evidence Graph Builder

**Files:**
- Create: `torchtitan/observability/state_estimator/bundle.py`
- Create: `torchtitan/observability/state_estimator/graph.py`
- Create: `tests/unit_tests/observability/state_estimator/test_bundle.py`
- Create: `tests/unit_tests/observability/state_estimator/test_graph.py`

**Interfaces:**
- Consumes:
  - `DerivedPaths`
  - `QualityFinding`
  - fixture builder from Task 2.
- Produces:
  - `RunEvidenceBundle`
  - `load_bundle(attempt_path: Path) -> RunEvidenceBundle`
  - `build_evidence_graph(attempt_path: Path) -> dict[str, Any]`
  - `write_evidence_graph(attempt_path: Path) -> DerivedPaths`

- `RunEvidenceBundle` fields:
  - `attempt_path: Path`
  - `manifest: dict[str, Any]`
  - `artifact_rows: list[dict[str, Any]]`
  - `event_rows: list[dict[str, Any]]`
  - `outcome_rows: list[dict[str, Any]]`
  - `quality_findings: list[QualityFinding]`

- [ ] **Step 1: Write failing bundle and graph tests**

Create `tests/unit_tests/observability/state_estimator/test_bundle.py`:

```python
from __future__ import annotations

import pytest

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)


def test_load_bundle_reads_manifest_artifacts_events_and_outcomes(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    bundle = load_bundle(attempt_path)

    assert bundle.manifest["run_id"] == "fixture-run"
    assert len(bundle.artifact_rows) == 1
    assert len(bundle.event_rows) == 1
    assert len(bundle.outcome_rows) == 1
    assert bundle.quality_findings == []


def test_load_bundle_fails_without_manifest(tmp_path):
    with pytest.raises(ValueError, match="manifest.json is required"):
        load_bundle(tmp_path)
```

Create `tests/unit_tests/observability/state_estimator/test_graph.py`:

```python
from __future__ import annotations

import json

from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)
from torchtitan.observability.state_estimator.graph import (
    build_evidence_graph,
    write_evidence_graph,
)


def test_graph_contains_run_process_event_and_outcome_entities(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    graph = build_evidence_graph(attempt_path)

    entity_ids = {entity["id"] for entity in graph["entities"]}
    assert "run:fixture-run" in entity_ids
    assert "attempt:fixture-run/attempt-001" in entity_ids
    assert "process:trainer.core.global_rank_000000" in entity_ids
    assert "outcome:trainer.core.global_rank_000000" in entity_ids
    assert graph["quality"] == []


def test_write_graph_outputs_derived_artifacts(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    paths = write_evidence_graph(attempt_path)

    assert json.loads(paths.evidence_graph.read_text())["schema_version"] == 1
    assert json.loads(paths.entities.read_text())["schema_version"] == 1
    assert json.loads(paths.quality.read_text())["schema_version"] == 1
    assert paths.timeline.exists()
```

- [ ] **Step 2: Run graph tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_bundle.py tests/unit_tests/observability/state_estimator/test_graph.py
```

Expected: import failure for `bundle` and `graph`.

- [ ] **Step 3: Implement bundle reader**

Create `torchtitan/observability/state_estimator/bundle.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read raw run-evidence bundles for offline state estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from torchtitan.observability.state_estimator.schema import (
    EntityRef,
    QualityFinding,
    read_json_object,
)


@dataclass(slots=True)
class RunEvidenceBundle:
    attempt_path: Path
    manifest: dict[str, Any]
    artifact_rows: list[dict[str, Any]] = field(default_factory=list)
    event_rows: list[dict[str, Any]] = field(default_factory=list)
    outcome_rows: list[dict[str, Any]] = field(default_factory=list)
    quality_findings: list[QualityFinding] = field(default_factory=list)


def _read_jsonl(path: Path, *, quality: list[QualityFinding]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            quality.append(
                QualityFinding(
                    kind="malformed_jsonl",
                    severity="warning",
                    evidence_state="malformed",
                    message=f"{path}:{line_number}: {exc}",
                    source_path=str(path),
                )
            )
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            quality.append(
                QualityFinding(
                    kind="non_object_jsonl",
                    severity="warning",
                    evidence_state="malformed",
                    message=f"{path}:{line_number}: row is not an object",
                    source_path=str(path),
                )
            )
    return rows


def load_bundle(attempt_path: Path) -> RunEvidenceBundle:
    manifest_path = attempt_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path}: manifest.json is required")
    manifest = read_json_object(manifest_path)
    quality: list[QualityFinding] = []
    artifact_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []

    indexes = attempt_path / "indexes"
    if indexes.exists():
        for path in sorted(indexes.glob("artifacts.*.jsonl")):
            artifact_rows.extend(_read_jsonl(path, quality=quality))
    else:
        quality.append(
            QualityFinding(
                kind="missing_indexes_dir",
                severity="warning",
                evidence_state="missing",
                message="indexes directory is missing",
                source_path=str(indexes),
            )
        )

    structured_logs = attempt_path / "structured_logs"
    if structured_logs.exists():
        for path in sorted(structured_logs.glob("*.jsonl")):
            event_rows.extend(_read_jsonl(path, quality=quality))
    else:
        quality.append(
            QualityFinding(
                kind="missing_structured_logs_dir",
                severity="warning",
                evidence_state="uncollected",
                message="structured_logs directory is missing",
                source_path=str(structured_logs),
            )
        )

    processes = attempt_path / "processes"
    if processes.exists():
        for path in sorted(processes.glob("*/outcome.json")):
            try:
                outcome_rows.append(read_json_object(path))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                quality.append(
                    QualityFinding(
                        kind="malformed_outcome",
                        severity="warning",
                        evidence_state="malformed",
                        message=str(exc),
                        source_path=str(path),
                        entity=EntityRef(kind="outcome", id=path.parent.name),
                    )
                )
    else:
        quality.append(
            QualityFinding(
                kind="missing_processes_dir",
                severity="warning",
                evidence_state="missing",
                message="processes directory is missing",
                source_path=str(processes),
            )
        )

    return RunEvidenceBundle(
        attempt_path=attempt_path,
        manifest=manifest,
        artifact_rows=artifact_rows,
        event_rows=event_rows,
        outcome_rows=outcome_rows,
        quality_findings=quality,
    )
```

- [ ] **Step 4: Implement evidence graph writer**

Create `torchtitan/observability/state_estimator/graph.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build deterministic derived evidence graphs from run-evidence bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.schema import (
    DerivedPaths,
    SCHEMA_VERSION,
    write_json_atomic,
)


def _entity(kind: str, id: str, **attrs: Any) -> dict[str, Any]:
    return {"kind": kind, "id": f"{kind}:{id}", "attrs": attrs}


def _edge(kind: str, source: str, target: str, **attrs: Any) -> dict[str, Any]:
    return {"kind": kind, "source": source, "target": target, "attrs": attrs}


def build_evidence_graph(attempt_path: Path) -> dict[str, Any]:
    bundle = load_bundle(attempt_path)
    run_id = bundle.manifest["run_id"]
    attempt_id = bundle.manifest["attempt_id"]
    entities: list[dict[str, Any]] = [
        _entity("run", run_id),
        _entity("attempt", f"{run_id}/{attempt_id}", run_id=run_id, attempt_id=attempt_id),
    ]
    edges: list[dict[str, Any]] = [
        _edge("has_attempt", f"run:{run_id}", f"attempt:{run_id}/{attempt_id}")
    ]
    seen_processes: set[str] = set()
    observations: list[dict[str, Any]] = []

    for row in bundle.artifact_rows:
        process_id = row.get("process_id")
        if isinstance(process_id, str) and process_id not in seen_processes:
            seen_processes.add(process_id)
            entities.append(_entity("process", process_id, global_rank=row.get("global_rank")))
            edges.append(
                _edge(
                    "attempt_process",
                    f"attempt:{run_id}/{attempt_id}",
                    f"process:{process_id}",
                )
            )
        artifact_id = str(row.get("artifact_id"))
        entities.append(
            _entity(
                "artifact",
                artifact_id,
                producer=row.get("producer"),
                kind=row.get("kind"),
                state=row.get("state"),
                path=row.get("path"),
            )
        )
        if isinstance(process_id, str):
            edges.append(_edge("process_artifact", f"process:{process_id}", f"artifact:{artifact_id}"))

    for row in bundle.outcome_rows:
        process_id = row.get("process_id")
        if isinstance(process_id, str) and process_id not in seen_processes:
            seen_processes.add(process_id)
            entities.append(_entity("process", process_id, global_rank=row.get("global_rank")))
        if isinstance(process_id, str):
            outcome_id = process_id
            entities.append(_entity("outcome", outcome_id, outcome=row.get("outcome")))
            edges.append(_edge("process_outcome", f"process:{process_id}", f"outcome:{outcome_id}"))

    for row in bundle.event_rows:
        process_id = str(row.get("process_id", "unknown"))
        event_id = f"{process_id}:{row.get('event_seq', len(observations))}"
        observation = {
            "schema_version": SCHEMA_VERSION,
            "id": event_id,
            "kind": "structured_event",
            "entity": {"kind": "process", "id": process_id},
            "event_time_ns": row.get("wall_time_ns"),
            "ingestion_time_ns": None,
            "source_path": "structured_logs",
            "quality": "present",
            "payload": row,
        }
        observations.append(observation)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "entities": entities,
        "edges": edges,
        "observations": sorted(
            observations,
            key=lambda item: (
                item["event_time_ns"] is None,
                item["event_time_ns"] or 0,
                item["id"],
            ),
        ),
        "quality": [finding.to_json() for finding in bundle.quality_findings],
    }


def write_evidence_graph(attempt_path: Path) -> DerivedPaths:
    paths = DerivedPaths.from_attempt_path(attempt_path)
    graph = build_evidence_graph(attempt_path)
    write_json_atomic(paths.evidence_graph, graph)
    write_json_atomic(
        paths.entities,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": graph["run_id"],
            "attempt_id": graph["attempt_id"],
            "entities": graph["entities"],
        },
    )
    write_json_atomic(
        paths.quality,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": graph["run_id"],
            "attempt_id": graph["attempt_id"],
            "findings": graph["quality"],
        },
    )
    paths.timeline.parent.mkdir(parents=True, exist_ok=True)
    paths.timeline.write_text(
        "".join(
            __import__("json").dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in graph["observations"]
        )
    )
    return paths
```

- [ ] **Step 5: Run graph tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_bundle.py tests/unit_tests/observability/state_estimator/test_graph.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/bundle.py \
  torchtitan/observability/state_estimator/graph.py \
  tests/unit_tests/observability/state_estimator/test_bundle.py \
  tests/unit_tests/observability/state_estimator/test_graph.py
git commit -m "Build offline training evidence graphs" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 4: Classical Deterministic Modeling Pass

**Files:**
- Create: `torchtitan/observability/state_estimator/estimators.py`
- Create: `tests/unit_tests/observability/state_estimator/test_estimators.py`

**Interfaces:**
- Consumes:
  - `build_evidence_graph(attempt_path: Path) -> dict[str, Any]`
- Produces:
  - `estimate_belief(graph: Mapping[str, Any], *, skew_threshold_ns: int = 500_000_000) -> dict[str, Any]`
  - `write_belief_summary(attempt_path: Path) -> DerivedPaths`

- Initial model outputs:
  - `process_liveness`
  - `phase_durations`
  - `peer_skew_findings`
  - `fault_hypotheses`
  - `observability_warnings`
  - `claim_calibration="heuristic"`

- [ ] **Step 1: Write failing estimator tests**

Create `tests/unit_tests/observability/state_estimator/test_estimators.py`:

```python
from __future__ import annotations

from torchtitan.observability.state_estimator.estimators import estimate_belief


def test_estimator_marks_failed_process_and_heuristic_calibration():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [
            {"kind": "process", "id": "process:rank0", "attrs": {"global_rank": 0}},
            {"kind": "outcome", "id": "outcome:rank0", "attrs": {"outcome": "failed"}},
        ],
        "edges": [
            {"kind": "process_outcome", "source": "process:rank0", "target": "outcome:rank0", "attrs": {}}
        ],
        "observations": [],
        "quality": [],
    }

    summary = estimate_belief(graph)

    assert summary["claim_calibration"] == "heuristic"
    assert summary["process_liveness"]["process:rank0"]["outcome"] == "failed"
    assert summary["fault_hypotheses"][0]["mode"] == "process_failure"
    assert summary["fault_hypotheses"][0]["score"] == 1.0


def test_estimator_reports_peer_skew_without_root_cause_overclaim():
    graph = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "attempt_id": "attempt-001",
        "entities": [],
        "edges": [],
        "quality": [],
        "observations": [
            {
                "id": "rank0:0",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank0"},
                "event_time_ns": 1_000,
                "payload": {"phase": "train/step", "step": 1, "global_rank": 0},
            },
            {
                "id": "rank1:0",
                "kind": "structured_event",
                "entity": {"kind": "process", "id": "rank1"},
                "event_time_ns": 900_001_000,
                "payload": {"phase": "train/step", "step": 1, "global_rank": 1},
            },
        ],
    }

    summary = estimate_belief(graph, skew_threshold_ns=500_000_000)

    assert summary["peer_skew_findings"][0]["phase"] == "train/step"
    assert summary["peer_skew_findings"][0]["step"] == 1
    assert summary["peer_skew_findings"][0]["skew_ns"] == 900_000_000
    assert summary["peer_skew_findings"][0]["root_cause"] == "ambiguous"
```

- [ ] **Step 2: Run estimator tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_estimators.py
```

Expected: import failure for `estimators`.

- [ ] **Step 3: Implement deterministic estimator**

Create `torchtitan/observability/state_estimator/estimators.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic offline estimators for training run evidence graphs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from torchtitan.observability.state_estimator.graph import build_evidence_graph
from torchtitan.observability.state_estimator.schema import (
    DerivedPaths,
    SCHEMA_VERSION,
    write_json_atomic,
)


def _process_liveness(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entities = {entity["id"]: entity for entity in graph.get("entities", [])}
    liveness: dict[str, dict[str, Any]] = {}
    for edge in graph.get("edges", []):
        if edge.get("kind") != "process_outcome":
            continue
        process_id = edge["source"]
        outcome_id = edge["target"]
        outcome = entities.get(outcome_id, {}).get("attrs", {}).get("outcome", "unknown")
        liveness[process_id] = {"outcome": outcome}
    return liveness


def _peer_skew_findings(
    observations: list[Mapping[str, Any]], *, skew_threshold_ns: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        payload = observation.get("payload", {})
        phase = payload.get("phase")
        step = payload.get("step")
        if isinstance(phase, str) and isinstance(step, int):
            groups[(phase, step)].append(observation)
    findings: list[dict[str, Any]] = []
    for (phase, step), rows in sorted(groups.items()):
        times = [
            row.get("event_time_ns")
            for row in rows
            if isinstance(row.get("event_time_ns"), int)
        ]
        if len(times) < 2:
            continue
        skew_ns = max(times) - min(times)
        if skew_ns > skew_threshold_ns:
            latest = max(
                rows,
                key=lambda row: row.get("event_time_ns") if isinstance(row.get("event_time_ns"), int) else -1,
            )
            findings.append(
                {
                    "phase": phase,
                    "step": step,
                    "skew_ns": skew_ns,
                    "latest_entity": latest.get("entity"),
                    "root_cause": "ambiguous",
                    "candidate_modes": [
                        "host_or_data_stall",
                        "compute_degradation",
                        "network_degradation",
                        "normal_workload_skew",
                    ],
                }
            )
    return findings


def estimate_belief(
    graph: Mapping[str, Any], *, skew_threshold_ns: int = 500_000_000
) -> dict[str, Any]:
    liveness = _process_liveness(graph)
    fault_hypotheses: list[dict[str, Any]] = []
    if any(item["outcome"] == "failed" for item in liveness.values()):
        fault_hypotheses.append(
            {
                "mode": "process_failure",
                "score": 1.0,
                "calibration": "heuristic",
                "evidence": "process outcome reported failed",
            }
        )
    peer_skew = _peer_skew_findings(
        list(graph.get("observations", [])), skew_threshold_ns=skew_threshold_ns
    )
    if peer_skew:
        fault_hypotheses.append(
            {
                "mode": "peer_relative_skew",
                "score": 0.5,
                "calibration": "heuristic",
                "evidence": "phase arrival skew exceeded threshold",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": graph.get("run_id"),
        "attempt_id": graph.get("attempt_id"),
        "claim_calibration": "heuristic",
        "process_liveness": liveness,
        "phase_durations": [],
        "peer_skew_findings": peer_skew,
        "fault_hypotheses": fault_hypotheses,
        "observability_warnings": list(graph.get("quality", [])),
    }


def write_belief_summary(attempt_path: Path) -> DerivedPaths:
    paths = DerivedPaths.from_attempt_path(attempt_path)
    summary = estimate_belief(build_evidence_graph(attempt_path))
    write_json_atomic(paths.belief_summary, summary)
    paths.diagnosis.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training State Estimator Diagnosis",
        "",
        f"- run_id: {summary['run_id']}",
        f"- attempt_id: {summary['attempt_id']}",
        f"- claim_calibration: {summary['claim_calibration']}",
        f"- fault_hypotheses: {len(summary['fault_hypotheses'])}",
        f"- peer_skew_findings: {len(summary['peer_skew_findings'])}",
        "",
    ]
    paths.diagnosis.write_text("\n".join(lines))
    return paths
```

- [ ] **Step 4: Run estimator tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_estimators.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/estimators.py \
  tests/unit_tests/observability/state_estimator/test_estimators.py
git commit -m "Add classical training state estimators" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 5: Data Collection Plan and Quality Report

**Files:**
- Modify: `torchtitan/observability/state_estimator/schema.py`
- Create: `torchtitan/observability/state_estimator/data_collection.py`
- Create: `tests/unit_tests/observability/state_estimator/test_data_collection.py`
- Modify: `.scratch/training-state-estimator/schema.md`

**Interfaces:**
- Consumes:
  - `RunEvidenceBundle`
  - `QualityFinding`
- Produces:
  - `DataSignal`
  - `DataCollectionPlan`
  - `build_data_collection_plan(bundle: RunEvidenceBundle) -> DataCollectionPlan`
  - `DataCollectionPlan.to_json() -> dict[str, Any]`

- Initial signal classes:
  - `required_raw`: manifest, process outcome, artifact index.
  - `tier0_semantic`: structured events, phase, step, rank, process identity.
  - `tier0_hardware`: DCGM or hardware summaries when indexed.
  - `tier1_trace`: profiler, memory, or Flight Recorder artifacts.
  - `tier2_incident`: incident snapshots or diagnostic bundles.
  - `derived`: evidence graph, belief summary, diagnosis, evaluation.

- [ ] **Step 1: Write failing data collection tests**

Create `tests/unit_tests/observability/state_estimator/test_data_collection.py`:

```python
from __future__ import annotations

from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.data_collection import (
    build_data_collection_plan,
)
from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)


def test_data_collection_plan_marks_required_raw_signals_present(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    plan = build_data_collection_plan(load_bundle(attempt_path)).to_json()

    states = {signal["name"]: signal["state"] for signal in plan["signals"]}
    assert states["manifest"] == "present"
    assert states["process_outcome"] == "present"
    assert states["artifact_index"] == "present"
    assert states["structured_events"] == "present"


def test_data_collection_plan_marks_uncollected_optional_hardware(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    plan = build_data_collection_plan(load_bundle(attempt_path)).to_json()

    dcgm = next(signal for signal in plan["signals"] if signal["name"] == "dcgm")
    assert dcgm["state"] == "uncollected"
    assert dcgm["required"] is False
```

- [ ] **Step 2: Run data collection tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_data_collection.py
```

Expected: import failure for `data_collection`.

- [ ] **Step 3: Implement data collection signal planner**

Create `torchtitan/observability/state_estimator/data_collection.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Describe available and missing signals for offline state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchtitan.observability.state_estimator.bundle import RunEvidenceBundle
from torchtitan.observability.state_estimator.schema import EvidenceState, SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class DataSignal:
    name: str
    tier: str
    state: EvidenceState
    required: bool
    producer: str | None = None
    kind: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "state": self.state,
            "required": self.required,
            "producer": self.producer,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DataCollectionPlan:
    run_id: str
    attempt_id: str
    signals: tuple[DataSignal, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "signals": [signal.to_json() for signal in self.signals],
        }


def _has_artifact(bundle: RunEvidenceBundle, *, producer: str | None = None, kind: str | None = None) -> bool:
    for row in bundle.artifact_rows:
        if producer is not None and row.get("producer") != producer:
            continue
        if kind is not None and row.get("kind") != kind:
            continue
        return True
    return False


def build_data_collection_plan(bundle: RunEvidenceBundle) -> DataCollectionPlan:
    signals = [
        DataSignal("manifest", "required_raw", "present", True),
        DataSignal("process_outcome", "required_raw", "present" if bundle.outcome_rows else "missing", True),
        DataSignal("artifact_index", "required_raw", "present" if bundle.artifact_rows else "missing", True),
        DataSignal("structured_events", "tier0_semantic", "present" if bundle.event_rows else "uncollected", False, producer="structured_logger", kind="torchtitan.structured_events"),
        DataSignal("dcgm", "tier0_hardware", "present" if _has_artifact(bundle, producer="dcgm") else "uncollected", False, producer="dcgm"),
        DataSignal("profiler", "tier1_trace", "present" if _has_artifact(bundle, kind="pytorch.profiler.trace") else "uncollected", False, kind="pytorch.profiler.trace"),
        DataSignal("flight_recorder", "tier1_trace", "present" if _has_artifact(bundle, kind="pytorch.flight_recorder.dump") else "uncollected", False, kind="pytorch.flight_recorder.dump"),
        DataSignal("incident", "tier2_incident", "present" if _has_artifact(bundle, producer="incident_recorder") else "uncollected", False, producer="incident_recorder"),
    ]
    return DataCollectionPlan(
        run_id=str(bundle.manifest["run_id"]),
        attempt_id=str(bundle.manifest["attempt_id"]),
        signals=tuple(signals),
    )
```

- [ ] **Step 4: Wire quality output into graph artifacts**

Modify `graph.write_evidence_graph()` so `quality.json` includes `data_collection`:

```python
from torchtitan.observability.state_estimator.bundle import load_bundle
from torchtitan.observability.state_estimator.data_collection import build_data_collection_plan

# inside write_evidence_graph:
bundle = load_bundle(attempt_path)
collection_plan = build_data_collection_plan(bundle).to_json()
graph = build_evidence_graph(attempt_path)
...
write_json_atomic(
    paths.quality,
    {
        "schema_version": SCHEMA_VERSION,
        "run_id": graph["run_id"],
        "attempt_id": graph["attempt_id"],
        "findings": graph["quality"],
        "data_collection": collection_plan,
    },
)
```

- [ ] **Step 5: Run data collection tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_data_collection.py tests/unit_tests/observability/state_estimator/test_graph.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/data_collection.py \
  torchtitan/observability/state_estimator/graph.py \
  tests/unit_tests/observability/state_estimator/test_data_collection.py \
  .scratch/training-state-estimator/schema.md
git commit -m "Add state estimator data collection plan" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 6: Fixed-Lag Incident Timeline Modeling

**Files:**
- Create: `torchtitan/observability/state_estimator/timeline.py`
- Create: `tests/unit_tests/observability/state_estimator/test_timeline.py`
- Modify: `torchtitan/observability/state_estimator/estimators.py`

**Interfaces:**
- Consumes:
  - graph observations.
- Produces:
  - `IncidentWindow`
  - `build_incident_timeline(graph: Mapping[str, Any], *, center_time_ns: int | None, window_before_ns: int, window_after_ns: int) -> list[dict[str, Any]]`
  - `select_failure_center_time_ns(graph: Mapping[str, Any]) -> int | None`

- [ ] **Step 1: Write failing timeline tests**

Create `tests/unit_tests/observability/state_estimator/test_timeline.py`:

```python
from __future__ import annotations

from torchtitan.observability.state_estimator.timeline import (
    build_incident_timeline,
    select_failure_center_time_ns,
)


def test_timeline_orders_by_event_time_then_entity():
    graph = {
        "observations": [
            {"id": "b", "event_time_ns": 20, "entity": {"kind": "process", "id": "rank1"}, "payload": {"phase": "backward"}},
            {"id": "a", "event_time_ns": 10, "entity": {"kind": "process", "id": "rank0"}, "payload": {"phase": "forward"}},
        ],
        "entities": [],
    }

    timeline = build_incident_timeline(
        graph,
        center_time_ns=15,
        window_before_ns=10,
        window_after_ns=10,
    )

    assert [row["id"] for row in timeline] == ["a", "b"]
    assert timeline[0]["relative_time_ns"] == -5
    assert timeline[1]["relative_time_ns"] == 5


def test_failure_center_uses_latest_observation_when_process_failed():
    graph = {
        "entities": [
            {"kind": "outcome", "id": "outcome:rank0", "attrs": {"outcome": "failed"}},
        ],
        "observations": [
            {"id": "a", "event_time_ns": 10, "entity": {"kind": "process", "id": "rank0"}, "payload": {}},
            {"id": "b", "event_time_ns": 30, "entity": {"kind": "process", "id": "rank0"}, "payload": {}},
        ],
    }

    assert select_failure_center_time_ns(graph) == 30
```

- [ ] **Step 2: Run timeline tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_timeline.py
```

Expected: import failure for `timeline`.

- [ ] **Step 3: Implement timeline module**

Create `torchtitan/observability/state_estimator/timeline.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-lag incident timeline reconstruction for offline evidence graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class IncidentWindow:
    center_time_ns: int
    window_before_ns: int
    window_after_ns: int


def select_failure_center_time_ns(graph: Mapping[str, Any]) -> int | None:
    has_failure = any(
        entity.get("kind") == "outcome"
        and entity.get("attrs", {}).get("outcome") == "failed"
        for entity in graph.get("entities", [])
    )
    if not has_failure:
        return None
    times = [
        observation.get("event_time_ns")
        for observation in graph.get("observations", [])
        if isinstance(observation.get("event_time_ns"), int)
    ]
    return max(times) if times else None


def build_incident_timeline(
    graph: Mapping[str, Any],
    *,
    center_time_ns: int | None,
    window_before_ns: int,
    window_after_ns: int,
) -> list[dict[str, Any]]:
    if center_time_ns is None:
        center_time_ns = select_failure_center_time_ns(graph)
    if center_time_ns is None:
        return []
    start = center_time_ns - window_before_ns
    end = center_time_ns + window_after_ns
    rows: list[dict[str, Any]] = []
    for observation in graph.get("observations", []):
        event_time = observation.get("event_time_ns")
        if not isinstance(event_time, int) or event_time < start or event_time > end:
            continue
        row = dict(observation)
        row["relative_time_ns"] = event_time - center_time_ns
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["event_time_ns"],
            row.get("entity", {}).get("kind", ""),
            row.get("entity", {}).get("id", ""),
            row.get("id", ""),
        ),
    )
```

- [ ] **Step 4: Include incident timeline in diagnosis output**

Modify `estimators.write_belief_summary()`:

```python
from torchtitan.observability.state_estimator.timeline import build_incident_timeline

# after summary:
timeline = build_incident_timeline(
    graph,
    center_time_ns=None,
    window_before_ns=300_000_000_000,
    window_after_ns=30_000_000_000,
)
summary["incident_timeline"] = timeline
```

In `diagnosis.md`, append:

```python
lines.extend(
    [
        "## Incident Timeline",
        "",
        *(f"- {row['relative_time_ns']} ns: {row.get('id')}" for row in timeline[:20]),
        "",
    ]
)
```

- [ ] **Step 5: Run timeline and estimator tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_timeline.py tests/unit_tests/observability/state_estimator/test_estimators.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/timeline.py \
  torchtitan/observability/state_estimator/estimators.py \
  tests/unit_tests/observability/state_estimator/test_timeline.py
git commit -m "Add fixed-lag training incident timelines" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 7: Probe Recommendation Modeling

**Files:**
- Create: `torchtitan/observability/state_estimator/probes.py`
- Create: `tests/unit_tests/observability/state_estimator/test_probes.py`
- Modify: `torchtitan/observability/state_estimator/estimators.py`

**Interfaces:**
- Consumes:
  - `belief_summary`.
- Produces:
  - `ProbeRecommendation`
  - `recommend_probes(summary: Mapping[str, Any]) -> list[dict[str, Any]]`

- Recommendation schema:
  - `kind`
  - `competing_hypotheses`
  - `information_target`
  - `cost_class`
  - `risk_class`
  - `required_authority`
  - `live_safe`
  - `source_evidence`

- [ ] **Step 1: Write failing probe tests**

Create `tests/unit_tests/observability/state_estimator/test_probes.py`:

```python
from __future__ import annotations

from torchtitan.observability.state_estimator.probes import recommend_probes


def test_peer_skew_recommends_compute_and_collective_separation():
    summary = {
        "peer_skew_findings": [
            {
                "phase": "train/step",
                "step": 1,
                "root_cause": "ambiguous",
                "candidate_modes": [
                    "host_or_data_stall",
                    "compute_degradation",
                    "network_degradation",
                ],
            }
        ],
        "observability_warnings": [],
        "fault_hypotheses": [],
    }

    recommendations = recommend_probes(summary)

    assert recommendations[0]["kind"] == "separate_late_arrival_from_network"
    assert recommendations[0]["live_safe"] is False
    assert recommendations[0]["required_authority"] == "operator_stopped_job"


def test_no_ambiguity_yields_no_probe():
    summary = {
        "peer_skew_findings": [],
        "observability_warnings": [],
        "fault_hypotheses": [
            {"mode": "process_failure", "score": 1.0, "calibration": "heuristic"}
        ],
    }

    assert recommend_probes(summary) == []
```

- [ ] **Step 2: Run probe tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_probes.py
```

Expected: import failure for `probes`.

- [ ] **Step 3: Implement probe recommendations**

Create `torchtitan/observability/state_estimator/probes.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recommendation-only active diagnostic probes for estimator ambiguity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ProbeRecommendation:
    kind: str
    competing_hypotheses: tuple[str, ...]
    information_target: str
    cost_class: str
    risk_class: str
    required_authority: str
    live_safe: bool
    source_evidence: str

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "competing_hypotheses": list(self.competing_hypotheses),
            "information_target": self.information_target,
            "cost_class": self.cost_class,
            "risk_class": self.risk_class,
            "required_authority": self.required_authority,
            "live_safe": self.live_safe,
            "source_evidence": self.source_evidence,
        }


def recommend_probes(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[ProbeRecommendation] = []
    for finding in summary.get("peer_skew_findings", []):
        candidates = tuple(finding.get("candidate_modes", []))
        if (
            finding.get("root_cause") == "ambiguous"
            and "network_degradation" in candidates
            and ("compute_degradation" in candidates or "host_or_data_stall" in candidates)
        ):
            recommendations.append(
                ProbeRecommendation(
                    kind="separate_late_arrival_from_network",
                    competing_hypotheses=candidates,
                    information_target="rank arrival time versus collective progress time",
                    cost_class="medium",
                    risk_class="requires_stopped_or_diagnostic_allocation",
                    required_authority="operator_stopped_job",
                    live_safe=False,
                    source_evidence=f"peer skew in {finding.get('phase')} step {finding.get('step')}",
                )
            )
    return [recommendation.to_json() for recommendation in recommendations]
```

- [ ] **Step 4: Add recommendations to belief summary and diagnosis**

Modify `estimators.write_belief_summary()`:

```python
from torchtitan.observability.state_estimator.probes import recommend_probes

summary["probe_recommendations"] = recommend_probes(summary)
```

Add to `diagnosis.md`:

```python
lines.extend(
    [
        "## Probe Recommendations",
        "",
        *(f"- {probe['kind']}: {probe['information_target']}" for probe in summary["probe_recommendations"]),
        "",
    ]
)
```

- [ ] **Step 5: Run probe and estimator tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_probes.py tests/unit_tests/observability/state_estimator/test_estimators.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/probes.py \
  torchtitan/observability/state_estimator/estimators.py \
  tests/unit_tests/observability/state_estimator/test_probes.py
git commit -m "Add state estimator probe recommendations" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 8: Learned Modeling Extension Interfaces

**Files:**
- Create: `torchtitan/observability/state_estimator/learned.py`
- Create: `tests/unit_tests/observability/state_estimator/test_learned.py`

**Interfaces:**
- Produces:
  - `TraceSegment`
  - `LikelihoodFactor`
  - `ResidualPrediction`
  - `SensorQualityPrediction`
  - `FaultModePrior`
  - `load_learned_extension(module_name: str | None) -> object | None`

- [ ] **Step 1: Write failing learned interface tests**

Create `tests/unit_tests/observability/state_estimator/test_learned.py`:

```python
from __future__ import annotations

import importlib
import sys

from torchtitan.observability.state_estimator.learned import (
    FaultModePrior,
    LikelihoodFactor,
    TraceSegment,
    load_learned_extension,
)


def test_learned_schemas_serialize_without_torch_dependency():
    assert "torch" not in sys.modules

    segment = TraceSegment(
        id="segment-1",
        entity={"kind": "process", "id": "rank0"},
        modality="structured_events",
        start_time_ns=10,
        end_time_ns=20,
        features={"count": 2},
    )
    factor = LikelihoodFactor(
        source_segment_id="segment-1",
        target_state="host_or_data_stall",
        log_likelihood=-1.25,
        calibration="heuristic",
        metadata={"model": "null"},
    )

    assert segment.to_json()["modality"] == "structured_events"
    assert factor.to_json()["calibration"] == "heuristic"


def test_null_extension_loader_returns_none():
    assert load_learned_extension(None) is None


def test_fault_mode_prior_schema_marks_advisory():
    prior = FaultModePrior(mode="network_degradation", score=0.2, calibration="heuristic")

    assert prior.to_json()["advisory"] is True
```

- [ ] **Step 2: Run learned tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_learned.py
```

Expected: import failure for `learned`.

- [ ] **Step 3: Implement learned interface schemas**

Create `torchtitan/observability/state_estimator/learned.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional learned-model extension interfaces for state estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TraceSegment:
    id: str
    entity: Mapping[str, str]
    modality: str
    start_time_ns: int | None
    end_time_ns: int | None
    features: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": dict(self.entity),
            "modality": self.modality,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "features": dict(self.features),
        }


@dataclass(frozen=True, slots=True)
class LikelihoodFactor:
    source_segment_id: str
    target_state: str
    log_likelihood: float
    calibration: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "source_segment_id": self.source_segment_id,
            "target_state": self.target_state,
            "log_likelihood": self.log_likelihood,
            "calibration": self.calibration,
            "advisory": True,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ResidualPrediction:
    target: str
    predicted_residual: float
    sigma: float
    calibration: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "predicted_residual": self.predicted_residual,
            "sigma": self.sigma,
            "calibration": self.calibration,
            "advisory": True,
        }


@dataclass(frozen=True, slots=True)
class SensorQualityPrediction:
    sensor_id: str
    mode: str
    score: float
    calibration: str

    def to_json(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "mode": self.mode,
            "score": self.score,
            "calibration": self.calibration,
            "advisory": True,
        }


@dataclass(frozen=True, slots=True)
class FaultModePrior:
    mode: str
    score: float
    calibration: str

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "score": self.score,
            "calibration": self.calibration,
            "advisory": True,
        }


def load_learned_extension(module_name: str | None) -> object | None:
    if module_name is None:
        return None
    return importlib.import_module(module_name)
```

- [ ] **Step 4: Run learned tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_learned.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/learned.py \
  tests/unit_tests/observability/state_estimator/test_learned.py
git commit -m "Define learned state estimator extension interfaces" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 9: Evaluation Harness

**Files:**
- Create: `torchtitan/observability/state_estimator/evaluation.py`
- Create: `tests/unit_tests/observability/state_estimator/test_evaluation.py`

**Interfaces:**
- Consumes:
  - `belief_summary.json`
  - labeled evaluation manifest.
- Produces:
  - `EvaluationCase`
  - `EvaluationResult`
  - `evaluate_case(case: EvaluationCase, summary: Mapping[str, Any]) -> EvaluationResult`
  - `write_evaluation_report(cases: Sequence[EvaluationCase], summaries: Mapping[str, Mapping[str, Any]], output_dir: Path) -> dict[str, Any]`

- Evaluation metrics:
  - `detected`
  - `expected_mode_found`
  - `recommended_probe_found`
  - `false_alert`
  - `observability_warning_count`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/unit_tests/observability/state_estimator/test_evaluation.py`:

```python
from __future__ import annotations

from torchtitan.observability.state_estimator.evaluation import (
    EvaluationCase,
    evaluate_case,
)


def test_evaluation_detects_expected_fault_mode_and_probe():
    case = EvaluationCase(
        id="case-1",
        attempt_path="fixture",
        expected_mode="peer_relative_skew",
        expected_probe="separate_late_arrival_from_network",
        nominal=False,
    )
    summary = {
        "fault_hypotheses": [{"mode": "peer_relative_skew"}],
        "probe_recommendations": [{"kind": "separate_late_arrival_from_network"}],
        "observability_warnings": [],
    }

    result = evaluate_case(case, summary).to_json()

    assert result["detected"] is True
    assert result["expected_mode_found"] is True
    assert result["recommended_probe_found"] is True
    assert result["false_alert"] is False


def test_nominal_case_counts_unexpected_hypothesis_as_false_alert():
    case = EvaluationCase(
        id="nominal",
        attempt_path="fixture",
        expected_mode=None,
        expected_probe=None,
        nominal=True,
    )
    summary = {
        "fault_hypotheses": [{"mode": "peer_relative_skew"}],
        "probe_recommendations": [],
        "observability_warnings": [{"kind": "missing_optional_artifact"}],
    }

    result = evaluate_case(case, summary).to_json()

    assert result["detected"] is True
    assert result["false_alert"] is True
    assert result["observability_warning_count"] == 1
```

- [ ] **Step 2: Run evaluation tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_evaluation.py
```

Expected: import failure for `evaluation`.

- [ ] **Step 3: Implement evaluation harness**

Create `torchtitan/observability/state_estimator/evaluation.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation helpers for offline training state-estimator outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from torchtitan.observability.state_estimator.schema import SCHEMA_VERSION, write_json_atomic


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    attempt_path: str
    expected_mode: str | None
    expected_probe: str | None
    nominal: bool


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    detected: bool
    expected_mode_found: bool
    recommended_probe_found: bool
    false_alert: bool
    observability_warning_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "detected": self.detected,
            "expected_mode_found": self.expected_mode_found,
            "recommended_probe_found": self.recommended_probe_found,
            "false_alert": self.false_alert,
            "observability_warning_count": self.observability_warning_count,
        }


def evaluate_case(case: EvaluationCase, summary: Mapping[str, Any]) -> EvaluationResult:
    modes = {item.get("mode") for item in summary.get("fault_hypotheses", [])}
    probes = {item.get("kind") for item in summary.get("probe_recommendations", [])}
    detected = bool(modes)
    expected_mode_found = case.expected_mode in modes if case.expected_mode else not detected
    recommended_probe_found = case.expected_probe in probes if case.expected_probe else True
    false_alert = case.nominal and detected
    return EvaluationResult(
        case_id=case.id,
        detected=detected,
        expected_mode_found=expected_mode_found,
        recommended_probe_found=recommended_probe_found,
        false_alert=false_alert,
        observability_warning_count=len(summary.get("observability_warnings", [])),
    )


def write_evaluation_report(
    cases: Sequence[EvaluationCase],
    summaries: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    results = [
        evaluate_case(case, summaries[case.id]).to_json()
        for case in cases
    ]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(results),
        "detected_count": sum(1 for result in results if result["detected"]),
        "false_alert_count": sum(1 for result in results if result["false_alert"]),
        "expected_mode_found_count": sum(1 for result in results if result["expected_mode_found"]),
        "recommended_probe_found_count": sum(1 for result in results if result["recommended_probe_found"]),
        "results": results,
    }
    write_json_atomic(output_dir / "evaluation_summary.json", aggregate)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training State Estimator Evaluation",
        "",
        f"- cases: {aggregate['case_count']}",
        f"- detected: {aggregate['detected_count']}",
        f"- false_alerts: {aggregate['false_alert_count']}",
        "",
    ]
    (output_dir / "evaluation_report.md").write_text("\n".join(lines))
    return aggregate
```

- [ ] **Step 4: Run evaluation tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_evaluation.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/evaluation.py \
  tests/unit_tests/observability/state_estimator/test_evaluation.py
git commit -m "Add training state estimator evaluation harness" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 10: Offline Analyzer CLI

**Files:**
- Create: `torchtitan/observability/state_estimator/cli.py`
- Create: `tests/unit_tests/observability/state_estimator/test_cli.py`

**Interfaces:**
- Consumes:
  - `write_evidence_graph`
  - `write_belief_summary`
- Produces:
  - `analyze_attempt(attempt_path: Path, *, overwrite: bool = True) -> dict[str, Any]`
  - CLI payload:
    `python3 -m torchtitan.observability.state_estimator.cli --attempt-path <path>`
  - Rootfs CLI invocation:
    `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m torchtitan.observability.state_estimator.cli --attempt-path <path>'`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit_tests/observability/state_estimator/test_cli.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys

from torchtitan.observability.state_estimator.cli import analyze_attempt
from torchtitan.observability.state_estimator.fixtures import (
    ProcessFixture,
    build_minimal_evidence_bundle,
)


def test_analyze_attempt_writes_graph_and_belief_outputs(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
                phases=[("train/step", 1, 1_000, 2_000)],
            )
        ],
    )

    result = analyze_attempt(attempt_path)

    assert result["ok"] is True
    assert (attempt_path / "derived" / "state_estimator" / "evidence_graph.json").exists()
    assert (attempt_path / "derived" / "state_estimator" / "belief_summary.json").exists()


def test_cli_runs_against_fixture_bundle(tmp_path):
    attempt_path = build_minimal_evidence_bundle(
        tmp_path,
        run_id="fixture-run",
        attempt_id="attempt-001",
        processes=[
            ProcessFixture(
                process_id="trainer.core.global_rank_000000",
                global_rank=0,
                local_rank=0,
                world_size=1,
            )
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchtitan.observability.state_estimator.cli",
            "--attempt-path",
            str(attempt_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0
    assert json.loads(proc.stdout)["ok"] is True
```

- [ ] **Step 2: Run CLI tests and verify red**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_cli.py
```

Expected: import failure for `cli`.

- [ ] **Step 3: Implement CLI**

Create `torchtitan/observability/state_estimator/cli.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline CLI for training state-estimator analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torchtitan.observability.state_estimator.estimators import write_belief_summary
from torchtitan.observability.state_estimator.graph import write_evidence_graph


def analyze_attempt(attempt_path: Path, *, overwrite: bool = True) -> dict[str, Any]:
    graph_paths = write_evidence_graph(attempt_path)
    belief_paths = write_belief_summary(attempt_path)
    return {
        "ok": True,
        "attempt_path": str(attempt_path),
        "evidence_graph": str(graph_paths.evidence_graph),
        "belief_summary": str(belief_paths.belief_summary),
        "diagnosis": str(belief_paths.diagnosis),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-path", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze_attempt(args.attempt_path)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_cli.py
```

Commit:

```bash
git add torchtitan/observability/state_estimator/cli.py \
  tests/unit_tests/observability/state_estimator/test_cli.py
git commit -m "Add offline training state estimator CLI" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 11: Integration Verification and Documentation Update

**Files:**
- Modify: `.scratch/training-state-estimator/spec.md`
- Modify: `.scratch/training-state-estimator/issues/09-first-end-to-end-offline-analyzer.md`
- Optional modify: `docs/run_evidence.md` only if the analyzer adds a documented derived-artifact convention.

**Interfaces:**
- Consumes all previous task outputs.
- Produces:
  - one verified host-side command sequence;
  - updated tracker comments with evidence;
  - documentation pointer to the CLI and derived artifacts.

- [ ] **Step 1: Run the complete state-estimator unit suite**

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator'
```

Expected: all tests pass.

- [ ] **Step 2: Run the observability-adjacent unit tests**

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/test_run_evidence.py tests/unit_tests/observability/test_structured_logging.py'
```

Expected: all tests pass.

- [ ] **Step 3: Run static compile for new modules**

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile torchtitan/observability/state_estimator/*.py'
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Run CLI on a synthetic fixture through pytest**

Run:

```bash
scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator/test_cli.py::test_cli_runs_against_fixture_bundle'
```

Expected: pass.

- [ ] **Step 5: Update tracker issue 09 with completion evidence**

Append to `.scratch/training-state-estimator/issues/09-first-end-to-end-offline-analyzer.md`:

```markdown
## Completion Evidence

- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/state_estimator'`
- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m pytest -q tests/unit_tests/observability/test_run_evidence.py tests/unit_tests/observability/test_structured_logging.py'`
- `scripts/rootfs/enter_rootfs.sh -- bash -lc 'cd /workspace/torchtitan && python3 -m py_compile torchtitan/observability/state_estimator/*.py'`

The analyzer remains offline and advisory. It does not alter raw evidence,
training control flow, optimizer commit behavior, checkpoint validity,
membership, or recovery policy.
```

- [ ] **Step 6: Run final diff hygiene**

Run:

```bash
git diff --check -- torchtitan/observability/state_estimator tests/unit_tests/observability/state_estimator .scratch/training-state-estimator
```

Expected: no whitespace errors.

- [ ] **Step 7: Commit final docs and tracker evidence**

Commit:

```bash
git add .scratch/training-state-estimator
git commit -m "Document training state estimator implementation evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

## Execution Order

Implement tasks in order. Task 1 through Task 3 create the data substrate. Task
4 through Task 7 add modeling and recommendation output. Task 8 defines learned
interfaces without enabling learned behavior. Task 9 evaluates outputs. Task 10
packages the analyzer. Task 11 verifies the complete slice and records evidence.

Mandatory orchestration loop for every task:

1. The main agent writes a precise task brief with only the needed plan/spec
   excerpts, file paths, rootfs command rules, prohibited actions, expected
   artifacts, and verification commands.
2. A fresh implementation subagent performs the task. It must run Python
   commands through `scripts/rootfs/enter_rootfs.sh`, report changed files, and
   include verification evidence.
3. A fresh review subagent reviews the task for spec conformance, repository
   guidance, rootfs/dependency compliance, behavioral tests, safety boundaries,
   and unintended scope.
4. A fresh finalizer subagent reviews the implementation trace and review
   trace together. It identifies challenges, issues, bottlenecks, missed
   dependencies, ambiguous instructions, and process failures. It recommends
   plan edits or task/workstream-specific skills, and states whether the task
   implementation must be rerun.
5. The main agent acts only as orchestrator: integrate accepted outputs, record
   trace links and finalizer findings in the ledger or tracker, patch the plan
   or skills when the finalizer finds reusable process improvements, and rerun
   the task with a fresh implementation subagent when required.

Do not reuse a subagent for a different task. Do not let the main agent
implement a task locally unless a task is trivial, inseparable from
orchestration, or subagent tooling is unavailable; record that exception and
send the resulting change through the same review and finalizer loop.

Do not run tasks that write the same files in parallel. In particular,
`estimators.py` is touched by Tasks 4, 6, and 7 and must be serialized.

## Data Collection Plan

The first implementation collects by reading existing evidence, not by adding
new hot-path producers.

Required raw signals:

- `manifest.json`
- `indexes/artifacts.<process_id>.jsonl`
- `processes/<process_id>/outcome.json`

Tier 0 semantic signals:

- structured event JSONL rows indexed by `producer="structured_logger"`;
- run, attempt, process, rank, role, actor, host, PID, step, phase, wall clock,
  monotonic clock, and event sequence fields when present.

Tier 0 hardware signals:

- DCGM or hardware telemetry artifacts when indexed by producers;
- absence is represented as `uncollected`.

Tier 1 trace signals:

- profiler trace artifacts;
- memory snapshots;
- Flight Recorder dump declarations or dumps.

Tier 2 incident signals:

- incident records and diagnostic artifacts once those producers exist.

Derived signals:

- evidence graph;
- entity index;
- timeline;
- data-collection quality report;
- belief summary;
- diagnosis report;
- evaluation report.

## Modeling Plan

Classical v1 models:

- process liveness from process outcomes;
- phase and step timelines from structured events;
- peer-relative phase skew from event time grouped by phase and step;
- ambiguity-preserving fault hypotheses;
- fixed-lag incident timeline around failed outcomes or explicit windows;
- probe recommendations from ambiguous hypothesis sets.

Learned models:

- no model runs in v1;
- extension schemas allow future likelihood factors, residual predictions,
  sensor-quality predictions, and fault-mode priors;
- all learned output is advisory and marked with calibration state.

Safety model:

- no estimator output can change optimizer commit, checkpoint validity,
  membership, reroute, eviction, restart, or recovery policy;
- probe recommendations include required authority and live-safe flags.

## Evaluation Plan

Host-side first:

- synthetic bundle fixtures for nominal, failed process, malformed evidence,
  and peer skew;
- deterministic expectations for detection, expected mode, expected probe, and
  false alert behavior.

Metrics:

- detected count;
- false alert count on nominal fixtures;
- expected mode found count;
- expected probe found count;
- observability warning count.

Later extensions:

- real small run-evidence bundles;
- injected-fault bundles;
- detection latency;
- failure-domain localization;
- calibration after enough labeled cases exist.

## Self-Review

- Spec coverage: the plan covers data collection, evidence graph construction,
  deterministic modeling, fixed-lag timelines, probe recommendations, learned
  extension interfaces, evaluation, CLI integration, and verification.
- Safety boundary: every task preserves advisory-only behavior and avoids
  training control changes.
- Dependency boundary: rootfs is required for Python execution, while no task
  requires GPU allocation, distributed launch, torch import, or optional ML
  dependencies.
- Type consistency: exported names are introduced before downstream tasks use
  them.
