# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-local, append-only evidence for a core training run attempt."""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torchtitan.config.configurable import Configurable
from torchtitan.distributed.parallel_dims import ParallelDims


class RunEvidenceError(RuntimeError):
    """Base error raised when run-attempt evidence cannot be recorded."""


class EvidenceContractError(RunEvidenceError):
    """An evidence producer violated the append-only artifact contract."""


class EvidenceCollisionError(RunEvidenceError):
    """A process attempted to reuse a run-attempt process index."""


class EvidenceWriteError(RunEvidenceError):
    """A filesystem operation required to record evidence failed."""


class ArtifactState(str, enum.Enum):
    DECLARED = "declared"
    COMPLETE = "complete"
    FAILED = "failed"
    RETIRED = "retired"


class ArtifactRelation(str, enum.Enum):
    INPUT = "input"
    OUTPUT = "output"


_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_URI_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_ACTIVE_EVIDENCE: RunEvidence | None = None
_PHASE_STACK: ContextVar[tuple[str, ...]] = ContextVar("run_evidence_phase", default=())
_logger = logging.getLogger(__name__)
_TRANSITIONS = {
    ArtifactState.DECLARED: {
        ArtifactState.COMPLETE,
        ArtifactState.FAILED,
        ArtifactState.RETIRED,
    },
    ArtifactState.COMPLETE: {ArtifactState.RETIRED},
}


class RunEvidence(Configurable):
    """Own one process-local recorder and expose it through module functions."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = True
        folder: str = "run_evidence"

    def __init__(
        self,
        config: Config,
        *,
        dump_folder: str,
        job_config: Mapping[str, Any],
        role: str,
        actor_id: str,
    ) -> None:
        self.config = config
        self.dump_folder = dump_folder
        self.job_config = dict(job_config)
        self.role = role
        self.actor_id = actor_id
        self.run_id: str | None = None
        self.attempt_id: str | None = None
        self._index_file: Any | None = None
        self._process_id: str | None = None
        self._process_context: dict[str, Any] = {}
        self._distributed_context: dict[str, Any] = {}
        self._artifacts: dict[str, ArtifactState] = {}
        self._artifact_seq = 0
        self._event_seq = 0
        self._lock = threading.Lock()
        self._entered_monotonic_ns: int | None = None

    def __enter__(self) -> RunEvidence:
        global _ACTIVE_EVIDENCE
        if not self.config.enable:
            return self
        if _ACTIVE_EVIDENCE is not None:
            raise RuntimeError("RunEvidence is already active in this process")
        self._validate_config()
        self.run_id, self.attempt_id = self._resolve_identity()
        self._publish_manifest()
        self._open_process_index()
        self._entered_monotonic_ns = time.monotonic_ns()
        _ACTIVE_EVIDENCE = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _ACTIVE_EVIDENCE
        outcome_error: EvidenceWriteError | None = None
        try:
            if self.config.enable and self._entered_monotonic_ns is not None:
                try:
                    self._write_outcome(exc_type, exc_value)
                except EvidenceWriteError as error:
                    if exc_type is None:
                        outcome_error = error
                    else:
                        _logger.exception(
                            "run evidence outcome write failed while handling another exception"
                        )
            if self._index_file is not None:
                try:
                    self._index_file.close()
                except OSError as error:
                    if exc_type is None and outcome_error is None:
                        outcome_error = EvidenceWriteError(
                            "failed to close run evidence process index"
                        )
                        outcome_error.__cause__ = error
                    else:
                        _logger.exception(
                            "run evidence index close failed while handling another exception"
                        )
                finally:
                    self._index_file = None
        finally:
            if _ACTIVE_EVIDENCE is self:
                _ACTIVE_EVIDENCE = None
        if outcome_error is not None:
            raise outcome_error
        return False

    def bind_distributed(
        self, parallel_dims: ParallelDims, device: torch.device
    ) -> None:
        self._distributed_context = {
            name: getattr(parallel_dims, name)
            for name in (
                "dp_replicate",
                "dp_shard",
                "cp",
                "tp",
                "pp",
                "ep",
                "world_size",
            )
        }
        self._distributed_context["device_type"] = device.type
        if device.index is not None:
            self._distributed_context["device_index"] = device.index

    def _validate_config(self) -> None:
        folder = self.config.folder
        if (
            not folder
            or Path(folder).is_absolute()
            or len(Path(folder).parts) != 1
            or folder in {".", ".."}
        ):
            raise ValueError(
                "RunEvidence.Config.folder must be one relative path segment"
            )
        if _URI_RE.match(self.dump_folder):
            raise ValueError("RunEvidence.dump_folder must be a local filesystem path")

    def _resolve_identity(self) -> tuple[str, str]:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        elastic_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
        if elastic_id.lower() == "none":
            elastic_id = ""
        run_id = os.environ.get("TORCHTITAN_RUN_ID", "") or elastic_id
        attempt_base = os.environ.get("TORCHTITAN_ATTEMPT_ID", "") or elastic_id
        if world_size != 1:
            if not run_id:
                raise ValueError(
                    "TORCHTITAN_RUN_ID is required for a multi-process run"
                )
            if not attempt_base:
                raise ValueError(
                    "TORCHTITAN_ATTEMPT_ID is required for a multi-process run"
                )
        run_id = run_id or uuid.uuid4().hex
        attempt_base = attempt_base or uuid.uuid4().hex
        restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
        attempt_id = (
            f"{attempt_base}-restart-{restart_count}"
            if restart_count > 0
            else attempt_base
        )
        for name, value in (("run_id", run_id), ("attempt_id", attempt_id)):
            if not _IDENTIFIER_RE.fullmatch(value):
                raise ValueError(f"{name} must match {_IDENTIFIER_RE.pattern}")
        return run_id, attempt_id

    @property
    def _attempt_path(self) -> Path:
        assert self.run_id is not None
        assert self.attempt_id is not None
        return (
            Path(self.dump_folder) / self.config.folder / self.run_id / self.attempt_id
        )

    def _publish_manifest(self) -> None:
        attempt_path = self._attempt_path
        try:
            attempt_path.mkdir(parents=True, exist_ok=True)
            manifest = self._manifest()
            manifest_path = attempt_path / "manifest.json"
            try:
                with manifest_path.open("x", encoding="utf-8") as handle:
                    handle.write(_canonical_json(manifest) + "\n")
            except FileExistsError:
                existing = self._read_manifest(manifest_path)
                self._validate_shared_manifest(existing, manifest)
        except RunEvidenceError:
            raise
        except OSError as error:
            raise EvidenceWriteError(
                "failed to publish run evidence manifest"
            ) from error

    def _manifest(self) -> dict[str, Any]:
        normalized = json.loads(_canonical_json(self.job_config))
        config_json = _canonical_json(normalized)
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "config": {
                "normalized": normalized,
                "sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            },
            "source": _source_state(),
            "command": " ".join(sys.argv),
            "runtime": {"python": sys.version, "torch": torch.__version__},
        }

    def _read_manifest(self, path: Path) -> dict[str, Any]:
        last_error: OSError | json.JSONDecodeError | None = None
        for _ in range(3):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                last_error = error
                time.sleep(0.01)
        raise EvidenceWriteError(
            "existing run evidence manifest could not be read"
        ) from last_error

    def _validate_shared_manifest(
        self, existing: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> None:
        fields = (
            "schema_version",
            "run_id",
            "attempt_id",
            "config",
            "source",
            "command",
            "runtime",
        )
        if any(existing.get(field) != expected[field] for field in fields):
            raise EvidenceContractError("existing run evidence manifest does not match")

    def _open_process_index(self) -> None:
        global_rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._process_id = f"{self.role}.{self.actor_id}.global_rank_{global_rank:06d}"
        self._process_context = {
            "process_id": self._process_id,
            "role": self.role,
            "actor_id": self.actor_id,
            "host_name": socket.gethostname(),
            "pid": os.getpid(),
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": world_size,
        }
        index_path = (
            self._attempt_path / "indexes" / f"artifacts.{self._process_id}.jsonl"
        )
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            self._index_file = index_path.open("x", encoding="utf-8")
        except FileExistsError as error:
            raise EvidenceCollisionError(
                f"run evidence process index already exists: {self._process_id}"
            ) from error
        except OSError as error:
            raise EvidenceWriteError(
                "failed to create run evidence process index"
            ) from error

    def _record_artifact(
        self,
        *,
        producer: str,
        kind: str,
        path: str | os.PathLike[str],
        state: ArtifactState,
        relation: ArtifactRelation,
        artifact_id: str | None,
        step: int | None,
        metadata: Mapping[str, Any] | None,
    ) -> str:
        if not isinstance(state, ArtifactState):
            raise EvidenceContractError("artifact state must be an ArtifactState")
        if not isinstance(relation, ArtifactRelation):
            raise EvidenceContractError("artifact relation must be an ArtifactRelation")
        normalized_path, path_type, local_path = self._normalize_path(path)
        normalized_metadata = self._normalize_metadata(metadata)
        expected_id = self._artifact_id(producer, kind, relation, normalized_path)
        if artifact_id is not None and artifact_id != expected_id:
            raise EvidenceContractError("artifact_id does not match artifact identity")
        artifact_id = expected_id
        previous_state = self._artifacts.get(artifact_id)
        if previous_state is not None and state not in _TRANSITIONS.get(
            previous_state, set()
        ):
            raise EvidenceContractError(
                f"invalid artifact transition: {previous_state.value} -> {state.value}"
            )
        if (
            state is ArtifactState.COMPLETE
            and local_path is not None
            and not local_path.exists()
        ):
            raise EvidenceContractError(
                f"completed local artifact does not exist: {local_path}"
            )
        row = {
            "schema_version": 1,
            "record_type": "artifact",
            "artifact_id": artifact_id,
            "producer": producer,
            "kind": kind,
            "relation": relation.value,
            "state": state.value,
            "path": normalized_path,
            "path_type": path_type,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "artifact_seq": self._artifact_seq,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            **self._process_context,
            "metadata": normalized_metadata,
        }
        if step is not None:
            row["step"] = step
        phase = self._phase()
        if phase is not None:
            row["phase"] = phase
        with self._lock:
            self._append_row(row)
            self._artifacts[artifact_id] = state
            self._artifact_seq += 1
        return artifact_id

    def _normalize_path(
        self, path: str | os.PathLike[str]
    ) -> tuple[str, str, Path | None]:
        raw_path = os.fspath(path)
        if _URI_RE.match(raw_path):
            return raw_path, "uri", None
        dump_path = Path(self.dump_folder).resolve()
        local_path = Path(raw_path)
        resolved_path = (
            local_path if local_path.is_absolute() else dump_path / local_path
        ).resolve()
        try:
            return (
                resolved_path.relative_to(dump_path).as_posix(),
                "dump_relative",
                resolved_path,
            )
        except ValueError:
            return str(resolved_path), "external_absolute", resolved_path

    def _normalize_metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        if metadata is not None and not isinstance(metadata, Mapping):
            raise EvidenceContractError("artifact metadata must be a mapping")
        try:
            return json.loads(_canonical_json(dict(metadata or {})))
        except (TypeError, ValueError) as error:
            raise EvidenceContractError(
                "artifact metadata must be JSON serializable"
            ) from error

    def _artifact_id(
        self,
        producer: str,
        kind: str,
        relation: ArtifactRelation,
        normalized_path: str,
    ) -> str:
        identity = {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "producer": producer,
            "kind": kind,
            "relation": relation.value,
            "path": normalized_path,
        }
        return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[
            :32
        ]

    def _append_row(self, row: Mapping[str, Any]) -> None:
        assert self._index_file is not None
        try:
            self._index_file.write(_canonical_json(row) + "\n")
            self._index_file.flush()
        except (OSError, TypeError, ValueError) as error:
            raise EvidenceWriteError(
                "failed to append run evidence artifact"
            ) from error

    def _event_context(self) -> dict[str, Any]:
        with self._lock:
            event_seq = self._event_seq
            self._event_seq += 1
        context = {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "event_seq": event_seq,
            **self._process_context,
            **self._distributed_context,
        }
        phase = self._phase()
        if phase is not None:
            context["phase"] = phase
        return context

    @contextmanager
    def _bind_phase(self, phase: str) -> Iterator[None]:
        if not phase:
            raise ValueError("phase must not be empty")
        token = _PHASE_STACK.set((*_PHASE_STACK.get(), phase))
        try:
            yield
        finally:
            _PHASE_STACK.reset(token)

    def _phase(self) -> str | None:
        phase_stack = _PHASE_STACK.get()
        return "/".join(phase_stack) if phase_stack else None

    def _write_outcome(self, exc_type: Any, exc_value: Any) -> None:
        assert self._process_id is not None
        assert self._entered_monotonic_ns is not None
        if exc_type is None:
            outcome = "succeeded"
        elif issubclass(exc_type, KeyboardInterrupt):
            outcome = "interrupted"
        else:
            outcome = "failed"
        record: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "process_outcome",
            "outcome": outcome,
            "elapsed_monotonic_ns": time.monotonic_ns() - self._entered_monotonic_ns,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            **self._process_context,
        }
        if exc_type is not None:
            record["exception_type"] = exc_type.__name__
            record["exception_message"] = str(exc_value)[:512]
        outcome_path = (
            self._attempt_path / "processes" / self._process_id / "outcome.json"
        )
        try:
            outcome_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=".outcome.", dir=outcome_path.parent, text=True
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(_canonical_json(record) + "\n")
                os.replace(temporary_path, outcome_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except OSError as error:
            raise EvidenceWriteError(
                "failed to write run evidence process outcome"
            ) from error


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_state() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": False}


def record_artifact(
    *,
    producer: str,
    kind: str,
    path: str | os.PathLike[str],
    state: ArtifactState = ArtifactState.COMPLETE,
    relation: ArtifactRelation = ArtifactRelation.OUTPUT,
    artifact_id: str | None = None,
    step: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    evidence = _ACTIVE_EVIDENCE
    if evidence is None:
        return None
    handling_exception = sys.exc_info()[0] is not None
    try:
        return evidence._record_artifact(
            producer=producer,
            kind=kind,
            path=path,
            state=state,
            relation=relation,
            artifact_id=artifact_id,
            step=step,
            metadata=metadata,
        )
    except EvidenceWriteError:
        if handling_exception:
            _logger.exception(
                "run evidence artifact append failed while handling another exception"
            )
            return None
        raise


def event_context() -> dict[str, Any]:
    evidence = _ACTIVE_EVIDENCE
    return {} if evidence is None else evidence._event_context()


def bind_phase(phase: str) -> AbstractContextManager[None]:
    evidence = _ACTIVE_EVIDENCE
    return nullcontext() if evidence is None else evidence._bind_phase(phase)
