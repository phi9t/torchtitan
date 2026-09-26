# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observed single-rank Qwen3 tracer bullet for the formal verifier.

The runtime adapter in this module is deliberately experiment-owned. It builds
the ordinary ``qwen3_debugmodel`` configuration through ``ConfigManager``, lets
the core ``Trainer`` retain control of the step, and observes the step through
PyTorch hooks plus an identity-preserving dataloader wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger
from torchtitan.trainer import Trainer


RAW_SCHEMA = "qwen3.formal.raw.v0"
SCOUT_SCHEMA = "qwen3.formal.scout.v0"
SCOUT_A_RUN_ID = "qfv-scout-a-seed42-fix1"
SCOUT_A_ATTEMPT_ID = "single-rank-cuda-fix1"
SOURCE_MANIFEST_SCHEMA = "qwen3.formal.scout.source-manifest.v1"
RUNTIME_MANIFEST_SCHEMA = "qwen3.formal.scout.runtime-manifest.v1"
STAGE_MANIFEST_SCHEMA = "qwen3.formal.scout.stage-manifest.v0"
EVIDENCE_MANIFEST_SCHEMA = "qwen3.formal.scout.evidence-manifest.v1"
SCOUT_A_REQUIRED_STAGES = (
    ("source_manifest", "checker/source-manifest.log"),
    ("focused_pytest", "checker/focused-pytest.log"),
    ("cuda_pytest", "checker/cuda-pytest.log"),
    ("owning_pytest", "checker/owning-pytest.log"),
    ("artifact_sync", "checker/artifact-sync.log"),
    ("formal_networked", "checker/formal-networked.log"),
    ("formal_no_fetch", "checker/formal-no-fetch.log"),
    ("lint", "checker/lint.log"),
)

_EXPECTED_LIFECYCLE = (
    "step.started",
    "batch.observed",
    "forward.started",
    "forward.completed",
    "backward.started",
    "gradient.ready",
    "backward.completed",
    "optimizer.started",
    "optimizer.mutated",
    "step.completed",
)
_PHASE_BY_KIND = {
    "step.started": "step",
    "batch.observed": "input",
    "forward.started": "forward",
    "forward.completed": "forward",
    "backward.started": "backward",
    "gradient.ready": "backward",
    "backward.completed": "backward",
    "optimizer.started": "optimizer",
    "optimizer.mutated": "optimizer",
    "step.completed": "step",
}
_LEAN_KIND = {
    "step.started": "stepStarted",
    "batch.observed": "batchObserved",
    "forward.started": "forwardStarted",
    "forward.completed": "forwardCompleted",
    "backward.started": "backwardStarted",
    "gradient.ready": "gradientReady",
    "gradient.missing": "gradientMissing",
    "backward.completed": "backwardCompleted",
    "optimizer.started": "optimizerStarted",
    "optimizer.mutated": "optimizerMutated",
    "step.completed": "stepCompleted",
}
_LEAN_PHASE = {
    "step": "step",
    "input": "input",
    "forward": "forward",
    "backward": "backward",
    "optimizer": "optimizer",
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ScoutAPaths:
    attempt_dir: Path
    raw_trace: Path
    normalized_trace: Path
    generated_tla_facts: Path
    generated_tla_bad_facts: Path
    generated_lean_facts: Path
    generated_lean_bad_facts: Path
    runtime_manifest: Path


def scout_a_cli_args(dump_folder: str | Path) -> list[str]:
    """Return the complete ConfigManager CLI profile for Scout A."""

    return [
        "--module",
        "qwen3",
        "--config",
        "qwen3_debugmodel",
        "--dump-folder",
        str(dump_folder),
        "--training.local-batch-size",
        "1",
        "--training.global-batch-size",
        "1",
        "--training.seq-len",
        "128",
        "--training.steps",
        "1",
        "--training.dtype",
        "bfloat16",
        "--optimizer.param-groups.0.optimizer-name",
        "AdamW",
        "--parallelism.data-parallel-replicate-degree",
        "1",
        "--parallelism.data-parallel-shard-degree",
        "1",
        "--parallelism.tensor-parallel-degree",
        "1",
        "--parallelism.pipeline-parallel-degree",
        "1",
        "--parallelism.context-parallel-degree",
        "1",
        "--parallelism.expert-parallel-degree",
        "1",
        "--parallelism.no-enable-sequence-parallel",
        "--checkpoint.no-enable",
        "--checkpoint.no-initial-load-in-hf",
        "--checkpoint.no-initial-load-model-only",
        "--checkpoint.load-step",
        "-1",
        "--debug.seed",
        "42",
        "--debug.deterministic",
        "--debug.no-deterministic-warn-only",
        "--debug.no-enable-structured-logging",
        "--compile.no-enable",
        "--dataloader.dataset",
        "c4_test",
        "--dataloader.num-workers",
        "0",
        "--hf-assets-path",
        "./tests/assets/tokenizer",
        "activation-checkpoint:none",
    ]


def build_scout_a_config(
    dump_folder: str | Path,
) -> tuple[Trainer.Config, list[str]]:
    """Select qwen3_debugmodel and apply Scout A through normal CLI precedence."""

    cli_args = scout_a_cli_args(dump_folder)
    config = ConfigManager().parse_args(cli_args)
    if not isinstance(config, Trainer.Config):
        raise TypeError("qwen3_debugmodel did not produce Trainer.Config")
    _scout_a_profile(config)
    return config, cli_args


def _scout_a_profile(config: Trainer.Config) -> dict[str, object]:
    model_spec = config.model_spec
    if model_spec is None:
        raise ValueError("Scout A requires a model spec")
    parallelism = config.parallelism
    degrees = {
        "data_parallel_replicate": parallelism.data_parallel_replicate_degree,
        "data_parallel_shard": parallelism.data_parallel_shard_degree,
        "context_parallel": parallelism.context_parallel_degree,
        "tensor_parallel": parallelism.tensor_parallel_degree,
        "pipeline_parallel": parallelism.pipeline_parallel_degree,
        "expert_parallel": parallelism.expert_parallel_degree,
    }
    expected = {
        "model": (model_spec.name, model_spec.flavor),
        "batch": (
            config.training.local_batch_size,
            config.training.global_batch_size,
        ),
        "sequence_and_steps": (config.training.seq_len, config.training.steps),
        "dtype": config.training.dtype,
        "degrees": degrees,
        "sequence_parallel": parallelism.enable_sequence_parallel,
        "seed_and_determinism": (
            config.debug.seed,
            config.debug.deterministic,
            config.debug.deterministic_warn_only,
        ),
        "dataset": config.dataloader.dataset,
        "tokenizer": config.hf_assets_path,
        "checkpoint": (
            config.checkpoint.enable,
            config.checkpoint.initial_load_path,
            config.checkpoint.initial_load_in_hf,
            config.checkpoint.create_seed_checkpoint,
        ),
        "compile": config.compile.enable,
        "activation_checkpoint": config.activation_checkpoint,
    }
    required = {
        "model": ("qwen3", "debugmodel"),
        "batch": (1, 1),
        "sequence_and_steps": (128, 1),
        "dtype": "bfloat16",
        "degrees": {name: 1 for name in degrees},
        "sequence_parallel": False,
        "seed_and_determinism": (42, True, False),
        "dataset": "c4_test",
        "tokenizer": "./tests/assets/tokenizer",
        "checkpoint": (False, None, False, False),
        "compile": False,
        "activation_checkpoint": None,
    }
    if expected != required:
        raise ValueError(
            "Scout A config does not match its fixed profile: "
            + _canonical_json_text(expected).strip()
        )
    optimizer_names = [group.optimizer_name for group in config.optimizer.param_groups]
    if optimizer_names != ["AdamW"]:
        raise ValueError(
            f"Scout A requires one AdamW optimizer, found {optimizer_names}"
        )
    return {
        "config_module": "qwen3",
        "config_name": "qwen3_debugmodel",
        "model_name": model_spec.name,
        "model_flavor": model_spec.flavor,
        "optimizer": "AdamW",
        "dtype": config.training.dtype,
        "local_batch_size": config.training.local_batch_size,
        "global_batch_size": config.training.global_batch_size,
        "sequence_length": config.training.seq_len,
        "tokens_per_optimizer_step": 128,
        "gradient_accumulation_steps": 1,
        "optimizer_steps": config.training.steps,
        "seed": config.debug.seed,
        "deterministic": config.debug.deterministic,
        "dataset": config.dataloader.dataset,
        "tokenizer_path": "tests/assets/tokenizer",
        "checkpoint_load": False,
        "checkpoint_save": False,
    }


def normalize_raw_trace(raw_trace: Mapping[str, object]) -> dict[str, object]:
    """Normalize observed raw events into deterministic scout.v0 facts."""

    if raw_trace.get("schema") != RAW_SCHEMA:
        raise ValueError(f"raw trace schema must be {RAW_SCHEMA}")
    identity = _mapping(raw_trace, "identity")
    run_id = identity.get("run_id")
    attempt_id = identity.get("attempt_id")
    process_id = identity.get("process_id")
    if not isinstance(run_id, str) or not isinstance(attempt_id, str):
        raise TypeError("raw run_id and attempt_id must be strings")
    if not isinstance(process_id, str) or not process_id:
        raise TypeError("raw process_id must be a non-empty string")
    _validate_identifier(run_id, "run_id")
    _validate_identifier(attempt_id, "attempt_id")
    rank = _integer(identity, "rank")
    events = _mapping_list(raw_trace, "events")
    ordered = sorted(events, key=lambda event: _integer(event, "source_order"))
    expected_order = list(range(1, len(ordered) + 1))
    actual_order = [_integer(event, "source_order") for event in ordered]
    if actual_order != expected_order:
        raise ValueError(f"raw source_order must be contiguous, found {actual_order}")

    raw_event_ids = [str(event.get("raw_event_id", "")) for event in ordered]
    if len(set(raw_event_ids)) != len(raw_event_ids):
        raise ValueError("duplicate raw event ID")
    for event, raw_event_id in zip(ordered, raw_event_ids, strict=True):
        expected_raw_id = f"raw-r{rank}-{_integer(event, 'source_order'):06d}"
        if raw_event_id != expected_raw_id:
            raise ValueError(
                f"raw event ID must be {expected_raw_id}, found {raw_event_id}"
            )

    raw_to_normalized = {
        str(event["raw_event_id"]): (
            _normalized_event_id(
                run_id,
                attempt_id,
                rank,
                _integer(event, "source_order"),
            )
        )
        for event in ordered
    }
    normalized_events: list[dict[str, object]] = []
    provenance: dict[str, object] = {}
    lineage = deepcopy(_mapping(raw_trace, "lineage"))
    device = deepcopy(_mapping(raw_trace, "device"))
    mesh = deepcopy(_mapping(raw_trace, "mesh"))
    for raw_event in ordered:
        raw_event_id = str(raw_event["raw_event_id"])
        event_id = raw_to_normalized[raw_event_id]
        raw_digest = _sha256_json(raw_event)
        raw_predecessors = _string_list(raw_event, "causal_predecessors")
        unknown = [pred for pred in raw_predecessors if pred not in raw_to_normalized]
        if unknown:
            raise ValueError(
                f"raw event {raw_event_id} has unknown predecessors {unknown}"
            )
        event_provenance = {
            "raw_event_id": raw_event_id,
            "raw_sha256": raw_digest,
        }
        normalized_event = {
            "event_id": event_id,
            "run_id": str(identity["run_id"]),
            "attempt_id": str(identity["attempt_id"]),
            "process_id": str(identity["process_id"]),
            "rank": _integer(identity, "rank"),
            "local_rank": _integer(identity, "local_rank"),
            "world_size": _integer(identity, "world_size"),
            "device": deepcopy(device),
            "mesh": deepcopy(mesh),
            "clock": {
                "domain": "rank_source_order",
                "order": _integer(raw_event, "source_order"),
            },
            "step": _integer(raw_event, "step"),
            "phase": str(raw_event["phase"]),
            "lineage": deepcopy(lineage),
            "kind": str(raw_event["kind"]),
            "work": {
                "id": f"{event_id}:work",
                "role": str(raw_event["kind"]),
            },
            "causal_predecessors": [
                raw_to_normalized[pred] for pred in raw_predecessors
            ],
            "observation": deepcopy(raw_event.get("observation", {})),
            "provenance": event_provenance,
        }
        normalized_events.append(normalized_event)
        provenance[event_id] = event_provenance

    normalized: dict[str, object] = {
        "schema": SCOUT_SCHEMA,
        "trace_id": "",
        "run_id": str(identity["run_id"]),
        "attempt_id": str(identity["attempt_id"]),
        "process": {
            "process_id": str(identity["process_id"]),
            "rank": _integer(identity, "rank"),
            "local_rank": _integer(identity, "local_rank"),
            "world_size": _integer(identity, "world_size"),
        },
        "device": device,
        "mesh": mesh,
        "profile": deepcopy(_mapping(raw_trace, "profile")),
        "lineage": lineage,
        "events": normalized_events,
        "provenance": provenance,
    }
    normalized["trace_id"] = _expected_trace_id(normalized)
    validate_normalized_trace(normalized)
    return normalized


def validate_normalized_trace(trace: Mapping[str, object]) -> None:
    """Reject incomplete identities, provenance, lifecycle, and causal order."""

    if trace.get("schema") != SCOUT_SCHEMA:
        raise ValueError(f"normalized trace schema must be {SCOUT_SCHEMA}")
    run_id = str(trace.get("run_id", ""))
    attempt_id = str(trace.get("attempt_id", ""))
    _validate_identifier(run_id, "run_id")
    _validate_identifier(attempt_id, "attempt_id")
    process = _mapping(trace, "process")
    if (
        _integer(process, "rank") != 0
        or _integer(process, "local_rank") != 0
        or _integer(process, "world_size") != 1
    ):
        raise ValueError("Scout A normalized process identity must be rank 0 of 1")
    mesh = _mapping(trace, "mesh")
    if mesh.get("degrees") != [1, 1, 1, 1, 1, 1]:
        raise ValueError("Scout A normalized mesh degrees must all equal one")
    events = _mapping_list(trace, "events")
    normalized_event_ids = [str(event.get("event_id", "")) for event in events]
    if len(set(normalized_event_ids)) != len(normalized_event_ids):
        raise ValueError("duplicate normalized event ID")
    if [str(event.get("kind")) for event in events] != list(_EXPECTED_LIFECYCLE):
        raise ValueError(
            "Scout A lifecycle must be exactly "
            + ", ".join(_EXPECTED_LIFECYCLE)
            + "; gradient.ready must precede optimizer.mutated and "
            "optimizer.mutated must precede step.completed"
        )
    event_ids = {str(event.get("event_id")) for event in events}
    root_provenance = _mapping(trace, "provenance")
    if set(root_provenance) != event_ids:
        raise ValueError("root provenance must cover every normalized event")
    prior: set[str] = set()
    for expected_order, event in enumerate(events, start=1):
        event_id = str(event.get("event_id", ""))
        expected_event_id = _normalized_event_id(
            run_id,
            attempt_id,
            _integer(process, "rank"),
            expected_order,
        )
        if event_id != expected_event_id:
            raise ValueError(
                f"normalized event ID must be {expected_event_id}, found {event_id}"
            )
        if event.get("run_id") != run_id or event.get("attempt_id") != attempt_id:
            raise ValueError(f"event {event_id} has inconsistent run/attempt identity")
        event_process = {
            "process_id": event.get("process_id"),
            "rank": event.get("rank"),
            "local_rank": event.get("local_rank"),
            "world_size": event.get("world_size"),
        }
        if event_process != process:
            raise ValueError(f"event {event_id} disagrees with root process identity")
        if event.get("device") != trace.get("device"):
            raise ValueError(f"event {event_id} disagrees with root device identity")
        if event.get("mesh") != trace.get("mesh"):
            raise ValueError(f"event {event_id} disagrees with root mesh identity")
        if event.get("lineage") != trace.get("lineage"):
            raise ValueError(f"event {event_id} disagrees with root lineage identity")
        clock = _mapping(event, "clock")
        if clock != {"domain": "rank_source_order", "order": expected_order}:
            raise ValueError(f"event {event_id} has invalid logical clock/order")
        kind = str(event.get("kind"))
        if event.get("phase") != _PHASE_BY_KIND[kind]:
            raise ValueError(f"event {event_id} has invalid phase for {kind}")
        if event.get("step") != 1:
            raise ValueError(f"event {event_id} has invalid step identity")
        work = _mapping(event, "work")
        if work != {"id": f"{event_id}:work", "role": kind}:
            raise ValueError(f"event {event_id} has invalid work identity")
        if not isinstance(event.get("provenance"), dict):
            raise ValueError(f"event {event_id} is missing raw provenance")
        event_provenance = _mapping(event, "provenance")
        if not event_provenance.get("raw_event_id") or not re.fullmatch(
            r"[0-9a-f]{64}", str(event_provenance.get("raw_sha256", ""))
        ):
            raise ValueError(f"event {event_id} is missing raw provenance")
        if root_provenance.get(event_id) != event_provenance:
            raise ValueError(f"event {event_id} provenance map disagrees")
        predecessors = _string_list(event, "causal_predecessors")
        expected_predecessors = (
            [] if expected_order == 1 else [normalized_event_ids[expected_order - 2]]
        )
        if predecessors != expected_predecessors or any(
            predecessor not in prior for predecessor in predecessors
        ):
            raise ValueError(f"event {event_id} violates causal order")
        prior.add(event_id)
    batch = next(event for event in events if event["kind"] == "batch.observed")
    batch_observation = _mapping(batch, "observation")
    lineage = _mapping(trace, "lineage")
    if batch_observation.get("batch_sha256") != lineage.get("first_batch_sha256"):
        raise ValueError("first-batch identity disagrees with lineage")
    optimizer = next(event for event in events if event["kind"] == "optimizer.mutated")
    optimizer_observation = _mapping(optimizer, "observation")
    if optimizer_observation.get("mutated") is not True or optimizer_observation.get(
        "before_sha256"
    ) == optimizer_observation.get("after_sha256"):
        raise ValueError("optimizer.mutated does not prove parameter mutation")
    expected_trace_id = _expected_trace_id(trace)
    if trace.get("trace_id") != expected_trace_id:
        raise ValueError(
            f"trace_id must be {expected_trace_id}, found {trace.get('trace_id')!r}"
        )


def export_scout_a_tla_facts(
    trace: Mapping[str, object], *, module_name: str = "ScoutAFacts"
) -> str:
    """Export observed values only; lifecycle propositions live elsewhere."""

    validate_normalized_trace(trace)
    events = _mapping_list(trace, "events")
    return _export_tla_fact_module(trace, events, module_name=module_name)


def export_scout_a_tla_bad_facts(trace: Mapping[str, object]) -> str:
    """Export one controlled mutation for the named negative TLC check."""

    validate_normalized_trace(trace)
    events = deepcopy(_mapping_list(trace, "events"))
    gradient = next(event for event in events if event["kind"] == "gradient.ready")
    gradient["kind"] = "gradient.missing"
    return _export_tla_fact_module(
        trace,
        events,
        module_name="ScoutABadFacts",
        mutation="gradient_ready_fact_missing",
    )


def _export_tla_fact_module(
    trace: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    *,
    module_name: str,
    mutation: str | None = None,
) -> str:
    event_ids = [str(event["event_id"]) for event in events]
    predecessors = []
    raw_sources = []
    for event in events:
        event_id = str(event["event_id"])
        pred_set = (
            "{"
            + ", ".join(
                _tla_string(value)
                for value in _string_list(event, "causal_predecessors")
            )
            + "}"
        )
        predecessors.append((_tla_string(event_id), pred_set))
        provenance = _mapping(event, "provenance")
        raw_sources.append(
            (_tla_string(event_id), _tla_string(str(provenance["raw_event_id"])))
        )
    lines = [
        f"------------------------------ MODULE {module_name} ------------------------------",
        "EXTENDS Naturals, Sequences, FiniteSets, TLC",
        "",
        f"RunId == {_tla_string(str(trace['run_id']))}",
        f"AttemptId == {_tla_string(str(trace['attempt_id']))}",
        f"TraceId == {_tla_string(str(trace['trace_id']))}",
    ]
    if mutation is not None:
        lines.append(f"MutationName == {_tla_string(mutation)}")
    lines.extend(
        [
            f"EventIds == <<{', '.join(_tla_string(value) for value in event_ids)}>>",
            "EventKinds == <<"
            + ", ".join(_tla_string(str(event["kind"])) for event in events)
            + ">>",
            "EventPhases == <<"
            + ", ".join(_tla_string(str(event["phase"])) for event in events)
            + ">>",
            "EventPredecessors == " + _tla_function(predecessors),
            "RawSource == " + _tla_function(raw_sources),
            "",
            "=============================================================================",
            "",
        ]
    )
    return "\n".join(lines)


def export_scout_a_lean_facts(
    trace: Mapping[str, object], *, namespace: str = "ScoutAFacts"
) -> str:
    """Export Lean values only; definitions and theorems stay maintained."""

    validate_normalized_trace(trace)
    return _export_lean_fact_module(
        trace,
        _mapping_list(trace, "events"),
        namespace=namespace,
    )


def export_scout_a_lean_bad_facts(trace: Mapping[str, object]) -> str:
    """Export the same controlled missing-gradient fact for Lean."""

    validate_normalized_trace(trace)
    events = deepcopy(_mapping_list(trace, "events"))
    gradient = next(event for event in events if event["kind"] == "gradient.ready")
    gradient["kind"] = "gradient.missing"
    return _export_lean_fact_module(
        trace,
        events,
        namespace="ScoutABadFacts",
        mutation="gradient_ready_fact_missing",
    )


def _export_lean_fact_module(
    trace: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    *,
    namespace: str,
    mutation: str | None = None,
) -> str:
    lines = [
        "import ScoutLifecycle",
        "",
        f"namespace Qwen3Formal.{namespace}",
        "",
        "open Qwen3Formal",
        "",
        f"def runId : String := {_lean_string(str(trace['run_id']))}",
        f"def attemptId : String := {_lean_string(str(trace['attempt_id']))}",
        f"def traceId : String := {_lean_string(str(trace['trace_id']))}",
    ]
    if mutation is not None:
        lines.append(f"def mutationName : String := {_lean_string(mutation)}")
    lines.extend(["", "def observedEvents : List ObservedEvent := ["])
    for event in events:
        predecessors = ", ".join(
            _lean_string(value) for value in _string_list(event, "causal_predecessors")
        )
        provenance = _mapping(event, "provenance")
        lines.extend(
            [
                "  {",
                f"    id := {_lean_string(str(event['event_id']))}",
                f"    rawSourceId := {_lean_string(str(provenance['raw_event_id']))}",
                f"    kind := .{_LEAN_KIND[str(event['kind'])]}",
                f"    phase := .{_LEAN_PHASE[str(event['phase'])]}",
                f"    predecessors := [{predecessors}]",
                "  },",
            ]
        )
    lines.extend(["]", "", f"end Qwen3Formal.{namespace}", ""])
    return "\n".join(lines)


def artifact_sync_mismatches(
    trace: Mapping[str, object],
    formal_dir: str | Path,
    fixture_path: str | Path,
) -> tuple[str, ...]:
    """Return deterministic fixture/generated paths that disagree."""

    validate_normalized_trace(trace)
    formal_path = Path(formal_dir)
    expected = {
        Path(fixture_path): _canonical_json_text(trace),
        formal_path / "ScoutAFacts.tla": export_scout_a_tla_facts(trace),
        formal_path / "ScoutABadFacts.tla": export_scout_a_tla_bad_facts(trace),
        formal_path / "ScoutAFacts.lean": export_scout_a_lean_facts(trace),
        formal_path / "ScoutABadFacts.lean": export_scout_a_lean_bad_facts(trace),
    }
    mismatches = []
    for path, expected_text in expected.items():
        if not path.is_file() or path.read_text() != expected_text:
            mismatches.append(str(path))
    return tuple(mismatches)


def sync_checked_in_artifacts(
    trace: Mapping[str, object],
    formal_dir: str | Path,
    fixture_path: str | Path,
) -> None:
    """Write the compact normalized fixture and generated fact modules."""

    validate_normalized_trace(trace)
    formal_path = Path(formal_dir)
    fixture = Path(fixture_path)
    formal_path.mkdir(parents=True, exist_ok=True)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(_canonical_json_text(trace))
    (formal_path / "ScoutAFacts.tla").write_text(export_scout_a_tla_facts(trace))
    (formal_path / "ScoutABadFacts.tla").write_text(export_scout_a_tla_bad_facts(trace))
    (formal_path / "ScoutAFacts.lean").write_text(export_scout_a_lean_facts(trace))
    (formal_path / "ScoutABadFacts.lean").write_text(
        export_scout_a_lean_bad_facts(trace)
    )


# Process documents are workflow trackers and review artifacts, not executable
# or checked-in source. They are recorded in the manifest for review, but their
# bytes are deliberately excluded from ``source_id``: a sealed attempt reports
# its own evidence identity, and that report is written into the tracker, so
# hashing the tracker into the identity it reports is circular. Excluding them
# also lets ordinary ticket status transitions happen without invalidating the
# bundle that justifies them.
_PROCESS_SOURCE_ROOTS: tuple[str, ...] = (".scratch", ".superpowers")
_PROCESS_SOURCE_PREDICATE_TEXT = "first path component is one of: " + ", ".join(
    _PROCESS_SOURCE_ROOTS
)
_PROCESS_INFORMATIONAL_NOTE = (
    "seal-time snapshot; process document bytes are not covered by source_id and "
    "are not re-verified; adding or removing a process path still changes "
    "status_sha256 and therefore source_id"
)
# A process path may never carry executable or formal source. This is an
# ALLOWLIST rather than a denylist: a denylist silently admits whatever it
# forgot -- extensionless files, unlisted interpreted languages, new formal
# backends -- and each of those would be real source sitting outside the
# verified set. Process documents are prose, so prose extensions are all that
# is permitted. Suffixes are matched case-insensitively; executable mode bits
# and symlinks are rejected separately.
_PROCESS_ALLOWED_SUFFIXES = (".md", ".txt")
# Generated checker products must never enter the source identity. TLC writes
# state and fingerprint files next to the module being checked unless a metadir
# is supplied, so an ad-hoc checker run inside the checkout silently leaves
# artifacts that a later seal would attest to as source. The spec keeps
# generated state spaces out of the checkout entirely, so this fails closed
# rather than ignoring them: ignoring them would let the pollution persist
# invisibly.
# TLC and Lean write these beside the checked module when no output directory
# is supplied. The rule is scoped to the formal directory rather than the whole
# tree, because matching bare directory names like "build" or "states" anywhere
# would reject legitimate source.
#
# Structural limit, stated rather than papered over: this guard only ever sees
# paths that appear in the captured Git status, so anything gitignored --
# `build/` among them -- cannot reach it. The guard stops generated products
# being sealed as verified SOURCE; it is not, and cannot be, a general
# assurance that the checkout is free of generated artifacts. Keeping the
# checker's output directory outside the checkout is what provides that, and
# the Bazel targets already do so via TEST_TMPDIR.
_GENERATED_CHECKER_ROOT = "experiments/qwen3_formal_verifier/formal"
_GENERATED_CHECKER_SUFFIXES = (".st", ".fp", ".bin", ".olean", ".ilean")
_GENERATED_CHECKER_DIRS = ("states", ".lake")
# TLC names error-trace artifacts <Spec>_TTrace_<timestamp>, so the marker is
# embedded rather than a prefix.
_GENERATED_CHECKER_MARKERS = ("TTrace_",)

_SOURCE_IDENTITY_KEYS = frozenset(
    {"schema", "head", "status_sha256", "entries", "source_id"}
)
_PROCESS_PRESENT_KEYS = frozenset(
    {"status", "path", "kind", "mode", "size_bytes", "sha256_at_seal", "verified"}
)
_PROCESS_DELETED_KEYS = frozenset({"status", "path", "kind", "verified"})
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def is_process_source_path(path_text: str) -> bool:
    """Return True when a repository-relative path is a process document.

    Matching is by first path component, never by string prefix, so
    ``.scratchpad/x`` and ``a/.scratch/b`` are not process paths.
    """

    parts = PurePosixPath(path_text).parts
    return bool(parts) and parts[0] in _PROCESS_SOURCE_ROOTS


def _reject_executable_process_path(
    path_text: str,
    *,
    mode: int | None = None,
    is_symlink: bool = False,
) -> None:
    """Refuse a process path that carries executable or formal source.

    ``mode`` is the file permission bits when known. Executable bits and
    symlinks are rejected because either could carry source that would then sit
    outside the verified set.
    """

    if PurePosixPath(path_text).suffix.lower() not in _PROCESS_ALLOWED_SUFFIXES:
        raise ValueError(
            "process source paths may only be prose documents "
            f"({', '.join(_PROCESS_ALLOWED_SUFFIXES)}), and may not carry "
            f"executable or formal source: {path_text}"
        )
    if is_symlink:
        raise ValueError(f"process source paths may not be symlinks: {path_text}")
    if mode is not None and mode & 0o111:
        raise ValueError(f"process source paths may not be executable: {path_text}")


def _reject_generated_checker_product(path_text: str) -> None:
    """Refuse a path that is a generated model-checker artifact.

    TLC writes ``.st``/``.fp`` state products beside the checked module when no
    metadir is given. Sealing those as verified source would attest to
    throwaway bytes and would contradict the spec's rule that generated state
    spaces stay outside the checkout.
    """

    candidate = PurePosixPath(path_text)
    if not path_text.startswith(f"{_GENERATED_CHECKER_ROOT}/"):
        return
    if (
        candidate.suffix.lower() in _GENERATED_CHECKER_SUFFIXES
        or any(part in _GENERATED_CHECKER_DIRS for part in candidate.parts[:-1])
        or any(marker in candidate.name for marker in _GENERATED_CHECKER_MARKERS)
    ):
        raise ValueError(
            "generated model-checker products may not be part of the source "
            f"identity; remove it from the checkout: {path_text}"
        )


def _source_identity_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    """Return only the fields that ``source_id`` covers.

    Raises ``ValueError`` rather than asserting: this runs on manifests loaded
    from disk, so a malformed manifest is invalid input, not a programmer error.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("source manifest must be a mapping")
    payload = {key: manifest[key] for key in _SOURCE_IDENTITY_KEYS if key in manifest}
    if set(payload) != _SOURCE_IDENTITY_KEYS:
        raise ValueError(
            "source manifest must carry exactly the identity key set, found "
            f"{sorted(payload)}"
        )
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise ValueError("source manifest entries must be a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("source manifest entries must be mappings")
        if is_process_source_path(str(entry.get("path", ""))):
            raise ValueError(
                "process paths must never reach the verified source entry list: "
                f"{entry.get('path')!r}"
            )
    return payload


def _validate_process_section(
    manifest: Mapping[str, object],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    """Validate a manifest's process section and return its entries.

    The section is informational, which makes it the one place a forged manifest
    could shrink lint coverage while still passing source verification. Every
    field is therefore checked, and a ``deleted`` record whose file is actually
    present is rejected -- that is exactly the forgery that would drop a live
    document from lint.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("source manifest must be a mapping")
    process = manifest.get("process_informational")
    if not isinstance(process, Mapping):
        raise ValueError("source manifest lacks a process_informational section")
    entries = process.get("entries")
    if not isinstance(entries, list):
        raise ValueError("process_informational entries must be a list")
    validated: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("process_informational entries must be mappings")
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text:
            raise ValueError("process_informational entries must carry a path")
        if not is_process_source_path(path_text):
            raise ValueError(
                f"process_informational path fails the process predicate: {path_text}"
            )
        if entry.get("verified") is not False:
            raise ValueError(
                f"process_informational entries must record verified=false: {path_text}"
            )
        kind = entry.get("kind")
        if kind == "deleted":
            expected_keys = _PROCESS_DELETED_KEYS
        elif kind in {"file", "symlink"}:
            expected_keys = _PROCESS_PRESENT_KEYS
        else:
            raise ValueError(
                f"process_informational entry has an unsupported kind: {path_text}"
            )
        if set(entry) != expected_keys:
            raise ValueError(
                "process_informational entry has an unexpected key set for "
                f"{path_text}: {sorted(entry)}"
            )
        if kind == "symlink":
            raise ValueError(f"process source paths may not be symlinks: {path_text}")
        if kind != "deleted":
            digest = entry.get("sha256_at_seal")
            if not isinstance(digest, str) or not _SHA256_HEX_RE.match(digest):
                raise ValueError(
                    "process_informational entries need a hex sha256_at_seal: "
                    f"{path_text}"
                )
            mode_text = entry.get("mode")
            if not isinstance(mode_text, str) or not re.fullmatch(
                r"[0-7]{4}", mode_text
            ):
                raise ValueError(
                    f"process_informational entry has an invalid mode: {path_text}"
                )
            _reject_executable_process_path(path_text, mode=int(mode_text, 8))
            if not isinstance(entry.get("size_bytes"), int):
                raise ValueError(
                    f"process_informational entry has an invalid size: {path_text}"
                )
        else:
            _reject_executable_process_path(path_text)
            if repo_root is not None and (repo_root / path_text).exists():
                raise ValueError(
                    "process_informational entry is recorded deleted but the file "
                    f"is present: {path_text}"
                )
        validated.append(entry)
    return validated


def _require_safe_lint_path(path_text: str) -> None:
    """Reject a manifest path that must never reach lint's argument list.

    ``lint-paths`` output is fed straight to ``git add`` and ``pre-commit
    --files``. A path escaping the checkout, or one that an option parser would
    read as a flag, is invalid input from disk regardless of what else in the
    manifest validated, so it is rejected here rather than at each consumer.
    """

    if "\0" in path_text or "\n" in path_text:
        raise ValueError(f"lint path contains a delimiter: {path_text!r}")
    if path_text.startswith("-"):
        raise ValueError(f"lint path would parse as an option: {path_text!r}")
    candidate = PurePosixPath(path_text)
    if candidate.is_absolute() or any(
        component in {"", ".", ".."} for component in candidate.parts
    ):
        raise ValueError(f"lint path escapes the checkout: {path_text!r}")


def source_manifest_lint_paths(
    manifest: Mapping[str, object],
    *,
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    """Return every present path the manifest covers, verified and process alike.

    Lint must keep covering process documents even though their bytes leave
    ``source_id``. Reading only ``entries`` would silently shrink lint coverage
    while the lint log still reported success, so the process section is fully
    validated here rather than trusted.

    ``repo_root`` enables the same deleted-but-present rejection that sealing
    performs. The remaining seal-time check, the process-entry roster
    comparison, needs the host-captured status bytes; the ``lint-paths`` command
    therefore takes them too, so lint-time and seal-time validation agree.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("source manifest must be a mapping")
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(
            f"source manifest schema must be {SOURCE_MANIFEST_SCHEMA}, "
            f"found {manifest.get('schema')!r}"
        )
    _source_identity_payload(manifest)
    process_entries = _validate_process_section(manifest, repo_root=repo_root)
    verified_entries = manifest["entries"]
    if not isinstance(verified_entries, list):
        raise ValueError("source manifest entries must be a list")
    paths: set[str] = set()
    for entry in (*verified_entries, *process_entries):
        if not isinstance(entry, dict):
            raise ValueError("source manifest entries must be mappings")
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text:
            raise ValueError("source manifest entries must carry a path")
        _require_safe_lint_path(path_text)
        if entry.get("kind") != "deleted":
            paths.add(path_text)
    return tuple(sorted(paths))


def build_source_manifest(
    repo_root: str | Path,
    head: str,
    status_bytes: bytes,
    *,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Hash the complete uncommitted Git status without using Git in Insula.

    Executable and checked-in source is hashed into ``source_id``. Process
    documents are recorded separately and informationally; see
    ``_PROCESS_SOURCE_ROOTS``.
    """

    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise ValueError(f"HEAD must be a 40-64 character lowercase hex ID: {head}")
    root = Path(repo_root).absolute()
    root_status = root.lstat()
    if not stat.S_ISDIR(root_status.st_mode):
        raise ValueError(f"source root is not a directory: {root}")
    records: list[dict[str, object]] = []
    process_records: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for encoded_entry in status_bytes.split(b"\0"):
        if not encoded_entry:
            continue
        if len(encoded_entry) < 4 or encoded_entry[2:3] != b" ":
            raise ValueError("source status is not Git porcelain v1 -z data")
        status_code = encoded_entry[:2].decode("ascii")
        if "R" in status_code or "C" in status_code:
            raise ValueError("renamed/copied source paths are unsupported in Scout A")
        path_text = encoded_entry[3:].decode("utf-8")
        source_path = Path(path_text)
        if source_path.is_absolute() or any(
            component in {"", ".", ".."} for component in source_path.parts
        ):
            raise ValueError(f"invalid source status path: {path_text!r}")
        if path_text in seen_paths:
            raise ValueError(f"duplicate source status path: {path_text}")
        _reject_generated_checker_product(path_text)
        seen_paths.add(path_text)
        absolute = root / source_path
        try:
            source_status = absolute.lstat()
        except FileNotFoundError:
            if "D" not in status_code:
                raise ValueError(
                    f"source status path is missing without deletion status: {path_text}"
                ) from None
            if is_process_source_path(path_text):
                _reject_executable_process_path(path_text)
                process_records.append(
                    {
                        "status": status_code,
                        "path": path_text,
                        "kind": "deleted",
                        "verified": False,
                    }
                )
            else:
                records.append(
                    {"status": status_code, "path": path_text, "kind": "deleted"}
                )
            continue
        mode = stat.S_IMODE(source_status.st_mode)
        if stat.S_ISLNK(source_status.st_mode):
            link_target = os.readlink(absolute)
            content = link_target.encode()
            kind = "symlink"
        elif stat.S_ISREG(source_status.st_mode):
            content = absolute.read_bytes()
            link_target = None
            kind = "file"
        else:
            raise ValueError(f"source status path is not a file: {path_text}")
        record: dict[str, object] = {
            "status": status_code,
            "path": path_text,
            "kind": kind,
            "mode": f"{mode:04o}",
            "size_bytes": len(content),
        }
        if is_process_source_path(path_text):
            # Rejects forbidden suffixes, executable mode bits, and symlinks, so
            # nothing that could carry source leaves the verified set.
            _reject_executable_process_path(
                path_text, mode=mode, is_symlink=kind == "symlink"
            )
            # ``sha256_at_seal`` is deliberately not named ``sha256``: no code
            # path and no reader may mistake it for a verified digest.
            record["sha256_at_seal"] = _sha256_bytes(content)
            record["verified"] = False
            process_records.append(record)
        else:
            if link_target is not None:
                record["target"] = link_target
            record["sha256"] = _sha256_bytes(content)
            records.append(record)
    records.sort(key=lambda record: str(record["path"]))
    process_records.sort(key=lambda record: str(record["path"]))
    assert len(records) + len(process_records) == len(
        seen_paths
    ), "verified and process records must partition the reported status paths"
    manifest: dict[str, object] = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "head": head,
        "status_sha256": _sha256_bytes(status_bytes),
        "entries": records,
        "source_id": "",
    }
    manifest["source_id"] = f"sha256:{_sha256_json(_source_identity_payload(manifest))}"
    manifest["process_informational"] = {
        "note": _PROCESS_INFORMATIONAL_NOTE,
        "predicate": _PROCESS_SOURCE_PREDICATE_TEXT,
        "entries": process_records,
    }
    if expected is not None:
        if expected.get("schema") != SOURCE_MANIFEST_SCHEMA:
            raise ValueError(
                f"source manifest schema must be {SOURCE_MANIFEST_SCHEMA}, "
                f"found {expected.get('schema')!r}"
            )
        if _source_identity_payload(manifest) != _source_identity_payload(expected):
            raise ValueError(
                "source manifest no longer matches the current source tree"
            )
        # Process bytes may drift, but the process entry LIST may not. Without
        # this a forged manifest could simply omit a process entry: the omitted
        # document would silently leave lint coverage while source verification
        # still passed, because process bytes are outside source_id. Compare
        # the recorded roster against the one freshly derived from the same
        # status bytes.
        expected_roster = {
            (str(entry.get("path")), str(entry.get("kind")))
            for entry in _validate_process_section(expected)
        }
        actual_roster = {
            (str(entry["path"]), str(entry["kind"])) for entry in process_records
        }
        if expected_roster != actual_roster:
            missing = sorted(path for path, _ in actual_roster - expected_roster)
            unexpected = sorted(path for path, _ in expected_roster - actual_roster)
            raise ValueError(
                "source manifest process roster does not match the status: "
                f"missing={missing} unexpected={unexpected}"
            )
    return manifest


def write_source_manifest(
    *,
    status_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path,
    head: str,
    attempt_dir: str | Path,
) -> Path:
    """Create one immutable source manifest from host-produced Git status."""

    _require_rootfs()
    attempt = Path(attempt_dir)
    status_file = Path(status_path)
    status_bytes = _require_regular_readonly(status_file, root=attempt)
    manifest = build_source_manifest(repo_root, head, status_bytes)
    output = Path(output_path)
    _write_immutable(output, _canonical_json_text(manifest), root=attempt)
    return output


def verify_source_manifest(
    manifest_path: str | Path,
    status_path: str | Path,
    *,
    repo_root: str | Path,
    expected_head: str,
    attempt_dir: str | Path,
) -> dict[str, object]:
    """Verify HEAD, dirty paths, modes, and bytes against the source manifest."""

    attempt = Path(attempt_dir)
    manifest_bytes = _require_regular_readonly(Path(manifest_path), root=attempt)
    status_bytes = _require_regular_readonly(Path(status_path), root=attempt)
    manifest = json.loads(manifest_bytes)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
    ):
        raise ValueError("source manifest has an unsupported schema")
    if manifest.get("head") != expected_head:
        raise ValueError(
            f"source manifest HEAD must be {expected_head}, found {manifest.get('head')!r}"
        )
    _validate_process_section(manifest, repo_root=Path(repo_root))
    return build_source_manifest(
        repo_root,
        expected_head,
        status_bytes,
        expected=manifest,
    )


class _ObservedDataloader:
    def __init__(self, wrapped: Iterable[Any], observer: "_RuntimeObserver") -> None:
        self._wrapped = wrapped
        self._observer = observer

    def __iter__(self) -> Iterator[Any]:
        for batch in self._wrapped:
            self._observer.observe_batch(batch)
            yield batch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _RuntimeObserver:
    def __init__(
        self,
        *,
        trainer: Trainer,
        run_id: str,
        attempt_id: str,
        profile: Mapping[str, object],
        config_digest: str,
        model_init_digest: str,
    ) -> None:
        self.trainer = trainer
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.profile = dict(profile)
        self.config_digest = config_digest
        self.model_init_digest = model_init_digest
        self.events: list[dict[str, object]] = []
        self.first_batch: dict[str, object] | None = None
        self._handles: list[Any] = []
        self._before_optimizer_digest: str | None = None
        self._batch_observed = False
        self._gradient_observed = False
        self._gradient_hook_count = 0
        self._gradient_name: str | None = None
        self._gradient_identity: dict[str, object] | None = None
        self._forward_count = 0
        self._backward_count = 0

    def install(self) -> None:
        if len(self.trainer.model_parts) != 1:
            raise ValueError("Scout A requires exactly one model part")
        model = self.trainer.model_parts[0]
        self._handles.extend(
            [
                model.register_forward_pre_hook(self._forward_pre_hook),
                model.register_forward_hook(self._forward_hook),
                model.register_full_backward_pre_hook(self._backward_pre_hook),
                model.register_full_backward_hook(self._backward_hook),
            ]
        )
        tracked_name, tracked_param = next(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self._handles.append(
            tracked_param.register_post_accumulate_grad_hook(
                lambda parameter: self._gradient_hook(tracked_name, parameter)
            )
        )
        if len(self.trainer.optimizers.optimizers) != 1:
            raise ValueError("Scout A requires exactly one built optimizer")
        optimizer = self.trainer.optimizers.optimizers[0]
        self._handles.append(optimizer.register_step_pre_hook(self._optimizer_pre_hook))
        self._handles.append(
            optimizer.register_step_post_hook(self._optimizer_post_hook)
        )

    def remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def emit(
        self,
        kind: str,
        phase: str,
        *,
        observation: Mapping[str, object],
    ) -> None:
        source_order = len(self.events) + 1
        raw_event_id = f"raw-r0-{source_order:06d}"
        predecessors = [] if not self.events else [str(self.events[-1]["raw_event_id"])]
        self.events.append(
            {
                "raw_event_id": raw_event_id,
                "source_order": source_order,
                "source_clock": {
                    "domain": "observer_source_order",
                    "value": source_order,
                },
                "kind": kind,
                "phase": phase,
                "step": 1,
                "causal_predecessors": predecessors,
                "observation": dict(observation),
            }
        )

    def observe_batch(self, batch: object) -> None:
        if self._batch_observed:
            return
        if (
            not isinstance(batch, (list, tuple))
            or len(batch) != 2
            or not isinstance(batch[0], dict)
            or not isinstance(batch[1], torch.Tensor)
        ):
            raise TypeError("Scout A dataloader yielded an unsupported batch")
        input_dict, labels = batch
        tensors = {
            f"input.{name}": value
            for name, value in input_dict.items()
            if isinstance(value, torch.Tensor)
        }
        tensors["labels"] = labels
        tensor_identities = {
            name: _tensor_identity(tensor) for name, tensor in sorted(tensors.items())
        }
        batch_digest = _sha256_json(tensor_identities)
        self.first_batch = {
            "batch_sha256": batch_digest,
            "tensors": tensor_identities,
        }
        self._batch_observed = True
        self.emit("batch.observed", "input", observation=self.first_batch)

    def _forward_pre_hook(self, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        del module
        self._forward_count += 1
        if self._forward_count != 1:
            raise RuntimeError("Scout A observed more than one core model forward")
        self.emit(
            "forward.started",
            "forward",
            observation={
                "hook": "Module.register_forward_pre_hook",
                "num_args": len(args),
            },
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        output: object,
    ) -> None:
        del module, args
        output_tensors = list(_walk_tensors(output))
        self.emit(
            "forward.completed",
            "forward",
            observation={
                "hook": "Module.register_forward_hook",
                "output_shapes": [list(tensor.shape) for tensor in output_tensors],
            },
        )

    def _backward_pre_hook(
        self,
        module: torch.nn.Module,
        grad_output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        del module
        self._backward_count += 1
        if self._backward_count != 1:
            raise RuntimeError("Scout A observed more than one core model backward")
        self.emit(
            "backward.started",
            "backward",
            observation={
                "hook": "Module.register_full_backward_pre_hook",
                "num_grad_outputs": (
                    1 if isinstance(grad_output, torch.Tensor) else len(grad_output)
                ),
            },
        )

    def _gradient_hook(self, name: str, parameter: torch.Tensor) -> None:
        if parameter.grad is None:
            raise RuntimeError(f"gradient hook for {name} ran without a gradient")
        self._gradient_hook_count += 1
        self._gradient_name = name
        self._gradient_identity = _tensor_identity(parameter.grad)

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: torch.Tensor | tuple[torch.Tensor, ...],
        grad_output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        del module
        if self._gradient_hook_count == 0 or self._gradient_identity is None:
            raise RuntimeError(
                "model backward completed before the tracked parameter gradient was ready"
            )
        if self._gradient_observed:
            raise RuntimeError("Scout A observed gradient readiness more than once")
        self._gradient_observed = True
        self.emit(
            "gradient.ready",
            "backward",
            observation={
                "hook": "Parameter.register_post_accumulate_grad_hook",
                "parameter": self._gradient_name,
                "hook_calls": self._gradient_hook_count,
                "gradient": self._gradient_identity,
            },
        )
        self.emit(
            "backward.completed",
            "backward",
            observation={
                "hook": "Module.register_full_backward_hook",
                "num_grad_inputs": len(grad_input),
                "num_grad_outputs": len(grad_output),
            },
        )

    def _optimizer_pre_hook(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs
        self._before_optimizer_digest = _optimizer_parameter_digest(optimizer)
        self.emit(
            "optimizer.started",
            "optimizer",
            observation={
                "hook": "Optimizer.register_step_pre_hook",
                "optimizer": type(optimizer).__name__,
                "parameter_sha256": self._before_optimizer_digest,
            },
        )

    def _optimizer_post_hook(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs
        if self._before_optimizer_digest is None:
            raise RuntimeError("optimizer post-hook ran before its pre-hook")
        after_digest = _optimizer_parameter_digest(optimizer)
        mutated = self._before_optimizer_digest != after_digest
        if not mutated:
            raise RuntimeError("AdamW step did not mutate observed parameters")
        self.emit(
            "optimizer.mutated",
            "optimizer",
            observation={
                "hook": "Optimizer.register_step_post_hook",
                "optimizer": type(optimizer).__name__,
                "before_sha256": self._before_optimizer_digest,
                "after_sha256": after_digest,
                "mutated": mutated,
            },
        )

    def raw_trace(self) -> dict[str, object]:
        if self.first_batch is None:
            raise RuntimeError("Scout A never observed its first batch")
        properties = torch.cuda.get_device_properties(self.trainer.device)
        return {
            "schema": RAW_SCHEMA,
            "identity": {
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "process_id": "rank-0",
                "rank": torch.distributed.get_rank(),
                "local_rank": int(os.environ["LOCAL_RANK"]),
                "world_size": torch.distributed.get_world_size(),
            },
            "device": {
                "kind": "cuda",
                "logical_id": str(self.trainer.device),
                "visible_index": self.trainer.device.index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            },
            "mesh": {
                "axes": [
                    "dp_replicate",
                    "dp_shard",
                    "cp",
                    "tp",
                    "pp",
                    "ep",
                ],
                "degrees": [1, 1, 1, 1, 1, 1],
                "coordinate": [0, 0, 0, 0, 0, 0],
            },
            "profile": dict(self.profile),
            "lineage": {
                "config_sha256": self.config_digest,
                "model_init_sha256": self.model_init_digest,
                "first_batch_sha256": self.first_batch["batch_sha256"],
            },
            "events": deepcopy(self.events),
        }


def run_scout_a(
    output_root: str | Path,
    *,
    run_id: str = SCOUT_A_RUN_ID,
    attempt_id: str = SCOUT_A_ATTEMPT_ID,
) -> ScoutAPaths:
    """Execute and observe one real optimizer step through the core Trainer."""

    _require_rootfs()
    _validate_identifier(run_id, "run_id")
    _validate_identifier(attempt_id, "attempt_id")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "0,"}:
        raise RuntimeError("Scout A requires CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Scout A requires exactly one visible CUDA device")
    if (
        int(os.environ.get("RANK", "-1")) != 0
        or int(os.environ.get("LOCAL_RANK", "-1")) != 0
        or int(os.environ.get("WORLD_SIZE", "-1")) != 1
    ):
        raise RuntimeError("Scout A must run under torchrun with one rank")

    output_root_path = Path(output_root)
    attempt_dir = output_root_path / run_id / attempt_id
    dump_folder = attempt_dir / "trainer"
    config, cli_args = build_scout_a_config(dump_folder)
    profile = _scout_a_profile(config)
    lineage_cli_args = list(cli_args)
    dump_folder_index = lineage_cli_args.index("--dump-folder") + 1
    lineage_cli_args[dump_folder_index] = "<attempt>/trainer"
    config_digest = _sha256_json({"cli": lineage_cli_args, "profile": profile})
    os.environ["TORCHTITAN_RUN_ID"] = run_id
    os.environ["TORCHTITAN_ATTEMPT_ID"] = attempt_id
    init_logger()

    trainer: Trainer | None = None
    observer: _RuntimeObserver | None = None
    try:
        trainer = config.build()
        if not isinstance(trainer, Trainer):
            raise TypeError("ConfigManager config did not build the core Trainer")
        if trainer.gradient_accumulation_steps != 1:
            raise ValueError("Scout A must build one gradient accumulation step")
        model_init_digest = _model_parameter_digest(trainer.model_parts)
        observer = _RuntimeObserver(
            trainer=trainer,
            run_id=run_id,
            attempt_id=attempt_id,
            profile=profile,
            config_digest=config_digest,
            model_init_digest=model_init_digest,
        )
        observer.install()
        trainer_dynamic: Any = trainer
        trainer_dynamic.dataloader = _ObservedDataloader(trainer.dataloader, observer)
        observer.emit(
            "step.started",
            "step",
            observation={
                "owner": f"{Trainer.__module__}.{Trainer.__qualname__}",
                "method": "Trainer.train",
                "config_manager": f"{ConfigManager.__module__}.{ConfigManager.__qualname__}",
            },
        )
        trainer.train()
        observer.emit(
            "step.completed",
            "step",
            observation={
                "owner": f"{Trainer.__module__}.{Trainer.__qualname__}",
                "trainer_step": trainer.step,
                "optimizer_steps": 1,
            },
        )
        raw_trace = observer.raw_trace()
        normalized = normalize_raw_trace(raw_trace)
    finally:
        if observer is not None:
            observer.remove()
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return _write_attempt_bundle(attempt_dir, raw_trace, normalized, cli_args)


def _write_attempt_bundle(
    attempt_dir: Path,
    raw_trace: Mapping[str, object],
    normalized: Mapping[str, object],
    cli_args: Sequence[str],
) -> ScoutAPaths:
    attempt_status = attempt_dir.lstat()
    if not stat.S_ISDIR(attempt_status.st_mode):
        raise ValueError(f"attempt directory must be a real directory: {attempt_dir}")
    source_manifest_value = os.environ.get("QFV_SCOUT_SOURCE_MANIFEST", "")
    expected_head = os.environ.get("QFV_SCOUT_EXPECTED_HEAD", "")
    source_root = os.environ.get("QFV_SCOUT_SOURCE_ROOT", "")
    if not source_manifest_value or not expected_head or not source_root:
        raise RuntimeError(
            "Scout A requires QFV_SCOUT_SOURCE_MANIFEST, "
            "QFV_SCOUT_EXPECTED_HEAD, and QFV_SCOUT_SOURCE_ROOT"
        )
    source_manifest_path = Path(source_manifest_value)
    expected_source_manifest = attempt_dir / "manifests" / "source.json"
    if source_manifest_path.absolute() != expected_source_manifest.absolute():
        raise ValueError(
            f"source manifest must be {expected_source_manifest}, "
            f"found {source_manifest_path}"
        )
    source_status_path = attempt_dir / "manifests" / "source-status.porcelain-v1-z"
    source = verify_source_manifest(
        source_manifest_path,
        source_status_path,
        repo_root=source_root,
        expected_head=expected_head,
        attempt_dir=attempt_dir,
    )
    raw_path = attempt_dir / "raw" / "observed.json"
    normalized_path = attempt_dir / "normalized" / "scout_a.json"
    generated_dir = attempt_dir / "generated"
    tla_path = generated_dir / "ScoutAFacts.tla"
    tla_bad_path = generated_dir / "ScoutABadFacts.tla"
    lean_path = generated_dir / "ScoutAFacts.lean"
    lean_bad_path = generated_dir / "ScoutABadFacts.lean"
    runtime_manifest = attempt_dir / "manifests" / "runtime.json"
    artifacts = {
        raw_path: _canonical_json_text(raw_trace),
        normalized_path: _canonical_json_text(normalized),
        tla_path: export_scout_a_tla_facts(normalized),
        tla_bad_path: export_scout_a_tla_bad_facts(normalized),
        lean_path: export_scout_a_lean_facts(normalized),
        lean_bad_path: export_scout_a_lean_bad_facts(normalized),
    }
    for path, contents in artifacts.items():
        _write_immutable(path, contents, root=attempt_dir)
    manifest_files = {
        str(path.relative_to(attempt_dir)): _evidence_file_record(
            path,
            root=attempt_dir,
        )
        for path in artifacts
    }
    for source_path in (source_status_path, source_manifest_path):
        manifest_files[
            str(source_path.relative_to(attempt_dir))
        ] = _evidence_file_record(source_path, root=attempt_dir)
    manifest: dict[str, object] = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "runtime_id": "",
        "run_id": normalized["run_id"],
        "attempt_id": normalized["attempt_id"],
        "trace_id": normalized["trace_id"],
        "source": {
            "head": source["head"],
            "source_id": source["source_id"],
            "manifest_path": str(source_manifest_path.relative_to(attempt_dir)),
            "status_path": str(source_status_path.relative_to(attempt_dir)),
        },
        "rootfs_sentinel": os.environ.get("TORCHTITAN_IN_ROOTFS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "trainer_class": f"{Trainer.__module__}.{Trainer.__qualname__}",
        "config_manager_class": f"{ConfigManager.__module__}.{ConfigManager.__qualname__}",
        "config_cli": list(cli_args),
        "files": manifest_files,
    }
    manifest["runtime_id"] = f"sha256:{_sha256_json(manifest)}"
    _write_immutable(
        runtime_manifest,
        _canonical_json_text(manifest),
        root=attempt_dir,
    )
    return ScoutAPaths(
        attempt_dir=attempt_dir,
        raw_trace=raw_path,
        normalized_trace=normalized_path,
        generated_tla_facts=tla_path,
        generated_tla_bad_facts=tla_bad_path,
        generated_lean_facts=lean_path,
        generated_lean_bad_facts=lean_bad_path,
        runtime_manifest=runtime_manifest,
    )


def _managed_evidence_path(attempt: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        component in {"", ".", ".."} for component in candidate.parts
    ):
        raise ValueError(f"invalid evidence-relative path: {relative!r}")
    return attempt / candidate


def _evidence_file_record(path: Path, *, root: Path) -> dict[str, object]:
    data = _require_regular_readonly(path, root=root)
    return {"sha256": _sha256_bytes(data), "size_bytes": len(data)}


def _verify_file_record(
    attempt: Path,
    relative: str,
    record: Mapping[str, object],
) -> bytes:
    path = _managed_evidence_path(attempt, relative)
    data = _require_regular_readonly(path, root=attempt)
    expected_size = record.get("size_bytes")
    expected_digest = record.get("sha256")
    if expected_size != len(data) or expected_digest != _sha256_bytes(data):
        raise ValueError(f"evidence digest/size mismatch: {relative}")
    return data


def _bundle_regular_files(attempt: Path) -> set[str]:
    """Return every file while rejecting symlinks and special entries."""

    attempt_status = attempt.lstat()
    if not stat.S_ISDIR(attempt_status.st_mode):
        raise ValueError(f"evidence bundle root is not a real directory: {attempt}")
    relative_files: set[str] = set()
    for current, directory_names, file_names in os.walk(attempt, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            directory = current_path / name
            directory_status = directory.lstat()
            if stat.S_ISLNK(directory_status.st_mode):
                raise ValueError(f"evidence bundle directory is a symlink: {directory}")
            if not stat.S_ISDIR(directory_status.st_mode):
                raise ValueError(
                    f"evidence bundle directory entry is not a directory: {directory}"
                )
        for name in file_names:
            path = current_path / name
            path_status = path.lstat()
            if stat.S_ISLNK(path_status.st_mode):
                raise ValueError(f"evidence bundle file is a symlink: {path}")
            if not stat.S_ISREG(path_status.st_mode):
                raise ValueError(f"evidence bundle entry is not a file: {path}")
            relative_files.add(str(path.relative_to(attempt)))
    return relative_files


def _verify_content_id(
    manifest: Mapping[str, object],
    field: str,
    label: str,
) -> None:
    candidate = deepcopy(dict(manifest))
    candidate[field] = ""
    expected = f"sha256:{_sha256_json(candidate)}"
    if manifest.get(field) != expected:
        raise ValueError(
            f"{label} {field} must be {expected}, found {manifest.get(field)!r}"
        )


def verify_runtime_bundle(
    attempt_dir: str | Path,
    *,
    expected_head: str,
    source_root: str | Path = ".",
) -> dict[str, object]:
    """Verify every runtime artifact, source byte, trace, and provenance link."""

    _require_rootfs()
    attempt = Path(attempt_dir)
    runtime_path = attempt / "manifests" / "runtime.json"
    runtime_bytes = _require_regular_readonly(runtime_path, root=attempt)
    runtime = json.loads(runtime_bytes)
    if (
        not isinstance(runtime, dict)
        or runtime.get("schema") != RUNTIME_MANIFEST_SCHEMA
    ):
        raise ValueError("runtime manifest has an unsupported schema")
    _verify_content_id(runtime, "runtime_id", "runtime manifest")
    files = _mapping(runtime, "files")
    expected_files = {
        "raw/observed.json",
        "normalized/scout_a.json",
        "generated/ScoutAFacts.tla",
        "generated/ScoutABadFacts.tla",
        "generated/ScoutAFacts.lean",
        "generated/ScoutABadFacts.lean",
        "manifests/source-status.porcelain-v1-z",
        "manifests/source.json",
    }
    if set(files) != expected_files:
        raise ValueError(
            f"runtime manifest files must be {sorted(expected_files)}, "
            f"found {sorted(files)}"
        )
    verified_bytes = {
        relative: _verify_file_record(
            attempt,
            relative,
            _mapping(files, relative),
        )
        for relative in sorted(expected_files)
    }
    source = _mapping(runtime, "source")
    if source.get("head") != expected_head:
        raise ValueError("runtime manifest HEAD disagrees with expected HEAD")
    source_manifest = verify_source_manifest(
        attempt / str(source.get("manifest_path", "")),
        attempt / str(source.get("status_path", "")),
        repo_root=source_root,
        expected_head=expected_head,
        attempt_dir=attempt,
    )
    if source.get("source_id") != source_manifest.get("source_id"):
        raise ValueError("runtime source identity disagrees with source manifest")

    raw = json.loads(verified_bytes["raw/observed.json"])
    normalized = json.loads(verified_bytes["normalized/scout_a.json"])
    if not isinstance(raw, dict) or not isinstance(normalized, dict):
        raise ValueError("runtime raw and normalized evidence must be objects")
    recomputed = normalize_raw_trace(raw)
    if normalized != recomputed:
        raise ValueError(
            "normalized trace or raw-event provenance does not match raw evidence"
        )
    validate_normalized_trace(normalized)
    for identity_key in ("run_id", "attempt_id", "trace_id"):
        if runtime.get(identity_key) != normalized.get(identity_key):
            raise ValueError(
                f"runtime manifest {identity_key} disagrees with normalized trace"
            )
    expected_generated = {
        "generated/ScoutAFacts.tla": export_scout_a_tla_facts(normalized).encode(),
        "generated/ScoutABadFacts.tla": export_scout_a_tla_bad_facts(
            normalized
        ).encode(),
        "generated/ScoutAFacts.lean": export_scout_a_lean_facts(normalized).encode(),
        "generated/ScoutABadFacts.lean": export_scout_a_lean_bad_facts(
            normalized
        ).encode(),
    }
    for relative, expected in expected_generated.items():
        if verified_bytes[relative] != expected:
            raise ValueError(f"generated runtime fact is stale: {relative}")
    return runtime


def _require_non_empty_stage_log(
    attempt: Path,
    evidence_path: str,
    *,
    stage_name: str,
) -> bytes:
    """Require a declared stage log to exist and to have captured something.

    This belongs at the sealing boundary, not only in the shell runner. The
    runner's ``-s`` guard is outside the sealed contract, so sealing itself
    accepted a zero-byte stage log: every mode, symlink and regular-file check
    passed while the log proved that no check had run. A missing or empty log
    now fails the seal instead of being sealed as evidence of nothing.

    Whitespace does not count. A stage that emits one blank line before dying
    would otherwise satisfy a byte-count check, which is the same hole one
    level up. This still proves only that the stage wrote something, not that
    it checked anything; the stage's own result marker proves that.
    """

    data = _require_regular_readonly(attempt / evidence_path, root=attempt)
    if not data.strip():
        raise ValueError(
            f"stage {stage_name} log is empty and proves nothing: {evidence_path}"
        )
    return data


def _read_stage_journal(attempt: Path) -> list[dict[str, object]]:
    journal_path = attempt / "checker" / "stages.tsv"
    journal_bytes = _require_regular_readonly(journal_path, root=attempt)
    try:
        journal_text = journal_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("stage journal must be UTF-8") from None
    lines = journal_text.splitlines()
    if len(lines) != len(SCOUT_A_REQUIRED_STAGES):
        raise ValueError("stage journal does not cover every required stage")
    stages = []
    for line, (required_name, required_log) in zip(
        lines,
        SCOUT_A_REQUIRED_STAGES,
        strict=True,
    ):
        fields = line.split("\t", 3)
        if len(fields) != 4:
            raise ValueError("stage journal row must have four tab-separated fields")
        name, status_text, evidence_path, command = fields
        try:
            exit_status = int(status_text)
        except ValueError:
            raise ValueError(f"stage {name} has a non-integer exit status") from None
        if name != required_name or evidence_path != required_log:
            raise ValueError(
                f"stage journal expected {required_name}/{required_log}, "
                f"found {name}/{evidence_path}"
            )
        if exit_status != 0:
            raise ValueError(f"required stage {name} did not exit successfully")
        if not command or "\n" in command or "\t" in command:
            raise ValueError(f"stage {name} has an invalid command record")
        _require_non_empty_stage_log(attempt, evidence_path, stage_name=name)
        stages.append(
            {
                "name": name,
                "command": command,
                "exit_status": exit_status,
                "evidence_path": evidence_path,
            }
        )
    return stages


def _stage_manifest(
    runtime: Mapping[str, object],
    stages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema": STAGE_MANIFEST_SCHEMA,
        "stage_id": "",
        "run_id": runtime["run_id"],
        "attempt_id": runtime["attempt_id"],
        "trace_id": runtime["trace_id"],
        "source": deepcopy(_mapping(runtime, "source")),
        "stages": deepcopy(list(stages)),
    }
    manifest["stage_id"] = f"sha256:{_sha256_json(manifest)}"
    return manifest


def finalize_evidence_manifest(
    attempt_dir: str | Path,
    *,
    expected_head: str,
    source_root: str | Path = ".",
) -> Path:
    """Verify runtime/stage evidence and seal the complete immutable bundle."""

    _require_rootfs()
    attempt = Path(attempt_dir)
    runtime = verify_runtime_bundle(
        attempt,
        expected_head=expected_head,
        source_root=source_root,
    )
    stages = _read_stage_journal(attempt)
    stage_manifest = _stage_manifest(runtime, stages)
    stage_path = attempt / "manifests" / "stages.json"
    _write_immutable(stage_path, _canonical_json_text(stage_manifest), root=attempt)

    runtime_files = _mapping(runtime, "files")
    relative_files = set(runtime_files)
    relative_files.update(
        {
            "manifests/runtime.json",
            "manifests/stages.json",
            "checker/stages.tsv",
            *(relative for _, relative in SCOUT_A_REQUIRED_STAGES),
        }
    )
    files = {
        relative: _evidence_file_record(
            _managed_evidence_path(attempt, relative),
            root=attempt,
        )
        for relative in sorted(relative_files)
    }
    source = _mapping(runtime, "source")
    manifest: dict[str, object] = {
        "schema": EVIDENCE_MANIFEST_SCHEMA,
        "evidence_id": "",
        "run_id": runtime["run_id"],
        "attempt_id": runtime["attempt_id"],
        "trace_id": runtime["trace_id"],
        "runtime_id": runtime["runtime_id"],
        "stage_id": stage_manifest["stage_id"],
        "source": deepcopy(source),
        "files": files,
    }
    manifest["evidence_id"] = f"sha256:{_sha256_json(manifest)}"
    output = attempt / "manifests" / "evidence.json"
    _write_immutable(output, _canonical_json_text(manifest), root=attempt)
    verify_evidence_bundle(
        attempt,
        expected_head=expected_head,
        source_root=source_root,
    )
    return output


def verify_evidence_bundle(
    attempt_dir: str | Path,
    *,
    expected_head: str,
    source_root: str | Path = ".",
) -> dict[str, object]:
    """Verify the complete sealed Scout A bundle against current source bytes."""

    _require_rootfs()
    attempt = Path(attempt_dir)
    evidence_path = attempt / "manifests" / "evidence.json"
    evidence_bytes = _require_regular_readonly(evidence_path, root=attempt)
    evidence = json.loads(evidence_bytes)
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != EVIDENCE_MANIFEST_SCHEMA
    ):
        raise ValueError("evidence manifest has an unsupported schema")
    _verify_content_id(evidence, "evidence_id", "evidence manifest")
    runtime = verify_runtime_bundle(
        attempt,
        expected_head=expected_head,
        source_root=source_root,
    )
    stage_path = attempt / "manifests" / "stages.json"
    stage_bytes = _require_regular_readonly(stage_path, root=attempt)
    stages = json.loads(stage_bytes)
    if not isinstance(stages, dict) or stages.get("schema") != STAGE_MANIFEST_SCHEMA:
        raise ValueError("stage manifest has an unsupported schema")
    _verify_content_id(stages, "stage_id", "stage manifest")
    expected_stages = _stage_manifest(runtime, _read_stage_journal(attempt))
    if stages != expected_stages:
        raise ValueError("stage manifest disagrees with the immutable stage journal")
    for key in ("run_id", "attempt_id", "trace_id"):
        if evidence.get(key) != runtime.get(key):
            raise ValueError(f"evidence manifest {key} disagrees with runtime")
    if evidence.get("runtime_id") != runtime.get("runtime_id"):
        raise ValueError("evidence manifest runtime_id disagrees with runtime")
    if evidence.get("stage_id") != stages.get("stage_id"):
        raise ValueError("evidence manifest stage_id disagrees with stages")
    if evidence.get("source") != runtime.get("source"):
        raise ValueError("evidence manifest source identity disagrees with runtime")
    files = _mapping(evidence, "files")
    expected_files = set(_mapping(runtime, "files"))
    expected_files.update(
        {
            "manifests/runtime.json",
            "manifests/stages.json",
            "checker/stages.tsv",
            *(relative for _, relative in SCOUT_A_REQUIRED_STAGES),
        }
    )
    if set(files) != expected_files:
        raise ValueError("evidence manifest does not cover the complete managed bundle")
    for relative in sorted(expected_files):
        _verify_file_record(attempt, relative, _mapping(files, relative))
    complete_files = expected_files | {"manifests/evidence.json"}
    discovered_files = _bundle_regular_files(attempt)
    if discovered_files != complete_files:
        raise ValueError(
            "evidence bundle contains unmanaged or missing files: "
            f"expected {sorted(complete_files)}, found {sorted(discovered_files)}"
        )
    return evidence


def _model_parameter_digest(model_parts: Sequence[torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for part_index, model in enumerate(model_parts):
        for name, parameter in sorted(model.named_parameters()):
            _update_tensor_digest(digest, f"part{part_index}.{name}", parameter)
    return digest.hexdigest()


def _optimizer_parameter_digest(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        names = group.get("param_names", [])
        for parameter_index, parameter in enumerate(group["params"]):
            name = (
                str(names[parameter_index])
                if parameter_index < len(names)
                else f"parameter-{parameter_index}"
            )
            _update_tensor_digest(
                digest,
                f"group{group_index}.{name}",
                parameter,
            )
    return digest.hexdigest()


def _update_tensor_digest(
    digest: Any,
    name: str,
    tensor: torch.Tensor,
) -> None:
    local = tensor.detach()
    if hasattr(local, "to_local"):
        local = local.to_local()
    materialized = local.float().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(tuple(materialized.shape)).encode())
    digest.update(materialized.numpy().tobytes())


def _tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    local = tensor.detach()
    if hasattr(local, "to_local"):
        local = local.to_local()
    materialized = local.cpu().contiguous()
    return {
        "dtype": str(materialized.dtype).removeprefix("torch."),
        "shape": list(materialized.shape),
        "sha256": _sha256_bytes(materialized.view(torch.uint8).numpy().tobytes()),
    }


def _walk_tensors(value: object) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _walk_tensors(value[key])


def _secure_managed_path(
    root: Path,
    path: Path,
    *,
    create_parents: bool = False,
) -> tuple[Path, Path]:
    """Resolve a managed evidence path, rejecting escapes and symlinks.

    ``create_parents`` is a sealing concern only. Verification must be able to
    run against a read-only copy of a sealed bundle -- exactly the situation a
    third party verifying the evidence is in -- so a missing parent is an
    error there rather than something to create.
    """

    root_absolute = root.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError:
        raise ValueError(
            f"managed evidence path escapes attempt root: {path_absolute}"
        ) from None
    root_status = root_absolute.lstat()
    if not stat.S_ISDIR(root_status.st_mode):
        kind = "symlink" if stat.S_ISLNK(root_status.st_mode) else "non-directory"
        raise ValueError(f"attempt root is a {kind}: {root_absolute}")
    current = root_absolute
    for component in relative.parts[:-1]:
        if component in {"", ".", ".."}:
            raise ValueError(f"invalid managed evidence path: {path_absolute}")
        current = current / component
        try:
            current_status = current.lstat()
        except FileNotFoundError:
            if not create_parents:
                raise FileNotFoundError(
                    f"managed evidence parent is missing: {current}"
                ) from None
            current.mkdir()
            current_status = current.lstat()
        if stat.S_ISLNK(current_status.st_mode):
            raise ValueError(f"managed evidence directory is a symlink: {current}")
        if not stat.S_ISDIR(current_status.st_mode):
            raise ValueError(f"managed evidence parent is not a directory: {current}")
    return root_absolute, path_absolute


def _require_regular_readonly(path: Path, *, root: Path) -> bytes:
    _, managed = _secure_managed_path(root, path)
    try:
        path_status = managed.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"required evidence file is missing: {managed}"
        ) from None
    if stat.S_ISLNK(path_status.st_mode):
        raise ValueError(f"evidence file must not be a symlink: {managed}")
    if not stat.S_ISREG(path_status.st_mode):
        raise ValueError(f"evidence path is not a regular file: {managed}")
    mode = stat.S_IMODE(path_status.st_mode)
    if mode != 0o444:
        raise PermissionError(
            f"immutable evidence must have read-only mode 0444, found {mode:04o}: {managed}"
        )
    return managed.read_bytes()


def _write_immutable(
    path: Path,
    contents: str | bytes,
    *,
    root: Path,
) -> None:
    _, managed = _secure_managed_path(root, path, create_parents=True)
    expected = contents.encode() if isinstance(contents, str) else contents
    try:
        existing_status = managed.lstat()
    except FileNotFoundError:
        existing_status = None
    if existing_status is not None:
        if stat.S_ISLNK(existing_status.st_mode):
            raise ValueError(f"immutable evidence must not be a symlink: {managed}")
        if not stat.S_ISREG(existing_status.st_mode):
            raise ValueError(f"immutable evidence is not a regular file: {managed}")
        actual = _require_regular_readonly(managed, root=root)
        if actual != expected:
            raise FileExistsError(
                f"immutable evidence already exists with different bytes: {managed}"
            )
        return

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(managed, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(expected)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    managed.chmod(0o444)
    _require_regular_readonly(managed, root=root)


def _require_rootfs() -> None:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError("Scout A Python must run with TORCHTITAN_IN_ROOTFS=1")


def _mapping(data: Mapping[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def _mapping_list(data: Mapping[str, object], key: str) -> list[dict[str, object]]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{key} must be a list of objects")
    return value


def _string_list(data: Mapping[str, object], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return value


def _integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _validate_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern}, found {value!r}"
        )


def _canonical_json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _sha256_json(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_event_id(
    run_id: str,
    attempt_id: str,
    rank: int,
    source_order: int,
) -> str:
    return f"{run_id}:{attempt_id}:r{rank}:e{source_order:06d}"


def _expected_trace_id(trace: Mapping[str, object]) -> str:
    content = deepcopy(dict(trace))
    content["trace_id"] = ""
    return f"sha256:{_sha256_json(content)}"


def _tla_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _tla_function(pairs: Sequence[tuple[str, str]]) -> str:
    if not pairs:
        return r"[x \in {} |-> x]"
    return " @@ ".join(f"({key} :> {value})" for key, value in pairs)


def _lean_string(value: str) -> str:
    return json.dumps(value)


def _default_formal_dir() -> Path:
    return Path("experiments/qwen3_formal_verifier/formal")


def _default_fixture() -> Path:
    return Path("experiments/qwen3_formal_verifier/fixtures/scout_a.normalized.json")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute the real Scout A step")
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--run-id", default=SCOUT_A_RUN_ID)
    run_parser.add_argument("--attempt-id", default=SCOUT_A_ATTEMPT_ID)

    sync_parser = subparsers.add_parser(
        "sync", help="check or write compact checked-in generated artifacts"
    )
    sync_parser.add_argument("--normalized", required=True)
    sync_parser.add_argument("--formal-dir", default=str(_default_formal_dir()))
    sync_parser.add_argument("--fixture", default=str(_default_fixture()))
    mode = sync_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")

    source_parser = subparsers.add_parser(
        "source", help="create the immutable source identity manifest"
    )
    source_parser.add_argument("--status-file", required=True)
    source_parser.add_argument("--output", required=True)
    source_parser.add_argument("--repo-root", required=True)
    source_parser.add_argument("--head", required=True)
    source_parser.add_argument("--attempt-dir", required=True)

    finalize_parser = subparsers.add_parser(
        "finalize", help="seal checker logs into the evidence manifest"
    )
    finalize_parser.add_argument("--attempt-dir", required=True)
    finalize_parser.add_argument("--expected-head", required=True)
    finalize_parser.add_argument("--source-root", default=".")

    verify_parser = subparsers.add_parser(
        "verify", help="verify the complete sealed evidence bundle"
    )
    verify_parser.add_argument("--attempt-dir", required=True)
    verify_parser.add_argument("--expected-head", required=True)
    verify_parser.add_argument("--source-root", default=".")

    lint_paths_parser = subparsers.add_parser(
        "lint-paths",
        help="print every covered path, verified and process alike, NUL separated",
    )
    lint_paths_parser.add_argument("--manifest", required=True)
    # Required, so lint can never silently fall back to shape-only validation:
    # with these the lint stage runs the same verification the seal does.
    lint_paths_parser.add_argument("--repo-root", required=True)
    lint_paths_parser.add_argument("--status-file", required=True)
    lint_paths_parser.add_argument("--head", required=True)
    lint_paths_parser.add_argument("--attempt-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "run":
        paths = run_scout_a(
            args.output_root,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
        )
        print(f"SCOUT_A_RUNTIME result=success attempt_dir={paths.attempt_dir}")
        print(f"SCOUT_A_RAW path={paths.raw_trace}")
        print(f"SCOUT_A_NORMALIZED path={paths.normalized_trace}")
        print(f"SCOUT_A_RUNTIME_MANIFEST path={paths.runtime_manifest}")
        return
    if args.command == "sync":
        _require_rootfs()
        trace = json.loads(Path(args.normalized).read_text())
        if args.write:
            sync_checked_in_artifacts(trace, args.formal_dir, args.fixture)
            print("SCOUT_A_ARTIFACT_SYNC result=written")
            return
        mismatches = artifact_sync_mismatches(trace, args.formal_dir, args.fixture)
        if mismatches:
            raise SystemExit(
                "Scout A generated artifacts are stale: " + ", ".join(mismatches)
            )
        print("SCOUT_A_ARTIFACT_SYNC result=clean")
        return
    if args.command == "source":
        manifest = write_source_manifest(
            status_path=args.status_file,
            output_path=args.output,
            repo_root=args.repo_root,
            head=args.head,
            attempt_dir=args.attempt_dir,
        )
        print(f"SCOUT_A_SOURCE_MANIFEST result=created path={manifest}")
        return
    if args.command == "finalize":
        manifest = finalize_evidence_manifest(
            args.attempt_dir,
            expected_head=args.expected_head,
            source_root=args.source_root,
        )
        print(f"SCOUT_A_EVIDENCE_MANIFEST result=sealed path={manifest}")
        return
    if args.command == "lint-paths":
        # Verify before deriving paths: the roster comparison inside
        # verify_source_manifest is the only check that can notice an absent
        # process entry, and shape validation alone cannot.
        manifest = verify_source_manifest(
            args.manifest,
            args.status_file,
            repo_root=args.repo_root,
            expected_head=args.head,
            attempt_dir=args.attempt_dir,
        )
        for path_text in source_manifest_lint_paths(
            manifest, repo_root=Path(args.repo_root)
        ):
            sys.stdout.buffer.write(path_text.encode() + b"\0")
        sys.stdout.buffer.flush()
        return
    assert args.command == "verify", f"unhandled command: {args.command}"
    evidence = verify_evidence_bundle(
        args.attempt_dir,
        expected_head=args.expected_head,
        source_root=args.source_root,
    )
    print(
        "SCOUT_A_EVIDENCE_VERIFY result=success "
        f"evidence_id={evidence['evidence_id']}"
    )


if __name__ == "__main__":
    main()
