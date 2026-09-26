# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observed four-rank DPxTP Qwen3 tracer bullet for the formal verifier.

Scout B keeps the core :class:`torchtitan.trainer.Trainer` unchanged. The
experiment owns its configuration guard, runtime observation, rank-bundle
merge, provisional normalization, and formal fact export.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from torchtitan.config import ConfigManager
from torchtitan.experiments.qwen3_formal_verifier.scout_a import (
    _bundle_regular_files,
    _canonical_json_text,
    _evidence_file_record,
    _model_parameter_digest,
    _optimizer_parameter_digest,
    _require_non_empty_stage_log,
    _require_regular_readonly,
    _require_rootfs,
    _secure_managed_path,
    _sha256_json,
    _tensor_identity,
    _verify_content_id,
    _verify_file_record,
    _walk_tensors,
    _write_immutable,
    source_manifest_lint_paths,
    verify_source_manifest,
    write_source_manifest,
)
from torchtitan.tools.logging import init_logger
from torchtitan.trainer import Trainer


SCOUT_B_RUN_ID = "qfv-scout-b-seed42"
SCOUT_B_ATTEMPT_ID = "four-rank-dp2-tp2-v1"
RAW_SCHEMA = "qwen3.formal.raw.v0"
SCOUT_SCHEMA = "qwen3.formal.scout.v0"
RUNTIME_MANIFEST_SCHEMA = "qwen3.formal.scout-b.runtime-manifest.v0"
STAGE_MANIFEST_SCHEMA = "qwen3.formal.scout-b.stage-manifest.v0"
EVIDENCE_MANIFEST_SCHEMA = "qwen3.formal.scout-b.evidence-manifest.v0"
RAW_EVENT_PROJECTION_SCHEMA = "qwen3.formal.raw-event-projection.v0"
# The contract carries its own version because it is compared by exact
# equality. Without it, adding a field makes every previously sealed bundle
# fail with a message a third party cannot tell apart from tampering.
PROVENANCE_CONTRACT_VERSION = "qwen3.formal.scout.provenance-contract.v1"
PROVENANCE_CONTRACT = {
    "version": PROVENANCE_CONTRACT_VERSION,
    "normalized_projection": {
        "schema": RAW_EVENT_PROJECTION_SCHEMA,
        "digest": "sha256",
        "excluded_paths": [
            "observed_time_ns",
            "observation.rank_raw_sha256 for bundle.collected",
            "observation.executor.thread_id for collective.*",
            "observation.producer.correlation_id for collective.*",
        ],
    },
    "full_raw_bytes": {
        "binding": "runtime_manifest.files",
        "digest": "sha256",
    },
    # Producer and stream attribution is INFERRED, not observed, and the
    # exported names -- CollectiveProducers, CollectiveStream,
    # producer.correlation_id -- read exactly like an observed join key. There
    # is no join key here: _collective_observations sorts one operation
    # family's Kineto kernels by start time, sorts that family's Flight
    # Recorder entries by record_id, and joins the two lists by position with
    # zip(entries, kernels, strict=True). Equal counts and equal order are the
    # whole argument. This block exists so the claim travels inside the sealed
    # bundle rather than living only in source comments and a design note.
    # Ticket 09's Kineto bridge replaces position with a joined key.
    "inferred_attribution": {
        "inferred": True,
        "observed": False,
        "method": "positional_zip_flight_entries_to_kineto_kernels",
        "join_key": None,
        "entry_order": "flight_recorder_record_id",
        "kernel_order": "kineto_kernel_start_ns",
        "fields": [
            "observation.producer.correlation_id for collective.*",
            "observation.producer.linked_correlation_id for collective.*",
            "observation.producer.external_id for collective.*",
            "observation.stream.device_index for collective.*",
            "observation.stream.resource_id for collective.*",
            "observation.stream.kernel for collective.*",
        ],
        "resource_id_meaning": (
            "opaque Kineto resource label, not a verified CUDA stream identity"
        ),
        "note": (
            "producer and stream attribution is INFERRED by position, not "
            "observed; ticket 09 replaces the positional zip with a joined key"
        ),
    },
}
SCOUT_B_REQUIRED_STAGES = (
    ("source_manifest", "checker/source-manifest.log"),
    ("focused_pytest", "checker/focused-pytest.log"),
    ("cuda_pytest", "checker/cuda-pytest.log"),
    ("owning_pytest", "checker/owning-pytest.log"),
    ("artifact_sync", "checker/artifact-sync.log"),
    ("scout_a_regression", "checker/scout-a-regression.log"),
    ("formal_networked", "checker/formal-networked.log"),
    ("formal_no_fetch", "checker/formal-no-fetch.log"),
    ("lint", "checker/lint.log"),
)
MESH_AXES = ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep")
MESH_DEGREES = (1, 2, 1, 2, 1, 1)
_CORE_LIFECYCLE = (
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
    "bundle.collected",
)
_COLLECTIVE_LIFECYCLE = ("enqueued", "started", "completed")
_FORMAL_EVENT_KIND_CODES = {
    kind: code
    for code, kind in enumerate(
        (*_CORE_LIFECYCLE, *(f"collective.{state}" for state in _COLLECTIVE_LIFECYCLE))
    )
}
_FORMAL_EVENT_KIND_BASE = 16
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ScoutBPaths:
    attempt_dir: Path
    rank_raw_traces: tuple[Path, Path, Path, Path]
    normalized_bundle: Path
    generated_tla_facts: Path
    generated_tla_bad_facts: Path
    generated_lean_facts: Path
    generated_lean_bad_facts: Path
    runtime_manifest: Path


def scout_b_cli_args(dump_folder: str | Path) -> list[str]:
    """Return the complete ConfigManager CLI profile for Scout B."""

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
        "2",
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
        "2",
        "--parallelism.tensor-parallel-degree",
        "2",
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


def build_scout_b_config(
    dump_folder: str | Path,
) -> tuple[Trainer.Config, list[str]]:
    """Select qwen3_debugmodel and apply Scout B via normal CLI precedence."""

    cli_args = scout_b_cli_args(dump_folder)
    config = ConfigManager().parse_args(cli_args)
    if not isinstance(config, Trainer.Config):
        raise TypeError("qwen3_debugmodel did not produce Trainer.Config")
    _scout_b_profile(config)
    return config, cli_args


def _scout_b_profile(config: Trainer.Config) -> dict[str, object]:
    model_spec = config.model_spec
    if model_spec is None:
        raise ValueError("Scout B requires a model spec")
    parallelism = config.parallelism
    observed = {
        "model": (model_spec.name, model_spec.flavor),
        "batch": (
            config.training.local_batch_size,
            config.training.global_batch_size,
        ),
        "sequence_and_steps": (config.training.seq_len, config.training.steps),
        "dtype": config.training.dtype,
        "degrees": {
            "data_parallel_replicate": parallelism.data_parallel_replicate_degree,
            "data_parallel_shard": parallelism.data_parallel_shard_degree,
            "context_parallel": parallelism.context_parallel_degree,
            "tensor_parallel": parallelism.tensor_parallel_degree,
            "pipeline_parallel": parallelism.pipeline_parallel_degree,
            "expert_parallel": parallelism.expert_parallel_degree,
        },
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
        "batch": (1, 2),
        "sequence_and_steps": (128, 1),
        "dtype": "bfloat16",
        "degrees": {
            "data_parallel_replicate": 1,
            "data_parallel_shard": 2,
            "context_parallel": 1,
            "tensor_parallel": 2,
            "pipeline_parallel": 1,
            "expert_parallel": 1,
        },
        "sequence_parallel": False,
        "seed_and_determinism": (42, True, False),
        "dataset": "c4_test",
        "tokenizer": "./tests/assets/tokenizer",
        "checkpoint": (False, None, False, False),
        "compile": False,
        "activation_checkpoint": None,
    }
    if observed != required:
        raise ValueError(f"Scout B config does not match its fixed profile: {observed}")
    optimizer_names = [group.optimizer_name for group in config.optimizer.param_groups]
    if optimizer_names != ["AdamW"]:
        raise ValueError(
            f"Scout B requires one AdamW optimizer, found {optimizer_names}"
        )
    return {
        "config_module": "qwen3",
        "config_name": "qwen3_debugmodel",
        "model_name": model_spec.name,
        "model_flavor": model_spec.flavor,
        "model_identity": f"{model_spec.name}/{model_spec.flavor}",
        "optimizer": "AdamW",
        "dtype": config.training.dtype,
        "local_batch_size": config.training.local_batch_size,
        "global_batch_size": config.training.global_batch_size,
        "sequence_length": config.training.seq_len,
        "tokens_per_optimizer_step": 256,
        "gradient_accumulation_steps": 1,
        "optimizer_steps": config.training.steps,
        "seed": config.debug.seed,
        "deterministic": config.debug.deterministic,
        "dataset": config.dataloader.dataset,
        "tokenizer_path": "tests/assets/tokenizer",
        "checkpoint_load": False,
        "checkpoint_save": False,
    }


def _raw_event_projection(raw_event: Mapping[str, object]) -> dict[str, object]:
    """Return the deterministic event projection named by the v0 contract.

    Full raw files, including omitted timestamps, preliminary rank-file
    digests, and process-local correlation/thread handles, remain byte-bound by
    the immutable runtime manifest.
    """

    projection = deepcopy(dict(raw_event))
    projection.pop("observed_time_ns", None)
    if projection.get("kind") == "bundle.collected":
        observation = projection.get("observation")
        if isinstance(observation, dict):
            observation.pop("rank_raw_sha256", None)
    if str(projection.get("kind", "")).startswith("collective."):
        observation = projection.get("observation")
        if isinstance(observation, dict):
            executor = observation.get("executor")
            if isinstance(executor, dict):
                executor.pop("thread_id", None)
            producer = observation.get("producer")
            if isinstance(producer, dict):
                producer.pop("correlation_id", None)
    return projection


def _raw_trace_projection(raw_trace: Mapping[str, object]) -> dict[str, object]:
    projection = deepcopy(dict(raw_trace))
    projection["events"] = [
        _raw_event_projection(event) for event in _mapping_list(raw_trace, "events")
    ]
    return projection


def _event_provenance(raw_event: Mapping[str, object]) -> dict[str, object]:
    return {
        "raw_event_id": str(raw_event.get("raw_event_id", "")),
        "projection_schema": RAW_EVENT_PROJECTION_SCHEMA,
        "raw_event_projection_sha256": _sha256_json(_raw_event_projection(raw_event)),
    }


def _producer_with_identity(producer: Mapping[str, object]) -> dict[str, object]:
    observed = deepcopy(dict(producer))
    observed.pop("identity", None)
    stable_mapping = deepcopy(observed)
    stable_mapping.pop("correlation_id", None)
    return {
        **observed,
        "identity": f"sha256:{_sha256_json(stable_mapping)}",
    }


def _expected_local_predecessors(
    events: Sequence[Mapping[str, object]], *, event_id_key: str
) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    last_core_id: str | None = None
    last_work_id: dict[str, str] = {}
    completed_work_ids: list[str] = []
    for event in events:
        event_id = str(event.get(event_id_key, ""))
        kind = str(event.get("kind", ""))
        if kind.startswith("collective."):
            observation = _mapping(event, "observation")
            work_id = str(observation.get("work_id", ""))
            lifecycle = str(observation.get("lifecycle", ""))
            if lifecycle == "enqueued":
                predecessors: list[str] = []
            else:
                prior_work_id = last_work_id.get(work_id)
                predecessors = [] if prior_work_id is None else [prior_work_id]
            last_work_id[work_id] = event_id
            if lifecycle == "completed":
                completed_work_ids.append(event_id)
        elif kind == "bundle.collected":
            predecessors = (
                [last_core_id, *completed_work_ids]
                if last_core_id is not None
                else list(completed_work_ids)
            )
        else:
            predecessors = [] if last_core_id is None else [last_core_id]
            last_core_id = event_id
        expected[event_id] = predecessors
    return expected


def merge_rank_traces(
    raw_traces: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Validate and merge one complete four-rank bundle without total ordering."""

    if len(raw_traces) != 4:
        raise ValueError(
            f"Scout B requires exactly ranks 0,1,2,3; found {len(raw_traces)} traces"
        )
    ranks = [_integer(_mapping(trace, "identity"), "rank") for trace in raw_traces]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"Scout B rank bundle contains a duplicate rank: {ranks}")
    if set(ranks) != {0, 1, 2, 3}:
        raise ValueError(f"Scout B requires exactly ranks 0,1,2,3; found {ranks}")
    ordered_raw = [trace for _, trace in sorted(zip(ranks, raw_traces, strict=True))]

    identities = [_mapping(trace, "identity") for trace in ordered_raw]
    run_attempts = {
        (str(identity.get("run_id", "")), str(identity.get("attempt_id", "")))
        for identity in identities
    }
    if len(run_attempts) != 1:
        raise ValueError("Scout B ranks disagree on run/attempt identity")
    run_id, attempt_id = next(iter(run_attempts))
    _validate_identifier(run_id, "run_id")
    _validate_identifier(attempt_id, "attempt_id")

    for expected_rank, trace in enumerate(ordered_raw):
        _validate_raw_rank_trace(trace, expected_rank)

    meshes = [_mapping(trace, "mesh") for trace in ordered_raw]
    if any(
        mesh.get("axes") != list(MESH_AXES) or mesh.get("degrees") != list(MESH_DEGREES)
        for mesh in meshes
    ):
        raise ValueError("Scout B ranks disagree on the fixed topology")
    expected_coordinates = {
        rank: [0, rank // 2, 0, rank % 2, 0, 0] for rank in range(4)
    }
    if any(
        mesh.get("coordinate") != expected_coordinates[rank]
        for rank, mesh in enumerate(meshes)
    ):
        raise ValueError("Scout B topology has an invalid rank coordinate")

    profiles = [_mapping(trace, "profile") for trace in ordered_raw]
    if any(profile != profiles[0] for profile in profiles[1:]):
        raise ValueError("Scout B ranks disagree on model/profile identity")
    lineages = [_mapping(trace, "lineage") for trace in ordered_raw]
    model_identities = {str(lineage.get("model_identity", "")) for lineage in lineages}
    if len(model_identities) != 1 or "" in model_identities:
        raise ValueError("Scout B ranks disagree on model identity")
    config_identities = {str(lineage.get("config_sha256", "")) for lineage in lineages}
    if len(config_identities) != 1 or not all(
        re.fullmatch(r"[0-9a-f]{64}", value) for value in config_identities
    ):
        raise ValueError("Scout B ranks disagree on config identity")

    rank_traces = [
        _normalize_rank_trace(trace, run_id=run_id, attempt_id=attempt_id)
        for trace in ordered_raw
    ]
    input_bundle = _input_bundle(rank_traces)
    process_groups = _canonical_process_groups(rank_traces)
    cross_rank_edges = _cross_rank_edges(rank_traces)
    normalized: dict[str, object] = {
        "schema": SCOUT_SCHEMA,
        "trace_id": "",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "topology": {
            "axes": list(MESH_AXES),
            "degrees": list(MESH_DEGREES),
            "world_size": 4,
            "rank_coordinates": {
                str(rank): coordinate
                for rank, coordinate in expected_coordinates.items()
            },
            "process_groups": process_groups,
        },
        "profile": deepcopy(profiles[0]),
        "lineage": {
            "config_sha256": next(iter(config_identities)),
            "model_identity": next(iter(model_identities)),
            "model_init_local_sha256_by_rank": {
                str(rank): str(lineage.get("model_init_local_sha256", ""))
                for rank, lineage in enumerate(lineages)
            },
        },
        "ordering": {
            "per_rank": "observed_source_order",
            "cross_rank_total_order": False,
        },
        "provenance_contract": deepcopy(PROVENANCE_CONTRACT),
        "input_bundle": input_bundle,
        "rank_traces": rank_traces,
        "cross_rank_edges": cross_rank_edges,
        "provenance": {
            "rank_raw_trace_projection_sha256": {
                str(rank): _sha256_json(_raw_trace_projection(raw_trace))
                for rank, raw_trace in enumerate(ordered_raw)
            }
        },
    }
    normalized["trace_id"] = _expected_trace_id(normalized)
    validate_normalized_bundle(normalized)
    return normalized


def _validate_raw_rank_trace(trace: Mapping[str, object], expected_rank: int) -> None:
    if trace.get("schema") != RAW_SCHEMA:
        raise ValueError(f"raw trace schema must be {RAW_SCHEMA}")
    identity = _mapping(trace, "identity")
    if (
        _integer(identity, "rank") != expected_rank
        or _integer(identity, "local_rank") != expected_rank
        or _integer(identity, "world_size") != 4
        or identity.get("process_id") != f"rank-{expected_rank}"
    ):
        raise ValueError(f"rank {expected_rank} has invalid process identity")
    events = _mapping_list(trace, "events")
    actual_orders = [_integer(event, "source_order") for event in events]
    expected_orders = list(range(1, len(events) + 1))
    if actual_orders != expected_orders:
        raise ValueError(
            f"rank {expected_rank} source order must be contiguous: {actual_orders}"
        )
    raw_ids = [str(event.get("raw_event_id", "")) for event in events]
    expected_ids = [f"raw-r{expected_rank}-{order:06d}" for order in expected_orders]
    if raw_ids != expected_ids or len(set(raw_ids)) != len(raw_ids):
        raise ValueError(f"rank {expected_rank} has invalid raw event identity")
    seen: set[str] = set()
    for event in events:
        _integer(event, "observed_time_ns")
        predecessors = _string_list(event, "causal_predecessors")
        if any(predecessor not in seen for predecessor in predecessors):
            raise ValueError(f"rank {expected_rank} has invalid causal source order")
        seen.add(str(event["raw_event_id"]))
    core_kinds = [
        str(event.get("kind"))
        for event in events
        if not str(event.get("kind", "")).startswith("collective.")
    ]
    if core_kinds != list(_CORE_LIFECYCLE):
        raise ValueError(f"rank {expected_rank} has an invalid core lifecycle")
    batch = next(event for event in events if event.get("kind") == "batch.observed")
    if _mapping(batch, "observation").get("batch_sha256") != _mapping(
        trace, "lineage"
    ).get("first_batch_sha256"):
        raise ValueError(f"rank {expected_rank} first-batch lineage disagrees")
    optimizer = next(
        event for event in events if event.get("kind") == "optimizer.mutated"
    )
    optimizer_observation = _mapping(optimizer, "observation")
    if optimizer_observation.get("mutated") is not True or optimizer_observation.get(
        "before_sha256"
    ) == optimizer_observation.get("after_sha256"):
        raise ValueError(f"rank {expected_rank} lacks optimizer mutation evidence")
    collected = next(
        event for event in events if event.get("kind") == "bundle.collected"
    )
    collection = _mapping(collected, "observation")
    if collection != {
        "primitive": "torch.distributed.all_gather_object",
        "participants": [0, 1, 2, 3],
    } and (
        collection.get("primitive") != "torch.distributed.all_gather_object"
        or collection.get("participants") != [0, 1, 2, 3]
    ):
        raise ValueError(f"rank {expected_rank} lacks complete collection evidence")
    _validate_raw_process_groups(trace, expected_rank)
    _validate_raw_placements(trace, expected_rank)
    _validate_raw_collectives(events, expected_rank)
    expected_predecessors = _expected_local_predecessors(
        events, event_id_key="raw_event_id"
    )
    if any(
        _string_list(event, "causal_predecessors")
        != expected_predecessors[str(event["raw_event_id"])]
        for event in events
    ):
        raise ValueError(f"rank {expected_rank} local causal graph is incomplete")


def _validate_raw_process_groups(trace: Mapping[str, object], rank: int) -> None:
    groups = _mapping_list(trace, "process_groups")
    by_axis: dict[str, list[dict[str, object]]] = defaultdict(list)
    for group in groups:
        by_axis[str(group.get("axis", ""))].append(group)
    expected = {
        "dp_shard": [rank % 2, rank % 2 + 2],
        "tp": [rank // 2 * 2, rank // 2 * 2 + 1],
    }
    for axis, members in expected.items():
        matches = [
            group
            for group in by_axis.get(axis, [])
            if group.get("members") == members
            and group.get("backend") == "nccl"
            and _integer(group, "local_coordinate")
            == (rank // 2 if axis == "dp_shard" else rank % 2)
        ]
        if not matches:
            raise ValueError(
                f"rank {rank} process group does not match {axis} membership {members}"
            )


def _validate_raw_placements(trace: Mapping[str, object], rank: int) -> None:
    placements = _mapping_list(trace, "tensor_placements")
    if not placements:
        raise ValueError(f"rank {rank} has no tensor placement evidence")
    has_dp_shard = False
    has_tp_shard = False
    for tensor in placements:
        mesh_axes = tensor.get("mesh_axes")
        entries = tensor.get("placements")
        if not isinstance(mesh_axes, list) or not isinstance(entries, list):
            raise ValueError(f"rank {rank} has malformed tensor placement evidence")
        if len(mesh_axes) != len(entries):
            raise ValueError(f"rank {rank} tensor placement axes do not align")
        for axis, placement in zip(mesh_axes, entries, strict=True):
            if not isinstance(placement, dict) or placement.get("axis") != axis:
                raise ValueError(f"rank {rank} has malformed tensor placement evidence")
            if axis == "dp_shard" and placement.get("kind") == "shard":
                has_dp_shard = True
            if axis == "tp" and placement.get("kind") == "shard":
                has_tp_shard = True
    if not has_dp_shard or not has_tp_shard:
        raise ValueError(
            f"rank {rank} tensor placement evidence lacks DP-shard or TP-shard placement"
        )


def _validate_raw_collectives(
    events: Sequence[Mapping[str, object]], rank: int
) -> None:
    by_work: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for event in events:
        if not str(event.get("kind", "")).startswith("collective."):
            continue
        observation = _mapping(event, "observation")
        work_id = str(observation.get("work_id", ""))
        if not work_id:
            raise ValueError(f"rank {rank} collective lifecycle lacks a work ID")
        by_work[work_id].append(event)
    if not by_work:
        raise ValueError(f"rank {rank} has no collective lifecycle evidence")
    for work_id, work_events in by_work.items():
        lifecycles = [
            str(_mapping(event, "observation").get("lifecycle", ""))
            for event in work_events
        ]
        kinds = [
            str(event.get("kind", "")).removeprefix("collective.")
            for event in work_events
        ]
        if lifecycles != list(_COLLECTIVE_LIFECYCLE) or kinds != lifecycles:
            raise ValueError(
                f"rank {rank} collective lifecycle for {work_id} is {lifecycles}"
            )
        observations = [_mapping(event, "observation") for event in work_events]
        stable_fields = (
            "axis",
            "collective_id",
            "work_id",
            "operation",
            "process_group",
            "executor",
            "stream",
            "producer",
        )
        for field in stable_fields:
            if any(
                observation.get(field) != observations[0].get(field)
                for observation in observations[1:]
            ):
                label = (
                    "producer correlation"
                    if field == "producer"
                    else "collective lifecycle"
                )
                raise ValueError(f"rank {rank} {label} changes within {work_id}")
        group = _mapping(observations[0], "process_group")
        if rank not in group.get("members", []):
            raise ValueError(f"rank {rank} collective process group omits its rank")
        if not _mapping(observations[0], "executor").get("thread_id"):
            raise ValueError(f"rank {rank} collective lacks executor evidence")
        if _mapping(observations[0], "stream").get("resource_id") is None:
            raise ValueError(f"rank {rank} collective lacks stream evidence")
        producer = _mapping(observations[0], "producer")
        if not producer.get("correlation_id") or not producer.get(
            "linked_correlation_id"
        ):
            raise ValueError(f"rank {rank} collective lacks producer correlation")


def _normalize_rank_trace(
    raw_trace: Mapping[str, object], *, run_id: str, attempt_id: str
) -> dict[str, object]:
    identity = _mapping(raw_trace, "identity")
    rank = _integer(identity, "rank")
    raw_events = _mapping_list(raw_trace, "events")
    id_map = {
        str(event["raw_event_id"]): (
            f"event:{run_id}:{attempt_id}:r{rank}:{_integer(event, 'source_order')}"
        )
        for event in raw_events
    }
    events = []
    provenance: dict[str, object] = {}
    for raw_event in raw_events:
        raw_id = str(raw_event["raw_event_id"])
        event_id = id_map[raw_id]
        observation = deepcopy(raw_event.get("observation", {}))
        if str(raw_event.get("kind", "")).startswith("collective."):
            if not isinstance(observation, dict):
                raise TypeError("collective observation must be an object")
            observation["producer"] = _producer_with_identity(
                _mapping(observation, "producer")
            )
        event_provenance = _event_provenance(raw_event)
        event = {
            "event_id": event_id,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "process_id": f"rank-{rank}",
            "rank": rank,
            "clock": {
                "domain": "rank_source_order",
                "order": _integer(raw_event, "source_order"),
            },
            "step": _integer(raw_event, "step"),
            "phase": str(raw_event.get("phase", "")),
            "kind": str(raw_event.get("kind", "")),
            "causal_predecessors": [
                id_map[predecessor]
                for predecessor in _string_list(raw_event, "causal_predecessors")
            ],
            "observation": observation,
            "provenance": event_provenance,
        }
        events.append(event)
        provenance[event_id] = event_provenance
    placements = deepcopy(_mapping_list(raw_trace, "tensor_placements"))
    return {
        "process": {
            "process_id": f"rank-{rank}",
            "rank": rank,
            "local_rank": _integer(identity, "local_rank"),
            "world_size": _integer(identity, "world_size"),
        },
        "device": deepcopy(_mapping(raw_trace, "device")),
        "mesh": deepcopy(_mapping(raw_trace, "mesh")),
        "process_groups": deepcopy(_mapping_list(raw_trace, "process_groups")),
        "tensor_placements": placements,
        "placement_summary": {
            "num_tensors": len(placements),
            "has_dp_shard": any(
                placement.get("axis") == "dp_shard" and placement.get("kind") == "shard"
                for tensor in placements
                for placement in tensor.get("placements", [])
                if isinstance(placement, dict)
            ),
            "has_tp_shard": any(
                placement.get("axis") == "tp" and placement.get("kind") == "shard"
                for tensor in placements
                for placement in tensor.get("placements", [])
                if isinstance(placement, dict)
            ),
            "schema_sha256": _sha256_json(placements),
        },
        "lineage": deepcopy(_mapping(raw_trace, "lineage")),
        "events": events,
        "provenance": provenance,
    }


def _input_bundle(rank_traces: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_dp: dict[str, dict[str, object]] = {}
    for dp_coordinate, ranks in ((0, [0, 1]), (1, [2, 3])):
        digests = {
            str(_mapping(rank_traces[rank], "lineage").get("first_batch_sha256", ""))
            for rank in ranks
        }
        if len(digests) != 1:
            raise ValueError(
                f"TP peers at DP coordinate {dp_coordinate} disagree on local input identity"
            )
        digest = next(iter(digests))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("TP peers have a malformed local first-batch identity")
        by_dp[f"dp:{dp_coordinate}"] = {
            "batch_sha256": digest,
            "ranks": ranks,
        }
    ordered = [{"dp_key": dp_key, **by_dp[dp_key]} for dp_key in ("dp:0", "dp:1")]
    return {
        "dp_keys": ["dp:0", "dp:1"],
        "by_dp_coordinate": by_dp,
        "global_sha256": _sha256_json(ordered),
    }


def _canonical_process_groups(
    rank_traces: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, tuple[int, ...]], dict[str, object]] = {}
    for trace in rank_traces:
        for group in _mapping_list(trace, "process_groups"):
            axis = str(group.get("axis", ""))
            members_value = group.get("members")
            if not isinstance(members_value, list) or not all(
                isinstance(member, int) for member in members_value
            ):
                raise ValueError("process group members must be integer ranks")
            members = tuple(members_value)
            if axis not in {"dp_shard", "tp"}:
                continue
            key = (axis, members)
            candidate = {
                "canonical_id": f"{axis}:{','.join(str(member) for member in members)}",
                "axis": axis,
                "members": list(members),
                "backend": group.get("backend"),
            }
            if key in groups and groups[key] != candidate:
                raise ValueError("conflicting process group evidence")
            groups[key] = candidate
    expected = {
        ("tp", (0, 1)),
        ("tp", (2, 3)),
        ("dp_shard", (0, 2)),
        ("dp_shard", (1, 3)),
    }
    if set(groups) != expected:
        raise ValueError(
            f"Scout B process group coverage is incomplete: {sorted(groups)}"
        )
    return [groups[key] for key in sorted(groups)]


def _collective_issue_order(events: Sequence[Mapping[str, object]]) -> int:
    """Return the per-rank order of a work's ``collective.enqueued`` event.

    Only this event carries faithful ordering: it inherits the NCCL Flight
    Recorder ``record_id`` sequence. The ``started``/``completed`` events are
    appended contiguously after the step, so their relative order across
    different works asserts a serialization that did not occur.
    """

    for event in events:
        if str(_mapping(event, "observation").get("lifecycle", "")) == "enqueued":
            return _integer(_mapping(event, "clock"), "order")
    raise ValueError("collective work has no enqueued event to order by")


def _collective_work(
    rank_trace: Mapping[str, object],
) -> dict[str, list[Mapping[str, object]]]:
    work: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for event in _mapping_list(rank_trace, "events"):
        if str(event.get("kind", "")).startswith("collective."):
            work_id = str(_mapping(event, "observation").get("work_id", ""))
            work[work_id].append(event)
    return work


def _cross_rank_edges(
    rank_traces: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_collective: dict[str, dict[int, str]] = defaultdict(dict)
    collection_events: dict[int, str] = {}
    for trace in rank_traces:
        rank = _integer(_mapping(trace, "process"), "rank")
        for work_events in _collective_work(trace).values():
            completed = work_events[-1]
            observation = _mapping(completed, "observation")
            by_collective[str(observation["collective_id"])][rank] = str(
                completed["event_id"]
            )
        collected = next(
            event
            for event in _mapping_list(trace, "events")
            if event.get("kind") == "bundle.collected"
        )
        collection_events[rank] = str(collected["event_id"])
    edges: list[dict[str, object]] = []
    edge_index = 0
    for collective_id, events in sorted(by_collective.items()):
        ranks = sorted(events)
        for left_index, left in enumerate(ranks):
            for right in ranks[left_index + 1 :]:
                edge_index += 1
                evidence = {
                    "kind": "collective_group",
                    "collective_id": collective_id,
                }
                edges.append(
                    {
                        "edge_id": f"sync-{edge_index:06d}",
                        "relation": "synchronizes",
                        "ordered": False,
                        "endpoints": [events[left], events[right]],
                        "evidence": evidence,
                        "evidence_sha256": _sha256_json(evidence),
                    }
                )
    for left in range(4):
        for right in range(left + 1, 4):
            edge_index += 1
            evidence = {
                "kind": "all_gather_object",
                "participants": [0, 1, 2, 3],
            }
            edges.append(
                {
                    "edge_id": f"sync-{edge_index:06d}",
                    "relation": "synchronizes",
                    "ordered": False,
                    "endpoints": [collection_events[left], collection_events[right]],
                    "evidence": evidence,
                    "evidence_sha256": _sha256_json(evidence),
                }
            )
    return edges


def _validate_normalized_collectives(
    events: Sequence[Mapping[str, object]], rank: int
) -> None:
    _validate_raw_collectives(events, rank)
    for work_id, work_events in _collective_work({"events": list(events)}).items():
        for event in work_events:
            producer = _mapping(_mapping(event, "observation"), "producer")
            if producer != _producer_with_identity(producer):
                raise ValueError(
                    f"rank {rank} producer correlation identity is invalid for {work_id}"
                )


def validate_normalized_bundle(bundle: Mapping[str, object]) -> None:
    """Reject incomplete identity, order, topology, placement, and work facts."""

    if bundle.get("schema") != SCOUT_SCHEMA:
        raise ValueError(f"normalized bundle schema must be {SCOUT_SCHEMA}")
    contract = bundle.get("provenance_contract")
    if contract != PROVENANCE_CONTRACT:
        found = contract.get("version") if isinstance(contract, Mapping) else None
        if found is None:
            raise ValueError(
                "normalized provenance contract is missing, unversioned, or "
                "altered; expected " + PROVENANCE_CONTRACT_VERSION
            )
        raise ValueError(
            f"normalized provenance contract version {found} does not match "
            f"{PROVENANCE_CONTRACT_VERSION}: this bundle was sealed under a "
            "different contract, which is not the same as tampering"
        )
    run_id = str(bundle.get("run_id", ""))
    attempt_id = str(bundle.get("attempt_id", ""))
    _validate_identifier(run_id, "run_id")
    _validate_identifier(attempt_id, "attempt_id")
    ordering = _mapping(bundle, "ordering")
    if ordering != {
        "per_rank": "observed_source_order",
        "cross_rank_total_order": False,
    }:
        raise ValueError(
            "normalized bundle makes an unsupported cross-rank total order"
        )
    rank_traces = _mapping_list(bundle, "rank_traces")
    if len(rank_traces) != 4:
        raise ValueError("normalized bundle must contain four rank traces")
    all_event_ids: set[str] = set()
    placement_schemas: set[str] = set()
    collective_participants: dict[str, set[int]] = defaultdict(set)
    collective_members: dict[str, tuple[int, ...]] = {}
    for expected_rank, trace in enumerate(rank_traces):
        process = _mapping(trace, "process")
        if process != {
            "process_id": f"rank-{expected_rank}",
            "rank": expected_rank,
            "local_rank": expected_rank,
            "world_size": 4,
        }:
            raise ValueError("normalized rank process identity is incomplete")
        mesh = _mapping(trace, "mesh")
        expected_coordinate = [0, expected_rank // 2, 0, expected_rank % 2, 0, 0]
        if (
            mesh.get("axes") != list(MESH_AXES)
            or mesh.get("degrees") != list(MESH_DEGREES)
            or mesh.get("coordinate") != expected_coordinate
        ):
            raise ValueError("normalized topology is inconsistent")
        events = _mapping_list(trace, "events")
        provenance = _mapping(trace, "provenance")
        if set(provenance) != {str(event.get("event_id")) for event in events}:
            raise ValueError("normalized provenance does not cover every event")
        prior: set[str] = set()
        for order, event in enumerate(events, start=1):
            event_id = str(event.get("event_id", ""))
            expected_id = f"event:{run_id}:{attempt_id}:r{expected_rank}:{order}"
            if event_id != expected_id or event_id in all_event_ids:
                raise ValueError("normalized event identity is invalid or duplicated")
            all_event_ids.add(event_id)
            if _mapping(event, "clock") != {
                "domain": "rank_source_order",
                "order": order,
            }:
                raise ValueError(
                    "normalized event clock violates per-rank source order"
                )
            predecessors = _string_list(event, "causal_predecessors")
            if any(predecessor not in prior for predecessor in predecessors):
                raise ValueError(
                    "normalized causal edge violates per-rank source order"
                )
            prior.add(event_id)
            event_provenance = _mapping(event, "provenance")
            if (
                provenance.get(event_id) != event_provenance
                or event_provenance.get("projection_schema")
                != RAW_EVENT_PROJECTION_SCHEMA
                or not str(event_provenance.get("raw_event_id", ""))
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(event_provenance.get("raw_event_projection_sha256", "")),
                )
            ):
                raise ValueError("normalized event lacks projection provenance")
        expected_predecessors = _expected_local_predecessors(
            events, event_id_key="event_id"
        )
        if any(
            _string_list(event, "causal_predecessors")
            != expected_predecessors[str(event["event_id"])]
            for event in events
        ):
            raise ValueError(
                f"rank {expected_rank} normalized local causal graph is incomplete"
            )
        _validate_normalized_collectives(events, expected_rank)
        summary = _mapping(trace, "placement_summary")
        tensors = _mapping_list(trace, "tensor_placements")
        if (
            summary.get("has_dp_shard") is not True
            or summary.get("has_tp_shard") is not True
            or summary.get("num_tensors") != len(tensors)
            or not tensors
        ):
            raise ValueError("normalized tensor placement evidence is incomplete")
        # Recomputed, not trusted. This runs over JSON read from disk, so the
        # digest and the bytes it is supposed to cover are two independent
        # inputs; comparing digests across ranks says nothing unless each one
        # is first known to describe that rank's own placements. A stale
        # digest over tampered placements otherwise agrees with its peers and
        # hides exactly the fault the placement invariants exist to catch.
        if str(summary.get("schema_sha256", "")) != _sha256_json(tensors):
            raise ValueError(
                f"rank {expected_rank} placement schema digest does not cover "
                "its tensor placements"
            )
        placement_schemas.add(str(summary.get("schema_sha256", "")))
        for work_events in _collective_work(trace).values():
            observations = [_mapping(event, "observation") for event in work_events]
            collective_id = str(observations[0].get("collective_id", ""))
            members = tuple(
                _mapping(observations[0], "process_group").get("members", [])
            )
            if (
                collective_id in collective_members
                and collective_members[collective_id] != members
            ):
                raise ValueError("collective process group conflicts across ranks")
            collective_members[collective_id] = members
            collective_participants[collective_id].add(expected_rank)
    if len(placement_schemas) != 1:
        raise ValueError("normalized tensor placement schemas conflict across ranks")
    for collective_id, participants in collective_participants.items():
        if participants != set(collective_members[collective_id]):
            raise ValueError(
                f"collective {collective_id} is incomplete for its process group"
            )
    _validate_input_bundle(bundle, rank_traces)
    edges = _mapping_list(bundle, "cross_rank_edges")
    for edge in edges:
        if edge.get("relation") != "synchronizes" or edge.get("ordered") is not False:
            raise ValueError(
                "normalized bundle makes an unsupported cross-rank total order"
            )
        endpoints = edge.get("endpoints")
        if (
            not isinstance(endpoints, list)
            or len(endpoints) != 2
            or any(endpoint not in all_event_ids for endpoint in endpoints)
        ):
            raise ValueError("cross-rank synchronization edge has invalid endpoints")
        evidence = _mapping(edge, "evidence")
        if edge.get("evidence_sha256") != _sha256_json(evidence):
            raise ValueError("cross-rank synchronization evidence digest is invalid")
    if edges != _cross_rank_edges(rank_traces):
        raise ValueError("normalized cross-rank synchronization graph is incomplete")
    rank_projection_digests = _mapping(
        _mapping(bundle, "provenance"), "rank_raw_trace_projection_sha256"
    )
    if set(rank_projection_digests) != {"0", "1", "2", "3"} or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        for digest in rank_projection_digests.values()
    ):
        raise ValueError("normalized rank projection provenance is incomplete")
    expected_trace_id = _expected_trace_id(bundle)
    if bundle.get("trace_id") != expected_trace_id:
        raise ValueError(
            f"trace_id must be {expected_trace_id}, found {bundle.get('trace_id')!r}"
        )


def _validate_input_bundle(
    bundle: Mapping[str, object], rank_traces: Sequence[Mapping[str, object]]
) -> None:
    expected = _input_bundle(rank_traces)
    if bundle.get("input_bundle") != expected:
        raise ValueError("ordered DP-keyed global input identity is inconsistent")


_PLACEMENT_KINDS = ("shard", "replicate", "partial")
# Mesh axes that carry a model-parameter DTensor placement in this DPxTP run.
# Named explicitly because the divisibility check needs a degree for every axis
# a placement claims, and a silently missing axis would make it vacuous.
_PLACEMENT_MESH_AXES = ("dp_shard", "tp")


def _placement_facts(
    rank_traces: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Project per-parameter DTensor placements with their axis and dim.

    ``validate_normalized_bundle`` recomputes each rank's
    ``placement_summary.schema_sha256`` over that rank's own
    ``tensor_placements`` bytes and then requires the four digests to agree, so
    by the time this runs the ranks are known to have observed one placement
    list, local shapes included. That is what licenses exporting it once. The
    per-rank digests are exported alongside so a checker asserts the agreement
    itself instead of taking this projection's word for it.

    Observation point, stated because the facts are easy to over-read: the
    snapshot is taken once, after the model is built and before
    ``Trainer.train``, over ``model_parts[0].named_parameters()``. It is the
    parameter set the single observed AdamW step updates, but it is not a
    second sample taken inside the optimizer pre-hook, and no gradient
    placement is observed at all.
    """

    digests = [
        str(_mapping(trace, "placement_summary").get("schema_sha256", ""))
        for trace in rank_traces
    ]
    if len(set(digests)) != 1 or not digests[0]:
        raise ValueError("placement facts require one agreed placement schema")
    tensors = _mapping_list(rank_traces[0], "tensor_placements")
    degrees_by_axis = dict(
        zip(
            _string_list(_mapping(rank_traces[0], "mesh"), "axes"),
            _integer_list(_mapping(rank_traces[0], "mesh"), "degrees"),
            strict=True,
        )
    )
    mesh_axis_degrees: dict[str, int] = {}
    for axis in _PLACEMENT_MESH_AXES:
        if axis not in degrees_by_axis:
            raise ValueError(f"observed mesh has no degree for axis {axis}")
        mesh_axis_degrees[axis] = int(degrees_by_axis[axis])
    parameters: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    for tensor in tensors:
        name = str(tensor.get("name", ""))
        if not name:
            raise ValueError("placement evidence has an unnamed parameter")
        # Names key the exported TLA functions, so a duplicate would build a
        # function with a repeated key rather than a detectable fact.
        if any(parameter["name"] == name for parameter in parameters):
            raise ValueError(f"placement evidence names {name} twice")
        mesh_axes = _string_list(tensor, "mesh_axes")
        global_shape = _integer_list(tensor, "global_shape")
        local_shape = _integer_list(tensor, "local_shape")
        if not mesh_axes or not global_shape:
            raise ValueError(f"placement evidence for {name} is degenerate")
        if len(global_shape) != len(local_shape):
            raise ValueError(f"placement shapes for {name} do not align")
        entries = _mapping_list(tensor, "placements")
        if len(entries) != len(mesh_axes):
            raise ValueError(f"placement axes for {name} do not align")
        parameters.append(
            {
                "name": name,
                "mesh_axes": list(mesh_axes),
                "global_shape": list(global_shape),
                "local_shape": list(local_shape),
                "placement_ids": [],
            }
        )
        for axis, entry in zip(mesh_axes, entries, strict=True):
            if entry.get("axis") != axis:
                raise ValueError(f"placement axis for {name} is out of order")
            kind = str(entry.get("kind", ""))
            if kind not in _PLACEMENT_KINDS:
                raise ValueError(f"placement for {name} has unknown kind {kind!r}")
            if axis not in mesh_axis_degrees:
                raise ValueError(f"placement for {name} claims unknown axis {axis!r}")
            record: dict[str, Any] = {
                "placement_id": f"placement:{len(placements):04d}",
                "parameter": name,
                "axis": axis,
                "kind": kind,
                "strided": entry.get("strided") is True,
            }
            if kind == "shard":
                dim = entry.get("dim")
                if not isinstance(dim, int) or isinstance(dim, bool):
                    raise ValueError(f"shard placement for {name} has no integer dim")
                if not 0 <= dim < len(global_shape):
                    raise ValueError(
                        f"shard placement for {name} names dim {dim} outside its shape"
                    )
                record["shard_dim"] = dim
            elif "dim" in entry:
                raise ValueError(f"non-shard placement for {name} carries a shard dim")
            parameters[-1]["placement_ids"].append(record["placement_id"])
            placements.append(record)
    return {
        "schema_digest": digests[0],
        "schema_digest_by_rank": digests,
        "mesh_axis_degrees": mesh_axis_degrees,
        "parameters": parameters,
        "placements": placements,
    }


def _formal_projection(bundle: Mapping[str, object]) -> dict[str, Any]:
    validate_normalized_bundle(bundle)
    rank_traces = _mapping_list(bundle, "rank_traces")
    event_facts: list[dict[str, Any]] = []
    collectives: list[dict[str, Any]] = []
    for trace in rank_traces:
        rank = _integer(_mapping(trace, "process"), "rank")
        for event in _mapping_list(trace, "events"):
            provenance = _mapping(event, "provenance")
            event_facts.append(
                {
                    "event_id": str(event["event_id"]),
                    "rank": rank,
                    "order": _integer(_mapping(event, "clock"), "order"),
                    "kind": str(event["kind"]),
                    "raw_event_id": str(provenance["raw_event_id"]),
                    "projection_schema": str(provenance["projection_schema"]),
                    "projection_sha256": str(provenance["raw_event_projection_sha256"]),
                    "predecessors": list(_string_list(event, "causal_predecessors")),
                }
            )
        for work_id, events in sorted(_collective_work(trace).items()):
            observations = [_mapping(event, "observation") for event in events]
            first = observations[0]
            members_value = _mapping(first, "process_group").get("members")
            if not isinstance(members_value, list) or not all(
                isinstance(member, int) for member in members_value
            ):
                raise TypeError("formal collective members must be integer ranks")
            collectives.append(
                {
                    "work_id": work_id,
                    "rank": rank,
                    "collective_id": str(first.get("collective_id", "")),
                    "members": list(members_value),
                    "lifecycle": [
                        str(observation.get("lifecycle", ""))
                        for observation in observations
                    ],
                    "producers": [
                        str(_mapping(observation, "producer").get("identity", ""))
                        for observation in observations
                    ],
                    "event_ids": [str(event["event_id"]) for event in events],
                    # Communicator identity. The global key is canonical_id
                    # (axis:members:description), NOT runtime_pg_id: that is a
                    # per-rank local numbering, so rank 0's pg_id 4 and rank 1's
                    # pg_id 4 are different communicators. runtime_pg_id is
                    # exported only so a checker can assert the two agree.
                    "comm": str(
                        _mapping(first, "process_group").get("canonical_id", "")
                    ),
                    "comm_description": str(
                        _mapping(first, "process_group").get("description", "")
                    ),
                    "runtime_pg_id": _integer(
                        _mapping(first, "process_group"), "runtime_pg_id"
                    ),
                    "operation": str(first.get("operation", "")),
                    "axis": str(first.get("axis", "")),
                    # Opaque label only: stream identity is INFERRED by a
                    # positional zip of Flight Recorder entries to Kineto
                    # kernels, not observed. Do not build a guard on it.
                    "stream_id": _integer(_mapping(first, "stream"), "resource_id"),
                    # Issue order is the one faithful ordering among collective
                    # events: the enqueued event inherits flight_record_id
                    # order. The started/completed sub-order is synthetic, so it
                    # must never be used as evidence of ordering.
                    "issue_order": _collective_issue_order(events),
                    "has_executor": all(
                        bool(_mapping(observation, "executor").get("thread_id"))
                        for observation in observations
                    ),
                    "has_stream": all(
                        _mapping(observation, "stream").get("resource_id") is not None
                        for observation in observations
                    ),
                    "observation_count": 1,
                },
            )
    event_counts = [len(_mapping_list(trace, "events")) for trace in rank_traces]
    if len(set(event_counts)) != 1:
        raise ValueError("formal rank-local event counts must agree")
    events_per_rank = event_counts[0]
    for event in event_facts:
        kind = str(event["kind"])
        kind_code = _FORMAL_EVENT_KIND_CODES.get(kind)
        if kind_code is None:
            raise ValueError(f"formal event kind has no bounded code: {kind}")
        event["identity"] = (
            int(event["rank"]) * events_per_rank + int(event["order"]) - 1
        ) * _FORMAL_EVENT_KIND_BASE + kind_code
    event_by_id = {
        str(event["event_id"]): {**event, "index": index}
        for index, event in enumerate(event_facts)
    }
    event_facts = [event_by_id[str(event["event_id"])] for event in event_facts]

    def reference(event_id: str) -> dict[str, object]:
        event = event_by_id[event_id]
        return {
            "event_id": event_id,
            "index": event["index"],
            "rank": event["rank"],
            "order": event["order"],
            "kind": event["kind"],
            "identity": event["identity"],
        }

    for event in event_facts:
        event["predecessor_references"] = [
            reference(str(predecessor)) for predecessor in event["predecessors"]
        ]
    cross_rank_edges = deepcopy(_mapping_list(bundle, "cross_rank_edges"))
    for edge in cross_rank_edges:
        edge["endpoint_references"] = [
            reference(str(endpoint)) for endpoint in edge["endpoints"]
        ]
    for collective in collectives:
        collective["event_references"] = [
            reference(str(event_id)) for event_id in collective["event_ids"]
        ]
    collective_rank_counts: dict[int, int] = defaultdict(int)
    collective_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for index, collective in enumerate(collectives):
        rank = int(collective["rank"])
        rank_order = collective_rank_counts[rank]
        collective_rank_counts[rank] += 1
        collective_by_key[(str(collective["collective_id"]), rank)] = {
            **collective,
            "index": index,
            "rank_order": rank_order,
        }
    collectives = [
        collective_by_key[(str(collective["collective_id"]), int(collective["rank"]))]
        for collective in collectives
    ]

    def collective_reference(collective: Mapping[str, object]) -> dict[str, object]:
        return {
            "work_id": collective["work_id"],
            "index": collective["index"],
            "rank": collective["rank"],
            "rank_order": collective["rank_order"],
            "collective_id": collective["collective_id"],
        }

    for collective in collectives:
        collective["peer_references"] = [
            collective_reference(
                collective_by_key[(str(collective["collective_id"]), int(member))]
            )
            for member in collective["members"]
        ]
    input_by_dp = _mapping(_mapping(bundle, "input_bundle"), "by_dp_coordinate")
    if len(set(collective_rank_counts.values())) != 1:
        raise ValueError("formal rank-local event/work counts must agree")
    return {
        "trace_id": bundle["trace_id"],
        "projection_schema": RAW_EVENT_PROJECTION_SCHEMA,
        "event_id_prefix": f"event:{bundle['run_id']}:{bundle['attempt_id']}:",
        "events_per_rank": events_per_rank,
        "collectives_per_rank": next(iter(collective_rank_counts.values())),
        "event_ids_by_rank": [
            [
                str(event["event_id"])
                for event in event_facts
                if int(event["rank"]) == rank
            ]
            for rank in range(4)
        ],
        "collective_work_ids_by_rank": [
            [
                str(collective["work_id"])
                for collective in collectives
                if int(collective["rank"]) == rank
            ]
            for rank in range(4)
        ],
        "collective_ids_by_rank": [
            [
                str(collective["collective_id"])
                for collective in collectives
                if int(collective["rank"]) == rank
            ]
            for rank in range(4)
        ],
        "rank_coordinates": [
            {
                "rank": rank,
                "dp": rank // 2,
                "tp": rank % 2,
                "input_sha256": _mapping(rank_traces[rank], "lineage")[
                    "first_batch_sha256"
                ],
                "has_dp_shard": _mapping(rank_traces[rank], "placement_summary")[
                    "has_dp_shard"
                ],
                "has_tp_shard": _mapping(rank_traces[rank], "placement_summary")[
                    "has_tp_shard"
                ],
            }
            for rank in range(4)
        ],
        "dp_inputs": [
            input_by_dp["dp:0"]["batch_sha256"],
            input_by_dp["dp:1"]["batch_sha256"],
        ],
        "placement_facts": _placement_facts(rank_traces),
        "events": event_facts,
        "cross_rank_edges": cross_rank_edges,
        "collectives": collectives,
    }


def export_scout_b_tla_facts(
    bundle: Mapping[str, object], *, module_name: str = "ScoutBFacts"
) -> str:
    """Export only observed DPxTP values; maintained modules own predicates."""

    return _export_scout_b_tla_module(
        _formal_projection(bundle), module_name=module_name
    )


def export_scout_b_tla_bad_facts(bundle: Mapping[str, object]) -> str:
    """Export one producer-correlation mutation for the named TLC negative."""

    facts = deepcopy(_formal_projection(bundle))
    facts["collectives"][0]["producers"][-1] = "controlled-invalid-producer"
    return _export_scout_b_tla_module(
        facts,
        module_name="ScoutBBadFacts",
        mutation="collective_producer_completion_mismatch",
    )


def _export_scout_b_tla_module(
    facts: Mapping[str, Any], *, module_name: str, mutation: str | None = None
) -> str:
    ranks = facts["rank_coordinates"]
    events = facts["events"]
    edges = facts["cross_rank_edges"]
    collectives = facts["collectives"]
    placements = facts["placement_facts"]
    lines = [
        f"------------------------------ MODULE {module_name} ------------------------------",
        "EXTENDS Naturals, Sequences, FiniteSets, TLC",
        "",
        # Emitted into the artifact on purpose. The same warning existed only
        # as a Python comment, so grep -ril infer over the sealed bundle
        # returned nothing and a reader of the facts had no way to tell an
        # inference from an observation.
        r"\* INFERRED, NOT OBSERVED: CollectiveProducers and CollectiveStream",
        r"\* come from a positional zip of NCCL Flight Recorder entries,",
        r"\* ordered by record_id, to Kineto kernels, ordered by start time.",
        r"\* There is no join key between the two sources; equal counts and",
        r"\* equal order are the whole argument. Ticket 09 replaces this with",
        r"\* a joined key. Do not build an invariant on either operator.",
        "",
        f"TraceId == {_tla_string(str(facts['trace_id']))}",
    ]
    if mutation is not None:
        lines.append(f"MutationName == {_tla_string(mutation)}")
    lines.extend(
        [
            "RankSet == {0, 1, 2, 3}",
            "RankCoordinates == "
            + _tla_function(
                [
                    (str(rank["rank"]), f"<<{rank['dp']}, {rank['tp']}>>")
                    for rank in ranks
                ]
            ),
            "InputDigestByRank == "
            + _tla_function(
                [
                    (str(rank["rank"]), _tla_string(str(rank["input_sha256"])))
                    for rank in ranks
                ]
            ),
            "PlacementHasDpShard == "
            + _tla_function(
                [
                    (str(rank["rank"]), "TRUE" if rank["has_dp_shard"] else "FALSE")
                    for rank in ranks
                ]
            ),
            "PlacementHasTpShard == "
            + _tla_function(
                [
                    (str(rank["rank"]), "TRUE" if rank["has_tp_shard"] else "FALSE")
                    for rank in ranks
                ]
            ),
            # Structural per-parameter placements. The two booleans above are
            # kept verbatim so ScoutBPlacementValid still means exactly what it
            # meant; PlacementBooleansDerivable proves they are the aggregate of
            # these facts rather than an independent claim.
            #
            # OBSERVATION POINT: this snapshot is taken once, after the model is
            # built and before Trainer.train, over
            # model_parts[0].named_parameters(). It is the parameter set the one
            # observed AdamW step updates -- OptimizerBoundaryObserved pins that
            # the step is in the trace -- but it is not re-sampled inside the
            # optimizer pre-hook, and no gradient placement is observed at all.
            r"\* BEGIN structural placement facts",
            r"\* OBSERVED once per rank, after model construction and before",
            r"\* Trainer.train, over model_parts[0].named_parameters(). Every",
            r"\* rank agreed on the whole list, so it is exported once and",
            r"\* PlacementSchemaDigestByRank carries the per-rank digests.",
            r"\* Shard dims are 0-based as PyTorch reports them; TLA sequences",
            r"\* are 1-based, so a shape lookup is shape[dim + 1].",
            f"PlacementSchemaDigest == {_tla_string(str(placements['schema_digest']))}",
            "PlacementSchemaDigestByRank == "
            + _tla_function(
                [
                    (str(rank), _tla_string(str(digest)))
                    for rank, digest in enumerate(placements["schema_digest_by_rank"])
                ]
            ),
            "MeshAxisDegree == "
            + _tla_function(
                [
                    (_tla_string(axis), str(int(degree)))
                    for axis, degree in sorted(placements["mesh_axis_degrees"].items())
                ]
            ),
            "ParameterNames == "
            + _tla_sequence(
                [
                    _tla_string(str(parameter["name"]))
                    for parameter in placements["parameters"]
                ]
            ),
            "ParameterMeshAxes == "
            + _tla_function(
                [
                    (
                        _tla_string(str(parameter["name"])),
                        "<<"
                        + ", ".join(
                            _tla_string(str(axis)) for axis in parameter["mesh_axes"]
                        )
                        + ">>",
                    )
                    for parameter in placements["parameters"]
                ]
            ),
            "ParameterGlobalShape == "
            + _tla_function(
                [
                    (
                        _tla_string(str(parameter["name"])),
                        "<<"
                        + ", ".join(
                            str(int(size)) for size in parameter["global_shape"]
                        )
                        + ">>",
                    )
                    for parameter in placements["parameters"]
                ]
            ),
            "ParameterLocalShape == "
            + _tla_function(
                [
                    (
                        _tla_string(str(parameter["name"])),
                        "<<"
                        + ", ".join(str(int(size)) for size in parameter["local_shape"])
                        + ">>",
                    )
                    for parameter in placements["parameters"]
                ]
            ),
            "PlacementIds == "
            + _tla_sequence(
                [
                    _tla_string(str(placement["placement_id"]))
                    for placement in placements["placements"]
                ]
            ),
            "PlacementParameter == "
            + _tla_function(
                [
                    (
                        _tla_string(str(placement["placement_id"])),
                        _tla_string(str(placement["parameter"])),
                    )
                    for placement in placements["placements"]
                ]
            ),
            "PlacementAxis == "
            + _tla_function(
                [
                    (
                        _tla_string(str(placement["placement_id"])),
                        _tla_string(str(placement["axis"])),
                    )
                    for placement in placements["placements"]
                ]
            ),
            "PlacementKind == "
            + _tla_function(
                [
                    (
                        _tla_string(str(placement["placement_id"])),
                        _tla_string(str(placement["kind"])),
                    )
                    for placement in placements["placements"]
                ]
            ),
            # _StridedShard, which FSDP2 uses for the outer axis when an inner
            # axis shards the same tensor dim. StridedShardIsAnOuterComposedShard
            # checks exactly that correspondence.
            "PlacementStrided == "
            + _tla_function(
                [
                    (
                        _tla_string(str(placement["placement_id"])),
                        "TRUE" if placement["strided"] else "FALSE",
                    )
                    for placement in placements["placements"]
                ]
            ),
            # Domain-restricted to the shard placements on purpose: a replicate
            # or partial placement has no dim, and a sentinel value would be a
            # number a checker could accidentally read as one.
            "PlacementShardDim == "
            + _tla_function(
                [
                    (
                        _tla_string(str(placement["placement_id"])),
                        str(int(placement["shard_dim"])),
                    )
                    for placement in placements["placements"]
                    if placement["kind"] == "shard"
                ]
            ),
            r"\* END structural placement facts",
            f"RawEventProjectionSchema == {_tla_string(str(facts['projection_schema']))}",
            "EventIds == "
            + _tla_sequence([_tla_string(str(event["event_id"])) for event in events]),
            "EventRank == "
            + _tla_function(
                [
                    (_tla_string(str(event["event_id"])), str(event["rank"]))
                    for event in events
                ]
            ),
            "EventOrder == "
            + _tla_function(
                [
                    (_tla_string(str(event["event_id"])), str(event["order"]))
                    for event in events
                ]
            ),
            "EventKind == "
            + _tla_function(
                [
                    (
                        _tla_string(str(event["event_id"])),
                        _tla_string(str(event["kind"])),
                    )
                    for event in events
                ]
            ),
            "EventRawId == "
            + _tla_function(
                [
                    (
                        _tla_string(str(event["event_id"])),
                        _tla_string(str(event["raw_event_id"])),
                    )
                    for event in events
                ]
            ),
            "EventProjectionSchema == "
            + _tla_function(
                [
                    (
                        _tla_string(str(event["event_id"])),
                        _tla_string(str(event["projection_schema"])),
                    )
                    for event in events
                ]
            ),
            "EventProjectionDigest == "
            + _tla_function(
                [
                    (
                        _tla_string(str(event["event_id"])),
                        _tla_string(str(event["projection_sha256"])),
                    )
                    for event in events
                ]
            ),
            "EventPredecessors == "
            + _tla_function(
                [
                    (
                        _tla_string(str(event["event_id"])),
                        "<<"
                        + ", ".join(
                            _tla_string(str(predecessor))
                            for predecessor in event["predecessors"]
                        )
                        + ">>",
                    )
                    for event in events
                ]
            ),
            "CrossRankEdgeIds == "
            + _tla_sequence([_tla_string(str(edge["edge_id"])) for edge in edges]),
            "CrossRankEndpoints == "
            + _tla_function(
                [
                    (
                        _tla_string(str(edge["edge_id"])),
                        "<<"
                        + ", ".join(
                            _tla_string(str(endpoint)) for endpoint in edge["endpoints"]
                        )
                        + ">>",
                    )
                    for edge in edges
                ]
            ),
            "CrossRankRelation == "
            + _tla_function(
                [
                    (
                        _tla_string(str(edge["edge_id"])),
                        _tla_string(str(edge["relation"])),
                    )
                    for edge in edges
                ]
            ),
            "CrossRankOrdered == "
            + _tla_function(
                [
                    (
                        _tla_string(str(edge["edge_id"])),
                        "TRUE" if edge["ordered"] else "FALSE",
                    )
                    for edge in edges
                ]
            ),
            "CrossRankEvidenceDigest == "
            + _tla_function(
                [
                    (
                        _tla_string(str(edge["edge_id"])),
                        _tla_string(str(edge["evidence_sha256"])),
                    )
                    for edge in edges
                ]
            ),
            "CollectiveWorkIds == "
            + _tla_sequence(
                [_tla_string(str(work["work_id"])) for work in collectives]
            ),
            "CollectiveRank == "
            + _tla_function(
                [
                    (_tla_string(str(work["work_id"])), str(work["rank"]))
                    for work in collectives
                ]
            ),
            "CollectiveId == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        _tla_string(str(work["collective_id"])),
                    )
                    for work in collectives
                ]
            ),
            # Communicator identity, operation and issue order. These are the
            # facts a protocol model needs and the previous export discarded,
            # leaving collectives opaque. The communicator key is canonical_id:
            # runtime_pg_id is per-rank local numbering and maps to two
            # different communicators, so it is exported only for cross-checking.
            "CollectiveComm == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        _tla_string(str(work["comm"])),
                    )
                    for work in collectives
                ]
            ),
            "CollectiveOperation == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        _tla_string(str(work["operation"])),
                    )
                    for work in collectives
                ]
            ),
            "CollectiveAxis == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        _tla_string(str(work["axis"])),
                    )
                    for work in collectives
                ]
            ),
            "CollectiveRuntimePgId == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        str(int(work["runtime_pg_id"])),
                    )
                    for work in collectives
                ]
            ),
            # Issue order: the order of this work's collective.enqueued event,
            # which inherits the NCCL Flight Recorder record_id sequence. The
            # started/completed sub-order is synthetic and is NOT exported as
            # ordering evidence.
            "CollectiveIssueOrder == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        str(int(work["issue_order"])),
                    )
                    for work in collectives
                ]
            ),
            # Opaque label only: stream identity is inferred by a positional zip
            # of Flight Recorder entries to Kineto kernels, not observed.
            r"\* INFERRED by positional zip, not observed. Values are opaque",
            r"\* Kineto resource labels, not verified CUDA stream identities.",
            "CollectiveStream == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        str(int(work["stream_id"])),
                    )
                    for work in collectives
                ]
            ),
            "CollectiveMembers == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        "{"
                        + ", ".join(str(member) for member in work["members"])
                        + "}",
                    )
                    for work in collectives
                ]
            ),
            "CollectiveLifecycle == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        "<<"
                        + ", ".join(
                            _tla_string(str(state)) for state in work["lifecycle"]
                        )
                        + ">>",
                    )
                    for work in collectives
                ]
            ),
            r"\* INFERRED by positional zip, not observed. A producer identity",
            r"\* is a digest of Kineto correlation fields attributed to this",
            r"\* collective by position; it is not an observed join key.",
            "CollectiveProducers == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        "<<"
                        + ", ".join(
                            _tla_string(str(producer)) for producer in work["producers"]
                        )
                        + ">>",
                    )
                    for work in collectives
                ]
            ),
            "CollectiveEventIds == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        "<<"
                        + ", ".join(
                            _tla_string(str(event_id)) for event_id in work["event_ids"]
                        )
                        + ">>",
                    )
                    for work in collectives
                ]
            ),
            "CollectiveHasExecutor == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        "TRUE" if work["has_executor"] else "FALSE",
                    )
                    for work in collectives
                ]
            ),
            "CollectiveHasStream == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        "TRUE" if work["has_stream"] else "FALSE",
                    )
                    for work in collectives
                ]
            ),
            "CollectiveObservationCount == "
            + _tla_function(
                [
                    (
                        _tla_string(str(work["work_id"])),
                        str(work["observation_count"]),
                    )
                    for work in collectives
                ]
            ),
            "",
            "=============================================================================",
            "",
        ]
    )
    return "\n".join(lines)


def export_scout_b_lean_facts(
    bundle: Mapping[str, object], *, namespace: str = "ScoutBFacts"
) -> str:
    """Export only observed DPxTP values for maintained Lean definitions."""

    return _export_scout_b_lean_module(_formal_projection(bundle), namespace=namespace)


def export_scout_b_lean_bad_facts(bundle: Mapping[str, object]) -> str:
    """Export the same controlled producer-correlation mutation for Lean."""

    facts = deepcopy(_formal_projection(bundle))
    facts["collectives"][0]["producers"][-1] = "controlled-invalid-producer"
    return _export_scout_b_lean_module(
        facts,
        namespace="ScoutBBadFacts",
        mutation="collective_producer_completion_mismatch",
    )


def _export_scout_b_lean_module(
    facts: Mapping[str, Any], *, namespace: str, mutation: str | None = None
) -> str:
    def event_reference(reference: Mapping[str, object]) -> str:
        return (
            "{ "
            f"eventId := {_lean_string(str(reference['event_id']))}, "
            f"identity := {reference['identity']}, index := {reference['index']}, "
            f"rank := {reference['rank']}, "
            f"order := {reference['order']}, "
            f"kind := {_lean_string(str(reference['kind']))} "
            "}"
        )

    def collective_reference(reference: Mapping[str, object]) -> str:
        return (
            "{ "
            f"workId := {_lean_string(str(reference['work_id']))}, "
            f"index := {reference['index']}, rank := {reference['rank']}, "
            f"rankOrder := {reference['rank_order']}, "
            f"collectiveId := {_lean_string(str(reference['collective_id']))} "
            "}"
        )

    def chunked_string_list(values: Sequence[str]) -> str:
        chunks = [
            "["
            + ", ".join(
                _lean_string(str(value)) for value in values[index : index + 64]
            )
            + "]"
            for index in range(0, len(values), 64)
        ]
        return " ++ ".join(chunks) if chunks else "[]"

    def append_string_grid(name: str, rows: Sequence[Sequence[str]]) -> None:
        rank_names = []
        for rank, row in enumerate(rows):
            rank_name = f"{name}Rank{rank}"
            rank_names.append(rank_name)
            lines.append(
                f"def {rank_name} : List String := " + chunked_string_list(row)
            )
        fields = ", ".join(
            f"rank{rank} := {rank_name}" for rank, rank_name in enumerate(rank_names)
        )
        lines.append(f"def {name} : RankStringRows := {{ {fields} }}")

    def append_event_evidence(name: str, events: Sequence[Mapping[str, Any]]) -> None:
        lines.extend(["", f"def {name} : List EventEvidence := ["])
        for event_index, event in enumerate(events):
            if event_index and event_index % 64 == 0:
                lines.append("] ++ [")
            predecessors = ", ".join(
                event_reference(predecessor)
                for predecessor in event["predecessor_references"]
            )
            lines.append(
                "  { "
                f"eventId := {_lean_string(str(event['event_id']))}, "
                f"identity := {event['identity']}, index := {event['index']}, "
                f"rank := {event['rank']}, order := {event['order']}, "
                f"kind := {_lean_string(str(event['kind']))}, "
                f"rawEventId := {_lean_string(str(event['raw_event_id']))}, "
                f"projectionSchema := {_lean_string(str(event['projection_schema']))}, "
                f"projectionDigest := {_lean_string(str(event['projection_sha256']))}, "
                f"predecessors := [{predecessors}] "
                "},"
            )
        lines.append("]")

    def append_cross_rank_edges(name: str, edges: Sequence[Mapping[str, Any]]) -> None:
        lines.extend(["", f"def {name} : List CrossRankSynchronization := ["])
        for edge in edges:
            endpoints = ", ".join(
                event_reference(endpoint) for endpoint in edge["endpoint_references"]
            )
            lines.append(
                "  { "
                f"edgeId := {_lean_string(str(edge['edge_id']))}, "
                f"endpoints := [{endpoints}], "
                f"relation := {_lean_string(str(edge['relation']))}, "
                f"ordered := {str(edge['ordered']).lower()}, "
                f"evidenceDigest := {_lean_string(str(edge['evidence_sha256']))} "
                "},"
            )
        lines.append("]")

    def append_collectives(name: str, collectives: Sequence[Mapping[str, Any]]) -> None:
        lines.extend(
            [
                "",
                "-- producers and hasStream below are INFERRED by positional",
                "-- zip, not observed. See the module header.",
                f"def {name} : List CollectiveObservation := [",
            ]
        )
        for work_index, work in enumerate(collectives):
            if work_index and work_index % 64 == 0:
                lines.append("] ++ [")
            members = ", ".join(str(member) for member in work["members"])
            lifecycle = ", ".join(f".{state}" for state in work["lifecycle"])
            producers = ", ".join(
                _lean_string(str(producer)) for producer in work["producers"]
            )
            event_ids = ", ".join(
                event_reference(event) for event in work["event_references"]
            )
            peer_references = ", ".join(
                collective_reference(peer) for peer in work["peer_references"]
            )
            lines.append(
                "  { "
                f"workId := {_lean_string(str(work['work_id']))}, "
                f"index := {work['index']}, rank := {work['rank']}, "
                f"rankOrder := {work['rank_order']}, "
                f"collectiveId := {_lean_string(str(work['collective_id']))}, "
                f"members := [{members}], lifecycle := [{lifecycle}], "
                f"producers := [{producers}], eventReferences := [{event_ids}], "
                f"peerReferences := [{peer_references}], "
                f"hasExecutor := {str(work['has_executor']).lower()}, "
                f"hasStream := {str(work['has_stream']).lower()}, "
                f"observationCount := {work['observation_count']} "
                "},"
            )
        lines.append("]")

    lines = [
        "import ScoutDistributed",
        "",
        f"namespace Qwen3Formal.{namespace}",
        "",
        "open Qwen3Formal",
        "",
        "set_option maxRecDepth 100000",
        "set_option maxHeartbeats 0",
        "",
        # Emitted into the artifact on purpose; see the matching block in the
        # TLA export for why a source comment was not enough.
        "-- INFERRED, NOT OBSERVED: the producers and hasStream fields of",
        "-- collectiveLifecycle come from a positional zip of NCCL Flight",
        "-- Recorder entries, ordered by record_id, to Kineto kernels, ordered",
        "-- by start time. There is no join key between the two sources; equal",
        "-- counts and equal order are the whole argument. The underlying",
        "-- Kineto resource id is an opaque label, not a verified CUDA stream",
        "-- identity. Ticket 09 replaces this with a joined key.",
        "",
        f"def traceId : String := {_lean_string(str(facts['trace_id']))}",
    ]
    if mutation is not None:
        lines.append(f"def mutationName : String := {_lean_string(mutation)}")
    lines.extend(
        [
            "",
            "def rawEventProjectionSchema : String := "
            + _lean_string(str(facts["projection_schema"])),
            "def eventIdPrefix : String := "
            + _lean_string(str(facts["event_id_prefix"])),
            f"def eventsPerRank : Nat := {facts['events_per_rank']}",
            f"def collectivesPerRank : Nat := {facts['collectives_per_rank']}",
        ]
    )
    append_string_grid("eventIdsByRank", facts["event_ids_by_rank"])
    append_string_grid("collectiveWorkIdsByRank", facts["collective_work_ids_by_rank"])
    append_string_grid("collectiveIdsByRank", facts["collective_ids_by_rank"])
    lines.extend(["", "def rankCoordinates : List RankCoordinate := ["])
    for rank in facts["rank_coordinates"]:
        lines.append(
            "  { "
            f"rank := {rank['rank']}, dp := {rank['dp']}, tp := {rank['tp']}, "
            f"inputDigest := {_lean_string(str(rank['input_sha256']))}, "
            f"hasDpShard := {str(rank['has_dp_shard']).lower()}, "
            f"hasTpShard := {str(rank['has_tp_shard']).lower()} "
            "},"
        )
    lines.append("]")

    # Structural per-parameter placements. Grouped by parameter here, while the
    # TLA export keys flat placement ids: the two checkers pay different costs,
    # and a grouped list keeps the Lean kernel evaluation linear instead of
    # filtering 148 placements once per parameter and tensor dim.
    #
    # OBSERVATION POINT: one snapshot per rank, taken after model construction
    # and before Trainer.train over model_parts[0].named_parameters(). Not
    # re-sampled inside the optimizer pre-hook, and no gradient placement is
    # observed.
    placements = facts["placement_facts"]
    lines.extend(
        [
            "",
            "def placementSchemaDigest : String := "
            + _lean_string(str(placements["schema_digest"])),
            "def placementSchemaDigestByRank : List String := ["
            + ", ".join(
                _lean_string(str(digest))
                for digest in placements["schema_digest_by_rank"]
            )
            + "]",
            "def meshAxisDegrees : List MeshAxisDegree := ["
            + ", ".join(
                "{ " + f"axis := {_lean_string(axis)}, degree := {int(degree)}" + " }"
                for axis, degree in sorted(placements["mesh_axis_degrees"].items())
            )
            + "]",
            "",
            "def parameterPlacements : List ParameterPlacements := [",
        ]
    )
    placement_by_id = {
        str(placement["placement_id"]): placement
        for placement in placements["placements"]
    }
    for parameter_index, parameter in enumerate(placements["parameters"]):
        if parameter_index and parameter_index % 64 == 0:
            lines.append("] ++ [")
        records: list[str] = []
        for placement_id in parameter["placement_ids"]:
            placement = placement_by_id[str(placement_id)]
            shard_dim = (
                f"some {int(placement['shard_dim'])}"
                if placement["kind"] == "shard"
                else "none"
            )
            records.append(
                "{ "
                f"axis := {_lean_string(str(placement['axis']))}, "
                f"kind := {_lean_string(str(placement['kind']))}, "
                f"strided := {str(placement['strided']).lower()}, "
                f"shardDim := {shard_dim} "
                "}"
            )
        lines.append(
            "  { "
            f"parameter := {_lean_string(str(parameter['name']))}, "
            "meshAxes := ["
            + ", ".join(_lean_string(str(axis)) for axis in parameter["mesh_axes"])
            + "], globalShape := ["
            + ", ".join(str(int(size)) for size in parameter["global_shape"])
            + "], localShape := ["
            + ", ".join(str(int(size)) for size in parameter["local_shape"])
            + "], placements := ["
            + ", ".join(records)
            + "] },"
        )
    lines.append("]")

    event_rank_names: list[str] = []
    for rank in range(4):
        name = f"eventEvidenceRank{rank}"
        event_rank_names.append(name)
        append_event_evidence(
            name,
            [event for event in facts["events"] if int(event["rank"]) == rank],
        )
    lines.extend(
        [
            "",
            "def eventEvidenceByRank : RankEventRows := { "
            + ", ".join(
                f"rank{rank} := {name}" for rank, name in enumerate(event_rank_names)
            )
            + " }",
            "def eventEvidence : List EventEvidence := "
            + " ++ ".join(event_rank_names),
        ]
    )

    edges = list(facts["cross_rank_edges"])
    edge_chunk_size = max(1, (len(edges) + 3) // 4)
    edge_chunk_names: list[str] = []
    for chunk_index in range(4):
        name = f"crossRankSynchronizationsChunk{chunk_index}"
        edge_chunk_names.append(name)
        start = chunk_index * edge_chunk_size
        append_cross_rank_edges(name, edges[start : start + edge_chunk_size])
    lines.extend(
        [
            "",
            "def crossRankSynchronizationChunks : SynchronizationChunks := { "
            + ", ".join(
                f"chunk{index} := {name}" for index, name in enumerate(edge_chunk_names)
            )
            + " }",
            "def crossRankSynchronizations : List CrossRankSynchronization := "
            + " ++ ".join(edge_chunk_names),
        ]
    )

    collective_rank_names: list[str] = []
    for rank in range(4):
        name = f"collectiveLifecycleRank{rank}"
        collective_rank_names.append(name)
        append_collectives(
            name,
            [work for work in facts["collectives"] if int(work["rank"]) == rank],
        )
    lines.extend(
        [
            "",
            "def collectiveLifecycleByRank : RankCollectiveRows := { "
            + ", ".join(
                f"rank{rank} := {name}"
                for rank, name in enumerate(collective_rank_names)
            )
            + " }",
            "def collectiveLifecycle : List CollectiveObservation := "
            + " ++ ".join(collective_rank_names),
            "",
            f"end Qwen3Formal.{namespace}",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_sync_mismatches(
    bundle: Mapping[str, object], formal_dir: str | Path
) -> tuple[str, ...]:
    """Return checked-in Scout B fact modules that disagree with evidence."""

    validate_normalized_bundle(bundle)
    formal_path = Path(formal_dir)
    expected = {
        formal_path / "ScoutBFacts.tla": export_scout_b_tla_facts(bundle),
        formal_path / "ScoutBBadFacts.tla": export_scout_b_tla_bad_facts(bundle),
        formal_path / "ScoutBFacts.lean": export_scout_b_lean_facts(bundle),
        formal_path / "ScoutBBadFacts.lean": export_scout_b_lean_bad_facts(bundle),
    }
    return tuple(
        str(path)
        for path, expected_text in expected.items()
        if not path.is_file() or path.read_text() != expected_text
    )


def sync_checked_in_artifacts(
    bundle: Mapping[str, object], formal_dir: str | Path
) -> None:
    """Write deterministic fact-only Scout B exports."""

    validate_normalized_bundle(bundle)
    formal_path = Path(formal_dir)
    formal_path.mkdir(parents=True, exist_ok=True)
    (formal_path / "ScoutBFacts.tla").write_text(export_scout_b_tla_facts(bundle))
    (formal_path / "ScoutBBadFacts.tla").write_text(
        export_scout_b_tla_bad_facts(bundle)
    )
    (formal_path / "ScoutBFacts.lean").write_text(export_scout_b_lean_facts(bundle))
    (formal_path / "ScoutBBadFacts.lean").write_text(
        export_scout_b_lean_bad_facts(bundle)
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
    """Record rank-local Trainer behavior without modifying the core loop."""

    def __init__(
        self,
        *,
        trainer: Trainer,
        run_id: str,
        attempt_id: str,
        profile: Mapping[str, object],
        config_digest: str,
        model_init_digest: str,
        process_groups: Sequence[Mapping[str, object]],
        tensor_placements: Sequence[Mapping[str, object]],
        coordinate: Sequence[int],
    ) -> None:
        self.trainer = trainer
        self.rank = dist.get_rank()
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.profile = dict(profile)
        self.config_digest = config_digest
        self.model_init_digest = model_init_digest
        self.process_groups = deepcopy(list(process_groups))
        self.tensor_placements = deepcopy(list(tensor_placements))
        self.coordinate = list(coordinate)
        self.events: list[dict[str, object]] = []
        self.first_batch: dict[str, object] | None = None
        self._handles: list[Any] = []
        self._last_core_index: int | None = None
        self._step_completed_index: int | None = None
        self._collective_completed_indices: list[int] = []
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
            raise ValueError("Scout B requires exactly one model part")
        model = self.trainer.model_parts[0]
        self._handles.extend(
            [
                model.register_forward_pre_hook(self._forward_pre_hook),
                model.register_forward_hook(self._forward_hook),
                model.register_full_backward_pre_hook(self._backward_pre_hook),
                model.register_full_backward_hook(self._backward_hook),
            ]
        )
        tracked_name, tracked_parameter = next(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self._handles.append(
            tracked_parameter.register_post_accumulate_grad_hook(
                lambda parameter: self._gradient_hook(tracked_name, parameter)
            )
        )
        if len(self.trainer.optimizers.optimizers) != 1:
            raise ValueError("Scout B requires exactly one built optimizer")
        optimizer = self.trainer.optimizers.optimizers[0]
        self._handles.append(optimizer.register_step_pre_hook(self._optimizer_pre_hook))
        self._handles.append(
            optimizer.register_step_post_hook(self._optimizer_post_hook)
        )

    def remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def emit_core(
        self, kind: str, phase: str, *, observation: Mapping[str, object]
    ) -> None:
        predecessors = [] if self._last_core_index is None else [self._last_core_index]
        event_index = self._append_event(
            kind,
            phase,
            observation=observation,
            predecessor_indices=predecessors,
        )
        self._last_core_index = event_index
        if kind == "step.completed":
            self._step_completed_index = event_index

    def _append_event(
        self,
        kind: str,
        phase: str,
        *,
        observation: Mapping[str, object],
        predecessor_indices: Sequence[int],
        observed_time_ns: int | None = None,
    ) -> int:
        self.events.append(
            {
                "observed_time_ns": (
                    time.perf_counter_ns()
                    if observed_time_ns is None
                    else observed_time_ns
                ),
                "kind": kind,
                "phase": phase,
                "step": 1,
                "predecessor_indices": list(predecessor_indices),
                "observation": deepcopy(dict(observation)),
            }
        )
        return len(self.events) - 1

    def observe_batch(self, batch: object) -> None:
        if self._batch_observed:
            return
        if (
            not isinstance(batch, (list, tuple))
            or len(batch) != 2
            or not isinstance(batch[0], dict)
            or not isinstance(batch[1], torch.Tensor)
        ):
            raise TypeError("Scout B dataloader yielded an unsupported batch")
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
        self.first_batch = {
            "batch_sha256": _sha256_json(tensor_identities),
            "tensors": tensor_identities,
        }
        self._batch_observed = True
        self.emit_core("batch.observed", "input", observation=self.first_batch)

    def _forward_pre_hook(self, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        del module
        self._forward_count += 1
        if self._forward_count != 1:
            raise RuntimeError("Scout B observed more than one core model forward")
        self.emit_core(
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
        self.emit_core(
            "forward.completed",
            "forward",
            observation={
                "hook": "Module.register_forward_hook",
                "output_shapes": [
                    list(tensor.shape) for tensor in _walk_tensors(output)
                ],
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
            raise RuntimeError("Scout B observed more than one core model backward")
        self.emit_core(
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
                "model backward completed before the tracked gradient was ready"
            )
        if self._gradient_observed:
            raise RuntimeError("Scout B observed gradient readiness more than once")
        self._gradient_observed = True
        self.emit_core(
            "gradient.ready",
            "backward",
            observation={
                "hook": "Parameter.register_post_accumulate_grad_hook",
                "parameter": self._gradient_name,
                "hook_calls": self._gradient_hook_count,
                "gradient": self._gradient_identity,
            },
        )
        self.emit_core(
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
        self.emit_core(
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
        if self._before_optimizer_digest == after_digest:
            raise RuntimeError("AdamW step did not mutate observed parameters")
        self.emit_core(
            "optimizer.mutated",
            "optimizer",
            observation={
                "hook": "Optimizer.register_step_post_hook",
                "optimizer": type(optimizer).__name__,
                "before_sha256": self._before_optimizer_digest,
                "after_sha256": after_digest,
                "mutated": True,
            },
        )

    def add_collective_observations(
        self, observations: Sequence[Mapping[str, object]]
    ) -> None:
        for observation in observations:
            prior_index: int | None = None
            times = _mapping(observation, "runtime_times_ns")
            for lifecycle, time_key in (
                ("enqueued", "enqueued"),
                ("started", "started"),
                ("completed", "completed"),
            ):
                value = deepcopy(dict(observation))
                value.pop("runtime_times_ns")
                value["lifecycle"] = lifecycle
                event_index = self._append_event(
                    f"collective.{lifecycle}",
                    "distributed",
                    observation=value,
                    predecessor_indices=([] if prior_index is None else [prior_index]),
                    observed_time_ns=_integer(times, time_key),
                )
                prior_index = event_index
            if prior_index is not None:
                self._collective_completed_indices.append(prior_index)

    def observe_bundle_collection(self, rank_raw_sha256: Mapping[str, object]) -> None:
        if self._step_completed_index is None:
            raise RuntimeError("bundle collection preceded step completion")
        predecessors = [
            self._step_completed_index,
            *self._collective_completed_indices,
        ]
        self._append_event(
            "bundle.collected",
            "collection",
            observation={
                "primitive": "torch.distributed.all_gather_object",
                "participants": [0, 1, 2, 3],
                "rank_raw_sha256": dict(rank_raw_sha256),
            },
            predecessor_indices=predecessors,
        )

    def raw_trace(self) -> dict[str, object]:
        if self.first_batch is None:
            raise RuntimeError("Scout B never observed its first batch")
        events = []
        for source_order, template in enumerate(self.events, start=1):
            raw_event_id = f"raw-r{self.rank}-{source_order:06d}"
            predecessor_indices = template["predecessor_indices"]
            if not isinstance(predecessor_indices, list):
                raise TypeError("observer predecessor indices must be a list")
            events.append(
                {
                    "raw_event_id": raw_event_id,
                    "source_order": source_order,
                    "observed_time_ns": template["observed_time_ns"],
                    "source_clock": {
                        "domain": "rank_observation_order",
                        "value": source_order,
                    },
                    "kind": template["kind"],
                    "phase": template["phase"],
                    "step": template["step"],
                    "causal_predecessors": [
                        f"raw-r{self.rank}-{index + 1:06d}"
                        for index in predecessor_indices
                    ],
                    "observation": deepcopy(template["observation"]),
                }
            )
        properties = torch.cuda.get_device_properties(self.trainer.device)
        return {
            "schema": RAW_SCHEMA,
            "identity": {
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "process_id": f"rank-{self.rank}",
                "rank": self.rank,
                "local_rank": int(os.environ["LOCAL_RANK"]),
                "world_size": dist.get_world_size(),
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
                "axes": list(MESH_AXES),
                "degrees": list(MESH_DEGREES),
                "coordinate": list(self.coordinate),
            },
            "process_groups": deepcopy(self.process_groups),
            "tensor_placements": deepcopy(self.tensor_placements),
            "profile": dict(self.profile),
            "lineage": {
                "config_sha256": self.config_digest,
                "model_identity": self.profile["model_identity"],
                "model_init_local_sha256": self.model_init_digest,
                "first_batch_sha256": self.first_batch["batch_sha256"],
            },
            "events": events,
        }


def _runtime_topology(
    trainer: Trainer,
) -> tuple[list[int], list[dict[str, object]], list[dict[str, object]]]:
    rank = dist.get_rank()
    meshes = trainer.parallel_dims.get_all_one_dimensional_meshes()
    required = {"fsdp", "tp"}
    if not required.issubset(meshes):
        raise ValueError(
            f"Scout B runtime meshes lack {sorted(required - set(meshes))}"
        )
    fsdp_mesh = meshes["fsdp"]
    tp_mesh = meshes["tp"]
    fsdp_coordinate = fsdp_mesh.get_coordinate()
    tp_coordinate = tp_mesh.get_coordinate()
    if fsdp_coordinate is None or tp_coordinate is None:
        raise ValueError("Scout B rank is absent from a required runtime mesh")
    coordinate = [0, int(fsdp_coordinate[0]), 0, int(tp_coordinate[0]), 0, 0]
    required_coordinate = [0, rank // 2, 0, rank % 2, 0, 0]
    if coordinate != required_coordinate:
        raise ValueError(
            f"Scout B runtime coordinate must be {required_coordinate}, found {coordinate}"
        )
    process_groups = []
    for runtime_axis, axis in (("fsdp", "dp_shard"), ("tp", "tp")):
        mesh = meshes[runtime_axis]
        group = mesh.get_group()
        runtime_coordinate = mesh.get_coordinate()
        if runtime_coordinate is None:
            raise ValueError(f"Scout B rank is absent from the {runtime_axis} mesh")
        process_groups.append(
            {
                "axis": axis,
                "runtime_axis": runtime_axis,
                "runtime_name": str(group.group_name),
                "members": dist.get_process_group_ranks(group),
                "backend": dist.get_backend(group),
                "local_coordinate": int(runtime_coordinate[0]),
            }
        )
    placements = []
    for name, parameter in sorted(trainer.model_parts[0].named_parameters()):
        if not isinstance(parameter, DTensor):
            raise TypeError(f"Scout B parameter is not a DTensor: {name}")
        mesh_axis_names = parameter.device_mesh.mesh_dim_names
        if mesh_axis_names is None:
            raise ValueError(f"Scout B parameter mesh has no named axes: {name}")
        mesh_axes = [
            "dp_shard" if axis == "fsdp" else str(axis) for axis in mesh_axis_names
        ]
        placement_records = []
        for axis, placement in zip(mesh_axes, parameter.placements, strict=True):
            if isinstance(placement, Shard):
                record: dict[str, object] = {
                    "axis": axis,
                    "kind": "shard",
                    "dim": int(placement.dim),
                }
            elif type(placement).__name__ == "_StridedShard":
                placement_dim = getattr(placement, "dim", None)
                if not isinstance(placement_dim, int):
                    raise TypeError(
                        f"Scout B parameter {name} has an invalid strided shard"
                    )
                record = {
                    "axis": axis,
                    "kind": "shard",
                    "dim": placement_dim,
                    "strided": True,
                }
            elif isinstance(placement, Replicate):
                record = {"axis": axis, "kind": "replicate"}
            elif isinstance(placement, Partial):
                record = {"axis": axis, "kind": "partial"}
            else:
                raise TypeError(
                    f"Scout B parameter {name} has unsupported placement {placement!r}"
                )
            placement_records.append(record)
        placements.append(
            {
                "name": name,
                "global_shape": list(parameter.shape),
                "local_shape": list(parameter.to_local().shape),
                "mesh_axes": mesh_axes,
                "placements": placement_records,
            }
        )
    return coordinate, process_groups, placements


def _operation_family(name: str) -> str | None:
    compact = re.sub(r"[^a-z]", "", name.lower())
    for token, operation in (
        ("reducescatter", "reduce_scatter"),
        ("allgather", "all_gather"),
        ("allreduce", "all_reduce"),
        ("broadcast", "broadcast"),
    ):
        if token in compact:
            return operation
    return None


def _flight_snapshot() -> dict[str, object]:
    torch_c: Any = torch._C
    distributed_c10d = vars(torch_c)["_distributed_c10d"]
    dump_nccl_trace_json = vars(distributed_c10d)["_dump_nccl_trace_json"]
    raw = dump_nccl_trace_json()
    snapshot = json.loads(raw)
    if not isinstance(snapshot, dict):
        raise TypeError("NCCL Flight Recorder snapshot must be an object")
    return snapshot


def _flight_record_ids(snapshot: Mapping[str, object]) -> set[int]:
    return {
        _integer(entry, "record_id") for entry in _mapping_list(snapshot, "entries")
    }


def _parse_flight_group(
    snapshot: Mapping[str, object], entry: Mapping[str, object]
) -> tuple[str, list[int]]:
    pg_config = _mapping(snapshot, "pg_config")
    pg_key = str(entry.get("pg_id"))
    config = pg_config.get(pg_key)
    if not isinstance(config, dict):
        process_group = entry.get("process_group")
        if isinstance(process_group, list) and process_group:
            config = pg_config.get(str(process_group[0]))
    if not isinstance(config, dict):
        raise ValueError(f"Flight Recorder entry lacks group config: {entry}")
    ranks_value = config.get("ranks")
    if not isinstance(ranks_value, str):
        raise TypeError("Flight Recorder group ranks must be JSON text")
    members = json.loads(ranks_value)
    if not isinstance(members, list) or not all(
        isinstance(member, int) for member in members
    ):
        raise TypeError("Flight Recorder group ranks must be integer ranks")
    description = str(config.get("desc", ""))
    return description, members


def _group_axis(members: Sequence[int]) -> str:
    member_tuple = tuple(members)
    if member_tuple in {(0, 1), (2, 3)}:
        return "tp"
    if member_tuple in {(0, 2), (1, 3)}:
        return "dp_shard"
    if member_tuple == (0, 1, 2, 3):
        return "world"
    raise ValueError(f"Scout B observed an unsupported process group: {members}")


def _collective_observations(
    profiler: Any,
    snapshot: Mapping[str, object],
    baseline_record_ids: set[int],
) -> list[dict[str, object]]:
    flight_entries = []
    for entry in _mapping_list(snapshot, "entries"):
        if _integer(entry, "record_id") in baseline_record_ids:
            continue
        if entry.get("is_p2p") is True:
            continue
        operation = _operation_family(str(entry.get("profiling_name", "")))
        if operation is None:
            raise ValueError(
                "Scout B observed an unsupported NCCL collective: "
                f"{entry.get('profiling_name')!r}"
            )
        if entry.get("state") != "completed":
            raise ValueError("Scout B collective did not reach completed state")
        for key in (
            "time_created_ns",
            "time_discovered_started_ns",
            "time_discovered_completed_ns",
        ):
            _integer(entry, key)
        flight_entries.append((operation, entry))
    if not flight_entries:
        raise ValueError("Scout B observed no step collectives")

    kernels_by_operation: dict[str, list[Any]] = defaultdict(list)
    profiler_impl = profiler.profiler
    if profiler_impl is None:
        raise RuntimeError("PyTorch profiler implementation is unavailable")
    kineto_results = profiler_impl.kineto_results
    if kineto_results is None:
        raise RuntimeError("Kineto results are unavailable")
    for event in kineto_results.events():
        if str(event.activity_type()) != "kernel":
            continue
        operation = _operation_family(event.name())
        if operation is not None and "nccl" in event.name().lower():
            kernels_by_operation[operation].append(event)
    for kernels in kernels_by_operation.values():
        kernels.sort(key=lambda event: event.start_ns())

    entries_by_operation: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for operation, entry in flight_entries:
        entries_by_operation[operation].append(entry)
    observations = []
    rank = dist.get_rank()
    for operation in sorted(entries_by_operation):
        entries = sorted(
            entries_by_operation[operation],
            key=lambda item: _integer(item, "record_id"),
        )
        kernels = kernels_by_operation.get(operation, [])
        if len(kernels) != len(entries):
            raise ValueError(
                f"Scout B {operation} Flight/Kineto count differs: "
                f"{len(entries)} != {len(kernels)}"
            )
        for entry, kernel in zip(entries, kernels, strict=True):
            description, members = _parse_flight_group(snapshot, entry)
            axis = _group_axis(members)
            description_key = re.sub(r"[^A-Za-z0-9._-]+", "-", description).strip("-")
            collective_id = (
                f"{axis}:{','.join(str(member) for member in members)}:"
                f"{description_key}:seq{_integer(entry, 'collective_seq_id')}:"
                f"{operation}"
            )
            work_id = f"work:r{rank}:{collective_id}"
            observations.append(
                {
                    "axis": axis,
                    "collective_id": collective_id,
                    "work_id": work_id,
                    "operation": operation,
                    "process_group": {
                        "canonical_id": (
                            f"{axis}:{','.join(str(member) for member in members)}:"
                            f"{description_key}"
                        ),
                        "runtime_pg_id": _integer(entry, "pg_id"),
                        "description": description,
                        "members": members,
                        "backend": "nccl",
                    },
                    "executor": {
                        "thread_id": str(entry.get("thread_id", "")),
                        "thread_name": str(entry.get("thread_name", "")),
                    },
                    "stream": {
                        "device_index": int(kernel.device_index()),
                        "resource_id": int(kernel.device_resource_id()),
                        "kernel": str(kernel.name()),
                    },
                    "producer": {
                        "correlation_id": str(kernel.correlation_id()),
                        "linked_correlation_id": str(kernel.linked_correlation_id()),
                        "external_id": str(kernel.external_id()),
                    },
                    "runtime_times_ns": {
                        "enqueued": _integer(entry, "time_created_ns"),
                        "started": _integer(entry, "time_discovered_started_ns"),
                        "completed": _integer(entry, "time_discovered_completed_ns"),
                    },
                    "flight_record_id": _integer(entry, "record_id"),
                }
            )
    return sorted(observations, key=lambda item: int(item["flight_record_id"]))


def run_scout_b(
    output_root: str | Path,
    *,
    run_id: str = SCOUT_B_RUN_ID,
    attempt_id: str = SCOUT_B_ATTEMPT_ID,
) -> ScoutBPaths:
    """Execute and observe the exact real four-rank DPxTP Trainer step."""

    _require_rootfs()
    _validate_identifier(run_id, "run_id")
    _validate_identifier(attempt_id, "attempt_id")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise RuntimeError("Scout B requires CUDA_VISIBLE_DEVICES=0,1,2,3")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Scout B requires exactly four visible CUDA devices")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank != rank:
        raise RuntimeError("Scout B must run as four local torchrun ranks")
    if int(os.environ.get("TORCH_NCCL_TRACE_BUFFER_SIZE", "0")) <= 0:
        raise RuntimeError("Scout B requires the NCCL Flight Recorder buffer")
    if os.environ.get("TORCH_NCCL_ENABLE_TIMING") != "1":
        raise RuntimeError("Scout B requires TORCH_NCCL_ENABLE_TIMING=1")

    attempt_dir = Path(output_root) / run_id / attempt_id
    config, cli_args = build_scout_b_config(attempt_dir / "trainer")
    profile = _scout_b_profile(config)
    lineage_cli = list(cli_args)
    lineage_cli[lineage_cli.index("--dump-folder") + 1] = "<attempt>/trainer"
    config_digest = _sha256_json({"cli": lineage_cli, "profile": profile})
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
            raise ValueError("Scout B must build one gradient accumulation step")
        torch.cuda.synchronize()
        baseline = _flight_snapshot()
        baseline_record_ids = _flight_record_ids(baseline)
        coordinate, process_groups, placements = _runtime_topology(trainer)
        observer = _RuntimeObserver(
            trainer=trainer,
            run_id=run_id,
            attempt_id=attempt_id,
            profile=profile,
            config_digest=config_digest,
            model_init_digest=_model_parameter_digest(trainer.model_parts),
            process_groups=process_groups,
            tensor_placements=placements,
            coordinate=coordinate,
        )
        observer.install()
        trainer_dynamic: Any = trainer
        trainer_dynamic.dataloader = _ObservedDataloader(trainer.dataloader, observer)
        observer.emit_core(
            "step.started",
            "step",
            observation={
                "owner": f"{Trainer.__module__}.{Trainer.__qualname__}",
                "method": "Trainer.train",
                "config_manager": (
                    f"{ConfigManager.__module__}.{ConfigManager.__qualname__}"
                ),
            },
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            trainer.train()
        torch.cuda.synchronize()
        observer.emit_core(
            "step.completed",
            "step",
            observation={
                "owner": f"{Trainer.__module__}.{Trainer.__qualname__}",
                "trainer_step": trainer.step,
                "optimizer_steps": 1,
            },
        )
        snapshot = _flight_snapshot()
        observer.add_collective_observations(
            _collective_observations(profiler, snapshot, baseline_record_ids)
        )
        preliminary = observer.raw_trace()
        gathered_preliminary: list[object] = [None] * 4
        dist.all_gather_object(gathered_preliminary, preliminary)
        rank_raw_sha256 = {
            str(index): _sha256_json(value)
            for index, value in enumerate(gathered_preliminary)
        }
        observer.observe_bundle_collection(rank_raw_sha256)
        raw_trace = observer.raw_trace()
        gathered: list[object] = [None] * 4
        dist.all_gather_object(gathered, raw_trace)
        if not all(isinstance(value, dict) for value in gathered):
            raise TypeError("Scout B rank collection returned a non-object trace")
        raw_traces = [value for value in gathered if isinstance(value, dict)]
        normalized = merge_rank_traces(raw_traces)
        paths = _write_attempt_bundle(
            attempt_dir,
            raw_traces,
            normalized,
            cli_args,
            writer_rank=rank,
        )
        dist.barrier()
        return paths
    finally:
        if observer is not None:
            observer.remove()
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _write_attempt_bundle(
    attempt_dir: Path,
    raw_traces: Sequence[Mapping[str, object]],
    normalized: Mapping[str, object],
    cli_args: Sequence[str],
    *,
    writer_rank: int,
) -> ScoutBPaths:
    if not stat.S_ISDIR(attempt_dir.lstat().st_mode):
        raise ValueError(f"attempt directory must be a real directory: {attempt_dir}")
    source_manifest_value = os.environ.get("QFV_SCOUT_SOURCE_MANIFEST", "")
    expected_head = os.environ.get("QFV_SCOUT_EXPECTED_HEAD", "")
    source_root = os.environ.get("QFV_SCOUT_SOURCE_ROOT", "")
    if not source_manifest_value or not expected_head or not source_root:
        raise RuntimeError(
            "Scout B requires QFV_SCOUT_SOURCE_MANIFEST, "
            "QFV_SCOUT_EXPECTED_HEAD, and QFV_SCOUT_SOURCE_ROOT"
        )
    source_manifest_path = Path(source_manifest_value)
    source_status_path = attempt_dir / "manifests" / "source-status.porcelain-v1-z"
    source = verify_source_manifest(
        source_manifest_path,
        source_status_path,
        repo_root=source_root,
        expected_head=expected_head,
        attempt_dir=attempt_dir,
    )
    raw_paths = tuple(
        attempt_dir / "raw" / "ranks" / f"rank-{rank:05d}.json" for rank in range(4)
    )
    normalized_path = attempt_dir / "normalized" / "scout_b.json"
    generated_dir = attempt_dir / "generated"
    tla_path = generated_dir / "ScoutBFacts.tla"
    tla_bad_path = generated_dir / "ScoutBBadFacts.tla"
    lean_path = generated_dir / "ScoutBFacts.lean"
    lean_bad_path = generated_dir / "ScoutBBadFacts.lean"
    runtime_path = attempt_dir / "manifests" / "runtime.json"
    paths = ScoutBPaths(
        attempt_dir=attempt_dir,
        rank_raw_traces=raw_paths,  # type: ignore[arg-type]
        normalized_bundle=normalized_path,
        generated_tla_facts=tla_path,
        generated_tla_bad_facts=tla_bad_path,
        generated_lean_facts=lean_path,
        generated_lean_bad_facts=lean_bad_path,
        runtime_manifest=runtime_path,
    )
    if writer_rank == 0:
        # Sealing side: rank 0 creates raw/ranks once so the parallel writers
        # below never race on the parent directory.
        _secure_managed_path(attempt_dir, raw_paths[0], create_parents=True)
    dist.barrier()
    own_raw_path = raw_paths[writer_rank]
    _write_immutable(
        own_raw_path,
        _canonical_json_text(raw_traces[writer_rank]),
        root=attempt_dir,
    )
    dist.barrier()
    if writer_rank != 0:
        return paths
    artifacts = {
        normalized_path: _canonical_json_text(normalized),
        tla_path: export_scout_b_tla_facts(normalized),
        tla_bad_path: export_scout_b_tla_bad_facts(normalized),
        lean_path: export_scout_b_lean_facts(normalized),
        lean_bad_path: export_scout_b_lean_bad_facts(normalized),
    }
    for path, contents in artifacts.items():
        _write_immutable(path, contents, root=attempt_dir)
    manifest_paths = [*raw_paths, *artifacts]
    manifest_files = {
        str(path.relative_to(attempt_dir)): _evidence_file_record(
            path, root=attempt_dir
        )
        for path in manifest_paths
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
        "config_manager_class": (
            f"{ConfigManager.__module__}.{ConfigManager.__qualname__}"
        ),
        "config_cli": list(cli_args),
        "rank_count": 4,
        "files": manifest_files,
    }
    manifest["runtime_id"] = f"sha256:{_sha256_json(manifest)}"
    _write_immutable(runtime_path, _canonical_json_text(manifest), root=attempt_dir)
    return paths


def _mapping(data: Mapping[str, object], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def _mapping_list(data: Mapping[str, object], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{key} must be a list of objects")
    return value


def _string_list(data: Mapping[str, object], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return value


def _integer_list(data: Mapping[str, object], key: str) -> list[int]:
    value = data.get(key)
    if (
        not isinstance(value, list)
        or not all(isinstance(item, int) for item in value)
        or any(isinstance(item, bool) for item in value)
    ):
        raise TypeError(f"{key} must be a list of integers")
    return value


def _integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _validate_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} has unsupported characters: {value!r}")


def _expected_trace_id(bundle: Mapping[str, object]) -> str:
    rank_traces = _mapping_list(bundle, "rank_traces")
    semantic_ranks = []
    for trace in rank_traces:
        semantic_events = []
        for event in _mapping_list(trace, "events"):
            event_fact = {
                "event_id": event.get("event_id"),
                "rank": event.get("rank"),
                "clock": deepcopy(_mapping(event, "clock")),
                "step": event.get("step"),
                "phase": event.get("phase"),
                "kind": event.get("kind"),
                "causal_predecessors": list(_string_list(event, "causal_predecessors")),
                "provenance": deepcopy(_mapping(event, "provenance")),
            }
            semantic_events.append(event_fact)
        semantic_ranks.append(
            {
                "process": deepcopy(_mapping(trace, "process")),
                "mesh": deepcopy(_mapping(trace, "mesh")),
                "placement_summary": deepcopy(_mapping(trace, "placement_summary")),
                "first_batch_sha256": _mapping(trace, "lineage").get(
                    "first_batch_sha256"
                ),
                "events": semantic_events,
            }
        )
    content = {
        "schema": bundle.get("schema"),
        "run_id": bundle.get("run_id"),
        "attempt_id": bundle.get("attempt_id"),
        "topology": deepcopy(_mapping(bundle, "topology")),
        "profile": deepcopy(_mapping(bundle, "profile")),
        "model_identity": _mapping(bundle, "lineage").get("model_identity"),
        "input_bundle": deepcopy(_mapping(bundle, "input_bundle")),
        "ordering": deepcopy(_mapping(bundle, "ordering")),
        "provenance_contract": deepcopy(_mapping(bundle, "provenance_contract")),
        "provenance": deepcopy(_mapping(bundle, "provenance")),
        "rank_semantics": semantic_ranks,
        "cross_rank_edges": deepcopy(_mapping_list(bundle, "cross_rank_edges")),
    }
    return f"sha256:{_sha256_json(content)}"


def _tla_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _tla_function(pairs: Sequence[tuple[str, str]]) -> str:
    if not pairs:
        return r"[x \in {} |-> x]"
    terms = [f"({key} :> {value})" for key, value in pairs]
    while len(terms) > 1:
        terms = [
            (
                f"({terms[index]} @@ {terms[index + 1]})"
                if index + 1 < len(terms)
                else terms[index]
            )
            for index in range(0, len(terms), 2)
        ]
    return terms[0]


def _tla_sequence(values: Sequence[str], *, chunk_size: int = 64) -> str:
    if not values:
        return "<<>>"
    chunks = [
        "<<" + ", ".join(values[index : index + chunk_size]) + ">>"
        for index in range(0, len(values), chunk_size)
    ]
    return " \\o ".join(chunks)


def _lean_string(value: str) -> str:
    return json.dumps(value)


def _managed_evidence_path(attempt: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        component in {"", ".", ".."} for component in candidate.parts
    ):
        raise ValueError(f"invalid evidence-relative path: {relative!r}")
    return attempt / candidate


def _runtime_expected_files() -> set[str]:
    return {
        *(f"raw/ranks/rank-{rank:05d}.json" for rank in range(4)),
        "normalized/scout_b.json",
        "generated/ScoutBFacts.tla",
        "generated/ScoutBBadFacts.tla",
        "generated/ScoutBFacts.lean",
        "generated/ScoutBBadFacts.lean",
        "manifests/source-status.porcelain-v1-z",
        "manifests/source.json",
    }


def _verify_normalized_provenance(
    raw_traces: Sequence[Mapping[str, object]],
    normalized: Mapping[str, object],
) -> None:
    """Recompute every projected digest while raw byte hashes stay manifest-owned."""

    normalized_traces = _mapping_list(normalized, "rank_traces")
    if len(raw_traces) != len(normalized_traces):
        raise ValueError("normalized provenance rank coverage is incomplete")
    expected_rank_digests: dict[str, str] = {}
    for rank, (raw_trace, normalized_trace) in enumerate(
        zip(raw_traces, normalized_traces, strict=True)
    ):
        raw_events = {
            str(event.get("raw_event_id", "")): event
            for event in _mapping_list(raw_trace, "events")
        }
        normalized_events = _mapping_list(normalized_trace, "events")
        referenced_raw_ids: set[str] = set()
        for event in normalized_events:
            provenance = _mapping(event, "provenance")
            raw_event_id = str(provenance.get("raw_event_id", ""))
            raw_event = raw_events.get(raw_event_id)
            if raw_event is None or provenance != _event_provenance(raw_event):
                raise ValueError(
                    "normalized event provenance disagrees with referenced raw event"
                )
            referenced_raw_ids.add(raw_event_id)
        if referenced_raw_ids != set(raw_events):
            raise ValueError(
                "normalized event provenance raw-event coverage is incomplete"
            )
        expected_rank_digests[str(rank)] = _sha256_json(
            _raw_trace_projection(raw_trace)
        )
    actual_rank_digests = _mapping(
        _mapping(normalized, "provenance"),
        "rank_raw_trace_projection_sha256",
    )
    if actual_rank_digests != expected_rank_digests:
        raise ValueError(
            "normalized rank projection provenance disagrees with raw traces"
        )


def verify_runtime_bundle(
    attempt_dir: str | Path,
    *,
    expected_head: str,
    source_root: str | Path = ".",
) -> dict[str, object]:
    """Rebuild Scout B from four immutable raw traces and verify every link."""

    _require_rootfs()
    attempt = Path(attempt_dir)
    runtime_path = attempt / "manifests" / "runtime.json"
    runtime = json.loads(_require_regular_readonly(runtime_path, root=attempt))
    if (
        not isinstance(runtime, dict)
        or runtime.get("schema") != RUNTIME_MANIFEST_SCHEMA
    ):
        raise ValueError("Scout B runtime manifest has an unsupported schema")
    _verify_content_id(runtime, "runtime_id", "Scout B runtime manifest")
    if runtime.get("rank_count") != 4:
        raise ValueError("Scout B runtime manifest must cover four ranks")
    if runtime.get("rootfs_sentinel") != "1":
        raise ValueError("Scout B runtime manifest lacks rootfs evidence")
    if runtime.get("cuda_visible_devices") != "0,1,2,3":
        raise ValueError("Scout B runtime manifest has wrong CUDA visibility")
    files = _mapping(runtime, "files")
    expected_files = _runtime_expected_files()
    if set(files) != expected_files:
        raise ValueError("Scout B runtime manifest files are incomplete")
    verified = {
        relative: _verify_file_record(attempt, relative, _mapping(files, relative))
        for relative in sorted(expected_files)
    }
    source = _mapping(runtime, "source")
    if source.get("head") != expected_head:
        raise ValueError("Scout B runtime source HEAD disagrees")
    source_manifest = verify_source_manifest(
        _managed_evidence_path(attempt, str(source.get("manifest_path", ""))),
        _managed_evidence_path(attempt, str(source.get("status_path", ""))),
        repo_root=source_root,
        expected_head=expected_head,
        attempt_dir=attempt,
    )
    if source.get("source_id") != source_manifest.get("source_id"):
        raise ValueError("Scout B runtime source identity disagrees")
    raw_traces = []
    for rank in range(4):
        value = json.loads(verified[f"raw/ranks/rank-{rank:05d}.json"])
        if not isinstance(value, dict):
            raise TypeError("Scout B raw trace must be an object")
        raw_traces.append(value)
    normalized = json.loads(verified["normalized/scout_b.json"])
    if not isinstance(normalized, dict):
        raise TypeError("Scout B normalized bundle must be an object")
    _verify_normalized_provenance(raw_traces, normalized)
    recomputed = merge_rank_traces(raw_traces)
    if normalized != recomputed:
        raise ValueError("Scout B normalized bundle disagrees with raw rank evidence")
    for key in ("run_id", "attempt_id", "trace_id"):
        if runtime.get(key) != normalized.get(key):
            raise ValueError(f"Scout B runtime {key} disagrees with normalized bundle")
    expected_generated = {
        "generated/ScoutBFacts.tla": export_scout_b_tla_facts(normalized).encode(),
        "generated/ScoutBBadFacts.tla": export_scout_b_tla_bad_facts(
            normalized
        ).encode(),
        "generated/ScoutBFacts.lean": export_scout_b_lean_facts(normalized).encode(),
        "generated/ScoutBBadFacts.lean": export_scout_b_lean_bad_facts(
            normalized
        ).encode(),
    }
    for relative, expected in expected_generated.items():
        if verified[relative] != expected:
            raise ValueError(f"Scout B generated runtime fact is stale: {relative}")
    return runtime


def _read_stage_journal(attempt: Path) -> list[dict[str, object]]:
    journal = _require_regular_readonly(
        attempt / "checker" / "stages.tsv", root=attempt
    ).decode("utf-8")
    lines = journal.splitlines()
    if len(lines) != len(SCOUT_B_REQUIRED_STAGES):
        raise ValueError("Scout B stage journal does not cover every required stage")
    stages = []
    for line, (required_name, required_log) in zip(
        lines, SCOUT_B_REQUIRED_STAGES, strict=True
    ):
        fields = line.split("\t", 3)
        if len(fields) != 4:
            raise ValueError("Scout B stage journal row must have four fields")
        name, status_text, evidence_path, command = fields
        try:
            exit_status = int(status_text)
        except ValueError:
            raise ValueError(f"Scout B stage {name} status is not an integer") from None
        if name != required_name or evidence_path != required_log:
            raise ValueError(
                f"Scout B stage expected {required_name}/{required_log}, "
                f"found {name}/{evidence_path}"
            )
        if exit_status != 0:
            raise ValueError(f"Scout B stage {name} failed")
        if not command or "\n" in command or "\t" in command:
            raise ValueError(f"Scout B stage {name} has an invalid command")
        # Content, not just mode: sealing accepted a zero-byte stage log until
        # this check existed, because the runner's "-s" guard lives outside the
        # sealed contract. See _require_non_empty_stage_log in scout_a.
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
    runtime: Mapping[str, object], stages: Sequence[Mapping[str, object]]
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
    """Seal the complete runtime and checker closure for Scout B."""

    _require_rootfs()
    attempt = Path(attempt_dir)
    runtime = verify_runtime_bundle(
        attempt, expected_head=expected_head, source_root=source_root
    )
    stages = _read_stage_journal(attempt)
    stage_manifest = _stage_manifest(runtime, stages)
    stage_path = attempt / "manifests" / "stages.json"
    _write_immutable(stage_path, _canonical_json_text(stage_manifest), root=attempt)
    relative_files = set(_mapping(runtime, "files"))
    relative_files.update(
        {
            "manifests/runtime.json",
            "manifests/stages.json",
            "checker/stages.tsv",
            *(relative for _, relative in SCOUT_B_REQUIRED_STAGES),
        }
    )
    files = {
        relative: _evidence_file_record(
            _managed_evidence_path(attempt, relative), root=attempt
        )
        for relative in sorted(relative_files)
    }
    manifest: dict[str, object] = {
        "schema": EVIDENCE_MANIFEST_SCHEMA,
        "evidence_id": "",
        "run_id": runtime["run_id"],
        "attempt_id": runtime["attempt_id"],
        "trace_id": runtime["trace_id"],
        "runtime_id": runtime["runtime_id"],
        "stage_id": stage_manifest["stage_id"],
        "source": deepcopy(_mapping(runtime, "source")),
        "files": files,
    }
    manifest["evidence_id"] = f"sha256:{_sha256_json(manifest)}"
    output = attempt / "manifests" / "evidence.json"
    _write_immutable(output, _canonical_json_text(manifest), root=attempt)
    verify_evidence_bundle(
        attempt, expected_head=expected_head, source_root=source_root
    )
    return output


def verify_evidence_bundle(
    attempt_dir: str | Path,
    *,
    expected_head: str,
    source_root: str | Path = ".",
) -> dict[str, object]:
    """Verify the complete immutable Scout B runtime and checker bundle."""

    _require_rootfs()
    attempt = Path(attempt_dir)
    evidence = json.loads(
        _require_regular_readonly(attempt / "manifests" / "evidence.json", root=attempt)
    )
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema") != EVIDENCE_MANIFEST_SCHEMA
    ):
        raise ValueError("Scout B evidence manifest has an unsupported schema")
    _verify_content_id(evidence, "evidence_id", "Scout B evidence manifest")
    runtime = verify_runtime_bundle(
        attempt, expected_head=expected_head, source_root=source_root
    )
    stages = json.loads(
        _require_regular_readonly(attempt / "manifests" / "stages.json", root=attempt)
    )
    if not isinstance(stages, dict) or stages.get("schema") != STAGE_MANIFEST_SCHEMA:
        raise ValueError("Scout B stage manifest has an unsupported schema")
    _verify_content_id(stages, "stage_id", "Scout B stage manifest")
    if stages != _stage_manifest(runtime, _read_stage_journal(attempt)):
        raise ValueError("Scout B stage manifest disagrees with journal")
    for key in ("run_id", "attempt_id", "trace_id", "runtime_id"):
        if evidence.get(key) != runtime.get(key):
            raise ValueError(f"Scout B evidence {key} disagrees with runtime")
    if evidence.get("stage_id") != stages.get("stage_id"):
        raise ValueError("Scout B evidence stage identity disagrees")
    if evidence.get("source") != runtime.get("source"):
        raise ValueError("Scout B evidence source identity disagrees")
    files = _mapping(evidence, "files")
    expected_files = set(_mapping(runtime, "files"))
    expected_files.update(
        {
            "manifests/runtime.json",
            "manifests/stages.json",
            "checker/stages.tsv",
            *(relative for _, relative in SCOUT_B_REQUIRED_STAGES),
        }
    )
    if set(files) != expected_files:
        raise ValueError("Scout B evidence manifest does not cover its bundle")
    for relative in sorted(expected_files):
        _verify_file_record(attempt, relative, _mapping(files, relative))
    discovered = _bundle_regular_files(attempt)
    if discovered != expected_files | {"manifests/evidence.json"}:
        raise ValueError("Scout B evidence bundle contains unmanaged or missing files")
    return evidence


def _default_formal_dir() -> Path:
    return Path("experiments/qwen3_formal_verifier/formal")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute the real Scout B step")
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--run-id", default=SCOUT_B_RUN_ID)
    run_parser.add_argument("--attempt-id", default=SCOUT_B_ATTEMPT_ID)

    sync_parser = subparsers.add_parser(
        "sync", help="check or write checked-in Scout B facts"
    )
    sync_parser.add_argument("--normalized", required=True)
    sync_parser.add_argument("--formal-dir", default=str(_default_formal_dir()))
    sync_mode = sync_parser.add_mutually_exclusive_group(required=True)
    sync_mode.add_argument("--check", action="store_true")
    sync_mode.add_argument("--write", action="store_true")

    source_parser = subparsers.add_parser(
        "source", help="create an immutable source identity manifest"
    )
    source_parser.add_argument("--status-file", required=True)
    source_parser.add_argument("--output", required=True)
    source_parser.add_argument("--repo-root", required=True)
    source_parser.add_argument("--head", required=True)
    source_parser.add_argument("--attempt-dir", required=True)

    finalize_parser = subparsers.add_parser(
        "finalize", help="seal checker logs into Scout B evidence"
    )
    finalize_parser.add_argument("--attempt-dir", required=True)
    finalize_parser.add_argument("--expected-head", required=True)
    finalize_parser.add_argument("--source-root", default=".")

    verify_parser = subparsers.add_parser(
        "verify", help="verify complete sealed Scout B evidence"
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
        paths = run_scout_b(
            args.output_root, run_id=args.run_id, attempt_id=args.attempt_id
        )
        if int(os.environ.get("RANK", "-1")) == 0:
            print(f"SCOUT_B_RUNTIME result=success attempt_dir={paths.attempt_dir}")
            print(f"SCOUT_B_NORMALIZED path={paths.normalized_bundle}")
            print(f"SCOUT_B_RUNTIME_MANIFEST path={paths.runtime_manifest}")
        return
    if args.command == "sync":
        _require_rootfs()
        bundle = json.loads(Path(args.normalized).read_text())
        if args.write:
            sync_checked_in_artifacts(bundle, args.formal_dir)
        mismatches = artifact_sync_mismatches(bundle, args.formal_dir)
        if mismatches:
            raise SystemExit(
                "Scout B checked-in artifacts are stale: " + ", ".join(mismatches)
            )
        print("SCOUT_B_ARTIFACT_SYNC result=success")
        return
    if args.command == "source":
        output = write_source_manifest(
            status_path=args.status_file,
            output_path=args.output,
            repo_root=args.repo_root,
            head=args.head,
            attempt_dir=args.attempt_dir,
        )
        print(f"SCOUT_B_SOURCE_MANIFEST result=created path={output}")
        return
    if args.command == "finalize":
        output = finalize_evidence_manifest(
            args.attempt_dir,
            expected_head=args.expected_head,
            source_root=args.source_root,
        )
        print(f"SCOUT_B_EVIDENCE result=sealed path={output}")
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
    print("SCOUT_B_EVIDENCE result=verified " f"evidence_id={evidence['evidence_id']}")


if __name__ == "__main__":
    main()
