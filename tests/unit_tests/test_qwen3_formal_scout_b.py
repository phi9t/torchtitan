# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for the observed four-rank DPxTP formal-verifier scout."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from torchtitan.experiments.qwen3_formal_verifier.scout_b import (
    _canonical_json_text,
    _evidence_file_record,
    _expected_trace_id,
    _formal_projection,
    _operation_family,
    _PAYLOAD_SIZE_RELATIONS,
    _sha256_json,
    _write_attempt_bundle,
    _write_immutable,
    artifact_sync_diagnosis,
    artifact_sync_mismatches,
    build_scout_b_config,
    export_scout_b_lean_bad_facts,
    export_scout_b_lean_facts,
    export_scout_b_tla_bad_facts,
    export_scout_b_tla_facts,
    finalize_evidence_manifest,
    merge_rank_traces,
    scout_b_cli_args,
    SCOUT_B_REQUIRED_STAGES,
    sync_checked_in_artifacts,
    validate_normalized_bundle,
    verify_evidence_bundle,
    verify_runtime_bundle,
    write_source_manifest,
)


AXES = ["dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "run_scout_b.sh"


def _raw_event(
    rank: int,
    order: int,
    kind: str,
    phase: str,
    observation: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "raw_event_id": f"raw-r{rank}-{order:06d}",
        "source_order": order,
        "observed_time_ns": rank * 1_000_000 + order * 1_000,
        "kind": kind,
        "phase": phase,
        "step": 1,
        "causal_predecessors": ([] if order == 1 else [f"raw-r{rank}-{order - 1:06d}"]),
        "observation": {} if observation is None else observation,
    }


def _collective_events(
    *,
    rank: int,
    start: int,
    axis: str,
    members: list[int],
    seq: int = 1,
    operation: str = "all_reduce",
    input_sizes: list[list[int]] | None = None,
    output_sizes: list[list[int]] | None = None,
    input_dtypes: list[str] | None = None,
    output_dtypes: list[str] | None = None,
) -> list[dict[str, object]]:
    # Identity shaped as the collector builds it: the canonical communicator id
    # is a prefix of the collective id, which is what lets a checker tell a
    # canonical key from any other string.
    canonical_id = f"{axis}:{','.join(str(member) for member in members)}"
    collective_id = f"{canonical_id}:seq{seq}:{operation}"
    producer = f"kineto-r{rank}-{collective_id}"
    work_id = f"work-r{rank}-{collective_id}"
    # Payload identity, shaped as the NCCL Flight Recorder reports it: per-tensor
    # shape lists with parallel per-tensor dtype names. The default is an
    # all_reduce, whose input and output volumes are equal.
    common: dict[str, object] = {
        "axis": axis,
        "collective_id": collective_id,
        "work_id": work_id,
        "operation": operation,
        "payload": {
            "input_sizes": [[128, 256]] if input_sizes is None else input_sizes,
            "output_sizes": [[128, 256]] if output_sizes is None else output_sizes,
            "input_dtypes": ["BFloat16"] if input_dtypes is None else input_dtypes,
            "output_dtypes": ["BFloat16"] if output_dtypes is None else output_dtypes,
        },
        "process_group": {
            "canonical_id": canonical_id,
            "members": members,
            "backend": "nccl",
            # Per-rank local pg numbering. Distinct communicators may share a
            # runtime id across ranks, which is why canonical_id is the key.
            "runtime_pg_id": 1 if axis == "tp" else 2,
            "description": f"mesh_{axis}",
        },
        "executor": {"thread_id": f"thread-r{rank}", "thread_name": "python3"},
        "stream": {
            "device_index": rank,
            "resource_id": 17,
            "kernel": "ncclDevKernel_AllReduce",
        },
        "producer": {
            "correlation_id": producer,
            "linked_correlation_id": f"launch-r{rank}-{collective_id}",
        },
    }
    events = []
    for offset, lifecycle in enumerate(("enqueued", "started", "completed")):
        observation = deepcopy(common)
        observation["lifecycle"] = lifecycle
        events.append(
            _raw_event(
                rank,
                start + offset,
                f"collective.{lifecycle}",
                "distributed",
                observation,
            )
        )
    return events


def _rank_trace(rank: int) -> dict[str, Any]:
    dp = rank // 2
    tp = rank % 2
    tp_members = [dp * 2, dp * 2 + 1]
    dp_members = [tp, tp + 2]
    batch_sha = "a" * 64 if dp == 0 else "b" * 64
    events = [
        _raw_event(rank, 1, "step.started", "step"),
        _raw_event(
            rank,
            2,
            "batch.observed",
            "input",
            {"batch_sha256": batch_sha},
        ),
        _raw_event(rank, 3, "forward.started", "forward"),
        *_collective_events(
            rank=rank,
            start=4,
            axis="tp",
            members=tp_members,
        ),
        _raw_event(rank, 7, "forward.completed", "forward"),
        _raw_event(rank, 8, "backward.started", "backward"),
        # A reduce_scatter, so the fixture carries a collective whose input
        # volume is its member count times its output volume rather than only
        # volume-preserving all_reduces. The payload size relation is vacuous
        # over a fixture where every operation preserves volume.
        *_collective_events(
            rank=rank,
            start=9,
            axis="dp_shard",
            members=dp_members,
            operation="reduce_scatter",
            input_sizes=[[256, 256]],
            output_sizes=[[128, 256]],
        ),
        _raw_event(rank, 12, "gradient.ready", "backward"),
        _raw_event(rank, 13, "backward.completed", "backward"),
        _raw_event(rank, 14, "optimizer.started", "optimizer"),
        _raw_event(
            rank,
            15,
            "optimizer.mutated",
            "optimizer",
            {
                "mutated": True,
                "before_sha256": "c" * 64,
                "after_sha256": "d" * 64,
            },
        ),
        _raw_event(rank, 16, "step.completed", "step"),
        _raw_event(
            rank,
            17,
            "bundle.collected",
            "collection",
            {
                "primitive": "torch.distributed.all_gather_object",
                "participants": [0, 1, 2, 3],
            },
        ),
    ]
    last_core_id: str | None = None
    last_work_id: dict[str, str] = {}
    completed_work_ids: list[str] = []
    for event in events:
        raw_event_id = str(event["raw_event_id"])
        kind = str(event["kind"])
        if kind.startswith("collective."):
            observation = event["observation"]
            assert isinstance(observation, dict)
            work_id = str(observation["work_id"])
            lifecycle = str(observation["lifecycle"])
            event["causal_predecessors"] = (
                [] if lifecycle == "enqueued" else [last_work_id[work_id]]
            )
            last_work_id[work_id] = raw_event_id
            if lifecycle == "completed":
                completed_work_ids.append(raw_event_id)
        elif kind == "bundle.collected":
            assert last_core_id is not None
            event["causal_predecessors"] = [last_core_id, *completed_work_ids]
        else:
            event["causal_predecessors"] = (
                [] if last_core_id is None else [last_core_id]
            )
            last_core_id = raw_event_id
    return {
        "schema": "qwen3.formal.raw.v1",
        "identity": {
            "run_id": "scout-b-test",
            "attempt_id": "attempt-0",
            "process_id": f"rank-{rank}",
            "rank": rank,
            "local_rank": rank,
            "world_size": 4,
        },
        "device": {
            "kind": "cuda",
            "logical_id": f"cuda:{rank}",
            "visible_index": rank,
            "name": "NVIDIA B200",
        },
        "mesh": {
            "axes": AXES,
            "degrees": [1, 2, 1, 2, 1, 1],
            "coordinate": [0, dp, 0, tp, 0, 0],
        },
        "process_groups": [
            {
                "axis": "dp_shard",
                "runtime_name": f"dp-{tp}",
                "members": dp_members,
                "backend": "nccl",
                "local_coordinate": dp,
            },
            {
                "axis": "tp",
                "runtime_name": f"tp-{dp}",
                "members": tp_members,
                "backend": "nccl",
                "local_coordinate": tp,
            },
        ],
        "tensor_placements": [
            {
                "name": "tok_embeddings.weight",
                "global_shape": [2048, 256],
                "local_shape": [1024, 128],
                "mesh_axes": ["dp_shard", "tp"],
                "placements": [
                    {"axis": "dp_shard", "kind": "shard", "dim": 0},
                    {"axis": "tp", "kind": "shard", "dim": 0},
                ],
            },
            {
                "name": "norm.weight",
                "global_shape": [256],
                "local_shape": [128],
                "mesh_axes": ["dp_shard", "tp"],
                "placements": [
                    {"axis": "dp_shard", "kind": "shard", "dim": 0},
                    {"axis": "tp", "kind": "replicate"},
                ],
            },
        ],
        "profile": {
            "model_identity": "qwen3/debugmodel",
            "local_batch_size": 1,
            "global_batch_size": 2,
            "sequence_length": 128,
            "tokens_per_optimizer_step": 256,
            "gradient_accumulation_steps": 1,
            "optimizer_steps": 1,
            "seed": 42,
            "deterministic": True,
        },
        "lineage": {
            "config_sha256": "e" * 64,
            "model_identity": "qwen3/debugmodel",
            "model_init_local_sha256": f"{rank + 1:064x}",
            "first_batch_sha256": batch_sha,
        },
        "events": events,
    }


def _rank_bundle() -> list[dict[str, Any]]:
    return [_rank_trace(rank) for rank in range(4)]


def _create_sealable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_bytes: bytes = b"",
    head_commit_paths: bytes = b"",
    source_files: dict[str, str] | None = None,
    empty_stage_log: str | None = None,
) -> Path:
    """Seal a bundle, optionally over a source tree with real status entries.

    ``source_files`` maps repository-relative paths to contents and must agree
    with ``status_bytes``; both default to an empty source tree.

    ``empty_stage_log`` names one required stage whose log is written zero-byte,
    which is what a stage that aborted before emitting anything leaves behind.
    """

    for relative_path, contents in (source_files or {}).items():
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    attempt = tmp_path / "scout-b-test" / "attempt-0"
    (attempt / "manifests").mkdir(parents=True)
    status_path = attempt / "manifests" / "source-status.porcelain-v1-z"
    _write_immutable(status_path, status_bytes, root=attempt)
    head_paths_path = attempt / "manifests" / "head-commit-paths.name-only-z"
    _write_immutable(head_paths_path, head_commit_paths, root=attempt)
    source_path = attempt / "manifests" / "source.json"
    write_source_manifest(
        status_path=status_path,
        head_paths_path=head_paths_path,
        output_path=source_path,
        repo_root=tmp_path,
        head="a" * 40,
        attempt_dir=attempt,
    )
    monkeypatch.setenv("QFV_SCOUT_SOURCE_MANIFEST", str(source_path))
    monkeypatch.setenv("QFV_SCOUT_EXPECTED_HEAD", "a" * 40)
    monkeypatch.setenv("QFV_SCOUT_SOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(
        "torchtitan.experiments.qwen3_formal_verifier.scout_b.dist.barrier",
        lambda: None,
    )
    raw_traces = _rank_bundle()
    normalized = merge_rank_traces(raw_traces)
    for writer_rank in (1, 2, 3, 0):
        _write_attempt_bundle(
            attempt,
            raw_traces,
            normalized,
            ["--fixture"],
            writer_rank=writer_rank,
        )
    checker = attempt / "checker"
    checker.mkdir()
    journal_lines: list[str] = []
    for stage_name, relative_log in SCOUT_B_REQUIRED_STAGES:
        _write_immutable(
            attempt / relative_log,
            "" if stage_name == empty_stage_log else f"{stage_name}=ok\n",
            root=attempt,
        )
        journal_lines.append(f"{stage_name}\t0\t{relative_log}\tcommand:{stage_name}\n")
    _write_immutable(
        checker / "stages.tsv",
        "".join(journal_lines),
        root=attempt,
    )
    return attempt


def test_scout_b_profile_uses_the_fixed_qwen3_debugmodel_topology(
    tmp_path: Path,
) -> None:
    config, cli_args = build_scout_b_config(tmp_path / "trainer")

    assert cli_args == scout_b_cli_args(tmp_path / "trainer")
    assert config.model_spec is not None
    assert (config.model_spec.name, config.model_spec.flavor) == (
        "qwen3",
        "debugmodel",
    )
    assert config.training.local_batch_size == 1
    assert config.training.global_batch_size == 2
    assert config.training.seq_len == 128
    assert config.training.steps == 1
    assert config.training.dtype == "bfloat16"
    assert [group.optimizer_name for group in config.optimizer.param_groups] == [
        "AdamW"
    ]
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.parallelism.data_parallel_shard_degree == 2
    assert config.parallelism.tensor_parallel_degree == 2
    assert config.parallelism.context_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.parallelism.expert_parallel_degree == 1
    assert config.parallelism.enable_sequence_parallel is False
    assert config.debug.seed == 42
    assert config.debug.deterministic is True
    assert config.debug.deterministic_warn_only is False
    assert config.dataloader.dataset == "c4_test"
    assert config.hf_assets_path == "./tests/assets/tokenizer"
    assert config.checkpoint.enable is False


def test_rank_bundle_merge_is_deterministic_without_cross_rank_total_order() -> None:
    ranks = _rank_bundle()

    first = merge_rank_traces(ranks)
    second = merge_rank_traces(deepcopy(ranks))

    assert first == second
    assert first["schema"] == "qwen3.formal.scout.v0"
    assert [trace["process"]["rank"] for trace in first["rank_traces"]] == [
        0,
        1,
        2,
        3,
    ]
    for rank_trace in first["rank_traces"]:
        orders = [event["clock"]["order"] for event in rank_trace["events"]]
        assert orders == list(range(1, len(orders) + 1))
        assert all(event["provenance"] for event in rank_trace["events"])
    assert first["ordering"] == {
        "per_rank": "observed_source_order",
        "cross_rank_total_order": False,
    }
    assert first["cross_rank_edges"]
    assert all(
        edge["relation"] == "synchronizes" and edge["ordered"] is False
        for edge in first["cross_rank_edges"]
    )
    assert first["input_bundle"]["dp_keys"] == ["dp:0", "dp:1"]
    assert first["input_bundle"]["by_dp_coordinate"] == {
        "dp:0": {"batch_sha256": "a" * 64, "ranks": [0, 1]},
        "dp:1": {"batch_sha256": "b" * 64, "ranks": [2, 3]},
    }
    assert first["input_bundle"]["global_sha256"]
    assert first["provenance_contract"] == {
        # Versioned so a later shape change is distinguishable from tampering
        # rather than failing every older bundle with one ambiguous message.
        "version": "qwen3.formal.scout.provenance-contract.v2",
        "normalized_projection": {
            "schema": "qwen3.formal.raw-event-projection.v0",
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
        # Payload identity is the one collective fact that is OBSERVED rather
        # than attributed by position, and the contract has to say which it is:
        # a reader who cannot tell them apart would treat the inferred producer
        # identity and the recorded tensor shapes as evidence of the same kind.
        "observed_payload": {
            "inferred": False,
            "observed": True,
            "source": "nccl_flight_recorder_entry",
            "per_tensor": True,
            "fields": [
                "observation.payload.input_sizes for collective.*",
                "observation.payload.output_sizes for collective.*",
                "observation.payload.input_dtypes for collective.*",
                "observation.payload.output_dtypes for collective.*",
            ],
            "shape": (
                "input_sizes/output_sizes are per-tensor shape lists; "
                "input_dtypes/output_dtypes are the parallel per-tensor dtype "
                "names; an empty shape list is a 0-dim tensor of one element"
            ),
        },
        # Producer and stream attribution is inferred by a positional zip, and
        # the sealed bundle has to say so: the exported names read exactly like
        # an observed join key, and before this block "grep -ril infer" over
        # the whole sealed bundle returned nothing.
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
                "opaque Kineto resource label, not a verified CUDA stream " "identity"
            ),
            "note": (
                "producer and stream attribution is INFERRED by position, not "
                "observed; ticket 09 replaces the positional zip with a joined "
                "key"
            ),
        },
    }
    validate_normalized_bundle(first)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_rank", "exactly ranks"),
        ("duplicate_rank", "duplicate rank"),
        ("run", "run/attempt"),
        ("attempt", "run/attempt"),
        ("topology", "topology"),
        ("model", "model identity"),
        ("config", "config identity"),
        ("tp_input", "TP peers"),
    ],
)
def test_rank_bundle_rejects_incomplete_or_conflicting_identity(
    mutation: str,
    message: str,
) -> None:
    ranks = _rank_bundle()
    if mutation == "missing_rank":
        ranks.pop()
    elif mutation == "duplicate_rank":
        ranks[3]["identity"]["rank"] = 2
        ranks[3]["identity"]["process_id"] = "rank-2-copy"
    elif mutation == "run":
        ranks[3]["identity"]["run_id"] = "other-run"
    elif mutation == "attempt":
        ranks[3]["identity"]["attempt_id"] = "other-attempt"
    elif mutation == "topology":
        ranks[3]["mesh"]["degrees"] = [1, 1, 1, 4, 1, 1]
    elif mutation == "model":
        ranks[3]["lineage"]["model_identity"] = "other/model"
    elif mutation == "config":
        ranks[3]["lineage"]["config_sha256"] = "f" * 64
    else:
        ranks[1]["lineage"]["first_batch_sha256"] = "f" * 64
        ranks[1]["events"][1]["observation"]["batch_sha256"] = "f" * 64

    with pytest.raises(ValueError, match=message):
        merge_rank_traces(ranks)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mesh_membership", "process group"),
        ("placement", "tensor placement"),
        ("producer", "producer correlation"),
        ("lifecycle", "collective lifecycle"),
    ],
)
def test_rank_bundle_rejects_one_dimension_collective_corruption(
    mutation: str,
    message: str,
) -> None:
    ranks = _rank_bundle()
    if mutation == "mesh_membership":
        ranks[0]["process_groups"][1]["members"] = [0, 2]
    elif mutation == "placement":
        ranks[0]["tensor_placements"][0]["placements"][1] = {
            "axis": "tp",
            "kind": "replicate",
        }
        ranks[0]["tensor_placements"][1]["placements"][1] = {
            "axis": "tp",
            "kind": "replicate",
        }
    elif mutation == "producer":
        ranks[0]["events"][5]["observation"]["producer"][
            "correlation_id"
        ] = "other-producer"
    else:
        ranks[0]["events"][5]["observation"]["lifecycle"] = "started"

    with pytest.raises(ValueError, match=message):
        merge_rank_traces(ranks)


def test_normalized_bundle_rejects_cross_rank_happens_before_claim() -> None:
    normalized = merge_rank_traces(_rank_bundle())
    normalized["cross_rank_edges"][0]["relation"] = "happens_before"
    normalized["cross_rank_edges"][0]["ordered"] = True

    with pytest.raises(ValueError, match="cross-rank total order"):
        validate_normalized_bundle(normalized)


def test_scout_b_exports_and_artifact_sync_cover_complete_normalized_graph(
    tmp_path: Path,
) -> None:
    normalized = merge_rank_traces(_rank_bundle())

    exports = {
        "ScoutBFacts.tla": export_scout_b_tla_facts(normalized),
        "ScoutBBadFacts.tla": export_scout_b_tla_bad_facts(normalized),
        "ScoutBFacts.lean": export_scout_b_lean_facts(normalized),
        "ScoutBBadFacts.lean": export_scout_b_lean_bad_facts(normalized),
    }
    sync_checked_in_artifacts(normalized, tmp_path)

    assert artifact_sync_mismatches(normalized, tmp_path) == ()
    assert {
        path.name: path.read_text() for path in sorted(tmp_path.iterdir())
    } == exports
    first_trace = normalized["rank_traces"][0]
    first_event = first_trace["events"][1]
    first_work = next(
        event
        for event in first_trace["events"]
        if event["kind"] == "collective.started"
    )
    first_edge = normalized["cross_rank_edges"][0]
    observed_values = (
        first_event["event_id"],
        first_event["provenance"]["raw_event_id"],
        first_event["provenance"]["raw_event_projection_sha256"],
        first_event["causal_predecessors"][0],
        first_work["observation"]["work_id"],
        first_work["observation"]["producer"]["identity"],
        first_edge["edge_id"],
        first_edge["endpoints"][0],
        first_edge["endpoints"][1],
        first_edge["evidence_sha256"],
    )
    for facts in (exports["ScoutBFacts.tla"], exports["ScoutBFacts.lean"]):
        assert all(str(value) in facts for value in observed_values)
    # The original defect was that the source said INFERRED while the sealed
    # artifact did not, so a reader of the bundle alone could not tell the
    # producer and stream fields from an observed join. Assert the disclaimer
    # in the artifact, not only in the provenance contract: dropping the
    # header emission would otherwise surface only as a gate-run digest.
    for module in ("ScoutBFacts.tla", "ScoutBBadFacts.tla"):
        assert "INFERRED, NOT OBSERVED" in exports[module], module
        assert "positional zip" in exports[module], module
        assert "opaque" in exports[module], module
    for module in ("ScoutBFacts.lean", "ScoutBBadFacts.lean"):
        assert "INFERRED" in exports[module], module
        assert "positional zip" in exports[module], module
    assert "controlled-invalid-producer" in exports["ScoutBBadFacts.tla"]
    assert "controlled-invalid-producer" in exports["ScoutBBadFacts.lean"]
    assert exports["ScoutBFacts.tla"] != exports["ScoutBBadFacts.tla"]
    assert exports["ScoutBFacts.lean"] != exports["ScoutBBadFacts.lean"]
    for facts in (exports["ScoutBFacts.tla"], exports["ScoutBBadFacts.tla"]):
        assert "INVARIANT" not in facts
        assert "THEOREM" not in facts
    for facts in (exports["ScoutBFacts.lean"], exports["ScoutBBadFacts.lean"]):
        assert "theorem " not in facts
        assert ": Prop := True" not in facts


def test_normalization_preserves_observed_producer_identity() -> None:
    raw_traces = _rank_bundle()
    normalized = merge_rank_traces(raw_traces)
    raw_event = next(
        event
        for event in raw_traces[0]["events"]
        if event["kind"] == "collective.enqueued"
    )
    event = next(
        event
        for event in normalized["rank_traces"][0]["events"]
        if event["kind"] == "collective.enqueued"
    )
    raw_producer = raw_event["observation"]["producer"]
    producer = event["observation"]["producer"]
    stable_producer = deepcopy(raw_producer)
    stable_producer.pop("correlation_id")

    assert producer == {
        **raw_producer,
        "identity": f"sha256:{_sha256_json(stable_producer)}",
    }
    assert event["observation"]["stream"] == raw_event["observation"]["stream"]


def test_normalized_producer_mutation_is_rejected_before_export() -> None:
    normalized = merge_rank_traces(_rank_bundle())
    completed = next(
        event
        for event in normalized["rank_traces"][0]["events"]
        if event["kind"] == "collective.completed"
    )
    completed["observation"]["producer"]["correlation_id"] = "mutated"
    normalized["trace_id"] = _expected_trace_id(normalized)

    with pytest.raises(ValueError, match="producer correlation changes within"):
        validate_normalized_bundle(normalized)


def test_stable_producer_facts_change_projection_trace_and_exports() -> None:
    raw_traces = _rank_bundle()
    original = merge_rank_traces(raw_traces)
    work_id = next(
        event["observation"]["work_id"]
        for event in raw_traces[0]["events"]
        if event["kind"] == "collective.enqueued"
    )
    for event in raw_traces[0]["events"]:
        if event["observation"].get("work_id") == work_id:
            event["observation"]["producer"]["external_id"] = "stable-mutation"

    mutated = merge_rank_traces(raw_traces)

    assert original["provenance"]["rank_raw_trace_projection_sha256"]["0"] != (
        mutated["provenance"]["rank_raw_trace_projection_sha256"]["0"]
    )
    assert original["trace_id"] != mutated["trace_id"]
    assert export_scout_b_tla_facts(original) != export_scout_b_tla_facts(mutated)
    assert export_scout_b_lean_facts(original) != export_scout_b_lean_facts(mutated)


@pytest.mark.parametrize("mutation", ["local_predecessor", "cross_rank_edge"])
def test_normalized_bundle_rejects_incomplete_causal_graph(mutation: str) -> None:
    normalized = merge_rank_traces(_rank_bundle())
    if mutation == "local_predecessor":
        normalized["rank_traces"][0]["events"][1]["causal_predecessors"] = []
        message = "local causal graph is incomplete"
    else:
        normalized["cross_rank_edges"].pop()
        message = "cross-rank synchronization graph is incomplete"
    normalized["trace_id"] = _expected_trace_id(normalized)

    with pytest.raises(ValueError, match=message):
        validate_normalized_bundle(normalized)


def test_projection_provenance_is_trace_bound_and_exported() -> None:
    normalized = merge_rank_traces(_rank_bundle())
    original_trace_id = normalized["trace_id"]
    original_tla = export_scout_b_tla_facts(normalized)
    original_lean = export_scout_b_lean_facts(normalized)
    trace = normalized["rank_traces"][0]
    event = trace["events"][0]
    event["provenance"]["raw_event_projection_sha256"] = "f" * 64
    trace["provenance"][event["event_id"]] = deepcopy(event["provenance"])
    normalized["trace_id"] = _expected_trace_id(normalized)

    validate_normalized_bundle(normalized)
    assert normalized["trace_id"] != original_trace_id
    assert export_scout_b_tla_facts(normalized) != original_tla
    assert export_scout_b_lean_facts(normalized) != original_lean


def test_artifact_sync_writes_and_detects_stale_scout_b_fact_modules(
    tmp_path: Path,
) -> None:
    normalized = merge_rank_traces(_rank_bundle())

    assert len(artifact_sync_mismatches(normalized, tmp_path)) == 4
    sync_checked_in_artifacts(normalized, tmp_path)
    assert artifact_sync_mismatches(normalized, tmp_path) == ()

    stale = tmp_path / "ScoutBFacts.tla"
    stale.write_text(stale.read_text() + "\\* stale\n")
    assert artifact_sync_mismatches(normalized, tmp_path) == (str(stale),)


def test_stale_artifact_diagnosis_names_an_identity_mismatch(tmp_path: Path) -> None:
    """A stale-artifact failure must distinguish a wrong identity from a new trace.

    The run and attempt identity is part of every event id, so a run recorded
    under a different --attempt-id disagrees on all of them, the digest differs,
    and all four modules mismatch at once. That reads as a changed trace, and the
    obvious remedy then looks like --update-artifacts, which would replace the
    canonical trace with one recorded under the wrong identity. This cost a
    twenty-minute gate run to work out by hand.
    """
    normalized = merge_rank_traces(_rank_bundle())
    sync_checked_in_artifacts(normalized, tmp_path)
    assert artifact_sync_mismatches(normalized, tmp_path) == ()

    # Same trace recorded under a different attempt id. The identity has to be
    # rewritten everywhere it appears, including inside every event id, because
    # that is what the exporter produces -- and changing only the top-level field
    # is rejected by bundle validation, which is its own guard working.
    renamed_ranks = json.loads(
        json.dumps(_rank_bundle()).replace("attempt-0", "some-other-attempt")
    )
    renamed = merge_rank_traces(renamed_ranks)
    assert renamed["attempt_id"] == "some-other-attempt"
    mismatches = artifact_sync_mismatches(renamed, tmp_path)
    assert len(mismatches) == 4, mismatches

    diagnosis = artifact_sync_diagnosis(renamed, tmp_path, mismatches)
    assert "checked-in artifacts are stale" in diagnosis
    assert "attempt_id=some-other-attempt" in diagnosis
    assert "attempt_id=attempt-0" in diagnosis
    assert "identity mismatch" in diagnosis
    assert "NOT pass --update-artifacts" in diagnosis


def test_stale_artifact_diagnosis_stays_quiet_when_the_identity_matches(
    tmp_path: Path,
) -> None:
    """A genuinely changed trace must not be misreported as an identity mismatch.

    Without this the identity block could be printed unconditionally, which would
    send someone hunting a --attempt-id typo that does not exist while the real
    cause is a trace that actually moved.
    """
    normalized = merge_rank_traces(_rank_bundle())
    sync_checked_in_artifacts(normalized, tmp_path)

    stale = tmp_path / "ScoutBFacts.tla"
    stale.write_text(stale.read_text() + "\\* stale\n")
    mismatches = artifact_sync_mismatches(normalized, tmp_path)
    assert mismatches == (str(stale),)

    diagnosis = artifact_sync_diagnosis(normalized, tmp_path, mismatches)
    assert "checked-in artifacts are stale" in diagnosis
    assert "identity mismatch" not in diagnosis
    assert "--update-artifacts" not in diagnosis


def test_finalization_and_verification_seal_complete_scout_b_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)

    manifest = finalize_evidence_manifest(
        attempt,
        expected_head="a" * 40,
        source_root=tmp_path,
    )
    evidence = verify_evidence_bundle(
        attempt,
        expected_head="a" * 40,
        source_root=tmp_path,
    )

    assert manifest == attempt / "manifests" / "evidence.json"
    assert evidence["run_id"] == "scout-b-test"
    assert evidence["attempt_id"] == "attempt-0"
    assert evidence["trace_id"]
    assert evidence["runtime_id"]
    assert evidence["stage_id"]


@pytest.mark.parametrize("stage_name", [name for name, _ in SCOUT_B_REQUIRED_STAGES])
def test_sealing_refuses_a_zero_byte_stage_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_name: str,
) -> None:
    """Drive the real sealer with a zero-byte stage log for every stage.

    This is the experiment that exposed the hole: nine zero-byte stage logs
    were fed to the real sealer and all nine were accepted, because the only
    non-empty check lived in the shell runner, outside the sealed contract.
    """

    attempt = _create_sealable_bundle(
        tmp_path,
        monkeypatch,
        empty_stage_log=stage_name,
    )

    with pytest.raises(ValueError, match=f"stage {stage_name} log is empty"):
        finalize_evidence_manifest(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def test_scout_b_runner_routes_every_pytest_stage_through_the_skip_guard() -> None:
    runner_text = RUNNER.read_text()

    assert runner_text.count('bash -lc "$(scout_pytest_guard_program)"') == 3
    invocations = [
        line
        for line in runner_text.splitlines()
        if "pytest -q" in line and not line.strip().startswith("#")
    ]
    assert len(invocations) == 3
    for line in invocations:
        assert "-rs" in line and "--color=no" in line, line


def test_runtime_manifest_detects_full_raw_byte_tamper_in_volatile_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    raw = attempt / "raw" / "ranks" / "rank-00003.json"
    contents = json.loads(raw.read_text())
    contents["events"][0]["observed_time_ns"] += 1
    raw.chmod(0o644)
    raw.write_text(_canonical_json_text(contents))
    raw.chmod(0o444)

    with pytest.raises(ValueError, match="rank-00003.json"):
        verify_runtime_bundle(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def test_runtime_verifier_recomputes_projection_provenance_from_raw_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    relative = "raw/ranks/rank-00000.json"
    raw_path = attempt / relative
    raw = json.loads(raw_path.read_text())
    raw["events"][0]["phase"] = "tampered-semantic-phase"
    raw_path.chmod(0o644)
    raw_path.write_text(_canonical_json_text(raw))
    raw_path.chmod(0o444)

    runtime_path = attempt / "manifests" / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["files"][relative] = _evidence_file_record(raw_path, root=attempt)
    runtime["runtime_id"] = ""
    runtime["runtime_id"] = f"sha256:{_sha256_json(runtime)}"
    runtime_path.chmod(0o644)
    runtime_path.write_text(_canonical_json_text(runtime))
    runtime_path.chmod(0o444)

    with pytest.raises(
        ValueError,
        match="normalized event provenance disagrees with referenced raw event",
    ):
        verify_runtime_bundle(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def test_runtime_verification_rejects_manifest_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    runtime_path = attempt / "manifests" / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["source"]["manifest_path"] = "../outside.json"
    runtime["runtime_id"] = ""
    runtime["runtime_id"] = f"sha256:{_sha256_json(runtime)}"
    runtime_path.chmod(0o644)
    runtime_path.write_text(_canonical_json_text(runtime))
    runtime_path.chmod(0o444)

    with pytest.raises(ValueError, match="invalid evidence-relative path"):
        verify_runtime_bundle(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def test_complete_bundle_verification_detects_stage_manifest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = _create_sealable_bundle(tmp_path, monkeypatch)
    finalize_evidence_manifest(
        attempt,
        expected_head="a" * 40,
        source_root=tmp_path,
    )
    stages_path = attempt / "manifests" / "stages.json"
    stages = json.loads(stages_path.read_text())
    stages["stages"][0]["command"] = "tampered-command"
    stages_path.chmod(0o644)
    stages_path.write_text(_canonical_json_text(stages))
    stages_path.chmod(0o444)

    with pytest.raises(ValueError, match="stage manifest"):
        verify_evidence_bundle(
            attempt,
            expected_head="a" * 40,
            source_root=tmp_path,
        )


def test_supported_scout_b_runner_documents_the_complete_four_rank_gate() -> None:
    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    assert "four-rank DP-shard-2/TP-2 Qwen3 Trainer step" in result.stdout
    assert "complete source-bound Scout A regression gate" in result.stdout
    assert "networked formal check" in result.stdout
    assert "fresh no-fetch formal check" in result.stdout
    assert "evidence sealing and verification" in result.stdout


def test_scout_b_sealed_bundle_survives_tracker_document_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scout B keeps its own verify path, so it needs its own regression.

    A ticket may reach ``resolved`` and record its evidence identity after the
    bundle is sealed; executable source may not drift.
    """

    status = b"?? source.py\0?? .scratch/qwen3-formal-verifier/issues/03.md\0"
    attempt = _create_sealable_bundle(
        tmp_path,
        monkeypatch,
        status_bytes=status,
        source_files={
            "source.py": "value = 1\n",
            ".scratch/qwen3-formal-verifier/issues/03.md": "**Status:** claimed\n",
        },
    )
    finalize_evidence_manifest(
        attempt, expected_head="a" * 40, source_root=str(tmp_path)
    )

    ticket = tmp_path / ".scratch" / "qwen3-formal-verifier" / "issues" / "03.md"
    ticket.write_text("**Status:** resolved\n\n## Gate evidence\n\nid: sha256:abc\n")
    verify_evidence_bundle(attempt, expected_head="a" * 40, source_root=str(tmp_path))


def test_scout_b_sealed_bundle_still_rejects_executable_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = b"?? source.py\0?? .scratch/qwen3-formal-verifier/issues/03.md\0"
    attempt = _create_sealable_bundle(
        tmp_path,
        monkeypatch,
        status_bytes=status,
        source_files={
            "source.py": "value = 1\n",
            ".scratch/qwen3-formal-verifier/issues/03.md": "**Status:** claimed\n",
        },
    )
    finalize_evidence_manifest(
        attempt, expected_head="a" * 40, source_root=str(tmp_path)
    )

    (tmp_path / "source.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="no longer matches the current source tree"):
        verify_evidence_bundle(
            attempt, expected_head="a" * 40, source_root=str(tmp_path)
        )


def test_collective_facts_key_on_canonical_id_not_runtime_pg_id() -> None:
    """The communicator key must be global, not per-rank local.

    `runtime_pg_id` is assigned per rank, so the same id denotes different
    communicators on different ranks -- in the observed 2x2 run every id maps to
    two. Keying a rendezvous check on it would silently merge two distinct
    communicators and make the check meaningless.
    """

    bundle = merge_rank_traces(_rank_bundle())
    projection = _formal_projection(bundle)
    collectives = cast(list[dict[str, Any]], projection["collectives"])

    assert collectives, "fixture must produce collectives"
    for work in collectives:
        # The key is the canonical id carried by the process group.
        assert work["comm"], work
        assert ":" in str(work["comm"]), "canonical id is axis:members[:desc]"
        # Operation and axis are first-class, not substrings of an identifier.
        assert work["operation"]
        assert work["axis"]


def test_collective_issue_order_comes_from_the_enqueued_event() -> None:
    """Only the enqueued event carries faithful ordering.

    The started/completed events are appended contiguously after the step from
    the Flight Recorder, so their relative order across different works asserts
    a serialization that did not occur. Ordering evidence must therefore be read
    from the enqueued event alone.
    """

    bundle = merge_rank_traces(_rank_bundle())
    projection = _formal_projection(bundle)
    collectives = cast(list[dict[str, Any]], projection["collectives"])
    rank_traces = cast(list[dict[str, Any]], bundle["rank_traces"])

    # Recompute the expected issue order independently of the exporter.
    expected: dict[str, int] = {}
    for trace in rank_traces:
        for event in cast(list[dict[str, Any]], trace["events"]):
            if not str(event.get("kind", "")).startswith("collective."):
                continue
            observation = cast(dict[str, Any], event["observation"])
            if str(observation.get("lifecycle", "")) != "enqueued":
                continue
            key = f"{trace['process']['rank']}:{observation['work_id']}"
            expected[key] = int(cast(dict[str, Any], event["clock"])["order"])

    assert expected, "fixture must contain enqueued events"
    for work in collectives:
        key = f"{work['rank']}:{work['work_id']}"
        assert work["issue_order"] == expected[key], key


def _rewrite_placements(
    normalized: dict[str, Any],
    mutate: Callable[[list[dict[str, Any]]], None],
    *,
    ranks: tuple[int, ...] | None = None,
    restamp_digest: bool = True,
) -> None:
    """Mutate per-rank ``tensor_placements`` the way a real run would report them.

    ``placement_summary.schema_sha256`` is a digest OVER those placements, and
    ``validate_normalized_bundle`` recomputes it, so a mutation that leaves the
    digest stale is refused as tampering rather than reaching the projection.
    Restamping is therefore the default: it makes the bundle self-consistent, so
    a test of the projection tests the projection. ``restamp_digest=False`` is
    for the tampering case itself.

    ``trace_id`` covers ``placement_summary``, so restamping the digest moves it
    too and it is recomputed here. Note what that chain does NOT cover: the
    digest is in ``trace_id`` but the placement bytes are not, which is why
    recomputing the digest against those bytes is the only thing standing
    between tampered placements and an export.
    """

    for rank, trace in enumerate(normalized["rank_traces"]):
        if ranks is not None and rank not in ranks:
            continue
        mutate(trace["tensor_placements"])
        if restamp_digest:
            trace["placement_summary"]["schema_sha256"] = _sha256_json(
                trace["tensor_placements"]
            )
    if restamp_digest:
        normalized["trace_id"] = _expected_trace_id(normalized)


def test_placement_facts_carry_axis_and_dim_per_parameter() -> None:
    """Structural placements, not two booleans per rank.

    Drives the real exporters rather than inspecting a checked-in fixture, and
    reads the projection so the assertion is about values and not about text
    happening to appear somewhere in a 1.6 MB module.
    """

    normalized = merge_rank_traces(_rank_bundle())
    facts = _formal_projection(normalized)["placement_facts"]

    assert facts["mesh_axis_degrees"] == {"dp_shard": 2, "tp": 2}
    assert [parameter["name"] for parameter in facts["parameters"]] == [
        "tok_embeddings.weight",
        "norm.weight",
    ]
    assert facts["schema_digest_by_rank"] == [facts["schema_digest"]] * 4
    assert facts["placements"] == [
        {
            "placement_id": "placement:0000",
            "parameter": "tok_embeddings.weight",
            "axis": "dp_shard",
            "kind": "shard",
            "strided": False,
            "shard_dim": 0,
        },
        {
            "placement_id": "placement:0001",
            "parameter": "tok_embeddings.weight",
            "axis": "tp",
            "kind": "shard",
            "strided": False,
            "shard_dim": 0,
        },
        {
            "placement_id": "placement:0002",
            "parameter": "norm.weight",
            "axis": "dp_shard",
            "kind": "shard",
            "strided": False,
            "shard_dim": 0,
        },
        {
            "placement_id": "placement:0003",
            "parameter": "norm.weight",
            "axis": "tp",
            "kind": "replicate",
            "strided": False,
        },
    ]

    tla = export_scout_b_tla_facts(normalized)
    lean = export_scout_b_lean_facts(normalized)
    # The two booleans are kept verbatim so PlacementValid cannot silently
    # change meaning, and they now sit beside the structural facts.
    assert "PlacementHasDpShard == " in tla
    assert "PlacementHasTpShard == " in tla
    assert 'MeshAxisDegree == (("dp_shard" :> 2) @@ ("tp" :> 2))' in tla
    # A replicate placement has no dim, so it is absent from the
    # domain-restricted shard-dim function rather than carrying a sentinel.
    shard_dim_line = next(
        line for line in tla.splitlines() if line.startswith("PlacementShardDim == ")
    )
    assert '"placement:0003"' not in shard_dim_line
    assert '"placement:0002" :> 0' in shard_dim_line
    assert "shardDim := none" in lean
    assert "shardDim := some 0" in lean


def test_placement_facts_are_delimited_for_parse_cost_reporting() -> None:
    """The runners report the placement share of the parsed bytes.

    Parse cost is charged before a single state is generated, so the block has
    delimiters the runners can measure between. This drives the exporter and
    checks the delimited block actually contains the structural operators.
    """

    normalized = merge_rank_traces(_rank_bundle())
    tla = export_scout_b_tla_facts(normalized)

    lines = tla.splitlines()
    start = lines.index(r"\* BEGIN structural placement facts")
    end = lines.index(r"\* END structural placement facts")
    block = "\n".join(lines[start : end + 1])

    assert start < end
    for operator in (
        "PlacementSchemaDigest == ",
        "PlacementSchemaDigestByRank == ",
        "MeshAxisDegree == ",
        "ParameterNames == ",
        "ParameterMeshAxes == ",
        "ParameterGlobalShape == ",
        "ParameterLocalShape == ",
        "PlacementIds == ",
        "PlacementParameter == ",
        "PlacementAxis == ",
        "PlacementKind == ",
        "PlacementStrided == ",
        "PlacementShardDim == ",
    ):
        assert operator in block, operator
        # Defined exactly once, and inside the delimiters rather than beside
        # them, so the measured share is the whole export.
        assert tla.count(f"\n{operator}") == 1, operator


def test_placement_export_carries_a_partial_placement_to_the_checkers() -> None:
    """A Partial must reach the facts, because the formal layer rejects it.

    An exporter that refused a Partial would move the check into Python and
    leave the formal invariant unfalsifiable, so the Partial has to survive the
    projection and be rejected by the checkers instead. The rank digests are
    restamped so the bundle is what a real run reporting a Partial would look
    like, rather than a tampered one the digest check refuses first.
    """

    normalized = merge_rank_traces(_rank_bundle())
    _rewrite_placements(
        normalized,
        lambda tensors: tensors[1]["placements"].__setitem__(
            1, {"axis": "tp", "kind": "partial"}
        ),
    )

    facts = _formal_projection(normalized)["placement_facts"]

    partial = [
        placement for placement in facts["placements"] if placement["kind"] == "partial"
    ]
    assert [placement["placement_id"] for placement in partial] == ["placement:0003"]
    assert "shard_dim" not in partial[0]
    assert '"placement:0003" :> "partial"' in export_scout_b_tla_facts(normalized)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dim_outside_shape", "outside its shape"),
        ("dim_on_replicate", "carries a shard dim"),
        ("unknown_kind", "unknown kind"),
        ("shapes_disagree", "shapes for .* do not align"),
        # Caught upstream by the normalized validator, which is the earlier
        # and better place for it; pinned so the projection's own guard is not
        # the only thing standing between a rank disagreement and an export.
        ("digest_conflict", "placement schema digest does not cover"),
        ("stale_digest_over_tampered_placements", "digest does not cover"),
    ],
)
def test_placement_export_refuses_malformed_placement_evidence(
    mutation: str, message: str
) -> None:
    """Bad data from disk is a ValueError, not a silently degenerate export."""

    normalized = merge_rank_traces(_rank_bundle())
    mutations = {
        "dim_outside_shape": lambda tensors: tensors[1]["placements"][0].__setitem__(
            "dim", 3
        ),
        "dim_on_replicate": lambda tensors: tensors[1]["placements"][1].__setitem__(
            "dim", 0
        ),
        "unknown_kind": lambda tensors: tensors[1]["placements"][1].__setitem__(
            "kind", "sharded"
        ),
        "shapes_disagree": lambda tensors: tensors[0].__setitem__(
            "local_shape", [1024]
        ),
    }
    if mutation in mutations:
        _rewrite_placements(normalized, mutations[mutation])
    elif mutation == "digest_conflict":
        normalized["rank_traces"][2]["placement_summary"]["schema_sha256"] = "a" * 64
    else:
        # The tampering case: rank 1's placements are rewritten and its digest
        # is deliberately left stale, which is how a hand-edited bundle looks.
        _rewrite_placements(
            normalized,
            lambda tensors: tensors[1]["placements"][1].__setitem__("kind", "partial"),
            ranks=(1,),
            restamp_digest=False,
        )

    with pytest.raises(ValueError, match=message):
        _formal_projection(normalized)


def _payload_of(normalized: dict[str, Any], work_id: str) -> dict[str, Any]:
    collective = next(
        work
        for work in _formal_projection(normalized)["collectives"]
        if work["work_id"] == work_id
    )
    return cast(dict[str, Any], collective["payload"])


def _rewrite_payloads(
    normalized: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
    *,
    ranks: tuple[int, ...] | None = None,
    lifecycles: tuple[str, ...] = ("enqueued", "started", "completed"),
) -> None:
    """Rewrite collective payloads the way a real run would have reported them.

    Payload identity is a property of the work, so it is repeated across the
    three lifecycle events and the raw validator requires all three to agree.
    Mutating only one is the incoherent-trace case, which ``lifecycles`` selects.
    """

    for rank, trace in enumerate(normalized["rank_traces"]):
        if ranks is not None and rank not in ranks:
            continue
        for event in trace["events"]:
            if not str(event["kind"]).startswith("collective."):
                continue
            observation = event["observation"]
            if observation["lifecycle"] not in lifecycles:
                continue
            mutate(observation)
    normalized["trace_id"] = _expected_trace_id(normalized)


def test_collective_payload_carries_per_tensor_sizes_and_dtypes() -> None:
    """Sizes and dtypes are plural and per tensor, with derived element counts.

    Drives the real projection rather than inspecting a checked-in fixture. The
    element counts are what the size relation is stated over, because an
    all-gather may concatenate along any dimension, so the shapes alone do not
    give the relation.
    """

    normalized = merge_rank_traces(_rank_bundle())
    collectives = _formal_projection(normalized)["collectives"]

    assert collectives, "fixture must produce collectives"
    by_operation = {
        str(work["operation"]): cast(dict[str, Any], work["payload"])
        for work in collectives
    }
    assert set(by_operation) == {"all_reduce", "reduce_scatter"}
    assert by_operation["all_reduce"]["input_sizes"] == [[128, 256]]
    assert by_operation["all_reduce"]["input_dtypes"] == ["BFloat16"]
    assert by_operation["all_reduce"]["input_elements"] == 128 * 256
    assert by_operation["all_reduce"]["output_elements"] == 128 * 256
    assert by_operation["all_reduce"]["size_relation"] == "input_equals_output"
    # The inverse relation, and one the fixture really exhibits: two members, so
    # the input volume is twice the output volume.
    assert by_operation["reduce_scatter"]["input_elements"] == 256 * 256
    assert by_operation["reduce_scatter"]["output_elements"] == 128 * 256
    assert (
        by_operation["reduce_scatter"]["size_relation"]
        == "input_is_member_count_times_output"
    )
    for work in collectives:
        payload = cast(dict[str, Any], work["payload"])
        assert len(payload["input_sizes"]) == len(payload["input_dtypes"])
        assert len(payload["output_sizes"]) == len(payload["output_dtypes"])


def test_collective_payload_is_keyed_on_the_canonical_communicator_id() -> None:
    """Never on runtime_pg_id, which denotes different communicators by rank.

    Three separate claims, because they fail differently. The payload key is the
    observed canonical id; it is a prefix of the collective id, which no integer
    key could be; and it partitions the works by member set, which a per-rank
    local numbering does not -- that is the eight-versus-four confusion. The
    prefix claim is asserted here because the Lean route to it is not axiom-free;
    see payloadCommKeyIsCommunicatorIdentity.
    """

    normalized = merge_rank_traces(_rank_bundle())
    collectives = _formal_projection(normalized)["collectives"]

    members_by_comm: dict[str, list[int]] = {}
    runtime_ids = set()
    for work in collectives:
        payload = cast(dict[str, Any], work["payload"])
        comm = str(payload["comm"])
        assert comm == str(work["comm"])
        assert str(work["collective_id"]).startswith(f"{comm}:seq")
        assert comm != str(work["runtime_pg_id"])
        runtime_ids.add(int(work["runtime_pg_id"]))
        members = [int(member) for member in work["members"]]
        assert members_by_comm.setdefault(comm, members) == members
    # Guard against a vacuous pass: the fixture must really have a runtime id
    # that two different member sets share, or the keying claim tests nothing.
    by_runtime: dict[int, set[tuple[int, ...]]] = {}
    for work in collectives:
        by_runtime.setdefault(int(work["runtime_pg_id"]), set()).add(
            tuple(int(member) for member in work["members"])
        )
    assert any(len(values) > 1 for values in by_runtime.values()), by_runtime
    assert len(members_by_comm) > len(runtime_ids)


def test_collective_payload_facts_are_delimited_and_reach_both_checkers() -> None:
    """The exported facts, in both checkers, in the shape each one needs.

    The TLA export keys flat functions by work id, which TLC can join all-pairs.
    The Lean export adds a per-collective row and a distinct-communicator table,
    because the kernel cannot afford the same join; the two are the same facts.
    """

    normalized = merge_rank_traces(_rank_bundle())
    tla = export_scout_b_tla_facts(normalized)
    lean = export_scout_b_lean_facts(normalized)

    lines = tla.splitlines()
    start = lines.index(r"\* BEGIN collective payload facts")
    end = lines.index(r"\* END collective payload facts")
    block = "\n".join(lines[start : end + 1])
    assert start < end
    for operator in (
        "CollectivePayloadComm == ",
        "CollectivePayloadSizeRelation == ",
        "CollectivePayloadInputSizes == ",
        "CollectivePayloadOutputSizes == ",
        "CollectivePayloadInputDtypes == ",
        "CollectivePayloadOutputDtypes == ",
        "CollectivePayloadInputElements == ",
        "CollectivePayloadOutputElements == ",
    ):
        assert operator in block, operator
        # Defined exactly once, and inside the delimiters rather than beside
        # them, so the measured byte share is the whole export.
        assert tla.count(f"\n{operator}") == 1, operator
    # Sequences of sequences need separating spaces, or the lexer meets three
    # consecutive angle brackets.
    assert "<<<<" not in tla

    for definition in (
        "def collectivePayloads : List CollectivePayload := [",
        "def payloadCollectiveRows : List PayloadCollectiveRow := [",
        "def payloadCommTable : List PayloadCommMembership := [",
    ):
        assert lean.count(definition) == 1, definition
    collectives = _formal_projection(normalized)["collectives"]
    # One row per collective, one entry per distinct communicator key.
    assert lean.count("{ collectiveId := ") == len(
        {str(work["collective_id"]) for work in collectives}
    )
    assert lean.count("{ comm := ") == len(
        {str(cast(dict[str, Any], work["payload"])["comm"]) for work in collectives}
    )


def test_collective_payload_rows_take_one_member_without_checking_agreement() -> None:
    """The exporter must not decide the property the checkers exist to decide.

    A per-collective row is a representative, not a verdict: if the exporter
    refused a bundle whose members disagreed, the disagreement would never reach
    a checker and the formal property would be unfalsifiable. So a one-member
    mismatch exports cleanly, with the row keeping the other member's value.
    """

    normalized = merge_rank_traces(_rank_bundle())
    _rewrite_payloads(
        normalized,
        lambda observation: observation["payload"].__setitem__(
            "input_sizes", [[256, 256]]
        ),
        ranks=(1,),
    )

    lean = export_scout_b_lean_facts(normalized)

    # Exported, not refused, and the mutated member really differs from its peer.
    projection = _formal_projection(normalized)
    volumes = {
        int(work["rank"]): int(cast(dict[str, Any], work["payload"])["input_elements"])
        for work in projection["collectives"]
        if str(work["axis"]) == "tp" and int(work["rank"]) in {0, 1}
    }
    assert volumes[0] != volumes[1], volumes
    assert "def payloadCollectiveRows : List PayloadCollectiveRow := [" in lean


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_payload", "has no input_sizes"),
        ("dtype_count_differs", "pairs 1 input_sizes with 2 input_dtypes"),
        ("empty_dtypes", "has no input_dtypes"),
        ("zero_dimension", "malformed input_sizes shape"),
        ("empty_sizes", "has no output_sizes"),
        ("payload_changes_mid_lifecycle", "collective lifecycle changes within"),
        ("unknown_operation", "no payload size relation"),
    ],
)
def test_collective_payload_refuses_malformed_evidence(
    mutation: str, message: str
) -> None:
    """Bad data from disk is a ValueError, not a silently degenerate export.

    ``payload_changes_mid_lifecycle`` is the one that would otherwise be
    invisible: the facts read the first event of a work, so a payload that
    differed at completion would never appear in the export.
    """

    normalized = merge_rank_traces(_rank_bundle())
    mutations: dict[str, Callable[[dict[str, Any]], None]] = {
        "missing_payload": lambda observation: observation["payload"].pop(
            "input_sizes"
        ),
        "dtype_count_differs": lambda observation: observation["payload"].__setitem__(
            "input_dtypes", ["BFloat16", "Float"]
        ),
        "empty_dtypes": lambda observation: observation["payload"].__setitem__(
            "input_dtypes", []
        ),
        "zero_dimension": lambda observation: observation["payload"].__setitem__(
            "input_sizes", [[0, 256]]
        ),
        "empty_sizes": lambda observation: observation["payload"].__setitem__(
            "output_sizes", []
        ),
        "unknown_operation": lambda observation: observation.__setitem__(
            "operation", "reduce"
        ),
    }
    if mutation == "payload_changes_mid_lifecycle":
        _rewrite_payloads(
            normalized,
            lambda observation: observation["payload"].__setitem__(
                "input_sizes", [[64, 256]]
            ),
            lifecycles=("completed",),
        )
    else:
        _rewrite_payloads(normalized, mutations[mutation])

    with pytest.raises(ValueError, match=message):
        _formal_projection(normalized)


def test_raw_trace_schema_bump_names_the_schema_rather_than_a_missing_field() -> None:
    """A v0 trace carries no payload, and must fail saying exactly that.

    The bump exists for the same reason PROVENANCE_CONTRACT_VERSION does: without
    it every previously sealed trace fails on a missing-field message that a
    third party cannot tell apart from tampering.
    """

    ranks = _rank_bundle()
    for trace in ranks:
        trace["schema"] = "qwen3.formal.raw.v0"

    with pytest.raises(
        ValueError, match="raw trace schema must be qwen3.formal.raw.v1"
    ):
        merge_rank_traces(ranks)


def test_every_observable_operation_family_has_a_payload_size_relation() -> None:
    """The two tables must not drift apart.

    ``_operation_family`` decides which collectives the collector accepts at all;
    ``_PAYLOAD_SIZE_RELATIONS`` decides which ones have a checkable volume
    relation. A family in the first and not the second would reach the facts and
    then fail the export on every run, and the reverse would be a relation for an
    operation that can never be observed.
    """

    families = {
        _operation_family(name)
        for name in (
            "nccl:all_reduce",
            "nccl:all_gather_into_tensor",
            "nccl:reduce_scatter_tensor",
            "nccl:broadcast",
        )
    }

    assert None not in families
    assert families == set(_PAYLOAD_SIZE_RELATIONS)
