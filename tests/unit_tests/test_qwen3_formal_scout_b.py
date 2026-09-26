# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for the observed four-rank DPxTP formal-verifier scout."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from torchtitan.experiments.qwen3_formal_verifier.scout_b import (
    _canonical_json_text,
    _evidence_file_record,
    _expected_trace_id,
    _formal_projection,
    _sha256_json,
    _write_attempt_bundle,
    _write_immutable,
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
    collective_id: str,
) -> list[dict[str, object]]:
    producer = f"kineto-r{rank}-{collective_id}"
    work_id = f"work-r{rank}-{collective_id}"
    common: dict[str, object] = {
        "axis": axis,
        "collective_id": collective_id,
        "work_id": work_id,
        "operation": "all_reduce",
        "process_group": {
            "canonical_id": f"{axis}:{','.join(str(member) for member in members)}",
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
            collective_id=f"tp-dp{dp}-seq1",
        ),
        _raw_event(rank, 7, "forward.completed", "forward"),
        _raw_event(rank, 8, "backward.started", "backward"),
        *_collective_events(
            rank=rank,
            start=9,
            axis="dp_shard",
            members=dp_members,
            collective_id=f"dp-tp{tp}-seq1",
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
        "schema": "qwen3.formal.raw.v0",
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
    source_path = attempt / "manifests" / "source.json"
    write_source_manifest(
        status_path=status_path,
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
        "version": "qwen3.formal.scout.provenance-contract.v1",
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
