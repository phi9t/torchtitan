# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from torchtitan.observability.state_estimator.features import extract_feature_segments
from torchtitan.observability.state_estimator.learned import (
    FaultModePrior,
    LikelihoodFactor,
    load_learned_extension,
    load_learned_factor_bundle,
    TraceSegment,
)
from torchtitan.observability.state_estimator.observation import (
    ClockQuality,
    NormalizedObservation,
    ObservationEnvelope,
    ObservationKind,
)


FORBIDDEN_DEFAULT_IMPORTS = (
    "torch",
    "transformers",
    "networkx",
    "torch_geometric",
    "torchdiffeq",
    "torchdyn",
    "torchcde",
    "neural_cde",
    "neural_ode",
    "cuda",
    "cupy",
    "triton",
)
FORBIDDEN_CONTROL_WORDS = (
    "transaction",
    "recovery",
    "rollback",
    "commit",
    "retry",
    "checkpoint_valid",
    "checkpoint validity",
)


def _observation(
    observation_id: str,
    *,
    record_type: str,
    payload: dict,
    kind: ObservationKind = ObservationKind.STRUCTURED_EVENT,
    event_time_ns: int = 1_000,
    source_path: str = "structured_logs/events.jsonl",
) -> NormalizedObservation:
    return NormalizedObservation(
        id=observation_id,
        kind=kind,
        envelope=ObservationEnvelope(
            run_id="run-1",
            attempt_id="attempt-1",
            schema_version=1,
            record_type=record_type,
            source_path=source_path,
            raw_record_id=observation_id,
            process_id="rank0",
            role="trainer",
            actor_id="trainer-0",
            host_name="host-a",
            pid=1234,
            global_rank=0,
            local_rank=0,
            world_size=8,
            device_index=0,
            device_uuid="GPU-0",
            step=7,
            phase=payload.get("phase", "train_step"),
            event_time_ns=event_time_ns,
            ingestion_time_ns=event_time_ns + 10,
            monotonic_ns=event_time_ns - 100,
            event_seq=7,
            artifact_seq=None,
            sensor_id="sensor-a",
            clock_quality=ClockQuality.EVENT_AND_MONOTONIC,
            quality="present",
        ),
        payload=payload,
        raw_record=payload,
    )


def _valid_factor_bundle() -> dict:
    return {
        "schema_version": 1,
        "model_identity": {
            "name": "fixture-learned-factor",
            "version": "2026-08-18",
            "artifact_digest": "sha256:abc",
        },
        "advisory": True,
        "calibration": {
            "state": "uncalibrated",
            "method": "none",
            "evaluated_at": None,
        },
        "training_data_manifest": {
            "id": "manifest-1",
            "uri": "file://fixtures/train.jsonl",
            "digest": "sha256:def",
        },
        "source_segments": [
            {
                "id": "scalar:rank0:7",
                "modality": "scalar",
                "source_observation_ids": ["obs-scalar"],
            }
        ],
        "factors": [
            {
                "source_segment_id": "scalar:rank0:7",
                "target_state": "host_or_data_stall",
                "log_likelihood": -1.25,
                "calibration": "uncalibrated",
                "metadata": {"family": "fixture"},
            }
        ],
    }


def _write_bundle(tmp_path, data: dict):
    path = tmp_path / "learned_factors.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _loaded_forbidden_default_imports(import_statement: str) -> list[str]:
    script = f"""
import json
import sys

FORBIDDEN_DEFAULT_IMPORTS = {FORBIDDEN_DEFAULT_IMPORTS!r}
{import_statement}
print(json.dumps([
    name for name in FORBIDDEN_DEFAULT_IMPORTS if name in sys.modules
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_learned_schemas_serialize_without_torch_dependency():
    loaded_forbidden = _loaded_forbidden_default_imports(
        "from torchtitan.observability.state_estimator.learned import "
        "LikelihoodFactor, TraceSegment"
    )

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

    assert loaded_forbidden == []
    assert segment.to_json()["modality"] == "structured_events"
    assert factor.to_json()["calibration"] == "heuristic"


def test_null_extension_loader_returns_none():
    assert load_learned_extension(None) is None


def test_factor_bundle_loader_requires_explicit_path(tmp_path):
    path = _write_bundle(tmp_path, _valid_factor_bundle())

    assert load_learned_factor_bundle(None) is None

    bundle = load_learned_factor_bundle(path)

    assert bundle.model_identity.name == "fixture-learned-factor"
    assert bundle.advisory is True
    assert bundle.calibration.state == "uncalibrated"
    assert bundle.training_data_manifest.id == "manifest-1"
    assert bundle.factors[0].source_segment_id == "scalar:rank0:7"


def test_factor_bundle_allows_transaction_source_segment_evidence(tmp_path):
    data = _valid_factor_bundle()
    data["source_segments"][0] = {
        "id": "transaction:rank0:7",
        "modality": "transaction",
        "source_observation_ids": ["obs-transaction"],
    }
    data["factors"][0]["source_segment_id"] = "transaction:rank0:7"
    path = _write_bundle(tmp_path, data)

    bundle = load_learned_factor_bundle(path)

    assert bundle.source_segments[0].modality == "transaction"
    assert bundle.source_segments[0].source_observation_ids == ("obs-transaction",)
    assert bundle.factors[0].source_segment_id == "transaction:rank0:7"


def test_fault_mode_prior_schema_marks_advisory():
    prior = FaultModePrior(
        mode="network_degradation", score=0.2, calibration="heuristic"
    )

    assert prior.to_json()["advisory"] is True


def test_feature_extraction_covers_supported_modalities():
    observations = [
        _observation(
            "obs-scalar",
            record_type="scalar_metric",
            payload={
                "metric_name": "train/loss",
                "value": 2.5,
                "unit": "loss",
                "phase": "train_step",
            },
        ),
        _observation(
            "obs-event",
            record_type="structured_event",
            payload={"event_name": "batch_start", "severity": "info"},
            event_time_ns=1_100,
        ),
        _observation(
            "obs-trace",
            record_type="trace_summary",
            payload={
                "trace_id": "trace-1",
                "duration_ns": 1_000,
                "op_count": 12,
                "phase": "forward",
            },
            event_time_ns=1_200,
        ),
        _observation(
            "obs-comm",
            record_type="communication",
            payload={
                "collective": "all_reduce",
                "bytes": 4096,
                "duration_ns": 800,
                "group_size": 8,
            },
            event_time_ns=1_300,
        ),
        _observation(
            "obs-hw",
            record_type="hardware",
            payload={
                "sensor": "dcgm",
                "gpu_utilization": 91.5,
                "hbm_used_bytes": 1024,
                "temperature_c": 70,
            },
            event_time_ns=1_400,
        ),
        _observation(
            "obs-framework",
            record_type="framework",
            payload={
                "component": "dataloader",
                "state": "waiting",
                "queue_depth": 3,
            },
            event_time_ns=1_500,
        ),
        _observation(
            "obs-transaction",
            record_type="transaction",
            payload={
                "committed_step": 6,
                "speculative_step": 7,
                "process_group_epoch": 1,
                "replica_epoch": 2,
            },
            event_time_ns=1_600,
        ),
    ]

    segments = extract_feature_segments(observations)
    by_modality = {segment.modality: segment for segment in segments}

    assert set(by_modality) == {
        "scalar",
        "event",
        "trace_summary",
        "communication",
        "hardware",
        "framework",
        "transaction",
    }
    assert by_modality["scalar"].features["metric_name"] == "train/loss"
    assert by_modality["scalar"].features["value"] == 2.5
    assert by_modality["event"].features["event_name"] == "batch_start"
    assert by_modality["trace_summary"].features["duration_ns"] == 1_000
    assert by_modality["communication"].features["collective"] == "all_reduce"
    assert by_modality["hardware"].features["device_uuid"] == "GPU-0"
    assert by_modality["framework"].features["component"] == "dataloader"
    assert by_modality["transaction"].features["committed_step"] == 6


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("model_identity", "model identity"),
        ("advisory", "advisory"),
        ("calibration", "calibration"),
        ("training_data_manifest", "training-data manifest"),
    ],
)
def test_factor_bundle_rejects_missing_required_metadata(tmp_path, field, expected):
    data = _valid_factor_bundle()
    data.pop(field)
    path = _write_bundle(tmp_path, data)

    with pytest.raises(ValueError, match=expected):
        load_learned_factor_bundle(path)


def test_factor_bundle_rejects_missing_source_segment_identity(tmp_path):
    data = _valid_factor_bundle()
    data["factors"][0].pop("source_segment_id")
    path = _write_bundle(tmp_path, data)

    with pytest.raises(ValueError, match="source segment"):
        load_learned_factor_bundle(path)


def test_loaded_factors_remain_advisory_and_exclude_control_decision_terms(tmp_path):
    path = _write_bundle(tmp_path, _valid_factor_bundle())

    bundle = load_learned_factor_bundle(path)
    serialized = bundle.to_json()
    serialized_text = json.dumps(serialized, sort_keys=True)

    assert serialized["advisory"] is True
    assert all(factor["advisory"] is True for factor in serialized["factors"])
    assert not any(word in serialized_text for word in FORBIDDEN_CONTROL_WORDS)

    unsafe = _valid_factor_bundle()
    unsafe["factors"][0]["metadata"] = {"safe_to_commit": True}
    unsafe_path = _write_bundle(tmp_path, unsafe)
    with pytest.raises(ValueError, match="control decision"):
        load_learned_factor_bundle(unsafe_path)


@pytest.mark.parametrize(
    "unsafe_update",
    [
        lambda data: data["factors"][0].update({"target_state": "recovery"}),
        lambda data: data["factors"][0].update({"target_state": "rollback"}),
        lambda data: data["factors"][0].update({"target_state": "retry"}),
        lambda data: data["factors"][0].update({"target_state": "transaction_safety"}),
        lambda data: data["factors"][0].update({"target_state": "transaction_control"}),
        lambda data: data["factors"][0].update(
            {"target_state": "transaction_decision"}
        ),
        lambda data: data["factors"][0].update({"target_state": "checkpoint_valid"}),
        lambda data: data["factors"][0].update({"target_state": "checkpoint validity"}),
        lambda data: data["factors"][0].update({"metadata": {"should_commit": True}}),
    ],
)
def test_factor_bundle_rejects_control_decision_outputs(tmp_path, unsafe_update):
    data = _valid_factor_bundle()
    unsafe_update(data)
    path = _write_bundle(tmp_path, data)

    with pytest.raises(ValueError, match="control decision"):
        load_learned_factor_bundle(path)


def test_default_learned_imports_do_not_load_ml_or_gpu_dependencies():
    loaded_forbidden = _loaded_forbidden_default_imports(
        "import torchtitan.observability.state_estimator.learned"
    )

    assert loaded_forbidden == []
