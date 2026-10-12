# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional learned-model extension interfaces for state estimation."""

from __future__ import annotations

import importlib
import json

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceSegment:
    id: str
    entity: Mapping[str, str]
    modality: str
    start_time_ns: int | None
    end_time_ns: int | None
    features: Mapping[str, Any] = field(default_factory=dict)
    source_observation_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": dict(self.entity),
            "modality": self.modality,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "features": dict(self.features),
            "source_observation_ids": list(self.source_observation_ids),
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
class LearnedModelIdentity:
    name: str
    version: str
    artifact_digest: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "artifact_digest": self.artifact_digest,
        }


@dataclass(frozen=True, slots=True)
class CalibrationMetadata:
    state: str
    method: str
    evaluated_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "method": self.method,
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True, slots=True)
class TrainingDataManifest:
    id: str
    uri: str
    digest: str

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "uri": self.uri, "digest": self.digest}


@dataclass(frozen=True, slots=True)
class SourceSegmentReference:
    id: str
    modality: str
    source_observation_ids: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modality": self.modality,
            "source_observation_ids": list(self.source_observation_ids),
        }


@dataclass(frozen=True, slots=True)
class LearnedFactorBundle:
    model_identity: LearnedModelIdentity
    advisory: bool
    calibration: CalibrationMetadata
    training_data_manifest: TrainingDataManifest
    source_segments: tuple[SourceSegmentReference, ...]
    factors: tuple[LikelihoodFactor, ...]
    schema_version: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_identity": self.model_identity.to_json(),
            "advisory": self.advisory,
            "calibration": self.calibration.to_json(),
            "training_data_manifest": self.training_data_manifest.to_json(),
            "source_segments": [
                source_segment.to_json() for source_segment in self.source_segments
            ],
            "factors": [factor.to_json() for factor in self.factors],
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


def load_learned_factor_bundle(
    path: str | Path | None,
) -> LearnedFactorBundle | None:
    """Load an explicitly requested advisory learned-factor JSON bundle."""

    if path is None:
        return None
    bundle_path = Path(path)
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("learned factor bundle must be a JSON object")
    _reject_control_decision_fields(data)
    bundle = LearnedFactorBundle(
        schema_version=_optional_int(data.get("schema_version")),
        model_identity=_model_identity(_required_mapping(data, "model_identity")),
        advisory=_advisory_flag(data),
        calibration=_calibration(_required_mapping(data, "calibration")),
        training_data_manifest=_training_data_manifest(
            _required_mapping(data, "training_data_manifest")
        ),
        source_segments=_source_segments(data.get("source_segments")),
        factors=_likelihood_factors(data.get("factors")),
    )
    source_segment_ids = {
        source_segment.id for source_segment in bundle.source_segments
    }
    for factor in bundle.factors:
        if not factor.source_segment_id:
            raise ValueError("factor is missing source segment identity")
        if factor.source_segment_id not in source_segment_ids:
            raise ValueError(
                f"factor references unknown source segment {factor.source_segment_id!r}"
            )
    return bundle


def _required_mapping(data: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if field_name not in data:
        raise ValueError(f"learned factor bundle is missing {_field_label(field_name)}")
    value = data[field_name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{_field_label(field_name)} must be a JSON object")
    return value


def _model_identity(data: Mapping[str, Any]) -> LearnedModelIdentity:
    return LearnedModelIdentity(
        name=_required_string(data, "name", owner="model identity"),
        version=_required_string(data, "version", owner="model identity"),
        artifact_digest=_required_string(
            data, "artifact_digest", owner="model identity"
        ),
    )


def _advisory_flag(data: Mapping[str, Any]) -> bool:
    if "advisory" not in data:
        raise ValueError("learned factor bundle is missing advisory flag")
    if data["advisory"] is not True:
        raise ValueError("learned factor bundle advisory flag must be true")
    return True


def _calibration(data: Mapping[str, Any]) -> CalibrationMetadata:
    return CalibrationMetadata(
        state=_required_string(data, "state", owner="calibration"),
        method=_required_string(data, "method", owner="calibration"),
        evaluated_at=_optional_string(data.get("evaluated_at")),
    )


def _training_data_manifest(data: Mapping[str, Any]) -> TrainingDataManifest:
    return TrainingDataManifest(
        id=_required_string(data, "id", owner="training-data manifest"),
        uri=_required_string(data, "uri", owner="training-data manifest"),
        digest=_required_string(data, "digest", owner="training-data manifest"),
    )


def _source_segments(value: Any) -> tuple[SourceSegmentReference, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("learned factor bundle is missing source segment identity")
    segments: list[SourceSegmentReference] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"source segment {index} must be a JSON object")
        segment_id = _required_string(item, "id", owner="source segment")
        modality = _required_string(item, "modality", owner="source segment")
        observation_ids = item.get("source_observation_ids", [])
        if not isinstance(observation_ids, list) or not all(
            isinstance(observation_id, str) and observation_id
            for observation_id in observation_ids
        ):
            raise ValueError(
                "source segment source_observation_ids must be a list of strings"
            )
        segments.append(
            SourceSegmentReference(
                id=segment_id,
                modality=modality,
                source_observation_ids=tuple(observation_ids),
            )
        )
    return tuple(segments)


def _likelihood_factors(value: Any) -> tuple[LikelihoodFactor, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("factors must be a list")
    factors: list[LikelihoodFactor] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"factor {index} must be a JSON object")
        if "source_segment_id" not in item:
            raise ValueError("factor is missing source segment identity")
        factors.append(
            LikelihoodFactor(
                source_segment_id=_required_string(
                    item, "source_segment_id", owner="factor"
                ),
                target_state=_required_string(item, "target_state", owner="factor"),
                log_likelihood=_required_float(item, "log_likelihood", owner="factor"),
                calibration=_required_string(item, "calibration", owner="factor"),
                metadata=_optional_mapping(item.get("metadata")),
            )
        )
    return tuple(factors)


def _required_string(data: Mapping[str, Any], field_name: str, *, owner: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} is missing {field_name}")
    return value


def _required_float(data: Mapping[str, Any], field_name: str, *, owner: str) -> float:
    value = data.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{owner} is missing {field_name}")
    return float(value)


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON object")
    return dict(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional string field must be a string or null")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("schema_version must be an integer")
    return value


_CONTROL_DECISION_TERMS = (
    "recovery",
    "rollback",
    "commit",
    "retry",
    "checkpoint_valid",
    "checkpoint validity",
)
_TRANSACTION_CONTROL_DECISION_TERMS = (
    "transaction_safety",
    "transaction safety",
    "transaction_control",
    "transaction control",
    "transaction_decision",
    "transaction decision",
)


_SOURCE_EVIDENCE_PATHS = {
    ("source_segments", "*", "id"),
    ("source_segments", "*", "modality"),
    ("source_segments", "*", "source_observation_ids", "*"),
    ("factors", "*", "source_segment_id"),
}


def _reject_control_decision_fields(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, str(key))
            normalized_key = str(key).lower().replace("-", "_")
            if _has_control_decision_term(
                normalized_key,
                allow_transaction_evidence=_is_source_evidence_path(child_path),
            ):
                raise ValueError(
                    "learned factor bundle contains control decision metadata"
                )
            if isinstance(child, str):
                normalized_child = child.lower().replace("-", "_")
                if _has_control_decision_term(
                    normalized_child,
                    allow_transaction_evidence=_is_source_evidence_path(child_path),
                ):
                    raise ValueError(
                        "learned factor bundle contains control decision metadata"
                    )
            else:
                _reject_control_decision_fields(child, path=child_path)
    elif isinstance(value, list):
        for child in value:
            _reject_control_decision_fields(child, path=(*path, "*"))


def _is_source_evidence_path(path: tuple[str, ...]) -> bool:
    return path in _SOURCE_EVIDENCE_PATHS


def _has_control_decision_term(value: str, *, allow_transaction_evidence: bool) -> bool:
    if any(term in value for term in _CONTROL_DECISION_TERMS):
        return True
    if any(term in value for term in _TRANSACTION_CONTROL_DECISION_TERMS):
        return True
    return not allow_transaction_evidence and "transaction" in value


def _field_label(field_name: str) -> str:
    if field_name == "training_data_manifest":
        return "training-data manifest"
    return field_name.replace("_", " ")
