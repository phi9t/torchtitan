#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small standard-library schema validator for launch-governing artifacts.

This is intentionally a bootstrap subset of JSON Schema. It validates the
contracts needed by the rootfs runtime without introducing a dependency on the
runtime being provisioned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


class SchemaValidationError(ValueError):
    """Raised when an artifact does not match its schema."""


def _schema_path(schema_name: str) -> Path:
    return SCHEMA_DIR / f"{schema_name}.schema.json"


def load_schema(schema_name: str) -> dict[str, Any]:
    """Load a schema by logical name."""

    path = _schema_path(schema_name)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SchemaValidationError(f"schema not found: {schema_name}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"schema must be an object: {schema_name}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SchemaValidationError(f"unsupported schema_version in {path}")
    return payload


def validate(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate payload against schema_name and return it unchanged."""

    if not isinstance(payload, dict):
        raise SchemaValidationError(f"{schema_name}: payload must be an object")
    schema = load_schema(schema_name)
    _validate_node(schema_name, payload, schema, path=schema_name)
    return payload


def validate_and_write(path: Path, schema_name: str, payload: dict[str, Any]) -> None:
    """Validate and atomically write a JSON payload."""

    validate(schema_name, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a modded-nanogpt runtime artifact."
    )
    parser.add_argument("--schema", required=True)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.input.read_text())
        if not isinstance(payload, dict):
            raise SchemaValidationError("input JSON must be an object")
        validate(args.schema, payload)
    except (json.JSONDecodeError, SchemaValidationError) as exc:
        print(f"schema validation failed: {exc}")
        return 1
    return 0


def _validate_node(
    schema_name: str, value: Any, schema: dict[str, Any], *, path: str
) -> None:
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(
            f"{schema_name}: {path} expected {schema['const']!r}, found {value!r}"
        )

    expected_type = schema.get("type")
    if expected_type is not None:
        _validate_type(schema_name, value, expected_type, path=path)

    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise SchemaValidationError(
                f"{schema_name}: {path}.required must be a list"
            )
        for key in required:
            if key not in value:
                raise SchemaValidationError(
                    f"{schema_name}: {path} missing required field {key}"
                )

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise SchemaValidationError(
                f"{schema_name}: {path}.properties must be an object"
            )
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise SchemaValidationError(
                    f"{schema_name}: {path} has unknown top-level field(s): {', '.join(unknown)}"
                )
        for key, child_schema in properties.items():
            if key in value:
                if not isinstance(child_schema, dict):
                    raise SchemaValidationError(
                        f"{schema_name}: schema for {path}.{key} must be an object"
                    )
                _validate_node(
                    schema_name, value[key], child_schema, path=f"{path}.{key}"
                )

    if isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema is not None:
            if not isinstance(item_schema, dict):
                raise SchemaValidationError(
                    f"{schema_name}: {path}.items must be an object"
                )
            for index, item in enumerate(value):
                _validate_node(schema_name, item, item_schema, path=f"{path}[{index}]")


def _validate_type(
    schema_name: str, value: Any, expected_type: Any, *, path: str
) -> None:
    if isinstance(expected_type, list):
        if any(_type_matches(value, item) for item in expected_type):
            return
        expected = "|".join(str(item) for item in expected_type)
    else:
        if _type_matches(value, expected_type):
            return
        expected = str(expected_type)
    raise SchemaValidationError(
        f"{schema_name}: {path} expected type {expected}, found {type(value).__name__}"
    )


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "null":
        return value is None
    raise SchemaValidationError(f"unsupported schema type: {expected_type}")


if __name__ == "__main__":
    raise SystemExit(main())
