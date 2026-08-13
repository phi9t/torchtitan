# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Post-hoc reducer over one attempt's per-process run evidence outcomes.

This is intentionally not on the training hot path. It runs after an attempt,
reads every ``processes/<process_id>/outcome.json`` published by the recorder,
and writes one immutable ``aggregate_outcome.json`` sibling to ``manifest.json``.
It preserves the "no central collector" contract of the run evidence
foundation: no cross-rank barrier and no in-process aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from torchtitan.observability.run_evidence import (
    _canonical_json,
    EvidenceCollisionError,
    EvidenceContractError,
    EvidenceWriteError,
    RunEvidenceError,
)


def reduce_attempt_outcome(attempt_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Reduce the per-process outcomes under one attempt directory.

    ``attempt_dir`` is the directory that holds ``manifest.json`` and the
    ``processes/`` tree. The aggregate outcome is:

    - ``failed`` if any process failed;
    - ``interrupted`` if none failed and any process was interrupted;
    - ``succeeded`` only when every expected process succeeded, the world size
      is consistent, and the expected world size is positive;
    - ``incomplete`` otherwise, including any missing process outcome or an
      inconsistent world size. Missing is not success.

    ``expected_world_size`` is derived from the per-process ``world_size``, which
    each surviving process records from its launcher ``WORLD_SIZE``. The manifest
    carries no authoritative world size, so when no process outcome is present
    the expected size is ``None`` and ``missing_process_count`` is 0; that attempt
    is still ``incomplete`` rather than ``succeeded``.
    """
    attempt_path = Path(attempt_dir)
    manifest = _read_manifest(attempt_path)
    outcomes = _read_process_outcomes(attempt_path)

    world_sizes = {entry["world_size"] for entry in outcomes}
    consistent_world_size = len(world_sizes) <= 1
    expected_world_size = next(iter(world_sizes)) if len(world_sizes) == 1 else None

    process_outcome_count = len(outcomes)
    if expected_world_size is not None and consistent_world_size:
        missing_process_count = max(expected_world_size - process_outcome_count, 0)
    else:
        missing_process_count = 0

    failed_ranks = sorted(
        entry["global_rank"] for entry in outcomes if entry["outcome"] == "failed"
    )
    any_interrupted = any(entry["outcome"] == "interrupted" for entry in outcomes)
    all_succeeded = all(entry["outcome"] == "succeeded" for entry in outcomes)

    if failed_ranks:
        aggregate_outcome = "failed"
    elif any_interrupted:
        aggregate_outcome = "interrupted"
    elif (
        all_succeeded
        and consistent_world_size
        and expected_world_size is not None
        and expected_world_size > 0
        and missing_process_count == 0
        and process_outcome_count == expected_world_size
    ):
        aggregate_outcome = "succeeded"
    else:
        aggregate_outcome = "incomplete"

    reduced_wall_time_ns = max(
        (entry["elapsed_monotonic_ns"] for entry in outcomes), default=0
    )

    return {
        "schema_version": 1,
        "evidence_schema_version": 1,
        "record_type": "attempt_outcome_aggregate",
        "run_id": manifest["run_id"],
        "attempt_id": manifest["attempt_id"],
        "expected_world_size": expected_world_size,
        "process_outcome_count": process_outcome_count,
        "missing_process_count": missing_process_count,
        "aggregate_outcome": aggregate_outcome,
        "first_failure": failed_ranks[0] if failed_ranks else None,
        "consistent_world_size": consistent_world_size,
        "reduced_wall_time_ns": reduced_wall_time_ns,
        "outcomes": outcomes,
    }


def write_attempt_outcome(
    attempt_dir: str | os.PathLike[str], *, force: bool = False
) -> Path:
    """Write ``aggregate_outcome.json`` immutably next to ``manifest.json``.

    Without ``force`` a second write raises ``EvidenceCollisionError``, matching
    the immutability of ``manifest.json`` and each process outcome. ``force``
    replaces the aggregate atomically for a deliberate recomputation.
    """
    attempt_path = Path(attempt_dir)
    aggregate = reduce_attempt_outcome(attempt_path)
    output_path = attempt_path / "aggregate_outcome.json"
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=".aggregate_outcome.", dir=attempt_path, text=True
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_canonical_json(aggregate) + "\n")
            if force:
                os.replace(temporary_path, output_path)
            else:
                try:
                    os.link(temporary_path, output_path)
                except FileExistsError as error:
                    raise EvidenceCollisionError(
                        "run evidence aggregate outcome already exists"
                    ) from error
        finally:
            temporary_path.unlink(missing_ok=True)
    except RunEvidenceError:
        raise
    except OSError as error:
        raise EvidenceWriteError(
            "failed to write run evidence aggregate outcome"
        ) from error
    return output_path


def _read_manifest(attempt_path: Path) -> dict[str, Any]:
    manifest_path = attempt_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvidenceWriteError(
            f"attempt manifest could not be read: {manifest_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise EvidenceContractError(
            f"attempt manifest is not valid JSON: {manifest_path}"
        ) from error
    for field in ("run_id", "attempt_id"):
        if field not in manifest:
            raise EvidenceContractError(
                f"attempt manifest is missing required field: {field}"
            )
    return manifest


def _read_process_outcomes(attempt_path: Path) -> list[dict[str, Any]]:
    processes_dir = attempt_path / "processes"
    outcomes: list[dict[str, Any]] = []
    if not processes_dir.is_dir():
        return outcomes
    for outcome_file in sorted(processes_dir.glob("*/outcome.json")):
        try:
            text = outcome_file.read_text(encoding="utf-8")
        except OSError as error:
            raise EvidenceWriteError(
                f"process outcome could not be read: {outcome_file}"
            ) from error
        try:
            record = json.loads(text)
        except json.JSONDecodeError as error:
            raise EvidenceContractError(
                f"process outcome is not valid JSON: {outcome_file}"
            ) from error
        for field in ("outcome", "global_rank", "world_size", "elapsed_monotonic_ns"):
            if field not in record:
                raise EvidenceContractError(
                    f"process outcome is missing required field {field}: "
                    f"{outcome_file}"
                )
        outcomes.append(record)
    outcomes.sort(key=lambda entry: entry["global_rank"])
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m torchtitan.observability.aggregate_outcome",
        description=(
            "Reduce one run-attempt's per-process outcomes into an immutable "
            "aggregate_outcome.json. Runs post-hoc, not on the training path."
        ),
    )
    parser.add_argument(
        "attempt_dir",
        help="attempt directory containing manifest.json and processes/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing aggregate_outcome.json",
    )
    args = parser.parse_args(argv)
    output_path = write_attempt_outcome(args.attempt_dir, force=args.force)
    print(str(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
