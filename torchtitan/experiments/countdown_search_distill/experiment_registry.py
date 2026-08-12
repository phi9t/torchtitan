# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run registry helpers for Countdown experiment provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_HASH_LIMIT_BYTES = 64 * 1024 * 1024


def build_countdown_report_input(
    *,
    experiment_root: Path,
    mode: str,
    run_id: str,
    manifest: Path,
    max_hash_bytes: int = DEFAULT_HASH_LIMIT_BYTES,
) -> dict[str, Any]:
    """Build a compact, auditable input object for reports and promotion gates."""

    splits = ["dev", "iid_test", "ood_test"]
    arms = _arms_for_mode(mode)
    data_root = experiment_root / "data"
    results_root = experiment_root / "results"
    adapter_eval_root = results_root / "eval" / "adapters" / mode

    base_summaries = {
        split: _read_json_if_exists(results_root / "eval" / split / "base" / "summary.json")
        for split in splits
    }
    adapter_matrix = _read_json_if_exists(adapter_eval_root / f"adapter_matrix_{mode}.json")
    analysis = _build_countdown_analysis(
        results_root=results_root,
        adapter_eval_root=adapter_eval_root,
        splits=splits,
        arms=arms,
    )

    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "mode": mode,
            "experiment_root": str(experiment_root),
            "manifest": str(manifest),
            "rootfs_active": os.environ.get("TORCHTITAN_IN_ROOTFS") == "1",
        },
        "manifest": {
            "path": str(manifest),
            "stages": _read_manifest(manifest),
        },
        "artifacts": {
            "runtime_preflight": _artifact_metadata(
                results_root / "runtime_preflight.json",
                max_hash_bytes=max_hash_bytes,
            ),
            "split_registry": _artifact_metadata(
                data_root / "split_registry.json",
                max_hash_bytes=max_hash_bytes,
            ),
            "adapter_matrix": _artifact_metadata(
                adapter_eval_root / f"adapter_matrix_{mode}.json",
                max_hash_bytes=max_hash_bytes,
            ),
            "base_summaries": {
                split: _artifact_metadata(
                    results_root / "eval" / split / "base" / "summary.json",
                    max_hash_bytes=max_hash_bytes,
                )
                for split in splits
            },
            "adapter_summaries": {
                f"{split}/{arm}": _artifact_metadata(
                    adapter_eval_root / split / arm / "summary.json",
                    max_hash_bytes=max_hash_bytes,
                )
                for split in splits
                for arm in arms
            },
            "adapter_exports": {
                arm: _artifact_metadata(
                    results_root / "adapters" / mode / arm / "export_summary.json",
                    max_hash_bytes=max_hash_bytes,
                )
                for arm in arms
            },
        },
        "validations": {
            "runtime_preflight": _read_json_if_exists(results_root / "runtime_preflight.json"),
            "split_registry": _read_json_if_exists(data_root / "split_registry.json"),
            "adapter_matrix": adapter_matrix,
        },
        "metrics": {
            "base": {
                split: _compact_summary(summary)
                for split, summary in base_summaries.items()
            },
            "adapters": _compact_adapter_rows(adapter_matrix),
        },
        "analysis": analysis,
        "checks": _report_checks(
            mode=mode,
            base_summaries=base_summaries,
            adapter_matrix=adapter_matrix,
            split_registry=_read_json_if_exists(data_root / "split_registry.json"),
            manifest_stages=_read_manifest(manifest),
        ),
    }


def write_countdown_report_input(
    report_input: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_input, indent=2, sort_keys=True) + "\n")


def _build_countdown_analysis(
    *,
    results_root: Path,
    adapter_eval_root: Path,
    splits: list[str],
    arms: list[str],
) -> dict[str, Any]:
    base_by_split = {
        split: _read_evaluation_rows(
            results_root / "eval" / split / "base" / "evaluations.jsonl"
        )
        for split in splits
    }
    adapter_by_split_arm = {
        (split, arm): _read_evaluation_rows(
            adapter_eval_root / split / arm / "evaluations.jsonl"
        )
        for split in splits
        for arm in arms
    }
    return {
        "base_elicitable_subsets": _base_elicitable_subset_metrics(
            base_by_split,
            adapter_by_split_arm,
            splits=splits,
            arms=arms,
        ),
        "representative_examples": _representative_examples(
            base_by_split,
            adapter_by_split_arm,
            splits=splits,
            arms=arms,
        ),
    }


def _arms_for_mode(mode: str) -> list[str]:
    if mode == "reduced":
        return ["raw", "hindsight", "curriculum"]
    if mode == "full":
        return ["raw", "clean", "formatting", "hindsight", "curriculum"]
    if mode == "smoke":
        return ["debug_smoke"]
    raise ValueError(f"unknown Countdown mode: {mode}")


def _read_evaluation_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid evaluation row at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"evaluation row at {path}:{line_number} must be an object")
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str):
            raise ValueError(f"evaluation row at {path}:{line_number} missing problem_id")
        rows[problem_id] = row
    return rows


def _base_elicitable_subset_metrics(
    base_by_split: dict[str, dict[str, dict[str, Any]]],
    adapter_by_split_arm: dict[tuple[str, str], dict[str, dict[str, Any]]],
    *,
    splits: list[str],
    arms: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        base_rows = base_by_split[split]
        base_elicitable_ids = sorted(
            problem_id
            for problem_id, row in base_rows.items()
            if _solved_at(row) is not None and _solved_at(row) != 1
        )
        base_subset = [base_rows[problem_id] for problem_id in base_elicitable_ids]
        if not base_subset:
            continue
        base_metrics = _subset_metric_row(base_subset)
        rows.append(
            {
                "split": split,
                "arm": "base",
                "subset": "base_elicitable",
                "num_problems": len(base_subset),
                **base_metrics,
            }
        )
        for arm in arms:
            adapter_rows = adapter_by_split_arm[(split, arm)]
            subset = [
                adapter_rows[problem_id]
                for problem_id in base_elicitable_ids
                if problem_id in adapter_rows
            ]
            if len(subset) != len(base_elicitable_ids):
                continue
            rows.append(
                {
                    "split": split,
                    "arm": arm,
                    "subset": "base_elicitable",
                    "num_problems": len(subset),
                    **_subset_metric_row(subset),
                }
            )
    return rows


def _subset_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pass_at_1": _pass_at(rows, 1, strict=False),
        "pass_at_32": _pass_at(rows, 32, strict=False),
        "strict_format_pass_at_1": _pass_at(rows, 1, strict=True),
        "strict_format_pass_at_32": _pass_at(rows, 32, strict=True),
    }


def _pass_at(rows: list[dict[str, Any]], k: int, *, strict: bool) -> float:
    if not rows:
        return 0.0
    solved = 0
    for row in rows:
        solved_at = _strict_solved_at(row) if strict else _solved_at(row)
        if solved_at is not None and solved_at <= k:
            solved += 1
    return solved / len(rows)


def _representative_examples(
    base_by_split: dict[str, dict[str, dict[str, Any]]],
    adapter_by_split_arm: dict[tuple[str, str], dict[str, dict[str, Any]]],
    *,
    splits: list[str],
    arms: list[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    selectors = (
        ("win", _is_win),
        ("regression", _is_regression),
        ("unchanged_failure", _is_unchanged_failure),
        ("format_failure", _is_format_failure),
    )
    for split in splits:
        base_rows = base_by_split[split]
        if not base_rows:
            continue
        for arm in arms:
            adapter_rows = adapter_by_split_arm[(split, arm)]
            if not adapter_rows:
                continue
            common_problem_ids = sorted(set(base_rows) & set(adapter_rows))
            for category, predicate in selectors:
                for problem_id in common_problem_ids:
                    base_row = base_rows[problem_id]
                    adapter_row = adapter_rows[problem_id]
                    if predicate(base_row, adapter_row):
                        examples.append(
                            _example_row(
                                split=split,
                                arm=arm,
                                category=category,
                                base_row=base_row,
                                adapter_row=adapter_row,
                            )
                        )
                        break
    return examples


def _is_win(base_row: dict[str, Any], adapter_row: dict[str, Any]) -> bool:
    base_solved_at = _solved_at(base_row)
    adapter_solved_at = _solved_at(adapter_row)
    return (
        adapter_solved_at is not None
        and adapter_solved_at == 1
        and (base_solved_at is None or base_solved_at > 1)
    )


def _is_regression(base_row: dict[str, Any], adapter_row: dict[str, Any]) -> bool:
    return _solved_at(base_row) == 1 and _solved_at(adapter_row) != 1


def _is_unchanged_failure(base_row: dict[str, Any], adapter_row: dict[str, Any]) -> bool:
    return _solved_at(base_row) is None and _solved_at(adapter_row) is None


def _is_format_failure(base_row: dict[str, Any], adapter_row: dict[str, Any]) -> bool:
    del base_row
    return _solved_at(adapter_row) == 1 and _strict_solved_at(adapter_row) != 1


def _example_row(
    *,
    split: str,
    arm: str,
    category: str,
    base_row: dict[str, Any],
    adapter_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "arm": arm,
        "category": category,
        "problem_id": str(adapter_row.get("problem_id")),
        "problem": _compact_problem(adapter_row.get("problem", {})),
        "base": _compact_evaluation_for_example(base_row),
        "adapter": _compact_evaluation_for_example(adapter_row),
    }


def _compact_problem(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "numbers": value.get("numbers"),
        "target": value.get("target"),
        "canonical_solution": value.get("canonical_solution")
        or value.get("solution"),
    }


def _compact_evaluation_for_example(row: dict[str, Any]) -> dict[str, Any]:
    sample = _first_rollout(row)
    return {
        "solved_at": _solved_at(row),
        "strict_solved_at": _strict_solved_at(row),
        "bucket": _bucket(row),
        "sample_index": sample.get("sample_index"),
        "success": sample.get("success"),
        "final_value": sample.get("final_value"),
        "error": sample.get("error"),
        "text_excerpt": _text_excerpt(str(sample.get("text", ""))),
    }


def _solved_at(row: dict[str, Any]) -> int | None:
    value = row.get("solved_at")
    if value is not None:
        return int(value)
    for index, rollout in enumerate(_rollouts(row), start=1):
        if rollout.get("success") is True:
            return index
    return None


def _strict_solved_at(row: dict[str, Any]) -> int | None:
    target = _target(row)
    if target is None:
        return None
    for index, rollout in enumerate(_rollouts(row), start=1):
        if rollout.get("success") is True and _has_strict_final_line(
            str(rollout.get("text", "")),
            target,
        ):
            return index
    return None


def _bucket(row: dict[str, Any]) -> str:
    value = row.get("bucket")
    if isinstance(value, str):
        return value
    solved_at = _solved_at(row)
    if solved_at == 1:
        return "easy"
    if solved_at is None:
        return "unreached"
    return "elicitable"


def _target(row: dict[str, Any]) -> int | None:
    problem = row.get("problem")
    if not isinstance(problem, dict):
        return None
    target = problem.get("target")
    return None if target is None else int(target)


def _rollouts(row: dict[str, Any]) -> list[dict[str, Any]]:
    rollouts = row.get("rollouts", [])
    if not isinstance(rollouts, list):
        return []
    return [rollout for rollout in rollouts if isinstance(rollout, dict)]


def _first_rollout(row: dict[str, Any]) -> dict[str, Any]:
    rollouts = _rollouts(row)
    return rollouts[0] if rollouts else {}


def _has_strict_final_line(text: str, target: int) -> bool:
    return any(line.strip() == f"FINAL: {target}" for line in text.splitlines())


def _text_excerpt(text: str, *, max_chars: int = 600) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest row at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest row at {path}:{line_number} must be an object")
        rows.append(row)
    return rows


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _artifact_metadata(
    path: Path,
    *,
    max_hash_bytes: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return metadata
    stat = path.stat()
    metadata.update(
        {
            "mtime_ns": stat.st_mtime_ns,
            "type": "dir" if path.is_dir() else "file",
        }
    )
    if path.is_dir():
        metadata["sha256"] = None
        metadata["hash_skipped_reason"] = "directory"
        return metadata
    metadata["size_bytes"] = stat.st_size
    if stat.st_size > max_hash_bytes:
        metadata["sha256"] = None
        metadata["hash_skipped_reason"] = "file_too_large"
        return metadata
    metadata["sha256"] = _sha256_file(path)
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    pass_at_k = summary.get("pass_at_k", {})
    strict_pass_at_k = summary.get("strict_format_pass_at_k", {})
    return {
        "num_problems": summary.get("num_problems"),
        "pass_at_1": pass_at_k.get("1"),
        "pass_at_32": pass_at_k.get("32"),
        "strict_format_pass_at_1": strict_pass_at_k.get("1"),
        "strict_format_pass_at_32": strict_pass_at_k.get("32"),
        "bucket_counts": summary.get("bucket_counts", {}),
        "validity_breakdown": summary.get("validity_breakdown", {}),
        "format_breakdown": summary.get("format_breakdown", {}),
    }


def _compact_adapter_rows(adapter_matrix: dict[str, Any] | None) -> list[dict[str, Any]]:
    if adapter_matrix is None:
        return []
    rows = adapter_matrix.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [
        {
            "split": row.get("split"),
            "arm": row.get("arm"),
            "num_problems": row.get("num_problems"),
            "pass_at_1": row.get("pass_at_1"),
            "pass_at_32": row.get("pass_at_32"),
            "strict_format_pass_at_1": row.get("strict_format_pass_at_1"),
            "strict_format_pass_at_32": row.get("strict_format_pass_at_32"),
            "bucket_counts": row.get("bucket_counts", {}),
            "format_breakdown": row.get("format_breakdown", {}),
            "summary": row.get("summary"),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def _report_checks(
    *,
    mode: str,
    base_summaries: dict[str, dict[str, Any] | None],
    adapter_matrix: dict[str, Any] | None,
    split_registry: dict[str, Any] | None,
    manifest_stages: list[dict[str, Any]],
) -> dict[str, bool]:
    required_stages = {
        "smoke": {
            "preflight",
            "calibration",
            "collect",
            "train_debug_smoke",
            "base_eval_dev",
            "base_eval_iid_test",
            "base_eval_ood_test",
        },
        "reduced": {
            "preflight",
            "calibration_sweep",
            "calibration",
            "collect",
            "validate_splits",
            "train_raw",
            "train_hindsight",
            "train_curriculum",
            "base_eval_dev",
            "base_eval_iid_test",
            "base_eval_ood_test",
            "export_adapters",
            "eval_adapters",
        },
        "full": {
            "preflight",
            "calibration_sweep",
            "calibration",
            "collect",
            "validate_splits",
            "train_raw",
            "train_clean",
            "train_formatting",
            "train_hindsight",
            "train_curriculum",
            "base_eval_dev",
            "base_eval_iid_test",
            "base_eval_ood_test",
            "export_adapters",
            "eval_adapters",
        },
    }[mode]
    successful_stages = {
        str(row.get("stage"))
        for row in manifest_stages
        if int(row.get("return_code", -1)) == 0
    }
    return {
        "required_manifest_stages_succeeded": required_stages <= successful_stages,
        "split_registry_selected": bool(
            split_registry and split_registry.get("selected", False)
        ),
        "split_registry_no_overlap": bool(
            split_registry
            and split_registry.get("checks", {}).get("no_problem_key_overlap", False)
        ),
        "base_summaries_present": all(summary is not None for summary in base_summaries.values()),
        "adapter_matrix_selected": mode == "smoke"
        or bool(adapter_matrix and adapter_matrix.get("selected", False)),
    }
