# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create a reviewed Mini Kimi K3 launch-backend evidence report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review Mini-K3 r1 backend evidence before launch preflight."
    )
    parser.add_argument(
        "--forward-oracle-report",
        type=Path,
        required=True,
        help="Strict first-party logits comparison report.",
    )
    parser.add_argument(
        "--forward-trace-report",
        type=Path,
        required=True,
        help="Exact-state layer trace report from trace-forward-oracle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the launch-backend review JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = review_launch_backend(
            forward_oracle_report=args.forward_oracle_report,
            forward_trace_report=args.forward_trace_report,
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "kind": "mini_kimi_k3_launch_backend_review",
            "status": "fail",
            "model_flavor": "r1",
            "backend": "torchtitan.experiments.mini_kimi_k3",
            "config": "mini_kimi_k3_r1_contract",
            "evidence": {
                "forward_oracle_report": str(args.forward_oracle_report),
                "forward_trace_report": str(args.forward_trace_report),
            },
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        print("Mini-K3 launch backend review failed", file=sys.stderr)
        print(f"wrote launch backend report: {args.output}", file=sys.stderr)
        return 21
    print("Mini-K3 launch backend review passed", file=sys.stderr)
    print(f"wrote launch backend report: {args.output}", file=sys.stderr)
    return 0


def review_launch_backend(
    *,
    forward_oracle_report: Path,
    forward_trace_report: Path,
) -> dict[str, Any]:
    forward_oracle = _read_json(forward_oracle_report)
    forward_trace = _read_json(forward_trace_report)
    _validate_forward_oracle(forward_oracle)
    _validate_forward_trace(forward_trace)
    final = forward_trace["final"]
    return {
        "schema_version": 1,
        "kind": "mini_kimi_k3_launch_backend_review",
        "status": "pass",
        "model_flavor": "r1",
        "backend": "torchtitan.experiments.mini_kimi_k3",
        "config": "mini_kimi_k3_r1_contract",
        "evidence": {
            "forward_oracle_report": str(forward_oracle_report),
            "forward_trace_report": str(forward_trace_report),
            "forward_oracle_max_abs_diff": forward_oracle["results"]["max_abs_diff"],
            "forward_oracle_max_rel_diff": forward_oracle["results"]["max_rel_diff"],
            "final_logit_max_abs_diff": final["logits"]["max_abs_diff"],
            "final_norm_max_abs_diff": final["norm"]["max_abs_diff"],
            "final_output_residual_max_abs_diff": final["output_attention_residual"][
                "max_abs_diff"
            ],
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"report must be a JSON object: {path}")
    return payload


def _validate_forward_oracle(report: dict[str, Any]) -> None:
    if report.get("kind") != "mini_kimi_k3_forward_oracle":
        raise ValueError(
            "forward oracle report kind must be mini_kimi_k3_forward_oracle"
        )
    if report.get("schema_version") != 1:
        raise ValueError("forward oracle report schema_version must be 1")
    if report.get("status") != "pass":
        raise ValueError("forward oracle report must pass")
    if report.get("model_flavor") != "r1":
        raise ValueError("forward oracle report must target model_flavor r1")
    results = report.get("results")
    tolerances = report.get("tolerances")
    if not isinstance(results, dict) or not isinstance(tolerances, dict):
        raise ValueError("forward oracle report must include results and tolerances")
    for key in ("max_abs_diff", "max_rel_diff"):
        measured = results.get(key)
        tolerance = tolerances.get(key)
        if not isinstance(measured, int | float) or not isinstance(
            tolerance, int | float
        ):
            raise ValueError(f"forward oracle report {key} must be numeric")
        if measured > tolerance:
            raise ValueError(
                f"forward oracle report {key} exceeds tolerance "
                f"({measured} > {tolerance})"
            )


def _validate_forward_trace(report: dict[str, Any]) -> None:
    if report.get("kind") != "mini_kimi_k3_forward_trace":
        raise ValueError("forward trace report kind must be mini_kimi_k3_forward_trace")
    if report.get("schema_version") != 1:
        raise ValueError("forward trace report schema_version must be 1")
    if report.get("status") != "pass":
        raise ValueError("forward trace report must pass")
    if report.get("config") != "mini_kimi_k3_r1_contract":
        raise ValueError("forward trace report config must be mini_kimi_k3_r1_contract")
    final = report.get("final")
    if not isinstance(final, dict):
        raise ValueError("forward trace report must include final metrics")
    for section, key in (
        ("logits", "final logit"),
        ("norm", "final norm"),
        ("output_attention_residual", "final output residual"),
    ):
        metrics = final.get(section)
        if not isinstance(metrics, dict):
            raise ValueError(f"forward trace report missing {section} metrics")
        max_abs = metrics.get("max_abs_diff")
        if not isinstance(max_abs, int | float) or max_abs != 0.0:
            raise ValueError(
                f"forward trace report {key} max_abs_diff must be exactly 0.0"
            )


if __name__ == "__main__":
    raise SystemExit(main())
