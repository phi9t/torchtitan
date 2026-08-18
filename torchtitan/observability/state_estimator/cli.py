# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline CLI for training state-estimator analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torchtitan.observability.state_estimator.estimators import write_belief_summary
from torchtitan.observability.state_estimator.graph import write_evidence_graph
from torchtitan.observability.state_estimator.reports import (
    analyze_attempt_report,
    analyze_run_report,
    compare_attempt_reports,
    evaluate_report,
)


def analyze_attempt(attempt_path: Path, *, overwrite: bool = True) -> dict[str, Any]:
    graph_paths = write_evidence_graph(attempt_path)
    belief_paths = write_belief_summary(attempt_path)
    return {
        "ok": True,
        "command": "compat-analyze-attempt",
        "attempt_path": str(attempt_path),
        "evidence_graph": str(graph_paths.evidence_graph),
        "belief_summary": str(belief_paths.belief_summary),
        "diagnosis": str(belief_paths.diagnosis),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-path",
        type=Path,
        help="Compatibility path: analyze one attempt and write under attempt/derived.",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_attempt_parser = subparsers.add_parser("analyze-attempt")
    analyze_attempt_parser.add_argument("--attempt-path", required=True, type=Path)
    analyze_attempt_parser.add_argument("--output-dir", required=True, type=Path)

    analyze_run_parser = subparsers.add_parser("analyze-run")
    analyze_run_parser.add_argument("--run-path", required=True, type=Path)
    analyze_run_parser.add_argument("--output-dir", required=True, type=Path)

    compare_parser = subparsers.add_parser("compare-attempts")
    compare_parser.add_argument(
        "--attempt-path", required=True, action="append", type=Path
    )
    compare_parser.add_argument("--output-dir", required=True, type=Path)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--manifest-path", required=True, type=Path)
    evaluate_parser.add_argument("--output-dir", required=True, type=Path)

    return parser.parse_args(argv)


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "action": "repair required evidence and rerun offline analysis",
    }


def main() -> int:
    args = parse_args()
    try:
        if args.command == "analyze-attempt":
            result = analyze_attempt_report(args.attempt_path, args.output_dir)
        elif args.command == "analyze-run":
            result = analyze_run_report(args.run_path, args.output_dir)
        elif args.command == "compare-attempts":
            result = compare_attempt_reports(args.attempt_path, args.output_dir)
        elif args.command == "evaluate":
            result = evaluate_report(args.manifest_path, args.output_dir)
        elif args.attempt_path is not None:
            result = analyze_attempt(args.attempt_path)
        else:
            raise ValueError(
                "choose a subcommand or pass --attempt-path for compatibility"
            )
    except Exception as exc:
        print(json.dumps(_error_payload(exc), sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
