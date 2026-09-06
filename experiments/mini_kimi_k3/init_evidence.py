# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize a Mini Kimi K3 run-attempt evidence bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.mini_kimi_k3.evidence import initialize_launch_evidence_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize an immutable Mini Kimi K3 run-attempt bundle."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/mini_kimi_k3/results"),
        help="Root directory that will contain runs/<run-id>/<attempt-id>/.",
    )
    parser.add_argument("--run-id", required=True, help="Stable Mini-K3 run id.")
    parser.add_argument("--attempt-id", required=True, help="Unique attempt id.")
    parser.add_argument(
        "--mode",
        choices=("full", "tiny-plumbing"),
        default="full",
        help="Launch lane recorded in the run declaration.",
    )
    parser.add_argument(
        "--model-variant",
        default="r1",
        help="Mini-K3 model variant recorded in the run declaration.",
    )
    parser.add_argument(
        "--parent-attempt-id",
        help="Parent attempt id when this attempt resumes or supersedes another.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        bundle_dir = initialize_launch_evidence_bundle(
            results_root=args.results_root,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            mode=args.mode,
            model_variant=args.model_variant,
            parent_attempt_id=args.parent_attempt_id,
        )
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 21

    print(f"initialized Mini Kimi K3 evidence bundle: {bundle_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
