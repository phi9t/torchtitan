# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rootfs entrypoint for the Falcon overfit gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from torchtitan.experiments.falcon.overfit import run_overfit


def main(argv: list[str] | None = None) -> int:
    del argv
    report = run_overfit()
    out_dir = Path("experiments/falcon/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "overfit.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
