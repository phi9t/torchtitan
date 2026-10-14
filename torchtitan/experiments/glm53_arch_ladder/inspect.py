# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect S07 architecture-ladder configs."""

from __future__ import annotations

import argparse
import json

import torch

from .flavors import build_ladder_model_config, TARGET_ACTIVE_NON_EMBEDDING_PARAMS
from .params import parameter_counts


RUNGS = ("L1", "L2", "L3", "L4")


def build_rows(seq_len: int) -> list[dict[str, int | str | bool | float]]:
    rows = []
    lower = int(0.9 * TARGET_ACTIVE_NON_EMBEDDING_PARAMS)
    upper = int(1.1 * TARGET_ACTIVE_NON_EMBEDDING_PARAMS)
    with torch.device("meta"):
        for rung in RUNGS:
            model = build_ladder_model_config(rung, seq_len=seq_len).build()
            counts = parameter_counts(model)
            rows.append(
                {
                    "rung": rung,
                    "total": counts.total,
                    "non_embedding": counts.non_embedding,
                    "active_non_embedding": counts.active_non_embedding,
                    "target_ratio": counts.active_non_embedding
                    / TARGET_ACTIVE_NON_EMBEDDING_PARAMS,
                    "within_match_rule": lower <= counts.active_non_embedding <= upper,
                }
            )
    return rows


def _print_markdown(rows: list[dict[str, int | str | bool | float]]) -> None:
    print(
        "| Rung | Total params | Non-embedding params | "
        "Active non-embedding params | Target ratio | Match rule |"
    )
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        print(
            "| {rung} | {total:,} | {non_embedding:,} | "
            "{active_non_embedding:,} | {target_ratio:.3f} | {match} |".format(
                **row,
                match="PASS" if row["within_match_rule"] else "FAIL",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    rows = build_rows(args.seq_len)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_markdown(rows)


if __name__ == "__main__":
    main()
