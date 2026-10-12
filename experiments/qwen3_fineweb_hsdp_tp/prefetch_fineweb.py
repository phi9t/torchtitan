# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-time online prefetch of a FineWeb-edu slice to local JSON-lines.

This is the only step in the experiment that touches the network. It streams
the first --num-docs documents from HuggingFaceFW/fineweb-edu (sample-10BT)
and writes them as {"text": ...} JSON-lines, mirroring
tests/assets/c4_test/data.json so the existing local loader and
_process_c4_text work unchanged. Streaming with an early break avoids
downloading a full shard.

Usage (run once, online, from the repo root):
    python experiments/qwen3_fineweb_hsdp_tp/prefetch_fineweb.py
"""

import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-docs",
        type=int,
        default=2000,
        help="Number of documents to prefetch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "fineweb_test",
            "data.json",
        ),
        help="Output JSON-lines path.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    written = 0
    # Iterate the streaming dataset explicitly so we can close the underlying
    # HTTP connection before exiting; breaking out of a for-loop over a
    # streaming dataset can otherwise leave a background thread mid-request and
    # emit noisy teardown warnings during interpreter finalization.
    it = iter(ds)
    with open(args.output, "w") as f:
        while written < args.num_docs:
            try:
                sample = next(it)
            except StopIteration:
                break
            f.write(json.dumps({"text": sample["text"]}) + "\n")
            written += 1
    del it, ds

    print(f"Wrote {written} documents to {args.output}")


if __name__ == "__main__":
    main()
