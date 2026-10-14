# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional CUDA memory cap for GLM EP co-run torchrun workers.

This module is imported only when its directory is explicitly prepended to
``PYTHONPATH``. It is intentionally a no-op unless
``TORCHTITAN_CUDA_MEMORY_CAP_GIB`` is set.
"""

from __future__ import annotations

import os
import sys


def _apply_cuda_memory_cap() -> None:
    cap_raw = os.environ.get("TORCHTITAN_CUDA_MEMORY_CAP_GIB")
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if cap_raw is None or local_rank_raw is None:
        return

    try:
        cap_gib = float(cap_raw)
        local_rank = int(local_rank_raw)
    except ValueError as exc:
        raise ValueError(
            "TORCHTITAN_CUDA_MEMORY_CAP_GIB must be a number and LOCAL_RANK "
            "must be an integer."
        ) from exc
    if cap_gib <= 0:
        return

    import torch

    if not torch.cuda.is_available():
        return

    torch.cuda.set_device(local_rank)
    total_gib = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
    fraction = min(1.0, cap_gib / total_gib)
    torch.cuda.set_per_process_memory_fraction(fraction, device=local_rank)
    print(
        "TORCHTITAN_CUDA_MEMORY_CAP_GIB="
        f"{cap_gib:g} applied to local_rank={local_rank} fraction={fraction:.6f}",
        file=sys.stderr,
        flush=True,
    )


_apply_cuda_memory_cap()
