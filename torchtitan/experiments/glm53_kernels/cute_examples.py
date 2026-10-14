# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL package convention checks for GLM kernels.

The rootfs provides NVIDIA CUTLASS/CuTe DSL as the importable ``cutlass`` and
``cutlass.cute`` packages. Candidate CuTe kernels should live as Python JIT
sources in this package, not in L2 native wheels.
"""

from __future__ import annotations

from typing import Any

import torch


def cute_runtime_info() -> dict[str, Any]:
    """Return the installed CuTe DSL package identity."""

    import cutlass
    import cutlass.cute as cute

    return {
        "cutlass_version": getattr(cutlass, "__version__", "unknown"),
        "has_cute_kernel": hasattr(cute, "kernel"),
        "has_cute_jit": hasattr(cute, "jit"),
    }


def cute_scale_add_convention(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Eager scale-add wrapper used by checks until a CuTe kernel candidate lands."""

    info = cute_runtime_info()
    if not info["has_cute_kernel"] or not info["has_cute_jit"]:
        raise RuntimeError(f"CuTe DSL decorators are unavailable: {info}")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    return torch.add(y, x, alpha=float(alpha))

