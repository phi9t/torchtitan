from __future__ import annotations

import importlib.resources
from pathlib import Path

import torch


_LOADED = False


def _load_extension() -> None:
    global _LOADED
    if _LOADED:
        return
    package_dir = importlib.resources.files(__package__)
    candidates = sorted(Path(str(package_dir)).glob("_C*.so"))
    if not candidates:
        raise ImportError("glm53_kernel_examples native extension is not installed")
    torch.ops.load_library(str(candidates[0]))
    _LOADED = True


_load_extension()


@torch.library.register_fake("glm53_native::scale_add")
def _scale_add_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    del alpha
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)


def _scale_add_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, float],
    output: torch.Tensor,
) -> None:
    del output
    _x, _y, alpha = inputs
    ctx.alpha = float(alpha)


def _scale_add_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    alpha = ctx.alpha  # pyrefly: ignore[missing-attribute]
    return grad_out * alpha, grad_out, None


torch.library.register_autograd(
    "glm53_native::scale_add",
    _scale_add_backward,
    setup_context=_scale_add_setup_context,
)


def scale_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Return ``x * alpha + y`` through the native torch.library example op."""

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.dtype != y.dtype:
        raise TypeError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}")
    return torch.ops.glm53_native.scale_add(x, y, float(alpha))

