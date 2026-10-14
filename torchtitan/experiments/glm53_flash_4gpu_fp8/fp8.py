# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 frozen-weight modules for GLM-5.3-Flash.

Shape suffixes:
    T = tokens, D = hidden dim, I = input features, O = output features,
    E = experts, F = expert/intermediate features, B = block count.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.experiments.glm53_flash.model import LoRALinear


FP8_BLOCK_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn
SCALE_DTYPE = torch.float32
_FP8_MAX = torch.finfo(FP8_DTYPE).max
_FP8_MIN = torch.finfo(FP8_DTYPE).min


def mark_fp8_parameter(param: nn.Parameter, role: str) -> nn.Parameter:
    """Mark a parameter for FSDP dtype override routing."""
    setattr(param, "_glm53_flash_fp8_role", role)
    return param


def fp8_param_dtype_override(param: nn.Parameter) -> torch.dtype | None:
    """Keep FP8 qdata and FP32 scales out of BF16 mixed-precision casting."""
    role = getattr(param, "_glm53_flash_fp8_role", None)
    if role == "qdata":
        return FP8_DTYPE
    if role == "scale":
        return SCALE_DTYPE
    return None


def is_fp8_qdata_param(param: torch.Tensor) -> bool:
    return getattr(param, "_glm53_flash_fp8_role", None) == "qdata"


def is_fp8_scale_param(param: torch.Tensor) -> bool:
    return getattr(param, "_glm53_flash_fp8_role", None) == "scale"


def blockwise_scale_shape(
    weight_shape: tuple[int, ...],
    *,
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[int, ...]:
    """Return one scale per 128x128 block over the last two dimensions."""
    if len(weight_shape) < 2:
        raise ValueError(
            "FP8 blockwise quantization requires at least two dimensions, "
            f"got {weight_shape}."
        )
    rows, cols = weight_shape[-2:]
    return (
        *weight_shape[:-2],
        math.ceil(rows / block_size),
        math.ceil(cols / block_size),
    )


def quantize_blockwise_128(
    source: torch.Tensor,
    *,
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor blockwise over its last two dimensions.

    The returned qdata has the original shape and dtype ``float8_e4m3fn``.
    Scales are FP32 with shape ``source.shape[:-2] + [ceil(O/128), ceil(I/128)]``.
    """
    if source.ndim < 2:
        raise ValueError(
            "FP8 blockwise quantization requires at least two dimensions, "
            f"got {source.ndim}."
        )
    rows, cols = source.shape[-2:]
    padded_rows = math.ceil(rows / block_size) * block_size
    padded_cols = math.ceil(cols / block_size) * block_size
    source_float = source.float()
    if padded_rows != rows or padded_cols != cols:
        source_float = F.pad(
            source_float,
            (0, padded_cols - cols, 0, padded_rows - rows),
        )
    blocks = source_float.reshape(
        *source.shape[:-2],
        padded_rows // block_size,
        block_size,
        padded_cols // block_size,
        block_size,
    )
    amax = blocks.abs().amax(dim=(-3, -1))
    scale = (amax / _FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    scaled = (blocks / scale.unsqueeze(-2).unsqueeze(-1)).clamp(_FP8_MIN, _FP8_MAX)
    qdata = scaled.to(FP8_DTYPE).reshape(
        *source.shape[:-2],
        padded_rows,
        padded_cols,
    )
    return qdata[..., :rows, :cols].contiguous(), scale.to(SCALE_DTYPE).contiguous()


def dequantize_blockwise_128(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize qdata produced by :func:`quantize_blockwise_128`."""
    expected_scale_shape = blockwise_scale_shape(tuple(qdata.shape), block_size=block_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "FP8 scale shape mismatch: expected "
            f"{expected_scale_shape}, got {tuple(scale.shape)} for qdata "
            f"{tuple(qdata.shape)}."
        )
    expanded_scale = scale.repeat_interleave(block_size, dim=-2).repeat_interleave(
        block_size, dim=-1
    )
    expanded_scale = expanded_scale[..., : qdata.shape[-2], : qdata.shape[-1]]
    return (qdata.float() * expanded_scale).to(dtype)


def quantize_into_fp8_storage_(
    qdata_target: torch.Tensor,
    scale_target: torch.Tensor,
    source: torch.Tensor,
) -> None:
    """Quantize ``source`` and copy qdata/scales into persistent storage."""
    if tuple(qdata_target.shape) != tuple(source.shape):
        raise ValueError(
            "FP8 qdata target shape mismatch: "
            f"target={tuple(qdata_target.shape)} source={tuple(source.shape)}."
        )
    expected_scale_shape = blockwise_scale_shape(tuple(source.shape))
    if tuple(scale_target.shape) != expected_scale_shape:
        raise ValueError(
            "FP8 scale target shape mismatch: "
            f"expected={expected_scale_shape} got={tuple(scale_target.shape)}."
        )
    qdata, scale = quantize_blockwise_128(source)
    qdata_target.copy_(qdata.to(device=qdata_target.device), non_blocking=True)
    scale_target.copy_(scale.to(device=scale_target.device), non_blocking=True)


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return local storage for a DTensor or the input tensor otherwise."""
    if isinstance(value, DTensor):
        return value._local_tensor
    return value


def _new_scale_parameter(
    weight: torch.Tensor,
    *,
    requires_grad: bool = False,
) -> nn.Parameter:
    scale = nn.Parameter(
        torch.empty(
            blockwise_scale_shape(tuple(weight.shape)),
            device=weight.device,
            dtype=SCALE_DTYPE,
        ),
        requires_grad=requires_grad,
    )
    return mark_fp8_parameter(scale, "scale")


def _new_qdata_parameter(weight: torch.Tensor) -> nn.Parameter:
    qdata = nn.Parameter(
        torch.empty_like(weight, dtype=FP8_DTYPE),
        requires_grad=False,
    )
    return mark_fp8_parameter(qdata, "qdata")


class FP8FrozenLoRALinear(LoRALinear):
    """LoRA linear with frozen base storage in FP8 qdata plus FP32 scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        rank: int | None = None,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            rank=rank,
            alpha=alpha,
        )
        old_weight = self.weight
        self.weight = _new_qdata_parameter(old_weight)
        self.weight_scale = _new_scale_parameter(old_weight)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @classmethod
    def from_lora_linear(cls, module: LoRALinear) -> "FP8FrozenLoRALinear":
        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            rank=module.rank,
            alpha=module.alpha,
        )

    def load_base_from_bf16_(self, source: torch.Tensor) -> None:
        quantize_into_fp8_storage_(
            local_tensor(self.weight),
            local_tensor(self.weight_scale),
            source,
        )

    def dequantized_weight(self, *, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_blockwise_128(self.weight, self.weight_scale, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self.dequantized_weight(dtype=x.dtype)
        out = F.linear(x, base_weight, self.bias)
        if self.rank is None:
            return out
        assert self.lora_a is not None and self.lora_b is not None
        adapter = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return out + adapter * (self.alpha / self.rank)


class FP8FrozenExperts(nn.Module):
    """Routed GLM experts with frozen expert matrices stored in FP8."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        swiglu_limit: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.swiglu_limit = swiglu_limit
        gate_up = torch.empty(num_experts, 2 * intermediate_dim, hidden_dim)
        down = torch.empty(num_experts, hidden_dim, intermediate_dim)
        self.gate_up_proj = _new_qdata_parameter(gate_up)
        self.gate_up_proj_scale = _new_scale_parameter(gate_up)
        self.down_proj = _new_qdata_parameter(down)
        self.down_proj_scale = _new_scale_parameter(down)

    @classmethod
    def from_experts(cls, module: Any) -> "FP8FrozenExperts":
        return cls(
            num_experts=module.num_experts,
            hidden_dim=module.hidden_dim,
            intermediate_dim=module.intermediate_dim,
            swiglu_limit=module.swiglu_limit,
        )

    def load_gate_up_from_bf16_(self, source: torch.Tensor) -> None:
        quantize_into_fp8_storage_(
            local_tensor(self.gate_up_proj),
            local_tensor(self.gate_up_proj_scale),
            source,
        )

    def load_down_from_bf16_(self, source: torch.Tensor) -> None:
        quantize_into_fp8_storage_(
            local_tensor(self.down_proj),
            local_tensor(self.down_proj_scale),
            source,
        )

    def _gate_up_weight(self, expert_idx: torch.Tensor, *, dtype: torch.dtype):
        return dequantize_blockwise_128(
            self.gate_up_proj[expert_idx],
            self.gate_up_proj_scale[expert_idx],
            dtype=dtype,
        )

    def _down_weight(self, expert_idx: torch.Tensor, *, dtype: torch.dtype):
        return dequantize_blockwise_128(
            self.down_proj[expert_idx],
            self.down_proj_scale[expert_idx],
            dtype=dtype,
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_index_TK: torch.Tensor,
        topk_weights_TK: torch.Tensor,
    ) -> torch.Tensor:
        out_TD = torch.zeros_like(x_TD)
        routing_map_EKT = F.one_hot(
            topk_index_TK, num_classes=self.num_experts + 1
        ).permute(2, 1, 0)
        active_experts = torch.greater(routing_map_EKT.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in active_experts:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            slot_T, token_T = torch.where(routing_map_EKT[expert_idx])
            gate_up_weight = self._gate_up_weight(expert_idx, dtype=x_TD.dtype)
            down_weight = self._down_weight(expert_idx, dtype=x_TD.dtype)
            gate_up_TF2 = F.linear(x_TD[token_T], gate_up_weight)
            gate_TF, up_TF = gate_up_TF2.chunk(2, dim=-1)
            hidden_TF = F.silu(gate_TF.clamp(max=self.swiglu_limit)) * up_TF.clamp(
                min=-self.swiglu_limit, max=self.swiglu_limit
            )
            contrib_TD = F.linear(hidden_TF, down_weight)
            contrib_TD = contrib_TD * topk_weights_TK[token_T, slot_T, None]
            out_TD.index_add_(0, token_T, contrib_TD.to(out_TD.dtype))
        return out_TD

