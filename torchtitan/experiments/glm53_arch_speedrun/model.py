# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""B0 GPT substrate for the GLM-5.3 architecture speedrun.

Shape suffixes:
    B = batch, L = sequence length, D = model dimension, H = attention heads,
    K = head dimension, V = vocabulary size.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    get_fsdp_reshard_after_forward_policy,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.local_compile import apply_local_compile, local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base import BaseModel
from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook

from .aux_loss import SpeedrunIndexerAuxLoss, SpeedrunMtpAuxLoss
from .elements import maybe_apply_rope, SpeedrunElementsConfig
from .elements.adapter import collect_element_aux_losses, maybe_build_mtp
from .elements.attention import IndexPool, TopKSparseCausalAttention
from .elements.config import b0_config as element_b0_config
from .elements.kda import GatedRMSNorm, KDALayer
from .elements.mhc import collapse_streams, expand_streams, MHCResidualBlock


def b0_elements_config(*, seq_len: int = 1024, vocab_size: int = 50257):
    cfg = replace(
        element_b0_config(),
        max_seq_len=seq_len,
        vocab_size=vocab_size,
    )
    cfg.validate()
    return cfg


class SpeedrunRMSNorm(nn.Module):
    """RMSNorm over the last tensor dimension."""

    def __init__(self, hidden_size: int, *, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        out = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (out * self.weight.float()).to(x.dtype)


class PerHeadRMSNorm(nn.Module):
    """Per-head QK norm for tensors shaped ``[B, L, H, K]``."""

    def __init__(self, num_heads: int, head_dim: int, *, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(num_heads, head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x_BLHK: torch.Tensor) -> torch.Tensor:
        x_float = x_BLHK.float()
        out = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (out * self.weight.float()).to(x_BLHK.dtype)


class SpeedrunPositionEncoding(nn.Module):
    """RoPE/NoPE switch point used by B0 and A2."""

    def __init__(self, config: SpeedrunElementsConfig) -> None:
        super().__init__()
        self.rope_enabled = config.rope_enabled
        self.rope_theta = config.rope_theta

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        *,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return maybe_apply_rope(
            q_BLHK,
            k_BLHK,
            rope_enabled=self.rope_enabled,
            theta=self.rope_theta,
            positions=positions,
        )


@local_compile("glm53_speedrun_attention", batch_invariant=True)
def speedrun_attention(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHK: torch.Tensor,
) -> torch.Tensor:
    out_BHLK = F.scaled_dot_product_attention(
        q_BLHK.transpose(1, 2),
        k_BLHK.transpose(1, 2),
        v_BLHK.transpose(1, 2),
        is_causal=True,
    )
    return out_BHLK.transpose(1, 2).contiguous()


@local_compile("glm53_speedrun_mlp", batch_invariant=True)
def speedrun_squared_relu_mlp(
    x_BLD: torch.Tensor,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
) -> torch.Tensor:
    hidden = F.relu(up_proj(x_BLD)).square()
    return down_proj(hidden)


class SpeedrunAttention(nn.Module):
    """Softmax self-attention with QK norm and optional RoPE."""

    def __init__(self, config: SpeedrunElementsConfig, *, norm_eps: float) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = PerHeadRMSNorm(
            config.num_attention_heads,
            config.head_dim,
            eps=norm_eps,
        )
        self.k_norm = PerHeadRMSNorm(
            config.num_attention_heads,
            config.head_dim,
            eps=norm_eps,
        )
        self.position = SpeedrunPositionEncoding(config)

    def _split_heads(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return x_BLD.view(x_BLD.shape[0], x_BLD.shape[1], self.num_heads, self.head_dim)

    def forward(
        self,
        x_BLD: torch.Tensor,
        *,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        q_BLHK = self.q_norm(self._split_heads(self.q_proj(x_BLD)))
        k_BLHK = self.k_norm(self._split_heads(self.k_proj(x_BLD)))
        v_BLHK = self._split_heads(self.v_proj(x_BLD))
        q_BLHK, k_BLHK = self.position(q_BLHK, k_BLHK, positions=positions)
        out_BLHK = speedrun_attention(q_BLHK, k_BLHK, v_BLHK)
        return self.out_proj(out_BLHK.flatten(-2))


def _build_speedrun_attention(
    config: SpeedrunElementsConfig,
    *,
    layer_idx: int,
    norm_eps: float,
) -> nn.Module:
    if config.attention_kind(layer_idx) == "kda":
        return KDALayer(config)
    if config.sparse_attention_enabled:
        return TopKSparseCausalAttention(config)
    return SpeedrunAttention(config, norm_eps=norm_eps)


def _call_attention(
    attention: nn.Module,
    x_BLD: torch.Tensor,
    *,
    positions: torch.Tensor | None,
    attention_mask_BL: torch.Tensor | None,
) -> torch.Tensor:
    if isinstance(attention, KDALayer):
        return attention(x_BLD)
    if isinstance(attention, TopKSparseCausalAttention):
        return attention(
            x_BLD,
            positions=positions,
            attention_mask_BL=attention_mask_BL,
        )
    return attention(x_BLD, positions=positions)


class SpeedrunMLP(nn.Module):
    """ReLU^2 feed-forward block from the classic speedrun recipe."""

    def __init__(self, config: SpeedrunElementsConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.mlp_hidden_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.mlp_hidden_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return speedrun_squared_relu_mlp(x_BLD, self.up_proj, self.down_proj)


class SpeedrunResidualBlock(nn.Module):
    """Replaceable residual block seam for the speedrun arms."""

    def __init__(
        self,
        config: SpeedrunElementsConfig,
        *,
        layer_idx: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.mhc_enabled = config.mhc_enabled
        self.attention_norm = SpeedrunRMSNorm(config.hidden_size, eps=norm_eps)
        self.attention = _build_speedrun_attention(
            config,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
        )
        self.mlp_norm = SpeedrunRMSNorm(config.hidden_size, eps=norm_eps)
        self.mlp = SpeedrunMLP(config)
        if config.mhc_enabled:
            self.attention_mhc = MHCResidualBlock(
                config.hidden_size,
                nn.Identity(),
                num_streams=config.mhc_streams,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                eps=config.mhc_eps,
                identity_at_init=config.mhc_init == "gated_identity",
            )
            self.mlp_mhc = MHCResidualBlock(
                config.hidden_size,
                nn.Identity(),
                num_streams=config.mhc_streams,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                eps=config.mhc_eps,
                identity_at_init=config.mhc_init == "gated_identity",
            )
        else:
            self.attention_mhc = None
            self.mlp_mhc = None

    def _attention_update(
        self,
        x_BLD: torch.Tensor,
        *,
        positions: torch.Tensor | None,
        attention_mask_BL: torch.Tensor | None,
    ) -> torch.Tensor:
        return _call_attention(
            self.attention,
            self.attention_norm(x_BLD),
            positions=positions,
            attention_mask_BL=attention_mask_BL,
        )

    def _mlp_update(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(x_BLD))

    def forward(
        self,
        x_BLD_or_BLND: torch.Tensor,
        *,
        positions: torch.Tensor | None,
        attention_mask_BL: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.mhc_enabled:
            x_BLD = x_BLD_or_BLND
            x_BLD = x_BLD + self._attention_update(
                x_BLD,
                positions=positions,
                attention_mask_BL=attention_mask_BL,
            )
            x_BLD = x_BLD + self._mlp_update(x_BLD)
            return x_BLD

        assert self.attention_mhc is not None
        assert self.mlp_mhc is not None
        streams_BLND = x_BLD_or_BLND
        streams_BLND = self.attention_mhc(
            streams_BLND,
            lambda collapsed_BLD: self._attention_update(
                collapsed_BLD,
                positions=positions,
                attention_mask_BL=attention_mask_BL,
            ),
        )
        streams_BLND = self.mlp_mhc(
            streams_BLND,
            lambda collapsed_BLD: self._mlp_update(collapsed_BLD),
        )
        return streams_BLND


class SpeedrunHead(nn.Module):
    """Untied LM head with optional logit softcap."""

    def __init__(self, config: SpeedrunElementsConfig, *, logit_softcap: float) -> None:
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logit_softcap = logit_softcap

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        logits_BLV = self.proj(x_BLD)
        if self.logit_softcap > 0.0:
            logits_BLV = self.logit_softcap * torch.tanh(
                logits_BLV / self.logit_softcap
            )
        return logits_BLV


class Glm53ArchSpeedrunModel(BaseModel):
    """Trainer-compatible B0 causal LM for architecture-element speedruns."""

    supports_pipeline_parallel = False
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        elements: SpeedrunElementsConfig = field(default_factory=b0_elements_config)
        experiment_name: str = "glm53_arch_speedrun"
        experiment_flavor: str = "B0"
        initializer_std: float = 0.02
        norm_eps: float = 1.0e-6
        logit_softcap: float = 30.0
        gpu_memory_cap_gib: float = 24.0
        local_compile_regions: list[str] = field(
            default_factory=lambda: [
                "glm53_speedrun_attention",
                "glm53_speedrun_mlp",
                "loss",
            ]
        )

        @property
        def max_context_length(self) -> int:
            return self.elements.max_seq_len

        def __post_init__(self) -> None:
            self.elements.validate()
            if self.initializer_std <= 0:
                raise ValueError("initializer_std must be positive")
            if self.norm_eps <= 0:
                raise ValueError("norm_eps must be positive")
            if self.logit_softcap < 0:
                raise ValueError("logit_softcap must be non-negative")
            if self.gpu_memory_cap_gib < 0:
                raise ValueError("gpu_memory_cap_gib must be non-negative")

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            if parallelism.tensor_parallel_degree != 1:
                raise ValueError(
                    "B0 speedrun currently requires tensor_parallel_degree=1"
                )
            if parallelism.context_parallel_degree != 1:
                raise ValueError(
                    "B0 speedrun currently requires context_parallel_degree=1"
                )
            if parallelism.pipeline_parallel_degree != 1:
                raise ValueError("B0 speedrun does not support pipeline parallelism")
            if parallelism.expert_parallel_degree != 1:
                raise ValueError("B0 speedrun does not use expert parallelism")

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            del seq_len
            nparams = sum(param.numel() for param in model.parameters())
            return nparams, max(1, 6 * nparams)

    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        _set_cuda_memory_cap(config.gpu_memory_cap_gib)
        cfg = config.elements
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [
                SpeedrunResidualBlock(
                    cfg, layer_idx=layer_idx, norm_eps=config.norm_eps
                )
                for layer_idx in range(cfg.num_layers)
            ]
        )
        self.norm = SpeedrunRMSNorm(cfg.hidden_size, eps=config.norm_eps)
        self.head = SpeedrunHead(cfg, logit_softcap=config.logit_softcap)
        self.mtp_head = maybe_build_mtp(cfg)
        self.indexer_aux_loss = (
            SpeedrunIndexerAuxLoss(
                SpeedrunIndexerAuxLoss.Config(
                    coeff=1.0,
                    reduce_mesh="loss",
                )
            )
            if cfg.indexer_aux_loss_weight > 0.0
            else None
        )
        self.mtp_aux_loss = (
            SpeedrunMtpAuxLoss(
                SpeedrunMtpAuxLoss.Config(
                    coeff=1.0,
                    reduce_mesh="loss",
                )
            )
            if cfg.mtp_enabled and cfg.mtp_aux_loss_weight > 0.0
            else None
        )

    @property
    def lm_head(self) -> nn.Module:
        return self.head

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: object | None = None,
        mtp_labels: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_masks
        x_BLD = self.token_embeddings(tokens)
        if self.config.elements.mhc_enabled:
            x_BLD_or_BLND = expand_streams(
                x_BLD,
                num_streams=self.config.elements.mhc_streams,
            )
        else:
            x_BLD_or_BLND = x_BLD
        for block in self.blocks:
            x_BLD_or_BLND = block(
                x_BLD_or_BLND,
                positions=positions,
                attention_mask_BL=None,
            )
        x_BLD = (
            collapse_streams(x_BLD_or_BLND)
            if self.config.elements.mhc_enabled
            else x_BLD_or_BLND
        )
        x_BLD = self.norm(x_BLD)
        if self.training and aux_loss_denominators is not None:
            aux_denominator = aux_loss_denominators.reshape(-1)[0]
            if self.indexer_aux_loss is not None:
                indexer_aux = collect_element_aux_losses(self)
                if indexer_aux is not None:
                    x_BLD = self.indexer_aux_loss.inject(
                        indexer_aux * aux_denominator,
                        carrier=x_BLD,
                        denominator=aux_denominator,
                    )
            if (
                self.mtp_aux_loss is not None
                and self.mtp_head is not None
                and mtp_labels is not None
            ):
                mtp_aux = self.mtp_head.aux_loss(x_BLD, mtp_labels)
                x_BLD = self.mtp_aux_loss.inject(
                    mtp_aux * aux_denominator,
                    carrier=x_BLD,
                    denominator=aux_denominator,
                )
        if self._skip_lm_head:
            return x_BLD.reshape(-1, x_BLD.shape[-1]).clone()
        logits_BLV = self.head(x_BLD)
        return logits_BLV.reshape(-1, logits_BLV.shape[-1]).clone()

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del kwargs
        labels = input_dict.pop("labels")
        tokens = input_dict.pop("input")
        input_dict.pop("padding_mask", None)
        if self.training and (
            self.indexer_aux_loss is not None or self.mtp_aux_loss is not None
        ):
            input_dict["aux_loss_denominators"] = torch.empty(0)
        if self.config.elements.mtp_enabled:
            input_dict["mtp_labels"] = labels
        return tokens, labels.reshape(-1), input_dict

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile_regions: list[str],
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
    ) -> "Glm53ArchSpeedrunModel":
        del ac_config, dump_folder
        apply_local_compile(local_compile_regions)
        with parallelism_context.activate_spmd():
            self._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
        return self

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        self.config.set_sharding_(parallelism)
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cast_forward_inputs=False,
        )
        fsdp_config: dict[str, Any] = {
            "mesh": dp_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": reshard_after_forward,
            "dp_mesh_dims": dp_mesh_dims,
        }
        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        fully_shard(self.token_embeddings, **fsdp_config)
        for block in self.blocks:
            fully_shard(block, **fsdp_config)
        fully_shard([self.norm, self.head], **fsdp_config)
        fully_shard(self, **fsdp_config)
        enable_fsdp_symm_mem(self, parallelism.fsdp_symm_mem_scope)
        disable_fsdp_gradient_division(self)

    def verify_module_protocol(self) -> None:
        return

    @classmethod
    def _register_optimizer_hooks(
        cls,
        optimizers,
        model_parts,
        parallelism_context,
    ) -> None:
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        cfg = self.config
        residual_std = cfg.initializer_std / math.sqrt(2 * cfg.elements.num_layers)
        with torch.no_grad():
            for module_name, module in self.named_modules():
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=cfg.initializer_std)
                elif isinstance(module, nn.Linear):
                    std = (
                        residual_std
                        if module_name.endswith(("out_proj", "down_proj"))
                        else cfg.initializer_std
                    )
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv1d):
                    nn.init.normal_(module.weight, mean=0.0, std=cfg.initializer_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (SpeedrunRMSNorm, PerHeadRMSNorm)):
                    module.reset_parameters()
                elif isinstance(module, nn.LayerNorm):
                    module.reset_parameters()
                elif isinstance(module, GatedRMSNorm):
                    nn.init.ones_(module.weight)
                elif isinstance(module, IndexPool):
                    nn.init.zeros_(module.slot_bias)
                elif isinstance(module, KDALayer):
                    nn.init.zeros_(module.dt_bias)
                    nn.init.zeros_(module.A_log)
                elif isinstance(module, (SpeedrunIndexerAuxLoss, SpeedrunMtpAuxLoss)):
                    module._init_self_buffers(buffer_device=buffer_device)
                elif hasattr(module, "reset_parameters") and module is not self:
                    module.reset_parameters()


def _set_cuda_memory_cap(cap_gib: float) -> None:
    if cap_gib <= 0 or not torch.cuda.is_available():
        return
    total_gib = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory
    fraction = min(1.0, cap_gib * 1024**3 / total_gib)
    env_fraction = os.environ.get("GLM53_SPEEDRUN_GPU_MEMORY_FRACTION")
    if env_fraction is not None:
        fraction = float(env_fraction)
    torch.cuda.set_per_process_memory_fraction(fraction)


def muon_parameter_names(model: nn.Module) -> set[str]:
    """Return FQNs assigned to Muon: hidden 2D projection matrices only."""
    suffixes = (
        ".attention.q_proj.weight",
        ".attention.k_proj.weight",
        ".attention.v_proj.weight",
        ".attention.out_proj.weight",
        ".attention.o_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )
    return {
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and param.ndim == 2
        and name.startswith("blocks.")
        and name.endswith(suffixes)
    }


def adamw_parameter_names(model: nn.Module) -> set[str]:
    """Return FQNs assigned to AdamW by the B0 optimizer contract."""
    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    return trainable - muon_parameter_names(model)
