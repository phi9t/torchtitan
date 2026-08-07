# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList


# Shape suffix legend for this file:
# B: batch, C: latent channels, H/W: latent height/width, T: patch tokens,
# D: hidden size, P: patch size, L: conditioning hidden size.


def modulate(x_BTD: torch.Tensor, shift_BD: torch.Tensor, scale_BD: torch.Tensor):
    return x_BTD * (1 + scale_BD.unsqueeze(1)) + shift_BD.unsqueeze(1)


def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}.")
    omega_D = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega_D = 1.0 / (10000 ** (omega_D / (embed_dim / 2)))
    out_TD = torch.outer(pos.to(torch.float64), omega_D)
    return torch.cat([torch.sin(out_TD), torch.cos(out_TD)], dim=1).float()


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}.")
    grid_h_H = torch.arange(grid_height, dtype=torch.float32)
    grid_w_W = torch.arange(grid_width, dtype=torch.float32)
    yy_HW, xx_HW = torch.meshgrid(grid_h_H, grid_w_W, indexing="ij")
    emb_h_TD = get_1d_sincos_pos_embed(embed_dim // 2, yy_HW.reshape(-1))
    emb_w_TD = get_1d_sincos_pos_embed(embed_dim // 2, xx_HW.reshape(-1))
    return torch.cat([emb_h_TD, emb_w_TD], dim=1).to(device=device)


def get_timestep_embedding(t_B: torch.Tensor, embed_dim: int) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}.")
    half_dim = embed_dim // 2
    exponent_D = -math.log(10000) * torch.arange(
        half_dim, device=t_B.device, dtype=torch.float32
    )
    exponent_D = exponent_D / max(half_dim - 1, 1)
    emb_BD = t_B.float().unsqueeze(1) * torch.exp(exponent_D).unsqueeze(0)
    return torch.cat([torch.sin(emb_BD), torch.cos(emb_BD)], dim=1)


def patchify(x_BCHW: torch.Tensor, patch_size: int) -> torch.Tensor:
    bsz, channels, height, width = x_BCHW.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"latent shape {(height, width)} must be divisible by patch_size "
            f"{patch_size}."
        )
    grid_h = height // patch_size
    grid_w = width // patch_size
    x_BCTPP = x_BCHW.reshape(bsz, channels, grid_h, patch_size, grid_w, patch_size)
    x_BTHWCP = torch.einsum("bchpwq->bhwcpq", x_BCTPP)
    return x_BTHWCP.reshape(bsz, grid_h * grid_w, channels * patch_size**2)


def unpatchify(
    x_BTD: torch.Tensor,
    *,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    patch_size: int,
) -> torch.Tensor:
    bsz, seq_len, patch_dim = x_BTD.shape
    grid_h = latent_height // patch_size
    grid_w = latent_width // patch_size
    expected_seq_len = grid_h * grid_w
    expected_patch_dim = latent_channels * patch_size**2
    if seq_len != expected_seq_len:
        raise ValueError(f"seq_len must be {expected_seq_len}, got {seq_len}.")
    if patch_dim != expected_patch_dim:
        raise ValueError(f"patch dim must be {expected_patch_dim}, got {patch_dim}.")
    x_BHWCPP = x_BTD.reshape(
        bsz, grid_h, grid_w, latent_channels, patch_size, patch_size
    )
    x_BCHPWQ = torch.einsum("bhwcpq->bchpwq", x_BHWCPP)
    return x_BCHPWQ.reshape(bsz, latent_channels, latent_height, latent_width)


class TimestepEmbedder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        frequency_embed_size: int = 256

    def __init__(self, config: Config):
        super().__init__()
        self.frequency_embed_size = config.frequency_embed_size
        self.fc1 = Linear.Config(
            in_features=config.frequency_embed_size,
            out_features=config.hidden_size,
            bias=True,
        ).build()
        self.fc2 = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
        ).build()

    def forward(self, t_B: torch.Tensor) -> torch.Tensor:
        t_freq_BD = get_timestep_embedding(t_B, self.frequency_embed_size)
        return self.fc2(F.silu(self.fc1(t_freq_BD)))


class DiTSelfAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int

    def __init__(self, config: Config):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("num_heads must divide hidden_size.")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = Linear.Config(
            in_features=config.hidden_size,
            out_features=3 * config.hidden_size,
            bias=True,
        ).build()
        self.proj = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
        ).build()

    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_size = x_BTD.shape
        qkv_BTNH = self.qkv(x_BTD).reshape(
            bsz, seq_len, 3, self.num_heads, self.head_dim
        )
        q_BNTH, k_BNTH, v_BNTH = qkv_BTNH.permute(2, 0, 3, 1, 4).unbind(0)
        out_BNTH = F.scaled_dot_product_attention(q_BNTH, k_BNTH, v_BNTH)
        out_BTD = out_BNTH.transpose(1, 2).reshape(bsz, seq_len, hidden_size)
        return self.proj(out_BTD)


class DiTBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        mlp_ratio: float = 4.0

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm1 = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        ).build()
        self.attn = DiTSelfAttention.Config(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
        ).build()
        self.norm2 = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        ).build()
        mlp_hidden_size = int(config.hidden_size * config.mlp_ratio)
        self.mlp = nn.Sequential(
            Linear.Config(
                in_features=config.hidden_size,
                out_features=mlp_hidden_size,
                bias=True,
            ).build(),
            nn.GELU(approximate="tanh"),
            Linear.Config(
                in_features=mlp_hidden_size,
                out_features=config.hidden_size,
                bias=True,
            ).build(),
        )
        self.adaLN_modulation = Linear.Config(
            in_features=config.hidden_size,
            out_features=6 * config.hidden_size,
            bias=True,
            param_init={"weight": nn.init.zeros_, "bias": nn.init.zeros_},
        ).build()

    def verify_module_protocol(self) -> None:
        pass

    def forward(self, x_BTD: torch.Tensor, c_BD: torch.Tensor) -> torch.Tensor:
        (
            shift_msa_BD,
            scale_msa_BD,
            gate_msa_BD,
            shift_mlp_BD,
            scale_mlp_BD,
            gate_mlp_BD,
        ) = self.adaLN_modulation(c_BD).chunk(6, dim=1)
        attn_in_BTD = modulate(self.norm1(x_BTD), shift_msa_BD, scale_msa_BD)
        attn_out_BTD = self.attn(attn_in_BTD)
        x_BTD = x_BTD + gate_msa_BD.unsqueeze(1) * attn_out_BTD
        mlp_in_BTD = modulate(self.norm2(x_BTD), shift_mlp_BD, scale_mlp_BD)
        return x_BTD + gate_mlp_BD.unsqueeze(1) * self.mlp(mlp_in_BTD)


class DiTFinalLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        patch_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.norm_final = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        ).build()
        self.linear = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.patch_dim,
            bias=True,
            param_init={"weight": nn.init.zeros_, "bias": nn.init.zeros_},
        ).build()
        self.adaLN_modulation = Linear.Config(
            in_features=config.hidden_size,
            out_features=2 * config.hidden_size,
            bias=True,
            param_init={"weight": nn.init.zeros_, "bias": nn.init.zeros_},
        ).build()

    def forward(self, x_BTD: torch.Tensor, c_BD: torch.Tensor) -> torch.Tensor:
        shift_BD, scale_BD = self.adaLN_modulation(c_BD).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x_BTD), shift_BD, scale_BD))


class DiTModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        latent_channels: int = 4
        latent_height: int = 16
        latent_width: int = 16
        patch_size: int = 2
        hidden_size: int = 192
        num_layers: int = 4
        num_heads: int = 3
        mlp_ratio: float = 4.0
        num_classes: int = 10

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            del config, kwargs

        @property
        def latent_patch_tokens(self) -> int:
            return (self.latent_height // self.patch_size) * (
                self.latent_width // self.patch_size
            )

        @property
        def patch_dim(self) -> int:
            return self.latent_channels * self.patch_size**2

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            del seq_len
            nparams = sum(p.numel() for p in model.parameters())
            attn_flops = 4 * self.latent_patch_tokens * self.hidden_size**2
            attn_flops += 2 * self.latent_patch_tokens**2 * self.hidden_size
            mlp_flops = int(
                4
                * self.latent_patch_tokens
                * self.hidden_size
                * self.hidden_size
                * self.mlp_ratio
            )
            return nparams, self.num_layers * (attn_flops + mlp_flops)

    def __init__(self, config: Config):
        super().__init__()
        if config.latent_height % config.patch_size != 0:
            raise ValueError("latent_height must be divisible by patch_size.")
        if config.latent_width % config.patch_size != 0:
            raise ValueError("latent_width must be divisible by patch_size.")
        if config.hidden_size % 4 != 0:
            raise ValueError("hidden_size must be divisible by 4.")
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("num_heads must divide hidden_size.")

        self.config = config
        self.x_embedder = Linear.Config(
            in_features=config.patch_dim,
            out_features=config.hidden_size,
            bias=True,
        ).build()
        self.t_embedder = TimestepEmbedder.Config(
            hidden_size=config.hidden_size,
        ).build()
        self.y_embedder = Embedding.Config(
            num_embeddings=config.num_classes,
            embedding_dim=config.hidden_size,
        ).build()
        self.blocks = ModuleList(
            [
                DiTBlock.Config(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                ).build()
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = DiTFinalLayer.Config(
            hidden_size=config.hidden_size,
            patch_dim=config.patch_dim,
        ).build()
        self.register_buffer(
            "pos_embed",
            torch.empty(1, config.latent_patch_tokens, config.hidden_size),
            persistent=False,
        )

    def verify_module_protocol(self) -> None:
        failures = []
        for fqn, mod in self.named_modules():
            if isinstance(mod, (nn.GELU, nn.Sequential)):
                continue
            if not isinstance(mod, Module):
                failures.append((fqn, type(mod).__name__))
        if failures:
            details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
            raise RuntimeError(
                f"The following modules do not satisfy the Module protocol: {details}"
            )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        pos_embed_TD = get_2d_sincos_pos_embed(
            self.config.hidden_size,
            self.config.latent_height // self.config.patch_size,
            self.config.latent_width // self.config.patch_size,
            device=buffer_device,
        )
        self.pos_embed = pos_embed_TD.unsqueeze(0)

    def forward(
        self,
        x_BCHW: torch.Tensor,
        timesteps_B: torch.Tensor,
        y_B: torch.Tensor,
    ) -> torch.Tensor:
        x_BTD = self.x_embedder(patchify(x_BCHW, self.config.patch_size))
        x_BTD = x_BTD + self.pos_embed.to(device=x_BTD.device, dtype=x_BTD.dtype)
        c_BD = self.t_embedder(timesteps_B) + self.y_embedder(y_B)
        for block in self.blocks:
            x_BTD = block(x_BTD, c_BD)
        pred_BTD = self.final_layer(x_BTD, c_BD)
        return unpatchify(
            pred_BTD,
            latent_channels=self.config.latent_channels,
            latent_height=self.config.latent_height,
            latent_width=self.config.latent_width,
            patch_size=self.config.patch_size,
        )
