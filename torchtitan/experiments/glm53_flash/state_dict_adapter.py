# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF <-> TorchTitan state-dict adapter for GLM-5.3-Flash BF16 weights."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.base.state_dict_adapter import StateDictAdapter


logger = logging.getLogger(__name__)


_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.weight$"
)
_NATIVE_GATE_UP_RE = re.compile(r"^layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
_NATIVE_DOWN_RE = re.compile(r"^layers\.(\d+)\.mlp\.experts\.down_proj$")


class Glm53FlashStateDictAdapter(StateDictAdapter):
    """Adapter for text decoder weights from `zai-org/GLM-5.3-Flash-BF16`."""

    def __init__(self, model_config, hf_assets_path: str | None) -> None:
        super().__init__(model_config, hf_assets_path)
        self.glm_config = model_config.glm_config
        self.last_skip_counts: dict[str, int] = {}

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        expert_parts: dict[tuple[int, int], dict[str, torch.Tensor]] = defaultdict(dict)
        skip_counts: dict[str, int] = defaultdict(int)
        unmapped: list[str] = []

        for hf_key, value in hf_state_dict.items():
            skip_reason = self._skip_reason(hf_key)
            if skip_reason is not None:
                skip_counts[skip_reason] += 1
                continue
            expert_match = _EXPERT_RE.match(hf_key)
            if expert_match is not None:
                layer, expert, projection = expert_match.groups()
                expert_parts[(int(layer), int(expert))][projection] = value
                continue
            native_key = self._map_hf_key(hf_key)
            if native_key is None:
                unmapped.append(hf_key)
                continue
            if native_key.endswith(".dt_bias"):
                value = value.reshape(
                    self.glm_config.linear_num_heads,
                    self.glm_config.linear_head_dim,
                )
            state_dict[native_key] = value

        self._merge_expert_parts(state_dict, expert_parts)
        if unmapped:
            raise KeyError(f"unmapped GLM HF keys: {unmapped[:20]}")
        self.last_skip_counts = dict(skip_counts)
        if skip_counts:
            logger.info("Skipped GLM HF keys: %s", dict(skip_counts))
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}
        unmapped: list[str] = []
        for native_key, value in state_dict.items():
            if ".lora_a" in native_key or ".lora_b" in native_key:
                continue
            gate_up_match = _NATIVE_GATE_UP_RE.match(native_key)
            if gate_up_match is not None:
                layer = int(gate_up_match.group(1))
                gate_up_E2FD = value
                for expert in range(gate_up_E2FD.shape[0]):
                    gate_FD, up_FD = gate_up_E2FD[expert].chunk(2, dim=0)
                    hf_state_dict[
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
                    ] = gate_FD
                    hf_state_dict[
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
                    ] = up_FD
                continue
            down_match = _NATIVE_DOWN_RE.match(native_key)
            if down_match is not None:
                layer = int(down_match.group(1))
                down_EDF = value
                for expert in range(down_EDF.shape[0]):
                    hf_state_dict[
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
                    ] = down_EDF[expert]
                continue

            hf_key = self._map_native_key(native_key)
            if hf_key is None:
                unmapped.append(native_key)
                continue
            if native_key.endswith(".dt_bias"):
                value = value.reshape(-1)
            hf_state_dict[hf_key] = value
        if unmapped:
            raise KeyError(f"unmapped GLM native keys: {unmapped[:20]}")
        return hf_state_dict

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
        *,
        thread_count: int | None = None,
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "GLM-5.3-Flash experiment supports BF16 loading only; "
                "weight_scale_inv keys are skipped."
            )
        return HuggingFaceStorageReader(path, thread_count=thread_count or 2)

    @staticmethod
    def _skip_reason(hf_key: str) -> str | None:
        if hf_key.endswith("weight_scale_inv") or ".weight_scale_inv" in hf_key:
            return "weight_scale_inv"
        if hf_key.startswith("model.vision") or hf_key.startswith("vision_tower"):
            return "vision_tower"
        if "mm_projector" in hf_key:
            return "vision_tower"
        if re.match(r"^model\.language_model\.layers\.45\.", hf_key):
            return "mtp"
        if ".shared_head." in hf_key:
            return "mtp"
        return None

    @staticmethod
    def _root_map() -> dict[str, str]:
        return {
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            "model.language_model.lm_head.weight": "lm_head.weight",
        }

    @staticmethod
    def _layer_suffix_map() -> dict[str, str]:
        return {
            "hc_attn_fn": "attn_hc.fn",
            "hc_attn_base": "attn_hc.base",
            "hc_attn_scale": "attn_hc.scale",
            "hc_ffn_fn": "ffn_hc.fn",
            "hc_ffn_base": "ffn_hc.base",
            "hc_ffn_scale": "ffn_hc.scale",
            "input_layernorm.weight": "input_layernorm.weight",
            "post_attention_layernorm.weight": "post_attention_layernorm.weight",
            "mlp.gate_proj.weight": "mlp.gate_proj.weight",
            "mlp.up_proj.weight": "mlp.up_proj.weight",
            "mlp.down_proj.weight": "mlp.down_proj.weight",
            "mlp.gate.weight": "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
            "mlp.shared_experts.gate_proj.weight": "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.up_proj.weight": "mlp.shared_experts.up_proj.weight",
            "mlp.shared_experts.down_proj.weight": "mlp.shared_experts.down_proj.weight",
            "self_attn.q_proj.weight": "self_attn.q_proj.weight",
            "self_attn.k_proj.weight": "self_attn.k_proj.weight",
            "self_attn.v_proj.weight": "self_attn.v_proj.weight",
            "self_attn.q_conv1d.weight": "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight": "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight": "self_attn.v_conv1d.weight",
            "self_attn.f_a_proj.weight": "self_attn.f_a_proj.weight",
            "self_attn.f_b_proj.weight": "self_attn.f_b_proj.weight",
            "self_attn.b_proj.weight": "self_attn.b_proj.weight",
            "self_attn.g_a_proj.weight": "self_attn.g_a_proj.weight",
            "self_attn.g_b_proj.weight": "self_attn.g_b_proj.weight",
            "self_attn.o_norm.weight": "self_attn.o_norm.weight",
            "self_attn.o_proj.weight": "self_attn.o_proj.weight",
            "self_attn.A_log": "self_attn.A_log",
            "self_attn.dt_bias": "self_attn.dt_bias",
            "self_attn.q_a_proj.weight": "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight": "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight": "self_attn.kv_b_proj.weight",
            "self_attn.indexer.wq_b.weight": "self_attn.indexer.wq_b.weight",
            "self_attn.indexer.wk.weight": "self_attn.indexer.wk.weight",
            "self_attn.indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.index_kpool_compress_ape": (
                "self_attn.indexer.index_pool.index_kpool_compress_ape"
            ),
            "self_attn.indexer.index_kpool_compress_gate": (
                "self_attn.indexer.index_pool.index_kpool_compress_gate"
            ),
        }

    def _map_hf_key(self, hf_key: str) -> str | None:
        root = self._root_map().get(hf_key)
        if root is not None:
            return root
        match = _LAYER_RE.match(hf_key)
        if match is None:
            return None
        layer, suffix = match.groups()
        native_suffix = self._layer_suffix_map().get(suffix)
        if native_suffix is None:
            return None
        return f"layers.{layer}.{native_suffix}"

    def _map_native_key(self, native_key: str) -> str | None:
        root_reverse = {v: k for k, v in self._root_map().items()}
        if native_key in root_reverse:
            hf_key = root_reverse[native_key]
            return "lm_head.weight" if native_key == "lm_head.weight" else hf_key
        match = re.match(r"^layers\.(\d+)\.(.+)$", native_key)
        if match is None:
            return None
        layer, suffix = match.groups()
        suffix_reverse = {v: k for k, v in self._layer_suffix_map().items()}
        hf_suffix = suffix_reverse.get(suffix)
        if hf_suffix is None:
            return None
        return f"model.language_model.layers.{layer}.{hf_suffix}"

    def _merge_expert_parts(
        self,
        state_dict: dict[str, Any],
        expert_parts: dict[tuple[int, int], dict[str, torch.Tensor]],
    ) -> None:
        if not expert_parts:
            return
        by_layer: dict[int, dict[int, dict[str, torch.Tensor]]] = defaultdict(dict)
        for (layer, expert), parts in expert_parts.items():
            by_layer[layer][expert] = parts
        for layer, experts in by_layer.items():
            missing = set(range(self.glm_config.num_experts)) - set(experts)
            if missing:
                raise KeyError(
                    f"layer {layer} is missing expert weights for {sorted(missing)[:8]}"
                )
            gate_up = []
            down = []
            for expert in range(self.glm_config.num_experts):
                parts = experts[expert]
                required = {"gate", "up", "down"}
                if set(parts) != required:
                    raise KeyError(
                        f"layer {layer} expert {expert} has projections "
                        f"{sorted(parts)}, expected {sorted(required)}"
                    )
                gate_up.append(torch.cat([parts["gate"], parts["up"]], dim=0))
                down.append(parts["down"])
            state_dict[f"layers.{layer}.mlp.experts.gate_up_proj"] = torch.stack(
                gate_up, dim=0
            )
            state_dict[f"layers.{layer}.mlp.experts.down_proj"] = torch.stack(
                down, dim=0
            )
