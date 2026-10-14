# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HF <-> TorchTitan adapter for the isolated GLM EP prototype."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.experiments.glm53_flash.state_dict_adapter import (
    Glm53FlashStateDictAdapter,
)


_EP_W13_RE = re.compile(r"^layers\.(\d+)\.mlp\.experts\.w13\.weight$")
_EP_W2_RE = re.compile(r"^layers\.(\d+)\.mlp\.experts\.w2\.weight$")
_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.weight$"
)


class Glm53FlashEPStateDictAdapter(Glm53FlashStateDictAdapter):
    """State-dict adapter for common grouped expert keys in the EP prototype."""

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        expert_parts: dict[tuple[int, int], dict[str, torch.Tensor]] = defaultdict(dict)
        for hf_key, value in hf_state_dict.items():
            expert_match = _EXPERT_RE.match(hf_key)
            if expert_match is not None:
                layer, expert, projection = expert_match.groups()
                expert_parts[(int(layer), int(expert))][projection] = value
                continue
            filtered[hf_key] = value

        state_dict = super().from_hf(filtered)
        self._merge_ep_expert_parts(state_dict, expert_parts)
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        filtered = {
            key: value
            for key, value in state_dict.items()
            if _EP_W13_RE.match(key) is None and _EP_W2_RE.match(key) is None
        }
        hf_state_dict = super().to_hf(filtered)
        for native_key, value in state_dict.items():
            w13_match = _EP_W13_RE.match(native_key)
            if w13_match is not None:
                layer = int(w13_match.group(1))
                local_w13_E2FD, experts = self._local_ep_experts(
                    "layers.{}.mlp.experts.w13.weight",
                    value,
                )
                for local_expert, expert in enumerate(experts):
                    gate_FD, up_FD = local_w13_E2FD[local_expert].unbind(dim=0)
                    hf_state_dict[
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
                    ] = gate_FD
                    hf_state_dict[
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
                    ] = up_FD
                continue

            w2_match = _EP_W2_RE.match(native_key)
            if w2_match is not None:
                layer = int(w2_match.group(1))
                local_w2_EDF, experts = self._local_ep_experts(
                    "layers.{}.mlp.experts.w2.weight",
                    value,
                )
                for local_expert, expert in enumerate(experts):
                    hf_state_dict[
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
                    ] = local_w2_EDF[local_expert]
        return hf_state_dict

    def _map_hf_key(self, hf_key: str) -> str | None:
        return super()._map_hf_key(hf_key)

    def _map_native_key(self, native_key: str) -> str | None:
        if _EP_W13_RE.match(native_key) is not None:
            return None
        if _EP_W2_RE.match(native_key) is not None:
            return None
        return super()._map_native_key(native_key)

    def _merge_ep_expert_parts(
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
            w13_key = "layers.{}.mlp.experts.w13.weight"
            w2_key = "layers.{}.mlp.experts.w2.weight"
            start, end = self.local_experts_indices.get(
                w13_key, (0, self.glm_config.num_experts)
            )
            expected = set(range(start, end))
            missing = expected - set(experts)
            if missing:
                raise KeyError(
                    f"layer {layer} is missing EP expert weights for "
                    f"{sorted(missing)[:8]}"
                )
            w13_parts = []
            w2_parts = []
            validate_only = (
                w13_key in self.grouped_expert_weight_mesh
                and w2_key in self.grouped_expert_weight_mesh
            )
            for expert in range(start, end):
                parts = experts[expert]
                required = {"gate", "up", "down"}
                if set(parts) != required:
                    raise KeyError(
                        f"layer {layer} expert {expert} has projections "
                        f"{sorted(parts)}, expected {sorted(required)}"
                    )
                if validate_only:
                    continue
                w13_parts.append(torch.stack([parts["gate"], parts["up"]], dim=0))
                w2_parts.append(parts["down"])
            if validate_only:
                continue
            state_dict[
                f"layers.{layer}.mlp.experts.w13.weight"
            ] = self._wrap_grouped_expert_dtensor(w13_key, torch.stack(w13_parts))
            state_dict[
                f"layers.{layer}.mlp.experts.w2.weight"
            ] = self._wrap_grouped_expert_dtensor(w2_key, torch.stack(w2_parts))

    def _local_ep_experts(
        self,
        abstract_key: str,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, range]:
        if isinstance(value, DTensor):
            self._record_grouped_expert_dtensor(abstract_key, value)
            start, end = self.local_experts_indices[abstract_key]
            return value._local_tensor, range(start, end)
        return value, range(value.shape[0])
