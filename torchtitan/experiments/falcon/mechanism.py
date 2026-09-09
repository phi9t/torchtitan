# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stateless mechanism observations for Falcon mixer runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.experiments.falcon.falcon import (
    _apply_phi,
    _resolve_phi,
    falcon_observation_terms,
    FalconAlignment,
    FalconPhi,
    FalconVariant,
)


_METRIC_NAMES = (
    "key_energy_pre",
    "key_energy_post",
    "beta",
    "actual_lambda",
    "eta",
    "gamma",
    "negative_inverse_log_gamma",
    "state_norm_raw",
    "state_norm_canonical",
    "update_norm",
    "read_norm",
    "clamp_fraction",
)


@dataclass
class _HeadSamples:
    labels: dict[str, Any]
    values: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in _METRIC_NAMES}
    )


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64)
    )
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p05": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
    }


class MechanismDiagnostics:
    """Collect per-layer/head Falcon mechanism observations.

    The object is supplied by a caller for a single forward pass or runner
    probe. Model modules never own it, so diagnostics do not become checkpoint
    state and cannot leak across attempts.
    """

    def __init__(self) -> None:
        self._heads: dict[tuple[int, int], _HeadSamples] = {}

    def record_falcon(
        self,
        *,
        layer: int,
        q_BLNK: torch.Tensor,
        k_BLNK: torch.Tensor,
        v_BLNV: torch.Tensor,
        beta_BLN: torch.Tensor,
        lambda_BLN: torch.Tensor,
        variant: FalconVariant,
        alignment: FalconAlignment,
        phi: FalconPhi,
        qk_norm_eps: float,
        nlms_denom_eps: float,
        lambda_scale: float,
        state_canonical_scale: float,
        state_norm_canonical_label: str,
        eps_gamma: float = 1.0e-6,
    ) -> None:
        with torch.no_grad():
            q_work = q_BLNK.detach().float()
            k_raw = k_BLNK.detach().float()
            v_work = v_BLNV.detach().float()
            beta_work = beta_BLN.detach().float()
            lambda_work = lambda_BLN.detach().float()
            phi_kind = _resolve_phi(phi, True)
            q_phi = _apply_phi(q_work, phi_kind, qk_norm_eps)
            k_phi = _apply_phi(k_raw, phi_kind, qk_norm_eps)
            x_BLNK, eta_BLN1, gamma_BLN, y_BLNV = falcon_observation_terms(
                q_phi,
                k_phi,
                v_work,
                beta_work,
                lambda_work,
                variant=variant,
                alignment=alignment,
                nlms_denom_eps=nlms_denom_eps,
                eps_gamma=eps_gamma,
            )
            batch, seq_len, heads, key_dim = q_phi.shape
            value_dim = v_work.shape[-1]
            state_BNKV = q_phi.new_zeros(batch, heads, key_dim, value_dim)
            key_energy_pre_source = k_raw.pow(2).sum(dim=-1)
            key_energy_post_source = k_phi.pow(2).sum(dim=-1)
            if alignment == "delayed":
                key_energy_pre = torch.zeros_like(key_energy_pre_source)
                key_energy_pre[:, 1:] = key_energy_pre_source[:, :-1]
                key_energy_post = torch.zeros_like(key_energy_post_source)
                key_energy_post[:, 1:] = key_energy_post_source[:, :-1]
            else:
                key_energy_pre = key_energy_pre_source
                key_energy_post = key_energy_post_source
            valid_start = 1 if alignment == "delayed" else 0

            for step in range(seq_len):
                x_BNK = x_BLNK[:, step]
                eta_BN1 = eta_BLN1[:, step]
                gamma_BN = gamma_BLN[:, step]
                if variant == "falcon1":
                    pred_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, x_BNK)
                    write_BNV = y_BLNV[:, step] - pred_BNV
                else:
                    write_BNV = y_BLNV[:, step]
                update_BNKV = torch.einsum("bnk,bnv->bnkv", x_BNK, eta_BN1 * write_BNV)
                state_BNKV = (
                    state_BNKV * gamma_BN.unsqueeze(-1).unsqueeze(-1) + update_BNKV
                )
                read_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, q_phi[:, step])

                if step < valid_start:
                    continue
                state_norm_raw = state_BNKV.pow(2).mean(dim=(-2, -1)).sqrt()
                state_norm_canonical = state_norm_raw * state_canonical_scale
                update_norm = update_BNKV.pow(2).mean(dim=(-2, -1)).sqrt()
                read_norm = read_BNV.pow(2).mean(dim=-1).sqrt()
                neg_inv_log_gamma = torch.where(
                    gamma_BN < 1.0,
                    -1.0 / torch.log(gamma_BN),
                    torch.full_like(gamma_BN, float("inf")),
                )
                clamp_fraction = (
                    (eta_BLN1[:, step, :, 0] * lambda_work[:, step])
                    >= (1.0 - eps_gamma)
                ).float()
                self._append_step(
                    layer=layer,
                    step=step,
                    labels={
                        "phi": phi_kind,
                        "alignment": alignment,
                        "variant": variant,
                        "qk_norm_eps_effective": qk_norm_eps,
                        "nlms_denom_eps_effective": nlms_denom_eps,
                        "lambda_scale": lambda_scale,
                        "state_canonical_scale": state_canonical_scale,
                        "state_norm_canonical_label": state_norm_canonical_label,
                        "lambda_label": "actual_lambda_after_scale",
                    },
                    tensors={
                        "key_energy_pre": key_energy_pre[:, step],
                        "key_energy_post": key_energy_post[:, step],
                        "beta": beta_work[:, step],
                        "actual_lambda": lambda_work[:, step],
                        "eta": eta_BLN1[:, step, :, 0],
                        "gamma": gamma_BN,
                        "negative_inverse_log_gamma": neg_inv_log_gamma,
                        "state_norm_raw": state_norm_raw,
                        "state_norm_canonical": state_norm_canonical,
                        "update_norm": update_norm,
                        "read_norm": read_norm,
                        "clamp_fraction": clamp_fraction,
                    },
                )

    def _append_step(
        self,
        *,
        layer: int,
        step: int,
        labels: dict[str, Any],
        tensors: dict[str, torch.Tensor],
    ) -> None:
        del step
        heads = next(iter(tensors.values())).shape[-1]
        for head in range(heads):
            samples = self._heads.setdefault(
                (layer, head),
                _HeadSamples(labels=dict(labels)),
            )
            if samples.labels != labels:
                raise ValueError("mechanism labels changed within a layer/head")
            for name, tensor in tensors.items():
                samples.values[name].extend(
                    float(value)
                    for value in tensor[:, head].detach().cpu().reshape(-1).tolist()
                    if math.isfinite(float(value))
                )

    def summary(self) -> dict[str, Any]:
        layers: list[dict[str, Any]] = []
        layer_ids = sorted({layer for layer, _head in self._heads})
        for layer in layer_ids:
            heads = []
            for (_layer, head), samples in sorted(self._heads.items()):
                if _layer != layer:
                    continue
                heads.append(
                    {
                        "head": head,
                        "labels": samples.labels,
                        "metrics": {
                            name: _stats(samples.values[name]) for name in _METRIC_NAMES
                        },
                    }
                )
            layers.append({"layer": layer, "heads": heads})
        return {
            "schema_version": 1,
            "record_type": "falcon_mechanism_diagnostics",
            "layers": layers,
        }
