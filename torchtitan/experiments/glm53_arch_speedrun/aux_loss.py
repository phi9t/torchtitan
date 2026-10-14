# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Auxiliary-loss metric hooks for the GLM-5.3 architecture speedrun."""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.models.common.aux_loss import AuxLoss


class SpeedrunIndexerAuxLoss(AuxLoss):
    """Training-loss hook and metric bucket for the sparse-indexer KL loss."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        pass

    @property
    def metric_name(self) -> str:
        return "glm53_indexer_aux_loss"


class SpeedrunMtpAuxLoss(AuxLoss):
    """Training-loss hook and metric bucket for MTP auxiliary CE."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        pass

    @property
    def metric_name(self) -> str:
        return "glm53_mtp_aux_loss"
