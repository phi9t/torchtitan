# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TrainingMicrobatch


@dataclass(kw_only=True, slots=True)
class SyntheticLatentMicrobatch(TrainingMicrobatch):
    labels: torch.Tensor
    class_labels: torch.Tensor
    loss_token_counts: torch.Tensor
    routing_token_counts: torch.Tensor

    def as_input_dict(self) -> dict[str, Any]:
        return {
            "input": self.labels,
            "class_labels": self.class_labels,
        }


class SyntheticLatentDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        latent_channels: int = 4
        latent_height: int = 16
        latent_width: int = 16
        num_classes: int = 10
        seed: int = 17

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        **kwargs: Any,
    ):
        del tokenizer, kwargs
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive.")
        if num_tokens_per_microbatch % max_context_length != 0:
            raise ValueError(
                "num_tokens_per_microbatch must be divisible by "
                "max_context_length for the DiT synthetic dataloader."
            )
        local_batch_size = num_tokens_per_microbatch // max_context_length
        self.config = config
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.max_context_length = max_context_length
        self.local_batch_size = local_batch_size
        self.sample_offset = 0

    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        while True:
            start = self.sample_offset
            global_ids_B = torch.arange(
                start,
                start + self.local_batch_size,
                dtype=torch.int64,
            )
            global_ids_B = global_ids_B * self.dp_world_size + self.dp_rank
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.seed + int(global_ids_B[0].item()))
            x0_BCHW = torch.randn(
                self.local_batch_size,
                self.config.latent_channels,
                self.config.latent_height,
                self.config.latent_width,
                generator=generator,
            )
            y_B = (global_ids_B % self.config.num_classes).to(torch.long)
            self.sample_offset += self.local_batch_size
            yield SyntheticLatentMicrobatch(
                labels=x0_BCHW,
                class_labels=y_B,
                loss_token_counts=torch.tensor(
                    self.local_batch_size * self.max_context_length,
                    dtype=torch.int64,
                ),
                routing_token_counts=torch.tensor(0, dtype=torch.int64),
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_offset": self.sample_offset,
            "dp_world_size": self.dp_world_size,
            "dp_rank": self.dp_rank,
            "max_context_length": self.max_context_length,
            "local_batch_size": self.local_batch_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict["dp_world_size"] != self.dp_world_size:
            raise ValueError(
                "SyntheticLatentDataLoader does not support dp_world_size changes "
                "across checkpoint restore."
            )
        if state_dict["dp_rank"] != self.dp_rank:
            raise ValueError(
                "SyntheticLatentDataLoader checkpoint rank does not match this rank."
            )
        if state_dict.get("max_context_length") != self.max_context_length:
            raise ValueError(
                "SyntheticLatentDataLoader does not support max_context_length "
                "changes across checkpoint restore."
            )
        if state_dict.get("local_batch_size") != self.local_batch_size:
            raise ValueError(
                "SyntheticLatentDataLoader does not support local batch size "
                "changes across checkpoint restore."
            )
        self.sample_offset = state_dict["sample_offset"]
