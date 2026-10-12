# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.trainer import Trainer

from .dataloader import SyntheticLatentDataLoader
from .model import DiTModel


class DiTTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        dataloader: SyntheticLatentDataLoader.Config
        diffusion_timesteps: int = 1000
        beta_start: float = 0.0001
        beta_end: float = 0.02

    def __init__(self, config: Config):
        assert config.model_spec is not None
        model_config = config.model_spec.model
        assert isinstance(model_config, DiTModel.Config)
        config.training.seq_len = model_config.latent_patch_tokens
        super().__init__(config)

        betas_T = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.diffusion_timesteps,
            device=self.device,
            dtype=torch.float32,
        )
        self.alpha_bar_T = torch.cumprod(1.0 - betas_T, dim=0)

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            batch_size = labels.shape[0]
            self.metrics_processor.ntokens_since_last_log += (
                batch_size * self.config.training.seq_len
            )
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield input_dict, labels

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        if self.gradient_accumulation_steps > 1:
            raise ValueError("DiT synthetic experiment does not support grad accum.")
        return super().train_step(data_iterator)

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: float,
    ) -> torch.Tensor:
        if self.parallel_dims.pp_enabled:
            raise ValueError("DiT synthetic experiment does not support PP.")
        assert isinstance(input_dict, dict)
        assert isinstance(labels, torch.Tensor)

        x0_BCHW = input_dict["input"]
        y_B = input_dict["class_labels"]
        batch_size = x0_BCHW.shape[0]
        x0_BCHW = x0_BCHW.to(
            device=self.device,
            dtype=TORCH_DTYPE_MAP[self.config.training.dtype],
        )
        y_B = y_B.to(self.device)

        with torch.no_grad():
            eps_BCHW = torch.randn_like(x0_BCHW)
            t_B = torch.randint(
                0,
                self.config.diffusion_timesteps,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )
            alpha_bar_B = self.alpha_bar_T[t_B].to(dtype=x0_BCHW.dtype)
            alpha_bar_B111 = alpha_bar_B.view(batch_size, 1, 1, 1)
            xt_BCHW = (
                alpha_bar_B111.sqrt() * x0_BCHW
                + (1.0 - alpha_bar_B111).sqrt() * eps_BCHW
            )

        self.ntokens_seen += batch_size * self.config.training.seq_len

        assert len(self.model_parts) == 1
        with self.train_context():
            pred_eps_BCHW = self.model_parts[0](xt_BCHW, t_B, y_B)
            loss, _ = self.loss_fn(pred_eps_BCHW, eps_BCHW, global_valid_tokens)
            del pred_eps_BCHW, eps_BCHW
            with spmd.no_typecheck():
                loss.backward()

        return loss
