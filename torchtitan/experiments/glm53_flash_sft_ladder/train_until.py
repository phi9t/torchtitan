#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment-local training entrypoint with an optional step cutoff."""

from __future__ import annotations

import argparse
import logging
import os
from typing import cast

import torch

from torchtitan.config import ConfigLoader
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)


def _parse_args() -> tuple[int | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stop-after-step", type=int)
    namespace, remaining = parser.parse_known_args()
    if namespace.stop_after_step is not None and namespace.stop_after_step < 0:
        parser.error("--stop-after-step must be non-negative")
    return namespace.stop_after_step, remaining


def _install_stop_after_step(trainer: Trainer, stop_after_step: int | None) -> None:
    if stop_after_step is None:
        return
    if stop_after_step > trainer.config.training.steps:
        raise ValueError(
            "--stop-after-step must be less than or equal to training.steps"
        )

    should_continue_training = trainer.should_continue_training

    def should_continue_until_step() -> bool:
        return (
            should_continue_training()
            and trainer.engine.num_completed_steps < stop_after_step
        )

    trainer.should_continue_training = should_continue_until_step
    logger.info(
        "Training will stop after step %s while preserving training.steps=%s.",
        stop_after_step,
        trainer.config.training.steps,
    )


def main() -> None:
    """Run TorchTitan training, optionally stopping after a checkpoint step."""

    init_logger()
    stop_after_step, config_args = _parse_args()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config = cast(Trainer.Config, ConfigLoader().load(config_args))

    with config.run_evidence.build(
        dump_folder=config.dump_folder,
        job_config=config.to_dict(),
        role="trainer",
        actor_id="glm53_flash_sft_ladder",
    ):
        try:
            sl.init_structured_logger(
                source="training",
                output_dir=config.dump_folder,
                enable=config.debug.enable_structured_logging,
            )
            sl.log_trace_instant("structured_logger_started")

            trainer: Trainer | None = None

            try:
                trainer = config.build()
                _install_stop_after_step(trainer, stop_after_step)

                if config.create_seed_checkpoint:
                    assert (
                        int(os.environ["WORLD_SIZE"]) == 1
                    ), "Must create seed checkpoint using a single device."
                    assert (
                        config.checkpointer is not None
                    ), "Must configure checkpointer when creating a seed checkpoint."
                    trainer.engine.save_checkpoint(last_step=True)
                    logger.info("Created seed checkpoint")
                else:
                    trainer.train()
            except Exception:
                if trainer:
                    trainer.close()
                raise
            else:
                trainer.close()
                if torch.distributed.is_initialized():
                    with sl.log_trace_span("torch_distributed_teardown"):
                        torch.distributed.destroy_process_group()
                logger.info("Process group destroyed")
        finally:
            sl.close_structured_logger()


if __name__ == "__main__":
    main()
