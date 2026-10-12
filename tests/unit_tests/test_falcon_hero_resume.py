# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pause/resume contract for the Falcon local-hero runner (ticket 08)."""

from __future__ import annotations

import os
from pathlib import Path


def _wire_single_process() -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("NGPU", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")


def _tiny_hero(dump: Path, bin_path: Path, *, steps: int, interval: int = 2):
    from experiments.falcon.hero_runner import build_hero_config

    config = build_hero_config(
        "A5",
        seed=0,
        steps=steps,
        dump_folder=dump,
        dry_run=True,
        checkpoint_interval=interval,
    )
    config.dataloader.bin_glob = str(bin_path)
    config.comm.mode = "fake_backend"
    return config


def test_hero_runner_pauses_and_resumes_optimizer_step(tmp_path: Path):
    from experiments.falcon.hero_runner import (
        clear_pause,
        install_pause_control,
        request_pause,
        write_synthetic_bin,
    )

    _wire_single_process()
    dump = tmp_path / "hero_A5"
    bin_path = tmp_path / "tokens.bin"
    write_synthetic_bin(bin_path, num_tokens=8192, vocab_size=50257)

    first = _tiny_hero(dump, bin_path, steps=4)
    trainer = first.build()
    try:
        trainer.train()
        assert trainer.step == 4
    finally:
        trainer.close()

    ckpt_root = dump / "checkpoint"
    assert ckpt_root.is_dir(), "hero run must write a checkpoint folder"
    assert any(ckpt_root.iterdir()), "hero run must leave at least one step checkpoint"

    # A leftover PAUSE must stop a longer job at the restored step.
    paused_cfg = _tiny_hero(dump, bin_path, steps=10)
    paused = paused_cfg.build()
    try:
        paused.checkpointer.load(step=-1)
        assert paused.step == 4
        request_pause(dump)
        install_pause_control(paused, dump)
        paused.train()
        assert paused.step == 4
    finally:
        paused.close()

    clear_pause(dump)
    resumed_cfg = _tiny_hero(dump, bin_path, steps=6)
    resumed = resumed_cfg.build()
    try:
        resumed.checkpointer.load(step=-1)
        assert resumed.step == 4
        resumed.train()
        assert resumed.step == 6
    finally:
        resumed.close()
