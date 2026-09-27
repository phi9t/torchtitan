# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the fault-detection seams wired into the core train loop.

These call Trainer._detect_step_stragglers directly on a stand-in object rather
than constructing a Trainer, because what needs checking is the gating: when the
seam performs a collective and when it does not. A seam that ran its all_gather
on a step where some ranks skipped it would hang the job and manufacture the very
fault it is looking for, so the negative cases matter more than the positive one.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from torchtitan.observability.run_evidence import IncidentClass
from torchtitan.trainer import Trainer


def _stand_in(*, step: int, log_every: int, threshold: float = 4.0):
    return SimpleNamespace(
        step=step,
        device=torch.device("cpu"),
        metrics_processor=SimpleNamespace(
            should_log=lambda candidate: candidate == 1 or candidate % log_every == 0
        ),
        config=SimpleNamespace(
            run_evidence=SimpleNamespace(straggler_slowdown_threshold=threshold)
        ),
    )


def _forbid_collectives(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("a collective was issued on a non-logging step")

    monkeypatch.setattr(torch.distributed, "all_gather", explode)


def test_no_collective_is_issued_on_a_non_logging_step(monkeypatch) -> None:
    """The gate is what keeps this off the per-step critical path."""
    _forbid_collectives(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    Trainer._detect_step_stragglers(_stand_in(step=3, log_every=5), 1_000)


def test_no_collective_is_issued_below_three_ranks(monkeypatch) -> None:
    """Skipping before the collective keeps small jobs free of it entirely.

    The detector declines under three ranks anyway -- with two samples the median
    lies between them -- so issuing the all_gather first would cost a collective
    to reach a guaranteed empty answer.
    """
    _forbid_collectives(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    for world_size in (1, 2):
        monkeypatch.setattr(
            torch.distributed, "get_world_size", lambda size=world_size: size
        )
        Trainer._detect_step_stragglers(_stand_in(step=5, log_every=5), 1_000)


def test_no_collective_is_issued_before_distributed_is_initialized(
    monkeypatch,
) -> None:
    _forbid_collectives(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    Trainer._detect_step_stragglers(_stand_in(step=5, log_every=5), 1_000)


def test_a_logging_step_gathers_and_forwards_the_configured_threshold(
    monkeypatch,
) -> None:
    """The positive path, asserting what reaches the detector.

    The threshold must come from config rather than from a constant in the seam,
    or the configured value would be silently ignored.
    """
    durations = [100, 104, 98, 900]

    def fake_all_gather(output_list, tensor, *args, **kwargs):
        for slot, value in zip(output_list, durations):
            slot.fill_(value)

    seen: dict[str, object] = {}

    def fake_detect(gathered, **kwargs):
        seen["gathered"] = list(gathered)
        seen.update(kwargs)
        return (3,)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(
        "torchtitan.trainer.detect_step_stragglers", fake_detect, raising=True
    )

    Trainer._detect_step_stragglers(_stand_in(step=10, log_every=5, threshold=7.5), 900)

    assert seen["gathered"] == durations
    assert seen["incident_class"] is IncidentClass.COMPUTE_STRAGGLER
    assert seen["slowdown_threshold"] == 7.5
    assert seen["step"] == 10
    assert seen["phase"] == "train_step"


def test_the_first_step_is_a_logging_step(monkeypatch) -> None:
    """Step 1 logs, so the seam must tolerate being reached immediately.

    Worth pinning because a job shorter than log_freq would otherwise never
    exercise this path at all, and the check would look fine while never running.
    """
    called: list[int] = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda output_list, tensor, *a, **k: [s.fill_(100) for s in output_list],
    )
    monkeypatch.setattr(
        "torchtitan.trainer.detect_step_stragglers",
        lambda gathered, **kwargs: called.append(kwargs["step"]) or (),
        raising=True,
    )

    Trainer._detect_step_stragglers(_stand_in(step=1, log_every=1000), 100)
    assert called == [1]


def test_the_configured_threshold_default_is_usable() -> None:
    """A default at or below 1.0 would be refused by the detector at runtime."""
    from torchtitan.observability.run_evidence import RunEvidence

    assert RunEvidence.Config().straggler_slowdown_threshold > 1.0
