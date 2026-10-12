# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for deriving a DeviceMeshModel instance for a mesh of any degree.

The load-bearing test is the equivalence one: TLC over the DERIVED 2x2 instance
must explore exactly the state graph the shipped hand-written instance does,
38321 distinct states. Agreement on rank and communicator counts would not
establish that the generated instance means the same thing, and only equivalence
makes it safe to replace the hand-written quadruple.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from torchtitan.experiments.qwen3_formal_verifier.fidelity_diff import ToolchainPaths
from torchtitan.experiments.qwen3_formal_verifier.mesh_ladder import (
    derive_instance,
    render_cfg,
    render_module,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_DIR = REPO_ROOT / "experiments" / "qwen3_formal_verifier" / "formal"

# The shipped instance, transcribed from DeviceMeshModel.tla's CommMembers2x2 and
# CommOps2x2. The point of holding it here is to fail when either side moves.
SHIPPED_2X2_MEMBERS = {
    (0, 1): 1,  # tp
    (2, 3): 1,  # tp
    (0, 2): 3,  # fsdp, batch, loss
    (1, 3): 3,  # fsdp, batch, loss
}
SHIPPED_2X2_DISTINCT_STATES = 38321


def test_derived_2x2_reproduces_the_shipped_member_sets() -> None:
    instance = derive_instance(2, 2)
    assert len(instance.ranks) == 4
    assert len(instance.communicators) == 8

    counted: dict[tuple[int, ...], int] = {}
    for communicator in instance.communicators:
        counted[communicator.members] = counted.get(communicator.members, 0) + 1
    assert counted == SHIPPED_2X2_MEMBERS

    # Exactly one mesh name carries the full operation set, as observed.
    full = [c for c in instance.communicators if c.all_operations]
    assert len(full) == 2, full
    assert all(c.name.startswith("fsdp") for c in full)


def test_the_generator_does_not_emit_one_communicator_per_axis() -> None:
    """The specific error this generator exists to avoid.

    A mesh-derived instance that emitted one communicator per axis would produce
    two at 2x2 where the runtime creates eight, understating the set. That is the
    eight-versus-four keying error ticket 11 exists to prevent, and it would make
    every invariant vacuously easier to satisfy.
    """
    instance = derive_instance(2, 2)
    assert len(instance.communicators) > 2
    assert len(instance.communicators) == 8


@pytest.mark.parametrize(
    "dp_shard,tp,ranks,comms",
    [(2, 2, 4, 8), (2, 3, 6, 11), (3, 2, 6, 9), (2, 4, 8, 14)],
)
def test_instance_shape_scales_with_the_mesh(
    dp_shard: int, tp: int, ranks: int, comms: int
) -> None:
    instance = derive_instance(dp_shard, tp)
    assert len(instance.ranks) == ranks
    assert len(instance.communicators) == comms
    # Every communicator's members must be ranks of this mesh, or TypeOK fails
    # inside TLC rather than here.
    for communicator in instance.communicators:
        assert set(communicator.members) <= set(instance.ranks)


def test_degenerate_degrees_are_refused() -> None:
    for dp_shard, tp in ((0, 2), (2, 0), (-1, 2)):
        with pytest.raises(ValueError, match="must be positive"):
            derive_instance(dp_shard, tp)


def _run_tlc(tmp_path: Path, dp_shard: int, tp: int, max_issues: int) -> str:
    toolchain = ToolchainPaths.from_environment()
    missing = [str(p) for p in (toolchain.java, toolchain.tla2tools) if not p.exists()]
    if missing:
        pytest.skip(
            "pinned TLC toolchain is not materialized: "
            + ", ".join(missing)
            + "; run scripts/run_formal_checks.sh --networked once to vendor it"
        )

    work = tmp_path / f"d{dp_shard}t{tp}m{max_issues}"
    work.mkdir()
    shutil.copy(FORMAL_DIR / "DeviceMeshModel.tla", work / "DeviceMeshModel.tla")
    instance = derive_instance(dp_shard, tp)
    (work / "DeviceMeshMeshLadder.tla").write_text(render_module(instance))
    (work / "DeviceMeshMeshLadder.cfg").write_text(
        render_cfg(instance, max_issues=max_issues)
    )
    completed = subprocess.run(
        [
            str(toolchain.java),
            "-XX:+UseSerialGC",
            # tla2tools extracts the standard modules into java.io.tmpdir and
            # parses them back, so concurrent runs need private directories.
            f"-Djava.io.tmpdir={work}",
            "-cp",
            str(toolchain.tla2tools),
            "tlc2.TLC",
            "-config",
            "DeviceMeshMeshLadder.cfg",
            "-metadir",
            str(work / "states"),
            "-workers",
            "4",
            "DeviceMeshMeshLadder",
        ],
        cwd=work,
        text=True,
        capture_output=True,
        timeout=900,
    )
    return completed.stdout + completed.stderr


# The FINAL summary line, which is the only place the counts are comma-free and
# the queue depth is reported. TLC's periodic progress lines carry the same
# phrase with thousands separators -- "18,875 distinct states found" -- so a
# looser pattern matches a progress snapshot and a comma truncates it to 875.
# That is not a hypothetical: it is what this helper did first.
_TLC_SUMMARY_RE = re.compile(
    r"^([0-9]+) states generated, ([0-9]+) distinct states found, "
    r"([0-9]+) states left on queue",
    re.MULTILINE,
)


def _exhaustive_distinct_states(output: str) -> int:
    """Return the distinct state count, and require the search to have closed.

    A run with states left on queue explored a prefix of the state space, and a
    count taken from it would read like a complete result. Asserting the queue is
    empty is what makes the number mean "the whole graph".
    """
    match = _TLC_SUMMARY_RE.search(output)
    assert match, output[-2000:]
    generated, distinct, queued = (int(group) for group in match.groups())
    assert queued == 0, f"search did not close: {queued} states left on queue"
    assert generated >= distinct, (generated, distinct)
    return distinct


def test_the_derived_2x2_explores_the_shipped_state_graph(tmp_path: Path) -> None:
    """Equivalence, not just agreement on counts.

    If this number ever diverges from the shipped instance's, the generated
    instance means something different from the hand-written one and replacing
    it would silently change what every invariant is checked against.
    """
    output = _run_tlc(tmp_path, 2, 2, 2)
    assert "Model checking completed. No error has been found." in output, output[
        -2000:
    ]
    assert _exhaustive_distinct_states(output) == SHIPPED_2X2_DISTINCT_STATES


def test_the_invariants_hold_on_a_six_rank_mesh(tmp_path: Path) -> None:
    """The question the shipped 2x2 instance cannot answer.

    MaxIssues is 1 here for a measured reason rather than convenience: at 2 the
    six-rank state space did not close inside ten minutes, while at 1 it closes
    in seconds. A partial exploration proves nothing, so this asserts over a run
    that actually completed.
    """
    output = _run_tlc(tmp_path, 2, 3, 1)
    assert "Model checking completed. No error has been found." in output, output[
        -2000:
    ]
    assert "violated" not in output
    # Recorded so a change in the model that alters reachability is visible here
    # rather than only in a token nobody reads.
    assert _exhaustive_distinct_states(output) == 4625
