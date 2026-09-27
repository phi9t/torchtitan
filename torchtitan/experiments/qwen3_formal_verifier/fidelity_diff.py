# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Differential fidelity harness for the two ScoutB protocol models.

WHAT THIS IS FOR.

`experiments/qwen3_formal_verifier/formal/ScoutBModel.tla` and
`experiments/qwen3_formal_verifier/formal/ScoutBProtocol.lean` are two
hand-written models of one protocol. They are joined only by the prose
correspondence table in the Lean file's header, checked once by a reviewer
reading both. If they drift, the unbounded Lean proof describes a different
system from the one TLC checks, and nothing says so.

This module generates pseudo-random instances -- a topology and a state --
from a fixed seed, emits each instance twice (as a Lean module that evaluates
the shared predicates and as a TLA+ probe module plus cfg that evaluates the
same predicates at the same pinned state), runs `lean` and `tlc2.TLC`, parses
both outputs, and asserts the two records agree predicate by predicate.

THE NINE SHARED PREDICATES (ticket 27's table):

    Lean                            TLA+
    completedB s r k                Completed(r, k)
    fullyPendingB T s c             FullyPending(c)
    opsAgreeAtFrontB T s c          OpsAgreeAtFront(c)
    sameSiteB T c1 c2               SameSite(c1, c2)
    uniformOpsOKB T f               UniformProgramOpsOK(candidate)
    uniformCommsOKB T f             UniformProgramCommsOK(candidate)
    startAllowedB T s c             StartAllowed(c)
    allDoneB T s                    AllDone
    stuckB T s                      Stuck

EXEMPTIONS, AND WHY THERE ARE EXACTLY TWO.

The Lean header names ten divergences. Only divergence 3 -- partial functions
made total and Option-valued -- can make the two sides legitimately differ on
a shared predicate, and it does so in exactly two places. Both are carved out
by name below, each with a one-sided assertion that keeps the carve-out from
degenerating into a silent skip. See `Exemption` and `EXEMPTIONS`.

The other divergences do not need a carve-out here, and the harness actively
tests several of them:

  - divergences 1 and 2 (sets as characteristic functions, finite sets as
    lists) are exactly what makes Lean's `T.ranks.all (fun r => !member c r
    || ...)` claim to equal TLA's `\\A r \\in CommMembers[c] : ...`. Every
    generated Lean module discharges `membersInRanks`, `admissibleInOps` and
    `commsComplete` by `decide`, and every comparison of `fullyPending`,
    `sameSite`, `startAllowed`, `allDone` and `stuck` then checks the
    equality of the two quantifications on real data.
  - divergence 5 (`allLt k` ranging over `0..k-1` rather than `1..k-1`) is
    checked by every `uniformOps` and `uniformComms` comparison.
  - divergence 7 (TypeOK minus two conjuncts) is not a probed predicate, but
    every generated TLA+ cfg lists TLA's FULL `TypeOK` as an invariant, so a
    generated instance that is not well formed under the stronger TLA reading
    fails the run loudly instead of producing a meaningless comparison.
  - divergences 8, 9 and 10 concern `BlockedBy`, liveness and stuttering.
    None is a shared boolean predicate, so none is in scope.

WHAT A PASSING RUN DOES AND DOES NOT SAY. It says the two models agreed on
every shared predicate at every probed argument over N instances drawn from
one seed. It is not exhaustive and does not become so by raising N. The
result token therefore carries the seed and the instance count.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Comparison",
    "DEFAULT_INSTANCE_COUNT",
    "DEFAULT_SEED",
    "Disagreement",
    "EXEMPTIONS",
    "Exemption",
    "Instance",
    "PERTURBATIONS",
    "PredicateCoverage",
    "Report",
    "SHARED_PREDICATES",
    "ToolchainPaths",
    "compare_records",
    "evaluate_with_lean",
    "evaluate_with_tlc",
    "generate_instances",
    "main",
    "prepare_work_dir",
    "probe_keys",
    "render_lean_chunk",
    "render_tla_probe",
    "run_differential",
]


DEFAULT_SEED = 42
DEFAULT_INSTANCE_COUNT = 192

# Vendored toolchain locations inside the bwrap rootfs. `java` is not on PATH
# there, so the JDK is addressed explicitly.
DEFAULT_LEAN_BIN = (
    "/project/formal-cache/bazel/vendor/+http_archive+lean_4_34_0/bin/lean"
)
DEFAULT_JAVA_BIN = (
    "/project/formal-cache/bazel/vendor/"
    "rules_java++toolchains+remotejdk17_linux/bin/java"
)
DEFAULT_TLA2TOOLS_JAR = (
    "/project/formal-cache/bazel/vendor/+http_file+tla2tools_1_7_4/file/"
    "tla2tools.jar"
)

LEAN_MODEL_SOURCE = "ScoutBProtocol.lean"
TLA_MODEL_SOURCE = "ScoutBModel.tla"

# Names as they appear in the emitted records. `completedOutOfDomain` is not a
# tenth shared predicate: it is the Lean-only half of exemption E1.
SHARED_PREDICATES = (
    "allDone",
    "completed",
    "fullyPending",
    "opsAgreeAtFront",
    "sameSite",
    "startAllowed",
    "stuck",
    "uniformComms",
    "uniformOps",
)

LEAN_ONLY_PREDICATES = ("completedOutOfDomain",)

EXEMPT = "EXEMPT"

LEAN_RECORD_RE = re.compile(
    r"^QFVDIFF_LEAN inst=(\d+) pred=([A-Za-z]+) args=(\S+) value=(true|false)$"
)
TLA_RECORD_RE = re.compile(
    r'^<<"QFVDIFF_TLA", "([A-Za-z]+)", "(\S+)", (TRUE|FALSE|"EXEMPT")>>$'
)


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One topology plus one state, in a form both emitters can render.

    Ranks, communicators, operations and streams are all small integer
    indices. The Lean module renders them as `Fin n` inhabitants -- which is
    what makes `Topology`'s three proof obligations fall to `decide` -- and
    the TLA+ probe renders ranks as naturals and the rest as strings, matching
    `ScoutBModel.tla`'s own instance conventions.
    """

    index: int
    family: str
    num_ranks: int
    num_comms: int
    num_ops: int
    num_streams: int
    # members[c][r] and admissible[c][op] are the characteristic functions of
    # TLA's CommMembers[c] and CommOps[c]; see divergence 1.
    members: tuple[tuple[bool, ...], ...]
    admissible: tuple[tuple[bool, ...], ...]
    # TLA's StreamOfIssue is keyed on the operation only, so this is too.
    stream_of_op: tuple[int, ...]
    max_issues: int
    require_matched_issue_order: bool
    require_uniform_program_ops: bool
    require_uniform_program_comms: bool
    require_stream_order: bool
    # issued[r] is a sequence of (comm, op) pairs, in issue order.
    issued: tuple[tuple[tuple[int, int], ...], ...]
    running: tuple[bool, ...]
    done_on: tuple[int, ...]

    def comm_count(self, rank: int, comm: int) -> int:
        """TLA's `CommCount(r, c)`, computed directly from the instance data.

        Used only by the harness's one-sided exemption checks. It counts
        occurrences of a literal in a literal list and re-implements no guard.
        """
        return sum(1 for entry in self.issued[rank] if entry[0] == comm)

    def members_of(self, comm: int) -> tuple[int, ...]:
        return tuple(r for r in range(self.num_ranks) if self.members[comm][r])

    def check_well_formed(self) -> None:
        """Reject an instance TLA's `TypeOK` would reject.

        This duplicates the TLC-side `INVARIANT TypeOK` on purpose: catching a
        generator fault here names the instance, whereas catching it in TLC
        only aborts one probe run.
        """
        if self.num_ranks < 1 or self.num_comms < 1 or self.num_ops < 1:
            raise ValueError(f"instance {self.index}: empty carrier set")
        if self.num_streams < 1:
            raise ValueError(f"instance {self.index}: no streams")
        if len(self.issued) != self.num_ranks:
            raise ValueError(f"instance {self.index}: issued arity")
        for comm in range(self.num_comms):
            if not 0 <= self.done_on[comm] <= self.max_issues:
                # TypeOK's `doneOn \in [CommIds -> 0..MaxIssues]` conjunct.
                raise ValueError(
                    f"instance {self.index}: doneOn[{comm}]="
                    f"{self.done_on[comm]} outside 0..{self.max_issues}"
                )
        for rank in range(self.num_ranks):
            for comm, op in self.issued[rank]:
                if not self.members[comm][rank]:
                    raise ValueError(
                        f"instance {self.index}: rank {rank} issued on comm "
                        f"{comm} it does not belong to"
                    )
                if not self.admissible[comm][op]:
                    raise ValueError(
                        f"instance {self.index}: comm {comm} does not admit " f"op {op}"
                    )
        for op in range(self.num_ops):
            if not 0 <= self.stream_of_op[op] < self.num_streams:
                raise ValueError(f"instance {self.index}: stream of op {op}")


def _draw_membership(
    rng: random.Random, num_comms: int, num_ranks: int
) -> tuple[tuple[bool, ...], ...]:
    return tuple(
        tuple(rng.random() < 0.6 for _ in range(num_ranks)) for _ in range(num_comms)
    )


def _draw_admissible(
    rng: random.Random, num_comms: int, num_ops: int
) -> tuple[tuple[bool, ...], ...]:
    rows = []
    for _ in range(num_comms):
        row = [rng.random() < 0.6 for _ in range(num_ops)]
        if not any(row):
            # A communicator admitting nothing is legal but carries no issues,
            # so force one admissible operation on most of them to keep the
            # issue sequences from collapsing to empty.
            row[rng.randrange(num_ops)] = True
        rows.append(tuple(row))
    return tuple(rows)


def _issuable_comms(
    members: Sequence[Sequence[bool]],
    admissible: Sequence[Sequence[bool]],
    rank: int,
) -> list[int]:
    return [c for c in range(len(members)) if members[c][rank] and any(admissible[c])]


def _draw_uniform_random(rng: random.Random, index: int) -> Instance:
    num_ranks = rng.randint(1, 4)
    num_comms = rng.randint(1, 3)
    num_ops = rng.randint(1, 3)
    num_streams = rng.randint(1, min(3, num_ops))
    max_issues = rng.randint(1, 3)
    members = _draw_membership(rng, num_comms, num_ranks)
    admissible = _draw_admissible(rng, num_comms, num_ops)
    stream_of_op = tuple(rng.randrange(num_streams) for _ in range(num_ops))

    issued: list[tuple[tuple[int, int], ...]] = []
    for rank in range(num_ranks):
        choices = _issuable_comms(members, admissible, rank)
        if not choices:
            issued.append(())
            continue
        length = rng.randint(0, max_issues)
        entries = []
        for _ in range(length):
            comm = rng.choice(choices)
            ops = [op for op in range(num_ops) if admissible[comm][op]]
            entries.append((comm, rng.choice(ops)))
        issued.append(tuple(entries))

    running = tuple(rng.random() < 0.3 for _ in range(num_comms))
    done_on = []
    for comm in range(num_comms):
        member_ranks = [r for r in range(num_ranks) if members[comm][r]]
        if member_ranks and rng.random() < 0.5:
            # A CommFifo-respecting draw: the communicator has not run ahead
            # of any member. These are the states the protocol actually
            # reaches, so they must be well represented.
            counts = [sum(1 for e in issued[r] if e[0] == comm) for r in member_ranks]
            done_on.append(rng.randint(0, min(counts)))
        else:
            done_on.append(rng.randint(0, max_issues))
    return Instance(
        index=index,
        family="uniform_random",
        num_ranks=num_ranks,
        num_comms=num_comms,
        num_ops=num_ops,
        num_streams=num_streams,
        members=members,
        admissible=admissible,
        stream_of_op=stream_of_op,
        max_issues=max_issues,
        require_matched_issue_order=rng.random() < 0.75,
        require_uniform_program_ops=rng.random() < 0.6,
        require_uniform_program_comms=rng.random() < 0.4,
        require_stream_order=rng.random() < 0.75,
        issued=tuple(issued),
        running=running,
        done_on=tuple(done_on),
    )


def _spmd_family_name(terminal: bool, running_blocker: bool) -> str:
    if not terminal:
        return "spmd_permutation"
    return "spmd_terminal_running" if running_blocker else "spmd_terminal"


def _draw_spmd_permutation(
    rng: random.Random,
    index: int,
    *,
    terminal: bool,
    running_blocker: bool = False,
) -> Instance:
    """One SPMD program, permuted per rank.

    Every rank belongs to every communicator and every communicator admits
    every operation, so each rank's sequence is a permutation of one base
    program. This is the shape that makes `fullyPending` true, lets
    `opsAgreeAtFront` be false, and produces the stream-ordered circular waits
    that `stuck` is about. `terminal` additionally sets `doneOn` to the
    per-communicator issue count so that `AllDone` holds -- without it a random
    draw almost never reaches a terminal state.

    `running_blocker` keeps that same fully-issued, fully-counted shape but
    leaves one communicator running, so `AllDone` is false for exactly one
    reason. Without such an instance, dropping `running = {}` from `AllDone`
    would be an undetectable change: every other terminal state in the draw
    has an empty running set, and every non-terminal one already fails a later
    conjunct.
    """
    num_ranks = rng.randint(2, 4)
    num_comms = rng.randint(2, 3)
    num_ops = rng.randint(1, 3)
    num_streams = rng.randint(1, min(3, num_ops))
    max_issues = rng.randint(2, 3)
    members = tuple(tuple(True for _ in range(num_ranks)) for _ in range(num_comms))
    admissible = tuple(tuple(True for _ in range(num_ops)) for _ in range(num_comms))
    stream_of_op = tuple(rng.randrange(num_streams) for _ in range(num_ops))

    single_op = rng.random() < 0.5
    fixed_op = rng.randrange(num_ops)
    program = [
        (
            rng.randrange(num_comms),
            fixed_op if single_op else rng.randrange(num_ops),
        )
        for _ in range(max_issues)
    ]
    issued = []
    for _ in range(num_ranks):
        permuted = list(program)
        rng.shuffle(permuted)
        issued.append(tuple(permuted))

    if terminal:
        blocked = rng.randrange(num_comms) if running_blocker else None
        running = tuple(comm == blocked for comm in range(num_comms))
        done_on = tuple(
            sum(1 for entry in program if entry[0] == comm) for comm in range(num_comms)
        )
    else:
        running = tuple(rng.random() < 0.2 for _ in range(num_comms))
        done_on = tuple(
            rng.randint(0, sum(1 for e in program if e[0] == comm))
            for comm in range(num_comms)
        )
    return Instance(
        index=index,
        family=_spmd_family_name(terminal, running_blocker),
        num_ranks=num_ranks,
        num_comms=num_comms,
        num_ops=num_ops,
        num_streams=num_streams,
        members=members,
        admissible=admissible,
        stream_of_op=stream_of_op,
        max_issues=max_issues,
        require_matched_issue_order=rng.random() < 0.75,
        require_uniform_program_ops=rng.random() < 0.5,
        require_uniform_program_comms=rng.random() < 0.5,
        require_stream_order=rng.random() < 0.75,
        issued=tuple(issued),
        running=running,
        done_on=done_on,
    )


def _draw_front_op_mismatch(rng: random.Random, index: int) -> Instance:
    """A fully pending communicator whose members disagree at its front.

    This is the NCCL hazard the protocol exists to rule out, and it is the one
    shape a uniform draw almost never produces: `OpsAgreeAtFront(c)` is only
    comparable where `FullyPending(c)` holds (exemption E2), and reaching that
    region with a mismatch needs one communicator issued at least twice with
    two different operations, permuted differently per rank. Without this
    family the draw compared `opsAgreeAtFront = false` four times in 192
    instances, which is not enough to call the predicate exercised.
    """
    num_ranks = rng.randint(2, 3)
    num_comms = rng.randint(1, 2)
    num_ops = rng.randint(2, 3)
    num_streams = rng.randint(1, min(3, num_ops))
    max_issues = rng.randint(2, 3)
    members = tuple(tuple(True for _ in range(num_ranks)) for _ in range(num_comms))
    admissible = tuple(tuple(True for _ in range(num_ops)) for _ in range(num_comms))
    stream_of_op = tuple(rng.randrange(num_streams) for _ in range(num_ops))

    hot = rng.randrange(num_comms)
    program: list[tuple[int, int]] = []
    for slot in range(max_issues):
        if num_comms > 1 and slot == max_issues - 1 and rng.random() < 0.4:
            # One slot elsewhere, so the mismatch is not always the whole
            # program and the cross-communicator guards still see traffic.
            other = (hot + 1) % num_comms
            program.append((other, rng.randrange(num_ops)))
        else:
            program.append((hot, slot % num_ops))
    issued = []
    for _ in range(num_ranks):
        permuted = list(program)
        rng.shuffle(permuted)
        issued.append(tuple(permuted))
    return Instance(
        index=index,
        family="front_op_mismatch",
        num_ranks=num_ranks,
        num_comms=num_comms,
        num_ops=num_ops,
        num_streams=num_streams,
        members=members,
        admissible=admissible,
        stream_of_op=stream_of_op,
        max_issues=max_issues,
        require_matched_issue_order=rng.random() < 0.75,
        require_uniform_program_ops=rng.random() < 0.5,
        require_uniform_program_comms=rng.random() < 0.5,
        require_stream_order=rng.random() < 0.75,
        issued=tuple(issued),
        running=tuple(False for _ in range(num_comms)),
        done_on=tuple(0 for _ in range(num_comms)),
    )


def generate_instances(seed: int, count: int) -> list[Instance]:
    """Draw `count` well-formed instances from `seed`.

    The draw is prefix-stable: instances 0..k-1 are the same whether `count`
    is k or a larger number, so raising the instance count never reshuffles
    what a smaller run compared.
    """
    if count < 1:
        raise ValueError(f"instance count must be positive, got {count}")
    rng = random.Random(seed)
    instances: list[Instance] = []
    for index in range(count):
        roll = rng.random()
        if roll < 0.18:
            instance = _draw_spmd_permutation(rng, index, terminal=False)
        elif roll < 0.28:
            instance = _draw_spmd_permutation(rng, index, terminal=True)
        elif roll < 0.38:
            instance = _draw_spmd_permutation(
                rng, index, terminal=True, running_blocker=True
            )
        elif roll < 0.52:
            instance = _draw_front_op_mismatch(rng, index)
        else:
            instance = _draw_uniform_random(rng, index)
        instance.check_well_formed()
        instances.append(instance)
    return instances


# ---------------------------------------------------------------------------
# Probe keys
# ---------------------------------------------------------------------------

# Lean is probed two positions past the end of each rank's sequence. Both are
# outside TLA's domain, so both are exemption E1 rather than comparisons.
OUT_OF_DOMAIN_OFFSETS = (1, 3)


def probe_keys(instance: Instance) -> list[tuple[str, str]]:
    """The (predicate, args) keys both emitters must produce, in order.

    Both sides are generated from this one list, and the comparator requires
    the two parsed key sets to equal it exactly. A predicate quietly dropped
    from one emitter is therefore a failure, not a smaller comparison.
    """
    keys: list[tuple[str, str]] = [("stuck", "-"), ("allDone", "-")]
    keys.append(("uniformOps", "-"))
    keys.append(("uniformComms", "-"))
    for comm in range(instance.num_comms):
        keys.append(("fullyPending", f"c{comm}"))
    for comm in range(instance.num_comms):
        keys.append(("startAllowed", f"c{comm}"))
    for comm in range(instance.num_comms):
        keys.append(("opsAgreeAtFront", f"c{comm}"))
    for left in range(instance.num_comms):
        for right in range(instance.num_comms):
            keys.append(("sameSite", f"c{left}-c{right}"))
    for rank in range(instance.num_ranks):
        length = len(instance.issued[rank])
        for position in range(1, length + 1):
            keys.append(("completed", f"r{rank}-k{position}"))
        for offset in OUT_OF_DOMAIN_OFFSETS:
            keys.append(("completedOutOfDomain", f"r{rank}-k{length + offset}"))
    return keys


# ---------------------------------------------------------------------------
# Perturbations: the mechanism that proves the harness can fail
# ---------------------------------------------------------------------------

PERTURBATIONS: Mapping[str, str] = {
    "lean_same_site_negate": (
        "Emit Lean's sameSite as its negation. Must be caught on every "
        "instance by the sameSite comparison."
    ),
    "lean_stuck_drop_start_conjunct": (
        "Emit Lean's stuck without the `!startAllowedB` conjunct. Caught only "
        "on instances where some communicator could start, so it also shows "
        "the generated instances are not all trivially blocked."
    ),
    "lean_ops_agree_negate": (
        "Emit Lean's opsAgreeAtFront as its negation. Caught by the in-domain "
        "comparison and, separately, by exemption E2's one-sided check."
    ),
    "lean_completed_out_of_domain_negate": (
        "Emit Lean's out-of-domain completed as its negation. TLA compares "
        "nothing there, so this is caught only by exemption E1's one-sided "
        "check -- which is what shows E1 is a carve-out and not a skip."
    ),
    "tla_all_done_drop_running": (
        "Emit TLA's AllDone without its `running = {}` conjunct."
    ),
    "tla_completed_off_by_one": (
        "Emit TLA's Completed with the done counter shifted by one."
    ),
}


def _validate_perturbations(perturbations: Iterable[str]) -> frozenset[str]:
    chosen = frozenset(perturbations)
    unknown = sorted(chosen - set(PERTURBATIONS))
    if unknown:
        raise ValueError(
            "unknown perturbation(s): "
            + ", ".join(unknown)
            + "; known: "
            + ", ".join(sorted(PERTURBATIONS))
        )
    return chosen


# ---------------------------------------------------------------------------
# Lean emission
# ---------------------------------------------------------------------------

LEAN_PREAMBLE = """\
-- Generated by torchtitan.experiments.qwen3_formal_verifier.fidelity_diff.
-- Do not edit; do not commit. One section per differential instance.
import ScoutBProtocol

namespace Qwen3Formal.ScoutBProtocol

/-- Print one predicate value in the harness's parse format. -/
def emitLean (inst : Nat) (pred : String) (args : String) (v : Bool) :
    IO Unit :=
  IO.println s!"QFVDIFF_LEAN inst={inst} pred={pred} args={args} value={v}"
"""


def _lean_bool(value: bool) -> str:
    return "true" if value else "false"


def _lean_bool_list(values: Sequence[bool]) -> str:
    return "[" + ", ".join(_lean_bool(v) for v in values) + "]"


def _lean_nat_list(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def _render_lean_instance(instance: Instance, perturbations: frozenset[str]) -> str:
    idx = instance.index
    lines: list[str] = []
    add = lines.append
    add(f"section Inst{idx}")
    add(f"abbrev Rk{idx} := Fin {instance.num_ranks}")
    add(f"abbrev Cm{idx} := Fin {instance.num_comms}")
    add(f"abbrev Op{idx} := Fin {instance.num_ops}")
    add(f"abbrev St{idx} := Fin {instance.num_streams}")
    add("")
    member_rows = ", ".join(
        _lean_bool_list(instance.members[c]) for c in range(instance.num_comms)
    )
    add(f"def memberTable{idx} : List (List Bool) := [{member_rows}]")
    add(f"def member{idx} (c : Cm{idx}) (r : Rk{idx}) : Bool :=")
    add(f"  (memberTable{idx}.getD c.val []).getD r.val false")
    add("")
    adm_rows = ", ".join(
        _lean_bool_list(instance.admissible[c]) for c in range(instance.num_comms)
    )
    add(f"def admTable{idx} : List (List Bool) := [{adm_rows}]")
    add(f"def admissibleOp{idx} (c : Cm{idx}) (op : Op{idx}) : Bool :=")
    add(f"  (admTable{idx}.getD c.val []).getD op.val false")
    add("")
    add(
        f"def streamTable{idx} : List St{idx} := "
        + _lean_nat_list(instance.stream_of_op)
    )
    add(f"def streamOf{idx} (e : Issue Cm{idx} Op{idx}) : St{idx} :=")
    add(f"  streamTable{idx}.getD e.op.val 0")
    add("")
    add(f"def topo{idx} : Topology Rk{idx} Cm{idx} Op{idx} St{idx} where")
    add(f"  ranks := {_lean_nat_list(range(instance.num_ranks))}")
    add(f"  comms := {_lean_nat_list(range(instance.num_comms))}")
    add(f"  member := member{idx}")
    add(f"  opsAll := {_lean_nat_list(range(instance.num_ops))}")
    add(f"  admissibleOp := admissibleOp{idx}")
    add(f"  streamOf := streamOf{idx}")
    add(f"  maxIssues := {instance.max_issues}")
    add(
        "  requireMatchedIssueOrder := "
        + _lean_bool(instance.require_matched_issue_order)
    )
    add(
        "  requireUniformProgramOps := "
        + _lean_bool(instance.require_uniform_program_ops)
    )
    add(
        "  requireUniformProgramComms := "
        + _lean_bool(instance.require_uniform_program_comms)
    )
    add("  requireStreamOrder := " + _lean_bool(instance.require_stream_order))
    # Divergence 2: instantiating a Topology discharges the three fields that
    # tie the Lean types back to TLA's finite sets. `Fin n` carriers make all
    # three decidable, which is why the generator uses them.
    add("  membersInRanks := by decide")
    add("  admissibleInOps := by decide")
    add("  commsComplete := by decide")
    add("")
    issued_rows = []
    for rank in range(instance.num_ranks):
        entries = ", ".join(
            f"Issue.mk {comm} {op}" for comm, op in instance.issued[rank]
        )
        issued_rows.append(f"[{entries}]")
    add(
        f"def issuedTable{idx} : List (List (Issue Cm{idx} Op{idx})) := ["
        + ", ".join(issued_rows)
        + "]"
    )
    add(f"def runningTable{idx} : List Bool := " + _lean_bool_list(instance.running))
    add(f"def doneTable{idx} : List Nat := " + _lean_nat_list(instance.done_on))
    add(f"def st{idx} : State Rk{idx} Cm{idx} Op{idx} where")
    add(f"  issued := fun r => issuedTable{idx}.getD r.val []")
    add(f"  running := fun c => runningTable{idx}.getD c.val false")
    add(f"  doneOn := fun c => doneTable{idx}.getD c.val 0")
    add("")
    add("#eval do")
    for pred, args in probe_keys(instance):
        expr = _lean_expression(instance, pred, args, perturbations)
        add(f'  emitLean {idx} "{pred}" "{args}" ({expr})')
    add(f"end Inst{idx}")
    return "\n".join(lines)


def _lean_expression(
    instance: Instance, pred: str, args: str, perturbations: frozenset[str]
) -> str:
    idx = instance.index
    topo = f"topo{idx}"
    state = f"st{idx}"
    if pred == "stuck":
        if "lean_stuck_drop_start_conjunct" in perturbations:
            # PERTURBATION, never on the default path: stuckB's body with the
            # third conjunct removed.
            return (
                f"({topo}.comms.all fun c => !{state}.running c) && "
                f"({topo}.ranks.all fun r => ({topo}.comms.all fun c => "
                f"({topo}.opsAll.all fun op => "
                f"!issueAllowedB {topo} {state} r c op)))"
            )
        return f"stuckB {topo} {state}"
    if pred == "allDone":
        return f"allDoneB {topo} {state}"
    if pred == "uniformOps":
        return f"uniformOpsOKB {topo} {state}.issued"
    if pred == "uniformComms":
        return f"uniformCommsOKB {topo} {state}.issued"
    if pred == "fullyPending":
        comm = _parse_comm(args)
        return f"fullyPendingB {topo} {state} {comm}"
    if pred == "startAllowed":
        comm = _parse_comm(args)
        return f"startAllowedB {topo} {state} {comm}"
    if pred == "opsAgreeAtFront":
        comm = _parse_comm(args)
        body = f"opsAgreeAtFrontB {topo} {state} {comm}"
        if "lean_ops_agree_negate" in perturbations:
            return f"!({body})"
        return body
    if pred == "sameSite":
        left, right = _parse_comm_pair(args)
        body = f"sameSiteB {topo} ({left} : Cm{idx}) {right}"
        if "lean_same_site_negate" in perturbations:
            return f"!({body})"
        return body
    if pred in ("completed", "completedOutOfDomain"):
        rank, position = _parse_rank_position(args)
        body = f"completedB {state} ({rank} : Rk{idx}) {position}"
        if (
            pred == "completedOutOfDomain"
            and "lean_completed_out_of_domain_negate" in perturbations
        ):
            return f"!({body})"
        return body
    raise AssertionError(f"unhandled predicate {pred!r}")


def _parse_comm(args: str) -> int:
    match = re.fullmatch(r"c(\d+)", args)
    assert match is not None, args
    return int(match.group(1))


def _parse_comm_pair(args: str) -> tuple[int, int]:
    match = re.fullmatch(r"c(\d+)-c(\d+)", args)
    assert match is not None, args
    return int(match.group(1)), int(match.group(2))


def _parse_rank_position(args: str) -> tuple[int, int]:
    match = re.fullmatch(r"r(\d+)-k(\d+)", args)
    assert match is not None, args
    return int(match.group(1)), int(match.group(2))


def render_lean_chunk(
    instances: Sequence[Instance], perturbations: Iterable[str] = ()
) -> str:
    chosen = _validate_perturbations(perturbations)
    parts = [LEAN_PREAMBLE]
    for instance in instances:
        parts.append(_render_lean_instance(instance, chosen))
    parts.append("end Qwen3Formal.ScoutBProtocol")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# TLA+ emission
# ---------------------------------------------------------------------------


def _tla_comm(comm: int) -> str:
    return f'"c{comm}"'


def _tla_op(op: int) -> str:
    return f'"o{op}"'


def _tla_set(items: Iterable[str]) -> str:
    rendered = list(items)
    if not rendered:
        return "{}"
    return "{" + ", ".join(rendered) + "}"


def _tla_case(variable: str, arms: Sequence[tuple[str, str]], indent: str) -> str:
    """Render a CASE chain whose last arm is OTHER.

    A one-arm chain is emitted as the bare value, because `CASE ... [] OTHER`
    with a single arm would leave the guard unreachable and unread.
    """
    assert arms, "a CASE chain needs at least one arm"
    if len(arms) == 1:
        return arms[0][1]
    lines = [f"CASE {variable} = {arms[0][0]} -> {arms[0][1]}"]
    for guard, value in arms[1:-1]:
        lines.append(f"{indent}  [] {variable} = {guard} -> {value}")
    lines.append(f"{indent}  [] OTHER -> {arms[-1][1]}")
    return "\n".join(lines)


def _tla_probe_expression(
    instance: Instance, pred: str, args: str, perturbations: frozenset[str]
) -> str:
    if pred == "stuck":
        return "Stuck"
    if pred == "allDone":
        if "tla_all_done_drop_running" in perturbations:
            # PERTURBATION, never on the default path: AllDone without its
            # `running = {}` conjunct.
            return (
                "((\\A r \\in Ranks : Len(issued[r]) = MaxIssues)"
                " /\\ (\\A c \\in CommIds : \\A r \\in CommMembers[c] :"
                " CommCount(r, c) = doneOn[c]))"
            )
        return "AllDone"
    if pred == "uniformOps":
        return "UniformProgramOpsOK(issued)"
    if pred == "uniformComms":
        return "UniformProgramCommsOK(issued)"
    if pred == "fullyPending":
        return f"FullyPending({_tla_comm(_parse_comm(args))})"
    if pred == "startAllowed":
        return f"StartAllowed({_tla_comm(_parse_comm(args))})"
    if pred == "opsAgreeAtFront":
        comm = _tla_comm(_parse_comm(args))
        # EXEMPTION E2, divergence 3. `OpsAgreeAtFront(c)` reads
        # `OpAt(r, c, Front(c))`, whose `IdxOf` is a CHOOSE over
        # {k \in DOMAIN issued[r] : issued[r][k].comm = c /\ CommPos(r,k) =
        # Front(c)}. That set is empty exactly when CommCount(r, c) <
        # Front(c), and TLC raises an evaluation error on a CHOOSE with no
        # witness. `FullyPending(c)` is precisely the condition that no member
        # is short, so the predicate is compared there and declared exempt
        # otherwise. The guard is evaluated by TLC from the model's own
        # definition, not recomputed by the harness.
        return f'IF FullyPending({comm}) THEN OpsAgreeAtFront({comm}) ELSE "EXEMPT"'
    if pred == "sameSite":
        left, right = _parse_comm_pair(args)
        return f"SameSite({_tla_comm(left)}, {_tla_comm(right)})"
    if pred == "completed":
        rank, position = _parse_rank_position(args)
        if "tla_completed_off_by_one" in perturbations:
            # PERTURBATION, never on the default path.
            return (
                f"(CommPos({rank}, {position}) <="
                f" doneOn[issued[{rank}][{position}].comm] + 1)"
            )
        return f"Completed({rank}, {position})"
    if pred == "completedOutOfDomain":
        # EXEMPTION E1, divergence 3. `Completed(r, k)` reads `issued[r][k]`,
        # which is undefined outside `1..Len(issued[r])`; TLC raises an
        # evaluation error rather than returning a value. The TLA+ side
        # therefore emits EXEMPT and the comparator applies E1's one-sided
        # assertion to the Lean value instead of comparing.
        return '"EXEMPT"'
    raise AssertionError(f"unhandled predicate {pred!r}")


def render_tla_probe(
    instance: Instance, perturbations: Iterable[str] = ()
) -> tuple[str, str, str]:
    """Return (module_name, module_text, cfg_text) for one instance."""
    chosen = _validate_perturbations(perturbations)
    name = f"ScoutBDiffProbe{instance.index}"
    comms = [_tla_comm(c) for c in range(instance.num_comms)]
    ops = [_tla_op(o) for o in range(instance.num_ops)]

    member_arms = [
        (
            comms[c],
            _tla_set(str(r) for r in instance.members_of(c)),
        )
        for c in range(instance.num_comms)
    ]
    comm_ops_arms = [
        (
            comms[c],
            _tla_set(
                _tla_op(o) for o in range(instance.num_ops) if instance.admissible[c][o]
            ),
        )
        for c in range(instance.num_comms)
    ]
    stream_arms = [
        (_tla_op(o), f'"s{instance.stream_of_op[o]}"') for o in range(instance.num_ops)
    ]
    issued_arms = []
    for r in range(instance.num_ranks):
        if instance.issued[r]:
            rendered = ", ".join(
                f"[comm |-> {_tla_comm(c)}, op |-> {_tla_op(o)}]"
                for c, o in instance.issued[r]
            )
            issued_arms.append((str(r), f"<< {rendered} >>"))
        else:
            issued_arms.append((str(r), "<< >>"))
    done_arms = [
        (comms[c], str(instance.done_on[c])) for c in range(instance.num_comms)
    ]

    header = "-" * 28
    body: list[str] = []
    add = body.append
    add(f"{header} MODULE {name} {header}")
    add("\\* Generated by fidelity_diff.py. Do not edit; do not commit.")
    add("EXTENDS ScoutBModel, TLC")
    add("")
    add(f"ProbeRanks == {_tla_set(str(r) for r in range(instance.num_ranks))}")
    add(f"ProbeCommIds == {_tla_set(comms)}")
    add(f"ProbeOps == {_tla_set(ops)}")
    add(f"ProbeMaxIssues == {instance.max_issues}")
    add("ProbeCommMembers == [c \\in ProbeCommIds |->")
    add("  " + _tla_case("c", member_arms, "  ") + "]")
    add("ProbeCommOps == [c \\in ProbeCommIds |->")
    add("  " + _tla_case("c", comm_ops_arms, "  ") + "]")
    add("ProbeStreamOfIssue(e) ==")
    add("  " + _tla_case("e.op", stream_arms, "  "))
    add("")
    add("PinnedInit ==")
    add("  /\\ issued = [r \\in Ranks |->")
    add("       " + _tla_case("r", issued_arms, "       ") + "]")
    running = _tla_set(
        comms[c] for c in range(instance.num_comms) if instance.running[c]
    )
    add(f"  /\\ running = {running}")
    add("  /\\ doneOn = [c \\in CommIds |->")
    add("       " + _tla_case("c", done_arms, "       ") + "]")
    add("PinnedNext == UNCHANGED vars")
    add("PinnedSpec == PinnedInit /\\ [][PinnedNext]_vars")
    add("")
    add('Emit(p, a, v) == PrintT(<<"QFVDIFF_TLA", p, a, v>>)')
    add("")
    add("Probe ==")
    for pred, args in probe_keys(instance):
        expr = _tla_probe_expression(instance, pred, args, chosen)
        add(f'  /\\ Emit("{pred}", "{args}", {expr})')
    add("  /\\ TRUE")
    add("=" * 79)
    module_text = "\n".join(body) + "\n"

    cfg_lines = [
        "SPECIFICATION PinnedSpec",
        "INVARIANT Probe",
        # TLA's FULL TypeOK, including the two conjuncts divergence 7 drops
        # from the Lean encoding. A generated instance that violates it would
        # make the comparison meaningless, so TLC rejects it here.
        "INVARIANT TypeOK",
        "CONSTANT Ranks <- ProbeRanks",
        "CONSTANT CommIds <- ProbeCommIds",
        "CONSTANT CommMembers <- ProbeCommMembers",
        "CONSTANT CommOps <- ProbeCommOps",
        "CONSTANT MaxIssues <- ProbeMaxIssues",
        # `Ops` and `StreamOfIssue` are ordinary definitions in ScoutBModel,
        # not declared constants. TLC's definition override binds them the
        # same way, which is what lets the generator vary the operation
        # alphabet and the operation-to-stream map instead of inheriting the
        # module's single hard-wired 2x2 instance.
        "CONSTANT Ops <- ProbeOps",
        "CONSTANT StreamOfIssue <- ProbeStreamOfIssue",
        "CONSTANT RequireMatchedIssueOrder <- "
        + ("TRUE" if instance.require_matched_issue_order else "FALSE"),
        "CONSTANT RequireUniformProgramOps <- "
        + ("TRUE" if instance.require_uniform_program_ops else "FALSE"),
        "CONSTANT RequireUniformProgramComms <- "
        + ("TRUE" if instance.require_uniform_program_comms else "FALSE"),
        "CONSTANT RequireStreamOrder <- "
        + ("TRUE" if instance.require_stream_order else "FALSE"),
    ]
    return name, module_text, "\n".join(cfg_lines) + "\n"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_lean_output(text: str) -> dict[tuple[int, str, str], bool]:
    records: dict[tuple[int, str, str], bool] = {}
    for line in text.splitlines():
        match = LEAN_RECORD_RE.match(line.strip())
        if match is None:
            continue
        key = (int(match.group(1)), match.group(2), match.group(3))
        value = match.group(4) == "true"
        if key in records and records[key] != value:
            raise ValueError(f"Lean printed conflicting values for {key}")
        records[key] = value
    return records


def parse_tlc_output(text: str, index: int) -> dict[tuple[int, str, str], object]:
    """Parse one probe run's records.

    TLC evaluates the invariant once per new distinct state. `PinnedNext` is
    `UNCHANGED vars`, so there is exactly one distinct state and one copy of
    each record; a repeated key with a different value would mean that
    assumption broke, and is rejected.
    """
    records: dict[tuple[int, str, str], object] = {}
    for line in text.splitlines():
        match = TLA_RECORD_RE.match(line.strip())
        if match is None:
            continue
        key = (index, match.group(1), match.group(2))
        raw = match.group(3)
        value: object
        if raw == "TRUE":
            value = True
        elif raw == "FALSE":
            value = False
        else:
            value = EXEMPT
        if key in records and records[key] != value:
            raise ValueError(f"TLC printed conflicting values for {key}")
        records[key] = value
    return records


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    instance: int
    predicate: str
    args: str
    lean: object
    tla: object
    reason: str

    def render(self) -> str:
        return (
            f"instance={self.instance} predicate={self.predicate} "
            f"args={self.args} lean={self.lean} tla={self.tla} "
            f"reason={self.reason}"
        )


@dataclass(frozen=True)
class Exemption:
    """One named carve-out for a divergence in ScoutBProtocol.lean's header.

    `divergence` is the number from that header and `predicate` is the probe
    name whose TLA+ side reports EXEMPT. The one-sided obligation the Lean
    value must still satisfy inside the carve-out lives in
    `_exemption_obligation`; a carve-out with no obligation anywhere would be
    indistinguishable from a silent skip, which is what `rationale` has to
    justify.
    """

    name: str
    divergence: int
    predicate: str
    rationale: str


EXEMPTIONS: tuple[Exemption, ...] = (
    Exemption(
        name="completed_outside_issue_domain",
        divergence=3,
        predicate="completedOutOfDomain",
        rationale=(
            "TLA's Completed(r, k) reads issued[r][k], undefined for k outside "
            "1..Len(issued[r]). Measured, not assumed: TLC there reports "
            "'Attempted to apply tuple ... to integer 5 which is out of "
            "domain' and abandons the run, so there is no value to compare. "
            "Lean's completedB returns true. Obligation: Lean must be true, "
            "which is the direction divergence 3 claims -- guard easier to "
            "satisfy, reachable set larger, invariant claim stronger."
        ),
    ),
    Exemption(
        name="ops_agree_at_front_not_fully_pending",
        divergence=3,
        predicate="opsAgreeAtFront",
        rationale=(
            "TLA's OpsAgreeAtFront(c) reads OpAt(r, c, Front(c)), whose IdxOf "
            "is a CHOOSE with no witness when CommCount(r, c) < Front(c). "
            "Measured, not assumed: TLC there reports 'Attempted to compute "
            "the value of an expression of form CHOOSE x \\in S: P, but no "
            "element of S satisfied P' at ScoutBModel.tla's IdxOf. The "
            "predicate is compared wherever FullyPending(c) holds. Where no "
            "member has reached the front, both of Lean's opAtOn reads are "
            "none, so the obligation is that Lean is true. Where some but not "
            "all members are short, TLA has no value and nothing is asserted; "
            "startAllowed(c) and stuck remain compared on those instances and "
            "are where the guard is actually used."
        ),
    ),
)


@dataclass(frozen=True)
class PredicateCoverage:
    predicate: str
    compared: int
    lean_true: int
    lean_false: int
    exempt: int


@dataclass
class Comparison:
    compared: int = 0
    exempt: int = 0
    obligations_checked: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)
    vacuity_problems: list[str] = field(default_factory=list)
    coverage: dict[str, PredicateCoverage] = field(default_factory=dict)
    exemption_counts: dict[str, int] = field(default_factory=dict)
    exemption_obligations: dict[str, int] = field(default_factory=dict)


def _all_members_short_at_front(instance: Instance, comm: int) -> bool:
    """True when no member of `comm` has issued its Front(c)-th collective.

    This is the sub-case of exemption E2 in which divergence 3's direction
    claim is stated outright: both of Lean's `opAtOn` reads are `none`, so
    `opsAgreeAtFrontB` must be `true`. It counts occurrences in the generated
    issue lists and reproduces no guard.
    """
    front = instance.done_on[comm] + 1
    members = instance.members_of(comm)
    if not members:
        # Vacuous quantification on both sides; no obligation to state.
        return False
    return all(instance.comm_count(r, comm) < front for r in members)


def compare_records(
    instances: Sequence[Instance],
    lean_records: Mapping[tuple[int, str, str], bool],
    tla_records: Mapping[tuple[int, str, str], object],
    *,
    require_both_values: bool = True,
) -> Comparison:
    """Compare the two records key by key.

    Raises `ValueError` when either emitter did not produce exactly the keys
    `probe_keys` asked for, because a missing key is a silently smaller
    comparison rather than a passing one.

    `require_both_values` controls only whether the non-vacuity condition is
    evaluated; its findings land in `Comparison.vacuity_problems` and never
    suppress a disagreement.
    """
    result = Comparison()
    # Per predicate: [compared, lean_true, lean_false, exempt]. The true and
    # false counts deliberately exclude exempt records.
    tallies: dict[str, list[int]] = {
        pred: [0, 0, 0, 0] for pred in (*SHARED_PREDICATES, *LEAN_ONLY_PREDICATES)
    }
    for exemption in EXEMPTIONS:
        result.exemption_counts.setdefault(exemption.name, 0)
        result.exemption_obligations.setdefault(exemption.name, 0)

    expected: set[tuple[int, str, str]] = set()
    for instance in instances:
        for pred, args in probe_keys(instance):
            expected.add((instance.index, pred, args))

    missing_lean = sorted(expected - set(lean_records))
    if missing_lean:
        raise ValueError(
            f"Lean output is missing {len(missing_lean)} probe record(s), "
            f"first: {missing_lean[0]}"
        )
    extra_lean = sorted(set(lean_records) - expected)
    if extra_lean:
        raise ValueError(
            f"Lean output has {len(extra_lean)} unexpected record(s), "
            f"first: {extra_lean[0]}"
        )
    missing_tla = sorted(expected - set(tla_records))
    if missing_tla:
        raise ValueError(
            f"TLC output is missing {len(missing_tla)} probe record(s), "
            f"first: {missing_tla[0]}"
        )
    extra_tla = sorted(set(tla_records) - expected)
    if extra_tla:
        raise ValueError(
            f"TLC output has {len(extra_tla)} unexpected record(s), "
            f"first: {extra_tla[0]}"
        )

    by_index = {instance.index: instance for instance in instances}
    for key in sorted(expected):
        index, pred, args = key
        instance = by_index[index]
        lean_value = lean_records[key]
        tla_value = tla_records[key]
        tally = tallies[pred]
        if tla_value == EXEMPT:
            tally[3] += 1
            result.exempt += 1
            exemption = _exemption_for(pred)
            result.exemption_counts[exemption.name] += 1
            obligation = _exemption_obligation(instance, pred, args)
            if obligation is not None:
                result.obligations_checked += 1
                result.exemption_obligations[exemption.name] += 1
                if lean_value != obligation:
                    result.disagreements.append(
                        Disagreement(
                            instance=index,
                            predicate=pred,
                            args=args,
                            lean=lean_value,
                            tla=f"{EXEMPT}(obligation={obligation})",
                            reason=(
                                f"exemption {exemption.name} "
                                f"(divergence {exemption.divergence}) "
                                "one-sided obligation violated"
                            ),
                        )
                    )
            continue
        tally[0] += 1
        if lean_value:
            tally[1] += 1
        else:
            tally[2] += 1
        result.compared += 1
        if lean_value != tla_value:
            result.disagreements.append(
                Disagreement(
                    instance=index,
                    predicate=pred,
                    args=args,
                    lean=lean_value,
                    tla=tla_value,
                    reason="shared predicate disagreement",
                )
            )

    for pred, tally in tallies.items():
        result.coverage[pred] = PredicateCoverage(
            predicate=pred,
            compared=tally[0],
            lean_true=tally[1],
            lean_false=tally[2],
            exempt=tally[3],
        )

    if require_both_values:
        _check_non_vacuity(result)
    return result


def _exemption_for(predicate: str) -> Exemption:
    for exemption in EXEMPTIONS:
        if exemption.predicate == predicate:
            return exemption
    raise ValueError(
        f"predicate {predicate!r} was reported EXEMPT by the TLA+ probe but "
        "no named exemption covers it; a carve-out must be declared in "
        "EXEMPTIONS with its divergence number"
    )


def _exemption_obligation(instance: Instance, predicate: str, args: str) -> bool | None:
    """The one-sided obligation on Lean inside a carve-out, if any."""
    if predicate == "completedOutOfDomain":
        # E1, divergence 3: completedB is true outside the issue domain.
        return True
    if predicate == "opsAgreeAtFront":
        comm = _parse_comm(args)
        if _all_members_short_at_front(instance, comm):
            # E2, divergence 3, in the sub-case where both Options are none.
            return True
        return None
    raise AssertionError(f"no obligation rule for {predicate!r}")


def _check_non_vacuity(result: Comparison) -> None:
    """Record every shared predicate that never took both compared values.

    A differential harness whose predicates are constant over the whole draw
    proves nothing about the cases that matter, and would pass just as happily
    with a broken comparator. Five guards in this program were found to have
    exactly that shape, so the condition is checked rather than assumed.

    The problems are recorded rather than raised so that a run which both
    disagrees and is vacuous still reports the disagreements, which are the
    more informative finding.
    """
    for pred in SHARED_PREDICATES:
        coverage = result.coverage[pred]
        if coverage.compared == 0:
            result.vacuity_problems.append(f"{pred}: never compared")
        elif coverage.lean_true == 0:
            result.vacuity_problems.append(f"{pred}: never observed true")
        elif coverage.lean_false == 0:
            result.vacuity_problems.append(f"{pred}: never observed false")


# ---------------------------------------------------------------------------
# Running the checkers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolchainPaths:
    lean: Path
    java: Path
    tla2tools: Path

    @staticmethod
    def from_environment(env: Mapping[str, str] | None = None) -> "ToolchainPaths":
        source = os.environ if env is None else env
        return ToolchainPaths(
            lean=Path(source.get("QFV_LEAN_BIN", DEFAULT_LEAN_BIN)),
            java=Path(source.get("QFV_JAVA_BIN", DEFAULT_JAVA_BIN)),
            tla2tools=Path(source.get("QFV_TLA2TOOLS_JAR", DEFAULT_TLA2TOOLS_JAR)),
        )

    def missing(self) -> list[str]:
        return [
            str(path)
            for path in (self.lean, self.java, self.tla2tools)
            if not path.exists()
        ]


@dataclass
class Report:
    seed: int
    instance_count: int
    comparison: Comparison
    lean_seconds: float
    tlc_seconds: float
    wall_seconds: float
    perturbations: tuple[str, ...]
    work_dir: Path

    @property
    def ok(self) -> bool:
        return (
            not self.comparison.disagreements and not self.comparison.vacuity_problems
        )

    def tokens(self) -> list[str]:
        comparison = self.comparison
        lines = [
            "QFV_FIDELITY_DIFF "
            f"result={'success' if self.ok else 'failure'} "
            f"seed={self.seed} instances={self.instance_count} "
            f"predicates_compared={comparison.compared} "
            f"exempt_records={comparison.exempt} "
            f"exemption_obligations_checked={comparison.obligations_checked} "
            f"disagreements={len(comparison.disagreements)} "
            f"vacuity_problems={len(comparison.vacuity_problems)} "
            f"perturbations={'|'.join(self.perturbations) or 'none'} "
            f"lean_seconds={self.lean_seconds:.2f} "
            f"tlc_seconds={self.tlc_seconds:.2f} "
            f"wall_seconds={self.wall_seconds:.2f} "
            "scope=sampled-instances exhaustive=no"
        ]
        for exemption in EXEMPTIONS:
            lines.append(
                "QFV_FIDELITY_DIFF_EXEMPTION "
                f"name={exemption.name} divergence={exemption.divergence} "
                f"predicate={exemption.predicate} "
                f"records={comparison.exemption_counts.get(exemption.name, 0)} "
                "obligations_checked="
                f"{comparison.exemption_obligations.get(exemption.name, 0)}"
            )
        for predicate in (*SHARED_PREDICATES, *LEAN_ONLY_PREDICATES):
            coverage = comparison.coverage[predicate]
            lines.append(
                "QFV_FIDELITY_DIFF_COVERAGE "
                f"predicate={coverage.predicate} "
                f"compared={coverage.compared} "
                f"lean_true={coverage.lean_true} "
                f"lean_false={coverage.lean_false} "
                f"exempt={coverage.exempt}"
            )
        for problem in comparison.vacuity_problems:
            lines.append(f"QFV_FIDELITY_DIFF_VACUITY detail={problem}")
        for disagreement in comparison.disagreements:
            lines.append("QFV_FIDELITY_DIFF_DISAGREEMENT " + disagreement.render())
        return lines


def _repo_formal_dir() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    formal = repo_root / "experiments" / "qwen3_formal_verifier" / "formal"
    if not formal.is_dir():
        raise ValueError(f"formal model directory not found: {formal}")
    return formal


def _run_lean_chunk(toolchain: ToolchainPaths, work_dir: Path, path: Path) -> str:
    env = dict(os.environ)
    env["LEAN_PATH"] = str(work_dir)
    completed = subprocess.run(
        [str(toolchain.lean), path.name],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"lean failed on {path.name} (exit {completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout


def _run_tlc_probe(
    toolchain: ToolchainPaths, work_dir: Path, name: str, index: int
) -> tuple[int, str]:
    # tla2tools extracts the standard modules (Naturals, Sequences, TLC, ...)
    # into java.io.tmpdir and then parses them from there. Concurrent probes
    # sharing /tmp race on those files and SANY fails with an internal NPE on
    # a half-written module, so every probe gets a private tmpdir.
    tmpdir = work_dir / "tlc-tmp" / name
    tmpdir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(toolchain.java),
            "-XX:+UseSerialGC",
            "-Xmx512m",
            f"-Djava.io.tmpdir={tmpdir}",
            "-cp",
            str(toolchain.tla2tools),
            "tlc2.TLC",
            "-config",
            f"{name}.cfg",
            # TLC's default metadir is states/<timestamp> with one-second
            # granularity, so concurrent probes in one work dir would collide.
            "-metadir",
            f"tlc-meta/{name}",
            "-workers",
            "1",
            "-nowarning",
            name,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        if "Invariant TypeOK is violated" in output:
            raise ValueError(
                f"instance {index} is not well formed under TLA's TypeOK; the "
                "comparison would be meaningless. TLC said:\n" + output
            )
        raise ValueError(
            f"TLC failed on {name} (exit {completed.returncode}):\n" + output
        )
    return index, output


def prepare_work_dir(
    work_dir: Path, toolchain: ToolchainPaths | None = None
) -> tuple[Path, ToolchainPaths]:
    """Stage both model sources into `work_dir` and compile the Lean model.

    Returns the resolved directory and toolchain. Split out from
    `run_differential` so that focused tests can drive one checker at a time
    without re-staging.
    """
    resolved = ToolchainPaths.from_environment() if toolchain is None else toolchain
    missing = resolved.missing()
    if missing:
        raise ValueError("toolchain not found: " + ", ".join(missing))
    # Absolute: the checkers run with cwd=work_dir and LEAN_PATH must still
    # resolve, so a relative work dir would point at the wrong place.
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    formal = _repo_formal_dir()
    for source in (LEAN_MODEL_SOURCE, TLA_MODEL_SOURCE):
        shutil.copy2(formal / source, work_dir / source)

    olean = work_dir / "ScoutBProtocol.olean"
    lean_source = work_dir / LEAN_MODEL_SOURCE
    if not olean.exists() or olean.stat().st_mtime < lean_source.stat().st_mtime:
        completed = subprocess.run(
            [str(resolved.lean), "-o", olean.name, LEAN_MODEL_SOURCE],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                "compiling ScoutBProtocol.lean failed:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        if not olean.exists():
            raise ValueError(f"lean reported success but {olean} was not written")
    return work_dir, resolved


def evaluate_with_lean(
    instances: Sequence[Instance],
    *,
    work_dir: Path,
    toolchain: ToolchainPaths,
    perturbations: Iterable[str] = (),
    chunk_size: int = 24,
    max_workers: int | None = None,
) -> dict[tuple[int, str, str], bool]:
    """Emit and run the Lean half, returning the parsed records.

    Instances are batched into chunk files because `lean` costs a fixed
    startup per invocation; the chunks then run concurrently.
    """
    chosen = _validate_perturbations(perturbations)
    if chunk_size < 1:
        raise ValueError(f"lean chunk size must be positive, got {chunk_size}")
    workers = max_workers or min(16, (os.cpu_count() or 4))
    chunk_paths = []
    for position in range(0, len(instances), chunk_size):
        chunk = instances[position : position + chunk_size]
        path = work_dir / f"GeneratedChunk{position // chunk_size}.lean"
        path.write_text(render_lean_chunk(chunk, chosen), encoding="ascii")
        chunk_paths.append(path)
    records: dict[tuple[int, str, str], bool] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for output in pool.map(
            lambda path: _run_lean_chunk(toolchain, work_dir, path), chunk_paths
        ):
            records.update(parse_lean_output(output))
    return records


def evaluate_with_tlc(
    instances: Sequence[Instance],
    *,
    work_dir: Path,
    toolchain: ToolchainPaths,
    perturbations: Iterable[str] = (),
    max_workers: int | None = None,
) -> dict[tuple[int, str, str], object]:
    """Emit and run the TLA+ half, returning the parsed records.

    One probe module and cfg per instance, because the seven constants and the
    two overridden definitions differ per instance and TLC binds them once per
    run.
    """
    chosen = _validate_perturbations(perturbations)
    workers = max_workers or min(16, (os.cpu_count() or 4))
    probes = []
    for instance in instances:
        name, module_text, cfg_text = render_tla_probe(instance, chosen)
        (work_dir / f"{name}.tla").write_text(module_text, encoding="ascii")
        (work_dir / f"{name}.cfg").write_text(cfg_text, encoding="ascii")
        probes.append((instance.index, name))
    records: dict[tuple[int, str, str], object] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_tlc_probe, toolchain, work_dir, name, index)
            for index, name in probes
        ]
        for future in concurrent.futures.as_completed(futures):
            index, output = future.result()
            records.update(parse_tlc_output(output, index))
    return records


def run_differential(
    *,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_INSTANCE_COUNT,
    work_dir: Path,
    toolchain: ToolchainPaths | None = None,
    perturbations: Iterable[str] = (),
    lean_chunk_size: int = 24,
    max_workers: int | None = None,
    require_both_values: bool = True,
) -> Report:
    """Generate, run both checkers, and compare.

    `work_dir` is created if absent and is left populated, so a reported
    disagreement can be reproduced by hand from the exact generated modules.
    """
    chosen = _validate_perturbations(perturbations)
    started = time.monotonic()
    work_dir, resolved = prepare_work_dir(work_dir, toolchain)
    instances = generate_instances(seed, count)

    lean_started = time.monotonic()
    lean_records = evaluate_with_lean(
        instances,
        work_dir=work_dir,
        toolchain=resolved,
        perturbations=chosen,
        chunk_size=lean_chunk_size,
        max_workers=max_workers,
    )
    lean_seconds = time.monotonic() - lean_started

    tlc_started = time.monotonic()
    tla_records = evaluate_with_tlc(
        instances,
        work_dir=work_dir,
        toolchain=resolved,
        perturbations=chosen,
        max_workers=max_workers,
    )
    tlc_seconds = time.monotonic() - tlc_started

    comparison = compare_records(
        instances,
        lean_records,
        tla_records,
        require_both_values=require_both_values,
    )
    return Report(
        seed=seed,
        instance_count=len(instances),
        comparison=comparison,
        lean_seconds=lean_seconds,
        tlc_seconds=tlc_seconds,
        wall_seconds=time.monotonic() - started,
        perturbations=tuple(sorted(chosen)),
        work_dir=work_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fidelity_diff",
        description=(
            "Differential-test ScoutBModel.tla against ScoutBProtocol.lean on "
            "pseudo-random instances drawn from a fixed seed."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCE_COUNT)
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory for the generated modules and checker outputs.",
    )
    parser.add_argument("--lean-chunk-size", type=int, default=24)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--perturb",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Inject a named perturbation to demonstrate the comparator can "
            "fail. Repeatable. Known names: " + ", ".join(sorted(PERTURBATIONS))
        ),
    )
    parser.add_argument(
        "--allow-vacuous",
        action="store_true",
        help=(
            "Skip the guard requiring every shared predicate to take both "
            "values. Intended for small diagnostic runs only."
        ),
    )
    parser.add_argument(
        "--list-perturbations", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_perturbations:
        for name in sorted(PERTURBATIONS):
            print(f"{name}: {PERTURBATIONS[name]}")
        return 0
    try:
        report = run_differential(
            seed=args.seed,
            count=args.instances,
            work_dir=Path(args.work_dir),
            perturbations=args.perturb,
            lean_chunk_size=args.lean_chunk_size,
            max_workers=args.max_workers,
            require_both_values=not args.allow_vacuous,
        )
    except ValueError as error:
        print(f"QFV_FIDELITY_DIFF result=error detail={error}", file=sys.stderr)
        return 2
    for line in report.tokens():
        print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
