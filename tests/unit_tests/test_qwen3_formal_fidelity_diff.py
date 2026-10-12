# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for the TLA+/Lean fidelity differential harness (ticket 27).

The toolchain-backed tests here EXECUTE both checkers on generated modules.
They are the only evidence that the correspondence between DeviceMeshModel.tla and
DeviceMeshProtocol.lean is checked rather than asserted, so they are not allowed
to degrade into greps over the generated text. Where the vendored Lean and
tla2tools are not present the toolchain tests skip and the pure-Python
contracts still run; a skip is visible in the pytest summary and is not a pass.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from torchtitan.experiments.qwen3_formal_verifier.fidelity_diff import (
    _all_members_short_at_front,
    _exemption_for,
    compare_records,
    DEFAULT_INSTANCE_COUNT,
    DEFAULT_SEED,
    evaluate_with_lean,
    evaluate_with_tlc,
    EXEMPT,
    EXEMPTIONS,
    generate_instances,
    Instance,
    main,
    PERTURBATIONS,
    prepare_work_dir,
    probe_keys,
    render_lean_chunk,
    render_tla_probe,
    run_differential,
    SHARED_PREDICATES,
    ToolchainPaths,
)

# Small enough to keep every toolchain test a few seconds, large enough that
# the draw still contains all four generator families.
SMOKE_INSTANCES = 12
# The stuck-conjunct perturbation is only observable on an instance where some
# communicator could start while nothing else can move, which the draw first
# supplies around here. A smaller count would make that demonstration vacuous.
PERTURBATION_INSTANCES = 24


def _toolchain_or_skip() -> ToolchainPaths:
    toolchain = ToolchainPaths.from_environment()
    missing = toolchain.missing()
    if missing:
        pytest.skip("vendored formal toolchain not present: " + ", ".join(missing))
    return toolchain


@pytest.fixture(scope="module")
def staged_work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One staged work dir per module, so DeviceMeshProtocol is compiled once."""
    toolchain = _toolchain_or_skip()
    work_dir, _ = prepare_work_dir(
        tmp_path_factory.mktemp("fidelity-diff-shared"), toolchain
    )
    return work_dir


# ---------------------------------------------------------------------------
# Generator contracts
# ---------------------------------------------------------------------------


def test_generate_instances_is_deterministic_and_prefix_stable() -> None:
    first = generate_instances(DEFAULT_SEED, 40)
    assert generate_instances(DEFAULT_SEED, 40) == first
    # A longer draw must extend the shorter one, so raising the instance count
    # never reshuffles what a smaller run already compared.
    assert generate_instances(DEFAULT_SEED, 80)[:40] == first
    assert generate_instances(DEFAULT_SEED + 1, 40) != first


def test_generated_instances_are_well_formed_and_varied() -> None:
    instances = generate_instances(DEFAULT_SEED, DEFAULT_INSTANCE_COUNT)
    assert len(instances) == DEFAULT_INSTANCE_COUNT
    for instance in instances:
        # Raises on anything TLA's TypeOK would reject.
        instance.check_well_formed()
    families = {instance.family for instance in instances}
    assert families == {
        "front_op_mismatch",
        "spmd_permutation",
        "spmd_terminal",
        "spmd_terminal_running",
        "uniform_random",
    }
    # Breadth of shapes is the point; depth of state space is not.
    assert {instance.num_ranks for instance in instances} >= {1, 2, 3, 4}
    assert {instance.num_comms for instance in instances} >= {1, 2, 3}
    assert {instance.max_issues for instance in instances} >= {1, 2, 3}
    flags = {
        (
            instance.require_matched_issue_order,
            instance.require_uniform_program_ops,
            instance.require_uniform_program_comms,
            instance.require_stream_order,
        )
        for instance in instances
    }
    assert len(flags) >= 8


def test_generate_instances_rejects_empty_draw() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        generate_instances(DEFAULT_SEED, 0)


def test_check_well_formed_rejects_non_member_issue() -> None:
    instance = generate_instances(DEFAULT_SEED, 1)[0]
    broken = dataclasses.replace(
        instance,
        members=tuple(tuple(False for _ in row) for row in instance.members),
        issued=(((0, 0),),) + instance.issued[1:],
    )
    with pytest.raises(ValueError, match="does not belong to"):
        broken.check_well_formed()


def test_check_well_formed_rejects_done_on_above_max_issues() -> None:
    instance = generate_instances(DEFAULT_SEED, 1)[0]
    broken = dataclasses.replace(
        instance,
        done_on=tuple(instance.max_issues + 1 for _ in instance.done_on),
    )
    with pytest.raises(ValueError, match="outside 0.."):
        broken.check_well_formed()


# ---------------------------------------------------------------------------
# Emission contracts
# ---------------------------------------------------------------------------


def test_probe_keys_cover_every_shared_predicate() -> None:
    instances = generate_instances(DEFAULT_SEED, 40)
    seen = {pred for instance in instances for pred, _ in probe_keys(instance)}
    assert set(SHARED_PREDICATES) <= seen
    # The Lean-only half of exemption E1 must be probed too, otherwise the
    # carve-out would carry no obligation anywhere.
    assert "completedOutOfDomain" in seen


def test_rendered_modules_are_ascii_and_self_consistent() -> None:
    instances = generate_instances(DEFAULT_SEED, 8)
    lean_text = render_lean_chunk(instances)
    lean_text.encode("ascii")
    for instance in instances:
        name, module_text, cfg_text = render_tla_probe(instance)
        module_text.encode("ascii")
        cfg_text.encode("ascii")
        assert f"MODULE {name}" in module_text
        # Both definition overrides are load-bearing: Ops and StreamOfIssue are
        # ordinary definitions in DeviceMeshModel, so without them the generator
        # could not vary the operation alphabet or the stream map.
        assert "CONSTANT Ops <- ProbeOps" in cfg_text
        assert "CONSTANT StreamOfIssue <- ProbeStreamOfIssue" in cfg_text
        # TLA's FULL TypeOK guards the generated state's well-formedness.
        assert "INVARIANT TypeOK" in cfg_text
        for pred, args in probe_keys(instance):
            assert f'Emit("{pred}", "{args}"' in module_text
            assert f'emitLean {instance.index} "{pred}" "{args}"' in lean_text


def test_unknown_perturbation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown perturbation"):
        render_lean_chunk(generate_instances(DEFAULT_SEED, 1), ["nope"])
    with pytest.raises(ValueError, match="unknown perturbation"):
        render_tla_probe(generate_instances(DEFAULT_SEED, 1)[0], ["nope"])


# ---------------------------------------------------------------------------
# Comparator contracts, on synthetic records
# ---------------------------------------------------------------------------


def _synthetic_records(
    instance: Instance,
) -> tuple[dict[tuple[int, str, str], bool], dict[tuple[int, str, str], object]]:
    """Records that agree everywhere, for a comparator-only test."""
    lean: dict[tuple[int, str, str], bool] = {}
    tla: dict[tuple[int, str, str], object] = {}
    for pred, args in probe_keys(instance):
        key = (instance.index, pred, args)
        if pred == "completedOutOfDomain":
            lean[key] = True
            tla[key] = EXEMPT
        else:
            lean[key] = True
            tla[key] = True
    return lean, tla


def test_comparator_names_instance_and_predicate_on_disagreement() -> None:
    instance = generate_instances(DEFAULT_SEED, 1)[0]
    lean, tla = _synthetic_records(instance)
    target = (instance.index, "stuck", "-")
    lean[target] = False
    result = compare_records([instance], lean, tla, require_both_values=False)
    assert len(result.disagreements) == 1
    disagreement = result.disagreements[0]
    assert disagreement.instance == instance.index
    assert disagreement.predicate == "stuck"
    assert disagreement.lean is False
    assert disagreement.tla is True
    assert "instance=" in disagreement.render()
    assert "predicate=stuck" in disagreement.render()


def test_comparator_rejects_a_missing_probe_record() -> None:
    instance = generate_instances(DEFAULT_SEED, 1)[0]
    lean, tla = _synthetic_records(instance)
    del tla[(instance.index, "stuck", "-")]
    with pytest.raises(ValueError, match="TLC output is missing"):
        compare_records([instance], lean, tla, require_both_values=False)
    lean, tla = _synthetic_records(instance)
    del lean[(instance.index, "allDone", "-")]
    with pytest.raises(ValueError, match="Lean output is missing"):
        compare_records([instance], lean, tla, require_both_values=False)


def test_comparator_rejects_an_undeclared_exemption() -> None:
    instance = generate_instances(DEFAULT_SEED, 1)[0]
    lean, tla = _synthetic_records(instance)
    # A predicate with no named carve-out must not be allowed to go EXEMPT.
    tla[(instance.index, "stuck", "-")] = EXEMPT
    with pytest.raises(ValueError, match="no named exemption covers it"):
        compare_records([instance], lean, tla, require_both_values=False)


def test_non_vacuity_guard_reports_a_single_valued_predicate() -> None:
    instance = generate_instances(DEFAULT_SEED, 1)[0]
    lean, tla = _synthetic_records(instance)
    result = compare_records([instance], lean, tla, require_both_values=True)
    assert not result.disagreements
    # Every value in the synthetic records is true, so every shared predicate
    # that was compared must be flagged as never observed false.
    flagged = {problem.split(":")[0] for problem in result.vacuity_problems}
    assert flagged == set(SHARED_PREDICATES)
    compared = {
        pred for pred in SHARED_PREDICATES if result.coverage[pred].compared > 0
    }
    for problem in result.vacuity_problems:
        predicate, detail = problem.split(": ", 1)
        if predicate in compared:
            assert detail == "never observed false"
        else:
            assert detail == "never compared"


def test_exemption_obligation_catches_a_false_out_of_domain_completed() -> None:
    instance = generate_instances(DEFAULT_SEED, 1)[0]
    lean, tla = _synthetic_records(instance)
    target = next(key for key in lean if key[1] == "completedOutOfDomain")
    lean[target] = False
    result = compare_records([instance], lean, tla, require_both_values=False)
    assert len(result.disagreements) == 1
    disagreement = result.disagreements[0]
    assert disagreement.predicate == "completedOutOfDomain"
    assert "completed_outside_issue_domain" in disagreement.reason
    assert "divergence 3" in disagreement.reason


def test_every_exemption_names_a_divergence_and_a_predicate() -> None:
    assert EXEMPTIONS, "the harness must declare its carve-outs explicitly"
    for exemption in EXEMPTIONS:
        assert exemption.divergence == 3, (
            "only divergence 3 can make a shared predicate differ; a new "
            "carve-out needs its own justification in the module docstring"
        )
        assert _exemption_for(exemption.predicate) is exemption
        assert exemption.rationale.strip()
    assert len({e.predicate for e in EXEMPTIONS}) == len(EXEMPTIONS)


def test_all_members_short_at_front_matches_the_issue_counts() -> None:
    instances = generate_instances(DEFAULT_SEED, 60)
    checked = 0
    for instance in instances:
        for comm in range(instance.num_comms):
            members = instance.members_of(comm)
            front = instance.done_on[comm] + 1
            expected = bool(members) and all(
                instance.comm_count(rank, comm) < front for rank in members
            )
            assert _all_members_short_at_front(instance, comm) is expected
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Toolchain-backed contracts: both checkers actually run
# ---------------------------------------------------------------------------


def test_both_models_agree_on_a_small_draw(tmp_path: Path) -> None:
    toolchain = _toolchain_or_skip()
    report = run_differential(
        seed=DEFAULT_SEED,
        count=SMOKE_INSTANCES,
        work_dir=tmp_path / "agree",
        toolchain=toolchain,
        require_both_values=False,
    )
    assert report.comparison.disagreements == [], [
        d.render() for d in report.comparison.disagreements
    ]
    assert report.comparison.compared > 0
    for predicate in SHARED_PREDICATES:
        assert report.comparison.coverage[predicate].compared > 0, predicate
    # Both carve-outs must have been reached, otherwise the run says nothing
    # about whether they hold.
    for exemption in EXEMPTIONS:
        assert report.comparison.exemption_counts[exemption.name] > 0
    assert report.comparison.exemption_obligations["completed_outside_issue_domain"] > 0
    token = report.tokens()[0]
    assert f"seed={DEFAULT_SEED}" in token
    assert f"instances={SMOKE_INSTANCES}" in token
    assert "exhaustive=no" in token


def test_full_default_draw_is_non_vacuous(staged_work_dir: Path) -> None:
    """The default draw exercises both values of all nine predicates.

    A differential suite in which some predicate is constant would pass just
    as happily with a broken comparator, so the default instance count has to
    be large enough to falsify that.
    """
    toolchain = _toolchain_or_skip()
    report = run_differential(
        seed=DEFAULT_SEED,
        count=DEFAULT_INSTANCE_COUNT,
        work_dir=staged_work_dir,
        toolchain=toolchain,
        require_both_values=True,
    )
    assert report.comparison.vacuity_problems == []
    assert report.comparison.disagreements == [], [
        d.render() for d in report.comparison.disagreements
    ]
    assert report.ok


@pytest.mark.parametrize("perturbation", sorted(PERTURBATIONS))
def test_each_perturbation_is_reported(perturbation: str, tmp_path: Path) -> None:
    """Prove the harness can fail, one injected fault at a time.

    Each perturbation changes exactly one predicate on exactly one side of the
    comparison, in the generated module that the checker actually consumes, so
    a detection exercises emission, execution, parsing and comparison.
    """
    toolchain = _toolchain_or_skip()
    report = run_differential(
        seed=DEFAULT_SEED,
        count=PERTURBATION_INSTANCES,
        work_dir=tmp_path / perturbation,
        toolchain=toolchain,
        perturbations=[perturbation],
        require_both_values=False,
    )
    assert not report.ok, f"{perturbation} was not detected"
    assert report.comparison.disagreements
    expected_predicate = {
        "lean_same_site_negate": "sameSite",
        "lean_stuck_drop_start_conjunct": "stuck",
        "lean_ops_agree_negate": "opsAgreeAtFront",
        "lean_completed_out_of_domain_negate": "completedOutOfDomain",
        "tla_all_done_drop_running": "allDone",
        "tla_completed_off_by_one": "completed",
    }[perturbation]
    predicates = {d.predicate for d in report.comparison.disagreements}
    assert predicates == {expected_predicate}, predicates
    for disagreement in report.comparison.disagreements:
        assert 0 <= disagreement.instance < PERTURBATION_INSTANCES
        assert disagreement.args


def test_ill_formed_instance_is_rejected_by_tla_typeok(tmp_path: Path) -> None:
    """A state TLA's TypeOK rejects must abort, not be quietly compared.

    `check_well_formed` is bypassed here on purpose: the point is that the
    generated cfg's `INVARIANT TypeOK` is a real second line of defence, so a
    future generator change that produces a nonsense state cannot yield a
    meaningless agreement.
    """
    toolchain = _toolchain_or_skip()
    work_dir, _ = prepare_work_dir(tmp_path / "typeok", toolchain)
    instance = Instance(
        index=0,
        family="deliberately_ill_formed",
        num_ranks=2,
        num_comms=1,
        num_ops=1,
        num_streams=1,
        members=((True, False),),
        admissible=((True,),),
        stream_of_op=(0,),
        max_issues=2,
        require_matched_issue_order=True,
        require_uniform_program_ops=False,
        require_uniform_program_comms=False,
        require_stream_order=True,
        # Rank 1 is not a member of communicator 0.
        issued=((), ((0, 0),)),
        running=(False,),
        done_on=(0,),
    )
    with pytest.raises(ValueError, match="does not belong to"):
        instance.check_well_formed()
    with pytest.raises(ValueError, match="not well formed under TLA's TypeOK"):
        evaluate_with_tlc([instance], work_dir=work_dir, toolchain=toolchain)


def test_lean_and_tlc_halves_produce_the_same_probe_keys(
    staged_work_dir: Path,
) -> None:
    """Each half is run alone and must cover exactly the agreed key set.

    Comparing whole reports would hide a case where both emitters dropped the
    same predicate, so the key set is checked against `probe_keys` directly.
    """
    toolchain = _toolchain_or_skip()
    instances = generate_instances(DEFAULT_SEED, 4)
    expected = {
        (instance.index, pred, args)
        for instance in instances
        for pred, args in probe_keys(instance)
    }
    lean = evaluate_with_lean(instances, work_dir=staged_work_dir, toolchain=toolchain)
    tla = evaluate_with_tlc(instances, work_dir=staged_work_dir, toolchain=toolchain)
    assert set(lean) == expected
    assert set(tla) == expected
    # Only the declared carve-out predicates may be EXEMPT on the TLA+ side.
    exempt_predicates = {key[1] for key, value in tla.items() if value == EXEMPT}
    assert exempt_predicates <= {e.predicate for e in EXEMPTIONS}


def test_cli_reports_failure_for_a_perturbed_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _toolchain_or_skip()
    status = main(
        [
            "--seed",
            str(DEFAULT_SEED),
            "--instances",
            str(PERTURBATION_INSTANCES),
            "--work-dir",
            str(tmp_path / "cli"),
            "--perturb",
            "lean_same_site_negate",
            "--allow-vacuous",
        ]
    )
    captured = capsys.readouterr().out
    assert status == 1
    assert "QFV_FIDELITY_DIFF result=failure" in captured
    assert "QFV_FIDELITY_DIFF_DISAGREEMENT" in captured
    assert "predicate=sameSite" in captured
