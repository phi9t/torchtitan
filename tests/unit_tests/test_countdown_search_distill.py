# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

from torchtitan.experiments.countdown_search_distill.countdown import (
    CountdownProblem,
    generate_problem_pool,
    generate_problem,
    reachable_targets_by_subset,
    verify_solution,
    ProblemFilters,
)
from torchtitan.experiments.countdown_search_distill.datasets import (
    build_training_examples,
)
from torchtitan.experiments.countdown_search_distill.evaluate import (
    bucket_counts,
    bootstrap_pass_at_k,
    compute_compression,
    evaluate_rollouts,
    pass_at_k,
    retained_search_lift,
)
from torchtitan.experiments.countdown_search_distill.lora_export import (
    Qwen3LoRAExportConfig,
    convert_torchtitan_lora_tensors,
    peft_adapter_config,
    split_qwen3_fused_qkv_lora_b,
)


def test_verify_solution_rejects_unavailable_intermediate():
    problem = CountdownProblem(numbers=(75, 50, 8, 7, 6, 3), target=952)
    solution = "\n".join(
        [
            "75 - 50 = 25",
            "25 * 8 = 200",
            "7 - 3 = 4",
            "200 + 6 = 206",
            "206 * 4 = 824",
            "824 + 128 = 952",
            "FINAL: 952",
        ]
    )

    result = verify_solution(problem, solution)

    assert not result.success
    assert "unavailable" in result.error


def test_verify_solution_rejects_non_exact_division():
    problem = CountdownProblem(numbers=(8, 3), target=2)
    result = verify_solution(problem, "8 / 3 = 2\nFINAL: 2")

    assert not result.success
    assert result.error == "division is not exact: 8 / 3 = 2"


def test_verify_solution_accepts_valid_integer_trace():
    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    solution = "2 + 3 = 5\n5 * 4 = 20\nFINAL: 20"

    result = verify_solution(problem, solution)

    assert result.success
    assert result.final_value == 20
    assert result.valid_prefix_length == 2
    assert result.solution_depth == 2
    assert result.distance_to_target == 0
    assert result.states == ((2, 3, 4), (4, 5), (20,))
    assert result.operations[0].to_json()["op"] == "+"


def test_verify_solution_extracts_arithmetic_from_prose():
    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    solution = "\n".join(
        [
            "Let me solve it.",
            "2 + 3 = 5",
            "Then multiply.",
            "5 × 4 = 20",
            "Final: 20",
            "This reaches the target.",
        ]
    )

    result = verify_solution(problem, solution)

    assert result.success
    assert result.final_value == 20
    assert [operation.op for operation in result.operations] == ["+", "*"]


def test_verify_solution_accepts_implicit_final_target_state():
    problem = CountdownProblem(
        numbers=(13, 49, 36),
        target=98,
        require_all_numbers=True,
    )
    solution = "\n".join(
        [
            "13 + 49 = 62",
            "62 + 36 = 98",
        ]
    )

    result = verify_solution(problem, solution)

    assert result.success
    assert result.final_value == 98
    assert result.available_numbers == (98,)


def test_verify_solution_accepts_generated_solution():
    import random

    problem = generate_problem(
        rng=random.Random(1),
        num_numbers=4,
        number_min=1,
        number_max=10,
        target_min=10,
        target_max=99,
        min_solution_depth=2,
    )

    result = verify_solution(problem, "\n".join(problem.solution))

    assert result.success
    assert result.final_value == problem.target


def test_verify_solution_marks_first_invalid_and_unreachable():
    problem = CountdownProblem(numbers=(2, 3, 4), target=20)

    invalid = verify_solution(problem, "2 + 3 = 6\nFINAL: 20")

    assert not invalid.success
    assert invalid.first_invalid["reason"] == "wrong_declared_result"
    assert invalid.valid_prefix_length == 0

    unreachable = verify_solution(problem, "3 - 2 = 1\nFINAL: 20")

    assert not unreachable.success
    assert unreachable.first_unreachable is not None
    assert unreachable.first_unreachable["state"] == (1, 4)


def test_reachable_targets_by_subset_finds_canonical_solution():
    reachable = reachable_targets_by_subset((2, 3, 4))

    assert reachable[(2, 3, 4)][20].steps == (
        "2 + 3 = 5",
        "4 * 5 = 20",
    )


def test_generate_problem_pool_supports_filters():
    import random

    problems = generate_problem_pool(
        rng=random.Random(4),
        num_problems=3,
        num_numbers=4,
        filters=ProblemFilters(
            require_all_numbers=True,
            target_min=10,
            target_max=99,
            number_min=1,
            number_max=10,
            min_solution_depth=2,
        ),
    )

    assert len(problems) == 3
    assert all(problem.require_all_numbers for problem in problems)


def test_evaluation_buckets_and_pass_at_k():
    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    success = "2 + 3 = 5\n5 * 4 = 20\nFINAL: 20"
    failure = "2 + 3 = 6\nFINAL: 20"

    easy = evaluate_rollouts(problem, [success, failure])
    elicitable = evaluate_rollouts(problem, [failure, success])
    unreached = evaluate_rollouts(problem, [failure, failure])

    evaluations = [easy, elicitable, unreached]

    assert bucket_counts(evaluations) == {
        "easy": 1,
        "elicitable": 1,
        "unreached": 1,
    }
    assert pass_at_k(evaluations, [1, 2]) == {
        1: pytest.approx(1 / 3),
        2: pytest.approx(2 / 3),
    }


def test_search_compression_metrics():
    assert retained_search_lift(
        base_pass_at_1=0.10,
        base_pass_at_32=0.60,
        distilled_pass_at_1=0.35,
    ) == pytest.approx(0.5)
    assert compute_compression(
        base_curve={1: 0.1, 4: 0.3, 16: 0.5},
        distilled_curve={1: 0.2, 4: 0.5, 16: 0.7},
        target_success=0.5,
    ) == pytest.approx(4.0)


def test_bootstrap_pass_at_k_is_deterministic_on_fixtures():
    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    success = "2 + 3 = 5\n5 * 4 = 20\nFINAL: 20"
    failure = "2 + 3 = 6\nFINAL: 20"
    evaluations = [
        evaluate_rollouts(problem, [success]),
        evaluate_rollouts(problem, [failure]),
    ]

    intervals = bootstrap_pass_at_k(evaluations, [1], num_resamples=20, seed=0)

    assert set(intervals[1]) == {"mean", "ci_low", "ci_high"}
    assert 0.0 <= intervals[1]["ci_low"] <= intervals[1]["ci_high"] <= 1.0


def test_dataset_builders_do_not_leak_hints_into_unhinted_arms():
    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    success = "2 + 3 = 5\n5 * 4 = 20\nFINAL: 20"
    evaluation = evaluate_rollouts(problem, [success], problem_id="p0")

    examples = build_training_examples([evaluation])

    assert examples["raw"][0].question == problem.prompt().strip()
    assert examples["clean"][0].question == problem.prompt().strip()
    assert "Hint:" not in examples["raw"][0].question
    assert "Hint:" not in examples["clean"][0].question
    assert "Hint:" in examples["hindsight"][0].question
    assert {row.curriculum_stage for row in examples["curriculum"]} == {
        "hint_present",
        "hint_dropout",
        "hint_absent",
    }
    assert examples["raw"][0].to_json()["hint"] is None
    assert examples["raw"][0].to_json()["curriculum_stage"] is None
    assert examples["curriculum"][2].to_json()["hint"] is None


def test_fixture_cli_scores_rollouts(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    problems_path = tmp_path / "problems.jsonl"
    rollouts_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "evaluations.jsonl"
    summary_path = tmp_path / "summary.json"
    problems_path.write_text(json.dumps(problem.to_json()) + "\n")
    rollouts_path.write_text(
        json.dumps(
            {
                "problem": problem.to_json(),
                "rollouts": [
                    "2 + 3 = 6\nFINAL: 20",
                    "2 + 3 = 5\n5 * 4 = 20\nFINAL: 20",
                ],
            }
        )
        + "\n"
    )

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "evaluate-fixture",
            "--problems",
            str(problems_path),
            "--rollouts",
            str(rollouts_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ]
        main()
    finally:
        sys.argv = old_argv

    summary = json.loads(summary_path.read_text())
    assert summary["bucket_counts"] == {"easy": 0, "elicitable": 1, "unreached": 0}
    assert summary["pass_at_k"]["1"] == 0.0
    assert summary["pass_at_k"]["2"] == 1.0
    assert summary_path.with_suffix(".csv").exists()
    assert summary_path.with_suffix(".md").exists()


def test_prompt_variants_are_distinct():
    from torchtitan.experiments.countdown_search_distill.cli import _prompt_for_problem

    problem = CountdownProblem(numbers=(2, 3, 4), target=20)

    default_prompt = _prompt_for_problem(problem, "default")
    terse_prompt = _prompt_for_problem(problem, "terse")

    assert "Return a sequence of operations" in default_prompt
    assert "Return only equations" in terse_prompt
    assert default_prompt != terse_prompt


def test_chat_prompt_variant_uses_template(monkeypatch):
    from torchtitan.experiments.countdown_search_distill import cli

    class StubTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            assert kwargs["add_generation_prompt"]
            assert not kwargs["enable_thinking"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            return "CHAT:" + messages[1]["content"]

    class StubAutoTokenizer:
        @staticmethod
        def from_pretrained(model):
            assert model == "stub-model"
            return StubTokenizer()

    monkeypatch.setattr(cli, "AutoTokenizer", StubAutoTokenizer, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type(
            "TransformersStub",
            (),
            {"AutoTokenizer": StubAutoTokenizer},
        ),
    )
    args = type("Args", (), {"prompt_variant": "chat", "model": "stub-model"})()

    prompts = cli._build_vllm_prompts(
        [CountdownProblem(numbers=(2, 3, 4), target=20)],
        args,
    )

    assert prompts[0].startswith("CHAT:")
    assert "Return only equations" in prompts[0]


def test_build_datasets_cli_writes_all_arms(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    evaluation = evaluate_rollouts(
        problem,
        ["2 + 3 = 5\n5 * 4 = 20\nFINAL: 20"],
        problem_id="p0",
    )
    evaluations_path = tmp_path / "evaluations.jsonl"
    output_dir = tmp_path / "train"
    evaluations_path.write_text(json.dumps(evaluation.to_json()) + "\n")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "build-datasets",
            "--evaluations",
            str(evaluations_path),
            "--output-dir",
            str(output_dir),
        ]
        main()
    finally:
        sys.argv = old_argv

    for arm in ("raw", "clean", "hindsight", "curriculum"):
        assert (output_dir / f"{arm}.jsonl").exists()


def test_build_datasets_cli_can_fallback_to_canonical_raw(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    problem = CountdownProblem(
        numbers=(2, 3, 4),
        target=20,
        solution=("2 + 3 = 5", "5 * 4 = 20", "FINAL: 20"),
    )
    evaluation = evaluate_rollouts(
        problem,
        ["2 + 3 = 6\nFINAL: 20"],
        problem_id="p0",
    )
    evaluations_path = tmp_path / "evaluations.jsonl"
    output_dir = tmp_path / "train"
    evaluations_path.write_text(json.dumps(evaluation.to_json()) + "\n")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "build-datasets",
            "--evaluations",
            str(evaluations_path),
            "--output-dir",
            str(output_dir),
            "--matched-only",
            "--canonical-raw-fallback",
        ]
        main()
    finally:
        sys.argv = old_argv

    raw_rows = [
        json.loads(line)
        for line in (output_dir / "raw.jsonl").read_text().splitlines()
    ]
    assert raw_rows[0]["source_rollout_ids"] == ["p0:canonical"]
    assert raw_rows[0]["answer"] == "2 + 3 = 5\n5 * 4 = 20\nFINAL: 20"


def test_preflight_reduced_cli_checks_calibration_and_coverage(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    success = evaluate_rollouts(
        problem,
        ["2 + 3 = 5\n5 * 4 = 20\nFINAL: 20"],
        problem_id="p0",
    )
    calibration_path = tmp_path / "calibration.json"
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    decision_path = tmp_path / "decision.json"
    calibration_path.write_text(json.dumps({"selected": True}) + "\n")
    train_path.write_text(json.dumps(success.to_json()) + "\n")
    dev_path.write_text(json.dumps(success.to_json()) + "\n")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "preflight-reduced",
            "--calibration-decision",
            str(calibration_path),
            "--train-evaluations",
            str(train_path),
            "--dev-evaluations",
            str(dev_path),
            "--decision",
            str(decision_path),
            "--min-train-matched",
            "1",
            "--min-dev-matched",
            "1",
        ]
        main()
    finally:
        sys.argv = old_argv

    decision = json.loads(decision_path.read_text())
    assert decision["selected"]
    assert decision["observed"]["train_matched"] == 1
    assert decision["observed"]["dev_matched"] == 1


def test_preflight_reduced_cli_fails_low_coverage(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    problem = CountdownProblem(numbers=(2, 3, 4), target=20)
    failure = evaluate_rollouts(
        problem,
        ["2 + 3 = 6\nFINAL: 20"],
        problem_id="p0",
    )
    calibration_path = tmp_path / "calibration.json"
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    decision_path = tmp_path / "decision.json"
    calibration_path.write_text(json.dumps({"selected": True}) + "\n")
    train_path.write_text(json.dumps(failure.to_json()) + "\n")
    dev_path.write_text(json.dumps(failure.to_json()) + "\n")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "preflight-reduced",
            "--calibration-decision",
            str(calibration_path),
            "--train-evaluations",
            str(train_path),
            "--dev-evaluations",
            str(dev_path),
            "--decision",
            str(decision_path),
            "--min-train-matched",
            "1",
            "--min-dev-matched",
            "1",
        ]
        with pytest.raises(SystemExit, match="reduced preflight failed"):
            main()
    finally:
        sys.argv = old_argv

    decision = json.loads(decision_path.read_text())
    assert not decision["selected"]
    assert not decision["checks"]["train_matched_coverage"]
    assert not decision["checks"]["dev_matched_coverage"]


def test_select_sweep_cli_writes_selected_regime(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    weak_summary = tmp_path / "weak_summary.json"
    selected_summary = tmp_path / "selected_summary.json"
    candidates_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "selected.json"
    decision_path = tmp_path / "decision.json"
    weak_summary.write_text(
        json.dumps(
            {
                "num_problems": 10,
                "pass_at_k": {"1": 0.0, "32": 0.1},
                "bucket_counts": {"easy": 0, "elicitable": 1, "unreached": 9},
            }
        )
        + "\n"
    )
    selected_summary.write_text(
        json.dumps(
            {
                "num_problems": 10,
                "pass_at_k": {"1": 0.1, "32": 0.4},
                "bucket_counts": {"easy": 1, "elicitable": 3, "unreached": 6},
            }
        )
        + "\n"
    )
    candidates_path.write_text(
        "\n".join(
            [
                json.dumps({"name": "weak", "summary": str(weak_summary)}),
                json.dumps(
                    {
                        "name": "selected",
                        "summary": str(selected_summary),
                        "num_numbers": 4,
                        "target_min": 10,
                        "target_max": 99,
                    }
                ),
            ]
        )
        + "\n"
    )

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "select-sweep",
            "--candidates",
            str(candidates_path),
            "--output",
            str(output_path),
            "--decision",
            str(decision_path),
        ]
        main()
    finally:
        sys.argv = old_argv

    selected = json.loads(output_path.read_text())
    decision = json.loads(decision_path.read_text())
    assert selected["name"] == "selected"
    assert selected["num_numbers"] == 4
    assert decision["selected_name"] == "selected"


def test_select_sweep_cli_fails_without_selected_regime(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    summary_path = tmp_path / "summary.json"
    candidates_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "selected.json"
    decision_path = tmp_path / "decision.json"
    summary_path.write_text(
        json.dumps(
            {
                "num_problems": 10,
                "pass_at_k": {"1": 0.0, "32": 0.1},
                "bucket_counts": {"easy": 0, "elicitable": 1, "unreached": 9},
            }
        )
        + "\n"
    )
    candidates_path.write_text(
        json.dumps({"name": "weak", "summary": str(summary_path)}) + "\n"
    )

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "select-sweep",
            "--candidates",
            str(candidates_path),
            "--output",
            str(output_path),
            "--decision",
            str(decision_path),
        ]
        with pytest.raises(SystemExit, match="calibration sweep failed"):
            main()
    finally:
        sys.argv = old_argv

    decision = json.loads(decision_path.read_text())
    assert not decision["selected"]
    assert not output_path.exists()


def test_runtime_preflight_cli_checks_modules_and_assets(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n")
    (model_dir / "tokenizer_config.json").write_text("{}\n")
    (model_dir / "tokenizer.json").write_text("{}\n")
    (model_dir / "model.safetensors").write_text("")
    decision_path = tmp_path / "runtime_preflight.json"

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "preflight-runtime",
            "--model",
            str(model_dir),
            "--decision",
            str(decision_path),
            "--required-modules",
            "json",
            "--no-require-rootfs",
            "--no-require-gpu",
        ]
        main()
    finally:
        sys.argv = old_argv

    decision = json.loads(decision_path.read_text())
    assert decision["selected"]
    assert decision["checks"]["modules"]["json"]
    assert decision["checks"]["model_assets"]["safetensors"]


def test_runtime_preflight_cli_fails_missing_asset(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    decision_path = tmp_path / "runtime_preflight.json"

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "preflight-runtime",
            "--model",
            str(model_dir),
            "--decision",
            str(decision_path),
            "--required-modules",
            "json",
            "--no-require-rootfs",
            "--no-require-gpu",
        ]
        with pytest.raises(SystemExit, match="runtime preflight failed"):
            main()
    finally:
        sys.argv = old_argv

    decision = json.loads(decision_path.read_text())
    assert not decision["selected"]
    assert not decision["checks"]["model_assets"]["config_json"]


def test_validate_eval_matrix_cli_checks_split_sizes_and_rollouts(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    summary_dir = tmp_path / "eval" / "dev" / "raw"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_problems": 2,
                "pass_at_k": {
                    "1": 0.5,
                    "2": 0.5,
                    "4": 0.5,
                    "8": 0.5,
                    "16": 0.5,
                    "32": 0.5,
                },
                "bucket_counts": {"easy": 1, "elicitable": 1, "unreached": 0},
                "validity_breakdown": {"success": 16, "missing_final": 48},
            }
        )
        + "\n"
    )
    decision_path = tmp_path / "matrix.json"

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "validate-eval-matrix",
            "--eval-root",
            str(tmp_path / "eval"),
            "--decision",
            str(decision_path),
            "--splits",
            "dev",
            "--arms",
            "raw",
            "--expected-problems",
            "dev=2",
            "--num-rollouts",
            "32",
        ]
        main()
    finally:
        sys.argv = old_argv

    decision = json.loads(decision_path.read_text())
    assert decision["selected"]
    assert decision["checks"]["dev/raw"]["rollout_count"]
    assert decision["rows"][0]["pass_at_32"] == 0.5


def test_validate_eval_matrix_cli_fails_bad_rollout_count(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    summary_dir = tmp_path / "eval" / "dev" / "raw"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_problems": 2,
                "pass_at_k": {
                    "1": 0.5,
                    "2": 0.5,
                    "4": 0.5,
                    "8": 0.5,
                    "16": 0.5,
                    "32": 0.5,
                },
                "bucket_counts": {"easy": 1, "elicitable": 1, "unreached": 0},
                "validity_breakdown": {"success": 1},
            }
        )
        + "\n"
    )
    decision_path = tmp_path / "matrix.json"

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "validate-eval-matrix",
            "--eval-root",
            str(tmp_path / "eval"),
            "--decision",
            str(decision_path),
            "--splits",
            "dev",
            "--arms",
            "raw",
            "--expected-problems",
            "dev=2",
            "--num-rollouts",
            "32",
        ]
        with pytest.raises(SystemExit, match="eval matrix validation failed"):
            main()
    finally:
        sys.argv = old_argv

    decision = json.loads(decision_path.read_text())
    assert not decision["selected"]
    assert not decision["checks"]["dev/raw"]["rollout_count"]


def test_validate_splits_cli_writes_registry_for_disjoint_splits(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    registry_path = tmp_path / "split_registry.json"
    train_path.write_text(
        json.dumps(CountdownProblem(numbers=(2, 3, 4), target=20).to_json()) + "\n"
    )
    dev_path.write_text(
        json.dumps(CountdownProblem(numbers=(2, 3, 5), target=25).to_json()) + "\n"
    )

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "validate-splits",
            "--split",
            f"train={train_path}",
            f"dev={dev_path}",
            "--output",
            str(registry_path),
        ]
        main()
    finally:
        sys.argv = old_argv

    registry = json.loads(registry_path.read_text())
    assert registry["selected"]
    assert registry["checks"]["no_problem_key_overlap"]
    assert [entry["split"] for entry in registry["entries"]] == ["train", "dev"]
    assert registry["entries"][0]["problem_keys"] == ["2,3,4|20"]
    assert registry["overlaps"] == []


def test_validate_splits_cli_fails_on_problem_key_overlap(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    registry_path = tmp_path / "split_registry.json"
    train_path.write_text(
        json.dumps(CountdownProblem(numbers=(2, 3, 4), target=20).to_json()) + "\n"
    )
    dev_path.write_text(
        json.dumps(CountdownProblem(numbers=(4, 3, 2), target=20).to_json()) + "\n"
    )

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "validate-splits",
            "--split",
            f"train={train_path}",
            f"dev={dev_path}",
            "--output",
            str(registry_path),
        ]
        with pytest.raises(SystemExit, match="split validation failed"):
            main()
    finally:
        sys.argv = old_argv

    registry = json.loads(registry_path.read_text())
    assert not registry["selected"]
    assert not registry["checks"]["no_problem_key_overlap"]
    assert registry["overlaps"][0]["problem_key"] == "2,3,4|20"
    assert {item["split"] for item in registry["overlaps"][0]["occurrences"]} == {
        "train",
        "dev",
    }


def test_generate_pool_cli_excludes_existing_problem_keys(tmp_path):
    from torchtitan.experiments.countdown_search_distill.cli import main

    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "countdown",
            "generate-pool",
            "--output",
            str(train_path),
            "--num-problems",
            "1",
            "--seed",
            "7",
            "--num-numbers",
            "3",
            "--target-min",
            "10",
            "--target-max",
            "50",
            "--min-solution-depth",
            "2",
            "--require-all-numbers",
        ]
        main()
        sys.argv = [
            "countdown",
            "generate-pool",
            "--output",
            str(dev_path),
            "--num-problems",
            "1",
            "--seed",
            "7",
            "--num-numbers",
            "3",
            "--target-min",
            "10",
            "--target-max",
            "50",
            "--min-solution-depth",
            "2",
            "--require-all-numbers",
            "--exclude-problems",
            str(train_path),
        ]
        main()
    finally:
        sys.argv = old_argv

    train_row = json.loads(train_path.read_text().splitlines()[0])
    dev_row = json.loads(dev_path.read_text().splitlines()[0])
    assert train_row["problem_id"] != dev_row["problem_id"]


def test_qwen3_countdown_smoke_config_uses_chat_lora():
    pytest.importorskip("spmd_types")

    from torchtitan.hf_datasets.text_datasets import ChatDataLoader
    from torchtitan.models.qwen3.config_registry import (
        qwen3_debugmodel_countdown_lora_smoke,
    )

    config = qwen3_debugmodel_countdown_lora_smoke()

    assert isinstance(config.dataloader, ChatDataLoader.Config)
    assert config.training.seq_len == 512
    assert config.training.global_batch_size == 64
    assert not config.checkpoint.initial_load_in_hf
    assert config.model_spec is not None


def test_qwen3_fused_qkv_lora_b_split_preserves_grouped_order():
    import torch

    fused = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(24, 2)

    q, k, v = split_qwen3_fused_qkv_lora_b(
        fused,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
    )

    grouped = fused.reshape(2, 4, 3, 2)
    assert torch.equal(q, grouped[:, :2].reshape(12, 2))
    assert torch.equal(k, grouped[:, 2:3].reshape(6, 2))
    assert torch.equal(v, grouped[:, 3:].reshape(6, 2))


def test_convert_torchtitan_lora_tensors_emits_peft_qwen3_keys():
    import torch

    config = Qwen3LoRAExportConfig(
        rank=2,
        alpha=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
    )
    tensors = {
        "layers.0.attention.qkv_linear.wqkv.lora_a.weight": torch.ones(2, 5),
        "layers.0.attention.qkv_linear.wqkv.lora_b.weight": torch.arange(
            48, dtype=torch.float32
        ).reshape(24, 2),
        "layers.0.attention.wo.lora_a.weight": torch.ones(2, 12),
        "layers.0.attention.wo.lora_b.weight": torch.ones(5, 2),
        "layers.0.feed_forward.w1.lora_a.weight": torch.ones(2, 5),
        "layers.0.feed_forward.w1.lora_b.weight": torch.ones(7, 2),
        "layers.0.feed_forward.w2.lora_a.weight": torch.ones(2, 7),
        "layers.0.feed_forward.w2.lora_b.weight": torch.ones(5, 2),
        "layers.0.feed_forward.w3.lora_a.weight": torch.ones(2, 5),
        "layers.0.feed_forward.w3.lora_b.weight": torch.ones(7, 2),
        "lm_head.lora_a.weight": torch.ones(2, 9),
        "lm_head.lora_b.weight": torch.ones(9, 2),
    }

    exported = convert_torchtitan_lora_tensors(tensors, config)

    expected_keys = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight",
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight",
        "base_model.model.lm_head.lora_A.weight",
        "base_model.model.lm_head.lora_B.weight",
    }
    assert set(exported) == expected_keys
    assert torch.equal(
        exported["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        exported["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"],
    )
    assert exported[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    ].shape == (12, 2)
    assert exported[
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"
    ].shape == (6, 2)
    assert exported[
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"
    ].shape == (6, 2)


def test_peft_adapter_config_has_vllm_required_fields():
    config = peft_adapter_config(Qwen3LoRAExportConfig(rank=32, alpha=64))

    assert config["r"] == 32
    assert config["lora_alpha"] == 64
    assert config["bias"] == "none"
    assert config["peft_type"] == "LORA"
    assert set(config["target_modules"]) >= {"q_proj", "k_proj", "v_proj"}

