# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json

from torchtitan.experiments.scaffold_to_policy.arithmetic_words import (
    build_report_input,
    build_split_registry,
    evaluate_fixture_rollouts,
    generate_split,
    summarize_evaluations,
    verify_answer,
    write_json,
    write_jsonl,
)


def test_arithmetic_words_generation_is_deterministic():
    first = generate_split(seed=123, num_problems=5)
    second = generate_split(seed=123, num_problems=5)

    assert [problem.to_json() for problem in first] == [
        problem.to_json() for problem in second
    ]
    assert len({problem.problem_id for problem in first}) == 5


def test_arithmetic_words_verifier_requires_strict_final_line():
    problem = generate_split(seed=1, num_problems=1)[0]

    missing = verify_answer(problem, str(problem.answer))
    wrong = verify_answer(problem, f"FINAL: {problem.answer + 1}")
    correct = verify_answer(problem, f"work\nFINAL: {problem.answer}")

    assert not missing.success
    assert missing.error == "missing FINAL line"
    assert not wrong.success
    assert wrong.strict_final
    assert correct.success
    assert correct.strict_final


def test_arithmetic_words_summary_reports_pass_curves():
    problem = generate_split(seed=2, num_problems=1)[0]
    evaluation = evaluate_fixture_rollouts(
        problem,
        [
            "no final",
            f"FINAL: {problem.answer + 1}",
            f"FINAL: {problem.answer}",
        ],
    )

    summary = summarize_evaluations([evaluation], ks=(1, 2, 3))

    assert summary["pass_at_k"] == {"1": 0.0, "2": 0.0, "3": 1.0}
    assert summary["strict_format_pass_at_k"] == {"1": 0.0, "2": 0.0, "3": 1.0}
    assert summary["bucket_counts"] == {"easy": 0, "elicitable": 1, "unreached": 0}
    assert summary["failure_breakdown"]["missing FINAL line"] == 1
    assert summary["failure_breakdown"]["success"] == 1


def test_arithmetic_words_split_registry_rejects_overlap(tmp_path):
    problems = generate_split(seed=10, num_problems=2)
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    write_jsonl(train, [problem.to_json() for problem in problems])
    write_jsonl(dev, [problems[0].to_json()])

    registry = build_split_registry({"train": train, "dev": dev})

    assert not registry["selected"]
    assert registry["overlaps"] == [
        {
            "problem_id": problems[0].problem_id,
            "first_split": "train",
            "second_split": "dev",
        }
    ]


def test_arithmetic_words_report_input_validates_summary_counts(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    problems = generate_split(seed=20, num_problems=2)
    write_jsonl(dev, [problem.to_json() for problem in problems])
    split_registry = data_root / "split_registry.json"
    write_json(split_registry, build_split_registry({"dev": dev}))
    summary = results_root / "dev_summary.json"
    write_json(
        summary,
        summarize_evaluations(
            [
                evaluate_fixture_rollouts(problem, [f"FINAL: {problem.answer}"])
                for problem in problems
            ]
        ),
    )

    report_input = build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
    )

    assert all(report_input["checks"].values())
    assert report_input["run"]["lane"] == "reasoning"
    assert report_input["verifier"]["kind"] == "exact"
    assert json.loads(summary.read_text())["num_problems"] == 2
