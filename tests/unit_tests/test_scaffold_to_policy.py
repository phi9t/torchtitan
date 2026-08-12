# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json

import pytest

from torchtitan.experiments.scaffold_to_policy.arithmetic_words import (
    build_report_input,
    build_split_registry,
    evaluate_fixture_rollouts,
    generate_split,
    prompt_for_problem,
    summarize_evaluations,
    verify_answer,
    write_json,
    write_jsonl,
)
from torchtitan.experiments.scaffold_to_policy.cli import build_parser
from torchtitan.experiments.scaffold_to_policy import coding_style
from torchtitan.experiments.scaffold_to_policy import gsm_style
from torchtitan.experiments.scaffold_to_policy import math_style
from torchtitan.experiments.scaffold_to_policy import modular_sequences


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


def test_arithmetic_words_prompt_names_strict_output_contract():
    problem = generate_split(seed=1, num_problems=1)[0]

    prompt = prompt_for_problem(problem)

    assert problem.prompt in prompt
    assert "FINAL: <integer>" in prompt


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


def test_arithmetic_words_vllm_parser_defaults_to_chat_prompt():
    parser = build_parser()

    args = parser.parse_args(
        [
            "evaluate-arithmetic-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
        ]
    )

    assert args.prompt_variant == "chat"
    assert args.num_rollouts == 32
    assert args.max_model_len == 2048


def test_gsm_style_normalizes_common_final_answer_forms():
    assert gsm_style.normalize_answer("#### $1,250") == "1250"
    assert gsm_style.normalize_answer(r"\boxed{36}") == "36"
    assert gsm_style.normalize_answer(r"\frac{6}{4}") == "3/2"
    assert gsm_style.normalize_answer("7.5 meters") == "15/2"
    assert gsm_style.normalize_answer("-4") == "-4"


def test_gsm_style_verifier_accepts_final_and_gsm8k_markers():
    problem = gsm_style.GSMStyleProblem(
        problem_id="gsm-fixture",
        source="fixture",
        question="What is 600 + 650?",
        answer="$1,250",
        normalized_answer="1250",
    )

    final = gsm_style.verify_answer(problem, "work\nFINAL: $1,250")
    gsm8k = gsm_style.verify_answer(problem, "work\n#### 1250")
    wrong = gsm_style.verify_answer(problem, "work\nFINAL: 1,251")
    missing = gsm_style.verify_answer(problem, "work only")

    assert final.success
    assert gsm8k.success
    assert not wrong.success
    assert wrong.normalized_value == "1251"
    assert not missing.success
    assert missing.error == "missing final answer"


def test_gsm_style_report_input_validates_summary_counts(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    rows = [
        {
            "question": "What is 20 + 22?",
            "answer": "42",
            "source": "fixture",
        },
        {
            "question": "What is 3/4 + 1/2?",
            "answer": "5/4",
            "source": "fixture",
        },
    ]
    gsm_style.write_jsonl(dev, rows)
    problems = gsm_style.load_problems(dev)
    gsm_style.write_jsonl(dev, [problem.to_json() for problem in problems])
    split_registry = data_root / "split_registry.json"
    gsm_style.write_json(
        split_registry,
        gsm_style.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    gsm_style.write_json(
        summary,
        gsm_style.summarize_evaluations(
            [
                gsm_style.evaluate_fixture_rollouts(
                    problem,
                    [f"FINAL: {problem.answer}"],
                )
                for problem in problems
            ]
        ),
    )

    report_input = gsm_style.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=1,
    )

    assert all(report_input["checks"].values())
    assert report_input["run"]["task"] == "gsm_style"
    assert report_input["run"]["scaffold"]["budget"] == 1
    assert report_input["verifier"]["kind"] == "exact"


def test_gsm_style_import_public_rows_records_revision_source(tmp_path):
    rows = [
        {
            "question": "Mia has 3 bags with 4 shells each. How many shells?",
            "answer": "Mia has 3 * 4 = 12 shells.\n#### 12",
        },
        {
            "question": "A rope is 3 meters split in half. How long is each piece?",
            "answer": "Each piece is 3 / 2 = 1.5 meters.\n#### 1.5",
        },
    ]

    problems = gsm_style.import_public_rows(
        rows,
        source="openai/gsm8k:main:abc123:test",
        limit=1,
        offset=1,
    )
    provenance = gsm_style.build_public_provenance(
        dataset="openai/gsm8k",
        subset="main",
        revision="abc123",
        source_split="test",
        output=tmp_path / "dev.jsonl",
        limit=1,
        offset=1,
        problems=problems,
    )

    assert len(problems) == 1
    assert problems[0].answer == "1.5"
    assert problems[0].normalized_answer == "3/2"
    assert problems[0].source == "openai/gsm8k:main:abc123:test"
    assert provenance["revision"] == "abc123"
    assert provenance["num_problems"] == 1


def test_gsm_style_parser_has_fixture_commands():
    parser = build_parser()

    prepare = parser.parse_args(
        [
            "prepare-gsm-style-split",
            "--input",
            "raw.jsonl",
            "--output",
            "prepared.jsonl",
        ]
    )
    evaluate = parser.parse_args(
        [
            "evaluate-gsm-style-fixture",
            "--problems",
            "problems.jsonl",
            "--rollouts",
            "rollouts.jsonl",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
        ]
    )

    assert str(prepare.input) == "raw.jsonl"
    assert str(evaluate.summary) == "summary.json"


def test_gsm8k_import_parser_requires_pinned_revision():
    parser = build_parser()

    args = parser.parse_args(
        [
            "import-gsm8k-split",
            "--output",
            "dev.jsonl",
            "--provenance",
            "dev_provenance.json",
            "--revision",
            "abc123",
            "--limit",
            "8",
            "--offset",
            "4",
        ]
    )

    assert args.dataset == "openai/gsm8k"
    assert args.subset == "main"
    assert args.source_split == "test"
    assert args.revision == "abc123"
    assert args.limit == 8
    assert args.offset == 4


def test_gsm_style_vllm_parser_defaults_to_chat_prompt():
    parser = build_parser()

    args = parser.parse_args(
        [
            "evaluate-gsm-style-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
        ]
    )

    assert args.prompt_variant == "chat"
    assert args.num_rollouts == 32
    assert args.max_new_tokens == 256


def test_math_style_normalizes_boxed_numeric_and_symbolic_answers():
    assert math_style.normalize_answer(r"\boxed{2}") == "2"
    assert math_style.normalize_answer(r"\boxed{\frac{6}{4}}") == "3/2"
    assert math_style.normalize_answer("$7.5$") == "15/2"
    assert math_style.normalize_answer(r"\boxed{\sqrt{3}}") == "sqrt3"


def test_math_style_verifier_accepts_final_and_boxed_markers():
    problem = math_style.MathStyleProblem(
        problem_id="math-fixture",
        source="fixture",
        problem="How many roots?",
        answer=r"\boxed{2}",
        normalized_answer="2",
    )

    final = math_style.verify_answer(problem, "work\nFINAL: 2")
    boxed = math_style.verify_answer(problem, r"work therefore \boxed{2}")
    wrong = math_style.verify_answer(problem, "work\nFINAL: 3")
    missing = math_style.verify_answer(problem, "work only")

    assert final.success
    assert boxed.success
    assert not wrong.success
    assert wrong.normalized_value == "3"
    assert not missing.success
    assert missing.error == "missing final answer"


def test_math_style_import_public_rows_records_revision_source(tmp_path):
    rows = [
        {
            "problem": r"How many vertical asymptotes does $1/(x^2-1)$ have?",
            "level": "Level 3",
            "type": "Algebra",
            "solution": r"The roots are $1$ and $-1$, so there are \boxed{2}.",
        },
        {
            "problem": "What is half of 3?",
            "level": "Level 1",
            "type": "Algebra",
            "solution": r"Half of 3 is \boxed{\frac{3}{2}}.",
        },
    ]

    problems = math_style.import_public_rows(
        rows,
        source="EleutherAI/hendrycks_math:algebra:abc123:test",
        limit=1,
        offset=1,
    )
    provenance = math_style.build_public_provenance(
        dataset="EleutherAI/hendrycks_math",
        subset="algebra",
        revision="abc123",
        source_split="test",
        output=tmp_path / "dev.jsonl",
        limit=1,
        offset=1,
        problems=problems,
    )

    assert len(problems) == 1
    assert problems[0].answer == r"\frac{3}{2}"
    assert problems[0].normalized_answer == "3/2"
    assert problems[0].category == "Algebra"
    assert provenance["revision"] == "abc123"
    assert provenance["num_problems"] == 1


def test_math_style_report_input_validates_summary_counts(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    rows = [
        {
            "problem_id": "math-1",
            "source": "fixture",
            "problem": "What is 1 + 1?",
            "answer": "2",
            "normalized_answer": "2",
        },
        {
            "problem_id": "math-2",
            "source": "fixture",
            "problem": "What is 3/2?",
            "answer": r"\frac{3}{2}",
            "normalized_answer": "3/2",
        },
    ]
    math_style.write_jsonl(dev, rows)
    problems = math_style.load_problems(dev)
    split_registry = data_root / "split_registry.json"
    math_style.write_json(
        split_registry,
        math_style.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    math_style.write_json(
        summary,
        math_style.summarize_evaluations(
            [
                math_style.evaluate_fixture_rollouts(
                    problem,
                    [f"FINAL: {problem.answer}"],
                )
                for problem in problems
            ]
        ),
    )

    report_input = math_style.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=1,
    )

    assert all(report_input["checks"].values())
    assert report_input["run"]["task"] == "math_style"
    assert report_input["run"]["scaffold"]["budget"] == 1
    assert report_input["verifier"]["limitations"]


def test_math_style_load_evaluations_rescores_rollout_texts(tmp_path):
    problem = math_style.MathStyleProblem(
        problem_id="math-1",
        source="fixture",
        problem="What is 2 + 2?",
        answer="4",
        normalized_answer="4",
    )
    evaluations = [
        math_style.evaluate_fixture_rollouts(
            problem,
            ["FINAL: 5", "FINAL: 4"],
        )
    ]
    path = tmp_path / "evaluations.jsonl"
    math_style.write_jsonl(path, [evaluation.to_json() for evaluation in evaluations])

    loaded = math_style.load_evaluations(path)

    assert loaded[0].solved_at() == 2
    assert loaded[0].rollouts[0].verification.normalized_value == "5"
    assert loaded[0].rollouts[1].verification.success


def test_math_style_parsers_default_to_pinned_public_algebra_and_chat_prompt():
    parser = build_parser()

    imported = parser.parse_args(
        [
            "import-math-split",
            "--output",
            "dev.jsonl",
            "--provenance",
            "dev_provenance.json",
            "--revision",
            "abc123",
            "--limit",
            "8",
        ]
    )
    evaluated = parser.parse_args(
        [
            "evaluate-math-style-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
        ]
    )

    assert imported.dataset == "EleutherAI/hendrycks_math"
    assert imported.subset == "algebra"
    assert imported.source_split == "test"
    assert evaluated.prompt_variant == "chat"
    assert evaluated.num_rollouts == 32
    assert evaluated.max_new_tokens == 512


def test_coding_style_verifier_runs_python_tests():
    problem = coding_style.CodingStyleProblem(
        problem_id="HumanEval/fixture",
        source="fixture",
        prompt="def add_one(x):\n    ",
        test="def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0",
        entry_point="add_one",
    )

    correct = coding_style.verify_solution(problem, "return x + 1")
    wrong = coding_style.verify_solution(problem, "return x + 2")

    assert correct.success
    assert not wrong.success
    assert wrong.error == "assertion failure"


def test_coding_style_extracts_markdown_fenced_code():
    problem = coding_style.CodingStyleProblem(
        problem_id="HumanEval/fence",
        source="fixture",
        prompt="def square(x):\n    ",
        test="def check(candidate):\n    assert candidate(4) == 16",
        entry_point="square",
    )

    verified = coding_style.verify_solution(
        problem,
        "Here is the code:\n```python\ndef square(x):\n    return x * x\n```",
    )

    assert verified.success
    assert verified.extracted_code.startswith("def square")


def test_coding_style_import_public_rows_records_revision_source(tmp_path):
    rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers, threshold):\n    ",
            "canonical_solution": "return False",
            "test": "def check(candidate):\n    assert candidate([1.0, 2.0], 0.1) == False",
            "entry_point": "has_close_elements",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def separate_paren_groups(paren_string):\n    ",
            "canonical_solution": "return []",
            "test": "def check(candidate):\n    assert candidate('()') == ['()']",
            "entry_point": "separate_paren_groups",
        },
    ]

    problems = coding_style.import_public_rows(
        rows,
        source="openai/openai_humaneval:abc123:test",
        limit=1,
        offset=1,
    )
    provenance = coding_style.build_public_provenance(
        dataset="openai/openai_humaneval",
        subset=None,
        revision="abc123",
        source_split="test",
        output=tmp_path / "dev.jsonl",
        limit=1,
        offset=1,
        problems=problems,
    )

    assert len(problems) == 1
    assert problems[0].problem_id == "HumanEval/1"
    assert problems[0].entry_point == "separate_paren_groups"
    assert provenance["revision"] == "abc123"
    assert provenance["num_problems"] == 1


def test_coding_style_report_input_validates_summary_counts(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    rows = [
        {
            "problem_id": "HumanEval/fixture",
            "source": "fixture",
            "prompt": "def add_one(x):\n    ",
            "test": "def check(candidate):\n    assert candidate(1) == 2",
            "entry_point": "add_one",
        }
    ]
    coding_style.write_jsonl(dev, rows)
    problems = coding_style.load_problems(dev)
    split_registry = data_root / "split_registry.json"
    coding_style.write_json(
        split_registry,
        coding_style.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    coding_style.write_json(
        summary,
        coding_style.summarize_evaluations(
            [
                coding_style.evaluate_fixture_rollouts(
                    problems[0],
                    ["return x + 1"],
                )
            ]
        ),
    )

    report_input = coding_style.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=1,
    )

    assert all(report_input["checks"].values())
    assert report_input["run"]["task"] == "coding_style"
    assert report_input["run"]["lane"] == "coding"
    assert report_input["verifier"]["kind"] == "executable"


def test_coding_style_parsers_default_to_pinned_public_humaneval_and_chat_prompt():
    parser = build_parser()

    imported = parser.parse_args(
        [
            "import-humaneval-split",
            "--output",
            "dev.jsonl",
            "--provenance",
            "dev_provenance.json",
            "--revision",
            "abc123",
            "--limit",
            "4",
        ]
    )
    evaluated = parser.parse_args(
        [
            "evaluate-coding-style-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
        ]
    )

    assert imported.dataset == "openai/openai_humaneval"
    assert imported.subset is None
    assert imported.source_split == "test"
    assert evaluated.prompt_variant == "chat"
    assert evaluated.num_rollouts == 4
    assert evaluated.max_new_tokens == 512


def test_modular_sequences_generation_is_deterministic():
    first = modular_sequences.generate_split(seed=500, num_problems=4)
    second = modular_sequences.generate_split(seed=500, num_problems=4)

    assert [problem.to_json() for problem in first] == [
        problem.to_json() for problem in second
    ]
    assert len({problem.problem_id for problem in first}) == 4


def test_modular_sequences_verifier_requires_exact_final_value():
    problem = modular_sequences.generate_split(seed=501, num_problems=1)[0]

    missing = modular_sequences.verify_answer(problem, str(problem.answer))
    wrong = modular_sequences.verify_answer(problem, f"FINAL: {problem.answer + 1}")
    correct = modular_sequences.verify_answer(
        problem,
        f"work\nFINAL: {problem.answer}",
    )

    assert not missing.success
    assert missing.error == "missing FINAL line"
    assert not wrong.success
    assert wrong.strict_final
    assert correct.success
    assert correct.strict_final


def test_modular_sequences_summary_reports_pass_curves():
    problem = modular_sequences.generate_split(seed=502, num_problems=1)[0]
    evaluation = modular_sequences.evaluate_fixture_rollouts(
        problem,
        [
            "no final",
            f"FINAL: {problem.answer + 1}",
            f"FINAL: {problem.answer}",
        ],
    )

    summary = modular_sequences.summarize_evaluations([evaluation], ks=(1, 2, 3))

    assert summary["pass_at_k"] == {"1": 0.0, "2": 0.0, "3": 1.0}
    assert summary["strict_format_pass_at_k"] == {"1": 0.0, "2": 0.0, "3": 1.0}
    assert summary["bucket_counts"] == {"easy": 0, "elicitable": 1, "unreached": 0}
    assert summary["failure_breakdown"]["missing FINAL line"] == 1
    assert summary["failure_breakdown"]["success"] == 1


def test_modular_sequences_training_examples_use_verified_rollouts():
    problem = modular_sequences.generate_split(seed=502, num_problems=1)[0]
    evaluation = modular_sequences.evaluate_fixture_rollouts(
        problem,
        [
            "no final",
            f"FINAL: {problem.answer}",
        ],
    )

    examples = modular_sequences.build_training_examples([evaluation])

    assert len(examples) == 1
    assert examples[0].question == modular_sequences.prompt_for_problem(problem)
    assert examples[0].answer == f"FINAL: {problem.answer}"
    assert examples[0].problem_id == problem.problem_id
    assert examples[0].source_rollout_ids == (f"{problem.problem_id}:1",)


def test_modular_sequences_training_examples_skip_unreached_problems():
    problem = modular_sequences.generate_split(seed=503, num_problems=1)[0]
    evaluation = modular_sequences.evaluate_fixture_rollouts(
        problem,
        [
            "no final",
            f"FINAL: {problem.answer + 1}",
        ],
    )

    assert modular_sequences.build_training_examples([evaluation]) == []


def test_modular_sequences_split_registry_rejects_overlap(tmp_path):
    problems = modular_sequences.generate_split(seed=504, num_problems=2)
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    modular_sequences.write_jsonl(train, [problem.to_json() for problem in problems])
    modular_sequences.write_jsonl(dev, [problems[0].to_json()])

    registry = modular_sequences.build_split_registry({"train": train, "dev": dev})

    assert not registry["selected"]
    assert registry["overlaps"] == [
        {
            "problem_id": problems[0].problem_id,
            "first_split": "train",
            "second_split": "dev",
        }
    ]


def test_modular_sequences_report_input_validates_summary_counts(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    problems = modular_sequences.generate_split(seed=505, num_problems=2)
    modular_sequences.write_jsonl(dev, [problem.to_json() for problem in problems])
    split_registry = data_root / "split_registry.json"
    modular_sequences.write_json(
        split_registry,
        modular_sequences.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    modular_sequences.write_json(
        summary,
        modular_sequences.summarize_evaluations(
            [
                modular_sequences.evaluate_fixture_rollouts(
                    problem,
                    [f"FINAL: {problem.answer}"],
                )
                for problem in problems
            ]
        ),
    )

    report_input = modular_sequences.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
    )

    assert all(report_input["checks"].values())
    assert report_input["run"]["task"] == "modular_sequences"
    assert report_input["verifier"]["kind"] == "exact"


def test_modular_sequences_report_input_accepts_adapter_summary_names(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    problems = modular_sequences.generate_split(seed=506, num_problems=2)
    modular_sequences.write_jsonl(dev, [problem.to_json() for problem in problems])
    split_registry = data_root / "split_registry.json"
    modular_sequences.write_json(
        split_registry,
        modular_sequences.build_split_registry({"dev": dev}),
    )
    summary = results_root / "adapter_raw_dev_summary.json"
    modular_sequences.write_json(
        summary,
        modular_sequences.summarize_evaluations(
            [
                modular_sequences.evaluate_fixture_rollouts(
                    problem,
                    [f"FINAL: {problem.answer}"],
                )
                for problem in problems
            ]
        ),
    )

    report_input = modular_sequences.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"adapter_raw_dev": summary},
    )

    assert all(report_input["checks"].values())
    assert "adapter_raw_dev" in report_input["metrics"]["splits"]


def test_modular_sequences_report_input_builds_transfer_analysis(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    problems = modular_sequences.generate_split(seed=507, num_problems=3)
    modular_sequences.write_jsonl(dev, [problem.to_json() for problem in problems])
    split_registry = data_root / "split_registry.json"
    modular_sequences.write_json(
        split_registry,
        modular_sequences.build_split_registry({"dev": dev}),
    )
    base_evaluations = [
        modular_sequences.evaluate_fixture_rollouts(
            problems[0],
            ["no final", f"FINAL: {problems[0].answer}"],
        ),
        modular_sequences.evaluate_fixture_rollouts(
            problems[1],
            [f"FINAL: {problems[1].answer}"],
        ),
        modular_sequences.evaluate_fixture_rollouts(
            problems[2],
            ["no final"],
        ),
    ]
    adapter_evaluations = [
        modular_sequences.evaluate_fixture_rollouts(
            problems[0],
            [f"FINAL: {problems[0].answer}"],
        ),
        modular_sequences.evaluate_fixture_rollouts(
            problems[1],
            ["no final"],
        ),
        modular_sequences.evaluate_fixture_rollouts(
            problems[2],
            ["no final"],
        ),
    ]
    base_summary = results_root / "base_dev_summary.json"
    adapter_summary = results_root / "adapter_raw_dev_summary.json"
    base_rows = results_root / "base_dev_evaluations.jsonl"
    adapter_rows = results_root / "adapter_raw_dev_evaluations.jsonl"
    modular_sequences.write_json(
        base_summary,
        modular_sequences.summarize_evaluations(base_evaluations),
    )
    modular_sequences.write_json(
        adapter_summary,
        modular_sequences.summarize_evaluations(adapter_evaluations),
    )
    modular_sequences.write_jsonl(
        base_rows,
        [evaluation.to_json() for evaluation in base_evaluations],
    )
    modular_sequences.write_jsonl(
        adapter_rows,
        [evaluation.to_json() for evaluation in adapter_evaluations],
    )

    report_input = modular_sequences.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": base_summary, "adapter_raw_dev": adapter_summary},
        evaluation_paths={"dev": base_rows, "adapter_raw_dev": adapter_rows},
    )

    subset_rows = report_input["analysis"]["base_elicitable_subsets"]
    dev_base_subset = [
        row for row in subset_rows if row["split"] == "dev" and row["arm"] == "base"
    ][0]
    dev_raw_subset = [
        row
        for row in subset_rows
        if row["split"] == "dev" and row["arm"] == "adapter_raw"
    ][0]
    assert dev_base_subset["num_problems"] == 1
    assert dev_base_subset["pass_at_1"] == 0.0
    assert dev_base_subset["pass_at_8"] == 1.0
    assert dev_raw_subset["pass_at_1"] == 1.0
    examples = report_input["analysis"]["representative_examples"]
    assert {example["category"] for example in examples} == {
        "win",
        "regression",
        "unchanged_failure",
    }


def test_modular_sequences_vllm_parser_defaults_to_chat_prompt():
    parser = build_parser()

    args = parser.parse_args(
        [
            "evaluate-modular-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
        ]
    )

    assert args.prompt_variant == "chat"
    assert args.num_rollouts == 32
    assert args.max_new_tokens == 384


def test_modular_sequences_vllm_parser_accepts_concise_chat_prompt():
    parser = build_parser()

    args = parser.parse_args(
        [
            "evaluate-modular-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
            "--prompt-variant",
            "concise_chat",
        ]
    )

    assert args.prompt_variant == "concise_chat"


def test_modular_sequences_vllm_parser_accepts_lora_adapter():
    parser = build_parser()

    args = parser.parse_args(
        [
            "evaluate-modular-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
            "--lora-adapter",
            "adapter",
            "--lora-name",
            "raw",
            "--max-lora-rank",
            "16",
        ]
    )

    assert str(args.lora_adapter) == "adapter"
    assert args.lora_name == "raw"
    assert args.max_lora_rank == 16


def test_qwen3_modular_sequences_config_uses_scaffold_data_root(monkeypatch, tmp_path):
    pytest.importorskip("spmd_types")

    from torchtitan.hf_datasets.text_datasets import ChatDataLoader
    from torchtitan.models.qwen3.config_registry import (
        qwen3_1_7b_modular_sequences_lora_raw,
    )

    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    monkeypatch.setenv("TORCHTITAN_SCAFFOLD_TO_POLICY_DATA_ROOT", str(data_root))
    monkeypatch.setenv("TORCHTITAN_SCAFFOLD_TO_POLICY_RESULTS_ROOT", str(results_root))
    monkeypatch.setenv("TORCHTITAN_SCAFFOLD_TO_POLICY_STEPS", "7")
    monkeypatch.setenv("TORCHTITAN_SCAFFOLD_TO_POLICY_LORA_RANK", "8")

    config = qwen3_1_7b_modular_sequences_lora_raw()

    assert isinstance(config.dataloader, ChatDataLoader.Config)
    assert config.training.steps == 7
    assert config.dataloader.load_dataset_kwargs["data_files"] == str(
        data_root / "train" / "modular_sequences_raw.jsonl"
    )
    assert config.dump_folder == str(results_root / "train" / "modular_sequences_raw")
