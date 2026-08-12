# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import importlib.metadata
import json
from pathlib import Path

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
from torchtitan.experiments.scaffold_to_policy import arc_grid
from torchtitan.experiments.scaffold_to_policy import coding_style
from torchtitan.experiments.scaffold_to_policy import external_harness
from torchtitan.experiments.scaffold_to_policy import gsm_style
from torchtitan.experiments.scaffold_to_policy import math_style
from torchtitan.experiments.scaffold_to_policy import modular_sequences
from torchtitan.experiments.scaffold_to_policy import multiple_choice
from torchtitan.experiments.scaffold_to_policy import report_artifacts


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
    assert report_input["checks"]["artifact_provenance_labeled"]
    assert report_input["artifacts"]["freshness"]["num_artifacts"] == 2
    assert json.loads(summary.read_text())["num_problems"] == 2


def test_shared_report_input_maps_adapter_summary_names(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    split_registry = data_root / "split_registry.json"
    summary = results_root / "adapter_raw_dev_summary.json"
    write_json(
        split_registry,
        {
            "selected": True,
            "splits": {"dev": {"num_problems": 2}},
        },
    )
    write_json(summary, {"num_problems": 2})

    report_input = report_artifacts.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="20260812T010101Z-shared",
        task="fixture_task",
        lane="reasoning",
        scaffold={"type": "fixture", "budget": 1},
        split_registry=split_registry,
        summary_paths={"adapter_raw_dev": summary},
        verifier={"kind": "exact", "name": "fixture"},
    )

    assert all(report_input["checks"].values())
    assert "adapter_raw_dev" in report_input["metrics"]["splits"]
    assert report_input["artifacts"]["freshness"]["num_artifacts"] == 2


def test_latest_report_index_selects_latest_matching_task(tmp_path):
    manifests = tmp_path / "manifests"
    first = manifests / "report_input_20260812T010000Z-a.json"
    second = manifests / "report_input_20260812T020000Z-b.json"
    other = manifests / "report_input_20260812T030000Z-other.json"
    write_json(
        first,
        {
            "run": {
                "run_id": "20260812T010000Z-a",
                "task": "math_style",
                "lane": "reasoning",
            },
            "checks": {"selected": True},
        },
    )
    write_json(
        second,
        {
            "run": {
                "run_id": "20260812T020000Z-b",
                "task": "math_style",
                "lane": "reasoning",
            },
            "checks": {"selected": False},
        },
    )
    write_json(
        other,
        {
            "run": {
                "run_id": "20260812T030000Z-other",
                "task": "coding_style",
                "lane": "coding",
            },
            "checks": {"selected": True},
        },
    )

    index = report_artifacts.build_latest_report_index(
        manifests_dir=manifests,
        task="math_style",
    )

    assert index["selected"]
    assert index["num_candidates"] == 2
    assert index["latest"]["run_id"] == "20260812T020000Z-b"
    assert not index["latest"]["checks_passed"]
    assert index["latest"]["artifact"]["sha256"]


def test_blocker_report_input_records_artifact_provenance(tmp_path):
    parser = build_parser()
    results_root = tmp_path / "results"
    blocker = results_root / "eval" / "vllm_gpu_memory_preflight.json"
    output = results_root / "manifests" / "report_input_fixture.json"
    arc_grid.write_json(
        blocker,
        {
            "schema_version": 1,
            "kind": "vllm_gpu_memory_preflight",
            "selected": False,
            "reason": "insufficient free memory",
        },
    )

    args = parser.parse_args(
        [
            "write-blocker-report-input",
            "--results-root",
            str(results_root),
            "--run-id",
            "fixture",
            "--task",
            "coding_style",
            "--lane",
            "coding",
            "--blocker-type",
            "vllm_gpu_memory_preflight",
            "--artifact",
            f"gpu_memory={blocker}",
            "--limitation",
            "No model score was produced.",
            "--output",
            str(output),
        ]
    )
    args.func(args)

    report_input = json.loads(output.read_text())
    assert report_input["run"]["scaffold"]["budget"] == 0
    assert report_input["checks"]["blocker_artifacts_present"]
    assert report_input["checks"]["artifact_provenance_labeled"]
    assert not report_input["checks"]["benchmark_execution_completed"]
    assert not report_input["checks"]["blocker_selected"]
    assert report_input["artifacts"]["details"]["gpu_memory"]["sha256"]
    assert report_input["limitations"] == ["No model score was produced."]


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
    assert report_input["checks"]["artifact_provenance_labeled"]
    assert report_input["artifacts"]["freshness"]["num_artifacts"] == 2


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


def test_math_style_imports_aime_rows_with_integer_answers():
    rows = [
        {
            "id": 1,
            "problem": "Find 20 + 4.",
            "solution": r"\boxed{024}",
            "answer": "24",
        }
    ]

    problems = math_style.import_aime_rows(
        rows,
        source="HuggingFaceH4/aime_2024:main:train",
    )

    assert problems[0].problem_id == "AIME/1"
    assert problems[0].answer == "024"
    assert problems[0].normalized_answer == "24"
    assert math_style.verify_answer(problems[0], "work\nFINAL: 024").success


def test_import_aime_split_accepts_offline_raw_cache(tmp_path):
    parser = build_parser()
    raw_cache = tmp_path / "raw" / "aime.jsonl"
    output = tmp_path / "data" / "dev.jsonl"
    provenance_path = tmp_path / "data" / "dev_provenance.json"
    math_style.write_jsonl(
        raw_cache,
        [
            {
                "id": 1,
                "problem": "Find 20 + 4.",
                "solution": r"\boxed{024}",
                "answer": "24",
            }
        ],
    )

    args = parser.parse_args(
        [
            "import-aime-split",
            "--raw-cache",
            str(raw_cache),
            "--offline",
            "--output",
            str(output),
            "--provenance",
            str(provenance_path),
            "--revision",
            "main",
            "--limit",
            "1",
        ]
    )
    args.func(args)

    problems = math_style.load_problems(output)
    provenance = json.loads(provenance_path.read_text())
    assert problems[0].problem_id == "AIME/1"
    assert provenance["row_source"] == "raw_cache"
    assert provenance["offline"]
    assert provenance["raw_cache"] == str(raw_cache)
    assert provenance["raw_cache_artifact"]["sha256"]


def test_multiple_choice_verifier_requires_final_letter():
    problem = multiple_choice.MultipleChoiceProblem(
        problem_id="gpqa-fixture",
        source="fixture",
        question="Which option is correct?",
        choices=("correct", "wrong b", "wrong c", "wrong d"),
        answer="A",
    )

    missing = multiple_choice.verify_answer(problem, "The answer is A.")
    wrong = multiple_choice.verify_answer(problem, "FINAL: B")
    correct = multiple_choice.verify_answer(problem, "trace\nFINAL: A")

    assert not missing.success
    assert missing.error == "missing final answer"
    assert not wrong.success
    assert wrong.strict_final
    assert correct.success


def test_multiple_choice_imports_gpqa_rows_and_reports(tmp_path):
    rows = [
        {
            "Question": "Which physical statement is correct?",
            "Correct Answer": "A specialist fact.",
            "Incorrect Answer 1": "Distractor one.",
            "Incorrect Answer 2": "Distractor two.",
            "Incorrect Answer 3": "Distractor three.",
            "Explanation": "Because of the governing equation.",
        }
    ]

    problems = multiple_choice.import_gpqa_rows(
        rows,
        source="Idavidrein/gpqa:gpqa_diamond:main:train",
    )
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    multiple_choice.write_jsonl(dev, [problem.to_json() for problem in problems])
    split_registry = data_root / "split_registry.json"
    multiple_choice.write_json(
        split_registry,
        multiple_choice.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    multiple_choice.write_json(
        summary,
        multiple_choice.summarize_evaluations(
            [
                multiple_choice.evaluate_fixture_rollouts(
                    problems[0],
                    ["FINAL: C", "FINAL: A"],
                )
            ],
            ks=(1, 2),
        ),
    )

    report_input = multiple_choice.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=2,
    )

    assert problems[0].answer == "A"
    assert problems[0].choices[0] == "A specialist fact."
    assert json.loads(summary.read_text())["pass_at_k"] == {"1": 0.0, "2": 1.0}
    assert all(report_input["checks"].values())
    assert report_input["run"]["task"] == "multiple_choice"


def test_import_gpqa_split_accepts_offline_raw_cache(tmp_path):
    parser = build_parser()
    raw_cache = tmp_path / "raw" / "gpqa.jsonl"
    output = tmp_path / "data" / "dev.jsonl"
    provenance_path = tmp_path / "data" / "dev_provenance.json"
    multiple_choice.write_jsonl(
        raw_cache,
        [
            {
                "Question": "Which physical statement is correct?",
                "Correct Answer": "A specialist fact.",
                "Incorrect Answer 1": "Distractor one.",
                "Incorrect Answer 2": "Distractor two.",
                "Incorrect Answer 3": "Distractor three.",
            }
        ],
    )

    args = parser.parse_args(
        [
            "import-gpqa-split",
            "--raw-cache",
            str(raw_cache),
            "--offline",
            "--output",
            str(output),
            "--provenance",
            str(provenance_path),
            "--revision",
            "main",
            "--limit",
            "1",
        ]
    )
    args.func(args)

    problems = multiple_choice.load_problems(output)
    provenance = json.loads(provenance_path.read_text())
    assert problems[0].answer == "A"
    assert problems[0].choices[0] == "A specialist fact."
    assert provenance["row_source"] == "raw_cache"
    assert provenance["offline"]


def test_arc_grid_verifier_requires_exact_final_json_grid():
    problem = arc_grid.ARCGridProblem(
        problem_id="ARC-AGI-2/fixture/0",
        source="fixture",
        train_examples=(
            arc_grid.ARCExample(
                input_grid=((1, 0), (0, 1)),
                output_grid=((0, 1), (1, 0)),
            ),
        ),
        test_input=((2, 0), (0, 2)),
        test_output=((0, 2), (2, 0)),
    )

    missing = arc_grid.verify_answer(problem, "[[0,2],[2,0]]")
    malformed = arc_grid.verify_answer(problem, "FINAL: not json")
    wrong = arc_grid.verify_answer(problem, "FINAL: [[2,0],[0,2]]")
    correct = arc_grid.verify_answer(problem, "trace\nFINAL: [[0,2],[2,0]]")

    assert not missing.success
    assert missing.error == "missing final grid"
    assert not malformed.success
    assert malformed.strict_final
    assert not wrong.success
    assert wrong.error == "final grid does not match answer"
    assert correct.success


def test_arc_grid_strict_prompt_requests_one_line_final_only():
    problem = arc_grid.ARCGridProblem(
        problem_id="ARC-AGI-2/fixture/0",
        source="fixture",
        train_examples=(
            arc_grid.ARCExample(
                input_grid=((1, 0), (0, 1)),
                output_grid=((0, 1), (1, 0)),
            ),
        ),
        test_input=((2, 0), (0, 2)),
        test_output=((0, 2), (2, 0)),
    )

    prompt = arc_grid.strict_prompt_for_problem(problem)

    assert "Reply with exactly one line: FINAL: <json-grid>" in prompt
    assert "Do not include analysis" in prompt
    assert "Return a short reasoning trace" not in prompt


def test_arc_grid_compact_prompt_preserves_examples_and_final_contract():
    problem = arc_grid.ARCGridProblem(
        problem_id="ARC-AGI-2/fixture/0",
        source="fixture",
        train_examples=(
            arc_grid.ARCExample(
                input_grid=((1, 0), (0, 1)),
                output_grid=((0, 1), (1, 0)),
            ),
        ),
        test_input=((2, 0), (0, 2)),
        test_output=((0, 2), (2, 0)),
    )

    compact = arc_grid.compact_prompt_for_problem(problem)
    full = arc_grid.prompt_for_problem(problem)

    assert compact.startswith("ARC. Infer output. End FINAL:<json-grid>.")
    assert "E1 I=[[1,0],[0,1]] O=[[0,1],[1,0]]" in compact
    assert "T=[[2,0],[0,2]]" in compact
    assert len(compact) < len(full)


def test_arc_grid_packed_prompt_preserves_grids_with_shorter_serialization():
    problem = arc_grid.ARCGridProblem(
        problem_id="ARC-AGI-2/fixture/0",
        source="fixture",
        train_examples=(
            arc_grid.ARCExample(
                input_grid=((1, 0), (0, 1)),
                output_grid=((0, 1), (1, 0)),
            ),
        ),
        test_input=((2, 0), (0, 2)),
        test_output=((0, 2), (2, 0)),
    )

    packed = arc_grid.packed_prompt_for_problem(problem)
    compact = arc_grid.compact_prompt_for_problem(problem)

    assert "Digits are cells, / separates rows" in packed
    assert "E1 I=10/01 O=01/10" in packed
    assert "T=20/02" in packed
    assert "FINAL:<json-grid>" in packed
    assert len(packed) < len(compact)


def test_arc_grid_imports_tasks_and_reports(tmp_path):
    task_dir = tmp_path / "arc" / "training"
    task_dir.mkdir(parents=True)
    task_path = task_dir / "abc12345.json"
    task_path.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "input": [[1, 0], [0, 1]],
                        "output": [[0, 1], [1, 0]],
                    }
                ],
                "test": [
                    {
                        "input": [[2, 0], [0, 2]],
                        "output": [[0, 2], [2, 0]],
                    }
                ],
            }
        )
    )

    problems = arc_grid.import_arc_tasks(
        [task_path],
        source="https://github.com/arcprize/ARC-AGI-2.git:abc:training",
    )
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    arc_grid.write_jsonl(dev, [problem.to_json() for problem in problems])
    split_registry = data_root / "split_registry.json"
    arc_grid.write_json(
        split_registry,
        arc_grid.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    arc_grid.write_json(
        summary,
        arc_grid.summarize_evaluations(
            [
                arc_grid.evaluate_fixture_rollouts(
                    problems[0],
                    ["FINAL: [[9]]", "FINAL: [[0,2],[2,0]]"],
                )
            ],
            ks=(1, 2),
        ),
    )

    report_input = arc_grid.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=2,
    )

    assert problems[0].problem_id == "ARC-AGI-2/abc12345/0"
    assert "FINAL: <json-grid>" in arc_grid.prompt_for_problem(problems[0])
    assert json.loads(summary.read_text())["pass_at_k"] == {"1": 0.0, "2": 1.0}
    assert all(report_input["checks"].values())
    assert report_input["run"]["task"] == "arc_grid"
    assert report_input["verifier"]["name"] == "arc_grid_exact_json_v1"


def test_arc_grid_prompt_preflight_and_report_input(tmp_path):
    problem = arc_grid.ARCGridProblem(
        problem_id="ARC-AGI-2/fixture/0",
        source="fixture",
        train_examples=(
            arc_grid.ARCExample(
                input_grid=((1, 0), (0, 1)),
                output_grid=((0, 1), (1, 0)),
            ),
        ),
        test_input=((2, 0), (0, 2)),
        test_output=((0, 2), (2, 0)),
    )
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    arc_grid.write_jsonl(dev, [problem.to_json()])
    split_registry = data_root / "split_registry.json"
    arc_grid.write_json(
        split_registry,
        arc_grid.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    arc_grid.write_json(
        summary,
        arc_grid.summarize_evaluations(
            [arc_grid.evaluate_fixture_rollouts(problem, ["FINAL: [[0,2],[2,0]]"])]
        ),
    )
    preflight = results_root / "dev_prompt_preflight.json"
    arc_grid.write_json(
        preflight,
        arc_grid.build_prompt_preflight(
            problems=[problem],
            token_counts={problem.problem_id: 20},
            max_model_len=64,
            max_new_tokens=8,
        ),
    )

    report_input = arc_grid.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="fixture",
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=1,
        preflight_paths={"dev": preflight},
    )

    assert all(report_input["checks"].values())
    assert report_input["preflight"]["splits"]["dev"]["selected"]
    assert report_input["preflight"]["splits"]["dev"]["max_total_tokens"] == 28


def test_vllm_gpu_memory_preflight_parser_accepts_low_memory_config():
    parser = build_parser()

    args = parser.parse_args(
        [
            "preflight-vllm-gpu-memory",
            "--output",
            "memory.json",
            "--gpu-memory-utilization",
            "0.24",
            "--device-index",
            "0",
        ]
    )

    assert args.gpu_memory_utilization == 0.24
    assert args.device_index == 0


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


def test_coding_style_imports_mbpp_rows_as_executable_checks():
    rows = [
        {
            "task_id": 11,
            "prompt": (
                "Write a python function to remove first and last occurrence of "
                "a given character from the string."
            ),
            "code": "def remove_Occ(s, ch):\n    return s",
            "test_imports": [],
            "test_list": [
                'assert remove_Occ("hello","l") == "heo"',
                'assert remove_Occ("abcda","a") == "bcd"',
            ],
        }
    ]

    problems = coding_style.import_mbpp_rows(
        rows,
        source="google-research-datasets/mbpp:sanitized:main:test",
    )
    verified = coding_style.verify_solution(
        problems[0],
        "def remove_Occ(s, ch):\n"
        "    first = s.find(ch)\n"
        "    last = s.rfind(ch)\n"
        "    return ''.join(c for i, c in enumerate(s) if i not in {first, last})",
    )

    assert problems[0].problem_id == "MBPP/11"
    assert problems[0].entry_point == "remove_Occ"
    assert "def remove_Occ(arg1, arg2):" in problems[0].prompt
    assert "def check(candidate):" in problems[0].test
    assert verified.success


def test_coding_style_imports_bigcodebench_rows_as_unittest_checks():
    rows = [
        {
            "task_id": "BigCodeBench/fixture",
            "code_prompt": "def task_func(x):\n    ",
            "canonical_solution": "    return x + 1",
            "test": (
                "import unittest\n"
                "class TestCases(unittest.TestCase):\n"
                "    def test_increment(self):\n"
                "        self.assertEqual(task_func(1), 2)\n"
            ),
            "entry_point": "task_func",
        }
    ]

    problems = coding_style.import_bigcodebench_rows(
        rows,
        source="bigcode/bigcodebench-hard:v0.1.4",
    )
    correct = coding_style.verify_solution(problems[0], "return x + 1")
    wrong = coding_style.verify_solution(problems[0], "return x + 2")

    assert problems[0].problem_id == "BigCodeBench/fixture"
    assert "unittest.defaultTestLoader" in problems[0].test
    assert correct.success
    assert not wrong.success
    assert wrong.error == "assertion failure"


def test_import_bigcodebench_split_accepts_offline_raw_cache(tmp_path):
    parser = build_parser()
    raw_cache = tmp_path / "raw" / "bigcodebench.jsonl"
    output = tmp_path / "data" / "dev.jsonl"
    provenance_path = tmp_path / "data" / "dev_provenance.json"
    coding_style.write_jsonl(
        raw_cache,
        [
            {
                "task_id": "BigCodeBench/fixture",
                "code_prompt": "def add_one(x):\n    ",
                "entry_point": "add_one",
                "test": "def check(candidate):\n    assert candidate(1) == 2",
                "canonical_solution": "return x + 1",
            }
        ],
    )

    args = parser.parse_args(
        [
            "import-bigcodebench-split",
            "--raw-cache",
            str(raw_cache),
            "--offline",
            "--output",
            str(output),
            "--provenance",
            str(provenance_path),
            "--revision",
            "main",
            "--limit",
            "1",
        ]
    )
    args.func(args)

    problems = coding_style.load_problems(output)
    provenance = json.loads(provenance_path.read_text())
    assert problems[0].problem_id == "BigCodeBench/fixture"
    assert provenance["row_source"] == "raw_cache"
    assert provenance["offline"]


def test_coding_style_preserves_indented_completion_bodies():
    problem = coding_style.CodingStyleProblem(
        problem_id="BigCodeBench/indented",
        source="fixture",
        prompt="def task_func():\n",
        test=(
            "def check(candidate):\n"
            "    value = candidate()\n"
            "    assert value().answer() == 3\n"
        ),
        entry_point="task_func",
        canonical_solution=(
            "    class Inner:\n"
            "        def answer(self):\n"
            "            return 3\n"
            "    return Inner\n"
        ),
    )

    verified = coding_style.verify_solution(problem, problem.canonical_solution or "")

    assert verified.success
    assert "    class Inner:" in verified.extracted_code


def test_coding_style_timeout_verification_is_json_serializable():
    problem = coding_style.CodingStyleProblem(
        problem_id="timeout",
        source="fixture",
        prompt="def task_func():\n    ",
        test="def check(candidate):\n    candidate()",
        entry_point="task_func",
    )

    verified = coding_style.verify_solution(
        problem,
        "while True:\n        pass",
        timeout_seconds=0.01,
    )

    assert not verified.success
    assert verified.error == "timeout"
    json.dumps(verified.to_json())


def test_coding_style_preflights_canonical_solutions():
    problems = [
        coding_style.CodingStyleProblem(
            problem_id="passes",
            source="fixture",
            prompt="def task_func(x):\n    ",
            test="def check(candidate):\n    assert candidate(1) == 2",
            entry_point="task_func",
            canonical_solution="    return x + 1",
        ),
        coding_style.CodingStyleProblem(
            problem_id="missing",
            source="fixture",
            prompt="def missing_func(x):\n    ",
            test="def check(candidate):\n    assert candidate(1) == 2",
            entry_point="missing_func",
            canonical_solution=None,
        ),
    ]

    preflight = coding_style.preflight_canonical_solutions(problems)

    assert not preflight["selected"]
    assert preflight["num_problems"] == 2
    assert preflight["num_passed"] == 1
    assert preflight["failure_breakdown"] == {
        "success": 1,
        "missing canonical solution": 1,
    }
    assert preflight["records"][0]["success"]
    assert preflight["records"][1]["error"] == "missing canonical solution"


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


def test_coding_style_report_input_accepts_canonical_preflight(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dev = data_root / "dev.jsonl"
    problem = coding_style.CodingStyleProblem(
        problem_id="HumanEval/fixture",
        source="fixture",
        prompt="def add_one(x):\n    ",
        test="def check(candidate):\n    assert candidate(1) == 2",
        entry_point="add_one",
        canonical_solution="    return x + 1",
    )
    coding_style.write_jsonl(dev, [problem.to_json()])
    split_registry = data_root / "split_registry.json"
    coding_style.write_json(
        split_registry,
        coding_style.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    coding_style.write_json(
        summary,
        coding_style.summarize_evaluations(
            [coding_style.evaluate_fixture_rollouts(problem, ["return x + 1"])]
        ),
    )
    preflight = results_root / "dev_canonical_preflight.json"
    coding_style.write_json(
        preflight,
        coding_style.preflight_canonical_solutions([problem]),
    )

    report_input = coding_style.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id="20260812T000000Z-unscoped",
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=1,
        preflight_paths={"dev": preflight},
    )

    assert all(report_input["checks"].values())
    assert report_input["artifacts"]["preflights"] == {
        "dev": str(preflight),
    }
    assert report_input["preflight"]["splits"]["dev"]["selected"]
    assert report_input["checks"]["artifact_provenance_labeled"]
    assert report_input["artifacts"]["freshness"]["status_counts"] == {
        "reused_or_unscoped": 3,
    }
    assert (
        report_input["artifacts"]["details"]["summaries"]["dev"]["run_binding"]["status"]
        == "reused_or_unscoped"
    )


def test_coding_style_report_input_marks_run_scoped_artifacts_fresh(tmp_path):
    run_id = "20260812T235959Z-fixture"
    data_root = tmp_path / "data"
    results_root = tmp_path / "results" / run_id
    dev = data_root / "dev.jsonl"
    problem = coding_style.CodingStyleProblem(
        problem_id="HumanEval/fixture",
        source="fixture",
        prompt="def add_one(x):\n    ",
        test="def check(candidate):\n    assert candidate(1) == 2",
        entry_point="add_one",
        canonical_solution="    return x + 1",
    )
    coding_style.write_jsonl(dev, [problem.to_json()])
    split_registry = data_root / f"split_registry_{run_id}.json"
    coding_style.write_json(
        split_registry,
        coding_style.build_split_registry({"dev": dev}),
    )
    summary = results_root / "dev_summary.json"
    coding_style.write_json(
        summary,
        {
            **coding_style.summarize_evaluations(
                [coding_style.evaluate_fixture_rollouts(problem, ["return x + 1"])]
            ),
            "run_id": run_id,
        },
    )
    preflight = results_root / "dev_canonical_preflight.json"
    coding_style.write_json(
        preflight,
        coding_style.preflight_canonical_solutions([problem]),
    )

    report_input = coding_style.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id=run_id,
        split_registry=split_registry,
        summary_paths={"dev": summary},
        scaffold_budget=1,
        preflight_paths={"dev": preflight},
    )

    assert report_input["checks"]["artifact_provenance_labeled"]
    assert report_input["artifacts"]["freshness"]["status_counts"] == {"fresh": 3}
    assert report_input["artifacts"]["details"]["summaries"]["dev"]["sha256"]


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
    assert evaluated.gpu_memory_utilization is None

    contract = parser.parse_args(
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
            "--prompt-variant",
            "contract_chat",
            "--gpu-memory-utilization",
            "0.4",
        ]
    )

    assert contract.prompt_variant == "contract_chat"
    assert contract.gpu_memory_utilization == 0.4

    math = parser.parse_args(
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
            "--gpu-memory-utilization",
            "0.5",
        ]
    )
    multiple_choice = parser.parse_args(
        [
            "evaluate-multiple-choice-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
            "--gpu-memory-utilization",
            "0.6",
        ]
    )

    assert math.gpu_memory_utilization == 0.5
    assert multiple_choice.gpu_memory_utilization == 0.6

    latest = parser.parse_args(
        [
            "write-latest-report-index",
            "--manifests-dir",
            "results/manifests",
            "--output",
            "latest.json",
            "--task",
            "coding_style",
            "--no-require-selected",
        ]
    )

    assert latest.pattern == "report_input_*.json"
    assert latest.task == "coding_style"
    assert not latest.require_selected


def test_harder_reasoning_and_coding_parsers_accept_public_commands():
    parser = build_parser()

    aime = parser.parse_args(
        [
            "import-aime-split",
            "--output",
            "dev.jsonl",
            "--revision",
            "main",
            "--limit",
            "4",
            "--raw-cache",
            "raw/aime.jsonl",
            "--offline",
        ]
    )
    gpqa = parser.parse_args(
        [
            "import-gpqa-split",
            "--output",
            "dev.jsonl",
            "--revision",
            "main",
            "--limit",
            "4",
        ]
    )
    mbpp = parser.parse_args(
        [
            "import-mbpp-split",
            "--output",
            "dev.jsonl",
            "--revision",
            "main",
            "--limit",
            "4",
        ]
    )
    bigcodebench = parser.parse_args(
        [
            "import-bigcodebench-split",
            "--output",
            "dev.jsonl",
            "--revision",
            "main",
            "--limit",
            "4",
        ]
    )
    coding_preflight = parser.parse_args(
        [
            "preflight-coding-style-canonical",
            "--problems",
            "dev.jsonl",
            "--output",
            "preflight.json",
        ]
    )
    arc = parser.parse_args(
        [
            "import-arc-grid-split",
            "--task-dir",
            "src/ARC-AGI-2/data",
            "--output",
            "dev.jsonl",
            "--revision",
            "main",
            "--limit",
            "4",
        ]
    )
    arc_preflight = parser.parse_args(
        [
            "preflight-arc-grid-prompts",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "preflight.json",
            "--prompt-variant",
            "packed_chat",
        ]
    )
    multiple = parser.parse_args(
        [
            "evaluate-multiple-choice-vllm",
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
    arc_eval = parser.parse_args(
        [
            "evaluate-arc-grid-vllm",
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
    arc_eval_low_memory = parser.parse_args(
        [
            "evaluate-arc-grid-vllm",
            "--problems",
            "problems.jsonl",
            "--model",
            "./assets/hf/Qwen3-1.7B",
            "--output",
            "evaluations.jsonl",
            "--summary",
            "summary.json",
            "--gpu-memory-utilization",
            "0.24",
        ]
    )
    blocker = parser.parse_args(
        [
            "write-blocker-report-input",
            "--results-root",
            "results",
            "--run-id",
            "run",
            "--task",
            "math_style",
            "--lane",
            "reasoning",
            "--blocker-type",
            "vllm_gpu_memory_preflight",
            "--artifact",
            "gpu_memory=preflight.json",
            "--output",
            "report_input.json",
        ]
    )

    assert aime.dataset == "HuggingFaceH4/aime_2024"
    assert aime.raw_cache == Path("raw/aime.jsonl")
    assert aime.offline
    assert gpqa.subset == "gpqa_diamond"
    assert mbpp.dataset == "google-research-datasets/mbpp"
    assert bigcodebench.dataset == "bigcode/bigcodebench-hard"
    assert bigcodebench.source_split == "v0.1.4"
    assert coding_preflight.timeout_seconds == 5.0
    assert arc.source_split == "training"
    assert arc_preflight.max_model_len == 4096
    assert arc_preflight.prompt_variant == "packed_chat"
    assert multiple.prompt_variant == "chat"
    assert multiple.num_rollouts == 4
    assert arc_eval.prompt_variant == "chat"
    assert arc_eval.max_model_len == 4096
    assert arc_eval.gpu_memory_utilization is None
    assert arc_eval_low_memory.gpu_memory_utilization == 0.24
    assert blocker.artifact == ["gpu_memory=preflight.json"]
    assert blocker.lane == "reasoning"


def test_external_harness_smoke_ingestion_records_pins_and_rootfs(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    raw = results_root / "raw" / "harbor_terminal.json"
    ingested_path = results_root / "ingested" / "harbor_terminal.json"
    report_path = results_root / "manifests" / "report_input.json"

    raw_record = external_harness.write_harness_smoke(
        output=raw,
        run_id="fixture",
        harness_family="harbor_terminal",
        pins=external_harness.default_harbor_terminal_pins(),
        dry_run=True,
        task_subset="dry-run",
    )
    ingested = external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id="fixture",
        ingested_paths={"harbor_terminal": ingested_path},
    )
    external_harness.write_json(report_path, report_input)

    assert raw_record["rootfs"]["in_rootfs"]
    assert [pin["name"] for pin in raw_record["pins"]] == [
        "harbor",
        "terminal-bench-2-1",
    ]
    assert ingested["checks"]["dry_run_labeled"]
    assert ingested["metric"]["name"] == "dry_run_compatibility"
    assert all(report_input["checks"].values())
    assert report_input["harnesses"]["harbor_terminal"]["mode"] == "dry_run"
    assert report_input["checks"]["artifact_provenance_labeled"]
    assert report_input["artifacts"]["freshness"]["num_artifacts"] == 1


def test_external_harness_installed_preflight_records_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    raw = results_root / "raw" / "tau2_preflight.json"
    ingested_path = results_root / "ingested" / "tau2_preflight.json"
    report_path = results_root / "manifests" / "report_input.json"
    pytest_version = importlib.metadata.version("pytest")

    external_harness.write_installed_preflight(
        output=raw,
        run_id="fixture",
        harness_family="tau2",
        pins=[
            external_harness.HarnessPin(
                name="fixture-harness",
                repo="https://example.com/fixture.git",
                revision="fixture",
                package_module="pytest",
                package_name="pytest",
                package_version=pytest_version,
                role="unit-test fixture",
            )
        ],
        task_subset="tau2-preflight",
        cli_names=["tau2"],
    )
    ingested = external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id="fixture",
        ingested_paths={"tau2": ingested_path},
    )
    external_harness.write_json(report_path, report_input)

    assert ingested["mode"] == "installed_preflight"
    assert ingested["checks"]["installed_preflight_labeled"]
    assert ingested["metric"]["name"] == "installed_preflight"
    assert "package import and version preflight only" in ingested["limitations"]
    assert all(report_input["checks"].values())
    assert report_input["run"]["scaffold"]["type"] == "installed_preflight"


def test_external_harness_task_score_report_accepts_success(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    raw = results_root / "raw" / "tau2_score.json"
    ingested_path = results_root / "ingested" / "tau2_score.json"

    external_harness.write_json(
        raw,
        {
            "schema_version": 1,
            "run_id": "fixture",
            "harness_family": "tau2",
            "mode": "task_score_smoke",
            "task_subset": "mock",
            "rootfs": {"in_rootfs": True},
            "tools": {},
            "pins": [
                {
                    "name": "tau2-bench",
                    "installed": True,
                    "installed_version": "1.0.1",
                    "package_version": "1.0.1",
                }
            ],
            "raw_result": {
                "metric_name": "tau2_mock_score",
                "score": 1.0,
                "num_tasks": 1,
                "score_source": "tau2 evaluator all_ignore_basis",
                "task_metadata": {"task_id": "create_task_1"},
                "trajectory": [],
            },
        },
    )
    ingested = external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id="fixture",
        ingested_paths={"tau2": ingested_path},
    )

    assert ingested["checks"]["task_score_labeled"]
    assert ingested["metric"]["task_metadata"]["task_id"] == "create_task_1"
    assert all(report_input["checks"].values())
    assert report_input["run"]["scaffold"]["type"] == "task_score_smoke"


def test_external_harness_execution_probe_report_accepts_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    raw = results_root / "raw" / "terminal_probe.json"
    ingested_path = results_root / "ingested" / "terminal_probe.json"

    external_harness.write_json(
        raw,
        {
            "schema_version": 1,
            "run_id": "fixture",
            "harness_family": "harbor_terminal",
            "mode": "task_execution_probe",
            "task_subset": "headless-terminal",
            "rootfs": {"in_rootfs": True},
            "tools": {},
            "pins": [
                {
                    "name": "terminal-bench-2-1",
                    "installed": True,
                    "installed_version": "0.2.18",
                    "package_version": "0.2.18",
                }
            ],
            "raw_result": {
                "metric_name": "terminal_bench_execution_probe",
                "score": 0.0,
                "num_tasks": 0,
                "score_source": "terminal-bench CLI oracle task execution",
                "task_metadata": {"task_id": "headless-terminal"},
                "trajectory": [],
            },
        },
    )
    ingested = external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id="fixture",
        ingested_paths={"terminal": ingested_path},
    )

    assert ingested["checks"]["task_execution_probe_labeled"]
    assert not report_input["checks"]["task_execution_probes_succeeded"]
    assert report_input["run"]["scaffold"]["type"] == "task_execution_probe"


def test_tau2_execution_probe_summarizes_upstream_results(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    tau2_results = results_root / "simulations" / "probe" / "results.json"
    raw = results_root / "raw" / "tau2_probe.json"
    ingested_path = results_root / "ingested" / "tau2_probe.json"
    external_harness.write_json(
        tau2_results,
        {
            "simulations": [
                {
                    "task_id": "create_task_1",
                    "termination_reason": "infrastructure_error",
                    "reward_info": None,
                    "info": {
                        "error_type": "TypeError",
                        "error": "DummyUser.__init__() got an unexpected keyword argument 'tools'",
                        "failed_after_attempts": 1,
                    },
                }
            ]
        },
    )

    raw_record = external_harness.write_tau2_execution_probe(
        output=raw,
        run_id="fixture",
        task_id="create_task_1",
        command=["python", "-c", "print('probe')"],
        cwd=tmp_path,
        timeout_seconds=5.0,
        results_json=tau2_results,
    )
    ingested = external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id="fixture",
        ingested_paths={"tau2": ingested_path},
    )

    metadata = raw_record["raw_result"]["task_metadata"]
    assert raw_record["mode"] == "task_execution_probe"
    assert raw_record["raw_result"]["score"] == 0.0
    assert metadata["results_present"]
    assert metadata["num_simulations"] == 1
    assert metadata["num_evaluated"] == 0
    assert metadata["num_infra_errors"] == 1
    assert not metadata["execution_completed"]
    assert metadata["errors"][0]["error_type"] == "TypeError"
    assert ingested["checks"]["task_execution_probe_labeled"]
    assert not report_input["checks"]["task_execution_probes_succeeded"]
    assert not report_input["checks"]["task_execution_probes_completed"]


def test_tau2_execution_probe_accepts_completed_zero_reward_baseline(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    tau2_results = results_root / "simulations" / "probe" / "results.json"
    raw = results_root / "raw" / "tau2_probe.json"
    ingested_path = results_root / "ingested" / "tau2_probe.json"
    external_harness.write_json(
        tau2_results,
        {
            "simulations": [
                {
                    "task_id": "create_task_1",
                    "termination_reason": "agent_stop",
                    "reward_info": {"reward": 0.0},
                    "info": {},
                }
            ]
        },
    )

    raw_record = external_harness.write_tau2_execution_probe(
        output=raw,
        run_id="fixture",
        task_id="create_task_1",
        command=["python", "-c", "print('probe')"],
        cwd=tmp_path,
        timeout_seconds=5.0,
        results_json=tau2_results,
    )
    external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id="fixture",
        ingested_paths={"tau2": ingested_path},
    )

    metadata = raw_record["raw_result"]["task_metadata"]
    assert raw_record["raw_result"]["score"] == 0.0
    assert raw_record["raw_result"]["num_tasks"] == 1
    assert metadata["execution_completed"]
    assert metadata["average_reward"] == 0.0
    assert not report_input["checks"]["task_execution_probes_succeeded"]
    assert report_input["checks"]["task_execution_probes_completed"]
    assert report_input["checks"]["artifact_provenance_labeled"]
    assert report_input["artifacts"]["details"]["ingested"]["tau2"]["sha256"]


def test_terminal_bench_execution_probe_rejects_harbor_runtime_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    run_id = "fixture-terminal"
    harbor_result = results_root / "runs" / run_id / "result.json"
    trial_result = (
        results_root / "runs" / run_id / "headless-terminal__abc" / "result.json"
    )
    raw = results_root / "raw" / "terminal_probe.json"
    ingested_path = results_root / "ingested" / "terminal_probe.json"
    external_harness.write_json(
        harbor_result,
        {
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
                "n_errored_trials": 1,
                "evals": {
                    "oracle__adhoc": {
                        "n_trials": 0,
                        "n_errors": 1,
                        "exception_stats": {
                            "RuntimeError": ["headless-terminal__abc"],
                        },
                    },
                },
            },
        },
    )
    external_harness.write_json(
        trial_result,
        {
            "trial_name": "headless-terminal__abc",
            "exception_info": {
                "exception_type": "RuntimeError",
                "exception_message": "missing verifier bind mount",
            },
        },
    )

    raw_record = external_harness.write_terminal_bench_execution_probe(
        output=raw,
        run_id=run_id,
        task_id="headless-terminal",
        command=["python", "-c", "print('harbor returned zero')"],
        cwd=tmp_path,
        timeout_seconds=5.0,
        harbor_result_json=harbor_result,
        agent_name="oracle",
    )
    ingested = external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id=run_id,
        ingested_paths={"terminal": ingested_path},
    )

    metadata = raw_record["raw_result"]["task_metadata"]
    assert raw_record["raw_result"]["score"] == 0.0
    assert raw_record["raw_result"]["num_tasks"] == 0
    assert metadata["returncode"] == 0
    assert metadata["agent_name"] == "oracle"
    assert not metadata["execution_completed"]
    assert metadata["results_present"]
    assert metadata["num_trials"] == 0
    assert metadata["num_errors"] == 1
    assert metadata["num_trial_exceptions"] == 1
    assert metadata["trial_exceptions"][0]["exception_type"] == "RuntimeError"
    assert ingested["checks"]["task_execution_probe_labeled"]
    assert not report_input["checks"]["task_execution_probes_succeeded"]
    assert not report_input["checks"]["task_execution_probes_completed"]


def test_terminal_bench_execution_probe_accepts_completed_zero_score_baseline(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    results_root = tmp_path / "results"
    run_id = "fixture-terminal-nop"
    harbor_result = results_root / "runs" / run_id / "result.json"
    raw = results_root / "raw" / "terminal_probe.json"
    ingested_path = results_root / "ingested" / "terminal_probe.json"
    external_harness.write_json(
        harbor_result,
        {
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
                "n_errored_trials": 0,
                "evals": {
                    "nop__adhoc": {
                        "n_trials": 1,
                        "n_errors": 0,
                        "metrics": [{"mean": 0.0}],
                    },
                },
            },
        },
    )

    raw_record = external_harness.write_terminal_bench_execution_probe(
        output=raw,
        run_id=run_id,
        task_id="headless-terminal",
        command=["python", "-c", "print('harbor baseline')"],
        cwd=tmp_path,
        timeout_seconds=5.0,
        harbor_result_json=harbor_result,
        agent_name="nop",
    )
    external_harness.ingest_harness_smoke(
        raw_result=raw,
        output=ingested_path,
        results_root=results_root,
    )
    report_input = external_harness.build_report_input(
        results_root=results_root,
        run_id=run_id,
        ingested_paths={"terminal": ingested_path},
    )

    metadata = raw_record["raw_result"]["task_metadata"]
    assert raw_record["raw_result"]["score"] == 0.0
    assert raw_record["raw_result"]["num_tasks"] == 1
    assert metadata["agent_name"] == "nop"
    assert metadata["execution_completed"]
    assert metadata["mean_metric"] == 0.0
    assert not report_input["checks"]["task_execution_probes_succeeded"]
    assert report_input["checks"]["task_execution_probes_completed"]


def test_external_harness_parser_accepts_dry_run_commands():
    parser = build_parser()

    raw = parser.parse_args(
        [
            "write-external-harness-smoke",
            "--harness-family",
            "tau2",
            "--run-id",
            "fixture",
            "--task-subset",
            "retail-dry-run",
            "--output",
            "raw.json",
        ]
    )
    report = parser.parse_args(
        [
            "build-external-harness-report-input",
            "--results-root",
            "results",
            "--run-id",
            "fixture",
            "--ingested",
            "tau2=ingested.json",
            "--output",
            "report.json",
        ]
    )

    assert raw.harness_family == "tau2"
    assert raw.dry_run
    assert report.ingested == ["tau2=ingested.json"]


def test_external_harness_parser_accepts_installed_preflight_command():
    parser = build_parser()

    args = parser.parse_args(
        [
            "write-external-harness-preflight",
            "--harness-family",
            "harbor_terminal",
            "--run-id",
            "fixture",
            "--task-subset",
            "terminal-bench-preflight",
            "--cli-name",
            "terminal-bench",
            "--output",
            "raw.json",
        ]
    )

    assert args.harness_family == "harbor_terminal"
    assert args.cli_name == ["terminal-bench"]


def test_external_harness_parser_accepts_tau2_score_command():
    parser = build_parser()

    args = parser.parse_args(
        [
            "write-tau2-mock-score-smoke",
            "--run-id",
            "fixture",
            "--task-id",
            "create_task_1",
            "--evaluation-type",
            "all_ignore_basis",
            "--output",
            "raw.json",
        ]
    )

    assert args.task_id == "create_task_1"
    assert args.evaluation_type == "all_ignore_basis"


def test_external_harness_parser_accepts_terminal_bench_commands():
    parser = build_parser()

    result = parser.parse_args(
        [
            "write-terminal-bench-result-smoke",
            "--run-id",
            "fixture",
            "--task-id",
            "headless-terminal",
            "--output",
            "raw.json",
        ]
    )
    probe = parser.parse_args(
        [
            "write-terminal-bench-execution-probe",
            "--run-id",
            "fixture",
            "--task-id",
            "headless-terminal",
            "--cwd",
            ".",
            "--harbor-result-json",
            "result.json",
            "--agent-name",
            "nop",
            "--output",
            "raw.json",
            "tb",
            "runs",
            "create",
        ]
    )

    assert result.task_id == "headless-terminal"
    assert probe.harbor_result_json == Path("result.json")
    assert probe.agent_name == "nop"
    assert probe.command == ["tb", "runs", "create"]


def test_external_harness_parser_accepts_tau2_execution_probe_command():
    parser = build_parser()

    args = parser.parse_args(
        [
            "write-tau2-execution-probe",
            "--run-id",
            "fixture",
            "--task-id",
            "create_task_1",
            "--cwd",
            ".",
            "--results-json",
            "results.json",
            "--output",
            "raw.json",
            "tau2",
            "run",
        ]
    )

    assert args.task_id == "create_task_1"
    assert args.results_json == Path("results.json")
    assert args.command == ["tau2", "run"]


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
