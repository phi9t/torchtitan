# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset construction for Countdown scaffold-to-policy compression."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.countdown_search_distill.countdown import (
    CountdownProblem,
)
from torchtitan.experiments.countdown_search_distill.evaluate import (
    ProblemEvaluation,
)


@dataclass(frozen=True)
class TrainingExample:
    question: str
    answer: str
    condition: str
    problem_id: str
    source_rollout_ids: tuple[str, ...]
    hint: str | None = None
    curriculum_stage: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "condition": self.condition,
            "problem_id": self.problem_id,
            "source_rollout_ids": list(self.source_rollout_ids),
            "hint": self.hint,
            "curriculum_stage": self.curriculum_stage,
        }


def countdown_question(problem: CountdownProblem, hint: str | None = None) -> str:
    question = problem.prompt().strip()
    if hint is None:
        return question
    return f"{question}\n\nHint:\n{hint.strip()}"


def clean_teacher_rewrite(answer: str) -> str:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    return "\n".join(lines)


def formatting_teacher_rewrite(evaluation: ProblemEvaluation, answer: str) -> str:
    success = _shortest_success(evaluation)
    if success is None:
        return clean_teacher_rewrite(answer)
    operation_lines = [operation.raw for operation in success.verification.operations]
    return "\n".join([*operation_lines, f"FINAL: {evaluation.problem.target}"])


def synthesize_hint(evaluation: ProblemEvaluation) -> str | None:
    success = _shortest_success(evaluation)
    if success is None:
        return None
    operations = success.verification.operations
    if not operations:
        return None
    first_step = operations[0]
    last_step = operations[-1]
    return (
        "A verified path starts by combining "
        f"{first_step.left} {first_step.op} {first_step.right}. "
        f"Keep the target-producing value {last_step.result} available."
    )


def build_training_examples(
    evaluations: Sequence[ProblemEvaluation],
    *,
    conditions: Iterable[str] = (
        "raw",
        "clean",
        "formatting",
        "hindsight",
        "curriculum",
    ),
    matched_only: bool = False,
) -> dict[str, list[TrainingExample]]:
    selected_conditions = set(conditions)
    examples = {condition: [] for condition in selected_conditions}
    eligible = []
    for evaluation in evaluations:
        success = _shortest_success(evaluation)
        hint = synthesize_hint(evaluation)
        has_all = success is not None and hint is not None
        if matched_only and not has_all:
            continue
        if success is not None:
            eligible.append((evaluation, success, hint))

    for evaluation, success, hint in eligible:
        source_id = _rollout_id(evaluation.problem_id, success.sample_index)
        answer = success.text.strip()
        if "raw" in selected_conditions:
            examples["raw"].append(
                TrainingExample(
                    question=countdown_question(evaluation.problem),
                    answer=answer,
                    condition="raw",
                    problem_id=evaluation.problem_id,
                    source_rollout_ids=(source_id,),
                )
            )
        if "clean" in selected_conditions:
            examples["clean"].append(
                TrainingExample(
                    question=countdown_question(evaluation.problem),
                    answer=clean_teacher_rewrite(answer),
                    condition="clean",
                    problem_id=evaluation.problem_id,
                    source_rollout_ids=(source_id,),
                )
            )
        if "formatting" in selected_conditions:
            examples["formatting"].append(
                TrainingExample(
                    question=countdown_question(evaluation.problem),
                    answer=formatting_teacher_rewrite(evaluation, answer),
                    condition="formatting",
                    problem_id=evaluation.problem_id,
                    source_rollout_ids=(source_id,),
                )
            )
        if hint is not None and "hindsight" in selected_conditions:
            examples["hindsight"].append(
                TrainingExample(
                    question=countdown_question(evaluation.problem, hint),
                    answer=answer,
                    condition="hindsight",
                    problem_id=evaluation.problem_id,
                    source_rollout_ids=(source_id,),
                    hint=hint,
                )
            )
        if hint is not None and "curriculum" in selected_conditions:
            for stage, stage_hint in (
                ("hint_present", hint),
                ("hint_dropout", hint),
                ("hint_absent", None),
            ):
                examples["curriculum"].append(
                    TrainingExample(
                        question=countdown_question(evaluation.problem, stage_hint),
                        answer=answer,
                        condition="curriculum",
                        problem_id=evaluation.problem_id,
                        source_rollout_ids=(source_id,),
                        hint=stage_hint,
                        curriculum_stage=stage,
                    )
                )
    return examples


def write_training_jsonl(examples: Sequence[TrainingExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example.to_json(), sort_keys=True) + "\n")


def write_training_sets(
    examples_by_condition: dict[str, Sequence[TrainingExample]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition, examples in examples_by_condition.items():
        write_training_jsonl(examples, output_dir / f"{condition}.jsonl")


def build_canonical_raw_examples(
    evaluations: Sequence[ProblemEvaluation],
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for evaluation in evaluations:
        if not evaluation.problem.solution:
            continue
        examples.append(
            TrainingExample(
                question=countdown_question(evaluation.problem),
                answer="\n".join(evaluation.problem.solution).strip(),
                condition="raw",
                problem_id=evaluation.problem_id,
                source_rollout_ids=(f"{evaluation.problem_id}:canonical",),
            )
        )
    return examples


def _shortest_success(evaluation: ProblemEvaluation):
    successful = [
        rollout for rollout in evaluation.rollouts if rollout.verification.success
    ]
    if not successful:
        return None
    return min(successful, key=lambda rollout: (rollout.verification.steps_consumed, rollout.sample_index))


def _rollout_id(problem_id: str, sample_index: int) -> str:
    return f"{problem_id}:{sample_index}"
