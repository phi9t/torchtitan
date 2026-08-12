# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Multiple-choice exact-answer helpers for harder reasoning smokes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.scaffold_to_policy import report_artifacts


FINAL_RE = re.compile(r"^\s*FINAL:\s*([A-D])\s*$", re.IGNORECASE)
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


@dataclass(frozen=True)
class MultipleChoiceProblem:
    problem_id: str
    source: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str
    explanation: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "question": self.question,
            "choices": list(self.choices),
            "answer": self.answer,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class MultipleChoiceVerification:
    success: bool
    final_value: str | None
    strict_final: bool
    error: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "final_value": self.final_value,
            "strict_final": self.strict_final,
            "error": self.error,
        }


@dataclass(frozen=True)
class MultipleChoiceRollout:
    sample_index: int
    text: str
    verification: MultipleChoiceVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class MultipleChoiceProblemEvaluation:
    problem: MultipleChoiceProblem
    rollouts: tuple[MultipleChoiceRollout, ...]

    def solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success:
                return index
        return None

    def strict_solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success and rollout.verification.strict_final:
                return index
        return None

    def bucket(self) -> str:
        solved_at = self.solved_at()
        if solved_at == 1:
            return "easy"
        if solved_at is None:
            return "unreached"
        return "elicitable"

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem.problem_id,
            "problem": self.problem.to_json(),
            "solved_at": self.solved_at(),
            "strict_solved_at": self.strict_solved_at(),
            "bucket": self.bucket(),
            "rollouts": [rollout.to_json() for rollout in self.rollouts],
        }


def prompt_for_problem(problem: MultipleChoiceProblem) -> str:
    return "\n".join(
        [
            "Answer this multiple-choice reasoning question.",
            problem.question,
            "",
            f"A. {problem.choices[0]}",
            f"B. {problem.choices[1]}",
            f"C. {problem.choices[2]}",
            f"D. {problem.choices[3]}",
            "",
            "Return a short reasoning trace.",
            "The last line must be exactly FINAL: <A|B|C|D>.",
        ]
    )


def verify_answer(
    problem: MultipleChoiceProblem,
    text: str,
) -> MultipleChoiceVerification:
    final_value = _extract_final_value(text)
    if final_value is None:
        return MultipleChoiceVerification(
            success=False,
            final_value=None,
            strict_final=False,
            error="missing final answer",
        )
    if final_value != problem.answer:
        return MultipleChoiceVerification(
            success=False,
            final_value=final_value,
            strict_final=True,
            error=f"final value {final_value} does not match answer {problem.answer}",
        )
    return MultipleChoiceVerification(
        success=True,
        final_value=final_value,
        strict_final=True,
        error=None,
    )


def evaluate_fixture_rollouts(
    problem: MultipleChoiceProblem,
    rollouts: Sequence[str],
) -> MultipleChoiceProblemEvaluation:
    return MultipleChoiceProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            MultipleChoiceRollout(
                sample_index=index,
                text=text,
                verification=verify_answer(problem, text),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def pass_at_k(
    evaluations: Sequence[MultipleChoiceProblemEvaluation],
    ks: Sequence[int],
    *,
    strict: bool = False,
) -> dict[int, float]:
    if not evaluations:
        return {k: 0.0 for k in ks}
    results = {}
    for k in ks:
        solved = 0
        for evaluation in evaluations:
            solved_at = evaluation.strict_solved_at() if strict else evaluation.solved_at()
            if solved_at is not None and solved_at <= k:
                solved += 1
        results[k] = solved / len(evaluations)
    return results


def summarize_evaluations(
    evaluations: Sequence[MultipleChoiceProblemEvaluation],
    *,
    ks: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> dict[str, object]:
    buckets = {"easy": 0, "elicitable": 0, "unreached": 0}
    errors: dict[str, int] = {}
    total_rollouts = 0
    for evaluation in evaluations:
        buckets[evaluation.bucket()] += 1
        for rollout in evaluation.rollouts:
            total_rollouts += 1
            key = rollout.verification.error or "success"
            errors[key] = errors.get(key, 0) + 1
    return {
        "num_problems": len(evaluations),
        "total_rollouts": total_rollouts,
        "pass_at_k": {str(k): value for k, value in pass_at_k(evaluations, ks).items()},
        "strict_format_pass_at_k": {
            str(k): value for k, value in pass_at_k(evaluations, ks, strict=True).items()
        },
        "bucket_counts": buckets,
        "failure_breakdown": errors,
    }


def build_split_registry(split_paths: dict[str, Path]) -> dict[str, object]:
    seen: dict[str, str] = {}
    overlaps = []
    splits = {}
    for split, path in split_paths.items():
        problems = load_problems(path)
        split_ids = [problem.problem_id for problem in problems]
        for problem_id in split_ids:
            if problem_id in seen:
                overlaps.append(
                    {
                        "problem_id": problem_id,
                        "first_split": seen[problem_id],
                        "second_split": split,
                    }
                )
            else:
                seen[problem_id] = split
        splits[split] = {
            "path": str(path),
            "num_problems": len(problems),
            "problem_id_hash": _hash_lines(split_ids),
        }
    return {
        "selected": not overlaps,
        "task": "multiple_choice",
        "verifier": "multiple_choice_final_letter_v1",
        "splits": splits,
        "overlaps": overlaps,
    }


def import_gpqa_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[MultipleChoiceProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        question = str(row.get("Question") or row.get("question"))
        correct = str(row.get("Correct Answer") or row.get("correct_answer"))
        incorrect = [
            str(
                row.get(f"Incorrect Answer {index}")
                or row.get(f"incorrect_answer_{index}")
            )
            for index in range(1, 4)
        ]
        choices = (correct, incorrect[0], incorrect[1], incorrect[2])
        explanation_value = row.get("Explanation") or row.get("explanation")
        problems.append(
            MultipleChoiceProblem(
                problem_id=_problem_id(source, question, correct),
                source=source,
                question=question,
                choices=choices,
                answer="A",
                explanation=None if explanation_value is None else str(explanation_value),
            )
        )
    return problems


def build_public_provenance(
    *,
    dataset: str,
    subset: str | None,
    revision: str,
    source_split: str,
    output: Path,
    limit: int,
    offset: int,
    problems: Sequence[MultipleChoiceProblem],
) -> dict[str, object]:
    problem_ids = [problem.problem_id for problem in problems]
    return {
        "dataset": dataset,
        "subset": subset,
        "revision": revision,
        "source_split": source_split,
        "output": str(output),
        "limit": limit,
        "offset": offset,
        "num_problems": len(problems),
        "problem_id_hash": _hash_lines(problem_ids),
        "source": _public_source(dataset, subset, revision, source_split),
    }


def build_report_input(
    *,
    data_root: Path,
    results_root: Path,
    run_id: str,
    split_registry: Path,
    summary_paths: dict[str, Path],
    scaffold_budget: int,
) -> dict[str, object]:
    return report_artifacts.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id=run_id,
        task="multiple_choice",
        lane="reasoning",
        scaffold={
            "type": "no_tool_sampling",
            "budget": scaffold_budget,
        },
        split_registry=split_registry,
        summary_paths=summary_paths,
        verifier={
            "kind": "exact",
            "name": "multiple_choice_final_letter_v1",
            "output_contract": "A line exactly matching FINAL: <A|B|C|D>.",
            "limitations": [
                "choice order is fixed by importer",
                "no partial credit or semantic judging",
            ],
        },
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_problems(path: Path) -> list[MultipleChoiceProblem]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problems.append(_problem_from_json(row))
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def load_evaluations(path: Path) -> list[MultipleChoiceProblemEvaluation]:
    evaluations = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problem = _problem_from_json(row["problem"])
            rollouts = [str(rollout["text"]) for rollout in row["rollouts"]]
            evaluations.append(evaluate_fixture_rollouts(problem, rollouts))
        except Exception as exc:
            raise ValueError(
                f"invalid evaluation at {path}:{line_number}: {exc}"
            ) from exc
    return evaluations


def _problem_from_json(row: dict[str, object]) -> MultipleChoiceProblem:
    choices = tuple(str(choice) for choice in row["choices"])
    if len(choices) != 4:
        raise ValueError("multiple-choice problems require exactly four choices")
    answer = str(row["answer"]).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid answer letter: {answer}")
    return MultipleChoiceProblem(
        problem_id=str(row["problem_id"]),
        source=str(row.get("source", "multiple_choice")),
        question=str(row["question"]),
        choices=(choices[0], choices[1], choices[2], choices[3]),
        answer=answer,
        explanation=None if row.get("explanation") is None else str(row.get("explanation")),
    )


def _extract_final_value(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        match = FINAL_RE.search(line)
        if match is not None:
            return match.group(1).upper()
    return None


def _public_source(
    dataset: str,
    subset: str | None,
    revision: str,
    source_split: str,
) -> str:
    if subset:
        return f"{dataset}:{subset}:{revision}:{source_split}"
    return f"{dataset}:{revision}:{source_split}"


def _problem_id(source: str, question: str, answer: str) -> str:
    digest = hashlib.sha256(f"{source}|{question}|{answer}".encode()).hexdigest()
    return f"mc-{digest[:16]}"


def _hash_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()
