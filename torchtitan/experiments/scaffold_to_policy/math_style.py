# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""MATH-style exact-answer normalization for reasoning evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.IGNORECASE)
FRAC_RE = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")
BARE_FRAC_RE = re.compile(r"\\d?frac([+-]?\d+)([+-]?\d+)")
TEXT_RE = re.compile(r"\\(?:text|mathrm)\{([^{}]+)\}")
MATH_DELIMS_RE = re.compile(r"^\s*\$+|\$+\s*$")
NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"
)


@dataclass(frozen=True)
class MathStyleProblem:
    problem_id: str
    source: str
    problem: str
    answer: str
    normalized_answer: str
    solution: str | None = None
    level: str | None = None
    category: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "problem": self.problem,
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
            "solution": self.solution,
            "level": self.level,
            "category": self.category,
        }


@dataclass(frozen=True)
class MathStyleVerification:
    success: bool
    final_value: str | None
    normalized_value: str | None
    strict_final: bool
    error: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "final_value": self.final_value,
            "normalized_value": self.normalized_value,
            "strict_final": self.strict_final,
            "error": self.error,
        }


@dataclass(frozen=True)
class MathStyleRollout:
    sample_index: int
    text: str
    verification: MathStyleVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class MathStyleProblemEvaluation:
    problem: MathStyleProblem
    rollouts: tuple[MathStyleRollout, ...]

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


def normalize_answer(value: str) -> str | None:
    value = value.strip()
    boxed = _extract_last_command_arg(value, r"\boxed")
    if boxed is not None:
        value = boxed
    value = MATH_DELIMS_RE.sub("", value.strip())
    value = TEXT_RE.sub(r"\1", value)
    value = FRAC_RE.sub(r"\1/\2", value)
    value = BARE_FRAC_RE.sub(r"\1/\2", value)
    value = value.replace(r"\$", "")
    value = value.replace(r"\!", "")
    value = value.replace(",", "").strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace(" ", "")
    if not value:
        return None
    try:
        number = Fraction(value)
    except ValueError:
        pass
    else:
        if number.denominator == 1:
            return str(number.numerator)
        return f"{number.numerator}/{number.denominator}"
    if NUMBER_RE.fullmatch(value):
        token = value.replace(",", "").replace(" ", "")
        try:
            number = Fraction(token)
        except ValueError:
            return None
        if number.denominator == 1:
            return str(number.numerator)
        return f"{number.numerator}/{number.denominator}"
    return _normalize_symbolic_text(value)


def prompt_for_problem(problem: MathStyleProblem) -> str:
    return "\n".join(
        [
            "Solve this MATH problem.",
            problem.problem,
            "Return a short calculation trace.",
            "The last line must be exactly FINAL: <answer>.",
        ]
    )


def verify_answer(problem: MathStyleProblem, text: str) -> MathStyleVerification:
    final_value = _extract_final_value(text)
    if final_value is None:
        return MathStyleVerification(
            success=False,
            final_value=None,
            normalized_value=None,
            strict_final=False,
            error="missing final answer",
        )
    normalized = normalize_answer(final_value)
    if normalized is None:
        return MathStyleVerification(
            success=False,
            final_value=final_value,
            normalized_value=None,
            strict_final=True,
            error=f"could not normalize final answer {final_value!r}",
        )
    if normalized != problem.normalized_answer:
        return MathStyleVerification(
            success=False,
            final_value=final_value,
            normalized_value=normalized,
            strict_final=True,
            error=(
                f"final value {normalized} does not match answer "
                f"{problem.normalized_answer}"
            ),
        )
    return MathStyleVerification(
        success=True,
        final_value=final_value,
        normalized_value=normalized,
        strict_final=True,
        error=None,
    )


def evaluate_fixture_rollouts(
    problem: MathStyleProblem,
    rollouts: Sequence[str],
) -> MathStyleProblemEvaluation:
    return MathStyleProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            MathStyleRollout(
                sample_index=index,
                text=text,
                verification=verify_answer(problem, text),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def load_evaluations(path: Path) -> list[MathStyleProblemEvaluation]:
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


def pass_at_k(
    evaluations: Sequence[MathStyleProblemEvaluation],
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
    evaluations: Sequence[MathStyleProblemEvaluation],
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
        "task": "math_style",
        "verifier": "math_style_normalized_final_v1",
        "splits": splits,
        "overlaps": overlaps,
    }


def import_public_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[MathStyleProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        problem = str(row["problem"])
        solution = str(row["solution"])
        answer = extract_answer_from_solution(solution)
        normalized_answer = normalize_answer(answer)
        if normalized_answer is None:
            raise ValueError(f"could not normalize public answer {answer!r}")
        category = None if row.get("type") is None else str(row.get("type"))
        level = None if row.get("level") is None else str(row.get("level"))
        problems.append(
            MathStyleProblem(
                problem_id=_problem_id(source, problem, answer),
                source=source,
                problem=problem,
                answer=answer,
                normalized_answer=normalized_answer,
                solution=solution,
                level=level,
                category=category,
            )
        )
    return problems


def import_aime_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[MathStyleProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        problem = str(row.get("problem") or row.get("Problem"))
        answer = str(row.get("answer") or row.get("Answer")).zfill(3)
        normalized_answer = normalize_answer(answer)
        if normalized_answer is None:
            raise ValueError(f"could not normalize AIME answer {answer!r}")
        solution_value = row.get("solution") or row.get("Solution")
        problem_id = str(row.get("id") or row.get("ID") or _problem_id(source, problem, answer))
        if not problem_id.startswith("AIME/"):
            problem_id = f"AIME/{problem_id}"
        problems.append(
            MathStyleProblem(
                problem_id=problem_id,
                source=source,
                problem=problem,
                answer=answer,
                normalized_answer=normalized_answer,
                solution=None if solution_value is None else str(solution_value),
                level="AIME",
                category="contest_math",
            )
        )
    return problems


def extract_answer_from_solution(solution: str) -> str:
    boxed = _extract_last_command_arg(solution, r"\boxed")
    if boxed is None:
        raise ValueError("MATH solution is missing boxed answer")
    return boxed.strip()


def build_public_provenance(
    *,
    dataset: str,
    subset: str,
    revision: str,
    source_split: str,
    output: Path,
    limit: int,
    offset: int,
    problems: Sequence[MathStyleProblem],
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
    summaries = {
        split: json.loads(path.read_text()) for split, path in summary_paths.items()
    }
    registry = json.loads(split_registry.read_text())
    checks = {
        "split_registry_selected": bool(registry.get("selected", False)),
        "summaries_present": all(path.is_file() for path in summary_paths.values()),
        "summary_split_counts_match": all(
            summaries[split]["num_problems"]
            == registry["splits"][split]["num_problems"]
            for split in summary_paths
        ),
    }
    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "task": "math_style",
            "lane": "reasoning",
            "scaffold": {
                "type": "fixture_or_no_tool_sampling",
                "budget": scaffold_budget,
            },
        },
        "artifacts": {
            "data_root": str(data_root),
            "results_root": str(results_root),
            "split_registry": str(split_registry),
            "summaries": {split: str(path) for split, path in summary_paths.items()},
        },
        "verifier": {
            "kind": "exact",
            "name": "math_style_normalized_final_v1",
            "output_contract": "A line exactly matching FINAL: <answer>.",
            "normalization": [
                "FINAL and boxed answer markers",
                "integers, decimals, simple fractions, and LaTeX fractions",
                "literal symbolic strings after light LaTeX cleanup",
            ],
            "limitations": [
                "no symbolic algebra equivalence",
                "no interval, set, matrix, or multi-answer semantic matching",
            ],
        },
        "checks": checks,
        "metrics": {"splits": summaries},
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_problems(path: Path) -> list[MathStyleProblem]:
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


def _problem_from_json(row: dict[str, object]) -> MathStyleProblem:
    answer = str(row["answer"])
    normalized_answer = normalize_answer(answer)
    if normalized_answer is None:
        raise ValueError(f"could not normalize answer {answer!r}")
    return MathStyleProblem(
        problem_id=str(row["problem_id"]),
        source=str(row.get("source", "math_style")),
        problem=str(row["problem"]),
        answer=answer,
        normalized_answer=normalized_answer,
        solution=None if row.get("solution") is None else str(row.get("solution")),
        level=None if row.get("level") is None else str(row.get("level")),
        category=None if row.get("category") is None else str(row.get("category")),
    )


def _extract_final_value(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        match = FINAL_RE.search(line)
        if match is not None:
            return match.group(1).strip()
        boxed = _extract_last_command_arg(line, r"\boxed")
        if boxed is not None:
            return boxed.strip()
    return None


def _extract_last_command_arg(text: str, command: str) -> str | None:
    start = 0
    last = None
    while True:
        command_index = text.find(command, start)
        if command_index < 0:
            return last
        brace_index = command_index + len(command)
        while brace_index < len(text) and text[brace_index].isspace():
            brace_index += 1
        if brace_index >= len(text) or text[brace_index] != "{":
            start = command_index + len(command)
            continue
        depth = 0
        for index in range(brace_index, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    last = text[brace_index + 1 : index]
                    start = index + 1
                    break
        else:
            return last


def _normalize_symbolic_text(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    value = value.replace("\\cdot", "*")
    value = value.replace("\\times", "*")
    value = value.replace("\\sqrt", "sqrt")
    value = value.replace("\\in", "in")
    value = value.replace("{", "").replace("}", "")
    value = value.replace("[", "").replace("]", "")
    return value


def _public_source(
    dataset: str,
    subset: str,
    revision: str,
    source_split: str,
) -> str:
    return f"{dataset}:{subset}:{revision}:{source_split}"


def _problem_id(source: str, problem: str, answer: str) -> str:
    digest = hashlib.sha256(f"{source}|{problem}|{answer}".encode()).hexdigest()
    return f"math-{digest[:16]}"


def _hash_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()
