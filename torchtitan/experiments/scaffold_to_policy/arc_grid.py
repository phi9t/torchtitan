# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""ARC-style exact grid evaluation for scaffold-to-policy reasoning smokes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.scaffold_to_policy import report_artifacts


FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.IGNORECASE)


Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ARCExample:
    input_grid: Grid
    output_grid: Grid

    def to_json(self) -> dict[str, object]:
        return {
            "input": _grid_to_lists(self.input_grid),
            "output": _grid_to_lists(self.output_grid),
        }


@dataclass(frozen=True)
class ARCGridProblem:
    problem_id: str
    source: str
    train_examples: tuple[ARCExample, ...]
    test_input: Grid
    test_output: Grid

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "train": [example.to_json() for example in self.train_examples],
            "test_input": _grid_to_lists(self.test_input),
            "test_output": _grid_to_lists(self.test_output),
        }


@dataclass(frozen=True)
class ARCGridVerification:
    success: bool
    final_grid: Grid | None
    strict_final: bool
    error: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "final_grid": None if self.final_grid is None else _grid_to_lists(self.final_grid),
            "strict_final": self.strict_final,
            "error": self.error,
        }


@dataclass(frozen=True)
class ARCGridRollout:
    sample_index: int
    text: str
    verification: ARCGridVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class ARCGridProblemEvaluation:
    problem: ARCGridProblem
    rollouts: tuple[ARCGridRollout, ...]

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


def prompt_for_problem(problem: ARCGridProblem) -> str:
    lines = [
        "Solve this ARC grid transformation task.",
        "Each grid is a JSON array of rows. Values are integers 0 through 9.",
        "Infer the transformation from the examples, then output the test grid.",
        "",
    ]
    for index, example in enumerate(problem.train_examples, start=1):
        lines.extend(
            [
                f"Example {index} input:",
                _compact_grid_json(example.input_grid),
                f"Example {index} output:",
                _compact_grid_json(example.output_grid),
                "",
            ]
        )
    lines.extend(
        [
            "Test input:",
            _compact_grid_json(problem.test_input),
            "",
            "Return a short reasoning trace.",
            "The last line must be exactly FINAL: <json-grid>.",
        ]
    )
    return "\n".join(lines)


def strict_prompt_for_problem(problem: ARCGridProblem) -> str:
    lines = [
        "Solve this ARC grid transformation task.",
        "Each grid is a JSON array of rows. Values are integers 0 through 9.",
        "Infer the transformation from the examples.",
        "Do not include analysis, markdown, labels, code fences, or extra lines.",
        "Reply with exactly one line: FINAL: <json-grid>",
        "",
    ]
    for index, example in enumerate(problem.train_examples, start=1):
        lines.extend(
            [
                f"Example {index} input:",
                _compact_grid_json(example.input_grid),
                f"Example {index} output:",
                _compact_grid_json(example.output_grid),
                "",
            ]
        )
    lines.extend(
        [
            "Test input:",
            _compact_grid_json(problem.test_input),
            "",
            "One-line final answer only.",
        ]
    )
    return "\n".join(lines)


def compact_prompt_for_problem(problem: ARCGridProblem) -> str:
    lines = ["ARC. Infer output. End FINAL:<json-grid>."]
    for index, example in enumerate(problem.train_examples, start=1):
        lines.append(
            f"E{index} I={_compact_grid_json(example.input_grid)} "
            f"O={_compact_grid_json(example.output_grid)}"
        )
    lines.append(f"T={_compact_grid_json(problem.test_input)}")
    return "\n".join(lines)


def packed_prompt_for_problem(problem: ARCGridProblem) -> str:
    lines = ["ARC. Digits are cells, / separates rows. End FINAL:<json-grid>."]
    for index, example in enumerate(problem.train_examples, start=1):
        lines.append(
            f"E{index} I={_packed_grid(example.input_grid)} "
            f"O={_packed_grid(example.output_grid)}"
        )
    lines.append(f"T={_packed_grid(problem.test_input)}")
    return "\n".join(lines)


def verify_answer(problem: ARCGridProblem, text: str) -> ARCGridVerification:
    final_value = _extract_final_value(text)
    if final_value is None:
        return ARCGridVerification(
            success=False,
            final_grid=None,
            strict_final=False,
            error="missing final grid",
        )
    try:
        final_grid = parse_grid(final_value)
    except ValueError as exc:
        return ARCGridVerification(
            success=False,
            final_grid=None,
            strict_final=True,
            error=str(exc),
        )
    if final_grid != problem.test_output:
        return ARCGridVerification(
            success=False,
            final_grid=final_grid,
            strict_final=True,
            error="final grid does not match answer",
        )
    return ARCGridVerification(
        success=True,
        final_grid=final_grid,
        strict_final=True,
        error=None,
    )


def evaluate_fixture_rollouts(
    problem: ARCGridProblem,
    rollouts: Sequence[str],
) -> ARCGridProblemEvaluation:
    return ARCGridProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            ARCGridRollout(
                sample_index=index,
                text=text,
                verification=verify_answer(problem, text),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def pass_at_k(
    evaluations: Sequence[ARCGridProblemEvaluation],
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
    evaluations: Sequence[ARCGridProblemEvaluation],
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


def build_prompt_preflight(
    *,
    problems: Sequence[ARCGridProblem],
    token_counts: dict[str, int],
    max_model_len: int,
    max_new_tokens: int,
) -> dict[str, object]:
    records = []
    for problem in problems:
        token_count = token_counts[problem.problem_id]
        total_tokens = token_count + max_new_tokens
        records.append(
            {
                "problem_id": problem.problem_id,
                "prompt_tokens": token_count,
                "max_new_tokens": max_new_tokens,
                "total_tokens": total_tokens,
                "max_model_len": max_model_len,
                "selected": total_tokens <= max_model_len,
            }
        )
    num_selected = sum(1 for record in records if record["selected"])
    return {
        "schema_version": 1,
        "kind": "arc_grid_prompt_preflight",
        "num_problems": len(records),
        "num_selected": num_selected,
        "selected": num_selected == len(records),
        "max_prompt_tokens": max(
            (record["prompt_tokens"] for record in records),
            default=0,
        ),
        "max_total_tokens": max(
            (record["total_tokens"] for record in records),
            default=0,
        ),
        "records": records,
    }


def import_arc_tasks(
    task_paths: Sequence[Path],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[ARCGridProblem]:
    problems = []
    for row_index, task_path in enumerate(sorted(task_paths)):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        problems.extend(_problems_from_task_path(task_path, source=source))
    if limit is not None:
        return problems[:limit]
    return problems


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
        "task": "arc_grid",
        "verifier": "arc_grid_exact_json_v1",
        "splits": splits,
        "overlaps": overlaps,
    }


def build_public_provenance(
    *,
    repo_url: str,
    revision: str,
    source_split: str,
    task_dir: Path,
    output: Path,
    limit: int,
    offset: int,
    problems: Sequence[ARCGridProblem],
) -> dict[str, object]:
    problem_ids = [problem.problem_id for problem in problems]
    return {
        "repo_url": repo_url,
        "revision": revision,
        "source_split": source_split,
        "task_dir": str(task_dir),
        "output": str(output),
        "limit": limit,
        "offset": offset,
        "num_problems": len(problems),
        "problem_id_hash": _hash_lines(problem_ids),
        "source": _public_source(repo_url, revision, source_split),
    }


def build_report_input(
    *,
    data_root: Path,
    results_root: Path,
    run_id: str,
    split_registry: Path,
    summary_paths: dict[str, Path],
    scaffold_budget: int,
    preflight_paths: dict[str, Path] | None = None,
) -> dict[str, object]:
    return report_artifacts.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id=run_id,
        task="arc_grid",
        lane="reasoning",
        scaffold={
            "type": "no_tool_sampling",
            "budget": scaffold_budget,
        },
        split_registry=split_registry,
        summary_paths=summary_paths,
        preflight_paths=preflight_paths,
        preflight_check_name="preflight_prompts_fit_context",
        verifier={
            "kind": "exact",
            "name": "arc_grid_exact_json_v1",
            "output_contract": "A line exactly matching FINAL: <json-grid>.",
            "limitations": [
                "only exact grid equality is scored",
                "no partial credit or semantic judging",
                "small smoke only; not an ARC-AGI leaderboard claim",
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


def load_problems(path: Path) -> list[ARCGridProblem]:
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


def load_evaluations(path: Path) -> list[ARCGridProblemEvaluation]:
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


def parse_grid(value: str) -> Grid:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse final grid JSON: {exc.msg}") from exc
    return _parse_grid_object(parsed)


def _problems_from_task_path(task_path: Path, *, source: str) -> list[ARCGridProblem]:
    task = json.loads(task_path.read_text())
    train_examples = tuple(
        ARCExample(
            input_grid=_parse_grid_object(example["input"]),
            output_grid=_parse_grid_object(example["output"]),
        )
        for example in task["train"]
    )
    problems = []
    for test_index, test_example in enumerate(task["test"]):
        problems.append(
            ARCGridProblem(
                problem_id=f"ARC-AGI-2/{task_path.stem}/{test_index}",
                source=source,
                train_examples=train_examples,
                test_input=_parse_grid_object(test_example["input"]),
                test_output=_parse_grid_object(test_example["output"]),
            )
        )
    return problems


def _problem_from_json(row: dict[str, object]) -> ARCGridProblem:
    return ARCGridProblem(
        problem_id=str(row["problem_id"]),
        source=str(row.get("source", "arc_grid")),
        train_examples=tuple(
            ARCExample(
                input_grid=_parse_grid_object(example["input"]),
                output_grid=_parse_grid_object(example["output"]),
            )
            for example in row["train"]
        ),
        test_input=_parse_grid_object(row["test_input"]),
        test_output=_parse_grid_object(row["test_output"]),
    )


def _parse_grid_object(value: object) -> Grid:
    if not isinstance(value, list) or not value:
        raise ValueError("grid must be a non-empty list of rows")
    rows = []
    width = None
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("grid rows must be non-empty lists")
        parsed_row = []
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool):
                raise ValueError("grid cells must be integers")
            if cell < 0 or cell > 9:
                raise ValueError("grid cells must be between 0 and 9")
            parsed_row.append(cell)
        if width is None:
            width = len(parsed_row)
        elif len(parsed_row) != width:
            raise ValueError("grid rows must all have the same width")
        rows.append(tuple(parsed_row))
    return tuple(rows)


def _extract_final_value(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        match = FINAL_RE.search(line)
        if match is not None:
            return match.group(1).strip()
    return None


def _compact_grid_json(grid: Grid) -> str:
    return json.dumps(_grid_to_lists(grid), separators=(",", ":"))


def _packed_grid(grid: Grid) -> str:
    return "/".join("".join(str(cell) for cell in row) for row in grid)


def _grid_to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def _public_source(repo_url: str, revision: str, source_split: str) -> str:
    return f"{repo_url}:{revision}:{source_split}"


def _hash_lines(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()
