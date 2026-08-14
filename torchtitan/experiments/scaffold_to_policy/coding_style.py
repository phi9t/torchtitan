# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Executable-code evaluation helpers for scaffold-to-policy coding smokes."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.scaffold_to_policy import report_artifacts


FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
ASSERT_CALL_ARGS_RE = re.compile(
    r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:==|!=|is|in|<=|>=|<|>)",
    re.DOTALL,
)
MBPP_WRAPPER_CALLS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "float",
        "int",
        "isclose",
        "len",
        "list",
        "max",
        "min",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
)


@dataclass(frozen=True)
class CodingStyleProblem:
    problem_id: str
    source: str
    prompt: str
    test: str
    entry_point: str
    canonical_solution: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "prompt": self.prompt,
            "test": self.test,
            "entry_point": self.entry_point,
            "canonical_solution": self.canonical_solution,
        }


@dataclass(frozen=True)
class CodingStyleVerification:
    success: bool
    error: str | None
    extracted_code: str
    returncode: int | None
    stdout: str
    stderr: str

    def to_json(self) -> dict[str, object]:
        return {
            "success": self.success,
            "error": self.error,
            "extracted_code": self.extracted_code,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class CodingStyleRollout:
    sample_index: int
    text: str
    verification: CodingStyleVerification

    def to_json(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "text": self.text,
            **self.verification.to_json(),
        }


@dataclass(frozen=True)
class CodingStyleProblemEvaluation:
    problem: CodingStyleProblem
    rollouts: tuple[CodingStyleRollout, ...]

    def solved_at(self) -> int | None:
        for index, rollout in enumerate(self.rollouts, start=1):
            if rollout.verification.success:
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
            "bucket": self.bucket(),
            "rollouts": [rollout.to_json() for rollout in self.rollouts],
        }


def prompt_for_problem(problem: CodingStyleProblem) -> str:
    return "\n".join(
        [
            "Complete this Python function.",
            "Return only Python code for the function implementation.",
            "Do not include Markdown fences or explanatory prose.",
            "",
            problem.prompt.rstrip(),
        ]
    )


def verify_solution(
    problem: CodingStyleProblem,
    text: str,
    *,
    timeout_seconds: float = 5.0,
) -> CodingStyleVerification:
    extracted = extract_candidate_code(problem, text)
    program = "\n".join(
        [
            extracted.rstrip(),
            "",
            problem.test.rstrip(),
            "",
            f"check({problem.entry_point})",
            "",
        ]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="coding-style-") as tmpdir:
            completed = subprocess.run(
                [sys.executable, "-I", "-"],
                input=program,
                text=True,
                capture_output=True,
                cwd=tmpdir,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        return CodingStyleVerification(
            success=False,
            error="timeout",
            extracted_code=extracted,
            returncode=None,
            stdout=_subprocess_text(exc.stdout),
            stderr=_subprocess_text(exc.stderr),
        )
    if completed.returncode == 0:
        return CodingStyleVerification(
            success=True,
            error=None,
            extracted_code=extracted,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return CodingStyleVerification(
        success=False,
        error=_classify_failure(completed.stderr),
        extracted_code=extracted,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def extract_candidate_code(problem: CodingStyleProblem, text: str) -> str:
    candidate = text.rstrip()
    fences = FENCED_CODE_RE.findall(candidate.strip())
    if fences:
        candidate = fences[-1].strip()
    if not candidate[:1].isspace():
        candidate = _strip_leading_chat_text(candidate)
    if f"def {problem.entry_point}" in candidate:
        return _prompt_preamble(problem) + candidate
    if candidate[:1].isspace():
        return problem.prompt + candidate
    return problem.prompt + candidate.lstrip()


def evaluate_fixture_rollouts(
    problem: CodingStyleProblem,
    rollouts: Sequence[str],
    *,
    timeout_seconds: float = 5.0,
) -> CodingStyleProblemEvaluation:
    return CodingStyleProblemEvaluation(
        problem=problem,
        rollouts=tuple(
            CodingStyleRollout(
                sample_index=index,
                text=text,
                verification=verify_solution(
                    problem,
                    text,
                    timeout_seconds=timeout_seconds,
                ),
            )
            for index, text in enumerate(rollouts)
        ),
    )


def pass_at_k(
    evaluations: Sequence[CodingStyleProblemEvaluation],
    ks: Sequence[int],
) -> dict[int, float]:
    if not evaluations:
        return {k: 0.0 for k in ks}
    results = {}
    for k in ks:
        solved = 0
        for evaluation in evaluations:
            solved_at = evaluation.solved_at()
            if solved_at is not None and solved_at <= k:
                solved += 1
        results[k] = solved / len(evaluations)
    return results


def summarize_evaluations(
    evaluations: Sequence[CodingStyleProblemEvaluation],
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
        "bucket_counts": buckets,
        "failure_breakdown": errors,
    }


def preflight_canonical_solutions(
    problems: Sequence[CodingStyleProblem],
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    records = []
    for problem in problems:
        if problem.canonical_solution is None:
            records.append(
                {
                    "problem_id": problem.problem_id,
                    "has_canonical_solution": False,
                    "success": False,
                    "error": "missing canonical solution",
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                }
            )
            continue
        verification = verify_solution(
            problem,
            problem.canonical_solution,
            timeout_seconds=timeout_seconds,
        )
        records.append(
            {
                "problem_id": problem.problem_id,
                "has_canonical_solution": True,
                "success": verification.success,
                "error": verification.error,
                "returncode": verification.returncode,
                "stdout": verification.stdout,
                "stderr": verification.stderr,
            }
        )
    failure_breakdown: dict[str, int] = {}
    for record in records:
        key = str(record["error"] or "success")
        failure_breakdown[key] = failure_breakdown.get(key, 0) + 1
    num_passed = sum(1 for record in records if record["success"])
    return {
        "schema_version": 1,
        "kind": "coding_style_canonical_preflight",
        "num_problems": len(records),
        "num_passed": num_passed,
        "selected": len(records) == num_passed,
        "failure_breakdown": failure_breakdown,
        "records": records,
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
        "task": "coding_style",
        "verifier": "python_executable_tests_v1",
        "splits": splits,
        "overlaps": overlaps,
    }


def import_public_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[CodingStyleProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        prompt = str(row["prompt"])
        test = str(row["test"])
        entry_point = str(row["entry_point"])
        task_id = str(row.get("task_id") or _problem_id(source, prompt, entry_point))
        problems.append(
            CodingStyleProblem(
                problem_id=task_id,
                source=source,
                prompt=prompt,
                test=test,
                entry_point=entry_point,
                canonical_solution=(
                    None
                    if row.get("canonical_solution") is None
                    else str(row.get("canonical_solution"))
                ),
            )
        )
    return problems


def import_mbpp_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[CodingStyleProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        prompt = str(row["prompt"]).strip()
        test_imports = [str(value) for value in row.get("test_imports", [])]
        test_list = [str(value) for value in row["test_list"]]
        if not test_list:
            raise ValueError("MBPP row has no tests")
        canonical = None if row.get("code") is None else str(row.get("code"))
        entry_point = (
            _entry_point_from_code(canonical, test_list)
            if canonical is not None
            else _entry_point_from_asserts(test_list)
        )
        test = _mbpp_check_source(test_imports, test_list, entry_point)
        task_id = f"MBPP/{row['task_id']}"
        signature = _signature_from_first_assert(test_list[0], entry_point)
        problems.append(
            CodingStyleProblem(
                problem_id=task_id,
                source=source,
                prompt=_mbpp_prompt(prompt, entry_point, signature),
                test=test,
                entry_point=entry_point,
                canonical_solution=canonical,
            )
        )
    return problems


def import_bigcodebench_rows(
    rows: Iterable[dict[str, object]],
    *,
    source: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[CodingStyleProblem]:
    problems = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if limit is not None and len(problems) >= limit:
            break
        task_id = str(row["task_id"])
        entry_point = str(row["entry_point"])
        test = _bigcodebench_check_source(str(row["test"]), entry_point)
        problems.append(
            CodingStyleProblem(
                problem_id=task_id,
                source=source,
                prompt=str(row["code_prompt"]),
                test=test,
                entry_point=entry_point,
                canonical_solution=(
                    None
                    if row.get("canonical_solution") is None
                    else str(row.get("canonical_solution"))
                ),
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
    problems: Sequence[CodingStyleProblem],
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
    preflight_paths: dict[str, Path] | None = None,
    runtime_path: Path | None = None,
) -> dict[str, object]:
    return report_artifacts.build_report_input(
        data_root=data_root,
        results_root=results_root,
        run_id=run_id,
        task="coding_style",
        lane="coding",
        scaffold={
            "type": "fixture_or_no_tool_sampling",
            "budget": scaffold_budget,
        },
        split_registry=split_registry,
        summary_paths=summary_paths,
        preflight_paths=preflight_paths,
        preflight_check_name="preflight_canonical_solutions_pass",
        runtime_path=runtime_path,
        verifier={
            "kind": "executable",
            "name": "python_executable_tests_v1",
            "output_contract": "Python function code that satisfies public tests.",
            "execution": [
                "candidate code is extracted from text or Markdown fences",
                "tests run in a separate isolated Python subprocess",
                "each rollout has a wall-clock timeout",
            ],
            "limitations": [
                "no third-party package installation inside candidate tests",
                "not a secure sandbox for malicious code",
                "HumanEval smoke only; not a LiveCodeBench or Terminal-Bench claim",
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


def load_problems(path: Path) -> list[CodingStyleProblem]:
    problems = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            problems.append(
                CodingStyleProblem(
                    problem_id=str(row["problem_id"]),
                    source=str(row["source"]),
                    prompt=str(row["prompt"]),
                    test=str(row["test"]),
                    entry_point=str(row["entry_point"]),
                    canonical_solution=(
                        None
                        if row.get("canonical_solution") is None
                        else str(row.get("canonical_solution"))
                    ),
                )
            )
        except Exception as exc:
            raise ValueError(f"invalid problem at {path}:{line_number}: {exc}") from exc
    return problems


def load_evaluations(path: Path) -> list[CodingStyleProblemEvaluation]:
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


def _problem_from_json(row: dict[str, object]) -> CodingStyleProblem:
    return CodingStyleProblem(
        problem_id=str(row["problem_id"]),
        source=str(row["source"]),
        prompt=str(row["prompt"]),
        test=str(row["test"]),
        entry_point=str(row["entry_point"]),
        canonical_solution=(
            None
            if row.get("canonical_solution") is None
            else str(row.get("canonical_solution"))
        ),
    )


def _strip_leading_chat_text(candidate: str) -> str:
    lines = candidate.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "from ", "import ", "@")):
            return "\n".join(lines[index:]).strip()
    return candidate


def _prompt_preamble(problem: CodingStyleProblem) -> str:
    marker = f"def {problem.entry_point}"
    index = problem.prompt.find(marker)
    if index == -1:
        return ""
    return problem.prompt[:index]


def _classify_failure(stderr: str) -> str:
    if "AssertionError" in stderr:
        return "assertion failure"
    if "SyntaxError" in stderr:
        return "syntax error"
    if "NameError" in stderr:
        return "name error"
    if "IndentationError" in stderr:
        return "indentation error"
    if "TypeError" in stderr:
        return "type error"
    if "RecursionError" in stderr:
        return "recursion error"
    if stderr.strip():
        return stderr.strip().splitlines()[-1][:160]
    return "nonzero exit"


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _entry_point_from_asserts(test_list: Sequence[str]) -> str:
    names = []
    for test in test_list:
        test_names = _entry_point_calls_from_assert(test)
        if not test_names:
            raise ValueError(f"could not infer MBPP entry point from test: {test}")
        names.extend(test_names)
    if len(set(names)) != 1:
        raise ValueError(f"MBPP tests reference multiple entry points: {sorted(set(names))}")
    return names[0]


def _entry_point_from_code(code: str, test_list: Sequence[str]) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError("could not parse MBPP canonical code") from exc

    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    tested_names = {name for test in test_list for name in _call_names_from_assert(test)}
    matching_names = sorted(function_names & tested_names)
    if len(matching_names) != 1:
        raise ValueError(
            "expected one tested MBPP canonical function, got "
            f"{matching_names} from functions {sorted(function_names)} "
            f"and test calls {sorted(tested_names)}"
        )
    return matching_names[0]


def _entry_point_calls_from_assert(test: str) -> list[str]:
    statement = _single_assert(test)
    result_expr = _assert_result_expression(statement.test)
    nodes = [result_expr] if result_expr is not None else [statement.test]
    names = []
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue
            name = _called_function_name(node.func)
            if name is None:
                continue
            if _is_wrapper_call(node):
                continue
            names.append(name)
    return names


def _call_names_from_assert(test: str) -> list[str]:
    statement = _single_assert(test)
    names = []
    for node in ast.walk(statement.test):
        if not isinstance(node, ast.Call):
            continue
        name = _called_function_name(node.func)
        if name is not None:
            names.append(name)
    return names


def _single_assert(test: str) -> ast.Assert:
    try:
        tree = ast.parse(test)
    except SyntaxError as exc:
        raise ValueError(f"could not parse MBPP test: {test}") from exc

    statements = [node for node in tree.body if isinstance(node, ast.Assert)]
    if len(statements) != 1:
        raise ValueError(f"MBPP test must contain one assert: {test}")
    return statements[0]


def _assert_result_expression(node: ast.expr) -> ast.expr | None:
    if isinstance(node, ast.Compare) and node.left is not None:
        return node.left
    return node


def _is_wrapper_call(node: ast.Call) -> bool:
    name = _called_function_name(node.func)
    if name not in MBPP_WRAPPER_CALLS:
        return False
    return any(
        isinstance(descendant, ast.Call)
        for arg in node.args
        for descendant in ast.walk(arg)
    )


def _called_function_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _signature_from_first_assert(test: str, entry_point: str) -> str:
    match = ASSERT_CALL_ARGS_RE.search(test)
    if match is None or match.group(1) != entry_point:
        return "*args, **kwargs"
    args = match.group(2).strip()
    if not args:
        return ""
    return ", ".join(f"arg{index}" for index, _ in enumerate(_split_call_args(args), start=1))


def _split_call_args(args: str) -> list[str]:
    parts = []
    depth = 0
    in_string: str | None = None
    escaped = False
    start = 0
    for index, char in enumerate(args):
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(args[start:index].strip())
            start = index + 1
    parts.append(args[start:].strip())
    return [part for part in parts if part]


def _mbpp_prompt(prompt: str, entry_point: str, signature: str) -> str:
    return "\n".join(
        [
            f"# {prompt}",
            f"def {entry_point}({signature}):",
            "    ",
        ]
    )


def _mbpp_check_source(
    test_imports: Sequence[str],
    test_list: Sequence[str],
    entry_point: str,
) -> str:
    rewritten_tests = [
        _rewrite_mbpp_assert_call(test, entry_point) for test in test_list
    ]
    body = "\n    ".join(rewritten_tests)
    imports = "\n".join(test_imports)
    return "\n".join(
        line
        for line in [
            imports,
            f"def check(candidate):\n    {body}",
        ]
        if line
    )


def _rewrite_mbpp_assert_call(test: str, entry_point: str) -> str:
    try:
        tree = ast.parse(test)
    except SyntaxError as exc:
        raise ValueError(f"could not parse MBPP test: {test}") from exc

    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == entry_point
    ]
    if len(matches) != 1:
        raise ValueError(
            f"could not rewrite MBPP test for entry point {entry_point!r}: {test}"
        )

    call = matches[0]
    return test[: call.func.col_offset] + "candidate" + test[call.func.end_col_offset :]


def _bigcodebench_check_source(test: str, entry_point: str) -> str:
    return "\n".join(
        [
            test.rstrip(),
            "",
            "def check(candidate):",
            f"    globals()[{entry_point!r}] = candidate",
            "    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCases)",
            "    result = unittest.TextTestRunner(verbosity=0).run(suite)",
            "    assert result.wasSuccessful()",
        ]
    )


def _problem_id(source: str, prompt: str, entry_point: str) -> str:
    digest = hashlib.sha256(
        "\n".join([source, prompt, entry_point]).encode()
    ).hexdigest()[:16]
    return f"coding-{digest}"


def _hash_lines(lines: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _public_source(
    dataset: str,
    subset: str | None,
    revision: str,
    source_split: str,
) -> str:
    if subset:
        return f"{dataset}:{subset}:{revision}:{source_split}"
    return f"{dataset}:{revision}:{source_split}"
