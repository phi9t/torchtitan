# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Countdown problem generation and exact solution verification."""

from __future__ import annotations

import itertools
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction


_OP_RE = re.compile(
    r"^\s*(?P<a>-?\d+)\s*(?P<op>[+\-*/xX×÷])\s*(?P<b>-?\d+)\s*=\s*(?P<c>-?\d+)\s*$"
)
_FINAL_RE = re.compile(r"^\s*FINAL\s*:\s*(?P<value>-?\d+)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class CountdownProblem:
    numbers: tuple[int, ...]
    target: int
    min_solution_depth: int | None = None
    require_all_numbers: bool = False
    solution: tuple[str, ...] = ()

    def prompt(self) -> str:
        numbers = ", ".join(str(number) for number in self.numbers)
        return (
            f"Numbers: {numbers}\n"
            f"Target: {self.target}\n\n"
            "Return a sequence of operations. Each operation must consume two\n"
            "currently available numbers and produce one new positive integer.\n"
            "Use each input number at most once. Division must be exact.\n"
            "Format:\n"
            "a + b = c\n"
            "...\n"
            f"FINAL: {self.target}\n"
        )

    def to_json(self) -> dict[str, object]:
        return {
            "numbers": list(self.numbers),
            "target": self.target,
            "min_solution_depth": self.min_solution_depth,
            "require_all_numbers": self.require_all_numbers,
            "solution": list(self.solution),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "CountdownProblem":
        numbers = data["numbers"]
        if not isinstance(numbers, list):
            raise ValueError("numbers must be a list")
        solution = data.get("solution", [])
        if not isinstance(solution, list):
            raise ValueError("solution must be a list")
        return cls(
            numbers=tuple(int(number) for number in numbers),
            target=int(data["target"]),
            min_solution_depth=(
                None
                if data.get("min_solution_depth") is None
                else int(data["min_solution_depth"])
            ),
            require_all_numbers=bool(data.get("require_all_numbers", False)),
            solution=tuple(str(step) for step in solution),
        )


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    final_value: int | None
    steps_consumed: int
    available_numbers: tuple[int, ...]
    operations: tuple["ParsedOperation", ...] = ()
    states: tuple[tuple[int, ...], ...] = ()
    valid_prefix_length: int = 0
    first_invalid: dict[str, object] | None = None
    first_unreachable: dict[str, object] | None = None
    distance_to_target: int | None = None
    solution_depth: int | None = None
    recoverable: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ParsedOperation:
    raw: str
    left: int
    op: str
    right: int
    result: int
    line_number: int

    def to_json(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "left": self.left,
            "op": self.op,
            "right": self.right,
            "result": self.result,
            "line_number": self.line_number,
        }


@dataclass(frozen=True)
class SolutionCandidate:
    value: Fraction
    depth: int
    steps: tuple[str, ...]


@dataclass(frozen=True)
class ProblemFilters:
    require_all_numbers: bool = False
    division_required: bool = False
    target_min: int = 100
    target_max: int = 999
    number_min: int = 1
    number_max: int = 100
    min_solution_depth: int = 0
    min_distinct_solutions: int = 1


def _format_number(value: Fraction) -> str:
    if value.denominator != 1:
        raise ValueError(f"non-integer value {value}")
    return str(value.numerator)


def _result_for(a: Fraction, b: Fraction, op: str) -> Fraction | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0 or a % b != 0:
            return None
        return a / b
    raise ValueError(f"unsupported operator {op}")


def _normalize_op(op: str) -> str:
    if op in {"x", "X", "×"}:
        return "*"
    if op == "÷":
        return "/"
    return op


def _consume_once(values: Counter[int], number: int) -> bool:
    if values[number] <= 0:
        return False
    values[number] -= 1
    if values[number] == 0:
        del values[number]
    return True


def _state_tuple(values: Counter[int]) -> tuple[int, ...]:
    return tuple(sorted(values.elements()))


def _distance_to_target(available: Counter[int], target: int) -> int | None:
    values = list(available.elements())
    if not values:
        return None
    return min(abs(value - target) for value in values)


def _target_reachable_from_state(available: Counter[int], target: int) -> bool:
    values = _state_tuple(available)
    if target in values:
        return True
    if len(values) < 2:
        return False
    return target in _reachable_targets(values, require_all_numbers=False)


def parse_operation_line(line: str, line_number: int) -> ParsedOperation | None:
    op_match = _OP_RE.match(line)
    if op_match is None:
        return None
    return ParsedOperation(
        raw=line,
        left=int(op_match.group("a")),
        op=_normalize_op(op_match.group("op")),
        right=int(op_match.group("b")),
        result=int(op_match.group("c")),
        line_number=line_number,
    )


def extract_solution_lines(text: str) -> str:
    lines: list[str] = []
    seen_operation = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        operation = parse_operation_line(line, len(lines) + 1)
        if operation is not None:
            normalized = f"{operation.left} {operation.op} {operation.right} = {operation.result}"
            lines.append(normalized)
            seen_operation = True
            continue
        final_match = _FINAL_RE.match(line)
        if final_match is not None and seen_operation:
            lines.append(f"FINAL: {int(final_match.group('value'))}")
            break
    return "\n".join(lines)


def _verification_result(
    *,
    problem: CountdownProblem,
    success: bool,
    final_value: int | None,
    steps_consumed: int,
    available: Counter[int],
    operations: list[ParsedOperation],
    states: list[tuple[int, ...]],
    first_invalid: dict[str, object] | None,
    first_unreachable: dict[str, object] | None,
    compute_recoverability: bool,
    error: str | None = None,
) -> VerificationResult:
    distance_to_target = (
        abs(final_value - problem.target)
        if final_value is not None
        else _distance_to_target(available, problem.target)
    )
    return VerificationResult(
        success=success,
        final_value=final_value,
        steps_consumed=steps_consumed,
        available_numbers=_state_tuple(available),
        operations=tuple(operations),
        states=tuple(states),
        valid_prefix_length=steps_consumed,
        first_invalid=first_invalid,
        first_unreachable=first_unreachable,
        distance_to_target=distance_to_target,
        solution_depth=steps_consumed if success else None,
        recoverable=(
            _target_reachable_from_state(available, problem.target)
            if compute_recoverability
            else False
        ),
        error=error,
    )


def verify_solution(
    problem: CountdownProblem,
    text: str,
    *,
    compute_recoverability: bool = True,
) -> VerificationResult:
    text = extract_solution_lines(text)
    available = Counter(problem.numbers)
    final_value: int | None = None
    steps_consumed = 0
    saw_final = False
    operations: list[ParsedOperation] = []
    states: list[tuple[int, ...]] = [_state_tuple(available)]
    first_unreachable: dict[str, object] | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        final_match = _FINAL_RE.match(line)
        if final_match is not None:
            final_value = int(final_match.group("value"))
            saw_final = True
            break
        operation = parse_operation_line(line, line_number)
        if operation is None:
            error = f"invalid line: {line}"
            return _verification_result(
                problem=problem,
                success=False,
                final_value=final_value,
                steps_consumed=steps_consumed,
                available=available,
                operations=operations,
                states=states,
                first_invalid={
                    "line_number": line_number,
                    "line": line,
                    "reason": "parse_error",
                },
                first_unreachable=first_unreachable,
                compute_recoverability=compute_recoverability,
                error=error,
            )

        next_available = available.copy()
        if not _consume_once(next_available, operation.left) or not _consume_once(
            next_available, operation.right
        ):
            error = f"operation consumes unavailable numbers: {line}"
            return _verification_result(
                problem=problem,
                success=False,
                final_value=final_value,
                steps_consumed=steps_consumed,
                available=available,
                operations=operations,
                states=states,
                first_invalid={
                    "line_number": line_number,
                    "line": line,
                    "reason": "unavailable_operand",
                    "state": _state_tuple(available),
                },
                first_unreachable=first_unreachable,
                compute_recoverability=compute_recoverability,
                error=error,
            )

        result = _result_for(
            Fraction(operation.left), Fraction(operation.right), operation.op
        )
        if result is None:
            error = f"division is not exact: {line}"
            return _verification_result(
                problem=problem,
                success=False,
                final_value=final_value,
                steps_consumed=steps_consumed,
                available=available,
                operations=operations,
                states=states,
                first_invalid={
                    "line_number": line_number,
                    "line": line,
                    "reason": "non_exact_division",
                    "state": _state_tuple(available),
                },
                first_unreachable=first_unreachable,
                compute_recoverability=compute_recoverability,
                error=error,
            )
        if result.denominator != 1:
            error = f"operation produced non-integer result: {line}"
            return _verification_result(
                problem=problem,
                success=False,
                final_value=final_value,
                steps_consumed=steps_consumed,
                available=available,
                operations=operations,
                states=states,
                first_invalid={
                    "line_number": line_number,
                    "line": line,
                    "reason": "non_integer_result",
                    "state": _state_tuple(available),
                },
                first_unreachable=first_unreachable,
                compute_recoverability=compute_recoverability,
                error=error,
            )
        computed = result.numerator
        if computed <= 0:
            error = f"operation produced non-positive result: {line}"
            return _verification_result(
                problem=problem,
                success=False,
                final_value=final_value,
                steps_consumed=steps_consumed,
                available=available,
                operations=operations,
                states=states,
                first_invalid={
                    "line_number": line_number,
                    "line": line,
                    "reason": "non_positive_result",
                    "state": _state_tuple(available),
                },
                first_unreachable=first_unreachable,
                compute_recoverability=compute_recoverability,
                error=error,
            )
        if computed != operation.result:
            error = (
                f"declared result {operation.result} does not match {computed}: {line}"
            )
            return _verification_result(
                problem=problem,
                success=False,
                final_value=final_value,
                steps_consumed=steps_consumed,
                available=available,
                operations=operations,
                states=states,
                first_invalid={
                    "line_number": line_number,
                    "line": line,
                    "reason": "wrong_declared_result",
                    "computed": computed,
                    "state": _state_tuple(available),
                },
                first_unreachable=first_unreachable,
                compute_recoverability=compute_recoverability,
                error=error,
            )

        next_available[computed] += 1
        available = next_available
        operations.append(operation)
        steps_consumed += 1
        states.append(_state_tuple(available))
        if (
            compute_recoverability
            and first_unreachable is None
            and not _target_reachable_from_state(available, problem.target)
        ):
            first_unreachable = {
                "line_number": line_number,
                "line": line,
                "state": _state_tuple(available),
            }

    if not saw_final and available[problem.target] > 0:
        if not problem.require_all_numbers or len(available) == 1:
            return _verification_result(
                problem=problem,
                success=True,
                final_value=problem.target,
                steps_consumed=steps_consumed,
                available=available,
                operations=operations,
                states=states,
                first_invalid=None,
                first_unreachable=first_unreachable,
                compute_recoverability=compute_recoverability,
            )
    if not saw_final:
        return _verification_result(
            problem=problem,
            success=False,
            final_value=final_value,
            steps_consumed=steps_consumed,
            available=available,
            operations=operations,
            states=states,
            first_invalid={"reason": "missing_final"},
            first_unreachable=first_unreachable,
            compute_recoverability=compute_recoverability,
            error="missing FINAL line",
        )
    if final_value != problem.target:
        return _verification_result(
            problem=problem,
            success=False,
            final_value=final_value,
            steps_consumed=steps_consumed,
            available=available,
            operations=operations,
            states=states,
            first_invalid={
                "reason": "wrong_final",
                "final_value": final_value,
                "target": problem.target,
            },
            first_unreachable=first_unreachable,
            compute_recoverability=compute_recoverability,
            error=f"final value {final_value} does not match target {problem.target}",
        )
    if available[problem.target] <= 0:
        return _verification_result(
            problem=problem,
            success=False,
            final_value=final_value,
            steps_consumed=steps_consumed,
            available=available,
            operations=operations,
            states=states,
            first_invalid={"reason": "target_unavailable"},
            first_unreachable=first_unreachable,
            compute_recoverability=compute_recoverability,
            error="target is not currently available",
        )
    if problem.require_all_numbers and len(available) != 1:
        return _verification_result(
            problem=problem,
            success=False,
            final_value=final_value,
            steps_consumed=steps_consumed,
            available=available,
            operations=operations,
            states=states,
            first_invalid={
                "reason": "unused_numbers",
                "state": _state_tuple(available),
            },
            first_unreachable=first_unreachable,
            compute_recoverability=compute_recoverability,
            error="solution did not consume all input numbers",
        )
    return _verification_result(
        problem=problem,
        success=True,
        final_value=final_value,
        steps_consumed=steps_consumed,
        available=available,
        operations=operations,
        states=states,
        first_invalid=None,
        first_unreachable=first_unreachable,
        compute_recoverability=compute_recoverability,
    )


def _candidate_results(
    left: SolutionCandidate, right: SolutionCandidate
) -> Iterable[SolutionCandidate]:
    a = left.value
    b = right.value
    depth = left.depth + right.depth + 1
    prefix = left.steps + right.steps
    for op in ("+", "-", "*", "/"):
        for x, y, x_steps, y_steps in (
            (a, b, left.steps, right.steps),
            (b, a, right.steps, left.steps),
        ):
            if op in {"+", "*"} and x > y:
                continue
            result = _result_for(x, y, op)
            if result is None or result.denominator != 1 or result <= 0:
                continue
            step = (
                f"{_format_number(x)} {op} {_format_number(y)} = "
                f"{_format_number(result)}"
            )
            yield SolutionCandidate(result, depth, x_steps + y_steps + (step,))


def solve_countdown(
    numbers: tuple[int, ...],
    target: int,
    *,
    max_solutions_per_state: int = 24,
) -> tuple[SolutionCandidate, ...]:
    state_cache: dict[tuple[Fraction, ...], list[SolutionCandidate]] = {}

    def solve_state(state: tuple[Fraction, ...]) -> list[SolutionCandidate]:
        key = tuple(sorted(state))
        cached = state_cache.get(key)
        if cached is not None:
            return cached
        if len(key) == 1:
            candidates = [SolutionCandidate(key[0], 0, ())]
            state_cache[key] = candidates
            return candidates

        candidates_by_steps: dict[tuple[str, ...], SolutionCandidate] = {}
        indexes = range(len(key))
        for i, j in itertools.combinations(indexes, 2):
            rest = [value for index, value in enumerate(key) if index not in (i, j)]
            for left in solve_state((key[i],)):
                for right in solve_state((key[j],)):
                    for candidate in _candidate_results(left, right):
                        next_state = tuple(sorted(rest + [candidate.value]))
                        for downstream in solve_state(next_state):
                            steps = candidate.steps + downstream.steps
                            combined = SolutionCandidate(
                                downstream.value,
                                candidate.depth + downstream.depth,
                                steps,
                            )
                            candidates_by_steps.setdefault(steps, combined)

        ordered = sorted(
            candidates_by_steps.values(),
            key=lambda candidate: (candidate.depth, len(candidate.steps)),
        )[:max_solutions_per_state]
        state_cache[key] = ordered
        return ordered

    initial = tuple(Fraction(number) for number in numbers)
    solutions = [
        candidate
        for candidate in solve_state(initial)
        if candidate.value == target and candidate.steps
    ]
    return tuple(sorted(solutions, key=lambda candidate: candidate.depth))


def generate_problem(
    *,
    rng: random.Random,
    num_numbers: int = 6,
    number_min: int = 1,
    number_max: int = 100,
    target_min: int = 100,
    target_max: int = 999,
    min_solution_depth: int = 3,
    require_all_numbers: bool = False,
    max_attempts: int = 1000,
) -> CountdownProblem:
    for _ in range(max_attempts):
        numbers = tuple(rng.randint(number_min, number_max) for _ in range(num_numbers))
        all_values = _reachable_targets(
            numbers, require_all_numbers=require_all_numbers
        )
        candidate_targets = [
            value
            for value, candidate in all_values.items()
            if target_min <= value <= target_max
            and candidate.depth >= min_solution_depth
        ]
        if not candidate_targets:
            continue
        target = rng.choice(candidate_targets)
        solution = all_values[target]
        return CountdownProblem(
            numbers=numbers,
            target=target,
            min_solution_depth=min_solution_depth,
            require_all_numbers=require_all_numbers,
            solution=solution.steps + (f"FINAL: {target}",),
        )
    raise ValueError("could not generate a matching Countdown problem")


def solution_uses_division(candidate: SolutionCandidate) -> bool:
    return any(" / " in step for step in candidate.steps)


def count_distinct_solutions(
    numbers: tuple[int, ...],
    target: int,
    *,
    max_solutions_per_state: int = 256,
) -> int:
    return len(
        solve_countdown(
            numbers,
            target,
            max_solutions_per_state=max_solutions_per_state,
        )
    )


def reachable_targets_by_subset(
    numbers: tuple[int, ...],
) -> dict[tuple[int, ...], dict[int, SolutionCandidate]]:
    reachable: dict[tuple[int, ...], dict[int, SolutionCandidate]] = {}
    for subset_size in range(1, len(numbers) + 1):
        for indexes in itertools.combinations(range(len(numbers)), subset_size):
            subset = tuple(numbers[index] for index in indexes)
            key = tuple(sorted(subset))
            if key in reachable:
                continue
            if len(subset) == 1:
                value = subset[0]
                reachable[key] = {value: SolutionCandidate(Fraction(value), 0, ())}
            else:
                reachable[key] = _reachable_targets_for_subset(subset)
    return reachable


def generate_problem_pool(
    *,
    rng: random.Random,
    num_problems: int,
    num_numbers: int = 6,
    filters: ProblemFilters = ProblemFilters(),
    max_duplicate_attempts: int = 100,
    excluded_problem_keys: Iterable[tuple[tuple[int, ...], int]] = (),
) -> list[CountdownProblem]:
    problems: list[CountdownProblem] = []
    seen: set[tuple[tuple[int, ...], int]] = set(excluded_problem_keys)
    attempts = 0
    while len(problems) < num_problems:
        attempts += 1
        if attempts > num_problems * max_duplicate_attempts:
            raise ValueError("too many duplicate or filtered generated problems")
        if filters.require_all_numbers and filters.min_distinct_solutions <= 1:
            problem = _generate_constructive_all_number_problem(
                rng=rng,
                num_numbers=num_numbers,
                filters=filters,
            )
            key = problem_key(problem)
            if key in seen:
                continue
            seen.add(key)
            problems.append(problem)
            continue
        numbers = tuple(
            rng.randint(filters.number_min, filters.number_max)
            for _ in range(num_numbers)
        )
        solutions_by_target = _reachable_targets(
            numbers,
            require_all_numbers=filters.require_all_numbers,
        )
        candidates: list[tuple[int, SolutionCandidate]] = []
        for target, solution in solutions_by_target.items():
            if not (filters.target_min <= target <= filters.target_max):
                continue
            if solution.depth < filters.min_solution_depth:
                continue
            if filters.division_required and not solution_uses_division(solution):
                continue
            if filters.min_distinct_solutions > 1:
                num_solutions = count_distinct_solutions(numbers, target)
                if num_solutions < filters.min_distinct_solutions:
                    continue
            candidates.append((target, solution))
        if not candidates:
            continue
        target, solution = rng.choice(candidates)
        problem = CountdownProblem(
            numbers=numbers,
            target=target,
            min_solution_depth=filters.min_solution_depth,
            require_all_numbers=filters.require_all_numbers,
            solution=solution.steps + (f"FINAL: {target}",),
        )
        key = problem_key(problem)
        if key in seen:
            continue
        seen.add(key)
        problems.append(problem)
    return problems


def _generate_constructive_all_number_problem(
    *,
    rng: random.Random,
    num_numbers: int,
    filters: ProblemFilters,
    max_attempts: int = 1000,
) -> CountdownProblem:
    if filters.min_solution_depth > num_numbers - 1:
        raise ValueError("min_solution_depth cannot exceed all-number solution depth")

    for _ in range(max_attempts):
        numbers = tuple(
            rng.randint(filters.number_min, filters.number_max)
            for _ in range(num_numbers)
        )
        values = [(number, ()) for number in numbers]
        used_division = False

        while len(values) > 1:
            left_index, right_index = sorted(
                rng.sample(range(len(values)), 2), reverse=True
            )
            left_value, left_steps = values.pop(left_index)
            right_value, right_steps = values.pop(right_index)

            candidates: list[tuple[str, int, str]] = [
                (
                    "+",
                    left_value + right_value,
                    f"{left_value} + {right_value} = {left_value + right_value}",
                ),
                (
                    "*",
                    left_value * right_value,
                    f"{left_value} * {right_value} = {left_value * right_value}",
                ),
            ]
            if left_value > right_value:
                candidates.append(
                    (
                        "-",
                        left_value - right_value,
                        f"{left_value} - {right_value} = {left_value - right_value}",
                    )
                )
            elif right_value > left_value:
                candidates.append(
                    (
                        "-",
                        right_value - left_value,
                        f"{right_value} - {left_value} = {right_value - left_value}",
                    )
                )
            if right_value != 0 and left_value % right_value == 0:
                candidates.append(
                    (
                        "/",
                        left_value // right_value,
                        f"{left_value} / {right_value} = {left_value // right_value}",
                    )
                )
            if left_value != 0 and right_value % left_value == 0:
                candidates.append(
                    (
                        "/",
                        right_value // left_value,
                        f"{right_value} / {left_value} = {right_value // left_value}",
                    )
                )

            bounded_candidates = [
                candidate
                for candidate in candidates
                if 0 < candidate[1] <= max(filters.target_max * 4, filters.number_max)
            ]
            if not bounded_candidates:
                break
            op, result, step = rng.choice(bounded_candidates)
            used_division = used_division or op == "/"
            values.append((result, left_steps + right_steps + (step,)))

        if len(values) != 1:
            continue
        target, steps = values[0]
        if not (filters.target_min <= target <= filters.target_max):
            continue
        if filters.division_required and not used_division:
            continue
        return CountdownProblem(
            numbers=numbers,
            target=target,
            min_solution_depth=filters.min_solution_depth,
            require_all_numbers=True,
            solution=steps + (f"FINAL: {target}",),
        )
    raise ValueError("could not construct a matching all-number Countdown problem")


def _reachable_targets(
    numbers: tuple[int, ...], *, require_all_numbers: bool
) -> dict[int, SolutionCandidate]:
    if require_all_numbers:
        return _reachable_targets_for_subset(numbers)

    reachable: dict[int, SolutionCandidate] = {}
    min_subset_size = 2
    for subset_size in range(min_subset_size, len(numbers) + 1):
        for subset in itertools.combinations(numbers, subset_size):
            for value, candidate in _reachable_targets_for_subset(subset).items():
                previous = reachable.get(value)
                if previous is None or candidate.depth < previous.depth:
                    reachable[value] = candidate
    return reachable


def _reachable_targets_for_subset(
    numbers: tuple[int, ...]
) -> dict[int, SolutionCandidate]:
    state_cache: dict[tuple[Fraction, ...], dict[Fraction, SolutionCandidate]] = {}

    def solve_state(state: tuple[Fraction, ...]) -> dict[Fraction, SolutionCandidate]:
        key = tuple(sorted(state))
        cached = state_cache.get(key)
        if cached is not None:
            return cached
        if len(key) == 1:
            candidate = SolutionCandidate(key[0], 0, ())
            state_cache[key] = {key[0]: candidate}
            return state_cache[key]

        best: dict[Fraction, SolutionCandidate] = {}
        for i, j in itertools.combinations(range(len(key)), 2):
            rest = [value for index, value in enumerate(key) if index not in (i, j)]
            for pair_candidate in _candidate_results(
                SolutionCandidate(key[i], 0, ()),
                SolutionCandidate(key[j], 0, ()),
            ):
                next_state = tuple(sorted(rest + [pair_candidate.value]))
                for value, downstream in solve_state(next_state).items():
                    candidate = SolutionCandidate(
                        value=value,
                        depth=pair_candidate.depth + downstream.depth,
                        steps=pair_candidate.steps + downstream.steps,
                    )
                    previous = best.get(value)
                    if previous is None or candidate.depth < previous.depth:
                        best[value] = candidate
        state_cache[key] = best
        return best

    reachable: dict[int, SolutionCandidate] = {}
    for value, candidate in solve_state(
        tuple(Fraction(number) for number in numbers)
    ).items():
        if value.denominator == 1 and value > 0:
            reachable[value.numerator] = candidate
    return reachable


def problem_key(problem: CountdownProblem) -> tuple[tuple[int, ...], int]:
    return tuple(sorted(problem.numbers)), problem.target
