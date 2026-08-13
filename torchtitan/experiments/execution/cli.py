# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""begin/stage/finish CLI facade for the typed lifecycle (roadmap Section 4).

run_common.sh and existing task runners sequence shell commands; this facade
lets them drive the typed attempt bundle without becoming Python recipes. Each
subcommand is a separate process:

    python -m torchtitan.experiments.execution begin ...
    python -m torchtitan.experiments.execution stage ... -- COMMAND...
    python -m torchtitan.experiments.execution finish ...

begin freezes the declaration and writes the manifest; stage reattaches to the
bundle and runs one stage, returning the command's return code so a shell
caller can react; finish reattaches, reconstructs stage invocation ids from the
append-only event stream, and commits the immutable outcome and report input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torchtitan.experiments.execution import models, store
from torchtitan.experiments.execution.lifecycle import RunAttempt
from torchtitan.experiments.execution.preflight import profiles, query


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchtitan.experiments.execution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    begin = subparsers.add_parser("begin", help="freeze a declaration into an attempt")
    _add_locator_args(begin)
    begin.add_argument("--family", required=True)
    begin.add_argument("--task", required=True)
    begin.add_argument("--lane", required=True)
    begin.add_argument("--parent-attempt-id", default=None)
    begin.add_argument(
        "--fields",
        default=None,
        help="path to a JSON file with the remaining normalized declaration",
    )
    begin.set_defaults(handler=_handle_begin)

    stage = subparsers.add_parser("stage", help="run one stage of an existing attempt")
    _add_locator_args(stage)
    stage.add_argument("--stage-id", required=True)
    stage.add_argument("--name", required=True)
    stage.add_argument("--kind", required=True, choices=models.STAGE_KINDS)
    stage.add_argument("--adapter", required=True, choices=models.EXECUTION_ADAPTERS)
    stage.add_argument("--cwd", default=None)
    stage.add_argument(
        "--stage-extra",
        action="append",
        default=None,
        dest="stage_extras",
        metavar="KEY=JSON",
        help="opaque per-stage metadata recorded verbatim on the terminal "
        "event; repeat for multiple keys. VALUE is parsed as JSON.",
    )
    stage.add_argument(
        "--stage-extra-file",
        action="append",
        default=None,
        dest="stage_extra_files",
        metavar="KEY=PATH",
        help="opaque per-stage metadata read from a JSON file after the "
        "command runs; a missing file is skipped. Repeat for multiple keys.",
    )
    stage.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the stage argv, after a -- separator",
    )
    stage.set_defaults(handler=_handle_stage)

    finish = subparsers.add_parser(
        "finish", help="commit the immutable attempt outcome"
    )
    _add_locator_args(finish)
    finish.add_argument(
        "--attempt-outcome", required=True, choices=models.ATTEMPT_EXECUTION_OUTCOMES
    )
    finish.add_argument(
        "--report-input",
        required=True,
        help="path to the canonical report input JSON",
    )
    finish.add_argument(
        "--evaluations",
        required=True,
        help="path to a JSON object mapping condition key to status dimensions",
    )
    finish.set_defaults(handler=_handle_finish)

    preflight = subparsers.add_parser(
        "preflight", help="query readiness over composed profiles (no journal write)"
    )
    preflight.add_argument(
        "--profile",
        action="append",
        required=True,
        dest="profiles",
        help="a profile name to compose; repeat for multiple profiles",
    )
    preflight.add_argument(
        "--output",
        default=None,
        help="path to write the preflight artifact; stdout only when omitted",
    )
    preflight.add_argument(
        "--require-ready",
        action="store_true",
        help="exit nonzero when the composed preflight is blocked",
    )
    preflight.set_defaults(handler=_handle_preflight)

    return parser


def _add_locator_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--results-root", required=True)
    sub.add_argument("--run-id", required=True)
    sub.add_argument("--attempt-id", required=True)


def _handle_begin(args: argparse.Namespace) -> int:
    fields = json.loads(Path(args.fields).read_text()) if args.fields else {}
    declaration = models.RunDeclaration(
        run_id=args.run_id,
        family=args.family,
        task=args.task,
        lane=args.lane,
        fields=fields,
    )
    RunAttempt.create(
        declaration,
        attempt_id=args.attempt_id,
        results_root=Path(args.results_root),
        parent_attempt_id=args.parent_attempt_id,
    )
    return 0


def _handle_stage(args: argparse.Namespace) -> int:
    argv = _stage_argv(args.command)
    stage_extra = _parse_stage_extras(args.stage_extras)
    stage_extra_files = _parse_stage_extra_files(args.stage_extra_files)
    attempt = RunAttempt.attach(
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        results_root=Path(args.results_root),
    )
    spec = models.StageSpec(
        stage_id=args.stage_id,
        name=args.name,
        kind=args.kind,
        adapter=args.adapter,
        argv=argv,
        cwd=args.cwd,
    )
    event = attempt.run_stage(
        spec,
        stage_extra=stage_extra,
        stage_extra_files=stage_extra_files,
    )
    # Propagate the command's return code so a shell caller can react; a
    # missing return code (blocked/interrupted) is a nonzero facade failure.
    return event.return_code if event.return_code is not None else 1


def _handle_finish(args: argparse.Namespace) -> int:
    canonical_report_input = json.loads(Path(args.report_input).read_text())
    raw_evaluations = json.loads(Path(args.evaluations).read_text())
    evaluations = {
        key: models.ConditionStatus(**status) for key, status in raw_evaluations.items()
    }
    attempt = RunAttempt.attach(
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        results_root=Path(args.results_root),
    )
    attempt.finish(
        canonical_report_input=canonical_report_input,
        evaluations=evaluations,
        attempt_outcome=args.attempt_outcome,
    )
    return 0


def _handle_preflight(args: argparse.Namespace) -> int:
    env = profiles.probe_env()
    result = query.run_preflight(profile_names=args.profiles, env=env)
    if args.output:
        store.write_terminal_json(Path(args.output), result)
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
    # A query never writes the attempt journal; it only reports readiness. The
    # caller reacts to the exit code when --require-ready is set.
    if args.require_ready and result["readiness"] != "ready":
        print(
            "preflight blocked: " + ", ".join(result["blocker_codes"]),
            file=sys.stderr,
        )
        return 1
    return 0


def _stage_argv(command: list[str]) -> list[str]:
    # argparse.REMAINDER keeps the -- separator; drop a single leading one.
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("stage requires a command after '--'")
    return command


def _parse_stage_extras(raw: list[str] | None) -> dict[str, object] | None:
    """Parse repeated --stage-extra KEY=JSON options into a dict.

    The value after the first '=' is parsed as JSON so a caller can attach
    structured metadata (an object, list, number) and not only a string.
    """

    if not raw:
        return None
    extras: dict[str, object] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--stage-extra must be KEY=JSON, got {item!r}")
        extras[key] = json.loads(value)
    return extras


def _parse_stage_extra_files(raw: list[str] | None) -> dict[str, str] | None:
    """Parse repeated --stage-extra-file KEY=PATH options into a dict.

    The path is resolved after the stage command runs, so a caller can attach a
    status file the command itself produces. A missing file is skipped rather
    than failing the stage.
    """

    if not raw:
        return None
    mapping: dict[str, str] = {}
    for item in raw:
        key, sep, path = item.partition("=")
        if not sep or not key or not path:
            raise ValueError(f"--stage-extra-file must be KEY=PATH, got {item!r}")
        mapping[key] = path
    return mapping


if __name__ == "__main__":
    raise SystemExit(main())
