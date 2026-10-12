# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Guarded Mini Kimi K3 launch coordinator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torchtitan.experiments.execution import models
from torchtitan.experiments.execution.executor import Executor
from torchtitan.experiments.execution.lifecycle import RunAttempt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Mini Kimi K3 guarded launch sequence."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/mini_kimi_k3/results"),
        help="Root directory containing runs/<run-id>/<attempt-id>/.",
    )
    parser.add_argument("--run-id", required=True, help="Stable Mini-K3 run id.")
    parser.add_argument("--attempt-id", required=True, help="Attempt id to launch.")
    parser.add_argument(
        "--mode",
        choices=("full", "tiny-plumbing"),
        default="full",
        help="Launch lane to preflight before training.",
    )
    parser.add_argument(
        "--token-manifest",
        type=Path,
        help="JSON token-shard manifest to validate before launch.",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        help="Directory containing manifest source subdirectories and shard files.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Sequence length used to validate shard usability.",
    )
    parser.add_argument(
        "--tiny-smoke-report",
        type=Path,
        help="JSON report produced by experiments/mini_kimi_k3/run.sh tiny-smoke.",
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        help="Optional first-party Mini-K3 checkout root for model-contract comparison.",
    )
    parser.add_argument(
        "--oracle-r1-config",
        type=Path,
        help="Optional first-party r1 KimiLinearConfig JSON for model-contract comparison.",
    )
    parser.add_argument(
        "--forward-oracle-report",
        type=Path,
        help="Optional first-party forward comparison report for Mini-K3 r1 logits.",
    )
    parser.add_argument(
        "--launch-backend-report",
        type=Path,
        help="Optional reviewed launch-backend report for Mini-K3 r1 training.",
    )
    parser.add_argument(
        "--r1-training-smoke-report",
        type=Path,
        help="Optional run_train.sh smoke report for Mini-K3 r1 Trainer execution.",
    )
    parser.add_argument(
        "--r1-corpus-plan-audit-report",
        type=Path,
        help="Optional audit-r1-corpus-plan report for source-level corpus deficits.",
    )
    parser.add_argument(
        "--corpus-disk-readiness-report",
        type=Path,
        help="Optional corpus-disk-readiness report for raw shard filesystem capacity.",
    )
    parser.add_argument(
        "--stage1-source-resolution-report",
        type=Path,
        help="Optional probe-stage1-sources report for Stage 1 source resolution.",
    )
    parser.add_argument(
        "--stage1-input-inspection-report",
        type=Path,
        help="Optional inspect-stage1-inputs report for local Stage 1 inputs.",
    )
    parser.add_argument(
        "--stage1-remote-readiness-report",
        type=Path,
        help="Optional stage1-remote-readiness report for remote corpus access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_launch(args)


def run_launch(args: argparse.Namespace, *, executor: Executor | None = None) -> int:
    attempt = RunAttempt.attach(
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        results_root=args.results_root,
        executor=executor,
    )
    outcome_path = attempt.bundle_dir / "outcome.json"
    if outcome_path.exists():
        print(
            "Mini Kimi K3 attempt already has terminal outcome; "
            f"not rerunning {args.run_id}/{args.attempt_id}: {outcome_path}",
            file=sys.stderr,
        )
        return 21

    report_path = attempt.bundle_dir / "derived" / "preflight.json"
    argv = [
        sys.executable,
        "-m",
        "experiments.mini_kimi_k3.preflight",
        "--mode",
        args.mode,
        "--report",
        str(report_path),
        "--evidence-results-root",
        str(args.results_root),
        "--run-id",
        args.run_id,
        "--attempt-id",
        args.attempt_id,
        "--seq-len",
        str(args.seq_len),
    ]
    if args.token_manifest is not None:
        argv.extend(["--token-manifest", str(args.token_manifest)])
    if args.tokens_dir is not None:
        argv.extend(["--tokens-dir", str(args.tokens_dir)])
    if args.tiny_smoke_report is not None:
        argv.extend(["--tiny-smoke-report", str(args.tiny_smoke_report)])
    if args.oracle_root is not None:
        argv.extend(["--oracle-root", str(args.oracle_root)])
    if args.oracle_r1_config is not None:
        argv.extend(["--oracle-r1-config", str(args.oracle_r1_config)])
    if args.forward_oracle_report is not None:
        argv.extend(["--forward-oracle-report", str(args.forward_oracle_report)])
    if args.launch_backend_report is not None:
        argv.extend(["--launch-backend-report", str(args.launch_backend_report)])
    if args.r1_training_smoke_report is not None:
        argv.extend(["--r1-training-smoke-report", str(args.r1_training_smoke_report)])
    r1_corpus_plan_audit_report = getattr(args, "r1_corpus_plan_audit_report", None)
    if r1_corpus_plan_audit_report is not None:
        argv.extend(["--r1-corpus-plan-audit-report", str(r1_corpus_plan_audit_report)])
    corpus_disk_readiness_report = getattr(args, "corpus_disk_readiness_report", None)
    if corpus_disk_readiness_report is not None:
        argv.extend(
            ["--corpus-disk-readiness-report", str(corpus_disk_readiness_report)]
        )
    stage1_source_resolution_report = getattr(
        args, "stage1_source_resolution_report", None
    )
    if stage1_source_resolution_report is not None:
        argv.extend(
            [
                "--stage1-source-resolution-report",
                str(stage1_source_resolution_report),
            ]
        )
    stage1_input_inspection_report = getattr(
        args, "stage1_input_inspection_report", None
    )
    if stage1_input_inspection_report is not None:
        argv.extend(
            [
                "--stage1-input-inspection-report",
                str(stage1_input_inspection_report),
            ]
        )
    stage1_remote_readiness_report = getattr(
        args, "stage1_remote_readiness_report", None
    )
    if stage1_remote_readiness_report is not None:
        argv.extend(
            [
                "--stage1-remote-readiness-report",
                str(stage1_remote_readiness_report),
            ]
        )

    terminal = attempt.run_stage(
        models.StageSpec(
            stage_id="preflight",
            name="Mini Kimi K3 launch preflight",
            kind="preflight",
            adapter="rootfs_cpu",
            argv=argv,
            cwd=str(Path.cwd()),
        ),
        stage_extra_files={"preflight_report": str(report_path)},
        terminal_kind_by_return_code={21: "stage_blocked"},
    )
    if terminal.kind != "stage_succeeded":
        _finish_blocked_preflight(attempt, args=args, report_path=report_path)
        return terminal.return_code if terminal.return_code is not None else 1

    preflight_error = _validate_ready_preflight_report(
        report_path=report_path,
        expected_mode=args.mode,
    )
    if preflight_error is not None:
        print(
            f"Mini Kimi K3 launch blocked by preflight report: {preflight_error}",
            file=sys.stderr,
        )
        _finish_blocked_preflight(attempt, args=args, report_path=report_path)
        return 21

    train_terminal = attempt.run_stage(
        models.StageSpec(
            stage_id="train",
            name="Mini Kimi K3 r1 TorchTitan training",
            kind="train",
            adapter="rootfs_torchrun_sft",
            argv=_train_argv(args, attempt=attempt),
            cwd=str(Path.cwd()),
            depends_on=("preflight",),
            supports_resume=True,
        ),
    )
    attempt_outcome = (
        "completed" if train_terminal.kind == "stage_succeeded" else "failed"
    )
    training_status = (
        models.ConditionStatus(
            execution_outcome="completed",
            measurement="real",
            promotion="not_evaluated",
            stage_invocation_id=train_terminal.stage_invocation_id,
        )
        if train_terminal.kind == "stage_succeeded"
        else models.ConditionStatus(
            execution_outcome="failed",
            measurement="invalid",
            promotion="reject",
            stage_invocation_id=train_terminal.stage_invocation_id,
        )
    )
    attempt.finish(
        canonical_report_input={
            "schema_version": 1,
            "kind": "mini_kimi_k3_launch_report_input",
            "run": {
                "run_id": args.run_id,
                "attempt_id": args.attempt_id,
                "mode": args.mode,
            },
            "preflight_report": str(report_path),
        },
        evaluations={
            "preflight": models.ConditionStatus(
                execution_outcome="completed",
                measurement="real",
                promotion="not_evaluated",
            ),
            "training": training_status,
        },
        attempt_outcome=attempt_outcome,
    )
    return train_terminal.return_code if train_terminal.return_code is not None else 1


def _finish_blocked_preflight(
    attempt: RunAttempt,
    *,
    args: argparse.Namespace,
    report_path: Path,
) -> None:
    attempt.finish(
        canonical_report_input={
            "schema_version": 1,
            "kind": "mini_kimi_k3_launch_report_input",
            "run": {
                "run_id": args.run_id,
                "attempt_id": args.attempt_id,
                "mode": args.mode,
            },
            "preflight_report": str(report_path),
        },
        evaluations={
            "preflight": models.ConditionStatus(
                execution_outcome="blocked",
                measurement="not_run",
                promotion="not_evaluated",
            )
        },
        attempt_outcome="blocked",
    )


def _validate_ready_preflight_report(
    *,
    report_path: Path,
    expected_mode: str,
) -> str | None:
    if not report_path.is_file():
        return f"missing report at {report_path}"
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return f"invalid JSON at {report_path}: {exc}"
    if report.get("kind") != "mini_kimi_k3_preflight":
        return f"unexpected kind {report.get('kind')!r}"
    if report.get("mode") != expected_mode:
        return f"unexpected mode {report.get('mode')!r}; expected {expected_mode!r}"
    if report.get("status") != "ready":
        return f"status {report.get('status')!r} is not ready"
    return None


def _train_argv(args: argparse.Namespace, *, attempt: RunAttempt) -> list[str]:
    argv = [
        "/usr/bin/env",
        "MODULE=mini_kimi_k3",
        "CONFIG=mini_kimi_k3_r1_contract",
        "NGPU=1",
        f"TORCHTITAN_RUN_ID={args.run_id}",
        f"TORCHTITAN_ATTEMPT_ID={args.attempt_id}",
        "./run_train.sh",
        f"--dump_folder={attempt.bundle_dir / 'train'}",
        "--checkpoint.keep_latest_k=2",
    ]
    if args.token_manifest is not None:
        argv.append(f"--dataloader.token_manifest={args.token_manifest}")
    if args.tokens_dir is not None:
        argv.append(f"--dataloader.tokens_dir={args.tokens_dir}")
    return argv


if __name__ == "__main__":
    raise SystemExit(main())
