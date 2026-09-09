# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import torchtitan.experiments.falcon.addition as addition
from torchtitan.experiments.falcon.addition import (
    AdditionSplitManifest,
    AdditionVocab,
    encode_addition,
    evaluate_addition_suffixes,
    OnlineAdditionStream,
    problem_identity,
)


REPO_ROOT = Path(__file__).parents[2]


def _all_manifest_rows(manifest: AdditionSplitManifest) -> list[dict]:
    return [
        row for draw_id in ("id", "ood", "challenge") for row in manifest.draws[draw_id]
    ]


def test_manifest_draws_are_width_balanced_disjoint_and_deterministic():
    """Sampling a duplicate or width-skewed held-out identity must fail."""
    first = AdditionSplitManifest.generate(seed=7)
    second = AdditionSplitManifest.generate(seed=7)

    assert first.digest == second.digest
    assert first.to_dict() == second.to_dict()
    assert set(first.draws) == {"id", "ood", "challenge"}

    identities = set()
    for draw_id, rows in first.draws.items():
        assert len(rows) == 24 * 8
        assert {row["draw_id"] for row in rows} == {draw_id}
        for width in range(1, 25):
            width_rows = [row for row in rows if row["width"] == width]
            assert len(width_rows) == 8
        for row in rows:
            assert row["schema_version"] == first.schema_version
            assert row["digest"] == first.digest
            assert (
                row["target"]
                == str(row["a"] + row["b"]).rjust(row["width"] + 1, "0")[::-1]
            )
            identity = problem_identity(row["width"], row["a"], row["b"])
            assert identity not in identities
            identities.add(identity)

    assert problem_identity(3, 123, 456) == problem_identity(3, 456, 123)


def test_manifest_validation_rejects_tampering_and_cross_draw_duplicates():
    """A stale digest or reused commuted pair must never validate as clean eval."""
    manifest = AdditionSplitManifest.generate(seed=9)
    payload = manifest.to_dict()

    tampered = json.loads(json.dumps(payload))
    tampered["draws"]["id"][0]["target"] = "not-a-sum"
    tampered["digest"] = addition._digest_manifest_payload(tampered)
    for rows in tampered["draws"].values():
        for row in rows:
            row["digest"] = tampered["digest"]
    with pytest.raises(ValueError, match="target"):
        AdditionSplitManifest.from_dict(tampered)

    tampered = json.loads(json.dumps(payload))
    first_id = tampered["draws"]["id"][0]
    tampered["draws"]["ood"][0]["width"] = first_id["width"]
    tampered["draws"]["ood"][0]["a"] = first_id["b"]
    tampered["draws"]["ood"][0]["b"] = first_id["a"]
    tampered["draws"]["ood"][0]["prompt"] = f"{first_id['b']}+{first_id['a']}="
    tampered["draws"]["ood"][0]["target"] = first_id["target"]
    tampered["digest"] = addition._digest_manifest_payload(tampered)
    for rows in tampered["draws"].values():
        for row in rows:
            row["digest"] = tampered["digest"]
    with pytest.raises(ValueError, match="duplicate"):
        AdditionSplitManifest.from_dict(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        AdditionSplitManifest.from_dict(tampered)


def test_online_stream_is_deterministic_id_only_and_excludes_eval_identities():
    """Training samples must not leak either orientation of held-out identities."""
    manifest = AdditionSplitManifest.generate(seed=3)
    eval_identities = {
        problem_identity(row["width"], row["a"], row["b"])
        for row in _all_manifest_rows(manifest)
    }

    left = OnlineAdditionStream(seed=11, manifest=manifest)
    right = OnlineAdditionStream(seed=11, manifest=manifest)
    left_examples = [left.next_example() for _ in range(256)]
    right_examples = [right.next_example() for _ in range(256)]

    assert [(ex.width, ex.a, ex.b) for ex in left_examples] == [
        (ex.width, ex.a, ex.b) for ex in right_examples
    ]
    assert {ex.width for ex in left_examples} <= set(range(1, 17))
    assert all(
        problem_identity(ex.width, ex.a, ex.b) not in eval_identities
        for ex in left_examples
    )


def test_suffix_metrics_mask_prompt_padding_and_report_position_and_carry_accuracy():
    """Wrong prompt/padding logits must not hide answer suffix metric errors."""
    vocab = AdditionVocab()
    examples = [
        encode_addition(9, 1, width=1, vocab=vocab),  # target 01, carry count 1
        encode_addition(12, 34, width=2, vocab=vocab),  # target 640, carry count 0
    ]
    batch = {
        "input": torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor(
            [
                [vocab.pad_id, *vocab.encode("01"), vocab.pad_id],
                [vocab.pad_id, *vocab.encode("640")],
            ],
            dtype=torch.long,
        ),
        "loss_mask": torch.tensor(
            [[False, True, True, False], [False, True, True, True]]
        ),
        "width": torch.tensor([1, 2], dtype=torch.long),
        "a": torch.tensor([9, 12], dtype=torch.long),
        "b": torch.tensor([1, 34], dtype=torch.long),
    }
    logits = torch.full((2, 4, vocab.size), -100.0)
    predictions = torch.tensor(
        [
            [vocab.encode("8")[0], *vocab.encode("01"), vocab.encode("7")[0]],
            [vocab.encode("8")[0], *vocab.encode("649")],
        ],
        dtype=torch.long,
    )
    logits.scatter_(-1, predictions.unsqueeze(-1), 100.0)

    metrics = evaluate_addition_suffixes(logits, batch, vocab=vocab)

    assert metrics.exact_suffix_micro == pytest.approx(0.5)
    assert metrics.token_accuracy_micro == pytest.approx(4 / 5)
    assert metrics.exact_suffix_width_macro == pytest.approx(0.5)
    assert metrics.token_accuracy_width_macro == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics.output_position_accuracy == {
        0: pytest.approx(1.0),
        1: pytest.approx(1.0),
        2: pytest.approx(0.0),
    }
    assert metrics.carry_count_accuracy == pytest.approx(1.0)


def test_collate_preserves_large_ood_operands_without_int64_overflow():
    # Break caught: 24-digit OOD operands exceed torch.long, but carry metrics
    # still need exact Python integer operands.
    vocab = AdditionVocab()
    examples = [
        encode_addition(
            10**23 + 123,
            10**23 + 456,
            width=24,
            vocab=vocab,
        )
    ]
    batch = addition.AdditionDataset(16, 24, [], examples).collate(
        examples, vocab=vocab
    )

    assert batch["a"] == [10**23 + 123]
    assert batch["b"] == [10**23 + 456]


def test_addition_runner_rejects_multiple_arms_or_seeds_and_writes_failure_bundle(
    tmp_path: Path,
):
    """One runner invocation owns exactly one matrix cell and records failures."""
    cmd = [
        sys.executable,
        "-m",
        "experiments.falcon.addition_runner",
        "--arm",
        "A0,A1",
        "--seed",
        "0",
        "--steps",
        "4",
        "--dry-run",
        "--results-root",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert "exactly one arm" in result.stderr

    bundles = list((tmp_path / "runs").glob("falcon-addition-A0_A1-seed0/*"))
    assert len(bundles) == 1
    outcome = json.loads((bundles[0] / "outcome.json").read_text())
    manifest = json.loads((bundles[0] / "manifest.json").read_text())
    assert manifest["claim_label"] == "smoke"
    assert outcome["status"] == "failed"
    assert "exactly one arm" in outcome["reason"]
    artifact_index = json.loads((bundles[0] / "artifact_index.json").read_text())
    failure_artifact = artifact_index[0]
    failure_path = REPO_ROOT / failure_artifact["path"]
    assert failure_path.is_file()
    assert (
        failure_artifact["digest"]
        == hashlib.sha256(failure_path.read_bytes()).hexdigest()
    )


def test_addition_runner_dry_run_writes_one_success_bundle(tmp_path: Path):
    """Dry-run smoke must publish a valid native addition attempt bundle."""
    cmd = [
        sys.executable,
        "-m",
        "experiments.falcon.addition_runner",
        "--arm",
        "A0",
        "--seed",
        "0",
        "--steps",
        "4",
        "--dry-run",
        "--results-root",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout)
    bundle = Path(payload["bundle_path"])
    assert bundle.is_dir()
    manifest = json.loads((bundle / "manifest.json").read_text())
    outcome = json.loads((bundle / "outcome.json").read_text())
    artifact_index = json.loads((bundle / "artifact_index.json").read_text())
    assert manifest["run_id"] == "falcon-addition-A0-seed0"
    assert manifest["claim_label"] == "smoke"
    assert manifest["data_lineage"]["split_manifest_digest"]
    assert manifest["checkpoint_lineage"]["checkpoint_policy"] == "dry-run-none"
    assert outcome["status"] == "completed"
    manifest_artifact = next(
        artifact
        for artifact in artifact_index
        if artifact["kind"] == "addition_split_manifest"
    )
    manifest_path = REPO_ROOT / manifest_artifact["path"]
    assert (
        manifest_artifact["digest"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert (
        manifest_artifact["split_manifest_digest"]
        == manifest["data_lineage"]["split_manifest_digest"]
    )
