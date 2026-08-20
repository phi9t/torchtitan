# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.modded_nanogpt_b200 import optimized_kernel_certifier as certifier


def _args(
    tmp_path: Path, *, attention_backend: str = "fa2", mlp_backend: str = "triton"
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train_gpt.py").write_text("# fixture\n")
    (source / "triton_kernels.py").write_text("def fixture():\n    return None\n")
    (source / "triton_kernels.py").chmod(0o644)
    output = tmp_path / "runtime" / "optimized_kernel_report.json"
    return SimpleNamespace(
        source=source,
        output=output,
        attention_backend=attention_backend,
        mlp_backend=mlp_backend,
        run_id="issue08_cert_fixture",
        attempt_id="issue08_cert_fixture_attempt_001",
        environment_class="torchtitan-rootfs-b200",
        expected_gpus=8,
    )


def _passing_probe(name: str) -> dict:
    return {
        "support_status": "supported",
        "build_status": "passed",
        "smoke_status": "passed",
        "detail": {"probe": name},
    }


def test_selected_fa2_triton_report_is_launch_eligible(monkeypatch, tmp_path):
    args = _args(tmp_path)
    monkeypatch.setattr(certifier.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        certifier.preflight,
        "collect_rootfs_detail",
        lambda: {"marker": "1", "cwd": "/workspace/torchtitan"},
    )
    monkeypatch.setattr(
        certifier.preflight,
        "collect_environment_detail",
        lambda: {
            "torch": "2.13.0+cu132",
            "cuda_runtime": "13.2",
            "triton": "3.7.1",
            "flash_attention": "2.8.3.post1",
        },
    )
    monkeypatch.setattr(certifier, "_gpu_arch_evidence", lambda expected_gpus: [])
    monkeypatch.setattr(certifier, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(
        certifier, "_probe_attention_fa2", lambda: _passing_probe("fa2")
    )
    monkeypatch.setattr(
        certifier, "_probe_mlp_triton", lambda source: _passing_probe("mlp.triton")
    )
    monkeypatch.setattr(
        certifier, "_probe_torch_scaled_mm", lambda: _passing_probe("scaled_mm")
    )
    monkeypatch.setattr(
        certifier, "_probe_inductor_cache", lambda: _passing_probe("inductor")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_triton_tensor_descriptor",
        lambda: _passing_probe("tensor_descriptor"),
    )
    monkeypatch.setattr(
        certifier, "_probe_nccl", lambda expected_gpus: _passing_probe("nccl")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_source_triton",
        lambda source: _passing_probe("source_triton"),
    )
    monkeypatch.setattr(
        certifier, "_probe_source_dc_triton", lambda source: _passing_probe("source_dc")
    )
    monkeypatch.setattr(
        certifier,
        "_forbidden_component_rows",
        lambda: [
            certifier.make_row(
                name="forbidden.flashinfer",
                kind="package",
                role="forbidden_component",
                requested=True,
                required_for_selected_launch=True,
                support_status="absent",
                build_status="not_required",
                smoke_status="not_required",
            )
        ],
    )

    report = certifier.build_report(args)

    assert report["schema_version"] == 1
    assert report["launch_eligible"] is True
    assert report["selected_tuple"] == {
        "attention_backend": "fa2",
        "mlp_backend": "triton",
    }
    selected = {
        row["name"]: row
        for row in report["rows"]
        if row["selected_for_launch"] or row["required_for_selected_launch"]
    }
    assert selected["attention.fa2"]["launch_eligible"] is True
    assert selected["mlp.triton"]["launch_eligible"] is True
    assert selected["prereq.nccl_all_reduce"]["launch_eligible"] is True
    assert report["blockers"] == []


def test_unselected_unsupported_rows_are_preserved_without_blocking(
    monkeypatch, tmp_path
):
    args = _args(tmp_path)
    monkeypatch.setattr(certifier.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(certifier.preflight, "collect_rootfs_detail", lambda: {})
    monkeypatch.setattr(certifier.preflight, "collect_environment_detail", lambda: {})
    monkeypatch.setattr(certifier, "_gpu_arch_evidence", lambda expected_gpus: [])
    monkeypatch.setattr(certifier, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(
        certifier, "_probe_attention_fa2", lambda: _passing_probe("fa2")
    )
    monkeypatch.setattr(
        certifier, "_probe_mlp_triton", lambda source: _passing_probe("mlp.triton")
    )
    monkeypatch.setattr(
        certifier, "_probe_torch_scaled_mm", lambda: _passing_probe("scaled_mm")
    )
    monkeypatch.setattr(
        certifier, "_probe_inductor_cache", lambda: _passing_probe("inductor")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_triton_tensor_descriptor",
        lambda: _passing_probe("tensor_descriptor"),
    )
    monkeypatch.setattr(
        certifier, "_probe_nccl", lambda expected_gpus: _passing_probe("nccl")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_source_triton",
        lambda source: _passing_probe("source_triton"),
    )
    monkeypatch.setattr(
        certifier, "_probe_source_dc_triton", lambda source: _passing_probe("source_dc")
    )
    monkeypatch.setattr(
        certifier,
        "_forbidden_component_rows",
        lambda: [
            certifier.make_row(
                name="forbidden.flashinfer",
                kind="package",
                role="forbidden_component",
                requested=True,
                required_for_selected_launch=True,
                support_status="absent",
                build_status="not_required",
                smoke_status="not_required",
            )
        ],
    )

    report = certifier.build_report(args)
    rows = {row["name"]: row for row in report["rows"]}

    assert report["launch_eligible"] is True
    assert rows["attention.fa3"]["support_status"] == "unsupported"
    assert rows["attention.fa3"]["diagnostic_only"] is True
    assert rows["attention.fa3"]["launch_eligible"] is False
    assert rows["attention.fa4"]["support_status"] == "unsupported"
    assert rows["attention.flex"]["support_status"] == "unsupported"
    assert rows["attention.torch_sdpa"]["selected_for_launch"] is False
    assert rows["mlp.torch"]["diagnostic_only"] is True
    assert all(blocker["row"] != "attention.fa3" for blocker in report["blockers"])


def test_forbidden_present_component_blocks_launch(monkeypatch, tmp_path):
    args = _args(tmp_path)
    monkeypatch.setattr(certifier.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(certifier.preflight, "collect_rootfs_detail", lambda: {})
    monkeypatch.setattr(certifier.preflight, "collect_environment_detail", lambda: {})
    monkeypatch.setattr(certifier, "_gpu_arch_evidence", lambda expected_gpus: [])
    monkeypatch.setattr(certifier, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(
        certifier, "_probe_attention_fa2", lambda: _passing_probe("fa2")
    )
    monkeypatch.setattr(
        certifier, "_probe_mlp_triton", lambda source: _passing_probe("mlp.triton")
    )
    monkeypatch.setattr(
        certifier, "_probe_torch_scaled_mm", lambda: _passing_probe("scaled_mm")
    )
    monkeypatch.setattr(
        certifier, "_probe_inductor_cache", lambda: _passing_probe("inductor")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_triton_tensor_descriptor",
        lambda: _passing_probe("tensor_descriptor"),
    )
    monkeypatch.setattr(
        certifier, "_probe_nccl", lambda expected_gpus: _passing_probe("nccl")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_source_triton",
        lambda source: _passing_probe("source_triton"),
    )
    monkeypatch.setattr(
        certifier, "_probe_source_dc_triton", lambda source: _passing_probe("source_dc")
    )
    monkeypatch.setattr(
        certifier,
        "_forbidden_component_rows_for_source",
        lambda source: [
            certifier.make_row(
                name="forbidden.flashinfer",
                kind="package",
                role="forbidden_component",
                requested=True,
                required_for_selected_launch=True,
                support_status="forbidden_present",
                build_status="failed",
                smoke_status="not_run",
                failure_class="forbidden_component_present",
                blockers=[
                    {
                        "class": "forbidden_component_present",
                        "message": "flashinfer is importable",
                    }
                ],
            )
        ],
    )

    report = certifier.build_report(args)

    assert report["launch_eligible"] is False
    assert report["blockers"] == [
        {
            "row": "forbidden.flashinfer",
            "class": "forbidden_component_present",
            "message": "flashinfer is importable",
        }
    ]


def test_installed_unselected_flashinfer_is_visible_without_blocking(
    monkeypatch, tmp_path
):
    args = _args(tmp_path)
    fake_module = SimpleNamespace(__file__="/runtime/flashinfer/__init__.py")

    def fake_import(name: str):
        if name == "flashinfer":
            return fake_module
        return importlib.import_module(name)

    monkeypatch.setattr(certifier.importlib, "import_module", fake_import)

    rows = certifier._forbidden_component_rows_for_source(args.source)

    assert rows == [
        {
            "name": "forbidden.flashinfer",
            "kind": "package",
            "role": "forbidden_component",
            "requested": True,
            "selected_for_launch": False,
            "required_for_selected_launch": True,
            "diagnostic_only": False,
            "support_status": "installed_unselected",
            "build_status": "not_required",
            "smoke_status": "not_required",
            "launch_eligible": True,
            "blockers": [],
            "detail": {
                "package": "flashinfer",
                "module_file": "/runtime/flashinfer/__init__.py",
                "source_import_matches": [],
                "policy": "installed but absent from selected source manifest",
            },
        }
    ]


def test_selected_failure_blocks_launch(monkeypatch, tmp_path):
    args = _args(tmp_path)
    monkeypatch.setattr(certifier.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(certifier.preflight, "collect_rootfs_detail", lambda: {})
    monkeypatch.setattr(certifier.preflight, "collect_environment_detail", lambda: {})
    monkeypatch.setattr(certifier, "_gpu_arch_evidence", lambda expected_gpus: [])
    monkeypatch.setattr(certifier, "_source_commit", lambda source: "abc123")
    monkeypatch.setattr(
        certifier,
        "_probe_attention_fa2",
        lambda: {
            "support_status": "supported",
            "build_status": "passed",
            "smoke_status": "failed",
            "failure_class": "fa2_smoke_failed",
            "blockers": [
                {
                    "class": "fa2_smoke_failed",
                    "message": "FA2 varlen/window smoke failed",
                }
            ],
        },
    )
    monkeypatch.setattr(
        certifier, "_probe_mlp_triton", lambda source: _passing_probe("mlp.triton")
    )
    monkeypatch.setattr(
        certifier, "_probe_torch_scaled_mm", lambda: _passing_probe("scaled_mm")
    )
    monkeypatch.setattr(
        certifier, "_probe_inductor_cache", lambda: _passing_probe("inductor")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_triton_tensor_descriptor",
        lambda: _passing_probe("tensor_descriptor"),
    )
    monkeypatch.setattr(
        certifier, "_probe_nccl", lambda expected_gpus: _passing_probe("nccl")
    )
    monkeypatch.setattr(
        certifier,
        "_probe_source_triton",
        lambda source: _passing_probe("source_triton"),
    )
    monkeypatch.setattr(
        certifier, "_probe_source_dc_triton", lambda source: _passing_probe("source_dc")
    )
    monkeypatch.setattr(
        certifier,
        "_forbidden_component_rows",
        lambda: [
            certifier.make_row(
                name="forbidden.flashinfer",
                kind="package",
                role="forbidden_component",
                requested=True,
                required_for_selected_launch=True,
                support_status="absent",
                build_status="not_required",
                smoke_status="not_required",
            )
        ],
    )

    report = certifier.build_report(args)

    assert report["launch_eligible"] is False
    assert report["blockers"] == [
        {
            "row": "attention.fa2",
            "class": "fa2_smoke_failed",
            "message": "FA2 varlen/window smoke failed",
        }
    ]


def test_main_writes_report_and_schema_digest(monkeypatch, tmp_path):
    args = _args(tmp_path)
    monkeypatch.setattr(certifier, "parse_args", lambda: args)
    monkeypatch.setattr(
        certifier,
        "build_report",
        lambda args: {
            "schema_version": 1,
            "launch_eligible": True,
            "rows": [],
            "blockers": [],
        },
    )

    assert certifier.main() == 0

    report = json.loads(args.output.read_text())
    assert report["launch_eligible"] is True
    assert report["schema_digest"]["path"].endswith(
        "optimized_kernel_report.schema.json"
    )
    assert len(report["report_digest"]["sha256"]) == 64
