# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from types import SimpleNamespace

from experiments.modded_nanogpt_b200 import preflight


def test_environment_detail_records_flash_attention_version(monkeypatch):
    torch = SimpleNamespace(
        __version__="2.13.0+cu132",
        version=SimpleNamespace(cuda="13.2"),
    )
    triton = SimpleNamespace(__version__="3.7.1")
    monkeypatch.setattr(preflight, "load_torch", lambda: (torch, triton))
    monkeypatch.setattr(preflight.metadata, "version", lambda name: "2.8.3.post1")

    detail = preflight.collect_environment_detail()

    assert detail["torch"] == "2.13.0+cu132"
    assert detail["cuda_runtime"] == "13.2"
    assert detail["triton"] == "3.7.1"
    assert detail["flash_attention"] == "2.8.3.post1"


def test_full_runtime_contract_rejects_torch_cuda_triton_drift(monkeypatch):
    torch = SimpleNamespace(
        __version__="2.13.0+cu131",
        version=SimpleNamespace(cuda="13.1"),
    )
    triton = SimpleNamespace(__version__="3.7.0")
    monkeypatch.setattr(preflight, "load_torch", lambda: (torch, triton))

    try:
        preflight.check_runtime_contract(mode="full")
    except preflight.CheckFailure as exc:
        assert "runtime contract drift" in str(exc)
        assert exc.detail == {
            "expected": {
                "torch": "2.13.0+cu132",
                "cuda_runtime": "13.2",
                "triton": "3.7.1",
            },
            "actual": {
                "torch": "2.13.0+cu131",
                "cuda_runtime": "13.1",
                "triton": "3.7.0",
            },
            "mismatches": {
                "torch": {
                    "expected": "2.13.0+cu132",
                    "actual": "2.13.0+cu131",
                },
                "cuda_runtime": {
                    "expected": "13.2",
                    "actual": "13.1",
                },
                "triton": {
                    "expected": "3.7.1",
                    "actual": "3.7.0",
                },
            },
        }
    else:
        raise AssertionError("expected runtime contract failure")


def test_full_direct_runtime_dependencies_require_kernels_for_fa3(monkeypatch):
    available_imports = {
        "numpy",
        "tqdm",
        "huggingface_hub",
        "datasets",
        "tiktoken",
        "typing_extensions",
        "setuptools",
    }

    def fake_import(name):
        if name not in available_imports:
            raise ImportError(name)
        return object()

    monkeypatch.setattr(preflight.importlib, "import_module", fake_import)

    try:
        preflight.check_direct_runtime_dependencies(
            mode="full",
            lane="A",
            attention_backend="fa3",
        )
    except preflight.CheckFailure as exc:
        assert "missing direct runtime dependencies" in str(exc)
        assert exc.detail == {
            "missing": [{"package": "kernels", "import_name": "kernels"}],
            "required": [
                {"package": "numpy", "import_name": "numpy"},
                {"package": "tqdm", "import_name": "tqdm"},
                {"package": "huggingface_hub", "import_name": "huggingface_hub"},
                {"package": "datasets", "import_name": "datasets"},
                {"package": "tiktoken", "import_name": "tiktoken"},
                {
                    "package": "typing_extensions",
                    "import_name": "typing_extensions",
                },
                {"package": "setuptools", "import_name": "setuptools"},
                {"package": "kernels", "import_name": "kernels"},
            ],
        }
    else:
        raise AssertionError("expected missing kernels dependency failure")


def test_full_direct_runtime_dependencies_skip_kernels_for_lane_b_fa2(monkeypatch):
    imported: list[str] = []

    def fake_import(name):
        imported.append(name)
        return object()

    monkeypatch.setattr(preflight.importlib, "import_module", fake_import)

    detail = preflight.check_direct_runtime_dependencies(
        mode="full",
        lane="B",
        attention_backend="fa2",
    )

    assert "kernels" not in imported
    assert detail["missing"] == []
    assert detail["required"] == [
        {"package": "numpy", "import_name": "numpy"},
        {"package": "tqdm", "import_name": "tqdm"},
        {"package": "huggingface_hub", "import_name": "huggingface_hub"},
        {"package": "datasets", "import_name": "datasets"},
        {"package": "tiktoken", "import_name": "tiktoken"},
        {"package": "typing_extensions", "import_name": "typing_extensions"},
        {"package": "setuptools", "import_name": "setuptools"},
    ]


def test_main_checks_data_manifest_before_mlp_backend(monkeypatch, tmp_path):
    report = tmp_path / "preflight_report.json"
    args = SimpleNamespace(
        _nccl_worker=False,
        mode="full",
        lane="B",
        source=tmp_path / "source",
        data_manifest=tmp_path / "data_manifest.json",
        run_id="lane_b_full_gate",
        attempt_id="lane_b_full_gate_attempt_001",
        arm="B0",
        claim_label="B200 compatibility patchset",
        evidence_tier="full-single-attempt",
        environment_class=preflight.DEFAULT_ENVIRONMENT_CLASS,
        attention_backend="fa2",
        mlp_backend="torch",
        expected_gpus=2,
        expected_name="B200",
        skip_nccl=False,
        verify_sha=True,
        allow_previous_stall=False,
        report=report,
    )
    calls: list[str] = []

    monkeypatch.setattr(preflight, "parse_args", lambda: args)
    monkeypatch.setattr(preflight, "check_mode_policy", lambda parsed: calls.append("mode_policy") or {})
    monkeypatch.setattr(preflight, "check_rootfs", lambda: calls.append("rootfs") or None)
    monkeypatch.setattr(preflight, "collect_rootfs_detail", lambda: {})
    monkeypatch.setattr(preflight, "collect_environment_detail", lambda: calls.append("torch_import") or {})
    monkeypatch.setattr(preflight, "check_runtime_contract", lambda mode: calls.append("runtime_contract") or {})
    monkeypatch.setattr(
        preflight,
        "check_direct_runtime_dependencies",
        lambda mode, lane, attention_backend: calls.append("direct_runtime_dependencies") or {},
    )
    monkeypatch.setattr(preflight, "check_gpu_inventory", lambda expected_gpus, expected_name: calls.append("gpu_inventory") or [])
    monkeypatch.setattr(preflight, "check_torch_primitives", lambda: calls.append("torch_primitives") or None)
    monkeypatch.setattr(preflight, "check_nccl", lambda expected_gpus: calls.append("nccl_all_reduce") or None)
    monkeypatch.setattr(preflight, "check_source", lambda source, lane, attention_backend, mlp_backend: calls.append("source_policy") or "")
    monkeypatch.setattr(preflight, "check_attention", lambda lane, attention_backend: calls.append("attention_backend") or None)
    monkeypatch.setattr(
        preflight,
        "check_data_manifest",
        lambda data_manifest, mode, verify_sha: calls.append("data_manifest") or {"verified_sha": True},
    )

    def fail_mlp(source, mlp_backend, allow_previous_stall):
        calls.append("mlp_backend")
        raise preflight.CheckFailure("known full-job blocker")

    monkeypatch.setattr(preflight, "check_mlp_backend", fail_mlp)

    exit_code = preflight.main()

    assert exit_code == 21
    assert calls.index("data_manifest") < calls.index("mlp_backend")
    written = report.read_text()
    assert '"name": "data_manifest"' in written
    assert '"name": "mlp_backend"' in written


def test_main_writes_structured_runtime_contract_failure(monkeypatch, tmp_path):
    report = tmp_path / "preflight_report.json"
    args = SimpleNamespace(
        _nccl_worker=False,
        mode="full",
        lane="B",
        source=tmp_path / "source",
        data_manifest=tmp_path / "data_manifest.json",
        run_id="lane_b_full_gate",
        attempt_id="lane_b_full_gate_attempt_001",
        arm="B0",
        claim_label="B200 compatibility patchset",
        evidence_tier="full-single-attempt",
        environment_class=preflight.DEFAULT_ENVIRONMENT_CLASS,
        attention_backend="fa2",
        mlp_backend="triton",
        expected_gpus=2,
        expected_name="B200",
        skip_nccl=False,
        verify_sha=True,
        allow_previous_stall=False,
        report=report,
    )

    monkeypatch.setattr(preflight, "parse_args", lambda: args)
    monkeypatch.setattr(preflight, "check_mode_policy", lambda parsed: {})
    monkeypatch.setattr(preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(preflight, "collect_rootfs_detail", lambda: {})
    monkeypatch.setattr(preflight, "collect_environment_detail", lambda: {})

    def fail_runtime_contract(mode):
        raise preflight.CheckFailure(
            "runtime contract drift",
            detail={"mismatches": {"torch": {"expected": "2.13.0+cu132", "actual": "2.13.0+cu131"}}},
        )

    monkeypatch.setattr(preflight, "check_runtime_contract", fail_runtime_contract)
    monkeypatch.setattr(
        preflight,
        "check_direct_runtime_dependencies",
        lambda mode, lane, attention_backend: {},
    )

    exit_code = preflight.main()

    assert exit_code == 21
    data = json.loads(report.read_text())
    assert data["ok"] is False
    assert data["checks"][-1] == {
        "name": "runtime_contract",
        "ok": False,
        "error": "runtime contract drift",
        "detail": {
            "mismatches": {
                "torch": {
                    "expected": "2.13.0+cu132",
                    "actual": "2.13.0+cu131",
                }
            }
        },
    }
    assert data["failures"] == [
        {"name": "runtime_contract", "error": "runtime contract drift"}
    ]
    assert data["classification"]["claim_eligible"] is False


def test_full_manifest_requires_declared_verified_sha_when_verifying(
    monkeypatch,
    tmp_path,
):
    shard = tmp_path / "train_000000.bin"
    shard.write_bytes(b"abc")
    monkeypatch.setattr(preflight, "EXPECTED_FINEWEB_SHARDS", 1)
    monkeypatch.setattr(preflight, "EXPECTED_FINEWEB_BYTES", shard.stat().st_size)
    manifest = {
        "schema_version": preflight.SCHEMA_VERSION,
        "dataset": "fineweb10B",
        "token_budget": "900M",
        "source": {"commit": preflight.UPSTREAM_COMMIT},
        "command": ["python", "data/cached_fineweb10B.py", "9"],
        "files": [
            {
                "path": str(shard),
                "bytes": shard.stat().st_size,
                "sha256": preflight.sha256_file(shard),
            }
        ],
        "total_bytes": shard.stat().st_size,
    }
    manifest_path = tmp_path / "data_manifest.json"

    try:
        manifest_path.write_text(json.dumps(manifest) + "\n")
        preflight.check_data_manifest(
            manifest_path,
            mode="full",
            verify_sha=True,
        )
    except preflight.CheckFailure as exc:
        assert "verified_sha" in str(exc)
    else:
        raise AssertionError("expected missing verified_sha failure")

    manifest["verified_sha"] = False
    try:
        manifest_path.write_text(json.dumps(manifest) + "\n")
        preflight.check_data_manifest(
            manifest_path,
            mode="full",
            verify_sha=True,
        )
    except preflight.CheckFailure as exc:
        assert "verified_sha" in str(exc)
    else:
        raise AssertionError("expected false verified_sha failure")


def test_full_manifest_with_declared_verified_sha_still_checks_shard_hashes(
    monkeypatch,
    tmp_path,
):
    shard = tmp_path / "train_000000.bin"
    shard.write_bytes(b"abc")
    monkeypatch.setattr(preflight, "EXPECTED_FINEWEB_SHARDS", 1)
    monkeypatch.setattr(preflight, "EXPECTED_FINEWEB_BYTES", shard.stat().st_size)
    manifest = {
        "schema_version": preflight.SCHEMA_VERSION,
        "dataset": "fineweb10B",
        "token_budget": "900M",
        "source": {"commit": preflight.UPSTREAM_COMMIT},
        "command": ["python", "data/cached_fineweb10B.py", "9"],
        "files": [
            {
                "path": str(shard),
                "bytes": shard.stat().st_size,
                "sha256": preflight.sha256_file(shard),
            }
        ],
        "total_bytes": shard.stat().st_size,
        "verified_sha": True,
    }
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    detail = preflight.check_data_manifest(
        manifest_path,
        mode="full",
        verify_sha=True,
    )

    assert detail["verified_sha"] is True
    assert detail["manifest_verified_sha"] is True
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    try:
        preflight.check_data_manifest(
            manifest_path,
            mode="full",
            verify_sha=True,
        )
    except preflight.CheckFailure as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("expected shard hash verification failure")


def test_torch_mlp_full_policy_preserves_smoke_detail(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "triton_kernels.py").write_text("class FusedLinearReLUSquareFunction:\n    pass\n")

    monkeypatch.setattr(preflight, "_run_torch_mlp_smoke", lambda path: {"backend": "torch", "output_shape": [2, 16, 768]})

    try:
        preflight.check_mlp_backend(source, "torch", allow_previous_stall=False)
    except preflight.CheckFailure as exc:
        assert "passes a local smoke" in str(exc)
        assert exc.detail == {
            "backend": "torch",
            "local_smoke": {
                "backend": "torch",
                "output_shape": [2, 16, 768],
            },
            "full_mode_policy": "blocked_without_allow_previous_stall",
        }
    else:
        raise AssertionError("expected full-mode torch MLP policy failure")


def test_triton_mlp_full_policy_uses_local_smoke(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    monkeypatch.setattr(
        preflight,
        "_run_triton_mlp_smoke",
        lambda path: {"backend": "triton", "output_shape": [2, 16, 768]},
    )

    detail = preflight.check_mlp_backend(source, "triton", allow_previous_stall=False)

    assert detail == {
        "backend": "triton",
        "local_smoke": {
            "backend": "triton",
            "output_shape": [2, 16, 768],
        },
    }


def test_triton_mlp_full_policy_preserves_compile_blocker_detail(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    def fail_triton_smoke(path):
        raise preflight.CheckFailure(
            "Triton MLP smoke failed",
            detail={
                "backend": "triton",
                "blocked_kernel": "linear_relu_square_kernel",
                "blocked_arch": "sm100",
                "failure_class": "triton_compile",
                "compiler_pass": "TritonNvidiaGPUOptimizeTMemLayoutsPass",
            },
        )

    monkeypatch.setattr(preflight, "_run_triton_mlp_smoke", fail_triton_smoke)

    try:
        preflight.check_mlp_backend(source, "triton", allow_previous_stall=False)
    except preflight.CheckFailure as exc:
        assert "Triton MLP smoke failed" in str(exc)
        assert exc.detail == {
            "backend": "triton",
            "blocked_kernel": "linear_relu_square_kernel",
            "blocked_arch": "sm100",
            "failure_class": "triton_compile",
            "compiler_pass": "TritonNvidiaGPUOptimizeTMemLayoutsPass",
        }
    else:
        raise AssertionError("expected full-mode triton MLP policy failure")
