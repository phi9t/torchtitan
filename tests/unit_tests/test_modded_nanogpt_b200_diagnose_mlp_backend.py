# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments.modded_nanogpt_b200 import diagnose_mlp_backend, preflight

LANE_B_TRITON_KERNELS = Path(
    "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/triton_kernels.py"
)
LANE_B_TRAIN_GPT = Path(
    "experiments/modded_nanogpt_b200/sources/modded-nanogpt-b200-sdpa/train_gpt.py"
)


def test_direct_script_help_imports_from_repo_root():
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/modded_nanogpt_b200/diagnose_mlp_backend.py",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert proc.returncode == 0
    assert "Run a bounded MLP backend diagnostic" in proc.stdout


def test_lane_b_triton_backward_flattens_3d_mlp_tensors():
    source = LANE_B_TRITON_KERNELS.read_text()
    backward_source = source[source.index("    def backward(ctx, grad_output):") :]
    backward_source = backward_source[
        : backward_source.index("\n\n\ndef reduce_mlp_activation_scales")
    ]

    assert (
        "grad_flat = grad_output.view((-1, grad_output.shape[-1]))" in backward_source
    )
    assert "x_flat = x.view((-1, x.shape[-1]))" in backward_source
    assert "dW2 = post.T @ grad_flat" in backward_source
    assert "dpre = linear_relu_square(grad_flat, W2, aux=post)" in backward_source
    assert "dW1 = dpre.T @ x_flat" in backward_source
    assert "dW1 = dpre.T @ x\n" not in backward_source


def test_lane_b_softcapped_cross_entropy_backward_matches_forward_arity():
    source = LANE_B_TRITON_KERNELS.read_text()
    class_source = source[source.index("class FusedSoftcappedCrossEntropy") :]

    assert (
        "def forward(ctx, x, targets, mtp_weights, prefix_targets, prefix_weight, "
        "lm_head_weight, x_s, w_s, grad_s, grad_scale, A=23.0, B=5.0, C=7.5):"
        in class_source
    )
    assert (
        "return grad_x, None, None, None, None, grad_w, None, None, None, None, "
        "None, None, None" in class_source
    )


def test_lane_b_train_compile_decorators_obey_compile_disable_env():
    source = LANE_B_TRAIN_GPT.read_text()

    assert "TORCH_COMPILE_DISABLE" in source
    assert "def maybe_compile(*args, **kwargs):" in source
    assert "@maybe_compile" in source
    direct_compile_decorators = [
        line
        for line in source.splitlines()
        if line.strip() == "@torch.compile"
        or line.strip().startswith("@torch.compile(")
    ]
    assert direct_compile_decorators == []
    assert (
        "torch.compile(model, dynamic=False, fullgraph=model_compile_fullgraph)"
        in source
    )


def test_lane_b_compile_disabled_optimizer_avoids_eager_uint32_cuda_ops():
    source = LANE_B_TRAIN_GPT.read_text()
    helper_source = source[source.index("    def _cautious_wd_and_update_inplace") :]
    helper_source = helper_source[
        : helper_source.index("    @staticmethod", helper_source.index("return"))
    ]

    assert "p.to(torch.int32) * 65536" in helper_source
    assert "p.to(torch.uint32) << 16" not in helper_source
    assert "p_precise_raw.view(torch.float32)" in helper_source


def test_torch_diagnostic_writes_success_report(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "mlp_report.json"
    args = SimpleNamespace(
        source=source,
        backend="torch",
        output=output,
        timeout_seconds=180,
    )

    monkeypatch.setattr(diagnose_mlp_backend, "parse_args", lambda: args)
    monkeypatch.setattr(diagnose_mlp_backend.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "collect_rootfs_detail",
        lambda: {"marker": "1", "cwd": "/workspace/torchtitan"},
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "collect_environment_detail",
        lambda: {"torch": "2.13.0+cu132", "triton": "3.7.1"},
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "_run_torch_mlp_smoke",
        lambda path: {"backend": "torch", "output_shape": [2, 16, 768]},
    )

    assert diagnose_mlp_backend.main() == 0
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["ok"] is True
    assert report["backend"] == "torch"
    assert report["source"] == str(source)
    assert report["rootfs"]["marker"] == "1"
    assert report["environment"]["torch"] == "2.13.0+cu132"
    assert report["detail"] == {"backend": "torch", "output_shape": [2, 16, 768]}


def test_triton_diagnostic_writes_subprocess_failure_report(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "mlp_report.json"
    args = SimpleNamespace(
        source=source,
        backend="triton",
        output=output,
        timeout_seconds=180,
    )

    monkeypatch.setattr(diagnose_mlp_backend, "parse_args", lambda: args)
    monkeypatch.setattr(diagnose_mlp_backend.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight, "collect_environment_detail", lambda: {}
    )

    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "collect_rootfs_detail",
        lambda: {"marker": "1"},
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "collect_environment_detail",
        lambda: {"torch": "2.13.0+cu132"},
    )

    def fail_subprocess(cmd, timeout_seconds, cwd=None):
        assert os.environ["MODDED_NANOGPT_CE_COMPUTE_CAPABILITY"] == "100"
        return subprocess.CompletedProcess(
            cmd,
            1,
            "TritonNvidiaGPUOptimizeTMemLayoutsPass failed\n",
            None,
        )

    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "run_checked_subprocess",
        fail_subprocess,
    )

    assert diagnose_mlp_backend.main() == 21
    report = json.loads(output.read_text())
    assert report["ok"] is False
    assert report["backend"] == "triton"
    assert report["rootfs"]["marker"] == "1"
    assert report["environment"]["torch"] == "2.13.0+cu132"
    assert report["failure_class"] == "triton_compile_or_runtime"
    assert "TritonNvidiaGPUOptimizeTMemLayoutsPass" in report["stdout"]
    assert report["detail"]["blocked_kernel"] == "linear_relu_square_kernel"
    assert report["detail"]["blocked_arch"] == "sm100"


def test_triton_diagnostic_classifies_backward_shape_mismatch(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "mlp_report.json"
    args = SimpleNamespace(
        source=source,
        backend="triton",
        output=output,
        timeout_seconds=180,
    )

    monkeypatch.setattr(diagnose_mlp_backend, "parse_args", lambda: args)
    monkeypatch.setattr(diagnose_mlp_backend.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight, "collect_rootfs_detail", lambda: {}
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight, "collect_environment_detail", lambda: {}
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "run_checked_subprocess",
        lambda cmd, timeout_seconds, cwd=None: subprocess.CompletedProcess(
            cmd,
            1,
            "RuntimeError: Expected size for first two dimensions of batch2 tensor "
            "to be: [2, 32] but got: [2, 16].\n",
            None,
        ),
    )

    assert diagnose_mlp_backend.main() == 21
    report = json.loads(output.read_text())
    assert report["failure_class"] == "triton_backward_shape_mismatch"
    assert report["detail"]["failure_class"] == "triton_backward_shape_mismatch"
    assert report["detail"]["failure_site"] == "FusedLinearReLUSquareFunction.backward"


def test_triton_diagnostic_uses_shared_preflight_smoke(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "mlp_report.json"
    args = SimpleNamespace(
        source=source,
        backend="triton",
        output=output,
        timeout_seconds=77,
    )
    calls = []

    monkeypatch.setattr(diagnose_mlp_backend, "parse_args", lambda: args)
    monkeypatch.setattr(diagnose_mlp_backend.preflight, "check_rootfs", lambda: None)
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight, "collect_rootfs_detail", lambda: {"marker": "1"}
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight, "collect_environment_detail", lambda: {}
    )

    def shared_smoke(path, timeout_seconds):
        calls.append((path, timeout_seconds))
        return {"backend": "triton", "output_shape": [2, 16, 768]}

    monkeypatch.setattr(
        diagnose_mlp_backend.preflight, "_run_triton_mlp_smoke", shared_smoke
    )
    monkeypatch.setattr(
        diagnose_mlp_backend.preflight,
        "run_checked_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostic bypassed shared smoke")
        ),
    )

    assert diagnose_mlp_backend.main() == 0
    report = json.loads(output.read_text())
    assert calls == [(source, 77)]
    assert report["ok"] is True
    assert report["detail"] == {"backend": "triton", "output_shape": [2, 16, 768]}


def test_diagnostic_writes_rootfs_failure_report(monkeypatch, tmp_path):
    output = tmp_path / "mlp_report.json"
    args = SimpleNamespace(
        source=tmp_path / "source",
        backend="torch",
        output=output,
        timeout_seconds=180,
    )

    monkeypatch.setattr(diagnose_mlp_backend, "parse_args", lambda: args)

    def fail_rootfs():
        raise preflight.CheckFailure("must run inside scripts/rootfs/enter_rootfs.sh")

    monkeypatch.setattr(diagnose_mlp_backend.preflight, "check_rootfs", fail_rootfs)

    assert diagnose_mlp_backend.main() == 21
    report = json.loads(output.read_text())
    assert report["ok"] is False
    assert report["backend"] == "torch"
    assert report["failure_class"] == "rootfs"
    assert "must run inside" in report["error"]
