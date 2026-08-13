# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from torchtitan.observability.run_evidence import (
    ArtifactState,
    record_artifact as record_run_artifact,
    RunEvidence,
)
from torchtitan.tools.profiler import Profiler


class ProducerAbort(BaseException):
    pass


class EvidenceAbort(BaseException):
    pass


@pytest.fixture
def active_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "profiler-evidence-run")
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", "profiler-evidence-attempt")
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder=str(tmp_path),
        job_config={"training": {"steps": 8}},
        role="trainer",
        actor_id="core",
    ) as evidence:
        yield evidence


@pytest.fixture
def relative_active_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCHTITAN_RUN_ID", "profiler-relative-run")
    monkeypatch.setenv("TORCHTITAN_ATTEMPT_ID", "profiler-relative-attempt")
    with RunEvidence(
        RunEvidence.Config(),
        dump_folder="outputs",
        job_config={"training": {"steps": 8}},
        role="trainer",
        actor_id="core",
    ) as evidence:
        yield evidence


def artifact_rows(active_evidence, *, kind):
    index_path = next(
        (
            Path(active_evidence.dump_folder)
            / "run_evidence"
            / active_evidence.run_id
            / active_evidence.attempt_id
            / "indexes"
        ).glob("artifacts.*.jsonl")
    )
    return [
        row
        for row in (json.loads(line) for line in index_path.read_text().splitlines())
        if row["kind"] == kind
    ]


def test_trace_export_records_evidence_lifecycle(tmp_path, active_evidence):
    """Missing trace lifecycle registration would leave the index without the export."""
    profiler = Profiler(Profiler.Config(enable_profiling=True))
    fake_profiler = SimpleNamespace(
        step_num=7,
        export_chrome_trace=lambda path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(b"trace"),
        ),
    )
    output_file = tmp_path / "profiling" / "traces" / "rank0_trace.json.gz"

    profiler._export_trace(
        fake_profiler,
        output_file=str(output_file),
        post_processor=None,
    )

    rows = artifact_rows(active_evidence, kind="pytorch.profiler.trace")
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert {row["producer"] for row in rows} == {"pytorch_profiler"}
    assert {row["metadata"]["format"] for row in rows} == {"chrome_trace_json_gzip"}
    assert [row["step"] for row in rows] == [7, 7]


def test_trace_export_uses_relative_native_path_and_normalized_evidence_path(
    relative_active_evidence,
):
    """A relative dump folder must not be added twice by evidence normalization."""
    native_paths = []

    def export_trace(path):
        native_paths.append(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"trace")

    profiler = Profiler(Profiler.Config(enable_profiling=True))
    output_file = "outputs/profiling/traces/rank0_trace.json.gz"
    profiler._export_trace(
        SimpleNamespace(step_num=8, export_chrome_trace=export_trace),
        output_file=output_file,
        post_processor=None,
    )

    assert native_paths == [output_file]
    rows = artifact_rows(relative_active_evidence, kind="pytorch.profiler.trace")
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert {row["path"] for row in rows} == {"profiling/traces/rank0_trace.json.gz"}
    assert {row["path_type"] for row in rows} == {"dump_relative"}


def test_trace_export_records_evidence_failure_after_post_processing_error(
    tmp_path, active_evidence
):
    """A post-processor failure must leave the produced trace declared as failed."""
    profiler = Profiler(Profiler.Config(enable_profiling=True))
    fake_profiler = SimpleNamespace(
        step_num=11,
        export_chrome_trace=lambda path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(b"trace"),
        ),
    )
    output_file = tmp_path / "profiling" / "traces" / "rank0_trace.json.gz"

    with pytest.raises(RuntimeError, match="post-processing failed"):
        profiler._export_trace(
            fake_profiler,
            output_file=str(output_file),
            post_processor=lambda path: (_ for _ in ()).throw(
                RuntimeError("post-processing failed")
            ),
        )

    rows = artifact_rows(active_evidence, kind="pytorch.profiler.trace")
    assert [row["state"] for row in rows] == ["declared", "failed"]
    assert {row["metadata"]["format"] for row in rows} == {"chrome_trace_json_gzip"}
    assert [row["step"] for row in rows] == [11, 11]


def test_trace_export_records_evidence_failure_after_native_base_exception(
    tmp_path, active_evidence
):
    """A native BaseException must leave its declared trace in the failed state."""
    profiler = Profiler(Profiler.Config(enable_profiling=True))

    def fail_export(path):
        raise ProducerAbort("native export failed")

    with pytest.raises(ProducerAbort, match="native export failed"):
        profiler._export_trace(
            SimpleNamespace(step_num=12, export_chrome_trace=fail_export),
            output_file=str(tmp_path / "profiling" / "traces" / "rank0_trace.json.gz"),
            post_processor=None,
        )

    rows = artifact_rows(active_evidence, kind="pytorch.profiler.trace")
    assert [row["state"] for row in rows] == ["declared", "failed"]
    assert {row["metadata"]["format"] for row in rows} == {"chrome_trace_json_gzip"}
    assert [row["step"] for row in rows] == [12, 12]


def test_trace_export_preserves_producer_error_when_failed_evidence_append_fails(
    tmp_path, active_evidence
):
    """An artifact-index failure must not replace the native export failure."""
    profiler = Profiler(Profiler.Config(enable_profiling=True))
    failed_transitions = []

    def fail_export(path):
        raise ProducerAbort("native export failed")

    def fail_failed_transition(**kwargs):
        if kwargs["state"] is ArtifactState.FAILED:
            failed_transitions.append(kwargs["state"])
            raise EvidenceAbort("failed evidence append")
        return record_run_artifact(**kwargs)

    fake_profiler = SimpleNamespace(step_num=13, export_chrome_trace=fail_export)
    with mock.patch(
        "torchtitan.tools.profiler.record_artifact",
        side_effect=fail_failed_transition,
    ):
        with pytest.raises(ProducerAbort, match="native export failed"):
            profiler._export_trace(
                fake_profiler,
                output_file=str(tmp_path / "profiling" / "rank0_trace.json.gz"),
                post_processor=None,
            )
    assert failed_transitions == [ArtifactState.FAILED]


def test_memory_snapshot_records_evidence_lifecycle(tmp_path, active_evidence):
    """Missing snapshot lifecycle registration would make a native dump untraceable."""
    profiler = Profiler(Profiler.Config(enable_memory_snapshot=True, profile_freq=1))
    with mock.patch(
        "torchtitan.tools.profiler.device_module.memory._record_memory_history"
    ), mock.patch(
        "torchtitan.tools.profiler.device_module.memory._snapshot",
        return_value={"segments": []},
    ), mock.patch(
        "torch.distributed.get_rank", return_value=0
    ):
        memory_profiler = profiler.build_memory_profiler(
            global_step=0,
            base_folder=str(tmp_path),
            leaf_folder="",
        )
        memory_profiler.step()

    rows = artifact_rows(active_evidence, kind="pytorch.cuda.memory_snapshot")
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert {row["producer"] for row in rows} == {"pytorch_memory"}
    assert {row["metadata"]["format"] for row in rows} == {"python_pickle_v4"}
    assert [row["step"] for row in rows] == [1, 1]


def test_memory_snapshot_uses_relative_native_path_and_normalized_evidence_path(
    relative_active_evidence,
):
    """A relative snapshot path must complete against the real native file."""
    profiler = Profiler(Profiler.Config(enable_memory_snapshot=True, profile_freq=1))
    with mock.patch(
        "torchtitan.tools.profiler.device_module.memory._record_memory_history"
    ), mock.patch(
        "torchtitan.tools.profiler.device_module.memory._snapshot",
        return_value={"segments": []},
    ), mock.patch(
        "torch.distributed.get_rank", return_value=0
    ):
        memory_profiler = profiler.build_memory_profiler(
            global_step=0,
            base_folder="outputs",
            leaf_folder="",
        )
        memory_profiler.step()

    output_file = Path(
        "outputs/profiling/memory_snapshot/step_000000000001/000000_step_1.pickle"
    )
    assert output_file.exists()
    rows = artifact_rows(relative_active_evidence, kind="pytorch.cuda.memory_snapshot")
    assert [row["state"] for row in rows] == ["declared", "complete"]
    assert {row["path"] for row in rows} == {
        "profiling/memory_snapshot/step_000000000001/000000_step_1.pickle"
    }
    assert {row["path_type"] for row in rows} == {"dump_relative"}


def test_memory_snapshot_records_failure_for_native_base_exception(
    tmp_path, active_evidence
):
    """A native snapshot BaseException must transition the artifact to failed."""
    profiler = Profiler(Profiler.Config(enable_memory_snapshot=True, profile_freq=1))
    with mock.patch(
        "torchtitan.tools.profiler.device_module.memory._record_memory_history"
    ), mock.patch(
        "torchtitan.tools.profiler.pickle.dump",
        side_effect=ProducerAbort("native snapshot failed"),
    ), mock.patch(
        "torch.distributed.get_rank", return_value=0
    ):
        memory_profiler = profiler.build_memory_profiler(
            global_step=0,
            base_folder=str(tmp_path),
            leaf_folder="",
        )
        with pytest.raises(ProducerAbort, match="native snapshot failed"):
            memory_profiler.step()

    rows = artifact_rows(active_evidence, kind="pytorch.cuda.memory_snapshot")
    assert [row["state"] for row in rows] == ["declared", "failed"]


def test_memory_snapshot_preserves_native_base_exception_when_failed_append_raises(
    tmp_path, active_evidence
):
    """A failed-index BaseException cannot replace the native snapshot failure."""
    profiler = Profiler(Profiler.Config(enable_memory_snapshot=True, profile_freq=1))
    failed_transitions = []

    def fail_failed_transition(**kwargs):
        if kwargs["state"] is ArtifactState.FAILED:
            failed_transitions.append(kwargs["state"])
            raise EvidenceAbort("failed evidence append")
        return record_run_artifact(**kwargs)

    with mock.patch(
        "torchtitan.tools.profiler.device_module.memory._record_memory_history"
    ), mock.patch(
        "torchtitan.tools.profiler.pickle.dump",
        side_effect=ProducerAbort("native snapshot failed"),
    ), mock.patch(
        "torch.distributed.get_rank", return_value=0
    ), mock.patch(
        "torchtitan.tools.profiler.record_artifact",
        side_effect=fail_failed_transition,
    ):
        memory_profiler = profiler.build_memory_profiler(
            global_step=0,
            base_folder=str(tmp_path),
            leaf_folder="",
        )
        with pytest.raises(ProducerAbort, match="native snapshot failed"):
            memory_profiler.step()
    assert failed_transitions == [ArtifactState.FAILED]


class TestProfilerConfig(unittest.TestCase):
    def test_default_field_values(self):
        cfg = Profiler.Config()
        self.assertFalse(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "profiling/traces")
        self.assertEqual(cfg.profile_freq, 10)
        self.assertEqual(cfg.profiler_active, 1)
        self.assertEqual(cfg.profiler_warmup, 3)
        self.assertIsNone(cfg.profiler_repeat)
        self.assertIsNone(cfg.profiler_skip_first)
        self.assertIsNone(cfg.profiler_skip_first_wait)
        self.assertFalse(cfg.enable_memory_snapshot)
        self.assertEqual(cfg.save_memory_snapshot_folder, "profiling/memory_snapshot")

    def test_custom_field_values(self):
        cfg = Profiler.Config(
            enable_profiling=True,
            save_traces_folder="my_traces",
            profile_freq=50,
            profiler_repeat=2,
            profiler_skip_first=5,
            profiler_skip_first_wait=3,
        )
        self.assertTrue(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "my_traces")
        self.assertEqual(cfg.profile_freq, 50)
        self.assertEqual(cfg.profiler_repeat, 2)
        self.assertEqual(cfg.profiler_skip_first, 5)
        self.assertEqual(cfg.profiler_skip_first_wait, 3)

    def test_build_returns_profiler_instance(self):
        """Profiler.Config.build() auto-wires to Profiler via Configurable."""
        cfg = Profiler.Config()
        profiler = cfg.build()
        self.assertIsInstance(profiler, Profiler)


class TestProfilerInit(unittest.TestCase):
    def test_default_runtime_attrs(self):
        """Profiler initializes runtime attrs to safe defaults."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)


class TestProfilerDisabledPaths(unittest.TestCase):
    """Tests for the no-op / disabled paths that require no GPU."""

    def test_build_torch_profiler_disabled_returns_none(self):
        """build_torch_profiler returns None when profiling is disabled."""
        profiler = Profiler(Profiler.Config(enable_profiling=False))
        result = profiler.build_torch_profiler(
            global_step=0, base_folder="/tmp", leaf_folder=""
        )
        self.assertIsNone(result)

    def test_build_memory_profiler_disabled_returns_none(self):
        """build_memory_profiler returns None when memory snapshot is disabled."""
        profiler = Profiler(Profiler.Config(enable_memory_snapshot=False))
        result = profiler.build_memory_profiler(
            global_step=0, base_folder="/tmp", leaf_folder=""
        )
        self.assertIsNone(result)

    def test_runtime_args_stored_on_init(self):
        """Runtime kwargs passed to __init__ are stored on the instance."""
        profiler = Profiler(
            Profiler.Config(), global_step=42, base_folder="/data", leaf_folder="sub"
        )
        self.assertEqual(profiler._global_step, 42)
        self.assertEqual(profiler._base_folder, "/data")
        self.assertEqual(profiler._leaf_folder, "sub")

    def test_context_manager_step_is_noop(self):
        """With everything disabled, context manager and step() don't raise."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            self.assertIs(prof, profiler)
            self.assertIsNone(prof.torch_profiler)
            self.assertIsNone(prof.memory_profiler)
            prof.step()
            prof.step()

    def test_default_args_context_manager(self):
        """Profiler with default runtime args works as a context manager."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            prof.step()

    def test_step_noop_when_both_profilers_none(self):
        """step() is a no-op when torch_profiler and memory_profiler are both None."""
        profiler = Profiler(Profiler.Config())
        profiler.step()
        profiler.step()

    def test_exit_resets_profiler_attrs(self):
        """After __exit__, torch_profiler and memory_profiler are reset to None."""
        profiler = Profiler(Profiler.Config())
        with profiler:
            pass
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)

    def test_active_updates_runtime_args(self):
        """active() updates runtime args and returns self for context manager use."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")

        result = profiler.active(
            global_step=10, base_folder="/output", leaf_folder="replica_0"
        )
        self.assertIs(result, profiler)
        self.assertEqual(profiler._global_step, 10)
        self.assertEqual(profiler._base_folder, "/output")
        self.assertEqual(profiler._leaf_folder, "replica_0")

    def test_active_as_context_manager(self):
        """active() can be used as a context manager with 'with' statement."""
        profiler = Profiler(Profiler.Config())
        with profiler.active(global_step=5, base_folder="/tmp") as prof:
            self.assertIs(prof, profiler)
            self.assertEqual(prof._global_step, 5)
            prof.step()


class TestProfilerEnabledPaths(unittest.TestCase):
    """Tests for enabled profiler paths — uses mocked distributed rank."""

    def setUp(self):
        self.patcher_rank = mock.patch("torch.distributed.get_rank", return_value=0)
        self.patcher_rank.start()

    def tearDown(self):
        self.patcher_rank.stop()

    def test_build_torch_profiler_returns_active_handle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_freq=4,
                    profiler_warmup=1,
                    profiler_active=1,
                ),
                global_step=0,
                base_folder=tmpdir,
            )
            with profiler:
                self.assertIsNotNone(profiler.torch_profiler)


if __name__ == "__main__":
    unittest.main()
