from dataclasses import replace
from pathlib import Path

from torchtitan.experiments.qwen3_formal_verifier.qwen3_step import (
    build_torchtitan_qwen3_step_trace,
    export_qwen3_step_lean,
    export_qwen3_step_tla,
    verify_qwen3_step_trace,
    write_qwen3_step_formal_artifacts,
)
from torchtitan.experiments.qwen3_formal_verifier.schema import Trace


def test_torchtitan_qwen3_step_trace_models_dp_by_tp_mesh() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)
    report = verify_qwen3_step_trace(trace)

    assert report.ok is True
    assert report.issues == ()

    tp_work = _work_event(trace, "w-step3-layer0-tp-attn-dp0", "collective.enqueued")
    assert tp_work.extensions["mesh"]["coord"] == [0, 0]
    assert tp_work.process_group.id == "tp_group_dp0"
    assert tp_work.process_group.ranks == (0, 1)
    assert tp_work.work.op == "all_reduce"
    assert tp_work.extensions["work_role"] == "tp_attention_output_reduce"

    dp_work = _work_event(trace, "w-step3-layer0-dp-grad-tp1", "collective.enqueued")
    assert dp_work.process_group.id == "dp_group_tp1"
    assert dp_work.process_group.ranks == (1, 3)
    assert dp_work.extensions["work_role"] == "dp_gradient_sync"
    assert dp_work.extensions["shard"] == {"axis": "tp", "index": 1}


def test_torchtitan_qwen3_step_trace_includes_torch_semantic_events() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)

    down_proj = _event(trace, "torch.aten.completed", rank=0, role="qwen3_mlp_down_projection")
    assert down_proj.extensions["module"] == {
        "path": "model.layers.0.mlp.down_proj",
        "role": "qwen3_mlp_down_projection",
    }
    assert down_proj.extensions["aten"] == {
        "op": "mm",
        "dispatcher_key": "accelerator",
        "correlation_id": "aten-step3-layer0-rank0-mlp-down",
    }
    assert down_proj.extensions["tensor"] == {
        "placement": "tp_sharded",
        "mesh_axis": "tp",
        "shard_index": 0,
        "dtype": "bf16",
        "shape_digest": "qwen3.hidden_shard",
    }
    assert _event(trace, "torch.aten.completed", rank=1, role="qwen3_mlp_down_projection")

    tp_work = _work_event(trace, "w-step3-layer0-tp-mlp-dp0", "collective.enqueued")
    assert tp_work.extensions["correlation"] == {
        "producer_event": down_proj.event_id,
        "producer_aten": "aten-step3-layer0-rank0-mlp-down",
        "module": "model.layers.0.mlp.down_proj",
    }
    assert down_proj.event_id in tp_work.causal_predecessors

    grad_ready = _event(trace, "torch.autograd.grad_ready", rank=3, role="qwen3_layer_gradient")
    assert grad_ready.extensions["autograd"] == {
        "node": "Qwen3DecoderLayerBackward",
        "grad_role": "layer_param_grad",
        "bucket": "qwen3-layer0-tp1",
    }


def test_collective_lifecycle_events_keep_rank_local_torch_producer() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)
    producer_by_rank = {
        rank: _event(trace, "torch.aten.completed", rank=rank, role="qwen3_mlp_down_projection")
        for rank in (0, 1)
    }

    for kind in ("collective.enqueued", "collective.started", "collective.completed"):
        for rank, producer in producer_by_rank.items():
            event = _work_event_for_rank(trace, "w-step3-layer0-tp-mlp-dp0", kind, rank)

            assert event.extensions["correlation"]["producer_event"] == producer.event_id


def test_qwen3_step_verifier_rejects_dp_gradient_sync_on_wrong_tp_shard() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)
    bad_events = []
    for event in trace.events:
        if event.work is not None and event.work.id == "w-step3-layer0-dp-grad-tp1":
            extensions = dict(event.extensions)
            extensions["shard"] = {"axis": "tp", "index": 0}
            bad_events.append(replace(event, extensions=extensions))
        else:
            bad_events.append(event)
    report = verify_qwen3_step_trace(Trace.from_events(bad_events))

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"dp_gradient_shard_mismatch"}


def test_qwen3_step_verifier_rejects_step_end_before_required_dp_sync() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)
    bad_events = tuple(
        event
        for event in trace.events
        if not (
            event.kind == "collective.completed"
            and event.work is not None
            and event.work.id == "w-step3-layer0-dp-grad-tp1"
            and event.rank == 1
        )
    )
    report = verify_qwen3_step_trace(Trace.from_events(bad_events))

    assert "step_end_before_required_work" in {issue.code for issue in report.issues}


def test_qwen3_step_verifier_rejects_collective_without_torch_producer() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)
    bad_events = []
    for event in trace.events:
        if event.work is not None and event.work.id == "w-step3-layer0-tp-mlp-dp0":
            extensions = dict(event.extensions)
            extensions.pop("correlation", None)
            bad_events.append(replace(event, extensions=extensions))
        else:
            bad_events.append(event)
    report = verify_qwen3_step_trace(Trace.from_events(bad_events))

    assert "collective_missing_torch_producer" in {issue.code for issue in report.issues}


def test_qwen3_step_exports_tla_and_lean_sources() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)

    tla = export_qwen3_step_tla(trace)
    lean = export_qwen3_step_lean(trace)

    assert "Ranks == {0, 1, 2, 3}" in tla
    assert "CoordDP == (0 :> 0) @@ (1 :> 0) @@ (2 :> 1) @@ (3 :> 1)" in tla
    assert 'WorkShardTP == ("w-step3-layer0-dp-grad-tp0" :> 0) @@' in tla
    assert 'Axis == ("tp_group_dp0" :> "TP") @@' in tla
    assert '("w-step3-layer0-dp-grad-tp1" :> "dp_gradient_sync")' in tla
    assert '("w-step3-layer0-tp-mlp-dp0" :> "aten-step3-layer0-rank0-mlp-down")' in tla
    assert "DPShardAgreement ==" in tla
    assert "Coord[r].tp = WorkShardTP[w]" in tla
    assert "structure Trace where" in lean
    assert "def sameTpCoord" in lean
    assert "def meshWellFormed" in lean
    assert "def qwen3TraceTorchOps : List String :=" in lean
    assert '"aten-step3-layer0-rank1-mlp-down"' in lean
    assert "def qwen3TraceRanks : List Nat := [0, 1, 2, 3]" in lean
    assert "| 1 => { dp := 0, tp := 1 }" in lean
    assert '"w-step3-layer0-tp-mlp-dp1"' in lean


def test_qwen3_step_writes_formal_artifacts(tmp_path) -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)

    paths = write_qwen3_step_formal_artifacts(trace, tmp_path)

    assert paths.tla.name == "Qwen3StepTrace.tla"
    assert paths.lean.name == "Qwen3StepTrace.lean"
    assert paths.readme.name == "README.md"
    assert "MODULE Qwen3StepTrace" in paths.tla.read_text()
    assert "namespace Qwen3StepTrace" in paths.lean.read_text()
    assert "TorchTitan-style Qwen3 dense DPxTP" in paths.readme.read_text()


def test_checked_in_qwen3_formal_artifacts_match_exporter() -> None:
    trace = build_torchtitan_qwen3_step_trace(dp_size=2, tp_size=2, layers=1, step=3)
    formal_dir = Path("experiments/qwen3_formal_verifier/formal")

    assert (formal_dir / "Qwen3StepTrace.tla").read_text() == export_qwen3_step_tla(trace)
    assert (formal_dir / "Qwen3StepTrace.lean").read_text() == export_qwen3_step_lean(trace)


def _work_event(trace: Trace, work_id: str, kind: str):
    for event in trace.events:
        if event.kind == kind and event.work is not None and event.work.id == work_id:
            return event
    raise AssertionError(f"missing {kind} event for {work_id}")


def _work_event_for_rank(trace: Trace, work_id: str, kind: str, rank: int):
    for event in trace.events:
        if event.kind == kind and event.rank == rank and event.work is not None and event.work.id == work_id:
            return event
    raise AssertionError(f"missing {kind} event for rank {rank} work {work_id}")


def _event(trace: Trace, kind: str, rank: int, role: str):
    for event in trace.events:
        if event.kind == kind and event.rank == rank and event.extensions.get("torch_role") == role:
            return event
    raise AssertionError(f"missing {kind} event for rank {rank} role {role}")
