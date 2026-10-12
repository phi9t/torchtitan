from torchtitan.experiments.qwen3_formal_verifier.emulator import LogicalAcceleratorEmulator
from torchtitan.experiments.qwen3_formal_verifier.model_export import to_lean_facts, to_tla_constants
from torchtitan.experiments.qwen3_formal_verifier.tracer import TensorSemantic, TorchSemanticTracer
from torchtitan.experiments.qwen3_formal_verifier.verifier import verify_trace


def test_logical_accelerator_emulator_emits_backend_neutral_collective_and_p2p() -> None:
    emulator = LogicalAcceleratorEmulator(world_size=2, device_count=1, backend_kind="mlu-emulated")
    process_group = emulator.create_process_group("world", ranks=(0, 1))

    collective = emulator.enqueue_collective(
        rank=0,
        process_group=process_group,
        op="all_reduce",
        tensor_bytes=1024,
    )
    emulator.complete_work(rank=0, work=collective)
    p2p = emulator.enqueue_p2p(
        rank=1,
        process_group=process_group,
        peer=0,
        direction="recv",
        tensor_bytes=512,
        is_batched=True,
    )

    assert verify_trace(emulator.trace).ok is True
    assert [event.kind for event in emulator.trace.events[-5:]] == [
        "collective.enqueued",
        "collective.started",
        "collective.completed",
        "p2p.enqueued",
        "p2p.started",
    ]
    assert p2p.peer == 0
    assert p2p.direction == "recv"
    assert p2p.is_batched is True


def test_model_export_erases_backend_specific_extensions() -> None:
    emulator = LogicalAcceleratorEmulator(world_size=2, device_count=1, backend_kind="cuda-emulated")
    process_group = emulator.create_process_group("world", ranks=(0, 1))
    work = emulator.enqueue_collective(rank=0, process_group=process_group, op="all_reduce", tensor_bytes=8)
    emulator.complete_work(rank=0, work=work)

    constants = to_tla_constants(emulator.trace)
    facts = to_lean_facts(emulator.trace)

    assert constants["ranks"] == [0]
    assert constants["devices"] == ["dev-0"]
    assert constants["executors"] == ["stream-0-default"]
    assert constants["works"] == ["work-1"]
    assert ("event-3", "event-4") in constants["causal_edges"]
    assert "event_kind event_1 accelerator_device_registered" in facts
    assert all("backend_kind" not in fact for fact in facts)


def test_torch_semantic_tracer_records_module_aten_autograd_and_work_lifecycle() -> None:
    tracer = TorchSemanticTracer(world_size=2, backend_kind="generic-accelerator")
    process_group = tracer.create_process_group("tp_group_dp0", ranks=(0, 1), axis="tp")

    tracer.begin_step(rank=0, step=11, mesh={"axes": ["dp", "tp"], "shape": [1, 2], "coord": [0, 0]})
    producer = tracer.record_aten_completed(
        rank=0,
        role="qwen3_mlp_down_projection",
        module_path="model.layers.0.mlp.down_proj",
        op="mm",
        correlation_id="aten-step11-layer0-rank0-mlp-down",
        tensor=TensorSemantic(
            placement="tp_sharded",
            mesh_axis="tp",
            shard_index=0,
            dtype="bf16",
            shape_digest="qwen3.hidden_shard",
        ),
    )
    work = tracer.enqueue_collective(
        rank=0,
        process_group=process_group,
        op="all_reduce",
        work_role="tp_mlp_down_reduce",
        producer_event=producer,
        tensor_bytes=2048,
    )
    started = tracer.start_work(rank=0, work=work)
    tracer.complete_work(rank=0, work=work, causal_predecessors=(started.event_id,))
    grad_ready = tracer.record_autograd_grad_ready(
        rank=0,
        role="qwen3_layer_gradient",
        node="Qwen3DecoderLayerBackward",
        grad_role="layer_param_grad",
        bucket="qwen3-layer0-tp0",
    )

    assert verify_trace(tracer.trace).ok is True
    assert producer.extensions["module"]["path"] == "model.layers.0.mlp.down_proj"
    assert producer.extensions["aten"]["dispatcher_key"] == "accelerator"
    assert producer.extensions["tensor"]["placement"] == "tp_sharded"
    assert grad_ready.extensions["autograd"]["bucket"] == "qwen3-layer0-tp0"

    enqueued = _work_event(tracer.trace, work.id, "collective.enqueued")
    assert enqueued.extensions["process_group_axis"] == "tp"
    assert enqueued.extensions["work_role"] == "tp_mlp_down_reduce"
    assert enqueued.extensions["correlation"] == {
        "producer_event": producer.event_id,
        "producer_aten": "aten-step11-layer0-rank0-mlp-down",
        "module": "model.layers.0.mlp.down_proj",
    }
    assert producer.event_id in enqueued.causal_predecessors


def test_torch_semantic_tracer_records_p2p_identity() -> None:
    tracer = TorchSemanticTracer(world_size=2, backend_kind="generic-accelerator")
    process_group = tracer.create_process_group("world", ranks=(0, 1))

    send = tracer.enqueue_p2p(
        rank=0,
        process_group=process_group,
        peer=1,
        direction="send",
        tensor_bytes=4096,
        is_batched=False,
        work_role="activation_send",
    )

    event = _work_event(tracer.trace, send.id, "p2p.enqueued")
    assert event.work.peer == 1
    assert event.work.direction == "send"
    assert event.work.is_batched is False
    assert event.extensions["work_role"] == "activation_send"
    assert verify_trace(tracer.trace).ok is True


def _work_event(trace, work_id: str, kind: str):
    for event in trace.events:
        if event.kind == kind and event.work is not None and event.work.id == work_id:
            return event
    raise AssertionError(f"missing {kind} event for work {work_id}")
