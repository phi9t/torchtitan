from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .schema import DeviceRef, ExecutorRef, ProcessGroupRef, Timing, Trace, TraceEvent, WorkRef


@dataclass(frozen=True)
class TensorSemantic:
    placement: str
    dtype: str
    shape_digest: str
    mesh_axis: str | None = None
    shard_index: int | None = None

    def to_jsonable(self) -> dict[str, object]:
        data: dict[str, object] = {
            "placement": self.placement,
            "dtype": self.dtype,
            "shape_digest": self.shape_digest,
        }
        if self.mesh_axis is not None:
            data["mesh_axis"] = self.mesh_axis
        if self.shard_index is not None:
            data["shard_index"] = self.shard_index
        return data


@dataclass(frozen=True)
class _OpenWork:
    kind_prefix: str
    process_group: ProcessGroupRef
    executor: ExecutorRef
    device: DeviceRef
    producer_event: TraceEvent | None
    tensor_bytes: int | None
    extensions: dict[str, object]
    started_event_id: str | None = None


class TorchSemanticTracer:
    """Build backend-neutral trace events from Torch-level semantic hooks.

    The tracer intentionally does not import torch. Runtime integrations can
    call these methods from TorchDispatchMode, autograd hooks, or distributed
    wrappers while tests and formal exporters can run on a plain Python host.
    """

    def __init__(self, world_size: int, backend_kind: str = "generic-accelerator") -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.world_size = world_size
        self.backend_kind = backend_kind
        self._logical_time = 0
        self._event_sequence = 0
        self._work_sequence = 0
        self._events: list[TraceEvent] = []
        self._open_work: dict[str, _OpenWork] = {}
        self._process_group_axis: dict[str, str] = {}
        self._current_step: dict[int, int] = {}
        self._mesh_by_rank: dict[int, Mapping[str, object]] = {}

    @property
    def trace(self) -> Trace:
        return Trace.from_events(self._events)

    def create_process_group(
        self,
        id: str,
        ranks: tuple[int, ...],
        axis: str | None = None,
    ) -> ProcessGroupRef:
        if not ranks:
            raise ValueError("process group ranks must not be empty")
        if axis is not None:
            self._process_group_axis[id] = axis
        return ProcessGroupRef(id=id, ranks=tuple(ranks))

    def begin_step(self, rank: int, step: int, mesh: Mapping[str, object] | None = None) -> TraceEvent:
        self._current_step[rank] = step
        if mesh is not None:
            self._mesh_by_rank[rank] = dict(mesh)
        return self._emit("rank.step_begin", rank=rank, extensions={"step": step})

    def end_step(self, rank: int, causal_predecessors: tuple[str, ...] = ()) -> TraceEvent:
        return self._emit("rank.step_end", rank=rank, causal_predecessors=causal_predecessors)

    def record_aten_completed(
        self,
        rank: int,
        role: str,
        module_path: str,
        op: str,
        correlation_id: str,
        tensor: TensorSemantic,
        causal_predecessors: tuple[str, ...] = (),
        module_role: str | None = None,
        dispatcher_key: str = "accelerator",
    ) -> TraceEvent:
        return self._emit(
            "torch.aten.completed",
            rank=rank,
            causal_predecessors=causal_predecessors,
            extensions={
                "torch_role": role,
                "module": {"path": module_path, "role": module_role or role},
                "aten": {
                    "op": op,
                    "dispatcher_key": dispatcher_key,
                    "correlation_id": correlation_id,
                },
                "tensor": tensor.to_jsonable(),
            },
        )

    def record_autograd_grad_ready(
        self,
        rank: int,
        role: str,
        node: str,
        grad_role: str,
        bucket: str,
        causal_predecessors: tuple[str, ...] = (),
    ) -> TraceEvent:
        return self._emit(
            "torch.autograd.grad_ready",
            rank=rank,
            causal_predecessors=causal_predecessors,
            extensions={
                "torch_role": role,
                "autograd": {
                    "node": node,
                    "grad_role": grad_role,
                    "bucket": bucket,
                },
            },
        )

    def enqueue_collective(
        self,
        rank: int,
        process_group: ProcessGroupRef,
        op: str,
        work_role: str,
        producer_event: TraceEvent | None = None,
        tensor_bytes: int | None = None,
        executor: ExecutorRef | None = None,
    ) -> WorkRef:
        return self._enqueue_work(
            kind_prefix="collective",
            rank=rank,
            process_group=process_group,
            op=op,
            work_role=work_role,
            producer_event=producer_event,
            tensor_bytes=tensor_bytes,
            executor=executor,
        )

    def enqueue_p2p(
        self,
        rank: int,
        process_group: ProcessGroupRef,
        peer: int,
        direction: str,
        tensor_bytes: int,
        is_batched: bool,
        work_role: str,
        producer_event: TraceEvent | None = None,
        executor: ExecutorRef | None = None,
    ) -> WorkRef:
        if direction not in {"send", "recv"}:
            raise ValueError("direction must be send or recv")
        return self._enqueue_work(
            kind_prefix="p2p",
            rank=rank,
            process_group=process_group,
            op=direction,
            work_role=work_role,
            producer_event=producer_event,
            tensor_bytes=tensor_bytes,
            executor=executor,
            peer=peer,
            direction=direction,
            is_batched=is_batched,
        )

    def start_work(
        self,
        rank: int,
        work: WorkRef,
        causal_predecessors: tuple[str, ...] | None = None,
    ) -> TraceEvent:
        open_work = self._open_work[work.id]
        predecessors = causal_predecessors
        if predecessors is None:
            predecessors = tuple(
                event.event_id
                for event in reversed(self._events)
                if event.rank == rank and event.work is not None and event.work.id == work.id
            )[:1]
        started = self._emit(
            f"{open_work.kind_prefix}.started",
            rank=rank,
            process_group=open_work.process_group,
            work=work,
            executor=open_work.executor,
            device=open_work.device,
            causal_predecessors=predecessors,
            extensions=open_work.extensions,
        )
        self._open_work[work.id] = _OpenWork(
            kind_prefix=open_work.kind_prefix,
            process_group=open_work.process_group,
            executor=open_work.executor,
            device=open_work.device,
            producer_event=open_work.producer_event,
            tensor_bytes=open_work.tensor_bytes,
            extensions=open_work.extensions,
            started_event_id=started.event_id,
        )
        return started

    def complete_work(
        self,
        rank: int,
        work: WorkRef,
        causal_predecessors: tuple[str, ...] | None = None,
    ) -> TraceEvent:
        open_work = self._open_work.pop(work.id)
        if causal_predecessors is None:
            causal_predecessors = (open_work.started_event_id,) if open_work.started_event_id is not None else ()
        return self._emit(
            f"{open_work.kind_prefix}.completed",
            rank=rank,
            process_group=open_work.process_group,
            work=work,
            executor=open_work.executor,
            device=open_work.device,
            causal_predecessors=causal_predecessors,
            extensions=open_work.extensions,
        )

    def _enqueue_work(
        self,
        kind_prefix: str,
        rank: int,
        process_group: ProcessGroupRef,
        op: str,
        work_role: str,
        producer_event: TraceEvent | None,
        tensor_bytes: int | None,
        executor: ExecutorRef | None,
        peer: int | None = None,
        direction: str | None = None,
        is_batched: bool | None = None,
    ) -> WorkRef:
        self._work_sequence += 1
        work = WorkRef(
            id=f"work-{self._work_sequence}",
            sequence=self._work_sequence,
            op=op,
            peer=peer,
            direction=direction,
            is_batched=is_batched,
        )
        device = self._device(rank)
        effective_executor = executor or self._default_executor(rank)
        extensions = self._work_extensions(process_group, work_role, producer_event, tensor_bytes)
        enqueued = self._emit(
            f"{kind_prefix}.enqueued",
            rank=rank,
            process_group=process_group,
            work=work,
            executor=effective_executor,
            device=device,
            causal_predecessors=(producer_event.event_id,) if producer_event is not None else (),
            extensions=extensions,
        )
        self._open_work[work.id] = _OpenWork(
            kind_prefix=kind_prefix,
            process_group=process_group,
            executor=effective_executor,
            device=device,
            producer_event=producer_event,
            tensor_bytes=tensor_bytes,
            extensions=extensions,
            started_event_id=enqueued.event_id,
        )
        return work

    def _emit(
        self,
        kind: str,
        rank: int,
        process_group: ProcessGroupRef | None = None,
        work: WorkRef | None = None,
        executor: ExecutorRef | None = None,
        device: DeviceRef | None = None,
        timing: Timing | None = None,
        causal_predecessors: tuple[str, ...] = (),
        extensions: dict[str, object] | None = None,
    ) -> TraceEvent:
        self._event_sequence += 1
        self._logical_time += 1
        event_extensions: dict[str, object] = {"backend_kind": self.backend_kind}
        if rank in self._current_step:
            event_extensions["step"] = self._current_step[rank]
        if rank in self._mesh_by_rank:
            event_extensions["mesh"] = dict(self._mesh_by_rank[rank])
        if process_group is not None and process_group.id in self._process_group_axis:
            event_extensions["process_group_axis"] = self._process_group_axis[process_group.id]
        event_extensions.update(extensions or {})
        event = TraceEvent(
            event_id=f"event-{self._event_sequence}",
            kind=kind,
            logical_time=self._logical_time,
            rank=rank,
            local_rank=rank,
            world_size=self.world_size,
            device=device or self._device(rank),
            process_group=process_group,
            work=work,
            executor=executor,
            timing=timing or Timing(clock_domain="logical"),
            causal_predecessors=causal_predecessors,
            extensions=event_extensions,
        )
        self._events.append(event)
        return event

    def _work_extensions(
        self,
        process_group: ProcessGroupRef,
        work_role: str,
        producer_event: TraceEvent | None,
        tensor_bytes: int | None,
    ) -> dict[str, object]:
        extensions: dict[str, object] = {"work_role": work_role}
        if tensor_bytes is not None:
            extensions["tensor_bytes"] = tensor_bytes
        axis = self._process_group_axis.get(process_group.id)
        if axis is not None:
            extensions["process_group_axis"] = axis
        if producer_event is not None:
            extensions["correlation"] = _producer_correlation(producer_event)
        return extensions

    @staticmethod
    def _device(rank: int) -> DeviceRef:
        return DeviceRef(kind="accelerator", index=rank, id=f"dev-r{rank}")

    @staticmethod
    def _default_executor(rank: int) -> ExecutorRef:
        return ExecutorRef(kind="stream", id=f"rank-{rank}-default")


def _producer_correlation(producer_event: TraceEvent) -> dict[str, object]:
    aten = producer_event.extensions.get("aten", {})
    autograd = producer_event.extensions.get("autograd", {})
    module = producer_event.extensions.get("module", {})
    producer_id = None
    if isinstance(aten, dict) and aten.get("correlation_id") is not None:
        producer_id = aten["correlation_id"]
    elif isinstance(autograd, dict) and autograd.get("bucket") is not None:
        producer_id = autograd["bucket"]
    return {
        "producer_event": producer_event.event_id,
        "producer_aten": producer_id,
        "module": module.get("path") if isinstance(module, dict) else None,
    }
