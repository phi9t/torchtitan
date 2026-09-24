from __future__ import annotations

from dataclasses import dataclass

from .schema import DeviceRef, ExecutorRef, ProcessGroupRef, Timing, Trace, TraceEvent, WorkRef


@dataclass(frozen=True)
class _OpenWork:
    work: WorkRef
    kind_prefix: str
    event_id: str
    rank: int
    local_rank: int
    process_group: ProcessGroupRef
    device: DeviceRef
    executor: ExecutorRef
    tensor_bytes: int


class LogicalAcceleratorEmulator:
    """Deterministic CPU-only accelerator observability emulator."""

    def __init__(self, world_size: int, device_count: int, backend_kind: str = "cpu-emulated") -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if device_count <= 0:
            raise ValueError("device_count must be positive")
        self.world_size = world_size
        self.device_count = device_count
        self.backend_kind = backend_kind
        self._logical_time = 0
        self._event_sequence = 0
        self._work_sequence = 0
        self._events: list[TraceEvent] = []
        self._open_work: dict[str, _OpenWork] = {}
        self._default_process_group = ProcessGroupRef(id="world", ranks=tuple(range(world_size)))

        for device_index in range(device_count):
            device = self._device(device_index)
            executor = self._default_executor(device_index)
            self._emit("accelerator.device.registered", rank=0, device=device)
            self._emit("accelerator.stream.created", rank=0, device=device, executor=executor)

    @property
    def trace(self) -> Trace:
        return Trace.from_events(self._events)

    def create_process_group(self, id: str, ranks: tuple[int, ...]) -> ProcessGroupRef:
        if not ranks:
            raise ValueError("process group ranks must not be empty")
        return ProcessGroupRef(id=id, ranks=tuple(ranks))

    def begin_step(self, rank: int, step: int, device_index: int = 0) -> None:
        self._emit(
            "rank.step_begin",
            rank=rank,
            device=self._device(device_index),
            extensions={"step": step},
        )

    def end_step(self, rank: int, step: int, device_index: int = 0) -> None:
        self._emit(
            "rank.step_end",
            rank=rank,
            device=self._device(device_index),
            extensions={"step": step},
        )

    def enqueue_collective(
        self,
        rank: int,
        process_group: ProcessGroupRef | None,
        op: str,
        tensor_bytes: int,
        device_index: int = 0,
    ) -> WorkRef:
        pg = process_group or self._default_process_group
        return self._enqueue_work(
            kind_prefix="collective",
            rank=rank,
            process_group=pg,
            op=op,
            tensor_bytes=tensor_bytes,
            device_index=device_index,
        )

    def enqueue_p2p(
        self,
        rank: int,
        process_group: ProcessGroupRef | None,
        peer: int,
        direction: str,
        tensor_bytes: int,
        is_batched: bool,
        device_index: int = 0,
    ) -> WorkRef:
        if direction not in {"send", "recv"}:
            raise ValueError("direction must be send or recv")
        pg = process_group or self._default_process_group
        return self._enqueue_work(
            kind_prefix="p2p",
            rank=rank,
            process_group=pg,
            op=direction,
            tensor_bytes=tensor_bytes,
            device_index=device_index,
            peer=peer,
            direction=direction,
            is_batched=is_batched,
        )

    def complete_work(self, rank: int, work: WorkRef) -> None:
        open_work = self._open_work.pop(work.id)
        self._emit(
            f"{open_work.kind_prefix}.completed",
            rank=rank,
            device=open_work.device,
            process_group=open_work.process_group,
            work=open_work.work,
            executor=open_work.executor,
            timing=Timing(
                clock_domain="logical",
                start=float(open_work.event_id.rsplit("-", 1)[-1]),
                end=float(self._event_sequence + 1),
            ),
            causal_predecessors=(open_work.event_id,),
            extensions={"tensor_bytes": open_work.tensor_bytes},
        )

    def _enqueue_work(
        self,
        kind_prefix: str,
        rank: int,
        process_group: ProcessGroupRef,
        op: str,
        tensor_bytes: int,
        device_index: int,
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
        device = self._device(device_index)
        executor = self._default_executor(device_index)
        enqueued = self._emit(
            f"{kind_prefix}.enqueued",
            rank=rank,
            device=device,
            process_group=process_group,
            work=work,
            executor=executor,
            extensions={"tensor_bytes": tensor_bytes},
        )
        started = self._emit(
            f"{kind_prefix}.started",
            rank=rank,
            device=device,
            process_group=process_group,
            work=work,
            executor=executor,
            causal_predecessors=(enqueued.event_id,),
            extensions={"tensor_bytes": tensor_bytes},
        )
        self._open_work[work.id] = _OpenWork(
            work=work,
            kind_prefix=kind_prefix,
            event_id=started.event_id,
            rank=rank,
            local_rank=rank,
            process_group=process_group,
            device=device,
            executor=executor,
            tensor_bytes=tensor_bytes,
        )
        return work

    def _emit(
        self,
        kind: str,
        rank: int,
        device: DeviceRef,
        process_group: ProcessGroupRef | None = None,
        work: WorkRef | None = None,
        executor: ExecutorRef | None = None,
        timing: Timing | None = None,
        causal_predecessors: tuple[str, ...] = (),
        extensions: dict[str, object] | None = None,
    ) -> TraceEvent:
        self._event_sequence += 1
        self._logical_time += 1
        event = TraceEvent(
            event_id=f"event-{self._event_sequence}",
            kind=kind,
            logical_time=self._logical_time,
            rank=rank,
            local_rank=rank,
            world_size=self.world_size,
            device=device,
            process_group=process_group,
            work=work,
            executor=executor,
            timing=timing or Timing(clock_domain="logical"),
            causal_predecessors=causal_predecessors,
            extensions={"backend_kind": self.backend_kind, **(extensions or {})},
        )
        self._events.append(event)
        return event

    @staticmethod
    def _device(index: int) -> DeviceRef:
        return DeviceRef(kind="accelerator", index=index, id=f"dev-{index}")

    @staticmethod
    def _default_executor(device_index: int) -> ExecutorRef:
        return ExecutorRef(kind="stream", id=f"stream-{device_index}-default")
