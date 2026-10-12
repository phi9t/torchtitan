from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schema import DeviceRef, ExecutorRef, ProcessGroupRef, Timing, Trace, TraceEvent, WorkRef
from .verifier import VerificationIssue, VerificationReport, verify_trace


@dataclass(frozen=True)
class _Mesh:
    dp_size: int
    tp_size: int

    @property
    def world_size(self) -> int:
        return self.dp_size * self.tp_size

    def rank(self, dp: int, tp: int) -> int:
        return dp * self.tp_size + tp

    def coord(self, rank: int) -> tuple[int, int]:
        return divmod(rank, self.tp_size)

    def tp_group(self, dp: int) -> ProcessGroupRef:
        return ProcessGroupRef(id=f"tp_group_dp{dp}", ranks=tuple(self.rank(dp, tp) for tp in range(self.tp_size)))

    def dp_group(self, tp: int) -> ProcessGroupRef:
        return ProcessGroupRef(id=f"dp_group_tp{tp}", ranks=tuple(self.rank(dp, tp) for dp in range(self.dp_size)))


class _TraceBuilder:
    def __init__(self, mesh: _Mesh, step: int) -> None:
        self.mesh = mesh
        self.step = step
        self.logical_time = 0
        self.event_sequence = 0
        self.work_sequence = 0
        self.events: list[TraceEvent] = []

    def emit(
        self,
        kind: str,
        rank: int,
        layer: int | None = None,
        process_group: ProcessGroupRef | None = None,
        work: WorkRef | None = None,
        executor: ExecutorRef | None = None,
        causal_predecessors: tuple[str, ...] = (),
        extensions: dict[str, object] | None = None,
    ) -> TraceEvent:
        self.logical_time += 1
        self.event_sequence += 1
        dp, tp = self.mesh.coord(rank)
        event_extensions: dict[str, object] = {
            "model": "qwen3_dense",
            "source": "torchtitan",
            "step": self.step,
            "mesh": {"axes": ["dp", "tp"], "shape": [self.mesh.dp_size, self.mesh.tp_size], "coord": [dp, tp]},
        }
        if layer is not None:
            event_extensions["layer"] = layer
        if process_group is not None:
            axis = "tp" if process_group.id.startswith("tp_group_") else "dp"
            event_extensions["process_group_axis"] = axis
        event_extensions.update(extensions or {})
        event = TraceEvent(
            event_id=f"e-{self.event_sequence:06d}",
            kind=kind,
            logical_time=self.logical_time,
            rank=rank,
            local_rank=rank,
            world_size=self.mesh.world_size,
            device=DeviceRef(kind="accelerator", index=rank, id=f"dev-r{rank}"),
            process_group=process_group,
            work=work,
            executor=executor,
            timing=Timing(clock_domain="logical"),
            causal_predecessors=causal_predecessors,
            extensions=event_extensions,
        )
        self.events.append(event)
        return event

    def new_work(self, step: int, layer: int, role: str, axis: str, index: int, op: str = "all_reduce") -> WorkRef:
        self.work_sequence += 1
        return WorkRef(id=f"w-step{step}-layer{layer}-{role}-{axis}{index}", sequence=self.work_sequence, op=op)

    def emit_torch_aten(
        self,
        rank: int,
        layer: int,
        role: str,
        module_path: str,
        op: str,
        tensor: dict[str, object],
        predecessor_kind: str | None = None,
    ) -> TraceEvent:
        predecessor = _last_event_id(self.events, rank, predecessor_kind) if predecessor_kind else None
        correlation_id = f"aten-step{self.step}-layer{layer}-rank{rank}-{_role_suffix(role)}"
        return self.emit(
            "torch.aten.completed",
            rank=rank,
            layer=layer,
            causal_predecessors=(predecessor,) if predecessor else (),
            extensions={
                "torch_role": role,
                "module": {"path": module_path, "role": role},
                "aten": {"op": op, "dispatcher_key": "accelerator", "correlation_id": correlation_id},
                "tensor": tensor,
            },
        )

    def emit_grad_ready(self, rank: int, layer: int) -> TraceEvent:
        _, tp = self.mesh.coord(rank)
        predecessor = _last_event_id(self.events, rank, "collective.completed")
        return self.emit(
            "torch.autograd.grad_ready",
            rank=rank,
            layer=layer,
            causal_predecessors=(predecessor,) if predecessor else (),
            extensions={
                "torch_role": "qwen3_layer_gradient",
                "autograd": {
                    "node": "Qwen3DecoderLayerBackward",
                    "grad_role": "layer_param_grad",
                    "bucket": f"qwen3-layer{layer}-tp{tp}",
                },
            },
        )


@dataclass(frozen=True)
class FormalArtifactPaths:
    tla: Path
    lean: Path
    readme: Path


def build_torchtitan_qwen3_step_trace(dp_size: int, tp_size: int, layers: int, step: int) -> Trace:
    """Build a deterministic TorchTitan-style Qwen3 dense DPxTP training-step trace."""

    if dp_size <= 0 or tp_size <= 0 or layers <= 0:
        raise ValueError("dp_size, tp_size, and layers must be positive")

    mesh = _Mesh(dp_size=dp_size, tp_size=tp_size)
    builder = _TraceBuilder(mesh=mesh, step=step)

    for rank in range(mesh.world_size):
        builder.emit("rank.step_begin", rank=rank)

    for layer in range(layers):
        for rank in range(mesh.world_size):
            local_compute = builder.emit(
                "qwen3.forward.local_compute_completed",
                rank=rank,
                layer=layer,
                extensions={"phase": "attention_qkv_rope_softmax_value"},
            )
            builder.emit_torch_aten(
                rank=rank,
                layer=layer,
                role="qwen3_attention_output_projection",
                module_path=f"model.layers.{layer}.self_attn.o_proj",
                op="mm",
                tensor=_tp_tensor(builder, rank),
                predecessor_kind=local_compute.kind,
            )
        for dp in range(dp_size):
            _emit_group_collective(
                builder,
                layer=layer,
                group=mesh.tp_group(dp),
                work=builder.new_work(step, layer, role="tp-attn", axis="dp", index=dp),
                role="tp_attention_output_reduce",
                shard=None,
                predecessor_role="qwen3_attention_output_projection",
            )

        for rank in range(mesh.world_size):
            mlp_compute = builder.emit(
                "qwen3.forward.mlp_local_completed",
                rank=rank,
                layer=layer,
                extensions={"phase": "mlp_gate_up_silu_down"},
            )
            builder.emit_torch_aten(
                rank=rank,
                layer=layer,
                role="qwen3_mlp_down_projection",
                module_path=f"model.layers.{layer}.mlp.down_proj",
                op="mm",
                tensor=_tp_tensor(builder, rank),
                predecessor_kind=mlp_compute.kind,
            )
        for dp in range(dp_size):
            _emit_group_collective(
                builder,
                layer=layer,
                group=mesh.tp_group(dp),
                work=builder.new_work(step, layer, role="tp-mlp", axis="dp", index=dp),
                role="tp_mlp_down_reduce",
                shard=None,
                predecessor_role="qwen3_mlp_down_projection",
            )

        for rank in range(mesh.world_size):
            builder.emit_grad_ready(rank=rank, layer=layer)
        for tp in range(tp_size):
            _emit_group_collective(
                builder,
                layer=layer,
                group=mesh.dp_group(tp),
                work=builder.new_work(step, layer, role="dp-grad", axis="tp", index=tp),
                role="dp_gradient_sync",
                shard={"axis": "tp", "index": tp},
                predecessor_role="qwen3_layer_gradient",
            )

    for rank in range(mesh.world_size):
        predecessors = _completed_work_events_for_rank(builder.events, rank)
        builder.emit("qwen3.optimizer.updated", rank=rank, causal_predecessors=predecessors)
        optimizer = builder.events[-1]
        builder.emit("rank.step_end", rank=rank, causal_predecessors=(optimizer.event_id,))

    return Trace.from_events(builder.events)


def verify_qwen3_step_trace(trace: Trace) -> VerificationReport:
    base_report = verify_trace(trace)
    issues = list(base_report.issues)
    issues.extend(_mesh_group_issues(trace))
    issues.extend(_dp_gradient_shard_issues(trace))
    issues.extend(_collective_torch_producer_issues(trace))
    issues.extend(_step_closure_issues(trace))
    return VerificationReport(ok=not issues, issues=tuple(issues))


def export_qwen3_step_tla(trace: Trace) -> str:
    ranks = sorted({event.rank for event in trace.events})
    dps = sorted({_coord(event)[0] for event in trace.events})
    tps = sorted({_coord(event)[1] for event in trace.events})
    groups = _process_groups(trace)
    works = _works(trace)
    dp_mapping = [(rank, _coord_for_rank(trace, rank)[0]) for rank in ranks]
    tp_mapping = [(rank, _coord_for_rank(trace, rank)[1]) for rank in ranks]
    shard_mapping = [(work.id, _work_shard_tp(trace, work.id)) for work in works if _work_shard_tp(trace, work.id) is not None]
    producer_mapping = [(work.id, _work_producer_aten(trace, work.id)) for work in works if _work_producer_aten(trace, work.id)]
    lines = [
        "------------------------------ MODULE Qwen3StepTrace ------------------------------",
        "EXTENDS Naturals, FiniteSets",
        "",
        "CONSTANTS Ranks, DPs, TPs, ProcessGroups, Works",
        "",
        f"Ranks == {{{', '.join(str(rank) for rank in ranks)}}}",
        f"DPs == {{{', '.join(str(dp) for dp in dps)}}}",
        f"TPs == {{{', '.join(str(tp) for tp in tps)}}}",
        f"ProcessGroups == {{{', '.join(_tla_string(group.id) for group in groups)}}}",
        f"Works == {{{', '.join(_tla_string(work.id) for work in works)}}}",
        "",
        "Axis == " + _tla_mapping((group.id, _group_axis(group)) for group in groups),
        "Members == " + _tla_mapping((group.id, "{" + ", ".join(str(rank) for rank in group.ranks) + "}") for group in groups),
        "CoordDP == " + _tla_mapping(dp_mapping),
        "CoordTP == " + _tla_mapping(tp_mapping),
        "Coord == [r \\in Ranks |-> [dp |-> CoordDP[r], tp |-> CoordTP[r]]]",
        "WorkGroup == " + _tla_mapping((work.id, _work_group_id(trace, work.id)) for work in works),
        "WorkRole == " + _tla_mapping((work.id, _work_role(trace, work.id)) for work in works),
        "ProducerAten == " + _tla_mapping(producer_mapping),
        "WorkShardTP == " + _tla_mapping(shard_mapping),
        "",
        "DPGroupShape ==",
        "  \\A pg \\in ProcessGroups :",
        "    Axis[pg] = \"DP\" => \\A r1, r2 \\in Members[pg] : Coord[r1].tp = Coord[r2].tp",
        "",
        "TPGroupShape ==",
        "  \\A pg \\in ProcessGroups :",
        "    Axis[pg] = \"TP\" => \\A r1, r2 \\in Members[pg] : Coord[r1].dp = Coord[r2].dp",
        "",
        "DPShardAgreement ==",
        "  \\A w \\in DOMAIN WorkShardTP :",
        "    \\A r \\in Members[WorkGroup[w]] : Coord[r].tp = WorkShardTP[w]",
        "",
        "=============================================================================",
    ]
    return "\n".join(lines) + "\n"


def export_qwen3_step_lean(trace: Trace) -> str:
    ranks = sorted({event.rank for event in trace.events})
    work_ids = [work.id for work in _works(trace)]
    torch_ops = sorted(
        str(event.extensions["aten"]["correlation_id"])
        for event in trace.events
        if event.kind == "torch.aten.completed" and isinstance(event.extensions.get("aten"), dict)
    )
    work_lines = ", ".join(_lean_string(work_id) for work_id in work_ids)
    torch_op_lines = ", ".join(_lean_string(torch_op) for torch_op in torch_ops)
    coord_lines = [f"| {rank} => {{ dp := {_coord_for_rank(trace, rank)[0]}, tp := {_coord_for_rank(trace, rank)[1]} }}" for rank in ranks]
    coord_lines.append("| _ => { dp := 0, tp := 0 }")
    return "\n".join(
        [
            "namespace Qwen3StepTrace",
            "",
            "structure Coord where",
            "  dp : Nat",
            "  tp : Nat",
            "deriving DecidableEq, Repr",
            "",
            "structure Trace where",
            "  ranks : List Nat",
            "  coord : Nat -> Coord",
            "  works : List String",
            "deriving Repr",
            "",
            f"def qwen3TraceRanks : List Nat := [{', '.join(str(rank) for rank in ranks)}]",
            "def qwen3TraceCoord : Nat -> Coord",
            *coord_lines,
            f"def qwen3TraceWorks : List String := [{work_lines}]",
            f"def qwen3TraceTorchOps : List String := [{torch_op_lines}]",
            "",
            "def sameTpCoord (tr : Trace) (rs : List Nat) : Prop :=",
            "  forall r1, List.Mem r1 rs -> forall r2, List.Mem r2 rs -> (tr.coord r1).tp = (tr.coord r2).tp",
            "",
            "def sameDpCoord (tr : Trace) (rs : List Nat) : Prop :=",
            "  forall r1, List.Mem r1 rs -> forall r2, List.Mem r2 rs -> (tr.coord r1).dp = (tr.coord r2).dp",
            "",
            "def meshWellFormed (_tr : Trace) : Prop :=",
            "  True",
            "",
            "def qwen3Trace : Trace :=",
            "  { ranks := qwen3TraceRanks, coord := qwen3TraceCoord, works := qwen3TraceWorks }",
            "",
            "end Qwen3StepTrace",
            "",
        ]
    )


def write_qwen3_step_formal_artifacts(trace: Trace, output_dir: str | Path) -> FormalArtifactPaths:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = FormalArtifactPaths(
        tla=output_path / "Qwen3StepTrace.tla",
        lean=output_path / "Qwen3StepTrace.lean",
        readme=output_path / "README.md",
    )
    paths.tla.write_text(export_qwen3_step_tla(trace))
    paths.lean.write_text(export_qwen3_step_lean(trace))
    paths.readme.write_text(_formal_readme())
    return paths


def _formal_readme() -> str:
    return """# Qwen3 Step Formal Trace

This directory contains generated formal artifacts for a TorchTitan-style Qwen3 dense DPxTP training step.

- `Qwen3StepTrace.tla` captures the finite DPxTP mesh, process groups, work ids, work roles, and group membership used by the TLA+ transition model.
- `Qwen3StepTrace.lean` captures the same finite trace facts in a Lean-friendly shape for trace-level invariant proofs.

The source of truth is the Python trace IR emitted by `build_torchtitan_qwen3_step_trace`; regenerate these files through `write_qwen3_step_formal_artifacts` after schema or role changes.
"""


def _emit_group_collective(
    builder: _TraceBuilder,
    layer: int,
    group: ProcessGroupRef,
    work: WorkRef,
    role: str,
    shard: dict[str, object] | None,
    predecessor_role: str,
) -> None:
    executor = ExecutorRef(kind="stream", id=f"{group.id}-collective")
    producer_by_rank: dict[int, TraceEvent | None] = {}
    enqueued_by_rank: dict[int, TraceEvent] = {}
    started_by_rank: dict[int, TraceEvent] = {}
    for rank in group.ranks:
        predecessor = _last_torch_role_event(builder.events, rank, predecessor_role)
        producer_by_rank[rank] = predecessor
        enqueued_by_rank[rank] = builder.emit(
            "collective.enqueued",
            rank=rank,
            layer=layer,
            process_group=group,
            work=work,
            executor=executor,
            causal_predecessors=(predecessor.event_id,) if predecessor else (),
            extensions=_work_extensions(role, shard, predecessor),
        )
    for rank in group.ranks:
        started_by_rank[rank] = builder.emit(
            "collective.started",
            rank=rank,
            layer=layer,
            process_group=group,
            work=work,
            executor=executor,
            causal_predecessors=(enqueued_by_rank[rank].event_id,),
            extensions=_work_extensions(role, shard, producer_by_rank[rank]),
        )
    for rank in group.ranks:
        builder.emit(
            "collective.completed",
            rank=rank,
            layer=layer,
            process_group=group,
            work=work,
            executor=executor,
            causal_predecessors=(started_by_rank[rank].event_id,),
            extensions=_work_extensions(role, shard, producer_by_rank[rank]),
        )


def _work_extensions(role: str, shard: dict[str, object] | None, producer: TraceEvent | None) -> dict[str, object]:
    extensions: dict[str, object] = {"work_role": role}
    if shard is not None:
        extensions["shard"] = shard
    if producer is not None:
        aten = producer.extensions.get("aten", {})
        autograd = producer.extensions.get("autograd", {})
        module = producer.extensions.get("module", {})
        producer_id = None
        if isinstance(aten, dict) and aten.get("correlation_id") is not None:
            producer_id = aten.get("correlation_id")
        elif isinstance(autograd, dict):
            producer_id = autograd.get("bucket")
        extensions["correlation"] = {
            "producer_event": producer.event_id,
            "producer_aten": producer_id,
            "module": module.get("path") if isinstance(module, dict) else None,
        }
    return extensions


def _mesh_group_issues(trace: Trace) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    for event in trace.events:
        if event.process_group is None:
            continue
        axis = event.extensions.get("process_group_axis")
        ranks = event.process_group.ranks
        if axis == "dp" and len({_coord_for_rank(trace, rank)[1] for rank in ranks}) != 1:
            issues.append(VerificationIssue("dp_group_shape", "DP group members must share one TP coordinate", event.event_id))
        if axis == "tp" and len({_coord_for_rank(trace, rank)[0] for rank in ranks}) != 1:
            issues.append(VerificationIssue("tp_group_shape", "TP group members must share one DP coordinate", event.event_id))
    return issues


def _dp_gradient_shard_issues(trace: Trace) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    for event in trace.events:
        if event.work is None or event.process_group is None:
            continue
        if event.extensions.get("work_role") != "dp_gradient_sync":
            continue
        shard = event.extensions.get("shard")
        if not isinstance(shard, dict) or shard.get("axis") != "tp":
            issues.append(
                VerificationIssue(
                    "dp_gradient_missing_tp_shard",
                    "DP gradient sync must identify the TP shard it reduces",
                    event.event_id,
                    event.work.id,
                )
            )
            continue
        expected_tp = int(shard["index"])
        for rank in event.process_group.ranks:
            if _coord_for_rank(trace, rank)[1] != expected_tp:
                issues.append(
                    VerificationIssue(
                        "dp_gradient_shard_mismatch",
                        "DP gradient sync members must match the TP shard index",
                        event.event_id,
                        event.work.id,
                    )
                )
                break
    return _dedupe_issues(issues)


def _collective_torch_producer_issues(trace: Trace) -> list[VerificationIssue]:
    event_ids = {event.event_id for event in trace.events}
    issues: list[VerificationIssue] = []
    for event in trace.events:
        if event.work is None or event.kind != "collective.enqueued":
            continue
        correlation = event.extensions.get("correlation")
        if not isinstance(correlation, dict) or not correlation.get("producer_event") or not correlation.get("producer_aten"):
            issues.append(
                VerificationIssue(
                    "collective_missing_torch_producer",
                    "collective work must correlate to the Torch event that produced its payload",
                    event.event_id,
                    event.work.id,
                )
            )
            continue
        producer_event = str(correlation["producer_event"])
        if producer_event not in event_ids or producer_event not in event.causal_predecessors:
            issues.append(
                VerificationIssue(
                    "collective_producer_not_causal",
                    "collective producer event must exist and be a causal predecessor",
                    event.event_id,
                    event.work.id,
                )
            )
    return _dedupe_issues(issues)


def _step_closure_issues(trace: Trace) -> list[VerificationIssue]:
    completed_by_rank: dict[int, set[str]] = {}
    required_by_rank: dict[int, set[str]] = {}
    step_end_events: list[TraceEvent] = []
    for event in trace.events:
        if event.work is not None and event.kind == "collective.completed":
            completed_by_rank.setdefault(event.rank, set()).add(event.work.id)
        if event.work is not None and event.extensions.get("work_role") in {
            "tp_attention_output_reduce",
            "tp_mlp_down_reduce",
            "dp_gradient_sync",
        }:
            required_by_rank.setdefault(event.rank, set()).add(event.work.id)
        if event.kind == "rank.step_end":
            step_end_events.append(event)

    issues: list[VerificationIssue] = []
    for event in step_end_events:
        missing = required_by_rank.get(event.rank, set()) - completed_by_rank.get(event.rank, set())
        for work_id in sorted(missing):
            issues.append(
                VerificationIssue(
                    "step_end_before_required_work",
                    f"rank {event.rank} ended the step before required work {work_id} completed",
                    event.event_id,
                    work_id,
                )
            )
    return issues


def _completed_work_events_for_rank(events: list[TraceEvent], rank: int) -> tuple[str, ...]:
    return tuple(event.event_id for event in events if event.rank == rank and event.kind == "collective.completed")


def _last_event_id(events: list[TraceEvent], rank: int, kind: str) -> str | None:
    for event in reversed(events):
        if event.rank == rank and event.kind == kind:
            return event.event_id
    return None


def _last_torch_role_event(events: list[TraceEvent], rank: int, role: str) -> TraceEvent | None:
    for event in reversed(events):
        if event.rank == rank and event.extensions.get("torch_role") == role:
            return event
    return None


def _tp_tensor(builder: _TraceBuilder, rank: int) -> dict[str, object]:
    _, tp = builder.mesh.coord(rank)
    return {
        "placement": "tp_sharded",
        "mesh_axis": "tp",
        "shard_index": tp,
        "dtype": "bf16",
        "shape_digest": "qwen3.hidden_shard",
    }


def _role_suffix(role: str) -> str:
    suffix_by_role = {
        "qwen3_attention_output_projection": "attn-out",
        "qwen3_mlp_down_projection": "mlp-down",
        "qwen3_layer_gradient": "grad-ready",
    }
    return suffix_by_role.get(role, role.replace("_", "-"))


def _coord(event: TraceEvent) -> tuple[int, int]:
    mesh = event.extensions.get("mesh")
    if not isinstance(mesh, dict):
        raise ValueError(f"event {event.event_id} has no mesh extension")
    coord = mesh.get("coord")
    if not isinstance(coord, list) or len(coord) != 2:
        raise ValueError(f"event {event.event_id} has invalid mesh coord")
    return int(coord[0]), int(coord[1])


def _coord_for_rank(trace: Trace, rank: int) -> tuple[int, int]:
    for event in trace.events:
        if event.rank == rank:
            return _coord(event)
    raise ValueError(f"rank {rank} has no trace event")


def _process_groups(trace: Trace) -> list[ProcessGroupRef]:
    groups: dict[str, ProcessGroupRef] = {}
    for event in trace.events:
        if event.process_group is not None:
            groups[event.process_group.id] = event.process_group
    return sorted(groups.values(), key=lambda group: (_group_axis(group) != "TP", group.id))


def _works(trace: Trace) -> list[WorkRef]:
    works: dict[str, WorkRef] = {}
    for event in trace.events:
        if event.work is not None:
            works[event.work.id] = event.work
    role_order = {
        "tp_attention_output_reduce": 0,
        "tp_mlp_down_reduce": 1,
        "dp_gradient_sync": 2,
    }
    return sorted(works.values(), key=lambda work: (role_order.get(_work_role(trace, work.id), 99), work.id))


def _group_axis(group: ProcessGroupRef) -> str:
    return "TP" if group.id.startswith("tp_group_") else "DP"


def _work_group_id(trace: Trace, work_id: str) -> str:
    for event in trace.events:
        if event.work is not None and event.work.id == work_id and event.process_group is not None:
            return event.process_group.id
    raise ValueError(f"work {work_id} has no process group")


def _work_role(trace: Trace, work_id: str) -> str:
    for event in trace.events:
        if event.work is not None and event.work.id == work_id:
            return str(event.extensions["work_role"])
    raise ValueError(f"work {work_id} has no role")


def _work_shard_tp(trace: Trace, work_id: str) -> int | None:
    for event in trace.events:
        if event.work is None or event.work.id != work_id:
            continue
        shard = event.extensions.get("shard")
        if isinstance(shard, dict) and shard.get("axis") == "tp":
            return int(shard["index"])
        return None
    return None


def _work_producer_aten(trace: Trace, work_id: str) -> str | None:
    for event in trace.events:
        if event.work is None or event.work.id != work_id:
            continue
        correlation = event.extensions.get("correlation")
        if isinstance(correlation, dict) and correlation.get("producer_aten") is not None:
            return str(correlation["producer_aten"])
        return None
    return None


def _tla_mapping(items) -> str:
    parts = [f"({_tla_atom(key)} :> {_tla_value(value)})" for key, value in items]
    return " @@ ".join(parts)


def _tla_string(value: str) -> str:
    return f'"{value}"'


def _tla_atom(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    return _tla_string(str(value))


def _tla_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    value = str(value)
    if value.startswith("{"):
        return value
    return _tla_string(value)


def _lean_string(value: str) -> str:
    return f'"{value}"'


def _dedupe_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    seen: set[tuple[str, str | None]] = set()
    deduped: list[VerificationIssue] = []
    for issue in issues:
        key = (issue.code, issue.work_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped
