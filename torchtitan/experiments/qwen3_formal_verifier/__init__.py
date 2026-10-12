"""Generic accelerator observability and Qwen3 DPxTP formal verifier."""

from .emulator import LogicalAcceleratorEmulator
from .schema import DeviceRef, ExecutorRef, ProcessGroupRef, Timing, Trace, TraceEvent, WorkRef
from .tracer import TensorSemantic, TorchSemanticTracer
from .verifier import VerificationIssue, VerificationReport, verify_trace

__all__ = [
    "DeviceRef",
    "ExecutorRef",
    "LogicalAcceleratorEmulator",
    "ProcessGroupRef",
    "TensorSemantic",
    "Timing",
    "TorchSemanticTracer",
    "Trace",
    "TraceEvent",
    "VerificationIssue",
    "VerificationReport",
    "WorkRef",
    "verify_trace",
]
