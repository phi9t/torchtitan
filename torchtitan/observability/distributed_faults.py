# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Classify a failed collective into a v1 fault class, or decline to.

Two of the nine `IncidentClass` values describe distributed failures that a
surviving rank can only observe indirectly: `COLLECTIVE_HANG` and `RANK_DEATH`.
Both surface from Gloo as a plain `RuntimeError`, so the class has to come from
the message, and the two are distinguishable there. Measured on four ranks with a
four-second timeout:

  rank stays alive and never issues   -> "Timed out waiting 4000ms for recv
                                         operation to complete"
  rank exits and closes its sockets   -> "Connection closed by peer",
                                         "Read error"

Scope, stated because a signature table is only as wide as it was measured:

  - These signatures are **Gloo's**. NCCL reports timeouts and aborts through a
    different path with different text, and is NOT covered. A NCCL failure will
    fall through to "unclassified" rather than be guessed at.
  - An unrecognized error returns no class. The whole point of the fallback is
    that a wrong class sends a reader to the wrong subsystem, which is worse
    than an honest "unclassified": the raised error is still the primary
    finding either way.
  - A hang observed this way says a peer did not arrive. It does not say which
    peer, or why. `FaultConfidence.COLLECTIVE_TIMEOUT_INSUFFICIENT_EVIDENCE`
    exists for exactly that ambiguity and is what gets recorded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torchtitan.observability.run_evidence import (
    FaultAttributionLocus,
    FaultConfidence,
    IncidentCaptureState,
    IncidentClass,
    IncidentPolicy,
    record_incident,
)

# (signature, class, confidence). Order matters only in that the first match
# wins; the two groups are disjoint in every case measured.
GLOO_FAULT_SIGNATURES: tuple[tuple[str, IncidentClass, FaultConfidence], ...] = (
    (
        "timed out waiting",
        IncidentClass.COLLECTIVE_HANG,
        FaultConfidence.COLLECTIVE_TIMEOUT_INSUFFICIENT_EVIDENCE,
    ),
    (
        "connection closed by peer",
        IncidentClass.RANK_DEATH,
        FaultConfidence.SUSPECTED_PEER_FAULT,
    ),
    (
        "read error",
        IncidentClass.RANK_DEATH,
        FaultConfidence.SUSPECTED_PEER_FAULT,
    ),
)

UNCLASSIFIED = "unclassified"


def classify_collective_failure(
    error: BaseException,
) -> tuple[IncidentClass | None, FaultConfidence | None, str]:
    """Return the fault class, confidence, and the signature that decided it.

    The third element is the evidence for the decision and is recorded with the
    incident, so a reader can tell what the classification rested on instead of
    trusting it. It is ``UNCLASSIFIED`` when nothing matched.
    """
    text = str(error).lower()
    for signature, incident_class, confidence in GLOO_FAULT_SIGNATURES:
        if signature in text:
            return incident_class, confidence, signature
    return None, None, UNCLASSIFIED


def record_collective_failure(
    error: BaseException,
    *,
    step: int | None = None,
    last_operation: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Record a failed collective as a typed incident when it can be classified.

    Returns the recorded row, or None when the error is unclassified or no
    recorder is active. Declining to classify is a normal outcome, not an error:
    the caller is expected to let the original exception propagate regardless,
    because the raised error is the primary finding and a missing incident must
    never replace it.
    """
    incident_class, confidence, signature = classify_collective_failure(error)
    if incident_class is None:
        return None
    detail: dict[str, Any] = {
        "matched_signature": signature,
        "backend": "gloo",
        "error_type": type(error).__name__,
    }
    if metadata:
        detail.update(dict(metadata))
    return record_incident(
        incident_class=incident_class,
        capture_state=IncidentCaptureState.ABORT_AND_PRESERVE,
        policy=IncidentPolicy.ABORT_FATAL,
        summary=f"collective failed: {str(error).splitlines()[0][:200]}",
        detected_locus=FaultAttributionLocus.TRAINER_RANK,
        attribution_confidence=confidence,
        step=step,
        last_operation=last_operation,
        useful_work_preserved=False,
        terminal_disposition="abort",
        metadata=detail,
    )
