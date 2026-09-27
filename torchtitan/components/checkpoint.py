# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import os
import queue
import re
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, Literal, NoReturn

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)

# Private, and imported at module scope on purpose: this is the exact reader
# resolution dcp.load performs for a bare checkpoint_id, so the manifest the
# sentinel validates is the manifest the subsequent load will read. Importing it
# here makes a future torch reorganisation fail loudly at import time instead of
# silently at checkpoint-load time. checkpoint.py already depends on
# _consolidate_hf_safetensors above for the same class of reason.
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.run_evidence import (
    ArtifactRelation,
    ArtifactState,
    FaultAttributionLocus,
    FaultConfidence,
    IncidentCaptureState,
    IncidentClass,
    IncidentPolicy,
    record_artifact,
    record_incident,
)
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools import filesystem
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"

# The DCP manifest file name. Already the repository's signal for "this
# directory holds a DCP checkpoint" (see _find_load_step).
DCP_MANIFEST_NAME = ".metadata"

# Result tokens for the checkpoint-manifest-integrity sentinel. They go into the
# incident metadata and into the raised error, so a reader of an evidence bundle
# can tell which check fired without parsing prose.
MANIFEST_MISSING = "missing_manifest"
MANIFEST_UNREADABLE = "unreadable_manifest"
SHARD_MISSING = "missing_shard"
SHARD_TRUNCATED = "truncated_shard"


class CheckpointCorruptionError(ValueError):
    """A checkpoint's DCP manifest does not describe the files on disk.

    A ValueError subclass because the checkpoint path is user input and load()
    already reports an invalid user-supplied checkpoint path that way. The
    dedicated type exists so callers and tests can pin this failure by type
    rather than by message text.
    """


def _shard_byte_extents(storage_data: Any) -> dict[str, int] | None:
    """Map each shard file the manifest references to its required end offset.

    DCP's filesystem writer stores ``Metadata.storage_data`` as
    ``{MetadataIndex: _StorageInfo}``, where ``_StorageInfo`` carries
    ``relative_path``, a byte ``offset`` into that file, and a byte ``length``.
    The largest ``offset + length`` targeting a file is the smallest size that
    file can have and still hold everything the manifest promises, which is what
    makes truncation detectable from the manifest alone.

    ``Metadata.storage_data`` is typed ``Any`` because it is writer-private, so
    the fields are read by duck typing rather than by importing ``_StorageInfo``.
    Returns None when the layout is not recognized: a writer that stores
    something else must make the sentinel skip, not make it report a defect that
    is not there.
    """
    if not isinstance(storage_data, Mapping) or not storage_data:
        return None
    extents: dict[str, int] = {}
    for info in storage_data.values():
        relative_path = getattr(info, "relative_path", None)
        offset = getattr(info, "offset", None)
        length = getattr(info, "length", None)
        if not isinstance(relative_path, (str, os.PathLike)):
            return None
        # bool is an int subclass, but a bool offset or length would itself be a
        # malformed manifest, and treating it as 0/1 here would understate the
        # required size. Rejecting the layout is the conservative outcome.
        if type(offset) is not int or type(length) is not int:
            return None
        key = os.fspath(relative_path)
        extents[key] = max(extents.get(key, 0), offset + length)
    return extents


def _read_dcp_manifest(checkpoint_id: str, *, manifest_path: str) -> tuple[Any, Any]:
    """Read a DCP manifest and return its state_dict_metadata and storage_data.

    Resolves the reader exactly as dcp.load does for a bare checkpoint_id, so
    the manifest validated here is the manifest the subsequent load will read.

    Raises:
        CheckpointCorruptionError: if the manifest cannot be deserialized.
    """
    try:
        reader = cast(Any, _storage_setup(None, checkpoint_id, reader=True))
        manifest = reader.read_metadata()
        return manifest.state_dict_metadata, manifest.storage_data
    except Exception as error:
        # Deliberately broad. The manifest is the object in doubt, so the whole
        # range of deserialization failures is the defect being looked for:
        # UnpicklingError, EOFError, a truncated read, an AttributeError from an
        # object that is not a Metadata, or a ModuleNotFoundError from a pickle
        # naming a class this build does not have. Narrowing the except clause
        # would let a corrupt manifest escape as an opaque crash, which is
        # exactly what this sentinel exists to prevent.
        _report_checkpoint_corruption(
            [
                {
                    "defect": MANIFEST_UNREADABLE,
                    "path": manifest_path,
                    "detail": (
                        "the DCP manifest could not be read: "
                        f"{type(error).__name__}: {error}"
                    ),
                }
            ],
            checkpoint_id=checkpoint_id,
            cause=error,
        )


def _report_checkpoint_corruption(
    defects: Sequence[Mapping[str, Any]],
    *,
    checkpoint_id: str,
    cause: BaseException | None = None,
) -> NoReturn:
    """Record a typed CHECKPOINT_CORRUPTION incident, then raise.

    The incident is emitted through the module-level record_incident facade
    rather than through a RunEvidence instance. That is safe here and was not
    safe for the inconsistent-rank-config detector: this runs inside the
    trainer's lifecycle, well after RunEvidence.__enter__ has installed the
    active recorder, whereas that detector runs during __enter__ itself and had
    to bypass the facade to avoid being silently dropped. The durability of this
    row is asserted by reading it back off disk in
    test_truncated_shard_aborts_manager_load_before_dcp_load.

    No step is passed: dcp_load does not receive one, and the checkpoint_id in
    the metadata already names the step folder that was being loaded. The
    surrounding checkpoint artifact rows carry the numeric step for the join.
    """
    summary_parts: list[str] = []
    for defect in defects:
        part = f"{defect['defect']} at {defect['path']}"
        detail = defect.get("detail")
        if detail:
            part = f"{part} ({detail})"
        summary_parts.append(part)
    summary = "; ".join(summary_parts)
    record_incident(
        incident_class=IncidentClass.CHECKPOINT_CORRUPTION,
        # The sentinel runs before dcp.load touches any training state, so the
        # capture is complete at the moment of detection and the checkpoint on
        # disk is left exactly as found for post-hoc inspection.
        capture_state=IncidentCaptureState.ABORT_AND_PRESERVE,
        # ABORT_FATAL rather than CAPTURE_BEFORE_ABORT: there is no separate
        # capture step to run before aborting, and rather than
        # CONTINUE_BOUNDED_WARNING because loading a checkpoint whose manifest
        # does not describe its files would either fail inside the reader or
        # silently install partial state. That is the silent-correctness class
        # the repository requires to abort.
        policy=IncidentPolicy.ABORT_FATAL,
        summary=f"checkpoint manifest integrity failed for {checkpoint_id}: {summary}",
        detected_locus=FaultAttributionLocus.CHECKPOINT_PATH,
        # This process read the manifest and stat'ed the files itself; the defect
        # is a directly observed local fact, not something inferred from a peer.
        attribution_confidence=FaultConfidence.OBSERVED_LOCAL_FAULT,
        last_operation="checkpoint.validate_dcp_manifest_integrity",
        useful_work_preserved=False,
        terminal_disposition="abort",
        metadata={
            "checkpoint_id": str(checkpoint_id),
            "defect_kinds": sorted({str(defect["defect"]) for defect in defects}),
            "defects": [dict(defect) for defect in defects],
        },
    )
    error = CheckpointCorruptionError(
        f"checkpoint at {checkpoint_id} failed manifest integrity validation "
        f"before load: {summary}"
    )
    if cause is not None:
        raise error from cause
    raise error


def validate_dcp_manifest_integrity(checkpoint_id: str) -> None:
    """Validate a DCP checkpoint's manifest against the files on disk.

    This is the "checkpoint manifest integrity" sentinel of the minimum
    correctness set in docs/robust_training_reliability.md. It runs before
    dcp.load so a damaged checkpoint aborts with a named defect and a typed
    CHECKPOINT_CORRUPTION incident instead of whatever the reader happens to
    raise from inside a worker thread.

    Four defects are detected, each named by the token it reports:

    1. ``missing_manifest`` -- ``.metadata`` is absent from the checkpoint.
    2. ``unreadable_manifest`` -- ``.metadata`` is present but does not
       deserialize, or deserializes into something carrying no
       ``state_dict_metadata`` mapping.
    3. ``missing_shard`` -- the manifest references a shard file that is not
       there.
    4. ``truncated_shard`` -- a referenced shard file is shorter than the
       largest ``offset + length`` the manifest assigns to it.

    What this sentinel does NOT detect, stated rather than implied:

    - Bit flips or any other corruption inside shard data. The manifest records
      byte ranges, not checksums, so a shard of the right length with wrong
      contents passes. Detecting that needs per-shard digests DCP does not
      write.
    - A shard that is longer than required. Trailing bytes past the last
      recorded range are not read, so extra length is not evidence of a defect.
    - Anything in a HuggingFace safetensors checkpoint. That path uses a
      different storage reader and a ``model.safetensors.index.json`` manifest;
      it is not validated here, and dcp_load warns when it takes that path.
    - Corruption of the manifest that still deserializes into a well-formed
      Metadata whose ranges happen to fit the files present.

    Remote (fsspec) checkpoints are validated, not skipped: the cost is one
    isfile plus one size probe per referenced shard, which for DCP is one pair
    per writing rank, paid once per load. When a backend declines to report a
    size, truncation detection for that file is skipped with a warning rather
    than guessed.

    Raises:
        CheckpointCorruptionError: if any defect above is found.
    """
    manifest_path = filesystem.join(checkpoint_id, DCP_MANIFEST_NAME)
    if not filesystem.isfile(manifest_path):
        _report_checkpoint_corruption(
            [
                {
                    "defect": MANIFEST_MISSING,
                    "path": manifest_path,
                    "detail": "the DCP manifest is absent from the checkpoint",
                }
            ],
            checkpoint_id=checkpoint_id,
        )

    state_dict_metadata, storage_data = _read_dcp_manifest(
        checkpoint_id, manifest_path=manifest_path
    )

    if not isinstance(state_dict_metadata, Mapping):
        _report_checkpoint_corruption(
            [
                {
                    "defect": MANIFEST_UNREADABLE,
                    "path": manifest_path,
                    "detail": (
                        "the DCP manifest carries no state_dict_metadata mapping "
                        f"(found {type(state_dict_metadata).__name__})"
                    ),
                }
            ],
            checkpoint_id=checkpoint_id,
        )

    extents = _shard_byte_extents(storage_data)
    if extents is None:
        logger.warning(
            "Checkpoint manifest integrity: shard existence and length checks "
            f"are SKIPPED for {checkpoint_id}. Its manifest's storage_data is "
            "not the {MetadataIndex: _StorageInfo} mapping written by DCP's "
            "filesystem writer, so no shard file could be identified. A missing "
            "or truncated shard in this checkpoint is NOT detected."
        )
        return

    defects: list[dict[str, Any]] = []
    for relative_path, required_bytes in sorted(extents.items()):
        shard_path = filesystem.join(checkpoint_id, relative_path)
        if not filesystem.isfile(shard_path):
            defects.append(
                {
                    "defect": SHARD_MISSING,
                    "path": shard_path,
                    "required_bytes": required_bytes,
                    "detail": "referenced by the manifest but not present",
                }
            )
            continue
        try:
            actual_bytes = filesystem.getsize(shard_path)
        except OSError as error:
            # A backend that cannot report a size makes truncation undetectable
            # for this file. Warn rather than guess: an unknown size is not
            # evidence of a defect, and treating it as zero would abort a
            # healthy load.
            logger.warning(
                "Checkpoint manifest integrity: length check SKIPPED for "
                f"{shard_path} because its size could not be read "
                f"({type(error).__name__}: {error}). Truncation of this shard "
                "is NOT detected."
            )
            continue
        if actual_bytes < required_bytes:
            defects.append(
                {
                    "defect": SHARD_TRUNCATED,
                    "path": shard_path,
                    "required_bytes": required_bytes,
                    "actual_bytes": actual_bytes,
                    "detail": (
                        f"{required_bytes - actual_bytes} bytes short of the "
                        "manifest's largest recorded byte range"
                    ),
                }
            )

    if defects:
        _report_checkpoint_corruption(defects, checkpoint_id=checkpoint_id)


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` are backed by the same storage.

    For ``DTensor`` the local shard's storage is compared via ``_local_tensor``
    rather than ``to_local()``, which is autograd-aware; this is a read-only
    identity check on the local storage.
    """
    if isinstance(a, DTensor):
        a = a._local_tensor
    if isinstance(b, DTensor):
        b = b._local_tensor
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


class ModelWrapper(Stateful):
    """
    A wrapper for `nn.Module` (or a list of modules) that provides a unified `Stateful`
    interface for distributed checkpointing.

    This class serves two purposes:
        1. Flattening/Aggregation: It combines the state dicts of multiple
           different modules (like individual chunks in Pipeline Parallelism)
           into a single flat view so checkpointing code can interact
           with them through a unified interface.
        2. Stable-storage caching: It caches the flattened state dict and, on
           every `state_dict()` call, returns tensors backed by the same
           storage. Async DCP staging may cache pinned host buffers keyed by the
           source storage, so keeping the storage stable lets it reuse those
           buffers across saves (the fast checkpoint path). Parameter tensors
           already satisfy this because the cached view shares the parameter
           storage; tensors produced by module `state_dict` hooks (e.g. one that
           splits a fused parameter) may be freshly allocated each call, so they
           are refreshed in place to keep their storage stable while their values
           track the current parameters.

    Notes:
        - Calling `load_state_dict` updates the underlying modules and
        refreshes the cached state_dict.
        - The model architecture should not be structurally modified (e.g.,
        changing keys or replacing tensor references) after wrapping, or the
        cache will become stale.
    """

    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cached_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        # TorchTitan already makes model state_dict keys canonical.
        return {k: v for model in self.model for k, v in model.state_dict().items()}

    def state_dict(self) -> dict[str, Any]:
        # Recompute the state dict so hook-produced tensors reflect the current
        # parameters, then merge into the cache without changing storage objects.
        for key, value in self._get_state_dict().items():
            cached = self.cached_state_dict.get(key)
            if (
                cached is None
                or cached.shape != value.shape
                or cached.dtype != value.dtype
            ):
                self.cached_state_dict[key] = value
            elif not _shares_storage(cached, value):
                cached.copy_(value)
        return self.cached_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # strict=False because state_dict is the flattened checkpoint dict, which
        # mixes model FQN keys with non-model keys (optimizer, lr_scheduler, ...).
        for model in self.model:
            model.load_state_dict(state_dict, strict=False)
        # Refresh the cache so state_dict() reflects the freshly loaded values.
        self.cached_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and
        Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            # A single failed deletion (e.g. a transient remote error) must not
            # kill this daemon thread; otherwise keep_latest_k would silently
            # stop purging for the rest of the run.
            try:
                filesystem.rmtree(path)
            except Exception as e:
                logger.warning(
                    "Checkpointer failed to delete %s: %s. Skipping.", path, e
                )
                continue
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager(Configurable):
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the
    original model. rank1's would _also_ have a param_group[0], since it's index based,
    but referring to layers.1. When saving, these collide and one of them is lost.
    Then when reloading, only one stage can restore its optimizer states, others will
    error.

        The solution to this problem is optimizer flattening.
        TorchTitan's OptimizersContainer flattens optimizer state dicts to FQN-keyed
        flat dicts using the utilities in torchtitan/components/checkpoint_utils.py.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This
    compounds challenge (1) by also requiring us to reason about multiple 'optim'
    objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state
        dicts from each object into one state dict before saving/loading. We rely on the
        individual state_dicts to not collide, which is guaranteed for the model by
        correct pipeline splitting and for the optimizer by the flattening support
        described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the
    lr_schedulers with the assumption that all lr_schedulers have the same state_dict.

    Args:
        config (Checkpoint): The config used to configure the checkpointing.
        dataloader (BaseDataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize
            the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        sd_adapter (Optional[type[BaseStateDictAdapter]]): The adapter used to convert
            model state dicts between native format and other formats.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with config.folder

    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to enable checkpoint"""

        folder: str = "checkpoint"
        """
        The folder to store the checkpoints.
        When enable is set to true, checkpoints will be in
        {--dump_folder}/{--checkpoint.folder}.
        """

        interval: int = 500
        """Checkpointing interval in steps."""

        initial_load_path: str | None = None
        """
        This option specifies the path to the initial checkpoint to load, which is
        particularly useful for resuming training from a previous run with a
        different output path or when loading a checkpoint from a pre-trained model.
        If the checkpoint folder for the current run is not empty,
        located at {--dump_folder}/{--checkpoint.folder}, this option will be ignored.
        This feature allows users to load an initial checkpoint from a different folder
        and continue training, saving new checkpoints to the specified folder without
        affecting the existing ones.

        Note that the path should contain the absolute path to the checkpoint folder,
        including the step number, if any; for example,
        "//pre_train/checkpoints/llama3/llama3_8b/step_10000".
        """

        initial_load_model_only: bool = True
        """
        This option specifies if only the model should be loaded during the initial
        checkpoint load. The option is only used when `initial_load_path` is specified.
        If False, the checkpoint at `initial_load_path` is treated as a standard
        training checkpoint, including optimizer, lr scheduler, training states, etc.
        The default setting for this option is True. Note that you will have to use
        `--checkpoint.no_initial_load_model_only` to override the default setting.
        """

        initial_load_in_hf: bool = False
        """
        Enable the use of HuggingFace's safetensors format for checkpointing. This will
        load checkpoints in HF's model definition and safetensors format instead of the
        default torchtitan model definition and DCP format, after necessary model state
        dict transformation.
        If `initial_load_path` is not provided, this option will look for weights
        in `sd_adapter.hf_assets_path`. `initial_load_model_only` must be True
        because safetensors doesn't support saving non-tensors.
        The default value is False.
        """

        initial_load_in_hf_quantized: bool = False
        """
        Enable loading of HuggingFace's safetensors format with quantized state dict
        keys. The option is only used when `initial_load_path` and
        `initial_load_path_in_hf` is specified. This will load checkpoints in HF's model
        definition and dequantize on model weights if necessary. To support this
        parameter, the model need to define proper HuggingFaceStorageReader to perform
        dequantize.
        """

        last_save_model_only: bool = True
        """
        When last_save_model_only=True, only the model will be saved at the end of
        training, the last save. With this, checkpoints can be loaded using
        `torch.load(..., weights_only=True)` after conversion. When
        last_save_model_only=False, the full checkpoint will be saved. A full
        checkpoint includes model, optimizer and train_state, which can be used to
        resume training. The default value is True.
        """

        last_save_in_hf: bool = False
        """
        Enable the use of Hugging Face's safetensors format for checkpointing. This will
        save the final checkpoints in safetensors format instead of the default DCP
        format, after necessary model state dict transformation. There will be a
        performance cost in using this as we need to consolidate the sharded tensors to
        full tensors as a separate step. last_save_model_only must be true because
        safetensors doesn't support saving non-tensors. On load, this argument isn't
        needed as we will detect whether the loaded checkpoint is in safetensors format
        or not. The default value is False.
        """

        export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
        """
        Converts to the specified precision when training completes and
        last_save_model_only=true.
        """

        async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
        """
        Which async checkpoint mode to use. Currently there are 3 different modes.

        - "disabled": Synchronized checkpointing. The training loop is blocked until all
        data is successfully saved to the persistence storage device (disk).

        - "async": Uses threading and `torch.distributed.checkpoint.async_save`.
        The training loop is blocked only during the GPU-to-CPU memory transfer. Once
        data reaches host RAM, training resumes while a background thread manages the
        final write to disk. This reduces idle time but remains subject to GIL
        contention.

        - "async_with_pinned_mem": Uses a separate process and pre-allocated pinned
        shared memory.
        The training loop resumes almost immediately by overlapping the GPU-to-CPU DMA
        transfer with the next iteration's computation. The process then persists the
        data from pinned shared memory to disk.
        This eliminates GIL contention and minimizes the blocking window to near-zero
        (< 1s), at the cost of significantly higher fixed CPU RAM usage (pinned memory).
        If case of insufficient CPU memory, performance may degrade due to memory
        paging.

        "disabled" is the default mode.
        """

        keep_latest_k: int = 10
        """
        Keeps only the latest k checkpoints, and purging older ones. If 0, keep all
        checkpoints. K cannot be 1 as the last one may be in the process of being
        saved. As a result, the metadata of the last one may not be ready yet. The
        default value is 10 to avoid filling up the disk.
        """

        load_step: int = -1
        """Load the checkpoint at the specified step. If -1, load the latest
        checkpoint."""

        exclude_from_loading: list[str] = field(default_factory=list)
        """
        Exclude specific keys from being loaded from the checkpoint.
        Provide a comma-separated list of keys to exclude,
        e.g. 'optimizer,lr_scheduler,dataloader'.
        Keys shouldn't include 'model' key.
        """

        enable_first_step_checkpoint: bool = False
        """
        Enable the checkpoint save at first step. This will save a checkpoint
        immediately after the first step to ensure checkpointing functions correctly.
        This is useful when running on a new cluster or storage to verify checkpointing
        without waiting for many steps or checkpointing too frequently. The default
        value is False.
        """

        create_seed_checkpoint: bool = False
        """
        Initializes the full model without applying parallelisms, and then saves it as a
        seed checkpoint. Note: requires user to call train.py without specifying any
        parallelisms, e.g. NGPU=1. Could be implemented as a separate script, but this
        way shares more code.
        """

        load_only: bool = False
        """
        In certain scenarios, you may only need to load checkpoints for verification or
        debugging purposes, without saving any new checkpoints. For example, you might
        use seed checkpoints to validate model correctness. Enabling this option allows
        checkpoints to be loaded without saving any during the training.
        """

        def __post_init__(self):
            if not self.folder.strip():
                raise ValueError("The 'folder' field cannot be empty.")
            if self.interval < 1:
                raise ValueError("Checkpoint interval needs to be at least 1 step.")
            if self.keep_latest_k < 0:
                raise ValueError("keep_latest_k cannot be negative.")
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            if MODEL in self.exclude_from_loading:
                raise ValueError(f"{MODEL} key shouldn't be in exclude_from_loading.")

            if self.initial_load_path:
                self.initial_load_path = self.initial_load_path.strip()
                if not (
                    self.initial_load_path.startswith("/")
                    or filesystem.is_remote(self.initial_load_path)
                ):
                    raise ValueError(
                        "initial_load_path must be an absolute path or a remote "
                        f"URI (e.g. gs://...): {self.initial_load_path}"
                    )
            if self.initial_load_in_hf and not self.initial_load_model_only:
                raise ValueError("initial_load_in_hf requires initial_load_model_only.")
            if self.initial_load_in_hf_quantized and not (
                self.initial_load_in_hf and self.initial_load_path
            ):
                raise ValueError(
                    "initial_load_in_hf_quantized requires initial_load_in_hf "
                    "and initial_load_path."
                )
            if self.last_save_in_hf and not self.last_save_model_only:
                raise ValueError("last_save_in_hf requires last_save_model_only=True.")

            # Remote (fsspec) checkpoint IO supports only the native DCP format.
            # HF safetensors read/write to a remote URI is not implemented, so
            # reject the combination up front instead of failing deep in DCP.
            if self.last_save_in_hf and filesystem.is_remote(self.folder):
                raise ValueError(
                    "last_save_in_hf is not supported with a remote "
                    f"checkpoint.folder: {self.folder}"
                )
            if (
                self.initial_load_in_hf
                and self.initial_load_path
                and filesystem.is_remote(self.initial_load_path)
            ):
                raise ValueError(
                    "initial_load_in_hf is not supported with a remote "
                    f"initial_load_path: {self.initial_load_path}"
                )

            async_lowered = self.async_mode.lower()
            if async_lowered in ("disabled", "async", "async_with_pinned_mem"):
                self.async_mode = async_lowered
            else:
                raise ValueError(f"Invalid async_mode: {async_lowered}")

            if self.load_only and self.enable_first_step_checkpoint:
                logger.warning(
                    "checkpoint.load_only is True; enable_first_step_checkpoint "
                    "will be ignored."
                )
            if self.initial_load_model_only and not self.initial_load_path:
                logger.warning(
                    "initial_load_model_only=True has no effect without "
                    "an initial_load_path."
                )

    def __init__(
        self,
        config: Config,
        *,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
    ) -> None:

        self.enable = config.enable
        if not self.enable:
            return

        self.folder = filesystem.join(base_folder, config.folder)
        self.interval = config.interval

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        # Loading & Saving Policy
        self.load_only = config.load_only
        self.exclude_from_loading = config.exclude_from_loading
        self.initial_load_path = config.initial_load_path
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized

        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint
        self.last_save_model_only = config.last_save_model_only
        self.last_save_in_hf = config.last_save_in_hf
        self.export_dtype = TORCH_DTYPE_MAP[config.export_dtype]

        self.sd_adapter = sd_adapter
        if self.last_save_in_hf and self.sd_adapter is None:
            raise ValueError(
                "checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
            )

        # Async & Distributed Infrastructure
        try:
            self.async_mode = AsyncMode(config.async_mode)
        except ValueError as e:
            raise ValueError(
                f"Unknown checkpoint async_mode {config.async_mode}"
            ) from e

        self.pg: dist.ProcessGroup | None = None
        if self.async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

        self.stager: DefaultStager | None = None
        self.staging_future: Future | None = None
        self.save_future: Future | None = None
        self._pending_checkpoint_artifact: (
            tuple[str, str, int, dict[str, Any]] | None
        ) = None

        # Retention Policy (Purge)
        self.keep_latest_k = config.keep_latest_k
        self.purge_thread: threading.Thread | None = None
        if self.keep_latest_k > 0:
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()

        logger.info(
            "Checkpointing active. Checkpoints will be loaded from and saved "
            f"to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        """Execute the DCP saving process.

        This method orchestrates the state_dict transformation (e.g., to HuggingFace
        format), selects the appropriate storage writer, and dispatches the save
        operation based on the requested synchronicity mode.

        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): Unique identifier (usually a path) for the checkpoint.
            async_mode (AsyncMode): The saving/staging strategy.
            enable_garbage_collection (bool): To trigger a manual GC collect after save.
            to_hf (bool): If True, uses a HuggingFaceStorageWriter and adapts the
                state_dict to be compatible with safetensors and HF model definitions.

        Returns:
            - None: If saved synchronously (AsyncMode.DISABLED).
            - Future: If AsyncMode.ASYNC is used (tracks disk I/O).
            - AsyncSaveResponse: If AsyncMode.ASYNC_WITH_PINNED_MEM is used
              (tracks both staging and disk I/O).
        """

        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None

        # HF Format Conversion
        if to_hf:
            assert self.sd_adapter is not None, "sd_adapter is required for to_hf=True"
            state_dict = self.sd_adapter.to_hf(state_dict)
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping

            # If sharded, we save to a subdir then consolidate
            save_path = (
                os.path.join(checkpoint_id, "sharded")
                if fqn_to_index_mapping
                else checkpoint_id
            )
            storage_writer = HuggingFaceStorageWriter(
                path=save_path,
                save_distributed=True,
                fqn_to_index_mapping=fqn_to_index_mapping,
                enable_consolidation=not fqn_to_index_mapping,
            )
            # NOTE: If `fqn_to_index_mapping` is absent, all FQNs are saved into a
            # single unified file. In this case, the StorageWriter can handle
            # consolidation internally on a single rank. However, when a mapping
            # exists, the weights are distributed across multiple files (sharded).
            # The internal consolidation is disabled here and instead
            # `consolidate_safetensors_files_on_every_rank` is used later to manage
            # the multi-file merging process.

        # Execution Dispatch
        checkpoint_save_id = (
            None if to_hf else checkpoint_id
        )  # for HF the storage_writer handles the path

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        # Post-Processing
        if to_hf and fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Load a DCP into the provided state dictionary.

        This method handles both standard DCP sharded checkpoints and HuggingFace
        safetensors. If loading from HF, it utilizes an adapter to map FQNs and
        handle format-specific sharding logic.

        Args:
            state_dict (dict): The target dictionary to populate with checkpoint data.
            checkpoint_id (str): Path or identifier for the source checkpoint.
            from_hf (bool): If True, adapts the load process for HuggingFace model
                definitions and safetensors format.
            from_quantized (bool): Indicates if the source is in a quantized format
                (e.g., 4-bit/8-bit), requiring the storage reader to handle
                specialized data types and sharding structures.

        Raises:
            AssertionError: If `from_hf` is True but no `sd_adapter` is available.
            CheckpointCorruptionError: If the native DCP manifest does not
                describe the files on disk. See
                validate_dcp_manifest_integrity for what that covers and what it
                does not.
        """

        if from_hf:
            assert self.sd_adapter is not None, (
                "trying to load checkpoint in HF safetensors format, "
                "but sd_adapter is not provided."
            )

            # The manifest integrity sentinel below reads a DCP .metadata
            # pickle. A HF checkpoint has no such file: it is described by
            # model.safetensors.index.json and read through the adapter's own
            # storage reader. Warn rather than stay silent, since a user who
            # expects the sentinel to guard every load would otherwise get no
            # signal that it did not run.
            logger.warning(
                "Checkpoint manifest integrity is NOT validated for the "
                f"HuggingFace safetensors checkpoint at {checkpoint_id}. The "
                "sentinel covers native DCP .metadata manifests only; a missing "
                "or truncated safetensors shard is not detected here."
            )

            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            self.states[MODEL].load_state_dict(state_dict)
        else:
            # Validate before dcp.load, not after a failure: the point is to
            # name the defect and the file while nothing has been loaded, so the
            # incident attributes the abort to the checkpoint path instead of to
            # whatever the reader raises from a worker thread.
            validate_dcp_manifest_integrity(checkpoint_id)

            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @sl.log_trace_span("checkpoint_save")
    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        """Save the checkpoint for the current step.

        This function manages the checkpointing lifecycle for the current step.
        A save is performed if any of the following conditions are met:
        1. It is the initial seed checkpoint (step 0).
        2. The current step matches the configured saving interval.
        3. `last_step` is True, which forces a save regardless of the interval.
           This typically happens when the training reaches its final step.

        Args:
            curr_step (int): The current training step.
            last_step (bool, optional): Whether this is the final step of training.

        Returns:
            bool: True if a checkpoint was written (or staged, for async modes) on
            this step.
        """

        if not self._should_save(curr_step, last_step):
            return False

        sl.add_step_tag("checkpoint_save")

        self.maybe_wait_for_saving()

        begin = time.monotonic()
        checkpoint_phase = (
            "saving" if self.async_mode == AsyncMode.DISABLED else "staging"
        )
        logger.info(f"{checkpoint_phase.capitalize()} the checkpoint.")

        checkpoint_id = self._create_checkpoint_id(curr_step)
        model_only = last_step and self.last_save_model_only
        metadata = self._checkpoint_artifact_metadata(
            step=curr_step,
            model_only=model_only,
            from_hf=last_step and self.last_save_in_hf,
            async_mode=AsyncMode.DISABLED if last_step else self.async_mode,
        )
        artifact_id = self._declare_checkpoint_artifact(
            checkpoint_id=checkpoint_id,
            relation=ArtifactRelation.OUTPUT,
            step=curr_step,
            metadata=metadata,
        )

        if last_step:
            try:
                self._save_last_step(curr_step)
            except BaseException:
                self._fail_checkpoint_artifact(
                    artifact_id=artifact_id,
                    checkpoint_id=checkpoint_id,
                    relation=ArtifactRelation.OUTPUT,
                    step=curr_step,
                    metadata=metadata,
                )
                raise
            self._finish_checkpoint_artifact(
                artifact_id=artifact_id,
                checkpoint_id=checkpoint_id,
                relation=ArtifactRelation.OUTPUT,
                state=ArtifactState.COMPLETE,
                step=curr_step,
                metadata=metadata,
            )
            logger.info(
                f"Last step checkpoint completed in {time.monotonic() - begin:.2f}s"
            )
            return True

        try:
            states = self._flattened_model_states_sd()
            if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.stager is None:
                    self.stager = DefaultStager(
                        StagingOptions(
                            use_pinned_memory=True,
                            use_shared_memory=True,
                            use_async_staging=True,
                            use_non_blocking_copy=True,
                        )
                    )

                result = self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=self.async_mode,
                )
                # Calling GC here is not required for this path.

                assert isinstance(result, AsyncSaveResponse)
                self.staging_future = result.staging_completion
                self.save_future = result.upload_completion

            elif self.async_mode == AsyncMode.ASYNC:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                result = self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=self.async_mode,
                )
                GarbageCollection.collect("GC collection invoked by checkpointer.")

                assert isinstance(result, Future)
                self.save_future = result

            else:
                self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=AsyncMode.DISABLED,
                    enable_garbage_collection=True,
                )
        except BaseException:
            self._fail_checkpoint_artifact(
                artifact_id=artifact_id,
                checkpoint_id=checkpoint_id,
                relation=ArtifactRelation.OUTPUT,
                step=curr_step,
                metadata=metadata,
            )
            raise

        if self.save_future is None:
            self._finish_checkpoint_artifact(
                artifact_id=artifact_id,
                checkpoint_id=checkpoint_id,
                relation=ArtifactRelation.OUTPUT,
                state=ArtifactState.COMPLETE,
                step=curr_step,
                metadata=metadata,
            )
        elif artifact_id is not None:
            self._pending_checkpoint_artifact = (
                artifact_id,
                checkpoint_id,
                curr_step,
                metadata,
            )

        self._purge_stale_checkpoints()

        logger.info(
            f"Finished {checkpoint_phase} the checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )
        return True

    @sl.log_trace_span("checkpoint_load")
    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function orchestrates the states loading process.
        If the local checkpoint folder contains a valid checkpoint, it retrieves the
        checkpoint corresponding to the specified step, defaulting to the latest
        available if the `step` is -1. Otherwise, it attempts an initial load from a
        specified path (in either native or HF format) or performs loading using
        provided HF assets path from the state dict adapter.

        Args:
            step (int, optional): The training step to restore.
                Defaults to -1 (latest available).

        Returns:
            bool: Whether the checkpoint was successfully located and loaded.
        """

        if not self.enable:
            return False

        model_only = False
        from_hf = False
        from_quantized = False

        has_checkpoint_folder = filesystem.exists(self.folder)
        load_step = -1
        if has_checkpoint_folder:
            load_step = self._find_load_step() if step == -1 else step

        if step != -1 and not has_checkpoint_folder:
            raise FileNotFoundError(
                f"--checkpoint.load_step={step} not found because "
                f"checkpoint.folder {self.folder} does not exist"
            )

        if load_step == -1:
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized

            if from_hf:
                assert model_only, (
                    "Only model can be loaded when loading from "
                    "HF's safetensors checkpoint."
                )
            if from_quantized:
                assert from_hf, "Quantized checkpoint can only be loaded from HF format"

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not filesystem.isdir(checkpoint_id):
                    raise ValueError(
                        f"Checkpoint.initial_load_path is invalid: {checkpoint_id}"
                    )
                if from_hf:
                    logger.info(
                        "Loading from HF safetensors from "
                        f"--checkpoint.initial_load_path: {checkpoint_id}"
                    )

            elif from_hf:
                assert (
                    self.sd_adapter and self.sd_adapter.hf_assets_path
                ), "from_hf=True requires sd_adapter and hf_assets_path."
                checkpoint_id = self.sd_adapter.hf_assets_path
                if not filesystem.isdir(checkpoint_id):
                    raise ValueError(
                        "model.hf_assets_path is being used to load HF weights "
                        "but the path is not valid. Either make sure hf_assets_path is "
                        "correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    "Loading HF safetensors from "
                    f"--model.hf_assets_path: {checkpoint_id}"
                )

            else:
                logger.info("No checkpoint was provided, this is a fresh start.")
                return False

        else:
            # This is the fault-tolerance branch: checkpoint.folder already
            # contains valid checkpoints from a previous run, so we resume from
            # it and all initial_* options are ignored by design. This allows a
            # job to keep the same arguments across automatic restarts after
            # failures.
            step = load_step
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not filesystem.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} not found at {checkpoint_id}"
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()

        metadata = self._checkpoint_artifact_metadata(
            step=step,
            model_only=model_only,
            from_hf=from_hf,
        )
        artifact_id = self._declare_checkpoint_artifact(
            checkpoint_id=checkpoint_id,
            relation=ArtifactRelation.INPUT,
            step=step,
            metadata=metadata,
        )
        try:
            states = self._states_to_load(model_only)
            self.dcp_load(
                states,
                checkpoint_id=checkpoint_id,
                from_hf=from_hf,
                from_quantized=from_quantized,
            )
        except BaseException:
            self._fail_checkpoint_artifact(
                artifact_id=artifact_id,
                checkpoint_id=checkpoint_id,
                relation=ArtifactRelation.INPUT,
                step=step,
                metadata=metadata,
            )
            raise
        self._finish_checkpoint_artifact(
            artifact_id=artifact_id,
            checkpoint_id=checkpoint_id,
            relation=ArtifactRelation.INPUT,
            state=ArtifactState.COMPLETE,
            step=step,
            metadata=metadata,
        )

        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            "Finished loading the checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )

        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging process to complete if it is active.

        In `ASYNC_WITH_PINNED_MEM` mode, the checkpoint data is first staged from
        device (GPU) memory to pinned host (CPU) memory. This staging process is
        asynchronous and designed to overlap with the subsequent training
        computation (forward/backward passes).

        This method ensures that the staging process has finished before the next
        checkpoint cycle begins or before training completes, preventing memory
        contention or race conditions in the pinned memory buffers.

        Raises:
            RuntimeError: If a staging future is detected while asynchronous mode
                isn't ASYNC_WITH_PINNED_MEM.
        """

        if not self.enable or self.staging_future is None:
            return

        if self.async_mode != AsyncMode.ASYNC_WITH_PINNED_MEM:
            raise RuntimeError(
                "self.staging_future is not None, "
                "but self.async_mode isn't ASYNC_WITH_PINNED_MEM."
            )

        self.staging_future.result()
        self.staging_future = None

    def maybe_wait_for_saving(self) -> None:
        """Wait for any async background checkpoint saving operation to complete.

        This is a blocking call that ensures all checkpoint data has been fully
        saved to storage. Upon completion, the tracking future is cleared
        to signify that no background save operations are currently active.

        Raises:
            RuntimeError: If a save future is detected while asynchronous mode
                is DISABLED.
        """

        if not self.enable or self.save_future is None:
            return

        if self.async_mode == AsyncMode.DISABLED:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is DISABLED."
            )

        self._wait_for_save_future()
        self.save_future = None

    def _wait_for_save_future(self) -> None:
        """Wait for the current save future and resolve its core artifact."""
        assert self.save_future is not None
        try:
            self.save_future.result()
        except BaseException:
            if self._pending_checkpoint_artifact is not None:
                (
                    artifact_id,
                    checkpoint_id,
                    step,
                    metadata,
                ) = self._pending_checkpoint_artifact
                self._pending_checkpoint_artifact = None
                self._fail_checkpoint_artifact(
                    artifact_id=artifact_id,
                    checkpoint_id=checkpoint_id,
                    relation=ArtifactRelation.OUTPUT,
                    step=step,
                    metadata=metadata,
                )
            raise
        if self._pending_checkpoint_artifact is not None:
            (
                artifact_id,
                checkpoint_id,
                step,
                metadata,
            ) = self._pending_checkpoint_artifact
            self._finish_checkpoint_artifact(
                artifact_id=artifact_id,
                checkpoint_id=checkpoint_id,
                relation=ArtifactRelation.OUTPUT,
                state=ArtifactState.COMPLETE,
                step=step,
                metadata=metadata,
            )
            self._pending_checkpoint_artifact = None

    @staticmethod
    def _checkpoint_artifact_metadata(
        *,
        step: int,
        model_only: bool,
        from_hf: bool,
        async_mode: AsyncMode | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "format": (
                "huggingface_safetensors" if from_hf else "torch_distributed_checkpoint"
            ),
            "model_only": model_only,
            "step": step,
        }
        if async_mode is not None:
            metadata["async_mode"] = async_mode.value
        return metadata

    @staticmethod
    def _checkpoint_evidence_path(checkpoint_id: str) -> str:
        if filesystem.is_remote(checkpoint_id):
            return checkpoint_id
        return str(Path(checkpoint_id).resolve())

    def _declare_checkpoint_artifact(
        self,
        *,
        checkpoint_id: str,
        relation: ArtifactRelation,
        step: int,
        metadata: dict[str, Any],
    ) -> str | None:
        return record_artifact(
            producer="checkpoint",
            kind="torchtitan.checkpoint",
            path=self._checkpoint_evidence_path(checkpoint_id),
            state=ArtifactState.DECLARED,
            relation=relation,
            step=step,
            metadata=metadata,
        )

    def _finish_checkpoint_artifact(
        self,
        *,
        artifact_id: str | None,
        checkpoint_id: str,
        relation: ArtifactRelation,
        state: ArtifactState,
        step: int,
        metadata: dict[str, Any],
    ) -> None:
        if artifact_id is None:
            return
        record_artifact(
            producer="checkpoint",
            kind="torchtitan.checkpoint",
            path=self._checkpoint_evidence_path(checkpoint_id),
            state=state,
            relation=relation,
            artifact_id=artifact_id,
            step=step,
            metadata=metadata,
        )

    def _fail_checkpoint_artifact(
        self,
        *,
        artifact_id: str | None,
        checkpoint_id: str,
        relation: ArtifactRelation,
        step: int,
        metadata: dict[str, Any],
    ) -> None:
        try:
            self._finish_checkpoint_artifact(
                artifact_id=artifact_id,
                checkpoint_id=checkpoint_id,
                relation=relation,
                state=ArtifactState.FAILED,
                step=step,
                metadata=metadata,
            )
        except BaseException:
            try:
                logger.exception(
                    "failed to append run evidence while recording checkpoint failure"
                )
            except BaseException:
                pass

    def _find_load_step(self, folder: str = "") -> int:
        """Identify the highest available checkpoint step in the specified directory.

        This method scans the target folder for subdirectories matching the
        'step-N' pattern. A folder is only considered a valid checkpoint if
        it contains either a DCP metadata file or a HuggingFace safetensors
        index.

        Args:
            folder (str, optional): The directory to scan. Defaults to `self.folder`.

        Returns:
            int: The maximum step number found among valid checkpoints,
                or -1 if no valid checkpoints are detected.

        Note:
            This function is not remote friendly: it issues one listdir plus
            up to two isfile probes per step folder, each a network round trip
            on remote (fsspec) storage instead of a single batched listing.
            Acceptable for now since it only runs once at load time.
        """

        folder = folder or self.folder
        if not filesystem.isdir(folder):
            return -1

        pattern = r"step-(\d+)"
        valid_steps = []

        for filename in filesystem.listdir(folder):
            match = re.search(pattern, filename)
            if not match:
                continue

            # A checkpoint is valid only if it contains core metadata
            checkpoint_path = filesystem.join(folder, filename)
            is_dcp = filesystem.isfile(filesystem.join(checkpoint_path, ".metadata"))
            is_hf = filesystem.isfile(
                filesystem.join(checkpoint_path, "model.safetensors.index.json")
            )

            if is_dcp or is_hf:
                valid_steps.append(int(match.group(1)))

        return max(valid_steps) if valid_steps else -1

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        """Generate the standardized filesystem path for a checkpoint
        (e.g., 'checkpoints/step-100')."""
        folder = folder or self.folder
        return filesystem.join(folder, f"step-{step}")

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Extract and flatten model parameters into a single state dictionary.

        This method merges the internal state of the model object into the top-level
        dictionary while keeping auxiliary states (such as optimizers or lr_schedulers)
        unflattened. This ensures a consistent format for the DCP writer.

        Args:
            state_dict (dict[str, Any], optional): A custom dictionary to flatten.
                Defaults to None (uses the instance's internal states).

        Returns:
            dict[str, Any]: A unified dictionary containing both flattened model
                parameters and top-level auxiliary states.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determine which state objects should be restored during loading.

        This method filters the checkpointer's state dictionary based on the
        loading context. It supports partial restoration for specific steps
        (e.g., loading only model weights for step 0) and respects explicit
        exclusion rules for auxiliary states.

        Args:
            model_only (bool): If True, returns only the model's parameters,
                bypassing optimizers and other training metadata.

        Returns:
            dict[str, Any]: A prepared dictionary of states to be passed to
                the loader.
        """
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        return self._flattened_model_states_sd(states_to_load)

    def _save_last_step(self, curr_step: int) -> None:
        """Execute the final checkpoint save at the completion of training.

        This method handles the specific requirements for the final training
        artifact. It allows for saving model weights exclusively (stripping
        optimizer states), performing data type conversion for export, and
        optionally formatting the output for HuggingFace compatibility.

        Args:
            curr_step (int): The final training step index.
        """

        # If `last_save_model_only` is False, we save the full training state
        # without dtype conversion to ensure training can be resumed safely.
        # Otherwise, we assume training is fully complete and save only the model
        # with dtype conversion if the current dtype isn't equal to the export dtype.

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            if self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        """Determine whether a checkpoint should be saved based on
        the current step, interval, and training status."""

        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints.

        Extracted so subclasses (e.g. TorchFTCheckpointManager) can add
        additional guards (like participating_rank) without duplicating
        the purge loop in _purge_stale_checkpoints.
        """
        return (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and filesystem.isdir(self.folder)
        )

    def _purge_stale_checkpoints(self):
        """Remove older checkpoint directories from storage to maintain
        only the most recent 'k' copies."""
        if self._should_purge():
            discovered_checkpoints = []
            for filename in filesystem.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = filesystem.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
