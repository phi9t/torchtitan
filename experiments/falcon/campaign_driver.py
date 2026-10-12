# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small campaign-level entrypoints for Falcon continuation tickets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from torchtitan.experiments.falcon.addition import (
    ADDITION_DRAW_IDS,
    AdditionDataset,
    AdditionSplitManifest,
    AdditionVocab,
    evaluate_addition_suffixes,
    OnlineAdditionStream,
)
from torchtitan.experiments.falcon.evidence import (
    canonical_native_run_id,
    write_native_attempt_bundle,
)
from torchtitan.experiments.falcon.lm_eval import (
    aggregate_region_ce,
    build_fixed_region_registry,
    evaluate_regions,
)
from torchtitan.experiments.falcon.mechanism import MechanismDiagnostics
from torchtitan.experiments.falcon.promotion import (
    decide_gdn_addition,
    decide_mechanism_screen_eligibility,
    GdnAdditionProjectionInputs,
    MECHANISM_SCREEN_ARMS,
    MECHANISM_SCREEN_EQUIVALENT_PAIRS,
    MECHANISM_SCREEN_REQUIRED_REGIONS,
    MECHANISM_SCREEN_REQUIRED_SHAPE,
)


_RESULTS_ROOT = Path("experiments/falcon/results/evidence")
_ARTIFACT_ROOT = Path("experiments/falcon/results/addition_preflight")
_RESERVE_BYTES = 5 * 1024**3
_PROJECTED_PEAK_BYTES = 2 * 1024**3
_TRAINING_STEPS_PER_SEED = 2_000
_WARMUP_STEPS = 20
_BATCH_SIZE = 128
_AUTHORIZED_PREFLIGHT_ARM = "A1"
_AUTHORIZED_PREFLIGHT_SEED = 0
_AUTHORIZED_PREFLIGHT_STEPS = 100
_MECHANISM_PREFLIGHT_ROOT = Path("experiments/falcon/results/mechanism_preflight")
_MECHANISM_PREFLIGHT_STEPS = 3
_MECHANISM_PREFLIGHT_RESERVE_BYTES = 5 * 1024**3
_MECHANISM_PREFLIGHT_PROJECTED_PEAK_BYTES = 3 * 1024**3


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(value)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _repo_artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        return f"external-falcon-artifacts/{digest}/{path.name}"


def _sync_device() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _set_single_process_env(*, master_port: int) -> None:
    for key, value in (
        ("LOCAL_RANK", "0"),
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("NGPU", "1"),
    ):
        existing = os.environ.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(
                f"addition preflight requires {key}={value}, got {existing!r}"
            )
        os.environ[key] = value
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _disk_accounting(root: Path) -> dict[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    return {
        "free_bytes": usage.free,
        "projected_peak_bytes": _PROJECTED_PEAK_BYTES,
        "retained_artifact_bytes": _dir_size(root),
        "reserve_bytes": _RESERVE_BYTES,
    }


def _require_rootfs() -> None:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        raise RuntimeError(
            "campaign_driver must run inside the bwrap rootfs "
            "(TORCHTITAN_IN_ROOTFS=1). Launch via experiments/falcon/run.sh."
        )


def build_projection_inputs(
    preflight: dict[str, Any],
    disk: dict[str, int],
) -> GdnAdditionProjectionInputs:
    steady_samples = [float(x) for x in preflight["steady_state_step_seconds"]]
    mean_step = sum(steady_samples) / len(steady_samples)
    per_draw_eval = {
        draw_id: float(preflight["eval_seconds_by_draw"][draw_id])
        for draw_id in ADDITION_DRAW_IDS
    }
    checkpoint_seconds = float(preflight["checkpoint_seconds"])
    return GdnAdditionProjectionInputs(
        free_bytes=int(disk["free_bytes"]),
        projected_peak_bytes=int(disk["projected_peak_bytes"]),
        retained_artifact_bytes=int(disk["retained_artifact_bytes"]),
        reserve_bytes=int(disk["reserve_bytes"]),
        warmup_steps=int(preflight["warmup_steps"]),
        steady_state_step_seconds=steady_samples,
        per_draw_eval_seconds=per_draw_eval,
        checkpoint_seconds_per_seed=checkpoint_seconds,
        preflight_seconds=float(preflight["elapsed_sec"]),
        failed_attempt_seconds=list(preflight.get("failed_attempt_seconds", [])),
        retry_seconds_per_logical_run=(
            mean_step * _TRAINING_STEPS_PER_SEED
            + sum(per_draw_eval.values())
            + checkpoint_seconds
        ),
        training_steps_per_seed=_TRAINING_STEPS_PER_SEED,
        required_seeds=[0, 1, 2],
        covered_seeds=[0, 1, 2],
        device_count_per_logical_run=1,
    )


def _train_and_measure_addition(
    *,
    arm: str,
    seed: int,
    steps: int,
    artifact_dir: Path,
) -> tuple[dict[str, Any], Any, AdditionVocab, AdditionSplitManifest]:
    import torch

    from experiments.falcon.ablation_runner import _addition_model, _mixer_signature

    if steps != _AUTHORIZED_PREFLIGHT_STEPS:
        raise ValueError("addition-preflight is only authorized for exactly 100 steps")
    if steps <= _WARMUP_STEPS:
        raise ValueError(f"steps must exceed warmup_steps={_WARMUP_STEPS}")
    if arm != _AUTHORIZED_PREFLIGHT_ARM:
        raise ValueError("addition-preflight is only authorized for GDN arm A1")
    if seed != _AUTHORIZED_PREFLIGHT_SEED:
        raise ValueError("addition-preflight is only authorized for seed 0")
    if not torch.cuda.is_available():
        raise RuntimeError("addition-preflight requires a CUDA B200 device")

    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    vocab = AdditionVocab()
    manifest = AdditionSplitManifest.generate(seed=seed)
    stream = OnlineAdditionStream(seed=seed, manifest=manifest, vocab=vocab)
    seq_len = max(
        len(example.tokens)
        for draw_id in ADDITION_DRAW_IDS
        for example in manifest.examples(draw_id, vocab=vocab)
    )
    mixer, variant, alignment, phi = _mixer_signature(arm)
    model, model_config = _addition_model(
        mixer,
        variant,
        alignment,
        phi,
        vocab.size,
        seq_len,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=0.0)
    losses: list[float] = []
    step_seconds: list[float] = []
    model.train()
    for step in range(steps):
        examples = [stream.next_example() for _ in range(_BATCH_SIZE)]
        collated = AdditionDataset(16, 24, examples, []).collate(examples, vocab=vocab)
        _sync_device()
        step_started = time.perf_counter()
        tokens = collated["input"].to(device)
        labels = collated["labels"].to(device)
        mask = collated["loss_mask"].to(device)
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab.size),
            labels.reshape(-1),
            reduction="none",
        )
        loss = (loss * mask.reshape(-1).float()).sum() / mask.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _sync_device()
        step_elapsed = time.perf_counter() - step_started
        losses.append(float(loss.item()))
        if step + 1 > _WARMUP_STEPS:
            step_seconds.append(step_elapsed)

    eval_seconds_by_draw: dict[str, float] = {}
    eval_metrics_by_draw: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for draw_id in ADDITION_DRAW_IDS:
            examples = manifest.examples(draw_id, vocab=vocab)
            collated = AdditionDataset(16, 24, examples, []).collate(
                examples, vocab=vocab
            )
            _sync_device()
            eval_started = time.perf_counter()
            logits = model(collated["input"].to(device))
            _sync_device()
            eval_seconds_by_draw[draw_id] = time.perf_counter() - eval_started
            eval_metrics_by_draw[draw_id] = evaluate_addition_suffixes(
                logits.cpu(),
                {
                    key: value.cpu() if hasattr(value, "cpu") else value
                    for key, value in collated.items()
                },
                vocab=vocab,
            ).to_dict()

    checkpoint_path = artifact_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _sync_device()
    ckpt_started = time.perf_counter()
    torch.save(
        {
            "arm": arm,
            "seed": seed,
            "steps": steps,
            "model_config": model_config.__dict__,
            "model": model.state_dict(),
        },
        checkpoint_path,
    )
    _sync_device()
    checkpoint_seconds = time.perf_counter() - ckpt_started
    return (
        {
            "arm": arm,
            "seed": seed,
            "requested_steps": steps,
            "completed_steps": steps,
            "warmup_steps": _WARMUP_STEPS,
            "steady_state_step_seconds": step_seconds,
            "losses": losses,
            "eval_seconds_by_draw": eval_seconds_by_draw,
            "eval_metrics_by_draw": eval_metrics_by_draw,
            "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_path": str(checkpoint_path),
            "mixer": mixer,
            "variant": variant,
            "alignment": alignment,
            "phi": phi,
            "batch_size": _BATCH_SIZE,
            "seq_len": seq_len,
            "dtype": "float32",
            "device": "cuda:0",
            "split_manifest_digest": manifest.digest,
            "failed_attempt_seconds": [],
        },
        model,
        vocab,
        manifest,
    )


def _attempt_record(
    *,
    attempt_id: str,
    arm: str,
    seed: int,
    started_utc: str,
    ended_utc: str,
    steps: int,
    completed_steps: int,
    artifacts: list[dict[str, Any]],
    projection: dict[str, Any],
) -> dict[str, Any]:
    logical_run = {"workload": "b12-gdn-addition-preflight", "arm": arm, "seed": seed}
    return {
        "run_id": canonical_native_run_id(logical_run),
        "attempt_id": attempt_id,
        "lane": "science",
        "mode": "addition_preflight",
        "arm": arm,
        "claim_label": "spend_projection_preflight",
        "evidence_tier": "tier0",
        "environment_class": "rootfs_b200",
        "logical_run": logical_run,
        "processes": [
            {
                "process_id": "addition-preflight.rank0",
                "role": "trainer",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "host_name": socket.gethostname(),
            }
        ],
        "mesh": {"axes": {"dp": 1}},
        "device": {"type": "cuda", "index": 0, "uuid": "B200-or-current-cuda0"},
        "clocks": {"started_utc": started_utc, "ended_utc": ended_utc},
        "steps": {"requested": steps, "completed": completed_steps},
        "phases": [
            {"name": "addition_train_preflight", "outcome": "completed"},
            {"name": "addition_eval", "outcome": "completed"},
            {"name": "checkpoint_write", "outcome": "completed"},
            {"name": "promotion_decision", "outcome": "completed"},
        ],
        "data_lineage": {
            "split_manifest_digest": projection["inputs"].get(
                "split_manifest_digest", "recorded-in-artifact"
            ),
            "stream_policy": "online-id-widths-1-16-excludes-all-eval-draws",
            "eval_draws": ",".join(ADDITION_DRAW_IDS),
        },
        "checkpoint_lineage": {
            "checkpoint_policy": "preflight-measurement-only",
            "output_checkpoint": next(
                artifact["path"]
                for artifact in artifacts
                if artifact["kind"] == "addition_preflight_checkpoint"
            ),
        },
        "artifacts": artifacts,
        "outcome": {
            "status": "completed",
            "reason": None,
            "promotion_decision": projection["decision"],
            "projected_b200_hours": projection["projected_b200_hours"],
        },
    }


def run_addition_preflight(
    *,
    arm: str,
    seed: int,
    steps: int,
    results_root: Path,
    artifact_root: Path,
    master_port: int,
) -> dict[str, Any]:
    _require_rootfs()
    _set_single_process_env(master_port=master_port)
    attempt_id = f"b12-gdn-addition-preflight-{arm}-seed{seed}-{uuid.uuid4().hex[:12]}"
    artifact_dir = artifact_root / attempt_id
    started_utc = _utc_now()
    started = time.perf_counter()
    disk_before = _disk_accounting(artifact_root)
    preflight, _model, _vocab, manifest = _train_and_measure_addition(
        arm=arm,
        seed=seed,
        steps=steps,
        artifact_dir=artifact_dir,
    )
    preflight["elapsed_sec"] = time.perf_counter() - started
    manifest_path = artifact_dir / "split_manifest.json"
    manifest.write_json(manifest_path)
    preflight_path = artifact_dir / "preflight.json"
    preflight_digest = _write_json(preflight_path, preflight)
    projection_inputs = build_projection_inputs(preflight, disk_before)
    decision = decide_gdn_addition(projection_inputs)
    decision_payload = decision.to_dict()
    decision_payload["inputs"]["split_manifest_digest"] = manifest.digest
    decision_path = artifact_dir / "gdn_addition_decision.json"
    decision_digest = _write_json(decision_path, decision_payload)
    ended_utc = _utc_now()

    artifacts = [
        {
            "path": _repo_artifact_path(preflight_path),
            "kind": "addition_preflight_accounting",
            "digest": preflight_digest,
        },
        {
            "path": _repo_artifact_path(decision_path),
            "kind": "gdn_addition_decision",
            "digest": decision_digest,
        },
        {
            "path": _repo_artifact_path(manifest_path),
            "kind": "addition_split_manifest",
            "digest": _file_sha256(manifest_path),
            "split_manifest_digest": manifest.digest,
        },
        {
            "path": _repo_artifact_path(Path(preflight["checkpoint_path"])),
            "kind": "addition_preflight_checkpoint",
            "digest": _file_sha256(Path(preflight["checkpoint_path"])),
        },
    ]
    record = _attempt_record(
        attempt_id=attempt_id,
        arm=arm,
        seed=seed,
        started_utc=started_utc,
        ended_utc=ended_utc,
        steps=steps,
        completed_steps=preflight["completed_steps"],
        artifacts=artifacts,
        projection=decision_payload,
    )
    bundle_path = write_native_attempt_bundle(results_root, record)
    summary = {
        "attempt_id": attempt_id,
        "bundle_path": str(bundle_path),
        "artifact_dir": str(artifact_dir),
        "preflight_path": str(preflight_path),
        "decision_path": str(decision_path),
        "decision": decision_payload,
    }
    _write_json(artifact_dir / "summary.json", summary)
    return summary


def _validate_existing_preflight_artifacts(
    *,
    preflight: dict[str, Any],
    decision_payload: dict[str, Any],
    manifest_path: Path,
    checkpoint_path: Path,
) -> AdditionSplitManifest:
    if preflight.get("arm") != _AUTHORIZED_PREFLIGHT_ARM:
        raise ValueError("existing preflight artifact must be arm A1")
    if preflight.get("seed") != _AUTHORIZED_PREFLIGHT_SEED:
        raise ValueError("existing preflight artifact must be seed 0")
    for field in ("requested_steps", "completed_steps"):
        if preflight.get(field) != _AUTHORIZED_PREFLIGHT_STEPS:
            raise ValueError(f"existing preflight artifact must record {field}=100")
    if preflight.get("warmup_steps") != _WARMUP_STEPS:
        raise ValueError("existing preflight artifact has wrong warmup_steps")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing preflight checkpoint: {checkpoint_path}")
    manifest = AdditionSplitManifest.read_json(manifest_path)
    manifest_digest = str(preflight.get("split_manifest_digest"))
    if manifest.digest != manifest_digest:
        raise ValueError("preflight split manifest digest does not match manifest")
    inputs = decision_payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("decision payload must include projection inputs")
    if inputs.get("split_manifest_digest") != manifest.digest:
        raise ValueError("decision split manifest digest does not match manifest")
    if decision_payload.get("record_type") != "gdn_addition_decision":
        raise ValueError("decision payload must be a gdn_addition_decision")

    disk = {
        "free_bytes": inputs["free_bytes"],
        "projected_peak_bytes": inputs["projected_peak_bytes"],
        "retained_artifact_bytes": inputs["retained_artifact_bytes"],
        "reserve_bytes": inputs["reserve_bytes"],
    }
    recomputed = decide_gdn_addition(build_projection_inputs(preflight, disk)).to_dict()
    recomputed["inputs"]["split_manifest_digest"] = manifest.digest
    for field in ("decision", "reason", "projected_b200_hours", "disk", "arithmetic"):
        if decision_payload.get(field) != recomputed[field]:
            raise ValueError(f"decision payload does not match recomputed {field}")
    for field in (
        "warmup_steps",
        "steady_state_step_seconds",
        "per_draw_eval_seconds",
        "checkpoint_seconds_per_seed",
        "preflight_seconds",
        "failed_attempt_seconds",
        "retry_seconds_per_logical_run",
        "training_steps_per_seed",
        "required_seeds",
        "covered_seeds",
        "device_count_per_logical_run",
    ):
        if inputs.get(field) != recomputed["inputs"][field]:
            raise ValueError(f"decision inputs do not match recomputed {field}")
    return manifest


def publish_existing_addition_preflight(
    *,
    artifact_dir: Path,
    results_root: Path,
) -> dict[str, Any]:
    _require_rootfs()
    preflight_path = artifact_dir / "preflight.json"
    decision_path = artifact_dir / "gdn_addition_decision.json"
    manifest_path = artifact_dir / "split_manifest.json"
    checkpoint_path = artifact_dir / "checkpoint.pt"
    preflight = json.loads(preflight_path.read_text())
    decision_payload = json.loads(decision_path.read_text())
    manifest = _validate_existing_preflight_artifacts(
        preflight=preflight,
        decision_payload=decision_payload,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
    )
    artifacts = [
        {
            "path": _repo_artifact_path(preflight_path),
            "kind": "addition_preflight_accounting",
            "digest": _file_sha256(preflight_path),
        },
        {
            "path": _repo_artifact_path(decision_path),
            "kind": "gdn_addition_decision",
            "digest": _file_sha256(decision_path),
        },
        {
            "path": _repo_artifact_path(manifest_path),
            "kind": "addition_split_manifest",
            "digest": _file_sha256(manifest_path),
            "split_manifest_digest": manifest.digest,
        },
        {
            "path": _repo_artifact_path(checkpoint_path),
            "kind": "addition_preflight_checkpoint",
            "digest": _file_sha256(checkpoint_path),
        },
    ]
    attempt_id = artifact_dir.name
    logical_run = {
        "workload": "b12-gdn-addition-preflight",
        "arm": str(preflight["arm"]),
        "seed": int(preflight["seed"]),
    }
    existing_manifest = (
        results_root
        / "runs"
        / canonical_native_run_id(logical_run)
        / attempt_id
        / "manifest.json"
    )
    clocks = (
        json.loads(existing_manifest.read_text())["clocks"]
        if existing_manifest.is_file()
        else {"started_utc": _utc_now(), "ended_utc": _utc_now()}
    )
    record = _attempt_record(
        attempt_id=attempt_id,
        arm=str(preflight["arm"]),
        seed=int(preflight["seed"]),
        started_utc=clocks["started_utc"],
        ended_utc=clocks["ended_utc"],
        steps=int(preflight["requested_steps"]),
        completed_steps=int(preflight["completed_steps"]),
        artifacts=artifacts,
        projection=decision_payload,
    )
    bundle_path = write_native_attempt_bundle(results_root, record)
    summary = {
        "attempt_id": attempt_id,
        "bundle_path": str(bundle_path),
        "artifact_dir": str(artifact_dir),
        "preflight_path": str(preflight_path),
        "decision_path": str(decision_path),
        "decision": decision_payload,
    }
    _write_json(artifact_dir / "summary.json", summary)
    return summary


def _mechanism_shape(config) -> dict[str, int]:
    inner = config.model_spec.model.config
    return {
        "num_hidden_layers": int(inner.num_hidden_layers),
        "hidden_size": int(inner.hidden_size),
        "num_heads": int(inner.num_heads),
        "head_dim": int(inner.head_dim),
        "seq_len": int(config.training.seq_len),
        "tokens_per_step": int(config.training.local_batch_size)
        * int(config.training.seq_len),
    }


def _build_mechanism_preflight_config(
    *,
    arm: str,
    seed: int,
    steps: int,
):
    from torchtitan.config import ConfigManager
    from torchtitan.experiments.falcon.config_registry import (
        apply_mechanism_arm,
        falcon_science,
    )

    config = falcon_science()
    apply_mechanism_arm(config, arm)
    config.training.steps = steps
    config.lr_scheduler.total_steps = steps
    config.lr_scheduler.warmup_steps = min(200, max(1, steps))
    config.debug.seed = seed
    config.debug.enable_structured_logging = False
    config.checkpoint.enable = False
    config.metrics.log_freq = 1
    manager = ConfigManager()
    manager.config = config
    manager._validate_config()
    shape = _mechanism_shape(config)
    if shape != MECHANISM_SCREEN_REQUIRED_SHAPE:
        raise ValueError(f"M03 preflight must use exact science shape, got {shape}")
    return config


def _runtime_device() -> tuple[Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("M03 mechanism preflight requires a CUDA B200 device")
    index = 0
    props = torch.cuda.get_device_properties(index)
    name = torch.cuda.get_device_name(index)
    if "B200" not in name:
        raise RuntimeError(
            f"M03 mechanism preflight requires a CUDA B200 device, got {name!r}"
        )
    try:
        uuid_value = str(props.uuid)
    except AttributeError:
        uuid_value = name
    return torch.device(f"cuda:{index}"), {
        "type": "cuda",
        "index": index,
        "name": name,
        "uuid": uuid_value,
    }


def _grad_norm(parameters) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().float().pow(2).sum().double().cpu()
    return float(total.sqrt().item())


def _model_only_checkpoint(
    *,
    model,
    path: Path,
    arm: str,
    seed: int,
    completed_steps: int,
    shape: dict[str, int],
) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    _sync_device()
    started = time.perf_counter()
    torch.save(
        {
            "arm": arm,
            "seed": seed,
            "completed_steps": completed_steps,
            "shape": shape,
            "model": model.state_dict(),
        },
        path,
    )
    _sync_device()
    return {
        "path": _repo_artifact_path(path),
        "kind": "model_only_checkpoint",
        "step": completed_steps,
        "sha256": _file_sha256(path),
        "seconds": time.perf_counter() - started,
    }


def _take_mechanism_probe(
    *,
    model,
    tokens,
    step: int,
) -> dict[str, Any]:
    import torch

    diagnostics = MechanismDiagnostics()
    device = next(model.parameters()).device
    model.eval()
    _sync_device()
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(tokens.to(device), mechanism_diagnostics=diagnostics)
    _sync_device()
    elapsed = time.perf_counter() - started
    model.train()
    return {
        "step": step,
        "probe_batch_digest": hashlib.sha256(
            tokens.cpu().numpy().tobytes()
        ).hexdigest(),
        "logits_checksum": float(logits.float().sum().item()),
        "seconds": elapsed,
        "diagnostics": diagnostics.summary(),
    }


def _run_one_mechanism_preflight_arm(
    *,
    arm: str,
    seed: int,
    steps: int,
    artifact_dir: Path,
    master_port: int,
) -> dict[str, Any]:
    import torch

    _set_single_process_env(master_port=master_port)
    started_utc = _utc_now()
    torch.manual_seed(seed)
    config = _build_mechanism_preflight_config(arm=arm, seed=seed, steps=steps)
    shape = _mechanism_shape(config)
    device, device_record = _runtime_device()
    model = config.model_spec.model.build()
    model.init_states()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=0.1)
    data = config.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=None,
        seq_len=config.training.seq_len,
        local_batch_size=config.training.local_batch_size,
    )
    data_iter = iter(data)
    probe_tokens = (
        torch.arange(
            config.training.seq_len,
            dtype=torch.long,
        ).unsqueeze(0)
        % config.loss.global_vocab_size
    )
    probes = [_take_mechanism_probe(model=model, tokens=probe_tokens, step=0)]
    trajectory: list[dict[str, Any]] = []
    completed_steps = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch, labels = next(data_iter)
        tokens = batch["input"].to(device)
        labels = labels.to(device)
        _sync_device()
        step_started = time.perf_counter()
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.loss.global_vocab_size).float(),
            labels.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _grad_norm(model.parameters())
        optimizer.step()
        _sync_device()
        completed_steps = step
        trajectory.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "grad_norm": grad_norm,
                "step_seconds": time.perf_counter() - step_started,
            }
        )
        probes.append(
            _take_mechanism_probe(model=model, tokens=probe_tokens, step=step)
        )

    registry = build_fixed_region_registry()
    _sync_device()
    eval_started = time.perf_counter()
    region_rows = evaluate_regions(
        model,
        registry,
        seq_len=config.training.seq_len,
        vocab_size=config.loss.global_vocab_size,
    )
    _sync_device()
    eval_seconds = time.perf_counter() - eval_started
    checkpoint = _model_only_checkpoint(
        model=model,
        path=artifact_dir / f"{arm}_seed{seed}_model_only.pt",
        arm=arm,
        seed=seed,
        completed_steps=completed_steps,
        shape=shape,
    )
    ended_utc = _utc_now()
    return {
        "arm": arm,
        "seed": seed,
        "shape": shape,
        "device": device_record,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "elapsed_sec": time.perf_counter() - started,
        "trajectory": trajectory,
        "fixed_region_eval": {
            "registry": registry.to_dict(),
            "registry_digest": registry.registry_digest,
            "regions": region_rows,
            "aggregate": aggregate_region_ce(region_rows, registry),
            "seconds": eval_seconds,
        },
        "probes": probes,
        "checkpoint": checkpoint,
        "attempt": {
            "status": "completed",
            "attempt_id": f"mechanism-preflight-{arm}-seed{seed}",
            "failure_reason": None,
        },
        "clocks": {"started_utc": started_utc, "ended_utc": ended_utc},
    }


def _max_pair_diff(
    left: list[dict[str, Any]], right: list[dict[str, Any]], key: str
) -> float | None:
    by_step_left = {row["step"]: float(row[key]) for row in left}
    by_step_right = {row["step"]: float(row[key]) for row in right}
    common_steps = set(by_step_left) & set(by_step_right)
    if not common_steps:
        return None
    return max(abs(by_step_left[step] - by_step_right[step]) for step in common_steps)


def build_mechanism_screen_evidence(
    *,
    attempts: dict[str, dict[str, Any]],
    disk: dict[str, int],
    failed_attempts: list[dict[str, Any]] | None = None,
    retry_budget_attempts: int = 1,
) -> dict[str, Any]:
    discriminability: list[dict[str, Any]] = []
    equivalent: list[dict[str, Any]] = []
    equivalent_pairs = {
        tuple(sorted(pair)) for pair in MECHANISM_SCREEN_EQUIVALENT_PAIRS
    }
    for left in MECHANISM_SCREEN_ARMS:
        for right in MECHANISM_SCREEN_ARMS:
            if left >= right or left not in attempts or right not in attempts:
                continue
            row = {
                "arms": [left, right],
                "max_abs_loss_diff": _max_pair_diff(
                    attempts[left].get("trajectory", []),
                    attempts[right].get("trajectory", []),
                    "loss",
                ),
                "max_abs_grad_norm_diff": _max_pair_diff(
                    attempts[left].get("trajectory", []),
                    attempts[right].get("trajectory", []),
                    "grad_norm",
                ),
            }
            if tuple(sorted((left, right))) in equivalent_pairs:
                left_probes = attempts[left].get("probes", [])
                right_probes = attempts[right].get("probes", [])
                left_digest = (
                    left_probes[0].get("probe_batch_digest")
                    if left_probes and isinstance(left_probes[0], dict)
                    else None
                )
                right_digest = (
                    right_probes[0].get("probe_batch_digest")
                    if right_probes and isinstance(right_probes[0], dict)
                    else None
                )
                row["same_input_digest"] = (
                    left_digest
                    if isinstance(left_digest, str) and left_digest == right_digest
                    else ""
                )
                equivalent.append(row)
            else:
                discriminability.append(row)

    elapsed_seconds = sum(
        float(record.get("elapsed_sec", 0.0)) for record in attempts.values()
    )
    checkpoint_seconds = sum(
        float(record.get("checkpoint", {}).get("seconds", 0.0))
        for record in attempts.values()
    )
    eval_seconds = sum(
        float(record.get("fixed_region_eval", {}).get("seconds", 0.0))
        for record in attempts.values()
    )
    probe_seconds = sum(
        float(probe.get("seconds", 0.0))
        for record in attempts.values()
        for probe in record.get("probes", [])
    )
    retry_seconds = sum(
        float(row.get("elapsed_sec", 0.0)) for row in (failed_attempts or [])
    )
    steady_samples = [
        float(row["step_seconds"])
        for record in attempts.values()
        for row in record.get("trajectory", [])
    ]
    mean_step_seconds = (
        sum(steady_samples) / len(steady_samples) if steady_samples else 0.0
    )
    projection = {
        "logical_runs": 15,
        "screen_b200_hours": mean_step_seconds * 2_000 * 15 / 3600.0,
        "preflight_b200_hours": elapsed_seconds / 3600.0,
        "probe_b200_hours": probe_seconds * 15 / max(len(attempts), 1) / 3600.0,
        "fixed_region_eval_b200_hours": eval_seconds
        * 15
        / max(len(attempts), 1)
        / 3600.0,
        "checkpoint_b200_hours": checkpoint_seconds
        * 15
        / max(len(attempts), 1)
        / 3600.0,
        "failure_retry_b200_hours": retry_seconds / 3600.0,
        "safety_reserve_b200_hours": 0.05,
    }
    return {
        "schema_version": 1,
        "record_type": "mechanism_screen_preflight",
        "required_arms": list(MECHANISM_SCREEN_ARMS),
        "required_seed": 0,
        "required_regions": list(MECHANISM_SCREEN_REQUIRED_REGIONS),
        "required_probe_steps": list(
            range(
                0,
                max((r.get("completed_steps", 0) for r in attempts.values()), default=0)
                + 1,
            )
        ),
        "required_shape": dict(MECHANISM_SCREEN_REQUIRED_SHAPE),
        "attempts": attempts,
        "equivalent_arm_pairs": equivalent,
        "discriminability": discriminability,
        "disk": disk,
        "projection": projection,
        "failure_accounting": {
            "failed_attempts": failed_attempts or [],
            "retry_budget_attempts": retry_budget_attempts,
        },
        "omission_records": [],
    }


def _mechanism_preflight_outcome(
    decision: Any,
) -> dict[str, str | float]:
    return {
        "status": "completed",
        "reason": decision.reason,
        "promotion_decision": decision.decision,
        "projected_b200_hours": decision.projected_b200_hours,
    }


def run_mechanism_screen_preflight(
    *,
    seed: int,
    steps: int,
    artifact_root: Path,
    master_port: int,
) -> dict[str, Any]:
    _require_rootfs()
    if steps < 1:
        raise ValueError("mechanism screen preflight steps must be >= 1")
    attempt_id = f"mechanism-screen-preflight-seed{seed}-{uuid.uuid4().hex[:12]}"
    artifact_dir = artifact_root / attempt_id
    attempts: dict[str, dict[str, Any]] = {}
    failed_attempts: list[dict[str, Any]] = []
    for index, arm in enumerate(MECHANISM_SCREEN_ARMS):
        started = time.perf_counter()
        arm_started_utc = _utc_now()
        try:
            attempts[arm] = _run_one_mechanism_preflight_arm(
                arm=arm,
                seed=seed,
                steps=steps,
                artifact_dir=artifact_dir / arm,
                master_port=master_port + index,
            )
        except Exception as exc:
            arm_ended_utc = _utc_now()
            failed_attempts.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "elapsed_sec": time.perf_counter() - started,
                    "reason": str(exc),
                    "started_utc": arm_started_utc,
                    "ended_utc": arm_ended_utc,
                }
            )
            attempts[arm] = {
                "arm": arm,
                "seed": seed,
                "shape": dict(MECHANISM_SCREEN_REQUIRED_SHAPE),
                "requested_steps": steps,
                "completed_steps": 0,
                "trajectory": [],
                "fixed_region_eval": {"regions": []},
                "probes": [],
                "checkpoint": {},
                "attempt": {
                    "status": "failed",
                    "attempt_id": f"mechanism-preflight-{arm}-seed{seed}",
                    "failure_reason": str(exc),
                },
                "clocks": {
                    "started_utc": arm_started_utc,
                    "ended_utc": arm_ended_utc,
                },
            }
    disk = _disk_accounting(artifact_root)
    evidence = build_mechanism_screen_evidence(
        attempts=attempts,
        disk=disk,
        failed_attempts=failed_attempts,
    )
    evidence_path = artifact_dir / "mechanism_screen_preflight.json"
    evidence_digest = _write_json(evidence_path, evidence)
    decision = decide_mechanism_screen_eligibility(evidence)
    decision_path = artifact_dir / "mechanism_screen_eligibility.json"
    decision_digest = _write_json(decision_path, decision.to_dict())
    artifacts = [
        {
            "path": _repo_artifact_path(evidence_path),
            "kind": "mechanism_screen_preflight",
            "digest": evidence_digest,
        },
        {
            "path": _repo_artifact_path(decision_path),
            "kind": "mechanism_screen_eligibility",
            "digest": decision_digest,
        },
    ]
    for record in attempts.values():
        checkpoint = record.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("path"):
            artifacts.append(
                {
                    "path": str(checkpoint["path"]),
                    "kind": "mechanism_preflight_checkpoint",
                    "digest": str(checkpoint.get("sha256", "")),
                }
            )
    started_utc = min(
        (
            str(record.get("clocks", {}).get("started_utc"))
            for record in attempts.values()
            if isinstance(record.get("clocks"), dict)
            and record.get("clocks", {}).get("started_utc")
        ),
        default=_utc_now(),
    )
    ended_utc = _utc_now()
    native_record = {
        "run_id": canonical_native_run_id(
            {"workload": "mechanism-screen-preflight", "arm": "M0-M4", "seed": seed}
        ),
        "attempt_id": attempt_id,
        "lane": "science",
        "mode": "mechanism_screen_preflight",
        "arm": "M0-M4",
        "claim_label": "m03_preflight",
        "evidence_tier": "tier0",
        "environment_class": "rootfs_b200"
        if os.environ.get("TORCHTITAN_IN_ROOTFS") == "1"
        else "host",
        "logical_run": {
            "workload": "mechanism-screen-preflight",
            "arm": "M0-M4",
            "seed": seed,
        },
        "processes": [
            {
                "process_id": "mechanism-screen-preflight.rank0",
                "role": "trainer",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "host_name": socket.gethostname(),
            }
        ],
        "mesh": {"axes": {"dp": 1}},
        "device": next(
            (
                record["device"]
                for record in attempts.values()
                if isinstance(record.get("device"), dict)
            ),
            {"type": "unknown", "index": 0, "uuid": "unknown"},
        ),
        "clocks": {"started_utc": started_utc, "ended_utc": ended_utc},
        "steps": {
            "requested": steps * len(MECHANISM_SCREEN_ARMS),
            "completed": sum(
                int(record.get("completed_steps", 0)) for record in attempts.values()
            ),
        },
        "phases": [
            {"name": "mechanism_preflight_train", "outcome": "completed"},
            {"name": "fixed_region_eval", "outcome": "completed"},
            {"name": "model_only_checkpoints", "outcome": "completed"},
            {"name": "mechanism_screen_eligibility", "outcome": "completed"},
        ],
        "data_lineage": {
            "training_data": "fineweb10b-train-shards",
            "fixed_region_eval": ",".join(MECHANISM_SCREEN_REQUIRED_REGIONS),
            "matched_seed": str(seed),
        },
        "checkpoint_lineage": {
            "checkpoint_policy": "final-model-only-per-arm",
            "output_checkpoint": "see artifact_index.json",
        },
        "artifacts": artifacts,
        "outcome": _mechanism_preflight_outcome(decision),
    }
    bundle_path = write_native_attempt_bundle(
        Path("experiments/falcon/results/mechanism_preflight/evidence"),
        native_record,
    )
    summary = {
        "attempt_id": attempt_id,
        "bundle_path": str(bundle_path),
        "artifact_dir": str(artifact_dir),
        "evidence_path": str(evidence_path),
        "evidence_digest": evidence_digest,
        "decision_path": str(decision_path),
        "decision_digest": decision_digest,
        "decision": decision.to_dict(),
    }
    _write_json(artifact_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("addition-preflight")
    preflight.add_argument("--arm", required=True)
    preflight.add_argument("--seed", type=int, required=True)
    preflight.add_argument("--steps", type=int, required=True)
    preflight.add_argument("--results-root", type=Path, default=_RESULTS_ROOT)
    preflight.add_argument("--artifact-root", type=Path, default=_ARTIFACT_ROOT)
    preflight.add_argument("--master-port", type=int, default=29612)
    publish = subparsers.add_parser("publish-addition-preflight")
    publish.add_argument("--artifact-dir", required=True, type=Path)
    publish.add_argument("--results-root", type=Path, default=_RESULTS_ROOT)
    mechanism_preflight = subparsers.add_parser("mechanism-screen-preflight")
    mechanism_preflight.add_argument("--seed", type=int, default=0)
    mechanism_preflight.add_argument(
        "--steps", type=int, default=_MECHANISM_PREFLIGHT_STEPS
    )
    mechanism_preflight.add_argument(
        "--artifact-root", type=Path, default=_MECHANISM_PREFLIGHT_ROOT
    )
    mechanism_preflight.add_argument("--master-port", type=int, default=29640)
    args = parser.parse_args(argv)

    if args.command == "addition-preflight":
        payload = run_addition_preflight(
            arm=args.arm,
            seed=args.seed,
            steps=args.steps,
            results_root=args.results_root,
            artifact_root=args.artifact_root,
            master_port=args.master_port,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "publish-addition-preflight":
        payload = publish_existing_addition_preflight(
            artifact_dir=args.artifact_dir,
            results_root=args.results_root,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "mechanism-screen-preflight":
        payload = run_mechanism_screen_preflight(
            seed=args.seed,
            steps=args.steps,
            artifact_root=args.artifact_root,
            master_port=args.master_port,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
