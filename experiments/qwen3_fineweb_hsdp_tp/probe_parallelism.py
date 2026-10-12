# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only DeviceMesh / DTensor introspection for the HSDP+TP experiment.

Launch under torchrun with the same world size the experiment uses::

    NGPU=8 torchrun --nproc_per_node=8 \
        experiments/qwen3_fineweb_hsdp_tp/probe_parallelism.py \
        --parallelism.data_parallel_replicate_degree=2 \
        --parallelism.data_parallel_shard_degree=2 \
        --parallelism.tensor_parallel_degree=2

The script builds the real torchtitan trainer (so the model is parallelized by
exactly the production code path), then prints, on rank 0:

  1. The world mesh axis names and every enabled single-axis submesh size,
     via ``ParallelDims.get_all_one_dimensional_meshes()``.
  2. For a sample of parameters (attention wq/wo, FFN w1/w2, tok_embeddings,
     lm_head): ``param.placements`` and ``param.device_mesh.mesh_dim_names``,
     showing TP ``Shard`` composed with FSDP ``Shard(0)`` over the dense mesh.
  3. A synthetic total-norm reduction over a two-axis-sharded DTensor to
     surface the ``_NormPartial`` -> ``full_tensor()`` redistribution that the
     real grad-norm clip performs (mirrors distributed/utils.py:614-627).

Everything here uses read-only DTensor / DeviceMesh APIs. No weights are
trained and no config is mutated beyond what is passed on the command line.
"""

import os
import sys

# Under torchrun a bare script path puts the script's own directory on
# sys.path[0], not the repo root, so ``import torchtitan`` fails. Prepend the
# repo root (two levels up) so the local torchtitan package is importable
# regardless of launch directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.config import ConfigManager


IS_RANK0 = os.environ.get("RANK", "0") == "0"


def p(*args: object) -> None:
    """Print only on rank 0, flushing so torchrun tee ordering is stable."""
    if IS_RANK0:
        print(*args, flush=True)


def hr(title: str) -> None:
    p("\n" + "=" * 78)
    p(title)
    p("=" * 78)


def describe_param(name: str, param: torch.nn.Parameter) -> None:
    """Print the sharding of a single parameter."""
    if isinstance(param, DTensor):
        mesh = param.device_mesh
        axes = list(mesh.mesh_dim_names or ())
        placements = [str(pl) for pl in param.placements]
        # Zip mesh axis name -> placement so the composition is legible.
        pairs = ", ".join(f"{a}={pl}" for a, pl in zip(axes, placements))
        p(
            f"  {name:<34} global{tuple(param.shape)} "
            f"local{tuple(param.to_local().shape)}\n"
            f"  {'':<34} mesh_axes={axes} placements=[{pairs}]"
        )
    else:
        p(f"  {name:<34} (plain tensor, not a DTensor) shape={tuple(param.shape)}")


def find_first(model: torch.nn.Module, suffixes: tuple[str, ...]) -> None:
    """Print the first parameter whose qualified name ends with any suffix."""
    for suffix in suffixes:
        for name, param in model.named_parameters():
            if name.endswith(suffix):
                describe_param(name, param)
                return
    p(f"  (no parameter matching any of {suffixes})")


def probe_norm_partial(mesh) -> None:
    """Reproduce the ``_NormPartial`` grad-norm redistribution in miniature.

    We build a DTensor that is ``Shard(0)`` across the dense ``[dp_replicate,
    fsdp, tp]`` mesh -- the same layout FSDP2 leaves parameter gradients in --
    then take its p-norm. ``torch.linalg.vector_norm`` on a sharded DTensor
    yields a DTensor whose placements are ``_NormPartial`` on the sharded axes
    (each rank holds its shard's partial norm). Calling ``full_tensor()`` then
    all-reduces across those axes to materialize the scalar norm, emitting the
    "N sequential all_reduce" warning. This is exactly what
    ``distributed/utils.py`` ``clip_grad_norm_`` does at lines 614-627:
    ``get_total_norm`` builds the ``_NormPartial`` and ``full_tensor()``
    collapses it. The warning names the axes (``[dp_replicate, fsdp, tp]``)
    that get reduced in sequence.
    """
    from torch.distributed.tensor import distribute_tensor, Shard

    axes = list(mesh.mesh_dim_names or ())
    # A tensor big enough to shard on dim 0 across the full mesh world size.
    world = mesh.size()
    local = torch.arange(world * 4, dtype=torch.float32, device="cuda").reshape(
        world * 4, 1
    )
    # Shard(0) on every axis of the multi-axis dense mesh mirrors how a flat
    # FSDP+TP parameter gradient is laid out (sharded on fsdp, and on tp for
    # colwise weights). distribute_tensor takes one placement per mesh axis.
    placements = [Shard(0) for _ in axes]
    dt = distribute_tensor(local, mesh, placements)

    # vector_norm on a sharded DTensor leaves the result in _NormPartial state,
    # which we can inspect BEFORE it is reduced. This mirrors what
    # get_total_norm produces internally in clip_grad_norm_.
    partial_norm = torch.linalg.vector_norm(dt, ord=2.0)
    p(f"  vector_norm returned type={type(partial_norm).__name__}")
    if isinstance(partial_norm, DTensor):
        p(f"  placements (partial, pre-reduction): {list(partial_norm.placements)}")
        p(f"  mesh axes: {list(partial_norm.device_mesh.mesh_dim_names or ())}")
        # full_tensor() collapses the _NormPartial across every sharded axis.
        reduced = partial_norm.full_tensor()
        p(
            f"  full_tensor() reduces over {len(partial_norm.placements)} axes "
            f"-> scalar norm = {reduced.item():.6f}"
        )
        p(
            "  -> full_tensor() issues one all_reduce per _NormPartial axis, "
            "i.e. the sequential [dp_replicate, fsdp, tp] reduction whose "
            "warning appears above. This is the grad-norm collective the "
            "article traces."
        )
    else:
        p(f"  norm is a plain tensor: {partial_norm.item():.6f}")

    # Also exercise the exact API clip_grad_norm_ calls, so the observed
    # behavior matches the training path 1:1.
    total_norm = torch.nn.utils.get_total_norm([dt], norm_type=2.0)
    p(
        f"  get_total_norm([dt]) -> type={type(total_norm).__name__}, "
        f"value={total_norm.full_tensor().item() if isinstance(total_norm, DTensor) else total_norm.item():.6f}"
    )


def main() -> None:
    # Parse the same CLI surface as run_train.sh so the mesh + sharding match
    # the experiment exactly. torchrun sets --module/--config indirectly here;
    # we require them on the command line just like run_train.sh forwards them.
    argv = sys.argv[1:]
    if "--module" not in argv:
        argv = ["--module", "qwen3"] + argv
    if "--config" not in argv:
        argv = ["--config", "qwen3_debugmodel_fineweb"] + argv
    # Keep it to a single step; we never call train(), but a valid config is
    # required and this keeps any incidental setup cheap.
    argv = argv + ["--training.steps", "1"]

    config = ConfigManager().parse_args(argv)

    # Building the trainer runs the production path: ParallelDims.build_mesh(),
    # model meta-init, and parallelize_qwen3() (TP sharding + FSDP2 wrapping).
    # The pyrefly ignore mirrors torchtitan/train.py: the parsed config is
    # dynamically typed, so .build() is not statically resolvable.
    trainer = config.build()  # pyrefly: ignore [missing-attribute]

    try:
        parallel_dims = trainer.parallel_dims
        assert len(trainer.model_parts) == 1
        model = trainer.model_parts[0]

        hr("1. Device mesh: axis names and enabled single-axis submesh sizes")
        world_mesh = parallel_dims.world_mesh
        p(f"world_mesh.mesh_dim_names = {list(world_mesh.mesh_dim_names or ())}")
        p(f"world_mesh.size()         = {world_mesh.size()}")
        one_d = parallel_dims.get_all_one_dimensional_meshes()
        p("Enabled one-dimensional meshes (name -> size):")
        for name, mesh in one_d.items():
            p(f"  {name:<14} size={mesh.size()}")
        p(
            "Expected for dp_replicate=2 x dp_shard=2 x tp=2: "
            "batch=4, loss=4, dp_replicate=2, fsdp=2, tp=2"
        )

        hr("2. Sampled parameter placements (TP Shard composed with FSDP Shard)")
        p("tok_embeddings.weight (colwise on vocab; tied to lm_head here):")
        find_first(model, ("tok_embeddings.weight",))
        p("attention QKV projection (colwise, TP Shard on out dim):")
        # The debug model fuses QKV into a single wqkv; the non-fused variant
        # exposes wq. Try both so this probe works across flavors.
        find_first(
            model,
            (
                "attention.qkv_linear.wqkv.weight",
                "attention.qkv_linear.wq.weight",
                "attention.wq.weight",
            ),
        )
        p("attention output projection wo (rowwise, TP Shard on in dim):")
        find_first(model, ("attention.wo.weight",))
        p("FFN w1 (colwise):")
        find_first(model, ("feed_forward.w1.weight",))
        p("FFN w2 (rowwise):")
        find_first(model, ("feed_forward.w2.weight",))
        p("lm_head.weight (colwise on vocab; absent when weight-tied):")
        find_first(model, ("lm_head.weight", "output.weight"))

        hr("3. _NormPartial grad-norm redistribution (synthetic)")
        # Use the dense mesh that FSDP+TP parameters actually live on.
        dense_mesh = parallel_dims.get_optional_mesh(["dp_replicate", "fsdp", "tp"])
        if dense_mesh is None:
            # Pure-FSDP or TP-only run: fall back to whatever multi-axis dense
            # mesh exists, else a single sharded axis.
            dense_mesh = parallel_dims.get_optional_mesh(["fsdp", "tp"])
        if dense_mesh is None:
            dense_mesh = parallel_dims.get_optional_mesh(
                "fsdp"
            ) or parallel_dims.get_mesh("tp")
        p(f"Using dense mesh axes: {list(dense_mesh.mesh_dim_names or ())}")
        probe_norm_partial(dense_mesh)

        hr("probe complete")
    finally:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
