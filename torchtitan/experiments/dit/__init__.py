# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import DiTModel
from .trainer import DiTTrainer

__all__ = ["DiTModel", "DiTTrainer", "dit_configs", "model_registry"]


def parallelize_dit(
    model: DiTModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    del training
    if parallel_dims.dp_enabled or parallel_dims.tp_enabled or parallel_dims.pp_enabled:
        raise ValueError(
            "The DiT synthetic experiment currently supports only single-rank "
            "training. Use NGPU=1 and leave data/tensor/pipeline parallelism at 1."
        )
    if parallelism.spmd_backend != "default":
        raise ValueError("The DiT synthetic experiment requires spmd_backend=default.")

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)
    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)
    return model


def _dit_debug() -> DiTModel.Config:
    return DiTModel.Config(
        latent_channels=4,
        latent_height=16,
        latent_width=16,
        patch_size=2,
        hidden_size=192,
        num_layers=4,
        num_heads=3,
        mlp_ratio=4.0,
        num_classes=10,
    )


dit_configs = {
    "debug": _dit_debug,
}


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    if converters:
        raise ValueError("DiT experiment does not currently support model converters.")
    return ModelSpec(
        name="dit",
        flavor=flavor,
        model=dit_configs[flavor](),
        parallelize_fn=parallelize_dit,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
