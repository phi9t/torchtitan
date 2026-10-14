# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU validation helper for the isolated GLM EP matrix."""

from __future__ import annotations

import argparse
import json

from torchtitan.config.loader import ConfigLoader
from torchtitan.distributed.parallelism_context import (
    DistributedTopology,
    ParallelismContext,
)


def validate_config(
    *,
    config_name: str,
    world_size: int,
    ep: int,
) -> dict[str, object]:
    config = ConfigLoader._load_config(
        "torchtitan.experiments.glm53_flash_ep.config_registry",
        config_name,
    )
    config.parallelism.expert_parallel_degree = ep
    config.model.set_sharding_(config.parallelism)
    context = ParallelismContext.from_config(
        config.parallelism,
        DistributedTopology(world_size=world_size),
        dump_folder=config.dump_folder,
    )
    return {
        "config": config_name,
        "world_size": world_size,
        "expert_parallel_degree": ep,
        "dp_shard": context.dp_shard,
        "edp_shard": context.dp_shard * context.cp * context.tp // context.ep,
        "num_experts": config.model.glm_config.num_experts,
        "experts_per_ep_rank": config.model.glm_config.num_experts // ep,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=["glm53_flash_ep_debugmodel", "glm53_flash_ep_mini"],
    )
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--ep", type=int, action="append", required=True)
    args = parser.parse_args()

    results = [
        validate_config(config_name=config_name, world_size=args.world_size, ep=ep)
        for config_name in args.config
        for ep in args.ep
    ]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
