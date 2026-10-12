#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CLI guards for the modded-nanogpt B200 harness."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOTFS_WORKSPACE = Path("/workspace/torchtitan")
ROOTFS_SENTINEL = ROOTFS_WORKSPACE / "scripts/rootfs/enter_rootfs.sh"


def rootfs_main_error(wrapper: str) -> str | None:
    if os.environ.get("TORCHTITAN_IN_ROOTFS") != "1":
        return f"must run through {wrapper}; TORCHTITAN_IN_ROOTFS=1 is missing"
    if Path.cwd() != ROOTFS_WORKSPACE:
        return f"expected rootfs workspace {ROOTFS_WORKSPACE}, found {Path.cwd()}"
    if not ROOTFS_SENTINEL.exists():
        return f"rootfs workspace sentinel is missing: {ROOTFS_SENTINEL}"
    return None


def guard_rootfs_cli(wrapper: str) -> int | None:
    error = rootfs_main_error(wrapper)
    if error is None:
        return None
    print(f"error: {error}", file=sys.stderr)
    return 21
