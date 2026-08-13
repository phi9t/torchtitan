# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Typed run-attempt lifecycle for scaffold-to-policy and related experiments.

This package implements the runtime preflight roadmap's typed lifecycle
(Sections 3-7): immutable run declarations with normalized digests, attempts and
their parent linkage, a stage protocol with terminal events, artifact lineage,
and one canonical report input. It is the upstream core run-attempt layout, not
an experiment fork, so it lives at torchtitan/experiments/execution rather than
inside a single task folder.
"""
