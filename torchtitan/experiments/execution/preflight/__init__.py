# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable doctor profiles and benchmark-semantic preflight (roadmap 9-11).

This subpackage splits generic execution-prerequisite doctor clauses (Section
10) from task-owned semantic preflight (Section 11), and models the composable
execution profiles (Section 9) so an optional package's absence affects only
the profiles that declare it.
"""
