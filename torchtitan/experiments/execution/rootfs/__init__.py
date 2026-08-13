# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rootfs identity, content-addressed store, and atomic selection (roadmap 8).

This subpackage owns the host-testable logic behind the hardened rootfs build
and activation contract. Docker, bwrap, and the live filesystem stay in the
shell launcher/builder; the pure-Python model here computes the content-address
of a build from its locked inputs, validates the build manifest, and manages an
atomic selection record with retained prior selection, rollback, and quarantine.
"""
