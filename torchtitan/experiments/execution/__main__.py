# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Entry point for ``python -m torchtitan.experiments.execution``."""

from torchtitan.experiments.execution.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
