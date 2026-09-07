# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

x = torch.randn(256, 256, device="cuda")
for _ in range(20):
    x = x @ x
    x = x / (x.norm() + 1e-6)
torch.cuda.synchronize()
print(float(x[0, 0]))
