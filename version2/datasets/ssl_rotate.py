#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torchvision.transforms.functional as TF

class SSL_IMG_ROTATE(object):
    def __init__(self, num_angles=4, num_rotations_per_img=1):
        self.num_angles = num_angles
        self.num_rotations_per_img = num_rotations_per_img
        self.angles = torch.linspace(0, 360, num_angles + 1)[:-1]

    def __call__(self, sample):
        label = np.array(torch.randint(self.num_angles, [1]).item(), dtype=np.float32)
        img = TF.rotate(sample, self.angles[label])
        return {"data": img, "label": label}
