"""
Data process for exchange position pretext task
"""

import torch
import time
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from itertools import permutations

def exchange_pos(img, kernel_size, t1, t2):
    hs1, ws1 = t1
    hs2, ws2 = t2
    tmp1 = np.copy(img[hs1: hs1 + kernel_size, ws1: ws1 + kernel_size])
    tmp2 = np.copy(img[hs2: hs2 + kernel_size, ws2: ws2 + kernel_size])
    img[hs1: hs1 + kernel_size, ws1: ws1 + kernel_size] = tmp2
    img[hs2: hs2 + kernel_size, ws2: ws2 + kernel_size] = tmp1
    return img

class SSL_EXCHANGE_POS(object):
    def __init__(self, dimension=5, kernel_size=9):
        self.dimension = dimension
        self.kernel_size = kernel_size

    def __call__(self, sample):
        label = np.array(torch.randint(self.dimension, [1]).item(), dtype=np.int)
        img = np.array(sample)
        height, width = img.shape[0], img.shape[1]
        hs1, ws1 = np.random.randint(height - self.kernel_size), np.random.randint(width - self.kernel_size)
        hs2, ws2 = np.random.randint(height - self.kernel_size), np.random.randint(width - self.kernel_size)
        img = exchange_pos(img, self.kernel_size, (hs1, ws1), (hs2, ws2))

        return {"data" : img, "label" : np.array(sample, dtype=np.float32).flatten()}


