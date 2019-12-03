"""
Data process for jigsaw pretext task
"""

import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from itertools import permutations

def jigsaw_transform(img, pattern, dimension):
    (n, m) = img.size

    height_step = int(n / dimension)
    weight_step = int(m / dimension)

    dst = Image.new('RGB', (m, n))
    for i, ind in enumerate(pattern):
        left = ind % dimension * weight_step
        top = ind // dimension * height_step
        right = left + weight_step
        bottom = top + height_step

        tmp = img.crop((left, top, right, bottom))
        dst.paste(tmp, (i % dimension * weight_step, i // dimension * height_step))
    return dst

class SSL_IMG_JIGSAW(object):
    def __init__(self, indices, dimension=2):
        self.indices = set(indices)
        self.dimension = dimension
        self.all_permutations = list(permutations(list(range(self.dimension**2))))

    def __call__(self, sample):
        data, labels = [], []
        indices = self.indices if self.indices else set(range(len(sample["data"])))
        for idx in range(len(sample["data"])):
            if idx in indices:
                label = torch.randint(len(self.all_permutations), [1]).item()
                pattern = self.all_permutations[label]
                img = jigsaw_transform(sample["data"][idx], pattern, self.dimension)
                data.append(img)
                labels.append(label)

        sample["data"] = data
        sample["label"] = labels

        return sample
