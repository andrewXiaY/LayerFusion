from .ssl_jigsaw import SSL_IMG_JIGSAW
from .ssl_rotate import SSL_IMG_ROTATE
import numpy as np
import torch
from torchvision import transforms
class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['data'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.asarray(image, dtype=np.float32)
        image = image.transpose((2, 0, 1))
        return {'data': image,
                'label': label}

class Normalize(object):
    def __init__(self, par):
        self.norm = transforms.Normalize(par[0], par[1])

    def __call__(self, sample):
        image, label = sample['data'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image /= 255
        image = torch.from_numpy(image)
        image = self.norm(image)

        return {'data': image,
                'label': torch.from_numpy(label)}

TRANSFORMS = {"ssl_rotate": SSL_IMG_ROTATE,
              "ssl_jigsaw": SSL_IMG_JIGSAW,
              "to_tensor": ToTensor,
              "Normalize": Normalize,
              "ssl_rel_patch_loc": None,
              "ssl_exemplar": None}

        
