from .ssl_jigsaw import SSL_IMG_JIGSAW
from .ssl_rotate import SSL_IMG_ROTATE
from .ssl_moveblur import SSL_Move
from .ssl_exchange_pos import SSL_EXCHANGE_POS
from .ssl_noise_add import SSL_NOISE_ADD
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

        try:
            label = torch.from_numpy(label)
        except Exception as e:
            pass

        return {'data': image,
                'label': label}



TRANSFORMS = {"ssl_rotate": SSL_IMG_ROTATE,
              "ssl_jigsaw": SSL_IMG_JIGSAW,
              "ssl_moveblur": SSL_Move,
              "ssl_exchange_pos": SSL_EXCHANGE_POS,
              "ssl_noise_add": SSL_NOISE_ADD,
              "to_tensor": ToTensor,
              "Normalize": Normalize,
              "default": None,
              "ssl_rel_patch_loc": None,
              "ssl_exemplar": None}

        
