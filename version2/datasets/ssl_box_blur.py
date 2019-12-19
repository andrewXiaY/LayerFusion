# this is box blur

import numpy as np
import torch
import cv2
from PIL import Image
from PIL import ImageFilter



class SSL_BOX_BLUR(object):

    def __init__(self,radius_max=10):
        self.radius_max=radius_max


    def __call__(self, sample):
        radius=np.random.randint(self.radius_max)
        blurred = sample.filter(ImageFilter.BoxBlur(radius))

        return {'data': blurred,
                'label': radius
        }
