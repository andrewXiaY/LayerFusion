"""
Data process for exchange position pretext task
"""

import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

def add_noise(img, mode):
    if mode == "gaussian":
        row, col, ch= img.shape
        mean, var = 0, 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row,col,ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss

    elif mode == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)

    elif mode =="speckle":
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = img + img * gauss

    elif mode == "s&p":
        row,col,ch = img.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[coords] = 0
        noisy = out

    return noisy        


class SSL_NOISE_ADD(object):
    def __init__(self):
        self.mode = ["gaussian", "s&p", "poisson", "speckle"]
    
    def __call__(self, sample):
        label = np.random.randint(4)
        mode = self.mode[label]
        img = np.array(sample)
        img = add_noise(img, mode)
        return {"data" : img, "label" : label}