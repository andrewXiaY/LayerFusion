import numpy as np
import torch
from PIL import Image



class SSL_COLOR(object):

    def __init__(self, N=3):
        self.N=N


    def __call__(self, sample):
        rand=np.random.randint(3*self.N)
        color=rand%3
        degree=rand//3
        pixels = np.array(sample)

        if color is 0:
            pixels[:, :, 0] = np.round(pixels[:, :, 0]*(degree/(self.N - 1)))
        if color is 1:
            pixels[:, :, 1] = np.round(pixels[:, :, 1]*(degree/(self.N - 1)))
        if color is 2:
            pixels[:, :, 2] = np.round(pixels[:, :, 2]*(degree/(self.N - 1)))


        return {'data': Image.fromarray(pixels), 'label': rand}