import numpy as np
import torch
import cv2
from PIL import Image

class SSL_Move(object):

    def __init__(self):
        self.angles=np.array([0,90,180,270])
        self.degrees=np.array([1,10,20,30,40,50,60,70,80])


    def __call__(self, sample):
        rand = np.array(torch.randint(36, [1]).item(), dtype=np.int)

        angle = self.angles[rand%4]
        degree = self.degrees[rand//4]
        image = np.array(sample)

        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        return {'data': Image.fromarray(blurred),
                'label': rand}