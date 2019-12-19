from PIL import Image
import sys
import cv2

sys.path.append("..")
from datasets import *

img = Image.open("../data/unsupervised_train/3.png")
trans = ssl_noise_add.SSL_NOISE_ADD()


img = trans(img)
cv2.imwrite("noise.png", img["data"])