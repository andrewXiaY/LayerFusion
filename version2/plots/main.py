import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
from torchvision import utils

sys.path.append("..")

from models.alex_net import AlexNet
from models.fusion_net import FusionNet
from configs import *

def imshow(img, ind):    
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig(f"dsa_{ind}.png")

task = "pretrained_classification_r_jig"
path = "../checkpoints/" + task + "/model_50.pth"

parameters = [
    {"model_name": "alex_net", "path": "../checkpoints/ssl_rotate/model_20.pth", "out_features": 4},
    {"model_name": "alex_net", "path": "../checkpoints/ssl_jigsaw/model_20.pth", "out_features": 24}
]

model = FusionNet(parameters)
model.load_state_dict(torch.load(path))

model.eval()

kernels = model.fc_layers[-1][0].weight.detach()
for i in kernels:
    print(i.size())
#imshow(utils.make_grid(kernels.reshape(10, 1, , 16), padding = 1, normalize=True, scale_each=True), 1)