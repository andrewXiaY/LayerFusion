import torch.nn as nn
from .utils import Flatten


class VGG_A(nn.Module):
    def __init__(self):
        super(VGG_A, self).__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        flatten = Flatten()
        
        self._feature_blocks = nn.ModuleList([conv1, pool1, conv2, pool2, conv3, conv4, pool3,
                                              conv5, conv6, pool4, conv7, conv8, pool5])
        self.all_feat_names = ["conv1", "pool1", "conv2", "pool2", "conv3", "conv4", "pool3",
                               "conv5", "conv6", "pool4", "conv7", "conv8", "pool5"]

        assert len(self.all_feat_names) == len(self._feature_blocks)

    def forward(self, x, out_feat_keys=None):
        for f in self._feature_blocks:
            x = f(x)

        return x
