import torch.nn as nn
from .utils import Flatten


class VGG_A(nn.Module):
    def __init__(self):
        super(VGG_A, self).__init__(output_features)
        self.output_features = output_features
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
        
        fc1 = nn.Sequential(nn.Linear(256, 21),
                            nn.ReLU(inplace=True))
        fc2 = nn.Sequential(nn.Linear(21, 100),
                            nn.ReLU(inplace=True))
        fc3 = nn.Sequential(nn.Linear(20, 230),
                            nn.ReLU(inplace=True))
        outp = nn.Sequential(nn.Linear(230, self.output_features))

        self._feature_blocks = nn.ModuleList([conv1, pool1, conv2, pool2, conv3, conv4, pool3,
                                              conv5, conv6, pool4, conv7, conv8, pool5, fc1, fc2, fc3, outp])
        self.all_feat_names = ["conv1", "pool1", "conv2", "pool2", "conv3", "conv4", "pool3",
                               "conv5", "conv6", "pool4", "conv7", "conv8", "pool5", "fc1", "fc2", "fc3", "output"]

        assert len(self.all_feat_names) == len(self._feature_blocks)

    def forward(self, x, out_feat_keys=None):
        out = {}
        for ind, f in enumerate(self._feature_blocks):
            x = f(x)
            out[ind] = x

        return out
