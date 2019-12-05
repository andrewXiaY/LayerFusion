
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, out_features = 4):
        super(AlexNet, self).__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2), # => 43 * 43 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2) # => 21 * 21
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # => 21 * 21
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2) # => 10 * 10
    
        
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # => 10 * 10
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # => 10 * 10
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # => 10 * 10
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=3, stride=2) # => 4 * 4 

        flatten = nn.Sequential(nn.Flatten())

        fc1 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
        )

        fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
        )

        fc3 = nn.Sequential(
            nn.Linear(1024, out_features),
            nn.ReLU(inplace=True)
        )

        
        self._feature_blocks = nn.ModuleList(
            [conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, flatten, fc1, fc2, fc3]
        )
        self.all_feat_names = [
            "conv1", "pool1", "conv2", "pool2", "conv3", "conv4", 
            "conv5", "pool5", "flatten", "fc1", "fc2", "fc3"
        ]

        assert len(self.all_feat_names) == len(self._feature_blocks)

    def forward(self, x):     
        for layer in self._feature_blocks:
            x = layer(x)
        return x
