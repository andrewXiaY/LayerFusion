from torch.utils.data import Dataset
import torch
from . import TRANSFORMS
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

DEFAULT_GRAY_IMG_SIZE = 96

def get_mean_image(crop_size):
    img = Image.fromarray(128 * np.ones((crop_size, crop_size, 3), dtype=np.uint8))
    return img

class DiskImageDataset(Dataset):
    """Base Dataset class for loading images from Disk."""
    def __init__(self, path, ssl_transform, end=-1):
        assert os.path.exists(path)
        self.data_path = []
        for file in os.listdir(path)[:end]:
            if(file.endswith('.png')):
                self.data_path.append(os.path.join(path, file))
        
        self.transform = transforms.Compose([TRANSFORMS[ssl_transform](), 
                                             TRANSFORMS["to_tensor"](),
                                             TRANSFORMS["Normalize"](([0.44671062, 0.43980984, 0.40664645], [0.26034098, 0.25657727, 0.27126738]))])

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        path = self.data_path[idx]
        img = Image.open(path, mode='r').convert("RGB")
    
        if self.transform:
            img = self.transform(img)
        return img # contains data and labels

    def num_samples(self):
        return len(self.data_path)


class DefaultDataset(Dataset):
    def __init__(self, src):
        self.data = np.load(src[0])
        self.labels = np.load(src[1])
        self.transform = transforms.Compose([TRANSFORMS["to_tensor"](),
                                             TRANSFORMS["Normalize"](([0.44671062, 0.43980984, 0.40664645], 
                                                                      [0.26034098, 0.25657727, 0.27126738]
                                                                      ))])
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        path = self.data[idx]
        img = Image.open("/".join(path.split("\\")), mode='r').convert("RGB")
        label = self.labels[idx]
        transformed  = self.transform({"data": img, "label": label})

        return transformed # contains data and labels

    def num_samples(self):
        return self.data.shape[0]