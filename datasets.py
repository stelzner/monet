import torch
from torch.utils.data import Dataset
import numpy as np

class Sprites(Dataset):
    def __init__(self, np_file, train=True, transform=None):
        self.transform = transform
        data = np.load(np_file)
        self.images = data['x_train'] if train else data['x_test']
        # self.images = np.transpose(self.images, (0, 3, 1, 2))
        self.counts = data['count_train'] if train else data['count_test']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.counts[idx]

