import re
import torch
import numpy as np
from torch.utils.data import Dataset


class OddMNIST(Dataset):
    """Odd Mnist dataset."""

    def __init__(self, root):
        self.sample_paths = list(root.glob("**/*.npy"))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.sample_paths[idx]
        groups = re.match("processed_data/(\w+)/(\d+)/.*npy", str(path)).groups()
        label = int(groups[1])
        return np.load(path), label
