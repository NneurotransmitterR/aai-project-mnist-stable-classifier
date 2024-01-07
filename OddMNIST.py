import re
import torch
import os
import numpy as np
from torch.utils.data import Dataset


class OddMNIST(Dataset):
    """Odd Mnist dataset."""

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.sample_paths = list(root.glob("**/*.npy"))
        self.train = train
        self.root = root
        self.data, self.targets = self._load_data()

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.sample_paths[idx]
        groups = re.match("processed_data/(\w+)/(\d+)/.*npy", str(path)).groups()
        label = int(groups[1])
        return np.load(path), label

    def _load_data(self):
        data_dir = self.root
        data_list = []
        label_list = []
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                # Iterate over each file in the subdirectory
                for file in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file)
                    if file.endswith('.npy'):
                        # Load the numpy array and convert it to a Tensor
                        npy_data = np.load(file_path)
                        tensor_data = torch.from_numpy(npy_data)
                        data_list.append(tensor_data)
                        label_list.append(int(label))

        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list)
        return data_tensor, label_tensor
