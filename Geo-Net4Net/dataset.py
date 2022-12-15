import os
import numpy as np
import re

import torch
from torch.utils.data import Dataset

class FCDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        super(FCDataset, self).__init__()
        self.base_path = data_dir
        self.data_path = [os.path.join(self.base_path, name) for name in
                          sorted(os.listdir(data_dir), key=lambda name: int(re.findall("\d+", name)[0]))]

        self.labels = torch.from_numpy(np.loadtxt(label_dir, dtype=int)).expand(len(self.data_path), -1).long() \
            if isinstance(label_dir, str) else label_dir
        self.n_data = len(self.data_path)
        print(self.n_data)
        print(data_dir)
        if isinstance(label_dir, str):
            print(label_dir)

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        data = torch.from_numpy(np.load(self.data_path[idx]))
        data += 1e-3 * torch.eye(data.shape[-1])
        label = self.labels[idx]

        return data, label

class ReassignedFCDataset(Dataset):
    def __init__(self, dataset, pseudo_labels):
        super(ReassignedFCDataset, self).__init__()
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return *self.dataset[idx], self.pseudo_labels[idx]