import torch
import numpy as np
from torch.utils import data


class SegmentationDataset(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        x,y = np.load(input_ID), np.load(target_ID)

        if self.transform is not None:
            x, y = self.transform(x,y)

        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y