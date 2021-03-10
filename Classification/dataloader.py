from transformations import Normalize, Compose
from dataset import make_dataset, ClassificationDataset, BalancedBatchSampler
from torch.utils.data import DataLoader
import numpy as np

import pathlib

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext+'/**/*.npy') if file.is_file()]
    return filenames
root = pathlib.Path('/data/dk/datasets_CROPS/crops_fixed_scale_uint8/valid')

# class_to_idx = {'FP': 0, 'LAD': 1, 'LCX': 2, 'RCA': 3, 'LM': 4}
class_to_idx = ['bengin','TP','negative','nipple']
train_samples = make_dataset(root, class_to_idx, extensions='npy')
print(len(train_samples))
dataset_train = ClassificationDataset(train_samples)
print(len(dataset_train))
train_sampler = BalancedBatchSampler(dataset_train)
# dataloader_train = DataLoader(dataset=dataset_train, sampler=train_sampler, batch_size=4)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True)
#
#
x,y = next(iter(dataloader_train))

print(x)
print(y)
#
# print(f'x = shape: {x}; type: {x.dtype}')
# print(f'y = shape: {y}; type: {y.dtype}')
# y_r = np.reshape(y,(4,-1))
# print(y_r.shape)
# print(y_r.sum(axis=1))
