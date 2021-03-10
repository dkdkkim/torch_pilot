from transformations import Normalize, Compose
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import numpy as np

import pathlib

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext+'/**/*.npy') if file.is_file()]
    return filenames
root = pathlib.Path('/data/dk/datasets_CROPS/crops_fixed_scale_uint8')
inputs = get_filenames_of_path(root, 'valid/TP')
targets = [pathlib.Path(str(inp).replace('crops_fixed_scale_uint8','crops_lbl_fixed_rev')) for inp in inputs]
transforms = Compose([Normalize()])
dataset_valid = SegmentationDataset(inputs=inputs, targets=targets, transform=transforms)
dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=4, shuffle=True)

x,y = next(iter(dataloader_valid))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'y = shape: {y.shape}; type: {y.dtype}')
y_r = np.reshape(y,(4,-1))
print(y_r.shape)
print(y_r.sum(axis=1))
