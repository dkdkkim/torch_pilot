import torch, os, random
import numpy as np
from torch.utils import data
from glob import glob
def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            # return has_file_allowed_extension(x, extensions)
            return x.lower().endswith(extensions)
    # for target_class in sorted(class_to_idx.keys()):
    for target_class in sorted(class_to_idx):
        # class_index = class_to_idx[target_class] if target_class=='FP' else 1
        class_index = 0 if target_class in ['negative','nipple'] else 1 # two class setting
        target_dirs = glob(directory + '/%s'%target_class)
        # target_dir = os.path.join(directory, target_class)
        for target_dir in target_dirs:
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
    return instances

class ClassificationDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        # self.data_dirs = glob(root_dir + '/*.npy')
        # self.data_dirs = root_dir
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.root_dir)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.root_dir[idx][1]
        return (self.root_dir[idx][0], label)
    # def getitem_valid(self, batch):
    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         # print(idx)
    #         idx = idx.tolist()
    #     # print("CACdataset, ", idx)
    #     crop = np.load(self.root_dir[idx][0])
    #     crop = np.expand_dims(crop[crop.shape[0]//2-20:crop.shape[0]//2+20], axis=0)
    #     crop = crop.astype(np.float)
    #     crop /= 1400.
    #     crop = torch.from_numpy(crop).float()
    #     crop = Variable(crop, requires_grad=False)
    #
    #     label = self.root_dir[idx][1]
    #
    #     return (crop, label)

class BalancedBatchSampler(data.sampler.Sampler):
    def __init__(self, dataset, labels=None, shuffle=True):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        self.shuffle = shuffle
        for idx in range(len(dataset)):
            # print("batchsampler, ", idx)
            label = dataset[idx][1]
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        # print(0, len(self.dataset[0]))
        # print(1, len(self.dataset[1]))
        self.keys = list(self.dataset.keys())
        self.currentKey = 0
        self.indices = [-1] * len(self.keys)
        # print(self.indices)
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.dataset[0])
            random.shuffle(self.dataset[1])
            # random.shuffle(self.dataset[2])
            # random.shuffle(self.dataset[3])
        while self.indices[self.currentKey] < self.balanced_max - 1:
            self.indices[self.currentKey] += 1
            yield self.dataset[self.keys[self.currentKey]][self.indices[self.currentKey]]
            self.currentKey = (self.currentKey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)
    def __len__(self):
        return self.balanced_max * len(self.keys)