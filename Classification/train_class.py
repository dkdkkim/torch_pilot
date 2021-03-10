import pathlib, os, datetime, json
from transformations import Normalize, Compose, Unsqueeze, CenterCrop
from dataset import make_dataset, ClassificationDataset, BalancedBatchSampler
import torch
from densenet3D import densenet121
from trainer import Trainer
from torch.utils.data import DataLoader
from visual import plot_training

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


root = pathlib.Path('/data/dk/datasets_CROPS/crops_fixed_scale_uint8/')

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext+'/**/*0.5_0.npy') if file.is_file()]
    return filenames

root = pathlib.Path('/data/dk/datasets_CROPS/crops_fixed_scale_uint8')
class_to_idx = ['TP','negative']

train_samples = make_dataset(root / 'valid', class_to_idx, extensions='0.5_0.npy')
valid_samples = make_dataset(root / 'valid', class_to_idx, extensions='0.5_0.npy')

dataset_train = ClassificationDataset(train_samples)
dataset_valid = ClassificationDataset(valid_samples)

train_sampler = BalancedBatchSampler(dataset_train)

dataloader_train = DataLoader(dataset=dataset_train, sampler=train_sampler, batch_size=4)
dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=4, shuffle=True)

model = densenet121().cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

trainer = Trainer(model=model, device=torch.device('cuda'), criterion=criterion, optimizer=optimizer, training_DataLoader=dataloader_train,
                    validation_DataLoader=dataloader_valid, lr_scheduler=None, epochs=2, epoch=0)

training_losses, validation_losses, lr_rates = trainer.run_trainer()
with open(f"/data/dk/exp/torch_cls_test/training_losses.json",'w') as file:
    json.dump(training_losses,file)
with open(f"/data/dk/exp/torch_cls_test/validation_losses.json",'w') as file:
    json.dump(validation_losses,file)
with open(f"/data/dk/exp/torch_cls_test/lr_rates.json",'w') as file:
    json.dump(lr_rates,file)
plot_training(training_losses, validation_losses, lr_rates, gausian=True, sigma=1, figsize=(10,4))

# model_name = f"/data/dk/exp/torch_seg_test/model_{datetime.datetime.now().strftime('%y%m%d_%H%M')}.pt"
model_name = f"/data/dk/exp/torch_seg_test/model_test.pt"
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)




