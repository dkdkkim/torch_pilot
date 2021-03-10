import pathlib, os, datetime, json
from transformations import Normalize, Compose, Unsqueeze
from dataset import SegmentationDataset
import torch
from model_detection import Model
from trainer import Trainer
from torch.utils.data import DataLoader
from visual import plot_training

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


root = pathlib.Path('/data/dk/datasets_CROPS/crops_fixed_scale_uint8/')

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext+'/**/*0.5_0.npy') if file.is_file()]
    return filenames

inputs_train = get_filenames_of_path(root, 'train/TP/5')
targets_train = [pathlib.Path(str(inp).replace('crops_fixed_scale_uint8','crops_lbl_fixed_rev')) for inp in inputs_train]

inputs_valid = get_filenames_of_path(root, 'valid/TP/5')
targets_valid = [pathlib.Path(str(inp).replace('crops_fixed_scale_uint8','crops_lbl_fixed_rev')) for inp in inputs_valid]

transforms_training = Compose([Normalize(), Unsqueeze()])
transforms_validation = Compose([Normalize(), Unsqueeze()])

dataset_train = SegmentationDataset(inputs=inputs_train, targets=targets_train, transform=transforms_training)
dataset_valid = SegmentationDataset(inputs=inputs_valid, targets=targets_valid, transform=transforms_validation)

dataloder_training = DataLoader(dataset=dataset_train, batch_size=2, shuffle=True)
dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=2, shuffle=True)

model = Model(in_ch=1, classes=2).cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

trainer = Trainer(model=model, device=torch.device('cuda'), criterion=criterion, optimizer=optimizer, training_DataLoader=dataloder_training,
                    validation_DataLoader=dataloader_validation, lr_scheduler=None, epochs=3, epoch=0)

training_losses, validation_losses, lr_rates = trainer.run_trainer()
if not os.path.exists(f"/data/dk/exp/torch_seg_test/"):
    os.makedirs(f"/data/dk/exp/torch_seg_test/")
with open(f"/data/dk/exp/torch_seg_test/training_losses.json",'w') as file:
    json.dump(training_losses,file)
with open(f"/data/dk/exp/torch_seg_test/validation_losses.json",'w') as file:
    json.dump(validation_losses,file)
with open(f"/data/dk/exp/torch_seg_test/lr_rates.json",'w') as file:
    json.dump(lr_rates,file)
plot_training(training_losses, validation_losses, lr_rates, gausian=True, sigma=1, figsize=(10,4))

# model_name = f"/data/dk/exp/torch_seg_test/model_{datetime.datetime.now().strftime('%y%m%d_%H%M')}.pt"
model_name = f"/data/dk/exp/torch_seg_test/model_test.pt"
torch.save(model.state_dict(), pathlib.Path.cwd() / model_name)




