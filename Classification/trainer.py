import numpy as np
import torch
from tqdm import tqdm, trange


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0):

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.epoch = epoch

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def load_input(self, inputs):
        arrays = []
        for inp in inputs:
            arrays.append(np.expand_dims(np.load(inp),axis=0))
        output = np.concatenate(arrays)
        output = output / 255.
        output = np.expand_dims(output, axis=1)
        return torch.from_numpy(output).type(torch.float32)

    def train(self):

        self.model.train()
        train_losses = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = self.load_input(x).cuda(), y.cuda()
            self.optimizer.zero_grad()
            out = self.model(input)
            loss = self.criterion(out, target)
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()
            self.optimizer.step()

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def validate(self):

        from tqdm import tqdm, trange

        self.model.eval()
        valid_losses = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = self.load_input(x).cuda(), y.cuda()

            # input, target = x.to(self.device), y.to(self.device)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))

    def run_trainer(self):

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            ''' epoch counter '''
            self.epoch += 1

            ''' Training block '''
            self.train()

            ''' Validation block '''
            self.validate()

            ''' Learning rate scheduler block'''
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLRONPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])
                else:
                    self.lr_scheduler.batch()

        return self.training_loss, self.validation_loss, self.learning_rate



