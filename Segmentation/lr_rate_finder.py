import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import math


class LearningRateFinder:
    """
    Train a model using different learning rates within a range to find the optimal learning rate
    """

    def __init(self,
               model: nn.Module,
               criterion,
               optimizer,
               device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def fit(self,
            data_loader: torch.utils.data.DataLoader,
            steps=100,
            min_lr=1e-7,
            max_lr=0.1,
            constant_increment=False):
        """
        Trains the model for number of steps using varied learning rate and store the statistics
        """
        self.loss_history = {}
        self.model.train()
        current_lr = min_lr
        steps_counter = 0
        epochs = math.ceil(steps / len(data_loader))

        progressbar = trange(epochs, desc='Progress')
        for epoch in progressbar