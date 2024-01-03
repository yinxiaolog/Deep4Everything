import torch
import torchvision
from torch import nn

import utils.common as common
import config


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = config.hyper_params['lr']
        self.weight_decay = 0
        self.net = torchvision.models.resnet50()
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, features: torch.Tensor):
        return self.net(features)

    @property
    def loss_func(self):
        return nn.CrossEntropyLoss()

    @property
    def optim(self):
        return torch.optim.SGD(self.net.parameters(), lr=self.lr)

    def metric_func(self, dataloader):
        self.net.eval()
        device = next(iter(self.net.parameters())).device
        right = 0
        size = 0
        with torch.no_grad():
            for feature, label in dataloader:
                feature = feature.to(device)
                label = label.to(device)
                pred = self.net(feature)
                right += common.accuracy(pred, label)
                size += label.numel()
        return right / size
