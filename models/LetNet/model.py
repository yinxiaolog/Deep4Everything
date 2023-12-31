import torch
from torch import nn

import utils.common as common
import config


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = config.hyper_params['lr']
        self.weight_decay = 0
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 40), nn.Sigmoid(),
            nn.Linear(40, 10)
        )

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
