import torch
from torch import Tensor
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lr = config['train']['lr']
        self.sequential = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)

    def loss_func(self):
        return nn.CrossEntropyLoss()

    def optim(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
