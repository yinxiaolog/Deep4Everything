import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lr = config['train']['lr']
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, 10))

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def loss_func(self):
        return nn.CrossEntropyLoss()

    def optim(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

# X = torch.randn(size=(1, 1, 224, 224))
# net = ResNet({
#     "train": {
#         "lr": 0.1
#     }
# }).net
#
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)

# blk = Residual(3, 6, True, 2)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# print(Y.shape)
