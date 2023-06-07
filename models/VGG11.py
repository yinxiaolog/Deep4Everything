import torch
from torch import Tensor
from torch import nn


class VGG11(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.sequential = None
        self.lr = config['train']['lr']
        self.conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        self.conv_blks = []
        self.vgg()

    def vgg(self):
        in_channels = 1
        for (num_convs, out_channels) in self.conv_arch:
            self.conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.sequential = nn.Sequential(
            *self.conv_blks, nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def vgg_block(self, num_conv, in_channels, out_channels):
        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)

    def loss_func(self):
        return nn.CrossEntropyLoss()

    def optim(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

# X = torch.randn(size=(1, 1, 224, 224))
# net = VGG11({
#     "train": {
#         "lr": 0.1
#     }
# }).sequential
#
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)
