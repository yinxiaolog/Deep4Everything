import torch
import numpy as np
from torch import nn

x = torch.tensor(0.0, requires_grad=True)
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x], lr=0.01)


def f(x):
    result = a * torch.pow(x, 2) + b * x + c
    return result


for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()

print("y=",f(x).data,";","x=",x.data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x


net = Net()
print(net)

model = net
model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
model.abcd = nn.CrossEntropyLoss()

class Foo():
    a = 1

foo = Foo()
m = foo
m.b = 3
m.c = 4

if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)