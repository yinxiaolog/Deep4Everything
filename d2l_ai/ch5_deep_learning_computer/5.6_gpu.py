import time

import torch
from d2l_ai import torch as d2l
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))

print(torch.cuda.device_count())

print(d2l.try_all_gpus())
print(d2l.try_gpu(2))

x = torch.tensor([1, 2, 3])
print(x.device)

X = torch.ones(2, 3, device=d2l.try_gpu(1))
print(X)

Y = torch.rand(2, 3, device=d2l.try_gpu(1))
print(Y)

Z = X.cuda(1)
print(X)
print(Z)

print(Y + Z)

print(Z.cuda(1) is Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=d2l.try_gpu(1))
print(net)

print(net(X))

print(net[0].weight.data.device)

start = time.time()
for i in range(1000):
    A = torch.randn(100, 100, device=d2l.try_gpu(1))
    B = torch.randn(100, 100, device=d2l.try_gpu(1))
    C = torch.mm(A, B)
print(torch.normal(C))

end = time.time()
print((end - start))
