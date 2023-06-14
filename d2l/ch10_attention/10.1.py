import torch
from d2l import torch as d2l
from torch.nn import functional as F

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
d2l.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
d2l.plt.show()

X = torch.randn((10, 10))
X = F.softmax(X)
X = X.reshape((1, 1, 10, 10))
d2l.show_heatmaps(X, xlabel='Keys', ylabel='Queries')
d2l.plt.show()
