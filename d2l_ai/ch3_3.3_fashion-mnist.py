import torchvision
from d2l_ai import torch as d2l
from torch.utils import data
from torchvision import transforms

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True
)

print(len(mnist_train), len(mnist_test))
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=d2l.get_fashion_mnist_labels(y))
d2l.plt.show()

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} secs')
train_iter, test_iter = d2l.load_data_fashion_mnist(32, 64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    d2l.show_images(X.reshape(32, 64, 64), 4, 8, titles=d2l.get_fashion_mnist_labels(y))
    d2l.plt.show()
    break
