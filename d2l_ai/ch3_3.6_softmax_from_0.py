import torch
from d2l_ai import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


print(cross_entropy(y_hat, y))

lr = 0.1


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# d2l_ai.plt.show()

d2l.predict_ch3(net, test_iter)
d2l.plt.ion()
d2l.plt.show()
