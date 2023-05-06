import torch

from draw.plot import plot

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
print(x.grad)
y = 2 * torch.dot(x, x)
print(y)
y.backward()
print(x.grad)
print(x.grad == 4 * x)
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 2.5.2
x.grad.zero_()
y = x * x
print(y)
y.sum().backward()
# y.backward(torch.ones(len(x)))
print(y)
print(x.grad)

# 2.5.3
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


# 2.5.4
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
print(d)
d.backward()
print(a.grad)
print(a.grad == d / a)

# 2.5.6
x = torch.linspace(0, 6, 50)
x.requires_grad_(True)
y = torch.sin(x)
print(y)
y.sum().backward()
plot(x.detach(), [y.detach(), x.grad], 'x', 'f(x)', ['sin(x)', 'diff by torch'])
