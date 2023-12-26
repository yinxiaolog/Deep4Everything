import matplotlib.pyplot as plt
import torch
from d2l_ai import torch as d2l
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6
counts = multinomial.Multinomial(10000000, fair_probs).sample()
print(counts / 10000000)

counts = multinomial.Multinomial(30, fair_probs).sample((5000,))
cum_counts = counts.cumsum(dim=0)
print(cum_counts.sum(dim=1, keepdims=True))
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
plt.show()
