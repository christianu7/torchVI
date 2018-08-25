import torch
from torch.distributions.bernoulli import Bernoulli

# generate data
n, d = 100000, 100

# create data
torch.manual_seed(1)
X = torch.randn(n, d)
X[:, 0] = 1.
beta_true = torch.randn(d)
y = Bernoulli(logits=X.mm(beta_true.view(d, 1))).sample() 

