import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

# constants
n, d = 100000, 100
prior = Normal(0, 1) # prior on logistic regression parameters
EPOCHS = 100 

# generate data
from simulate_data import logistic_regr
torch.manual_seed(1)
X, y, beta_true = logistic_regr(n, d)

# function to sample from q using reparam method
def sample_q_normal(Lambda):
    mu = Lambda[:, 0]
    sigma = F.softplus(Lambda[:, 1])

    epsilon = torch.randn((mu.size()))
    z = mu + epsilon * sigma

    return z, mu, sigma


class LogisticRegression(nn.Module):
    """Bayesian Logistic Regression with VI"""
    def __init__(self, d):
        super(LogisticRegression, self).__init__()
        self.d = d
        self.Lambda = nn.Parameter(torch.randn((d, 2)))
        self.mu = self.Lambda[:, 0]
        self.sigma = F.softplus(self.Lambda[:, 1])
        self.z = self.Lambda[:, 0]

    def forward(self, input):
        self.mu = self.Lambda[:, 0]
        self.sigma = F.softplus(self.Lambda[:, 1])

        epsilon = torch.randn((self.mu.size()))
        self.z = self.mu + epsilon * self.sigma

        logits = input.mm(self.z.view(self.d, 1))

        return logits


class Elbo(nn.Module):
    def __init__(self, model):
        super(Elbo, self).__init__()
        self.model = model

    def forward(self, yhat, y):
        z, mu, sigma = self.model.z, self.model.mu, self.model.sigma

        log_lik = Bernoulli(logits=yhat).log_prob(y).sum()
        log_prior = prior.log_prob(z).sum()
        entropy = Normal(mu, sigma).log_prob(z).sum()

        return entropy - log_lik - log_prior


model = LogisticRegression(d)
criterion = Elbo(model)
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(EPOCHS):
    opt.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    opt.step()

    if epoch % 1 == 0:
        print(epoch, loss.item())

