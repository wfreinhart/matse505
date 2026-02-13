# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# ---
# id: Lecture12_0
# type: Foundational
# parent_lecture: Lecture12
# ---
#
#
#
# Today's topics:
# * streamlining `pytorch` code
# * saving and loading models
# * using `DataLoader` objects
# * `pytorch-lightning` workflows
#
# Let's work with our visual example from last time:
#
# We implemented a shallow neural network by defining a `torch.nn.Module` like so:
#
# It was trained like this:
#
# Finally, we evaluated it like this:
#
# We observed that `ReLU` leads to a piecewise linear function while `tanh` would produce smoother curves.

# %%
from scipy.special import legendre
import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng(0)

npoly = 5
a = 2*(rng.random(npoly) - 0.5)
xl = np.linspace(-1, 1, 101)

def f(x):
    y = np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)
    return y

ns = 200
xls = (rng.random(ns) - 0.5)*2  # draws samples uniformly on [-1, 1]
yls = f(xls)

fig, ax = plt.subplots()
_ = ax.plot(xl, f(xl), label='Function')
_ = ax.plot(xls, yls, '.', label='Data')
_ = ax.legend()

import torch
from torch import nn

class MLPRegressor(torch.nn.Module):

    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,1)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

import tqdm

torch.manual_seed(0)

# make torch tensors from np arrays
xlt = torch.Tensor(xls.reshape(-1, 1))
ylt = torch.Tensor(yls.reshape(-1, 1))

# define the model
model = MLPRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)

# do the training
pbar = tqdm.tqdm(np.arange(400))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(xlt)
    # Compute Loss
    loss = criterion(y_pred, ylt)

    pbar.set_postfix_str(f'train loss: {loss.item():.3e}')
    # Backward pass
    loss.backward()
    optimizer.step()

with torch.no_grad():
    model_in = torch.tensor(xl.reshape(-1, 1), dtype=torch.float)
    model_out = model(model_in).detach().numpy()

fig, ax = plt.subplots()
_ = ax.plot(xl, f(xl), label='Function')
_ = ax.plot(xls, f(xls), '.', label='Data')
_ = ax.plot(xl, model_out, label='NN')
_ = ax.legend()
