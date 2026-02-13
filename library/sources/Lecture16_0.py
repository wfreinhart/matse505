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
# id: Lecture16_0
# type: Foundational
# parent_lecture: Lecture16
# ---
#
#
#
# Today's topics:
# * Generative modeling
# * Variational Autoencoder
#
# To start with we need to repeat the dataset generation and model training from last time:
#
# We'll regenerate the Legendre polynomial dataset used last time:
#
# Then use the same model definitions:
#
# And finally create the train and validation datasets and then use them to train the model with pytorch-lightning.
#
# We were visualizing the 5D latent space according to the ground truth $a_0$ coefficient:

# %%
# !pip install pytorch_lightning

import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
import tqdm

rng = np.random.RandomState(0)

npoly = 5
a = 2*(rng.rand(npoly) - 0.5)
x = np.linspace(-1, 1, 101)

def f(x):
    y = np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)
    return y

fig, ax = plt.subplots()
ax.plot(x, f(x), '.')

rng = np.random.RandomState(0)

npoly = 5
x = np.linspace(-1, 1, 101)

def f(x, a):
    return np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)

def make_function():
    a = 2*(rng.rand(npoly) - 0.5)
    return a, f(x, a)

# need to make a whole bunch of functions!
N_samples = 1000
a = np.zeros([N_samples, npoly])
y = np.zeros([N_samples, len(x)])
for i in tqdm.tqdm(np.arange(N_samples)):
    this_a, this_f = make_function()
    a[i] = this_a
    y[i] = this_f

import torch
import torch.nn as nn
import pytorch_lightning as pl

class MLP(nn.Module):
    def __init__(self, hidden_dim, act):
        super(MLP, self).__init__()
        self.act = act
        layers = []
        for h in hidden_dim[:-1]:
            layers.append(nn.LazyLinear(h))
            layers.append(self.act)
        layers.append(nn.LazyLinear(hidden_dim[-1]))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x

class Autoencoder(pl.LightningModule):
    def __init__(self, input_size, latent_dim, hidden_dim, act=nn.ReLU()):
        super().__init__()

        self.encoder = MLP(list(hidden_dim)+[latent_dim], act)
        self.decoder = MLP(list(hidden_dim)[::-1]+[input_size], act)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out = self(x)
        loss = self.criterion(out, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out = self(x)
        loss = self.criterion(out, x)
        self.log('validation_loss', loss)
        return loss

from sklearn import model_selection
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

train_idx, val_idx = model_selection.train_test_split(np.arange(y.shape[0]), random_state=0)

yt = torch.tensor(y).float()
at = torch.tensor(a).float()

ds_train = [(yt[i], at[i]) for i in train_idx]
ds_val = [(yt[i], at[i]) for i in val_idx]

dl_train = DataLoader(ds_train, batch_size=128, shuffle=True)

torch.manual_seed(0)

model = Autoencoder(yt.shape[1], a.shape[1], [128, 64, 32, 16], act=nn.LeakyReLU())
out = model(yt[:1])

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model=model, train_dataloaders=dl_train)

with torch.no_grad():
    z = model.encoder(yt).detach().numpy()

fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2, c=a[:, 0])
