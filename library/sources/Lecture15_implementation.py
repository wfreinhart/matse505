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
# id: Lecture15_implementation
# type: Foundational
# parent_lecture: Lecture15
# ---
#
# ## Implementation
#
# We'll implement a simple autoencoder using `pytorch_lightning`.
# We need to start by installing it:
#
# Then we can define our `pl.LightningModule`.
# It will utilize a `MLP` submodule (implemented as a `nn.Module`).
#
# This can be reused for both our encoder and decoder:
#
# Let's set up a train/test split as we usually do (using indices instead of arrays or tensors):
#
# And the `DataLoader` objects:
#
# Finally, we will set up the model and train with `pytorch_lightning`:
#
# Now we can read the logs from file:
#
# And plot the results:
#
# We can generate the reconstruction to "see what the model sees" like so:
#
# The discrepancy is the source of the nonzero reconstruction loss in the train and validation curves.

# %%
# !pip install pytorch_lightning

import torch
import torch.nn as nn

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

import pytorch_lightning as pl

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

train_idx, val_idx = model_selection.train_test_split(np.arange(y.shape[0]), random_state=0)

from torch.utils.data import DataLoader

yt = torch.tensor(y).float()
at = torch.tensor(a).float()

# only needs to implement two methods:
# __len__(self) and __getitem__(self, index)
# -- list type works for this!
ds_train = [(yt[i], at[i]) for i in train_idx]
ds_val = [(yt[i], at[i]) for i in val_idx]

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=64, shuffle=False)
for batch, _ in dl_train:
    print(batch.shape)
    break

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

torch.manual_seed(0)

model = Autoencoder(yt.shape[1], a.shape[1], [64, 32, 16], act=nn.LeakyReLU())
out = model(yt[:1])

logger = CSVLogger("logs")  # create a csv with log information so we can plot it later

trainer = pl.Trainer(max_epochs=50, logger=logger, log_every_n_steps=len(dl_train))
trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_val)

import pandas as pd

log_path = 'logs/lightning_logs/version_0/metrics.csv'
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
else:
    colab_path = '/content/logs/lightning_logs/version_0/metrics.csv'
    if os.path.exists(colab_path):
        df = pd.read_csv(colab_path)
    else:
        print(f"Warning: Log file not found at {log_path} or {colab_path}")
        df = pd.DataFrame(columns=['epoch', 'train_loss', 'validation_loss'])
df.head()

fig, ax = plt.subplots()

ax.plot(df['epoch'], df['train_loss'], '.', label='Train')
ax.plot(df['epoch'], df['validation_loss'], '.', label='Validation')

ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()

idx = 0

with torch.no_grad():
    out = model(yt[[idx]]).detach().numpy()

fig, ax = plt.subplots()
ax.plot(y[idx], label='Data')
ax.plot(*out, label='Reconstruction')
ax.legend()
