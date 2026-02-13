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
# id: Lecture12_pytorch_lightning
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# # `pytorch-lightning`
#
# PyTorch Lightning is a lightweight wrapper around PyTorch that helps you organize your code and reduce boilerplate, making it easier to write and maintain complex deep learning projects.
# We'll go over how to use PyTorch Lightning to streamline your PyTorch code.
#
# The first step will be installation:
#
# ### `LightningModule`
#
# Let's see how we can streamline `MLPRegressor` code from before using PyTorch Lightning.
# The first thing we need to do is create a PyTorch Lightning module that wraps our existing PyTorch model.
# Here's what that might look like:
#
# As you can see, we've defined a new class `LitMLPRegressor` that extends `pl.LightningModule`.
# We've moved our model definition into this new class, but we've also defined two new methods: `training_step` and `configure_optimizers`.
# The `training_step` method defines what happens in each iteration of the training loop, while the `configure_optimizers` method defines the optimizer used during training.

# %%
# !pip install pytorch-lightning

import pytorch_lightning as pl
from torch import optim


# define the LightningModule
class LitMLPRegressor(pl.LightningModule):
    # the top part looks the same as torch.nn.Module:
    def __init__(self, output_size, hidden_size=(100, ), activation=nn.ReLU()):
        super(LitMLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(nn.LazyLinear(layer_size))
            layers.append(activation)

        layers.append(nn.LazyLinear(output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x

    # the bottom part is for Pytorch Lightning:
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        out = self.forward(x)
        loss = nn.functional.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the lightning model
pl_model = LitMLPRegressor(1, (100, ), activation=nn.Tanh())
