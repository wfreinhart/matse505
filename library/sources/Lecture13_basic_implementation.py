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
# id: Lecture13_basic_implementation
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Basic implementation
#
# We'll start by installing `pytorch-lightning` to simplify the training procedure.
#
# Remember that the basic features of a CNN are the convolutions, activations, and pooling, with fully connected layers at the end.
# We will design a `ConvBlock` object to avoid repeating the first three several times:

# %%
# !pip install pytorch-lightning

import torch.nn as nn
from torch import optim
import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ClassifierCNN(pl.LightningModule):
    def __init__(self, conv_channels, fc_dim, num_classes):
        super(ClassifierCNN, self).__init__()

        conv_blocks = []
        for i, c in enumerate(conv_channels):
            conv_blocks.append( ConvBlock(c) )

        self.conv = nn.Sequential(*conv_blocks)
        self.fc = nn.Sequential(nn.LazyLinear(fc_dim),
                                nn.LeakyReLU(),
                                nn.LazyLinear(num_classes))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)  # flatten the output for FC layer
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
