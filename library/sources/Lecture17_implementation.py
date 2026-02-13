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
# id: Lecture17_implementation
# type: Foundational
# parent_lecture: Lecture17
# ---
#
# ## Implementation
#
# We'll start with the Encoder that uses `Conv2d` layers like we did before:
#
# Then we have to make a Decoder that uses `ConvTranspose2d` layers:
#
# Finally we combine these into a single `ConvAutoencoder` class:
#
# Now we can train with `pytorch-lightning`:
# > Note these parameters are chosen very carefully and you will have to change padding, stride, etc. in order to make a different number of layers work.

# %%
# !pip install pytorch-lightning

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(self, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, conv_channels, embedding_dim):
        super(ConvEncoder, self).__init__()

        conv_blocks = []
        for i, c in enumerate(conv_channels):
            conv_blocks.append( ConvBlock(c) )

        self.conv = nn.Sequential(*conv_blocks)

        self.fc = nn.LazyLinear(embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)  # flatten the output for FC layer
        x = self.fc(x)
        return x

class ConvTBlock(nn.Module):
    def __init__(self, out_channels):
        super(ConvTBlock, self).__init__()
        self.conv = nn.LazyConvTranspose2d(out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, deconv_channels, pre_conv_dim):
        super(ConvDecoder, self).__init__()

        self.img_dim = pre_conv_dim
        self.fc = nn.LazyLinear(self.img_dim**2)

        deconv_blocks = []
        for i, c in enumerate(deconv_channels):
            deconv_blocks.append( ConvTBlock(c) )

        self.conv = nn.Sequential(*deconv_blocks)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.img_dim, self.img_dim)  # unflatten the output for conv layers
        x = self.conv(x)
        return x

class ConvAutoencoder(pl.LightningModule):
    def __init__(self, conv_channels, latent_dim, deconv_channels, img_size):
        super(ConvAutoencoder, self).__init__()

        self.encoder = ConvEncoder(conv_channels, latent_dim)

        pre_conv_dim = int( img_size / 2**len(conv_channels) )  # with stride=2, img halves each conv layer
        self.decoder = ConvDecoder(deconv_channels, pre_conv_dim)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        self.log('val_loss', loss)
        return loss

model = ConvAutoencoder(conv_channels=[8, 6, 4], latent_dim=32,
                        deconv_channels=[4, 8, 1], img_size=224)

with torch.no_grad():  # initialize lazy layer sizes
    x, y = dl_train.dataset[0]
    out = model(x.unsqueeze(0))

trainer = pl.Trainer(max_epochs=50, accelerator="gpu")
trainer.fit(model, dl_train)
