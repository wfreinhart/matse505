# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# > Presentations next week, due by start of class
# 
# Today's topics:
# * Peer review
# * Convolutional AutoEncoder

# %% [markdown]
# # Peer review of presentation plan

# %% [markdown]
# Worksheet [presentation plan worksheet](https://colab.research.google.com/drive/1IMtQneMv-aRgbfRkAisUvKRUWqiDvWE4?usp=sharing)
# 
# Work in pairs.
# Review each other's plan and then report a summary back to the class.
# 
# You will have 8 minutes total for discussion and about 30 seconds each to report out.
# 
# We will repeat twice so you will see 2 other students' project plans.

# %% [markdown]
# # Convolutional AutoEncoder
# 
# <img src="../lectures/assets/lecture17_unet_architecture.jpg" alt="Architecture of the U-Net for semantic segmentation" width=600>

# %% [markdown]
# ## Convolutional Transpose Layer
# 
# In order to create an autoencoder for convolutional networks, we need to be able to "de-compress" or "de-convolve" an image.
# 
# > [!NOTE]
# > The transposed convolution visualization from ResearchGate is currently unavailable due to access restrictions.
# 
# A **convolutional transpose layer** is a type of layer in a convolutional neural network that is used for upsampling or increasing the resolution of an input image or feature map.
# It is also sometimes referred to as a "deconvolutional layer," but this is not accurate since the convolutional transpose is *not an inverse of the convolution*.

# %% [markdown]
# The main hyperparameters of a convolutional transpose layer include:
# 
# * Output channels: The number of channels in the output feature map. This is equivalent to the number of filters used in a standard convolutional layer.
# 
# * Kernel size: The size of the convolutional filter that is applied during the transpose operation.
# 
# * Stride: The amount of pixels by which the convolutional filter is moved during the transpose operation.
# 
# * Padding: The amount of padding that is added to the input image or feature map to ensure that the output size matches the desired size.
# 
# * Output padding: The number of pixels to concatenate to the last row and column of the output image.

# %% [markdown]
# In convolution, padding and stride are used to control the size of the output feature map. In convolutional transpose, padding and stride are used to control the size of the output feature map as well as the amount of upsampling that is applied to the input.

# %% [markdown]
# ## Dataset

# %%
import zipfile, requests

url = 'https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/wfr5091_psu_edu/ERXYsfbOP4dGm7_M4oIh-0gBV3Ix19fKuSndDu4Ui6zHrQ?e=HOCHNN&download=1'
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open('data.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

zip_file = zipfile.ZipFile('data.zip')
zip_file.extractall('/content/')
zip_file.close()

# %%
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def rgb2grey(img):
    return img[0].unsqueeze(0)

transform = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),   # Convert the image to a PyTorch tensor
    transforms.Lambda(lambda x: rgb2grey(x))  # Convert the RGB image to greyscale
])

ds_train = datasets.ImageFolder('NEU-DET-SP/train', transform=transform)
dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
print(f'Number of training images (all classes): {len(ds_train)}')

ds_valid = datasets.ImageFolder('NEU-DET-SP/validation', transform=transform)
dl_valid = DataLoader(ds_valid, batch_size=64, shuffle=False)
print(f'Number of validation images (all classes): {len(ds_valid)}')

# %% [markdown]
# ## Implementation

# %%
!pip install pytorch-lightning

# %% [markdown]
# We'll start with the Encoder that uses `Conv2d` layers like we did before:

# %%
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

# %% [markdown]
# Then we have to make a Decoder that uses `ConvTranspose2d` layers:

# %%
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

# %% [markdown]
# Finally we combine these into a single `ConvAutoencoder` class:

# %%
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

# %% [markdown]
# Now we can train with `pytorch-lightning`:
# > Note these parameters are chosen very carefully and you will have to change padding, stride, etc. in order to make a different number of layers work.

# %%
model = ConvAutoencoder(conv_channels=[8, 6, 4], latent_dim=32,
                        deconv_channels=[4, 8, 1], img_size=224)

with torch.no_grad():  # initialize lazy layer sizes
    x, y = dl_train.dataset[0]
    out = model(x.unsqueeze(0))

trainer = pl.Trainer(max_epochs=50, accelerator="gpu")
trainer.fit(model, dl_train)

# %% [markdown]
# ## Evaluation

# %%
for name, loader in [('train', dl_train), ('valid', dl_valid)]:

    batch_loss = 0
    for batch in loader:
        x, _ = batch
        with torch.no_grad():
            x_hat = model(x)
        loss = model.criterion(x_hat, x)
        batch_loss += loss.item() / len(batch)
    print(f'MSE loss for {name} set: {batch_loss}')

# %% [markdown]
# ### Reconstruction

# %%
from matplotlib import pyplot as plt

i = -1
x, _ = dl_valid.dataset[i]
x_hat = model(x.unsqueeze(0))[0]

fig, axes = plt.subplots(1, 2)
for i, arr in enumerate([x, x_hat]):
    axes[i].imshow(arr.detach().squeeze(0).numpy(), 'Greys_r')

# %% [markdown]
# ### Interpolation
# 
# This is a linear interpolation in latent space:

# %%
import numpy as np

n_interp = 8
fig, axes = plt.subplots(1, n_interp+2, figsize=(12, 4))

i, j = (0, 35)
x, _ = dl_valid.dataset[i]
x_hat = model(x.unsqueeze(0))[0]
axes[0].imshow(x.detach().squeeze(0).numpy(), 'Greys_r')

x, _ = dl_valid.dataset[i]
z_i = model.encoder(x.unsqueeze(0))

x, _ = dl_valid.dataset[j]
z_j = model.encoder(x.unsqueeze(0))
axes[-1].imshow(x.detach().squeeze(0).numpy(), 'Greys_r')

for i, alpha in enumerate(np.linspace(0, 1, n_interp)):
    arr = model.decoder(z_i + alpha * (z_j - z_i))[0]
    axes[i+1].imshow(arr.detach().squeeze(0).numpy(), 'Greys_r')

# %% [markdown]
# Compare to a linear interpolation in real space:

# %%
fig, axes = plt.subplots(1, n_interp, figsize=(10, 4))

i, j = (0, 35)
x_i, _ = dl_valid.dataset[i]
x_j, _ = dl_valid.dataset[j]

for i, alpha in enumerate(np.linspace(0, 1, n_interp)):
    arr = x_i + alpha * (x_j - x_i)
    axes[i].imshow(arr.detach().squeeze(0).numpy(), 'Greys_r')

# %% [markdown]
# ## Latent space

# %%
import numpy as np

all_y = []
all_z = []
for x, y in dl_valid:
    all_y.append(y)
    with torch.no_grad():
        z = model.encoder(x)
    all_z.append(z.detach().numpy())

y = np.hstack(all_y)  # classes
z = np.vstack(all_z)  # latent codes

# %% [markdown]
# Perform a PCA embedding of the high-dimensional latent space:

# %%
from sklearn import decomposition

pca = decomposition.PCA()
coefs = pca.fit_transform(z)

# plot the explained variance of the embedding
fig, ax = plt.subplots()
_ = ax.plot(np.arange(1, pca.n_components_+1), np.cumsum(pca.explained_variance_ratio_), '.-')
_ = ax.set_xlabel('Components')
_ = ax.set_ylabel('Explained Variance')
ax.set_xscale('log')

# plot the class labels
fig, ax = plt.subplots()
ax.scatter(coefs[:, 0], coefs[:, 1], c=y)

# %% [markdown]
# Plot examples in the space:

# %%
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import cluster

# perform the clustering
km = cluster.KMeans(n_clusters=48, n_init='auto', random_state=0).fit(coefs[:, :2])

# determine which sample IDs are closest to the cluster centers
center_id = []
for i, c in enumerate(km.cluster_centers_):
    dist = np.linalg.norm(c - coefs[:, :2], axis=1)
    center_id.append( np.argmin(dist) )

# plot the results
fig, ax = plt.subplots(figsize=(10, 10))
_ = ax.scatter(*coefs[:, :2].T, c=y)
_ = ax.set_xlabel('PC 1')
_ = ax.set_xlabel('PC 2')
ax.set_aspect('equal')
# add image thumbnails as annotations to the scatter plot
thumbnail_size = (64, 64)
for id in center_id:
    img_tensor, _ = dl_valid.dataset[id]
    img = img_tensor.numpy().transpose()
    image_offset = OffsetImage(img, zoom=0.15, cmap='Greys_r')
    ab = AnnotationBbox(image_offset, coefs[id, :2], xycoords='data', frameon=False)
    ax.add_artist(ab)

# %% [markdown]
# Plot the reconstructions instead to "see what the model sees":

# %%
# plot the results
fig, ax = plt.subplots(figsize=(10, 10))
_ = ax.scatter(*coefs[:, :2].T, c=y)
_ = ax.set_xlabel('PC 1')
_ = ax.set_xlabel('PC 2')
ax.set_aspect('equal')
# add image thumbnails as annotations to the scatter plot
thumbnail_size = (64, 64)
for id in center_id:
    x, _ = dl_valid.dataset[id]
    img_tensor = model(x.unsqueeze(0)).squeeze(0)
    img = img_tensor.detach().numpy().transpose()
    image_offset = OffsetImage(img, zoom=0.15, cmap='Greys_r')
    ab = AnnotationBbox(image_offset, coefs[id, :2], xycoords='data', frameon=False)
    ax.add_artist(ab)

# %% [markdown]
# ## Final comments
# 
# These results show that "vanilla" AutoEncoder does not work well for image domain.
# There are *many* advances in generative models for images that can improve the results.
# Examples are adversarial training, transformers, and diffusion models.
# These require a lot of effort to implement and involve nuanced decisions about the transformations and distributions.

# %%

