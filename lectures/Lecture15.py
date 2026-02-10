# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's topics:
# * Autoencoder architecture
# * Latent spaces

# %% [markdown]
# # Autoencoder

# %% [markdown]
# ## Concepts
#
# An autoencoder is a type of neural network that is used for unsupervised learning, particularly in the domain of dimensionality reduction. The idea behind an autoencoder is to learn a compressed representation of some input data, and then use that compressed representation to generate a reconstruction of the original data. In essence, the network "encodes" the input data into a compressed representation, and then "decodes" that representation back into a reconstructed version of the original data.
#
# <img src="../lectures/assets/lecture15_autoencoder.jpg" alt="Schematic of an Autoencoder showing Encoder, Bottleneck, and Decoder" width=600>
#
# An autoencoder typically consists of two main parts: an **encoder** and a **decoder**. The encoder is a neural network that takes the input data and maps it to a lower-dimensional representation, while the decoder is a network that takes the lower-dimensional representation and maps it back to the original data space. The encoder and decoder are often symmetric, meaning that the encoder and decoder architectures are mirrored around the middle layer.
# *However, in practice the decoder may benefit from deeper networks.*
#
# <img src="../lectures/assets/lecture15_latent_space.jpg" alt="Visualization of a latent space with mapped data points" width=600>
#
# During training, an autoencoder tries to minimize the reconstruction error between the original input data and the reconstructed output data. This is typically done using a loss function like mean squared error or binary cross-entropy. By minimizing this reconstruction error, the autoencoder learns a compressed representation of the data that is able to capture the most important features of the data.
#

# %% [markdown]
# Autoencoders have a wide range of applications, including in image and signal processing, feature extraction, anomaly detection, and data denoising. They can also be used for generative tasks, such as generating new data that is similar to the input data.
#
# <img src="../lectures/assets/lecture15_cvae_latent.jpg" alt="Latent space visualization of a Conditional VAE" width=600>
#
# How does this work?
# In short, there are not `28 x 28 = 784` unique pieces of information in these images.
#
# <img src="../lectures/assets/lecture15_mnist.jpg" alt="Samples from the MNIST dataset of handwritten digits" width=600>

# %% [markdown]
# ## A simple example
#
# We will return to the polynomial example for simplicity.
# In future lectures we will implement autoencoders for images using the same principles but with CNNs.
# Let's create some sample polynomials with 5 basis functions (the Legendre polynomials).

# %%
import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

npoly = 5
a = 2*(rng.rand(npoly) - 0.5)
x = np.linspace(-1, 1, 101)

def f(x):
    y = np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)
    return y

fig, ax = plt.subplots()
ax.plot(x, f(x), '.')

# %% [markdown]
# These are 101-dimensional data artifacts, but they really only represent a 5-dimensional space (i.e., coefficients of the basis set).
# I select this example because there is an analytical relationship between the high-dimensional objects and the low-dimensional coefficients.

# %%
import tqdm

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

# %% [markdown]
# Now we'll use this as our dataset:

# %%
print( y.shape )

# %% [markdown]
# ## Implementation
#
# We'll implement a simple autoencoder using `pytorch_lightning`.
# We need to start by installing it:

# %%
# !pip install pytorch_lightning

# %% [markdown]
# Then we can define our `pl.LightningModule`.
# It will utilize a `MLP` submodule (implemented as a `nn.Module`).

# %%
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


# %% [markdown]
# This can be reused for both our encoder and decoder:

# %%
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


# %% [markdown]
# Let's set up a train/test split as we usually do (using indices instead of arrays or tensors):

# %%
from sklearn import model_selection

train_idx, val_idx = model_selection.train_test_split(np.arange(y.shape[0]), random_state=0)

# %% [markdown]
# And the `DataLoader` objects:

# %%
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

# %% [markdown]
# Finally, we will set up the model and train with `pytorch_lightning`:

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

torch.manual_seed(0)

model = Autoencoder(yt.shape[1], a.shape[1], [64, 32, 16], act=nn.LeakyReLU())
out = model(yt[:1])

logger = CSVLogger("logs")  # create a csv with log information so we can plot it later

trainer = pl.Trainer(max_epochs=50, logger=logger, log_every_n_steps=len(dl_train))
trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_val)

# %% [markdown]
# Now we can read the logs from file:

# %%
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

# %% [markdown]
# And plot the results:

# %%
fig, ax = plt.subplots()

ax.plot(df['epoch'], df['train_loss'], '.', label='Train')
ax.plot(df['epoch'], df['validation_loss'], '.', label='Validation')

ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()

# %% [markdown]
# We can generate the reconstruction to "see what the model sees" like so:

# %%
idx = 0

with torch.no_grad():
    out = model(yt[[idx]]).detach().numpy()

fig, ax = plt.subplots()
ax.plot(y[idx], label='Data')
ax.plot(*out, label='Reconstruction')
ax.legend()

# %% [markdown]
# The discrepancy is the source of the nonzero reconstruction loss in the train and validation curves.

# %% [markdown]
# # Latent spaces
#
# In machine learning and deep learning, a latent space is a compressed and abstract representation of the input data that is learned by a neural network during training. The term "latent" refers to the fact that this representation is not directly observable or measurable, but is inferred or deduced from the input data.
#
# The process of learning a latent space involves training a neural network, such as an autoencoder, to encode the input data into a lower-dimensional representation, and then decode this representation back into the original input data. The compressed representation that is learned by the network is often referred to as the encoding or latent space.
#
# The latent space is typically much smaller than the input space, which means that it represents a highly compressed and abstract version of the input data. However, the latent space is also designed to preserve important information about the input data, such as its structure, patterns, and relationships.
#
# The latent space can be thought of as a way to represent the essential features or attributes of the input data in a compact and efficient way. By using a latent space, it is possible to perform tasks such as data compression, data denoising, and data augmentation, as well as generative modeling and unsupervised learning.

# %% [markdown]
# ## Evaluation
#
# Let's start by plotting the first two dimensions of our latent space.
# We'll access it using the `model.encoder` module:

# %%
with torch.no_grad():
    z = model.encoder(yt).detach().numpy()

fig, ax = plt.subplots()
ax.scatter(z[:, 0], z[:, 1])

# %% [markdown]
# $z$ is more than 2 dimensional, so we can evaluate the entire latent space this way:

# %%
fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2)

# %% [markdown]
# On its own this doesn't tell us much.
# Let's try to relate it back to the coefficients of the polynomials using color in our scatter plot.

# %%
with torch.no_grad():
    z = model.encoder(yt).detach().numpy()

fig, ax = plt.subplots()
ax.scatter(z[:, 0], z[:, 1], c=a[:, 0])

# %% [markdown]
# We should see that there is a gradient in the color, indicating that the first coefficient is related to the learned latent space.
#
# If we look at another coefficient, it won't necessarily share the same correlation:

# %%
fig, ax = plt.subplots()
ax.scatter(z[:, 0], z[:, 1], c=a[:, 1])

# %% [markdown]
# It's possible that another combination of $z$ coordinates does correlate with the $a_1$ coefficient though:

# %%
fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2, c=a[:, 1])

# %% [markdown]
# ## Visualization
#
# It can be helpful to visualize the data samples directly in the latent space.

# %%
scale = 0.2

fig, ax = plt.subplots(figsize=(16, 16))

for i, this_y in enumerate(y):
    ax.plot(z[i, 0]+x*scale, z[i, 1]+this_y*scale)

ax.set_aspect('equal')

# %% [markdown]
# This is quite crowded, so it can be helpful to downselect using clustering techniques.
# In this case I use KMeans clustering simply to define uniformly sized groups in the $(z_0, z_1)$ space.

# %%
from sklearn import cluster

z_idx = (0, 1)
km = cluster.KMeans(n_clusters=64).fit(z[:, z_idx])

cluster_ids = []
for c in km.cluster_centers_:
    cluster_ids.append( np.argmin( np.linalg.norm(z[:, z_idx] - c, axis=1) ) )

scale = 0.5

fig, ax = plt.subplots(figsize=(16, 16))

for i in cluster_ids:
    ax.plot(z[i, z_idx[0]]+x*scale, z[i, z_idx[1]]+y[i]*scale)

ax.set_aspect('equal')

# %% [markdown]
# ## [Check your understanding]
#
# * Plot the reconstructed functions instead of the original data.
# * Investigate some other hyperplanes aside from $(z_0, z_1)$

# %%

# %% [markdown]
# ## Concept vectors
#
# Concept vectors, also known as concept embeddings or distributed representations, are a type of vector space model used in natural language processing and machine learning. The basic idea behind concept vectors is to represent words or concepts as vectors in a high-dimensional vector space, such that words or concepts that are semantically similar are close together in the vector space.
#
# The concept vector model is based on the distributional hypothesis, which states that words that appear in similar contexts tend to have similar meanings. To create concept vectors, a large corpus of text is analyzed to identify patterns in the co-occurrence of words. These patterns are used to construct a high-dimensional vector space, where each word is represented as a vector in the space.
#
# In this vector space, each dimension represents a different feature or context of the words. For example, one dimension may represent the frequency of the word in the corpus, while another dimension may represent the frequency of the word in a particular context.
#
# <img src="../lectures/assets/lecture15_latent_arithmetic.jpg" alt="Example of vector arithmetic in latent space for generating faces" width=600>
#
# Once the concept vectors are constructed, they can be used for various natural language processing tasks, such as word similarity, document classification, and sentiment analysis. In word similarity tasks, the similarity between two words is measured by the cosine similarity between their corresponding concept vectors. In document classification, the concept vectors of the words in a document are averaged to create a document vector, which is then used to classify the document.
#
# Concept vectors are a powerful tool for natural language processing and machine learning, as they provide a way to represent the meaning of words and concepts in a way that is both computationally efficient and semantically meaningful. They have been used in many successful applications, such as language translation, speech recognition, and information retrieval.

# %% [markdown]
# <img src="../lectures/assets/lecture15_gan_architecture.jpg" alt="Architecture of a Generative Adversarial Network" width=600>

# %% [markdown]
# With the benefit of known regression labels, we can actually fit the concept vectors directly using linear regression:

# %%
from sklearn import linear_model

lr = linear_model.LinearRegression().fit(z, a[:, [0]])
print(f'R2 = {lr.score(z, a[:, 0]):.3f}; coef = {lr.coef_}')

# %% [markdown]
# Once we have this vector, we can project the latent code onto the concept using a dot product:

# %%
z_proj = np.dot(z, lr.coef_.T)

fig, ax = plt.subplots()
ax.plot(z_proj, a[:, 0], '.')
ax.set_xlabel('$z \cdot v$')
ax.set_ylabel('$A_0$')

# %% [markdown]
# ## [Check your understanding]
#
# Find the concept vector for each polynomial coefficient.
# Do they all have strong correlation?
# Should we expect them to?

# %%
