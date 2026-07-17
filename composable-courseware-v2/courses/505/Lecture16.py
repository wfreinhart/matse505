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
# Today's topics:
# * Generative modeling
# * Variational Autoencoder

# %% [markdown]
# To start with we need to repeat the dataset generation and model training from last time:

# %%
!pip install pytorch_lightning

# %% [markdown]
# We'll regenerate the Legendre polynomial dataset used last time:

# %%
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

# %% [markdown]
# Then use the same model definitions:

# %%
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

# %% [markdown]
# And finally create the train and validation datasets and then use them to train the model with pytorch-lightning.

# %%
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

# %% [markdown]
# We were visualizing the 5D latent space according to the ground truth $a_0$ coefficient:

# %%
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

# %% [markdown]
# # Generative modeling

# %% [markdown]
# ## Concepts
# 
# Generative modeling is a type of machine learning approach in which a model is trained to learn the underlying distribution of a dataset, and then generate new data samples that are similar to the original data.
# It is called "generative" since we are "generating" new (synthetic) samples.
# 
# To use an autoencoder for generative modeling, we first train the autoencoder on a dataset of input data.
# During training, the autoencoder learns to encode the input data into a compressed representation, and then decode the compressed representation back into the original input data.
# Once the autoencoder is trained, **we can sample from the learned encoding or latent space to generate new data samples.**
# 
# There are a variety of ways to generate new data samples using a trained autoencoder:
# * Randomly sample from the learned encoding or latent space and then decode the samples to generate new data samples
#   * This requires the specification of a probability distribution to sample from. An arbitrary prior distribution such as uniform or normal can be used, or an empirical PDF based on the training data sample could be used.
# * Interpolate between two data samples in the latent space
#   * This can be done by encoding the two data samples of interest, taking a weighted average of the encodings, and then decoding the weighted average to generate a new data sample that is a combination of the two original data samples.
# * Sample from the latent space on a regular grid

# %% [markdown]
# ## Basic implementation

# %%
from sklearn import cluster

km = cluster.KMeans(n_clusters=64).fit(z[:, :2])

cluster_ids = []
for c in km.cluster_centers_:
    cluster_ids.append( np.argmin( np.linalg.norm(z[:, :2] - c, axis=1) ) )

scale = 0.5

fig, ax = plt.subplots(figsize=(16, 16))

for i in cluster_ids:
    ax.plot(z[i, 0]+x*scale, z[i, 1]+y[i]*scale)

ax.set_aspect('equal')

# %% [markdown]
# We don't have to only use real curves that were put through the encoder -- we can also generate "synthetic" curves by passing an arbitrary $z$ through the decoder.
# Here we'll use all zeros:

# %%
model.decoder(torch.zeros([1, a.shape[1]]).float())

# %% [markdown]
# We can plot this to see we recover a full (but synthetic) observation of $y$:

# %%
gen_y = model.decoder(torch.zeros([1, a.shape[1]]).float())

fig, ax = plt.subplots()
ax.plot(x, gen_y[0].detach().numpy())

# %% [markdown]
# We can also compare this to a real curve nearby to this point in latent space ( in this case, the origin):

# %%
close_to_origin = np.argmin(np.linalg.norm(z, axis=1))
print(z[close_to_origin])

# %% [markdown]
# The two curves are not going to be exactly the same since their latent codes are slightly different:

# %%
fig, ax = plt.subplots()
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y[0].detach().numpy(), label='Generated')
_ = ax.legend()

# %% [markdown]
# We can extend this to generate a series of synthetic curves spread across the latent space:

# %%
fig, ax = plt.subplots(figsize=(16, 16))

km = cluster.KMeans(n_clusters=64).fit(z)

for c in km.cluster_centers_:
    gen_y = model.decoder(torch.tensor(c).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    ax.plot(c[0]+x*scale, c[1]+this_y*scale)

ax.set_aspect('equal')

# %% [markdown]
# Why are some of the curves overlapping?
# 
# We could also do so over a regular grid:

# %%
steps = 10

z0v = np.linspace(z[:, 0].min(), z[:, 0].max(), steps)
z1v = np.linspace(z[:, 1].min(), z[:, 1].max(), steps)
z0g, z1g = np.meshgrid(z0v, z1v)

fig, ax = plt.subplots()
ax.scatter(z0g.flatten(), z1g.flatten())
ax.set_aspect('equal')

# %% [markdown]
# Now sample at those points:

# %%
zgrid = np.vstack([z0g.flatten(), z1g.flatten(),
                   [0]*steps**2, [0]*steps**2, [0]*steps**2]).T  # why is this here?

fig, ax = plt.subplots(figsize=(16, 16))

for c in zgrid:
    gen_y = model.decoder(torch.tensor(c).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    ax.plot(c[0]+x*scale, c[1]+this_y*scale)

ax.set_aspect('equal')

# %% [markdown]
# We can increase the number of samples to really see the fine details of this space:

# %%
steps = 50

z0v = np.linspace(z[:, 0].min(), z[:, 0].max(), steps)
z1v = np.linspace(z[:, 1].min(), z[:, 1].max(), steps)
z0g, z1g = np.meshgrid(z0v, z1v)

zgrid = np.vstack([z0g.flatten(), z1g.flatten(),
                   [0]*steps**2, [0]*steps**2, [0]*steps**2]).T  # why is this here?

fig, ax = plt.subplots(figsize=(16, 16))

scale = 0.1

for c in zgrid:
    gen_y = model.decoder(torch.tensor(c).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    ax.plot(c[0]+x*scale, c[1]+this_y*scale)

ax.set_aspect('equal')

# %% [markdown]
# ## Interpolation

# %% [markdown]
# Linear interpolation between observations or groups of observations is a very common scheme for generating new samples.
# Let's try interpolating between the maximum and minimum $z_0$ samples:

# %%
top = np.argmax(z[:, 0])
bot = np.argmin(z[:, 0])

fig, ax = plt.subplots()
_ = ax.plot(x, y[top], label='Top')
_ = ax.plot(x, y[bot], label='Bottom')
_ = ax.legend()

# %% [markdown]
# ## [Check your understanding]
# 
# Interpolate between some $z_i$ and $z_j$ samples using the vector $\vec{v} = z_j - z_i$ with a simple mixing rule like $z_\mathrm{mix} = z_i + f \vec{v}$.
# Plot the result and think about the result -- are the curves close to what you would have drawn?

# %%
fig, ax = plt.subplots()
_ = ax.plot(x, y[top], label='Top')
_ = ax.plot(x, y[bot], label='Bottom')

v = z[top] - z[bot]
for i in np.linspace(0, 1, 11)[1:-1]:
    this_z = z[bot] + i * v
    gen_y = model.decoder(torch.tensor(this_z).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    _ = ax.plot(x, this_y, '--')

# %% [markdown]
# ## Using concept vectors
# 
# We can also modify single curves using the concept vectors.
# Here's a sample curve that lies close to the origin:

# %%
close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

fig, axes = plt.subplots(1, 2)

ax = axes[0]
_ = ax.scatter(*z[close_to_origin, :2], label='Real')
_ = ax.legend()

ax = axes[1]
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.legend()

# %% [markdown]
# Let's compute the $a_0$ concept vector again:

# %%
from sklearn import linear_model

lr = linear_model.LinearRegression().fit(z, a[:, [0]])
print(f'R2 = {lr.score(z, a[:, 0]):.3f}; coef = {lr.coef_}')

# %% [markdown]
# We can normalize these coefficients to obtain a unit vector (direction):

# %%
vec_a0 = lr.coef_ / np.linalg.norm(lr.coef_)

# %% [markdown]
# Then we apply the vector to modify the selected latent code:

# %%
# compute the latent codes and decode the functions
z_plus = z[close_to_origin] + vec_a0
out_plus = model.decoder(torch.tensor(z_plus).float())
gen_y_plus = out_plus[0].detach().numpy()

z_minus = z[close_to_origin] - vec_a0
out_minus = model.decoder(torch.tensor(z_minus).float())
gen_y_minus = out_minus[0].detach().numpy()

# make the plot
fig, axes = plt.subplots(1, 2)

ax = axes[0]
_ = ax.scatter(*z[:, :2].T, label='All data')
_ = ax.scatter(*z[close_to_origin, :2], label='Real')
_ = ax.scatter(*z_plus[0, :2], label='+a0')
_ = ax.scatter(*z_minus[0, :2], label='-a0')
_ = ax.legend()

ax = axes[1]
_ = ax.plot(x, np.mean(y, axis=0), label='All data')
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y_plus, label='+a0')
_ = ax.plot(x, gen_y_minus, label='-a0')
_ = ax.legend()

# %% [markdown]
# If we increase the magnitude of the vector we will increase the effect:

# %%
z_plus = z[close_to_origin] + vec_a0 * 5
out_plus = model.decoder(torch.tensor(z_plus).float())
gen_y_plus = out_plus[0].detach().numpy()

z_minus = z[close_to_origin] - vec_a0 * 5
out_minus = model.decoder(torch.tensor(z_minus).float())
gen_y_minus = out_minus[0].detach().numpy()

fig, axes = plt.subplots(1, 2)

ax = axes[0]
_ = ax.scatter(*z[:, :2].T, label='All data')
_ = ax.scatter(*z[close_to_origin, :2], label='Real')
_ = ax.scatter(*z_plus[0, :2], label='+a0')
_ = ax.scatter(*z_minus[0, :2], label='-a0')
_ = ax.legend()

ax = axes[1]
_ = ax.plot(x, np.mean(y, axis=0), label='All data')
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y_plus, label='+a0')
_ = ax.plot(x, gen_y_minus, label='-a0')
_ = ax.legend()

# %% [markdown]
# ## [Check your understanding]
# 
# Try combining multiple concept vectors together.
# You can do different combinations of +, - for each vector to make more complex changes to the reconstructed functions.

# %%


# %% [markdown]
# # Variational Autoencoder

# %% [markdown]
# ## Concept
# 
# The autoencoder's latent space can be very poorly behaved.
# For instance, there is no guarantee that points very near to each other have a meaningful interpolation between them; the only thing that the model seeks is the faithful reconstruction of input samples.
# This limits the autoencoder's utility in generative modeling.
# 
# <img src="../lectures/assets/lecture16_vae_architecture.jpg" alt="Architecture of a Variational Autoencoder" width=600>
# 
# The Variational Autoencoder (VAE) utilizes the notion of distributions inside the bottleneck:
# 
# <img src="../lectures/assets/lecture16_reparameterization.jpg" alt="Illustration of the reparameterization trick in VAEs" width=600>
# 
# Thus, nearby points blend together and the latent space is forced to have some notion of smoothness during training in order to achieve a low loss.
# 
# 
# <img src="../lectures/assets/lecture16_vae_latent.jpg" alt="Latent space visualization of a trained VAE" width=600>
# 
# 
# In practice this is achieved by fitting a mean $\mu$ and standard deviation $\sigma$ using neural networks:
# 
# <img src="../lectures/assets/lecture16_adversarial_training.jpg" alt="Conceptual diagram of adversarial training in GANs" width=600>
# 
# > There is an amazing article describing this in great detail in [Understanding Variational Autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
# 
# This will require some careful programming because the behavior is different during training and inference!

# %% [markdown]
# ## Implementation
# 
# First we should define the loss function, the Kullback–Leibler divergence.
# This is a measure of dissimilarity between probability distributions.
# In practice, we are trying to make the latent distribution Normal, so we can just look up the KL divergence for a Normal distribution:

# %%
def kld_loss(x_rc, x, mu, logvar):
    mse_loss = nn.MSELoss(reduction='sum')(x_rc, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ( mse_loss + kl_divergence ) / len(x)

# %% [markdown]
# > Note the use of log variance to make the gradients better behaved.
# 
# Then we have to implement the probabilistic encoding:

# %%
class VariationalAutoencoder(pl.LightningModule):
    def __init__(self, input_size, latent_dim, hidden_dim, act=nn.ReLU()):
        super().__init__()

        self.encoder = MLP(hidden_dim, act)

        # new parts of the network:
        self.fc_mean = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)

        self.decoder = MLP(list(hidden_dim)[::-1]+[input_size], act)

    def encode(self, x):
        # pass through two-headed encoder network
        x = self.encoder(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def forward(self, x):
        # goes all the way through *with a probabilistic encoding*!
        # also returns mean and logvar for KLD calculation
        mean, logvar = self.encode(x)
        # convert from mean and variance to probabilistic z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        # decode stochastic latent code
        out = self.decoder(z)
        return out, mean, logvar

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out, mean, logvar = self(x)
        loss = kld_loss(out, x, mean, logvar)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out, mean, logvar = self(x)
        loss = kld_loss(out, x, mean, logvar)
        self.log('validation_loss', loss)
        return loss

# %% [markdown]
# Now we train the model using the usual pytorch-lightning training procedure:

# %%
torch.manual_seed(0)

model = VariationalAutoencoder(yt.shape[1], a.shape[1], [128, 64, 32, 16], act=nn.ELU())
out = model(yt[:1])

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model=model, train_dataloaders=dl_train)

# %% [markdown]
# We can see the stochastic effect by repeatedly reconstructing a single sample:

# %%
with torch.no_grad():
    zt, _ = model.encode(yt)
    z = zt.detach().numpy()

close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

fig, ax = plt.subplots()

for i in range(100):
    gen_y, _, _ = model(yt[close_to_origin].unsqueeze(0))
    _ = ax.plot(x, gen_y[0].detach().numpy(), 'b-', alpha=0.1)

_ = ax.plot(x, y[close_to_origin], 'k-')

# %% [markdown]
# If we don't want to invoke the stochastic part of the encoding, we can use the mean value without reparameterization:

# %%
close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

fig, ax = plt.subplots()

for i in range(100):
    gen_y, _, _ = model(yt[close_to_origin].unsqueeze(0))
    _ = ax.plot(x, gen_y[0].detach().numpy(), 'b-', alpha=0.1)

_ = ax.plot(x, y[close_to_origin], 'k-')

mu, logvar = model.encode(yt[close_to_origin].unsqueeze(0))
gen_y = model.decoder(mu)
_ = ax.plot(x, gen_y[0].detach().numpy(), 'k--')

# %% [markdown]
# You will see that the mean lies in the center of the generated samples (by construction).
# 
# There is an important difference between this result and the one from the AE -- the smoothness of the curve!
# Scroll back up to the AE results to see the difference.
# The probabilistic nature of the embedding forces some degree of averaging to achieve this effect.

# %% [markdown]
# ## Evaluating the latent space
# 
# The entire point of the VAE is to force the latent space to be approximately Normal.
# Let's see how it worked:

# %%
with torch.no_grad():
    zt, _ = model.encode(yt)
    z = zt.detach().numpy()

fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2, c=a[:, 0])

# %% [markdown]
# You will see here that the distributions are individually Normal-looking.
# In addition, we still have the expected relationships between $a$ coefficients and the resulting $z$ latent space.

# %%
from sklearn import linear_model

lr = linear_model.LinearRegression().fit(z, a[:, [0]])
print(f'R2 = {lr.score(z, a[:, 0]):.3f}; coef = {lr.coef_}')

# %% [markdown]
# In addition, we could check the reconstruction:

# %%
close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

gen_y = model.decoder(torch.tensor(a[close_to_origin]).unsqueeze(0).float())

fig, ax = plt.subplots()
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y[0].detach().numpy(), label='Generated')
_ = ax.legend()

# %% [markdown]
# In summary, we have given up fidelity for a "better-behaved" latent space (or at least one with prescribed statistics).
# 

# %% [markdown]
# ## [Check your understanding]
# 
# Try training the VAE with fewer and greater latent dimensions.
# * How does the loss change?
# * How does the shape of the latent space change?
# 
# Make a 2D grid of generated samples in these different cases.
# * How does the spatial variation in the latent space change with more and less dimensions?

# %%


# %% [markdown]
# ## beta-VAE
# 
# We can balance this using a "beta-VAE" that uses the following loss:
# 
# $\mathcal{L} = \mathrm{MSE} + \beta \; \mathrm{KLD}$
# 
# (this is where the "beta" comes from)
# 
# With a beta-VAE, we choose an empirical balance the Normality of the latent distribution with the reconstruction loss.

# %%
def kld_loss(x_rc, x, mu, logvar, beta=1):
    mse_loss = nn.MSELoss(reduction='sum')(x_rc, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ( mse_loss + beta * kl_divergence ) / len(x)


class BetaVariationalAutoencoder(pl.LightningModule):
    def __init__(self, input_size, latent_dim, hidden_dim, beta=1, act=nn.ReLU()):
        super().__init__()

        self.beta = beta

        self.encoder = MLP(hidden_dim, act)

        # new parts of the network:
        self.fc_mean = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)

        self.decoder = MLP(list(hidden_dim)[::-1]+[input_size], act)

    def encode(self, x):
        # pass through two-headed encoder network
        x = self.encoder(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def forward(self, x):
        # goes all the way through *with a probabilistic encoding*!
        # also returns mean and logvar for KLD calculation
        mean, logvar = self.encode(x)
        # convert from mean and variance to probabilistic z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        # decode stochastic latent code
        out = self.decoder(z)
        return out, mean, logvar

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out, mean, logvar = self(x)
        loss = kld_loss(out, x, mean, logvar, beta=self.beta)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out, mean, logvar = self(x)
        loss = kld_loss(out, x, mean, logvar, beta=self.beta)
        self.log('validation_loss', loss)
        return loss

# %% [markdown]
# We train as before:

# %%
torch.manual_seed(0)

model = BetaVariationalAutoencoder(yt.shape[1], a.shape[1], [128, 64, 32, 16], beta=1e-2, act=nn.ELU())
out = model(yt[:1])

dl_train = DataLoader(ds_train, batch_size=128, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False)

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model=model, train_dataloaders=dl_train)

# %% [markdown]
# > Note the much lower value of loss!
# 
# Now we can check the probabilistic reconstruction again:

# %%
with torch.no_grad():
    zt, _ = model.encode(yt)
    z = zt.detach().numpy()

close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

fig, ax = plt.subplots()

for i in range(100):
    gen_y, _, _ = model(yt[close_to_origin].unsqueeze(0))
    _ = ax.plot(x, gen_y[0].detach().numpy(), 'b-', alpha=0.1)

_ = ax.plot(x, y[close_to_origin], 'k-')

# %% [markdown]
# Next we can investigate the shape of the new latent space:

# %%
fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2, c=a[:, 0])

# %% [markdown]
# Note the reconstructions are not that great:

# %%
close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

gen_y = model.decoder(torch.tensor(a[close_to_origin]).unsqueeze(0).float())

fig, ax = plt.subplots()
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y[0].detach().numpy(), label='Generated')
_ = ax.legend()

# %% [markdown]
# Nevertheless, the space is quite smooth:

# %%
steps = 50

z0v = np.linspace(z[:, 0].min(), z[:, 0].max(), steps)
z1v = np.linspace(z[:, 1].min(), z[:, 1].max(), steps)
z0g, z1g = np.meshgrid(z0v, z1v)

zgrid = np.vstack([z0g.flatten(), z1g.flatten(),
                   [0]*steps**2, [0]*steps**2, [0]*steps**2]).T  # why is this here?

fig, ax = plt.subplots(figsize=(16, 16))

scale = 0.05

for c in zgrid:
    gen_y = model.decoder(torch.tensor(c).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    ax.plot(c[0]+x*scale, c[1]+this_y*scale)

ax.set_aspect('equal')

# %%

