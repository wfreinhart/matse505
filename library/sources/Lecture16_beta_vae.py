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
# id: Lecture16_beta_vae
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## beta-VAE
#
# We can balance this using a "beta-VAE" that uses the following loss:
#
# $\mathcal{L} = \mathrm{MSE} + \beta \; \mathrm{KLD}$
#
# (this is where the "beta" comes from)
#
# With a beta-VAE, we choose an empirical balance the Normality of the latent distribution with the reconstruction loss.
#
# We train as before:
#
# > Note the much lower value of loss!
#
# Now we can check the probabilistic reconstruction again:
#
# Next we can investigate the shape of the new latent space:
#
# Note the reconstructions are not that great:
#
# Nevertheless, the space is quite smooth:

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

torch.manual_seed(0)

model = BetaVariationalAutoencoder(yt.shape[1], a.shape[1], [128, 64, 32, 16], beta=1e-2, act=nn.ELU())
out = model(yt[:1])

dl_train = DataLoader(ds_train, batch_size=128, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=1024, shuffle=False)

trainer = pl.Trainer(max_epochs=100)
trainer.fit(model=model, train_dataloaders=dl_train)

with torch.no_grad():
    zt, _ = model.encode(yt)
    z = zt.detach().numpy()

close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

fig, ax = plt.subplots()

for i in range(100):
    gen_y, _, _ = model(yt[close_to_origin].unsqueeze(0))
    _ = ax.plot(x, gen_y[0].detach().numpy(), 'b-', alpha=0.1)

_ = ax.plot(x, y[close_to_origin], 'k-')

fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2, c=a[:, 0])

close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

gen_y = model.decoder(torch.tensor(a[close_to_origin]).unsqueeze(0).float())

fig, ax = plt.subplots()
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y[0].detach().numpy(), label='Generated')
_ = ax.legend()

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
