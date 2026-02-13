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
# id: Lecture16_implementation
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## Implementation
#
# First we should define the loss function, the Kullback–Leibler divergence.
# This is a measure of dissimilarity between probability distributions.
# In practice, we are trying to make the latent distribution Normal, so we can just look up the KL divergence for a Normal distribution:
#
# > Note the use of log variance to make the gradients better behaved.
#
# Then we have to implement the probabilistic encoding:
#
# Now we train the model using the usual pytorch-lightning training procedure:
#
# We can see the stochastic effect by repeatedly reconstructing a single sample:
#
# If we don't want to invoke the stochastic part of the encoding, we can use the mean value without reparameterization:
#
# You will see that the mean lies in the center of the generated samples (by construction).
#
# There is an important difference between this result and the one from the AE -- the smoothness of the curve!
# Scroll back up to the AE results to see the difference.
# The probabilistic nature of the embedding forces some degree of averaging to achieve this effect.

# %%
def kld_loss(x_rc, x, mu, logvar):
    mse_loss = nn.MSELoss(reduction='sum')(x_rc, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ( mse_loss + kl_divergence ) / len(x)

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

torch.manual_seed(0)

model = VariationalAutoencoder(yt.shape[1], a.shape[1], [128, 64, 32, 16], act=nn.ELU())
out = model(yt[:1])

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

close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

fig, ax = plt.subplots()

for i in range(100):
    gen_y, _, _ = model(yt[close_to_origin].unsqueeze(0))
    _ = ax.plot(x, gen_y[0].detach().numpy(), 'b-', alpha=0.1)

_ = ax.plot(x, y[close_to_origin], 'k-')

mu, logvar = model.encode(yt[close_to_origin].unsqueeze(0))
gen_y = model.decoder(mu)
_ = ax.plot(x, gen_y[0].detach().numpy(), 'k--')
