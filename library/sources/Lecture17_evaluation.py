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
# id: Lecture17_evaluation
# type: Foundational
# parent_lecture: Lecture17
# ---
#
# ## Evaluation
#
# ### Reconstruction
#
# ### Interpolation
#
# This is a linear interpolation in latent space:
#
# Compare to a linear interpolation in real space:

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

from matplotlib import pyplot as plt

i = -1
x, _ = dl_valid.dataset[i]
x_hat = model(x.unsqueeze(0))[0]

fig, axes = plt.subplots(1, 2)
for i, arr in enumerate([x, x_hat]):
    axes[i].imshow(arr.detach().squeeze(0).numpy(), 'Greys_r')

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

fig, axes = plt.subplots(1, n_interp, figsize=(10, 4))

i, j = (0, 35)
x_i, _ = dl_valid.dataset[i]
x_j, _ = dl_valid.dataset[j]

for i, alpha in enumerate(np.linspace(0, 1, n_interp)):
    arr = x_i + alpha * (x_j - x_i)
    axes[i].imshow(arr.detach().squeeze(0).numpy(), 'Greys_r')
