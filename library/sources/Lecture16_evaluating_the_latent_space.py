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
# id: Lecture16_evaluating_the_latent_space
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## Evaluating the latent space
#
# The entire point of the VAE is to force the latent space to be approximately Normal.
# Let's see how it worked:
#
# You will see here that the distributions are individually Normal-looking.
# In addition, we still have the expected relationships between $a$ coefficients and the resulting $z$ latent space.
#
# In addition, we could check the reconstruction:
#
# In summary, we have given up fidelity for a "better-behaved" latent space (or at least one with prescribed statistics).

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

from sklearn import linear_model

lr = linear_model.LinearRegression().fit(z, a[:, [0]])
print(f'R2 = {lr.score(z, a[:, 0]):.3f}; coef = {lr.coef_}')

close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

gen_y = model.decoder(torch.tensor(a[close_to_origin]).unsqueeze(0).float())

fig, ax = plt.subplots()
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y[0].detach().numpy(), label='Generated')
_ = ax.legend()
