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
# id: Lecture15_evaluation
# type: Foundational
# parent_lecture: Lecture15
# ---
#
# ## Evaluation
#
# Let's start by plotting the first two dimensions of our latent space.
# We'll access it using the `model.encoder` module:
#
# $z$ is more than 2 dimensional, so we can evaluate the entire latent space this way:
#
# On its own this doesn't tell us much.
# Let's try to relate it back to the coefficients of the polynomials using color in our scatter plot.
#
# We should see that there is a gradient in the color, indicating that the first coefficient is related to the learned latent space.
#
# If we look at another coefficient, it won't necessarily share the same correlation:
#
# It's possible that another combination of $z$ coordinates does correlate with the $a_1$ coefficient though:

# %%
with torch.no_grad():
    z = model.encoder(yt).detach().numpy()

fig, ax = plt.subplots()
ax.scatter(z[:, 0], z[:, 1])

fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2)

with torch.no_grad():
    z = model.encoder(yt).detach().numpy()

fig, ax = plt.subplots()
ax.scatter(z[:, 0], z[:, 1], c=a[:, 0])

fig, ax = plt.subplots()
ax.scatter(z[:, 0], z[:, 1], c=a[:, 1])

fig, axes = plt.subplots(z.shape[1], z.shape[1], figsize=(6, 6))
for i in range(z.shape[1]):
    for j in range(z.shape[1]):
        ax = axes[i, j]
        if i == j:
            ax.hist(z[:, i], bins=int(np.sqrt(z.shape[0])))
        else:
            ax.scatter(z[:, i], z[:, j], s=2, c=a[:, 1])
