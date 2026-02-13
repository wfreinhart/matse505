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
# id: Lecture06_where_manifold_learning_fails
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Where manifold learning fails
#
# Let's revisit our composition data from above.
#
# We can try `SpectralEmbedding` again:
#
# The `nearest_neighbors` affinity does not seem to be working.
# Let's go back to `rbf`:
#
# This is almost identical to PCA.
# What about diffusion maps?
#
# You can see that depending on the choice of $\varepsilon$ we will get wildly different results, eventually ending up back at something like the linear result.

# %%
x = data.loc[:, ' C':'Nb + Ta']
# here's a one-liner to encode the str labels as int:
_, y = np.unique([it[0] for it in data['Alloy code']], return_inverse=True)

Z = manifold.SpectralEmbedding(random_state=0).fit_transform(x)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

Z = manifold.SpectralEmbedding(affinity='rbf', random_state=0).fit_transform(x)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

dist = distance.squareform(distance.pdist(x))

epsilon = np.percentile(dist, 100)
L = np.exp(-dist**2/epsilon**2)

D = np.diag(np.sum(L, axis=1))
P = np.linalg.inv(D) @ L

w, v = np.linalg.eig(P)
plt.scatter(*v[:, 1:3].T, c=y)
