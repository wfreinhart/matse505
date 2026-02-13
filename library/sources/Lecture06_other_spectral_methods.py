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
# id: Lecture06_other_spectral_methods
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Other spectral methods
#
# Here are a few other methods from the `manifold` module in no particular order.
# You can read about their assumptions in the `scikit-learn` documentation or on other websites.

# %%
from sklearn import manifold

Z = manifold.Isomap().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

Z = manifold.LocallyLinearEmbedding().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

Z = manifold.MDS().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')
