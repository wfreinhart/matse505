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
# id: Lecture06_supervised_umap
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Supervised UMAP
#
# We can also do supervised and semi-supervised learning with UMAP, to create clusters that are informed by the labels.
# Here's the fully supervised case:
#
# In this particular case, it doesn't change much (if at all).
# For completeness, here's the semi-supervised case, where we artificially hide some of the labels:

# %%
Z = umap.UMAP(random_state=0).fit_transform(x, y=y)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

sparse_y = np.array(y)  # create a copy of the labels

remove_n = int(0.90 * x.shape[0])  # remove 90% of the labels!

rng = np.random.RandomState(0)  # set random state so we always get same result
remove_idx = rng.choice(np.arange(y.shape[0]), remove_n, replace=False)

sparse_y[remove_idx] = -1  # this indicates "missing" label

Z = umap.UMAP(random_state=0).fit_transform(x, y=sparse_y)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')
