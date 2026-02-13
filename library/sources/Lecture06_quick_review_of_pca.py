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
# id: Lecture06_quick_review_of_pca
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Quick review of PCA
#
# Let's review how the PCA projection works.
# We'll go back to the alloy compositions data for this exercise.
#
# Now that we have this $z$ embedding, we can evaluate the data in this low-dimensional space.
#
# Remember these projections are nothing more than the product of the original array (shifted to have zero mean) with the principal component vectors:

# %%
from sklearn import decomposition

x = data.loc[:, ' C':'Nb + Ta']

pca = decomposition.PCA().fit(x)
z = pca.transform(x)
print(z.shape)

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.scatter(*z[:, :2].T)  # shorthand way to plot the columns (1, 2) as (x, y)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# we use x.values because x is actually a DataFrame! .values converts to array
x_shift = x.values - np.mean(x.values, axis=0)
# we use the transpose of pca.components_ based on the implementation in sklearn
z_manual = np.dot(x_shift, pca.components_.T)

fig, ax = plt.subplots()
ax.scatter(*z_manual[:, :2].T)  # shorthand way to plot the columns (1, 2) as (x, y)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
