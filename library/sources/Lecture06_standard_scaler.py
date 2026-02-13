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
# id: Lecture06_standard_scaler
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Standard scaler
#
# Here we'll use the `StandardScaler`, which normalizes everything according to its standard deviation:
#
# $X_s = \frac{X - \mu}{\sigma}$
#
# This way all the values should end up with comparable magnitudes. Let's see how it affects the manifold structure:
#
# We can try looking at this manifold through the lens of different compositions.
#
# Remember - this was fitted to the properties! The fact that there are correlations with the compositions tells us something about how the chemistry influences properties -- for instance, we should look at V-containing steels to get properties like those on the left hand side.
#
# Just to show that feature scaling achieved something, this was the original projection:
#
# We can confirm that part of the space got smashed together along the center.

# %%
from sklearn import decomposition
from matplotlib import pyplot as plt

# compute the pca embedding
pca = decomposition.PCA().fit(x)
Po = pca.transform(x)

# plot the result
fig, ax = plt.subplots()
ax.scatter(Po[:, 0], Po[:, 1])

from sklearn import preprocessing

# rescale the data
scaler = preprocessing.StandardScaler().fit(x)
xs = scaler.transform(x)

# compute the pca embedding
pca = decomposition.PCA().fit(xs)
P = pca.transform(xs)

# plot the result
fig, ax = plt.subplots(1, 2)
ax[0].scatter(P[:, 0], P[:, 1])
ax[1].scatter(Po[:, 0], Po[:, 1])

fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data[' Al'])
plt.colorbar(im)

fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
plt.colorbar(im)

fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[0])

fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[1])

pca = decomposition.PCA().fit(x)
P = pca.transform(x)

fig, ax = plt.subplots()
im = ax.scatter(Po[:, 0], Po[:, 1], c=clean_data['V'])
plt.colorbar(im)

pca = decomposition.PCA().fit(x)
Po = pca.transform(x)

fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[0])

fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[1])

x.mean(axis=0)
