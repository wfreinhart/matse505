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
# id: Lecture05_principal_component_analysis
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Principal Component Analysis
#
# We are often faced with higher-dimensional data like in these alloy compositions and properties.
# Unfortunately there is not much we can do about our eyes being wired to understand 2D images.
# This makes it difficult to interpret information in higher dimensions like 3D scatter plots, unless the trends are very clear.
#
# [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) is a method to "project" the higher-dimensional data down to only a few dimensions so we can visualize or analyze it with methods that we are more comfortable with.
# If you prefer the nuts and bolts, you should know PCA is based on a spectral decomposition of the covariance matrix (it's pretty straightfoward linear algebra and not really "machine learning").
#
# For example, consider these data:
#
# <img src="../lectures/assets/pca_3d_to_2d.jpg" width=800 alt="Diagram illustrating Principal Component Analysis projecting 3D data onto a 2D principal component plane">
#
# Here we have 3 spatial dimensions plus a color.
# The PCA shows clear groupings in a 2D space which is a plane through the original 3D space.
#
# PCA is based on the eigenvectors of the covariance matrix.
# You are probably familiar with the concept of covariance in 2D, like the following:
#
# <img src="../lectures/assets/covariance_types.jpg" width=600 alt="Visual representation of different spatial covariance structures in data">
#
# This readily extends to higher dimensions, and can be calculated with builtins such as with `numpy.cov`:
# > The transpose is needed because `numpy.cov` assumes "Each row of m represents a variable, and each column a single observation of all those variables" -- the transpose of our `DataFrame`
#
# If we take the eigenvectors of this covariance matrix, we will get something special:
#
# `w` are the eigenvalues while `v` are the eigenvectors of the covariance matrix (each column is one eigenvector).
# What is the use of these?
# Let's start with the eigenvalues:
#
# The eigenvalues decay over the column index from 4e-1 to 3e-8.
# This is proportional to the variance in that dimension, so the first eigenvector will be associated with 10 million times greater variance than the last one.
# With this in mind, we can investigate the eigenvectors.
#
# From this we see that something like `+Cr, +Mo, -Mn, -Ceq` is the dominant direction of variance.
# In other words, alloys with high `Cr` and `Mo` have low `Mn` and vice versa.
# As shown above, this is the direction of maximal variance.
#
# We can project the `x` values onto the first two eigenvectors.
# You should think of this like rotating your viewpoint to view the maximally varying directions in the plane (just like the example above).
#
# So what is the point of this trick?
# Let's plot the above with the original alloy code labels.
#
# Unlike above where we searched for a suitable 3D representation (with only moderate success), here we have constructed a rotation using analysis of the dataset.
# The resulting projection reveals the ground truth labeling clearly (existing almost exclusively in the PC1 direction).

# %%
np.cov(x.T).shape

w, v = np.linalg.eig(np.cov(x.T))
print(w.shape, v.shape)

fig, ax = plt.subplots()
ax.plot(np.real(w), '.')
ax.set_yscale('log')

fig, ax = plt.subplots()
_ = ax.bar(x.columns, v[:, 0])

projected = (x.values @ v)
print( projected.shape )

fig, ax = plt.subplots()
_ = ax.scatter(projected[:, 0], projected[:, 1])

fig, ax = plt.subplots()
_ = ax.scatter(projected[:, 0], projected[:, 1], c=y)
_ = ax.set_xlabel('Principal Component 1')
_ = ax.set_ylabel('Principal Component 2')
