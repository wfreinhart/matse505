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
# id: Lecture06_spectral_methods
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Spectral methods
#
# The Spectral Embedding (Laplacian Eigenmaps) algorithm comprises three stages:
#
# **Weighted Graph Construction.**
# Transform the raw input data into graph representation using affinity (adjacency) matrix representation.
#
# **Graph Laplacian Construction.**
# The unnormalized Graph Laplacian is constructed as $L = D - A$ and normalized as
#
# $L = D^{-1/2} (D-A) D^{-1/2}$
#
# **Partial Eigenvalue Decomposition.**
# Eigenvalue decomposition is done on graph Laplacian
#
# This is very similar to the idea of PCA, except we will use a nonlinear, graph-based construction for $A$ instead of the covariance matrix.
# After constructing the matrix, the spectral decomposition (eigenvalue problem) is the same.
#
# One of the main decisions here is what constitutes adjacency.
# The default in `sklearn` is to build the nearest neighbors graph.
#
# The default for `affinity` is the nearest neighbor graph, which does not appear to work well.
# What if we try the `rbf` option like we did above for the kernel PCA?
#
# This actually works pretty well!
# Note that the curvature in $Z_1$ is entirely spurious and the discovered manifold is entirely 1D.
# This is a common feature of manifold learning approaches when there are extra dimensions.

# %%
from sklearn import manifold

Z = manifold.SpectralEmbedding().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

Z = manifold.SpectralEmbedding(affinity='rbf').fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')
