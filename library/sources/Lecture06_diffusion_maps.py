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
# id: Lecture06_diffusion_maps
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Diffusion maps
#
# A variant of the spectral embedding is to use a Gaussian kernel to emulate a "diffusion process" on the manifold:
#
# $k(x,y) = \exp \left( -\frac{||x-y||^2}{\epsilon^2} \right)$
#
# With this kernel we can compute an affinity matrix $L$ and then solve the eigenvector problem on the normalized diffusion matrix,
#
# $P = D^{-1} K$,
#
# this can be cast as an eigenvector problem to obtain a mapping in the diffusion space:

# %%
from scipy.spatial import distance

dist = distance.squareform(distance.pdist(S))

epsilon = np.percentile(dist, 1)
L = np.exp(-dist**2/epsilon**2)

D = np.diag(np.sum(L, axis=1))
P = np.linalg.inv(D) @ L

w, v = np.linalg.eig(P)
plt.scatter(*v[:, 1:3].T, c=t)
