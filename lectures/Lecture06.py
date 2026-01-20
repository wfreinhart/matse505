# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's topics:
# * Feature scaling
# * Reconstruction
# * Manifold learning
# * Semi-supervised learning

# %% [markdown]
# Let's load the alloys dataset from before...

# %%
import pandas as pd
import numpy as np

data = pd.read_csv('../datasets/steels.csv')  # load into pandas
data                            # show a view of the data file

# %% [markdown]
# Drop the outlier we found last time:

# %%
outlier = np.argmax(data[' Tensile Strength (MPa)'])
clean_data = data.drop(index=outlier)

# %% [markdown]
# And define a space to work in:

# %%
x = clean_data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']

# %% [markdown]
# # Feature scaling

# %% [markdown]
# One thing to watch out for when doing unsupervised representation learning is features with different units. If one of the features has magnitudes of $10^6$ and another has $10^1$, the big one will completely dominate the embedding. We can get around this using **feature scaling**.
#
# <img src="./assets/feature_scaling.jpg" width=600 alt="Comparison of data distribution before and after different scaling methods: StandardScaler, MinMaxScaler, and RobustScaler">
#
# Making sure the data are roughly isotropic really helps us capture more information in the embedding.

# %% [markdown]
# ## Standard scaler
#
# Here we'll use the `StandardScaler`, which normalizes everything according to its standard deviation:
#
# $X_s = \frac{X - \mu}{\sigma}$
#
# This way all the values should end up with comparable magnitudes. Let's see how it affects the manifold structure:

# %%
from sklearn import decomposition
from matplotlib import pyplot as plt

# compute the pca embedding
pca = decomposition.PCA().fit(x)
Po = pca.transform(x)

# plot the result
fig, ax = plt.subplots()
ax.scatter(Po[:, 0], Po[:, 1])

# %%
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

# %% [markdown]
# We can try looking at this manifold through the lens of different compositions.

# %%
fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data[' Al'])
plt.colorbar(im)

# %%
fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
plt.colorbar(im)

# %% [markdown]
# Remember - this was fitted to the properties! The fact that there are correlations with the compositions tells us something about how the chemistry influences properties -- for instance, we should look at V-containing steels to get properties like those on the left hand side.

# %%
fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[0])

fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[1])

# %% [markdown]
# Just to show that feature scaling achieved something, this was the original projection:

# %%
pca = decomposition.PCA().fit(x)
P = pca.transform(x)

fig, ax = plt.subplots()
im = ax.scatter(Po[:, 0], Po[:, 1], c=clean_data['V'])
plt.colorbar(im)

# %% [markdown]
# We can confirm that part of the space got smashed together along the center.

# %%
pca = decomposition.PCA().fit(x)
Po = pca.transform(x)

fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[0])

fig, ax = plt.subplots()
ax.bar(x.columns, pca.components_[1])

# %%
x.mean(axis=0)

# %% [markdown]
# ## Min-Max scaler
#
# This is the most straightforward preprocesser you could imagine -- it simply rescales each column to have a min value of 0 and max value of 1.
#
# $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$
#

# %%
scaler = preprocessing.MinMaxScaler().fit(x)
xs = scaler.transform(x)

pca = decomposition.PCA().fit(xs)
P = pca.transform(xs)

fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
plt.colorbar(im)

# %% [markdown]
# One interesting consequence of the `MinMaxScaler` is the mitigation of the outlier in `Tensile Strength (MPa)`.

# %% [markdown]
# ## Power transformer
#
# The power transformer is a scheme to make a non-Gaussian distribution look more Gaussian.
# Here's a visual example:
#
# <img src="./assets/power_transformer.jpg" width=600 alt="Effect of PowerTransformer on skewed data distributions to make them more Gaussian-like">
#
# Why would we do this?
# This is an attempt to avoid the long tails dominating the reduced space.

# %%
scaler = preprocessing.PowerTransformer().fit(x)
xs = scaler.transform(x)

pca = decomposition.PCA().fit(xs)
P = pca.transform(xs)

fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
plt.colorbar(im)

# %% [markdown]
# Here the power transformer provides a middle ground between the standard scaler and min-max scaler.
# Note that all three of these preprocessing methods make the point cloud more isotropic and portray additional features compared to the PCA on raw data.

# %% [markdown]
# A quick note on the inverse transform:

# %%
print( 'original', x.values[0] )
scaler = preprocessing.StandardScaler().fit(x)
xs = scaler.transform(x)
print( 'scaled', xs[0] )
print( 'reconstructed', scaler.inverse_transform(xs)[0] )

# %% [markdown]
# ## [Check your understanding]
#
# Try implementing the `StandardScaler` and `MinMaxScaler` yourself (i.e., using `numpy`).
#
# Check that you have it right by comparing to the result from the `sklearn` builtins.

# %%
xs = scaler.transform(x)
print( 'scaled', xs[0] )

mu = x.mean(axis=0)
sigma = x.std(axis=0)
xs = (x - mu) / sigma
print(xs.values[0])

# %% [markdown]
# # Reconstruction
#
# Something we haven't discussed with these dimensionality reduction approaches is the concept of reconstructing the high-dimensional object from the low-dimensional embedding.
# It's important to understand how this works, its limitations, and how it might be applied.

# %% [markdown]
# ## Quick review of PCA
#
# Let's review how the PCA projection works.
# We'll go back to the alloy compositions data for this exercise.

# %%
from sklearn import decomposition

x = data.loc[:, ' C':'Nb + Ta']

pca = decomposition.PCA().fit(x)
z = pca.transform(x)
print(z.shape)

# %% [markdown]
# Now that we have this $z$ embedding, we can evaluate the data in this low-dimensional space.

# %%
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.scatter(*z[:, :2].T)  # shorthand way to plot the columns (1, 2) as (x, y)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# %% [markdown]
# Remember these projections are nothing more than the product of the original array (shifted to have zero mean) with the principal component vectors:

# %%
# we use x.values because x is actually a DataFrame! .values converts to array
x_shift = x.values - np.mean(x.values, axis=0)
# we use the transpose of pca.components_ based on the implementation in sklearn
z_manual = np.dot(x_shift, pca.components_.T)

fig, ax = plt.subplots()
ax.scatter(*z_manual[:, :2].T)  # shorthand way to plot the columns (1, 2) as (x, y)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

# %% [markdown]
# ## Concept of reconstruction

# %% [markdown]
# We can also take these embeddings and "un-project" them back to the originals.
# (Remember this embedding operation was basically a rotation).
# Let's evaluate these in the 2D plane that corresponds to the greatest variance in the data:

# %%
ascending = np.argsort(pca.components_[0])
descending = ascending[::-1]
top2 = descending[:2]
print(top2)
print(x.columns[top2])

# %%
x_recon = np.dot(z_manual, pca.components_) + np.mean(x.values, axis=0)

fig, ax = plt.subplots()
ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
ax.scatter(*x.values[:, top2].T, label='Original',
           marker='s', edgecolor='tab:orange', facecolor='none')
ax.legend()

# %% [markdown]
# ## Information loss
#
# We can plot the cumulative explained variance to see how many of these are significant.

# %%
fig, ax = plt.subplots()
n = np.arange(pca.n_components_)+1
var_c = np.cumsum(pca.explained_variance_ratio_)
print(var_c)
ax.plot(n, var_c, '.-')
ax.set_xlabel('Components')
ax.set_ylabel('Cumulative explained variance')

# %% [markdown]
# While this is slightly subjective (or at least depends on the intended application), we can probably say that we won't be able to see a difference with more than 9 components (>99.99% explained variance).
# Let's investigate what happens when we use less or more components.

# %%
nc = 9
z_trunc = z_manual[:, :nc]
x_recon = np.dot(z_trunc, pca.components_[:nc, :]) + np.mean(x.values, axis=0)

fig, ax = plt.subplots()
ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
ax.scatter(*x.values[:, top2].T, label='Original',
           marker='s', edgecolor='tab:orange', facecolor='none')
ax.legend()


# %% [markdown]
# So far, so good. Let's make a function and try using subsequently fewer components.

# %%
def plot_reconstruction(nc):
    z_trunc = z_manual[:, :nc]
    x_recon = np.dot(z_trunc, pca.components_[:nc, :]) + np.mean(x.values, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
    ax.scatter(*x.values[:, top2].T, label='Original',
            marker='s', edgecolor='tab:orange', facecolor='none')
    ax.legend()
    ax.text(0.95, 0.05, f'{nc} components; {var_c[nc]*100:.2f}% explained variance',
            ha='right', transform=ax.transAxes)

    ax.set_xlabel(x.columns[top2[0]])
    ax.set_ylabel(x.columns[top2[1]])

    return fig

fig = plot_reconstruction(8)
fig = plot_reconstruction(6)
fig = plot_reconstruction(4)
fig = plot_reconstruction(2)

# %% [markdown]
# Here we see that the fidelity of the reconstruction decays rapidly as we move to fewer components.
# However, these were the dominant elements in the original space.
# If we consider some other elements we'll see much worse reconstruction much earlier.

# %%
nc = 8  # looked fine for [Cr, Mo]

z_trunc = z_manual[:, :nc]
x_recon = np.dot(z_trunc, pca.components_[:nc, :]) + np.mean(x.values, axis=0)

fig, ax = plt.subplots()
ax.scatter(*x_recon[:, :2].T, label='Reconstruction')
ax.scatter(*x.values[:, :2].T, label='Original',
        marker='s', edgecolor='tab:orange', facecolor='none')
ax.legend()
ax.text(0.95, 0.05, f'{nc} components; {var_c[nc]*100:.2f}% explained variance',
        ha='right', transform=ax.transAxes)

ax.set_xlabel(x.columns[0])
ax.set_ylabel(x.columns[1])

# %% [markdown]
# Since the principal components did not consider the variations in these (Si, C), they are not captured perfectly here.

# %% [markdown]
# ## [Check your understanding]
#
# Try repeating this reconstruction experiment using rescaled features.
# Follow these steps:
# * Apply a feature scaler to the data
# * Plot the PCA embedding from scaled features
# * Plot the explained variance curve
# * Try reconstructing the original data from the low-dimensional embeddings
# > Note: you will have to invert the feature scaling in addition to the embedding!

# %%

# %% [markdown]
# # Manifold learning
#
# Manifold learning is another term for nonlinear dimensionality reduction.
# Roughly speaking, a [manifold](https://en.wikipedia.org/wiki/Manifold) is an $n$-dimensional surface that is locally smooth.
# The term "manifold learning" therefore means searching for a smooth, low-dimensional surface in a high-dimensional space.
#
# Here's a nice motivating example that shows the difference between a **linear projection** and a **nonlinear manifold**:
#
# <img src="./assets/manifold_learning_spiral.jpg" height=400 alt="A 3D spiral dataset representing a manifold that can be unrolled into 2D">
#
# If we plot the data just based on the $x$ coordinate, we end up with the picture on the left. But if we find patterns in the data using unsupervised learning, we can "unroll" the spiral and get a new representation like the one on the right.
#
# Let's try this with a toy dataset before we move to our alloy compositions.

# %%
from sklearn import datasets
from plotly import express as px

S, t = datasets.make_swiss_roll(n_samples=400)

px.scatter_3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], color=t)

# %% [markdown]
# If we try to apply PCA on this data, we will not get what we want:

# %%
from sklearn import decomposition
from matplotlib import pyplot as plt

# fit the model
pca = decomposition.PCA().fit(S)
St = pca.transform(S)

fig, ax = plt.subplots()
ax.scatter(St[:, 0], St[:, 1], c=t)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# Why?
# Because PCA is a linear projection and linear methods can never reproduce nonlinear behavior.
# We need to introduce a nonlinearity in our learning pipeline...

# %% [markdown]
# ## Kernel PCA
#
# We can apply the "kernel trick" to use PCA for nonlinear data.
# First we apply the nonlinear kernel and then we do the linear projection.
# This is very similar to the example of polynomial features in a linear regression.
#
# Here is a great visual example of the kernel trick at work:
#
# <img src="./assets/kernel_trick.jpg" width=600 alt="Diagram illustrating the kernel trick: projecting linearly inseparable data into a higher dimension where it becomes separable">
#
# After this nonlinear transformation, the decision boundary can be very easily defined by a linear model.
#
# Let's try the `poly` kernel on the nonlinear data above.
# Note that for manifold learning, `sklearn` uses `fit()` and `transform()` just like the PCA interface instead of the `predict()` for supervised learning:

# %%
# fit the model
pca = decomposition.KernelPCA(kernel='poly').fit(S)

# project X using PCA
St = pca.transform(S)

# plot result
fig, ax = plt.subplots()
ax.scatter(St[:, 0], St[:, 1], c=t)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# Unfortunately, this result doesn't look any "better" than the linear PCA.
# We might also try the `rbf` option, which stands for Radial Basis Function.
# This is a localized basis function that links nearby points together, which makes sense when trying to identify this spiral manifold.

# %%
# fit the model
pca = decomposition.KernelPCA(kernel='rbf').fit(S)

# project X using PCA
St = pca.transform(S)

# plot result
fig, ax = plt.subplots()
ax.scatter(St[:, 0], St[:, 1], c=t)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# Nope, the RBF detects the wrong structure in the data.
# That's fine, there's no guarantee that more complex models will yield better results for any given problem.

# %% [markdown]
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

# %% [markdown]
# One of the main decisions here is what constitutes adjacency.
# The default in `sklearn` is to build the nearest neighbors graph.

# %%
from sklearn import manifold

Z = manifold.SpectralEmbedding().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %% [markdown]
# The default for `affinity` is the nearest neighbor graph, which does not appear to work well.
# What if we try the `rbf` option like we did above for the kernel PCA?

# %%
Z = manifold.SpectralEmbedding(affinity='rbf').fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %% [markdown]
# This actually works pretty well!
# Note that the curvature in $Z_1$ is entirely spurious and the discovered manifold is entirely 1D.
# This is a common feature of manifold learning approaches when there are extra dimensions.

# %% [markdown]
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

# %% [markdown]
# ## Other spectral methods

# %% [markdown]
# Here are a few other methods from the `manifold` module in no particular order.
# You can read about their assumptions in the `scikit-learn` documentation or on other websites.

# %%
from sklearn import manifold

Z = manifold.Isomap().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %%
Z = manifold.LocallyLinearEmbedding().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %%
Z = manifold.MDS().fit_transform(S)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=t)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %% [markdown]
# ## Where manifold learning fails
#
# Let's revisit our composition data from above.

# %%
x = data.loc[:, ' C':'Nb + Ta']
# here's a one-liner to encode the str labels as int:
_, y = np.unique([it[0] for it in data['Alloy code']], return_inverse=True)

# %% [markdown]
# We can try `SpectralEmbedding` again:

# %%
Z = manifold.SpectralEmbedding(random_state=0).fit_transform(x)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %% [markdown]
# The `nearest_neighbors` affinity does not seem to be working.
# Let's go back to `rbf`:

# %%
Z = manifold.SpectralEmbedding(affinity='rbf', random_state=0).fit_transform(x)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %% [markdown]
# This is almost identical to PCA.
# What about diffusion maps?

# %%
dist = distance.squareform(distance.pdist(x))

epsilon = np.percentile(dist, 100)
L = np.exp(-dist**2/epsilon**2)

D = np.diag(np.sum(L, axis=1))
P = np.linalg.inv(D) @ L

w, v = np.linalg.eig(P)
plt.scatter(*v[:, 1:3].T, c=y)

# %% [markdown]
# You can see that depending on the choice of $\varepsilon$ we will get wildly different results, eventually ending up back at something like the linear result.

# %% [markdown]
# # Semi-supervised learning
#
# As a final comment on unsupervised learning, we should note that it is possible to come across problems that are best solved with hybrid unsupervised + supervised learning schemes.
# In these cases, we might have a few labeled points (e.g., from a very expensive experiment or simulation) while the vast majority are unlabeled.
# More than either supervised or unsupervised, this can require a lot of expert knowledge to work well.

# %% [markdown]
# ## In `scikit-learn`
#
# Let's try out one of the `sklearn.semi_supervised` builtins:

# %%
from sklearn import semi_supervised

x = data.loc[:, ' C':'Nb + Ta']
p = decomposition.PCA().fit_transform(x)

# %% [markdown]
# Let's remove 99% of the data by replacing the class labels with `-1` (per the instructions in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading)):

# %%
import numpy as np

remove_n = int(0.99 * x.shape[0])  # remove 98% of the labels!

sparse_y = np.array(y)  # create a copy of the labels

rng = np.random.RandomState(0)  # set random state so we always get same result
remove_idx = rng.choice(np.arange(y.shape[0]), remove_n, replace=False)

sparse_y[remove_idx] = -1  # remove some labels

# %% [markdown]
# We can visualize the result of this "un-labeling" by splitting the dataset up in the scatter plot:

# %%
idx = sparse_y > 0  # separate the plotted points into labeled/unlabeled

fig, ax = plt.subplots()
ax.scatter(p[idx, 0], p[idx, 1], c=sparse_y[idx])
ax.plot(p[~idx, 0], p[~idx, 1], '.', color=np.ones(3)*0.8, zorder=0)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# Now we fit the model and check its performance against the full labeled dataset:

# %%
model = semi_supervised.LabelSpreading().fit(p, sparse_y)
print(model.score(p, y))

# %% [markdown]
# This performance may surprise you.
# Let's look at the result:

# %%
predicted_class = model.predict(p)
incorrect = (predicted_class^y).astype(bool)

fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=predicted_class)
ax.plot(p[incorrect, 0], p[incorrect, 1], 'rx')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# ## Unsupervised UMAP
#
# Uniform Manifold APproximation uses more sophisticated machinery to compute something like the spectral embedding.
# It ends up being able to resolve irregularly space data like [this visualization of UMAP topology](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html):
#
# <img src="./assets/umap_graph.jpg" width=600 alt="Visualization of a high-dimensional dataset projected into 2D using the UMAP manifold learning algorithm">
#
# To use the package, we first need to install the UMAP package using `pip`:

# %%
# !pip install umap-learn

# %% [markdown]
# Now we can import the `umap` module and fit a `UMAP` object.
# It has an interface just like `sklearn`:

# %%
import umap

Z = umap.UMAP(random_state=0).fit_transform(x)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %% [markdown]
# We can see here the result looks quite different from PCA -- instead of a few contiguous clusters there are many discrete clusters spread across the space.
# We can investigate this result using `plotly.express`:

# %%
px.scatter(x=Z[:, 0], y=Z[:, 1], color=y, hover_name=data['Alloy code'])

# %% [markdown]
# ## Supervised UMAP
#
# We can also do supervised and semi-supervised learning with UMAP, to create clusters that are informed by the labels.
# Here's the fully supervised case:

# %%
Z = umap.UMAP(random_state=0).fit_transform(x, y=y)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

# %% [markdown]
# In this particular case, it doesn't change much (if at all).
# For completeness, here's the semi-supervised case, where we artificially hide some of the labels:

# %%
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

# %%
