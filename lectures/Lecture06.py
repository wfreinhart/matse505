# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Lecture06

# %%
class Context(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault('data', None)
        self.setdefault('tensors', {})
        self.setdefault('model', None)
        self.setdefault('viz', None)

ctx = Context()

# %% [markdown]
# Today's topics:
# * Feature scaling
# * Reconstruction
# * Manifold learning
# * Semi-supervised learning
#
# Let's load the alloys dataset from before...
#
# Drop the outlier we found last time:
#
# And define a space to work in:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    import pandas as pd
    import numpy as np
    import os

    # Set the path to the data file
    filename = 'steels.csv'
    local_path = f'../datasets/{filename}'
    github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

    # Load the data: try local path first, fallback to GitHub for Colab
    if os.path.exists(local_path):
        data = pd.read_csv(local_path)
    else:
        data = pd.read_csv(github_url)
    data                            # show a view of the data file

    outlier = np.argmax(data[' Tensile Strength (MPa)'])
    clean_data = data.drop(index=outlier)

    x = clean_data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']
    ctx['data'] = data
    ctx['tensors']['clean_data'] = clean_data
    ctx['tensors']['filename'] = filename
    ctx['tensors']['github_url'] = github_url
    ctx['tensors']['local_path'] = local_path
    ctx['tensors']['outlier'] = outlier
    ctx['tensors']['x'] = x
run_module(ctx)

# %% [markdown]
# # Feature scaling
#
# One thing to watch out for when doing unsupervised representation learning is features with different units. If one of the features has magnitudes of $10^6$ and another has $10^1$, the big one will completely dominate the embedding. We can get around this using **feature scaling**.
#
# <img src="../lectures/assets/feature_scaling.jpg" width=600 alt="Comparison of data distribution before and after different scaling methods: StandardScaler, MinMaxScaler, and RobustScaler">
#
# Making sure the data are roughly isotropic really helps us capture more information in the embedding.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    clean_data = ctx.get('tensors', {}).get('clean_data')
    x = ctx.get('tensors', {}).get('x')
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
    ctx['model_artifacts']['scaler'] = scaler
    ctx['tensors']['pca'] = pca
run_module(ctx)

# %% [markdown]
# ## Min-Max scaler
#
# This is the most straightforward preprocesser you could imagine -- it simply rescales each column to have a min value of 0 and max value of 1.
#
# $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$
#
# One interesting consequence of the `MinMaxScaler` is the mitigation of the outlier in `Tensile Strength (MPa)`.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    clean_data = ctx.get('tensors', {}).get('clean_data')
    x = ctx.get('tensors', {}).get('x')
    scaler = preprocessing.MinMaxScaler().fit(x)
    xs = scaler.transform(x)

    pca = decomposition.PCA().fit(xs)
    P = pca.transform(xs)

    fig, ax = plt.subplots()
    im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
    plt.colorbar(im)
    ctx['model_artifacts']['scaler'] = scaler
    ctx['tensors']['pca'] = pca
run_module(ctx)

# %% [markdown]
# ## Power transformer
#
# The power transformer is a scheme to make a non-Gaussian distribution look more Gaussian.
# Here's a visual example:
#
# <img src="../lectures/assets/power_transformer.jpg" width=600 alt="Effect of PowerTransformer on skewed data distributions to make them more Gaussian-like">
#
# Why would we do this?
# This is an attempt to avoid the long tails dominating the reduced space.
#
# Here the power transformer provides a middle ground between the standard scaler and min-max scaler.
# Note that all three of these preprocessing methods make the point cloud more isotropic and portray additional features compared to the PCA on raw data.
#
# A quick note on the inverse transform:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    clean_data = ctx.get('tensors', {}).get('clean_data')
    x = ctx.get('tensors', {}).get('x')
    scaler = preprocessing.PowerTransformer().fit(x)
    xs = scaler.transform(x)

    pca = decomposition.PCA().fit(xs)
    P = pca.transform(xs)

    fig, ax = plt.subplots()
    im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
    plt.colorbar(im)

    print( 'original', x.values[0] )
    scaler = preprocessing.StandardScaler().fit(x)
    xs = scaler.transform(x)
    print( 'scaled', xs[0] )
    print( 'reconstructed', scaler.inverse_transform(xs)[0] )
    ctx['model_artifacts']['scaler'] = scaler
    ctx['tensors']['pca'] = pca
run_module(ctx)

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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Reconstruction
#
# Something we haven't discussed with these dimensionality reduction approaches is the concept of reconstructing the high-dimensional object from the low-dimensional embedding.
# It's important to understand how this works, its limitations, and how it might be applied.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Quick review of PCA
#
# Let's review how the PCA projection works.
# We'll go back to the alloy compositions data for this exercise.
#
# Now that we have this $z$ embedding, we can evaluate the data in this low-dimensional space.
#
# Remember these projections are nothing more than the product of the original array (shifted to have zero mean) with the principal component vectors:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
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
    ctx['tensors']['pca'] = pca
    ctx['tensors']['x'] = x
    ctx['tensors']['x_shift'] = x_shift
    ctx['tensors']['z'] = z
    ctx['tensors']['z_manual'] = z_manual
run_module(ctx)

# %% [markdown]
# ## Concept of reconstruction
#
# We can also take these embeddings and "un-project" them back to the originals.
# (Remember this embedding operation was basically a rotation).
# Let's evaluate these in the 2D plane that corresponds to the greatest variance in the data:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pca = ctx.get('tensors', {}).get('pca')
    x = ctx.get('tensors', {}).get('x')
    z_manual = ctx.get('tensors', {}).get('z_manual')
    ascending = np.argsort(pca.components_[0])
    descending = ascending[::-1]
    top2 = descending[:2]
    print(top2)
    print(x.columns[top2])

    x_recon = np.dot(z_manual, pca.components_) + np.mean(x.values, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
    ax.scatter(*x.values[:, top2].T, label='Original',
               marker='s', edgecolor='tab:orange', facecolor='none')
    ax.legend()
    ctx['tensors']['ascending'] = ascending
    ctx['tensors']['descending'] = descending
    ctx['tensors']['top2'] = top2
    ctx['tensors']['x_recon'] = x_recon
run_module(ctx)

# %% [markdown]
# ## Information loss
#
# We can plot the cumulative explained variance to see how many of these are significant.
#
# While this is slightly subjective (or at least depends on the intended application), we can probably say that we won't be able to see a difference with more than 9 components (>99.99% explained variance).
# Let's investigate what happens when we use less or more components.
#
# So far, so good. Let's make a function and try using subsequently fewer components.
#
# Here we see that the fidelity of the reconstruction decays rapidly as we move to fewer components.
# However, these were the dominant elements in the original space.
# If we consider some other elements we'll see much worse reconstruction much earlier.
#
# Since the principal components did not consider the variations in these (Si, C), they are not captured perfectly here.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pca = ctx.get('tensors', {}).get('pca')
    plot_reconstruction = ctx.get('tensors', {}).get('plot_reconstruction')
    top2 = ctx.get('tensors', {}).get('top2')
    x = ctx.get('tensors', {}).get('x')
    z_manual = ctx.get('tensors', {}).get('z_manual')
    fig, ax = plt.subplots()
    n = np.arange(pca.n_components_)+1
    var_c = np.cumsum(pca.explained_variance_ratio_)
    print(var_c)
    ax.plot(n, var_c, '.-')
    ax.set_xlabel('Components')
    ax.set_ylabel('Cumulative explained variance')

    nc = 9
    z_trunc = z_manual[:, :nc]
    x_recon = np.dot(z_trunc, pca.components_[:nc, :]) + np.mean(x.values, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
    ax.scatter(*x.values[:, top2].T, label='Original',
               marker='s', edgecolor='tab:orange', facecolor='none')
    ax.legend()

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
    ctx['tensors']['var_c'] = var_c
    ctx['tensors']['x_recon'] = x_recon
    ctx['tensors']['z_trunc'] = z_trunc
run_module(ctx)

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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Manifold learning
#
# Manifold learning is another term for nonlinear dimensionality reduction.
# Roughly speaking, a [manifold](https://en.wikipedia.org/wiki/Manifold) is an $n$-dimensional surface that is locally smooth.
# The term "manifold learning" therefore means searching for a smooth, low-dimensional surface in a high-dimensional space.
#
# Here's a nice motivating example that shows the difference between a **linear projection** and a **nonlinear manifold**:
#
# <img src="../lectures/assets/manifold_learning_spiral.jpg" height=400 alt="A 3D spiral dataset representing a manifold that can be unrolled into 2D">
#
# If we plot the data just based on the $x$ coordinate, we end up with the picture on the left. But if we find patterns in the data using unsupervised learning, we can "unroll" the spiral and get a new representation like the one on the right.
#
# Let's try this with a toy dataset before we move to our alloy compositions.
#
# If we try to apply PCA on this data, we will not get what we want:
#
# Why?
# Because PCA is a linear projection and linear methods can never reproduce nonlinear behavior.
# We need to introduce a nonlinearity in our learning pipeline...

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    datasets = ctx.get('tensors', {}).get('datasets')
    px = ctx.get('tensors', {}).get('px')
    from sklearn import datasets
    from plotly import express as px

    S, t = datasets.make_swiss_roll(n_samples=400)

    px.scatter_3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], color=t)

    from sklearn import decomposition
    from matplotlib import pyplot as plt

    # fit the model
    pca = decomposition.PCA().fit(S)
    St = pca.transform(S)

    fig, ax = plt.subplots()
    ax.scatter(St[:, 0], St[:, 1], c=t)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ctx['tensors']['pca'] = pca
run_module(ctx)

# %% [markdown]
# ## Kernel PCA
#
# We can apply the "kernel trick" to use PCA for nonlinear data.
# First we apply the nonlinear kernel and then we do the linear projection.
# This is very similar to the example of polynomial features in a linear regression.
#
# Here is a great visual example of the kernel trick at work:
#
# <img src="../lectures/assets/kernel_trick.jpg" width=600 alt="Diagram illustrating the kernel trick: projecting linearly inseparable data into a higher dimension where it becomes separable">
#
# After this nonlinear transformation, the decision boundary can be very easily defined by a linear model.
#
# Let's try the `poly` kernel on the nonlinear data above.
# Note that for manifold learning, `sklearn` uses `fit()` and `transform()` just like the PCA interface instead of the `predict()` for supervised learning:
#
# Unfortunately, this result doesn't look any "better" than the linear PCA.
# We might also try the `rbf` option, which stands for Radial Basis Function.
# This is a localized basis function that links nearby points together, which makes sense when trying to identify this spiral manifold.
#
# Nope, the RBF detects the wrong structure in the data.
# That's fine, there's no guarantee that more complex models will yield better results for any given problem.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    t = ctx.get('tensors', {}).get('t')
    # fit the model
    pca = decomposition.KernelPCA(kernel='poly').fit(S)

    # project X using PCA
    St = pca.transform(S)

    # plot result
    fig, ax = plt.subplots()
    ax.scatter(St[:, 0], St[:, 1], c=t)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # fit the model
    pca = decomposition.KernelPCA(kernel='rbf').fit(S)

    # project X using PCA
    St = pca.transform(S)

    # plot result
    fig, ax = plt.subplots()
    ax.scatter(St[:, 0], St[:, 1], c=t)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ctx['tensors']['pca'] = pca
run_module(ctx)

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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    manifold = ctx.get('tensors', {}).get('manifold')
    t = ctx.get('tensors', {}).get('t')
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
run_module(ctx)

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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    distance = ctx.get('tensors', {}).get('distance')
    t = ctx.get('tensors', {}).get('t')
    from scipy.spatial import distance

    dist = distance.squareform(distance.pdist(S))

    epsilon = np.percentile(dist, 1)
    L = np.exp(-dist**2/epsilon**2)

    D = np.diag(np.sum(L, axis=1))
    P = np.linalg.inv(D) @ L

    w, v = np.linalg.eig(P)
    plt.scatter(*v[:, 1:3].T, c=t)
    ctx['tensors']['dist'] = dist
    ctx['tensors']['epsilon'] = epsilon
run_module(ctx)

# %% [markdown]
# ## Other spectral methods
#
# Here are a few other methods from the `manifold` module in no particular order.
# You can read about their assumptions in the `scikit-learn` documentation or on other websites.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    manifold = ctx.get('tensors', {}).get('manifold')
    t = ctx.get('tensors', {}).get('t')
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
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    distance = ctx.get('tensors', {}).get('distance')
    it = ctx.get('tensors', {}).get('it')
    manifold = ctx.get('tensors', {}).get('manifold')
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
    ctx['tensors']['dist'] = dist
    ctx['tensors']['epsilon'] = epsilon
    ctx['tensors']['x'] = x
run_module(ctx)

# %% [markdown]
# # Semi-supervised learning
#
# As a final comment on unsupervised learning, we should note that it is possible to come across problems that are best solved with hybrid unsupervised + supervised learning schemes.
# In these cases, we might have a few labeled points (e.g., from a very expensive experiment or simulation) while the vast majority are unlabeled.
# More than either supervised or unsupervised, this can require a lot of expert knowledge to work well.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## In `scikit-learn`
#
# Let's try out one of the `sklearn.semi_supervised` builtins:
#
# Let's remove 99% of the data by replacing the class labels with `-1` (per the instructions in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading)):
#
# We can visualize the result of this "un-labeling" by splitting the dataset up in the scatter plot:
#
# Now we fit the model and check its performance against the full labeled dataset:
#
# This performance may surprise you.
# Let's look at the result:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    semi_supervised = ctx.get('tensors', {}).get('semi_supervised')
    y = ctx.get('tensors', {}).get('y')
    from sklearn import semi_supervised

    x = data.loc[:, ' C':'Nb + Ta']
    p = decomposition.PCA().fit_transform(x)

    import numpy as np

    remove_n = int(0.99 * x.shape[0])  # remove 98% of the labels!

    sparse_y = np.array(y)  # create a copy of the labels

    rng = np.random.RandomState(0)  # set random state so we always get same result
    remove_idx = rng.choice(np.arange(y.shape[0]), remove_n, replace=False)

    sparse_y[remove_idx] = -1  # remove some labels

    idx = sparse_y > 0  # separate the plotted points into labeled/unlabeled

    fig, ax = plt.subplots()
    ax.scatter(p[idx, 0], p[idx, 1], c=sparse_y[idx])
    ax.plot(p[~idx, 0], p[~idx, 1], '.', color=np.ones(3)*0.8, zorder=0)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    model = semi_supervised.LabelSpreading().fit(p, sparse_y)
    print(model.score(p, y))

    predicted_class = model.predict(p)
    incorrect = (predicted_class^y).astype(bool)

    fig, ax = plt.subplots()
    ax.scatter(p[:, 0], p[:, 1], c=predicted_class)
    ax.plot(p[incorrect, 0], p[incorrect, 1], 'rx')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ctx['model'] = model
    ctx['tensors']['idx'] = idx
    ctx['tensors']['incorrect'] = incorrect
    ctx['tensors']['predicted_class'] = predicted_class
    ctx['tensors']['remove_idx'] = remove_idx
    ctx['tensors']['remove_n'] = remove_n
    ctx['tensors']['rng'] = rng
    ctx['tensors']['sparse_y'] = sparse_y
    ctx['tensors']['x'] = x
run_module(ctx)

# %% [markdown]
# ## Unsupervised UMAP
#
# Uniform Manifold APproximation uses more sophisticated machinery to compute something like the spectral embedding.
# It ends up being able to resolve irregularly space data like [this visualization of UMAP topology](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html):
#
# <img src="../lectures/assets/umap_graph.jpg" width=600 alt="Visualization of a high-dimensional dataset projected into 2D using the UMAP manifold learning algorithm">
#
# To use the package, we first need to install the UMAP package using `pip`:
#
# Now we can import the `umap` module and fit a `UMAP` object.
# It has an interface just like `sklearn`:
#
# We can see here the result looks quite different from PCA -- instead of a few contiguous clusters there are many discrete clusters spread across the space.
# We can investigate this result using `plotly.express`:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    px = ctx.get('tensors', {}).get('px')
    umap = ctx.get('tensors', {}).get('umap')
    x = ctx.get('tensors', {}).get('x')
    y = ctx.get('tensors', {}).get('y')
    # !pip install umap-learn

    import umap

    Z = umap.UMAP(random_state=0).fit_transform(x)

    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1], c=y)
    ax.set_xlabel('$Z_0$')
    ax.set_ylabel('$Z_1$')

    px.scatter(x=Z[:, 0], y=Z[:, 1], color=y, hover_name=data['Alloy code'])
run_module(ctx)

# %% [markdown]
# ## Supervised UMAP
#
# We can also do supervised and semi-supervised learning with UMAP, to create clusters that are informed by the labels.
# Here's the fully supervised case:
#
# In this particular case, it doesn't change much (if at all).
# For completeness, here's the semi-supervised case, where we artificially hide some of the labels:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    umap = ctx.get('tensors', {}).get('umap')
    x = ctx.get('tensors', {}).get('x')
    y = ctx.get('tensors', {}).get('y')
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
    ctx['tensors']['remove_idx'] = remove_idx
    ctx['tensors']['remove_n'] = remove_n
    ctx['tensors']['rng'] = rng
    ctx['tensors']['sparse_y'] = sparse_y
run_module(ctx)
