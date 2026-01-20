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
# * Clustering
# * Dimensionality reduction with PCA

# %% [markdown]
# # Unsupervised learning
#
# Unsupervised learning means we have only $X$ features with  no $y$ labels.
# This is focused on understanding the data without trying to predict something directly.
# We'll discuss two flavors of unsupervised learning:
# * *Clustering* - dividing data into categorical groups
# * *Dimensionality reduction* - mapping data from high dimension to low dimension
#   * this can sometimes be referred to as "representation learning"

# %% [markdown]
# ## Clustering
#
# As discussed earlier, clustering is a ML scheme where we try to identify discrete groups of data points without having any labels associated.
# This stands in contrast to both flavors of supervised learning (regression and classification) which always have target labels.
#
# <img src="./assets/kmeans_diagram.jpg" height=300 alt="Schematic diagram of the K-Means clustering algorithm steps">
#
# This can be helpful for understanding:
# * patterns in the data,
# * outlier detection, or
# * even generating classification labels for a supervised classification task.
#
# **Note: clustering is not classification!**
#
# At first glance clustering seems a lot like classification.
# It's worth taking a moment to make sure you understand the difference -- there are no predefined labels here!
# This graphic may help make the point:
#
# <img src="./assets/classification_vs_clustering.jpg" width=600 alt="Comparison between supervised classification (with labels) and unsupervised clustering (without labels)">

# %% [markdown]
# ## Implementation in `sklearn`

# %% [markdown]
# Let's load the alloys dataset from before...

# %%
import pandas as pd
import numpy as np

data = pd.read_csv('../datasets/steels.csv')  # load into pandas
data                            # show a view of the data file

# %% [markdown]
# We start by defining our features, $X$.
#
# Unlike in the supervised case, we have no $y$ labels.
#
# Also, we don't need to do a train/test split because we have no labels to check against!

# %%
x = data.loc[:, ' C':'Nb + Ta']
x

# %% [markdown]
# ## Gaussian mixtures
#
# The first clustering model we consider is the Gaussian Mixture Model (GMM).
# This name is pretty literal, as you can see in this image:
#
# <img src="./assets/gmm_distribution_mixture.jpg" width=500 alt="A Gaussian Mixture Model representing a probability distribution as a sum of multiple Gaussian components">
#
# The idea is that each cluster of data represents a distribution with different parameters.
# We can fit the mixture of Gaussians to represent these clusters, perhaps accounting for some overlapping at the edges.
# It will look like this in 2D:
#
# <img src="./assets/gmm_2d_clusters.jpg" width=500 alt="2D visualization of clusters identified by a Gaussian Mixture Model">
#
# The GMM is a parametric model with each distribution having a mean $\mu_i$ and vector of covariance $\sigma_{ij}$ (i.e., each pair of distributions has a covariance).
# Depending on how the assumptions, the model can behave quite differently:
#
# <img src="./assets/gmm_covariances.jpg" width=500 alt="Visualization of different Gaussian Mixture Model covariance types: spherical, tied, diag, and full">

# %% [markdown]
# The implementation in `scikit-learn` is very straightforward to use.
# The interface is pretty similar to the supervised models: `fit()` and `predict()`.
# The only difference is `fit` takes only one argument: `x` (since again, there are no labels).

# %%
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4).fit(x)  # no y labels!
labels = gmm.predict(x)
print(labels)

# %% [markdown]
# This output is similar to the classification problems, but obviously we have no target labels to compare to.
# We can visualize this result in a 3D space:

# %%
from plotly import express as px

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

# %% [markdown]
# Some of the clusters intersect in this view.
# It's important to remember this is only 3 of 8 dimensions in the space -- and all 8 are considered by the GMM.
# We can view it in another slice to see a clearer picture:

# %%
px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=labels)

# %% [markdown]
# ## k-Means

# %% [markdown]
# Another common scheme is the $k$-means algorithm.
# $k$-means minimizes within-cluster variances of $k$ clusters.
# It basically means it looks for compact groups and splits the data along any "gaps" or "ridges" as shown in the figure below:
#
# <img src="./assets/kmeans_viz.jpg" width=500 alt="Visualization of K-Means clustering centroids and partition boundaries">
#
#

# %% [markdown]
# The interface is exactly like GMM since we're using a well-developed library:

# %%
from sklearn import cluster

model = cluster.KMeans().fit(x)
labels = model.predict(x)

# %% [markdown]
# We can visualize the outcome in 3D just like last time:

# %%
px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=labels)

# %% [markdown]
# We can compare this to the results of the `Alloy family` codes we developed last time:

# %%
from sklearn import preprocessing

data['Alloy family'] = [it[0] for it in data['Alloy code']]

encoder = preprocessing.LabelEncoder().fit(data['Alloy family'])
y = encoder.transform(data['Alloy family'])  # these are numerical so we can plot them!

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=y)

# %% [markdown]
# One thing we see right away is that we fitted too many clusters compared to the `Alloy family` codes.
#
# If we want to see how k-means compares to the labels, we can reduce the `n_clusters` to 4 (this is a **hyperparameter**).

# %%
model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

# %% [markdown]
# Here we see that several of the clusters do match nicely with the `Alloy family` codes, but two are sort of entangled. Remember there was no guarantee that these labels would match those! This just shows that there is an intrinsic, obvious distinction between the high-V compounds, the high-Cr compounds, and the rest.
#
# We can also observe that the 3D view of the data that gave the clearest distinction in the GMM does not correspond to the real labels:

# %%
px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=y)

# %% [markdown]
# This is a critical aspect of clustering methods that you must understand in order to deploy them responsibly!

# %% [markdown]
# ## Cutoff-based methods
#
# Not all clustering methods use a defined number of clusters ("$k$").
# Some provide a characteristic score for each.
# For instance, agglomerative clustering builds up a tree where each node merges two clusters from below (with leaf nodes being single points).
# At each level, a dissimilarity metric can be assessed.
# This permits a choice between either a fixed number of clusters OR a threshold dissimilarity at which to stop merging.
#
# <img src="./assets/hierarchical_clustering.jpg" width=600 alt="A dendrogram illustrating the stages of hierarchical agglomerative clustering">
#
# Note that depending on the dissimilarity metric used, the shapes of the clusters can be radically different.
#
# This is also an instance-based model, and the result of the clustering will be different for every new data sample.
# As a result, there are no separate `fit` / `predict` methods, and instead we can only perform `fit_predict` in one step (i.e., new data cannot be clustered using pre-trained model parameters).

# %%
model = cluster.AgglomerativeClustering()
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

# %%
model = cluster.AgglomerativeClustering(n_clusters=4)
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

# %%
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


model = cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model.fit(x)
plot_dendrogram(model)

# %%
model = cluster.AgglomerativeClustering(distance_threshold=0.3, n_clusters=None, linkage='single')
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

# %%
model = cluster.AgglomerativeClustering(n_clusters=3, linkage='single')
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=labels)

# %% [markdown]
# ## Density-based methods
#
# We can instead specify a minimum density of points that constitutes a continuous cluster.
# This avoids penalizing non-convex shapes (which are completely plausible for some data distributions).
# This has important consequences for the resulting shape of clusters:
#
# <img src="./assets/dbscan_vs_others.jpg" width=600 alt="Comparison of DBSCAN clustering performance on non-spherical datasets compared to K-Means">

# %% [markdown]
# ## Other methods
#
# `scikit-learn` has many methods available and they are all [well documented](https://scikit-learn.org/stable/modules/clustering.html).
# Here is an excellent visual guide to the performance characteristics of different approaches:
#
# <img src="./assets/sklearn_clustering_comparison.jpg" width=600 alt="Comprehensive comparison chart of different scikit-learn clustering algorithms on various dataset shapes">

# %% [markdown]
# ## [Check your understanding]
#
# Apply the DBSCAN algorithm to the clustering problem above.
# Experiment with the hyperparameters to find a reasonable result.

# %%

# %% [markdown]
# # Evaluating clustering methods
#
# There are many clustering methods implemented in `scikit-learn`, but which should we choose?
# And how many clusters should we use for a chosen model anyways?
# Unlike in supervised learning, we can't simply check how often we get the right answer...

# %% [markdown]
# ## With ground truth labels
#
# ...or can we?
#
# If we have labels (like in this case) we could check how well the chosen clustering method reflects these.
# The `adjusted_rand_score` evaluates something like accuracy while accounting for the fact that cluster indices can be in any order.
# Meanwhile the `normalized_mutual_info_score` is a measure of [mutual information](https://en.wikipedia.org/wiki/Mutual_information) (something kind of like a correlation) between the two labeling schemes.
# In either case, higher values are better (indicating greater correlation between the obtained labels and the ground truth).

# %%
from sklearn import metrics

def report_cluster_scores(labels):
    "Compare labels predicted by a clustering algorithm to ground truth."
    ars = metrics.adjusted_rand_score(y, labels)
    amis = metrics.adjusted_mutual_info_score(y, labels)

    print(f'{str(model):40s}: ARS = {ars:.3f}, AMIS = {amis:.3f}')


# apply the function to several clustering models...

model = GaussianMixture(n_components=4).fit(x)
labels = model.predict(x)
report_cluster_scores(labels)

model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)
report_cluster_scores(labels)

model = cluster.AgglomerativeClustering(n_clusters=4)
labels = model.fit_predict(x)
report_cluster_scores(labels)

# %% [markdown]
# Here we see very similar performance between all 3 model classes, with a slight edge to GMM and KMeans (tied).

# %% [markdown]
# ## Without labels
#
# In most clustering scenarios, ground truth labels are not known (this would typically lead to a classification problem).
# Thus we must make do with metrics that evaluate only the clusters themselves and not their accuracy relative to a target outcome.

# %% [markdown]
# The "Silhouette Coefficient" evaluates how well the clusters are separated from each other.
# It is calculated as:
#
# $s = \frac{b-a}{\max(a,b)}$,
#
# where $a$ is the mean distance between samples in the same cluster and $b$ is the mean distance between samples in next-nearest clusters.
# Therefore, $s=1$ corresponds to very high separation ($b \gg a$) while $s=0$ corresponds to very low separation ($b \ll a$).
# An intermediate value of $s=0.5$ means samples in different clusters are about twice as far apart as samples within clusters.
#
# We can try calculating the performance of our `KMeans` model:

# %%
model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)

metrics.silhouette_score(x, labels, metric='euclidean')

# %% [markdown]
# Let's consider how to use this score to evaluate a suitable number of clusters.
# We can write a `for` loop to iterate over a large number of possible clusters:

# %%
from matplotlib import pyplot as plt

k_list = np.arange(2, 16)
s = np.zeros(len(k_list))
for i, k in enumerate(k_list):

    model = cluster.KMeans(n_clusters=k, random_state=0).fit(x)
    labels = model.predict(x)
    s[i] = metrics.silhouette_score(x, labels, metric='euclidean')

fig, ax = plt.subplots()
ax.plot(k_list, s, 's')
ax.set_xlabel('$k$')
ax.set_ylabel('$s$')

# %% [markdown]
# This shows that $s$ increases substantially from $k=3$ to around $k=5$, then stagnates, and has non-monotonic increases until $k=15$.
# Among these options we need to evaluate how many clusters will be meaningful for our application.
# I think $k=5$ would be the first choice, then maybe $k=9$, and then $k=14$ if this is still few enough to be useful.

# %%
model = cluster.KMeans(n_clusters=14, random_state=0).fit(x)
labels = model.predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

# %% [markdown]
# What if we zoom out and try many more clusters?
# After all, it looks like $s$ is increasing with $k$...

# %%
from matplotlib import pyplot as plt
import tqdm  # a very useful package for progress bars

k_list = np.arange(5, 96, 5)
s = np.zeros(len(k_list))
for i, k in tqdm.tqdm(enumerate(k_list), total=len(k_list)):

    model = cluster.KMeans(n_clusters=k, random_state=0).fit(x)
    labels = model.predict(x)
    s[i] = metrics.silhouette_score(x, labels, metric='euclidean')

fig, ax = plt.subplots()
_ = ax.plot(k_list, s, 's')
_ = ax.set_xlabel('$k$')
_ = ax.set_ylabel('$s$')

# %% [markdown]
# This result shows that the Silhouette score generally increases as we move from 5 to 95 clusters.
# Now you have to ask yourself: is 95 clusters a useful result?
# In many cases, probably not.
# There is something unusual happening around $k=25$ clusters (a local dip in $s$ before it recovers again at $k=35$).
# Is this useful?
#
# This simply illustrates that metrics only provide guidance and typically no single metric can determine which model to use.
# The importance of different metrics will greatly depend on how you plan to deploy the models.
# There is often a general range of allowable hyperparameters (such as $k \le 10$) and then you can determine an optimal choice within this range.

# %% [markdown]
# ## [Check your understanding]
#
# Identify an optimal clustering model and hyperparameters for this alloys dataset based on one or more metrics.
# Think about how you would justify your choice with the metrics, characteristics of the algorithm, and some intended use case.

# %%

# %% [markdown]
# # Dimensionality reduction
#
# Here we will try to find a new representation for our data in a lower dimensional space.
# As such, we can also call this "representation learning."

# %% [markdown]
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
# <img src="./assets/pca_3d_to_2d.jpg" width=800 alt="Diagram illustrating Principal Component Analysis projecting 3D data onto a 2D principal component plane">
#
# Here we have 3 spatial dimensions plus a color.
# The PCA shows clear groupings in a 2D space which is a plane through the original 3D space.

# %% [markdown]
# PCA is based on the eigenvectors of the covariance matrix.
# You are probably familiar with the concept of covariance in 2D, like the following:
#
# <img src="./assets/covariance_types.jpg" width=600 alt="Visual representation of different spatial covariance structures in data">
#
# This readily extends to higher dimensions, and can be calculated with builtins such as with `numpy.cov`:
# > The transpose is needed because `numpy.cov` assumes "Each row of m represents a variable, and each column a single observation of all those variables" -- the transpose of our `DataFrame`

# %%
np.cov(x.T).shape

# %% [markdown]
# If we take the eigenvectors of this covariance matrix, we will get something special:

# %%
w, v = np.linalg.eig(np.cov(x.T))
print(w.shape, v.shape)

# %% [markdown]
# `w` are the eigenvalues while `v` are the eigenvectors of the covariance matrix (each column is one eigenvector).
# What is the use of these?
# Let's start with the eigenvalues:

# %%
fig, ax = plt.subplots()
ax.plot(np.real(w), '.')
ax.set_yscale('log')

# %% [markdown]
# The eigenvalues decay over the column index from 4e-1 to 3e-8.
# This is proportional to the variance in that dimension, so the first eigenvector will be associated with 10 million times greater variance than the last one.
# With this in mind, we can investigate the eigenvectors.

# %%
fig, ax = plt.subplots()
_ = ax.bar(x.columns, v[:, 0])

# %% [markdown]
# From this we see that something like `+Cr, +Mo, -Mn, -Ceq` is the dominant direction of variance.
# In other words, alloys with high `Cr` and `Mo` have low `Mn` and vice versa.
# As shown above, this is the direction of maximal variance.
#
# We can project the `x` values onto the first two eigenvectors.
# You should think of this like rotating your viewpoint to view the maximally varying directions in the plane (just like the example above).

# %%
projected = (x.values @ v)
print( projected.shape )

fig, ax = plt.subplots()
_ = ax.scatter(projected[:, 0], projected[:, 1])

# %% [markdown]
# So what is the point of this trick?
# Let's plot the above with the original alloy code labels.

# %%
fig, ax = plt.subplots()
_ = ax.scatter(projected[:, 0], projected[:, 1], c=y)
_ = ax.set_xlabel('Principal Component 1')
_ = ax.set_ylabel('Principal Component 2')

# %% [markdown]
# Unlike above where we searched for a suitable 3D representation (with only moderate success), here we have constructed a rotation using analysis of the dataset.
# The resulting projection reveals the ground truth labeling clearly (existing almost exclusively in the PC1 direction).

# %% [markdown]
# ## PCA with `scikit-learn`
#
# Of course there is always the `sklearn` interface for the same operation:

# %%
from sklearn import decomposition

# fit the model
pca = decomposition.PCA().fit(x)

# project X using PCA
p = pca.transform(x)
print(p.shape)

# %% [markdown]
# We see that the output of `pca.transform` is the same size as the original `x`.
# These are the coefficients of `x` projected onto the vectors identified by PCA.
# Let's see what these projections look like:

# %%
fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=y)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# If we want to understand where each `Alloy family` appears in the manifold, we could do something like this:

# %%
fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=y)

# label the centers
for i in range(4):
    center = np.mean(p[y==i], axis=0)
    ax.text(center[0], center[1], encoder.classes_[i])

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# Here we begin to see why the `KMeans` clustering was confused between the `C` and `L` families -- they do not form compact groupings in the space! Instead they are defined by some strict definitions about their elemental compositions (which we saw last time using `DecisionTreeClassifier`).
#
# We can plot the `KMeans` predictions in this space to see what it thinks should be the clusters:

# %%
model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)

fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=labels)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# %% [markdown]
# The clustering algorithm prioritizes compact groupings in the 2D space whereas the ground truth labels make more slender/anisotropic groups.

# %% [markdown]
# ## Evaluating the projection
#
# So far we have only looked at the first two of 14 components.
# What components should we choose?
# Maybe the 14th one is the best?
# It turns out we have a very systematic way of evaluating these projections using "explained variance":

# %%
fig, ax = plt.subplots()
ax.plot(pca.explained_variance_, '.-')
ax.set_xlabel('Component #')
ax.set_ylabel('Explained variance')

# %% [markdown]
# The explained variance tells us how much of the overall variance in data can be captured by each component.
# From the chart we see that `sklearn` already orders the components by decreasing explained variance.
# Furthermore, the first few components capture most of the variance.
# We can also plot the cumulative explained variance ratio to make it a little easier to decide:

# %%
fig, ax = plt.subplots()
ax.plot(np.cumsum(pca.explained_variance_ratio_), '.-')
ax.set_xlabel('Component #')
ax.set_ylabel('Cumulative explained variance')

# %% [markdown]
# Now we can see that the first 2 components capture 85% of the variance, and the first 3 capture 95%!

# %% [markdown]
# ## Outlier detection
#
# Another use of dimensionality reduction can be outlier detection. Let's switch our definition of $X$ from the compositions to the properties.

# %%
x = data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']

pca = decomposition.PCA().fit(x)
P = pca.transform(x)

fig, ax = plt.subplots()
ax.scatter(P[:, 0], P[:, 1])
ax.set_xlabel('$P_0$')
ax.set_ylabel('$P_1$')

# %% [markdown]
# Clearly something is strange with that one point off on its own. Let's check it out:

# %%
outlier = np.argmax(P[:, 0])  # find the largest value in Z_0
print(x.loc[outlier])

# %% [markdown]
# Look at that `Tensile Strength` value! If we examine the `Tensile Strength` data, we will see that it is indeed anomalous:

# %%
fig, ax = plt.subplots()
_ = ax.hist(x[' Tensile Strength (MPa)'], bins=100)
ax.set_xlabel('Tensile Strength (MPa)')
ax.set_ylabel('Count')

# %% [markdown]
# This value was almost certainly entered incorrectly (e.g., wrong decimal place). We can remove the outlier using `drop()`:

# %%
clean_data = data.drop(index=outlier)
clean_y = encoder.transform(clean_data['Alloy family'])

# %% [markdown]
# Then we can look at our manifold again:

# %%
clean_x = clean_data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']

pca = decomposition.PCA().fit(clean_x)
P = pca.transform(clean_x)

fig, ax = plt.subplots()
ax.scatter(P[:, 0], P[:, 1])
ax.set_xlabel('$P_0$')
ax.set_ylabel('$P_1$')

# %% [markdown]
# Of course here we see some more strange behavior. We could continue to investigate these additional outliers using this method (though we may not always want to remove them, some might be "real" special cases).
#
# To be clear, we could easily have found this anomaly by analyzing the `Tensile Strength (MPa)` column individually.
# However, the upshot is that we see that data point as an anomaly in the first Principal Component, without considering any column-wise statistics.

# %% [markdown]
# ## [Check your understanding]
#
# Keep identifying and removing outliers from the dataset until you are satisfied with the point cloud produced by the projection.
#
# Can you automate this process (i.e., determine a cutoff distance and automatically terminate the pruning once that cutoff is reached)?

# %%
fig, ax = plt.subplots()
_ = ax.bar(x.columns, pca.components_[:, 1])
