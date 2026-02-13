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
# id: Lecture05_cutoff_based_methods
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Cutoff-based methods
#
# Not all clustering methods use a defined number of clusters ("$k$").
# Some provide a characteristic score for each.
# For instance, agglomerative clustering builds up a tree where each node merges two clusters from below (with leaf nodes being single points).
# At each level, a dissimilarity metric can be assessed.
# This permits a choice between either a fixed number of clusters OR a threshold dissimilarity at which to stop merging.
#
# <img src="../lectures/assets/hierarchical_clustering.jpg" width=600 alt="A dendrogram illustrating the stages of hierarchical agglomerative clustering">
#
# Note that depending on the dissimilarity metric used, the shapes of the clusters can be radically different.
#
# This is also an instance-based model, and the result of the clustering will be different for every new data sample.
# As a result, there are no separate `fit` / `predict` methods, and instead we can only perform `fit_predict` in one step (i.e., new data cannot be clustered using pre-trained model parameters).

# %%
model = cluster.AgglomerativeClustering()
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

model = cluster.AgglomerativeClustering(n_clusters=4)
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

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

model = cluster.AgglomerativeClustering(distance_threshold=0.3, n_clusters=None, linkage='single')
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

model = cluster.AgglomerativeClustering(n_clusters=3, linkage='single')
labels = model.fit_predict(x)

px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=labels)
