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
# id: Lecture05_k_means
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## k-Means
#
# Another common scheme is the $k$-means algorithm.
# $k$-means minimizes within-cluster variances of $k$ clusters.
# It basically means it looks for compact groups and splits the data along any "gaps" or "ridges" as shown in the figure below:
#
# <img src="../lectures/assets/kmeans_viz.jpg" width=500 alt="Visualization of K-Means clustering centroids and partition boundaries">
#
# The interface is exactly like GMM since we're using a well-developed library:
#
# We can visualize the outcome in 3D just like last time:
#
# We can compare this to the results of the `Alloy family` codes we developed last time:
#
# One thing we see right away is that we fitted too many clusters compared to the `Alloy family` codes.
#
# If we want to see how k-means compares to the labels, we can reduce the `n_clusters` to 4 (this is a **hyperparameter**).
#
# Here we see that several of the clusters do match nicely with the `Alloy family` codes, but two are sort of entangled. Remember there was no guarantee that these labels would match those! This just shows that there is an intrinsic, obvious distinction between the high-V compounds, the high-Cr compounds, and the rest.
#
# We can also observe that the 3D view of the data that gave the clearest distinction in the GMM does not correspond to the real labels:
#
# This is a critical aspect of clustering methods that you must understand in order to deploy them responsibly!

# %%
from sklearn import cluster

model = cluster.KMeans().fit(x)
labels = model.predict(x)

px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=labels)

from sklearn import preprocessing

data['Alloy family'] = [it[0] for it in data['Alloy code']]

encoder = preprocessing.LabelEncoder().fit(data['Alloy family'])
y = encoder.transform(data['Alloy family'])  # these are numerical so we can plot them!

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=y)

model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

px.scatter_3d(x=data[' Cr'], y=data[' Mn'], z=data[' Al'], color=y)
