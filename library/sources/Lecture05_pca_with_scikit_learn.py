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
# id: Lecture05_pca_with_scikit_learn
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## PCA with `scikit-learn`
#
# Of course there is always the `sklearn` interface for the same operation:
#
# We see that the output of `pca.transform` is the same size as the original `x`.
# These are the coefficients of `x` projected onto the vectors identified by PCA.
# Let's see what these projections look like:
#
# If we want to understand where each `Alloy family` appears in the manifold, we could do something like this:
#
# Here we begin to see why the `KMeans` clustering was confused between the `C` and `L` families -- they do not form compact groupings in the space! Instead they are defined by some strict definitions about their elemental compositions (which we saw last time using `DecisionTreeClassifier`).
#
# We can plot the `KMeans` predictions in this space to see what it thinks should be the clusters:
#
# The clustering algorithm prioritizes compact groupings in the 2D space whereas the ground truth labels make more slender/anisotropic groups.

# %%
from sklearn import decomposition

# fit the model
pca = decomposition.PCA().fit(x)

# project X using PCA
p = pca.transform(x)
print(p.shape)

fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=y)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=y)

# label the centers
for i in range(4):
    center = np.mean(p[y==i], axis=0)
    ax.text(center[0], center[1], encoder.classes_[i])

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)

fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=labels)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
