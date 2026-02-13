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
# id: Lecture17_latent_space
# type: Foundational
# parent_lecture: Lecture17
# ---
#
# ## Latent space
#
# Perform a PCA embedding of the high-dimensional latent space:
#
# Plot examples in the space:
#
# Plot the reconstructions instead to "see what the model sees":

# %%
import numpy as np

all_y = []
all_z = []
for x, y in dl_valid:
    all_y.append(y)
    with torch.no_grad():
        z = model.encoder(x)
    all_z.append(z.detach().numpy())

y = np.hstack(all_y)  # classes
z = np.vstack(all_z)  # latent codes

from sklearn import decomposition

pca = decomposition.PCA()
coefs = pca.fit_transform(z)

# plot the explained variance of the embedding
fig, ax = plt.subplots()
_ = ax.plot(np.arange(1, pca.n_components_+1), np.cumsum(pca.explained_variance_ratio_), '.-')
_ = ax.set_xlabel('Components')
_ = ax.set_ylabel('Explained Variance')
ax.set_xscale('log')

# plot the class labels
fig, ax = plt.subplots()
ax.scatter(coefs[:, 0], coefs[:, 1], c=y)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import cluster

# perform the clustering
km = cluster.KMeans(n_clusters=48, n_init='auto', random_state=0).fit(coefs[:, :2])

# determine which sample IDs are closest to the cluster centers
center_id = []
for i, c in enumerate(km.cluster_centers_):
    dist = np.linalg.norm(c - coefs[:, :2], axis=1)
    center_id.append( np.argmin(dist) )

# plot the results
fig, ax = plt.subplots(figsize=(10, 10))
_ = ax.scatter(*coefs[:, :2].T, c=y)
_ = ax.set_xlabel('PC 1')
_ = ax.set_xlabel('PC 2')
ax.set_aspect('equal')
# add image thumbnails as annotations to the scatter plot
thumbnail_size = (64, 64)
for id in center_id:
    img_tensor, _ = dl_valid.dataset[id]
    img = img_tensor.numpy().transpose()
    image_offset = OffsetImage(img, zoom=0.15, cmap='Greys_r')
    ab = AnnotationBbox(image_offset, coefs[id, :2], xycoords='data', frameon=False)
    ax.add_artist(ab)

# plot the results
fig, ax = plt.subplots(figsize=(10, 10))
_ = ax.scatter(*coefs[:, :2].T, c=y)
_ = ax.set_xlabel('PC 1')
_ = ax.set_xlabel('PC 2')
ax.set_aspect('equal')
# add image thumbnails as annotations to the scatter plot
thumbnail_size = (64, 64)
for id in center_id:
    x, _ = dl_valid.dataset[id]
    img_tensor = model(x.unsqueeze(0)).squeeze(0)
    img = img_tensor.detach().numpy().transpose()
    image_offset = OffsetImage(img, zoom=0.15, cmap='Greys_r')
    ab = AnnotationBbox(image_offset, coefs[id, :2], xycoords='data', frameon=False)
    ax.add_artist(ab)
