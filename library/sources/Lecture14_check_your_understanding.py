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
# id: Lecture14_check_your_understanding
# type: Foundational
# parent_lecture: Lecture14
# ---
#
# ## [Check your understanding]
#
# Perform feature extraction using the new pretrained model (wrapped inside our ResNetClassifier).
# Reduce their dimensions with PCA and plot the explained variance and class labels.
# How has the result changed after fine-tuning (compared to our "Feature Extraction" section at the beginning)?
#
# Finally we will use the `OffsetImage` and `AnnotationBbox` objects from `pyplot` to draw the images at their locations in the PCA space.

# %%
# feature_model = nn.Sequential(*list(model_ft.children())[:-1])

ft_features = []
ft_labels = []

with torch.no_grad():
    prog = tqdm.tqdm(dl_train, total=len(dl_train))
    for x, y in prog:
        out = model_ft.pretrained(x)  # only use the pretrained model! the last layer is our classifier
        ft_features.append(out.detach().numpy())
        ft_labels.append(y.detach().numpy())

ft_features = np.vstack(ft_features)
ft_features = ft_features.reshape(ft_features.shape[:2])
ft_labels = np.hstack(ft_labels)

# perform the PCA embedding on image features
pca = decomposition.PCA()
z_ft = pca.fit_transform(ft_features)

# plot the explained variance of the embedding
fig, ax = plt.subplots()
_ = ax.plot(np.arange(1, 101), np.cumsum(pca.explained_variance_ratio_[:100]), '.-')
_ = ax.set_xlabel('Components')
_ = ax.set_ylabel('Explained Variance')
ax.set_xscale('log')

# plot the principal components of the image features with class labels
fig, ax = plt.subplots()
_ = ax.scatter(*z_ft[:, [0, 1]].T, c=ft_labels)
_ = ax.set_xlabel('PC 1')
_ = ax.set_ylabel('PC 2')
ax.set_aspect('equal')

from sklearn import cluster

# perform the clustering
km = cluster.KMeans(n_clusters=64, random_state=0).fit(z_ft[:, :2])

# determine which sample IDs are closest to the cluster centers
center_id = []
for i, c in enumerate(km.cluster_centers_):
    dist = np.linalg.norm(c - z_ft[:, :2], axis=1)
    center_id.append( np.argmin(dist) )

# plot the cluster centers
_ = ax.plot(*km.cluster_centers_.T, 'rx')
fig

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, ax = plt.subplots(figsize=(10, 10))
_ = ax.scatter(*z_ft[:, :2].T, c=train_labels)
_ = ax.set_xlabel('PC 1')
_ = ax.set_xlabel('PC 2')
ax.set_aspect('equal')
# add image thumbnails as annotations to the scatter plot
thumbnail_size = (64, 64)
for id in center_id:
    img_tensor, _ = dl_train.dataset[id]
    img = img_tensor.numpy().transpose()
    image_offset = OffsetImage(img, zoom=0.1)
    ab = AnnotationBbox(image_offset, z_ft[id, :2], xycoords='data')
    ax.add_artist(ab)
