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
# id: Lecture15_visualization
# type: Foundational
# parent_lecture: Lecture15
# ---
#
# ## Visualization
#
# It can be helpful to visualize the data samples directly in the latent space.
#
# This is quite crowded, so it can be helpful to downselect using clustering techniques.
# In this case I use KMeans clustering simply to define uniformly sized groups in the $(z_0, z_1)$ space.

# %%
scale = 0.2

fig, ax = plt.subplots(figsize=(16, 16))

for i, this_y in enumerate(y):
    ax.plot(z[i, 0]+x*scale, z[i, 1]+this_y*scale)

ax.set_aspect('equal')

from sklearn import cluster

z_idx = (0, 1)
km = cluster.KMeans(n_clusters=64).fit(z[:, z_idx])

cluster_ids = []
for c in km.cluster_centers_:
    cluster_ids.append( np.argmin( np.linalg.norm(z[:, z_idx] - c, axis=1) ) )

scale = 0.5

fig, ax = plt.subplots(figsize=(16, 16))

for i in cluster_ids:
    ax.plot(z[i, z_idx[0]]+x*scale, z[i, z_idx[1]]+y[i]*scale)

ax.set_aspect('equal')
