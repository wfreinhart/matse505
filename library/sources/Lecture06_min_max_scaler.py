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
# id: Lecture06_min_max_scaler
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Min-Max scaler
#
# This is the most straightforward preprocesser you could imagine -- it simply rescales each column to have a min value of 0 and max value of 1.
#
# $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$
#
# One interesting consequence of the `MinMaxScaler` is the mitigation of the outlier in `Tensile Strength (MPa)`.

# %%
scaler = preprocessing.MinMaxScaler().fit(x)
xs = scaler.transform(x)

pca = decomposition.PCA().fit(xs)
P = pca.transform(xs)

fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
plt.colorbar(im)
