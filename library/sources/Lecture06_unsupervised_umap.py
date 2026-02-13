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
# id: Lecture06_unsupervised_umap
# type: Foundational
# parent_lecture: Lecture06
# ---
#
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
# !pip install umap-learn

import umap

Z = umap.UMAP(random_state=0).fit_transform(x)

fig, ax = plt.subplots()
ax.scatter(Z[:, 0], Z[:, 1], c=y)
ax.set_xlabel('$Z_0$')
ax.set_ylabel('$Z_1$')

px.scatter(x=Z[:, 0], y=Z[:, 1], color=y, hover_name=data['Alloy code'])
