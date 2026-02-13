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
# id: Lecture05_density_based_methods
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Density-based methods
#
# We can instead specify a minimum density of points that constitutes a continuous cluster.
# This avoids penalizing non-convex shapes (which are completely plausible for some data distributions).
# This has important consequences for the resulting shape of clusters:
#
# <img src="../lectures/assets/dbscan_vs_others.jpg" width=600 alt="Comparison of DBSCAN clustering performance on non-spherical datasets compared to K-Means">
