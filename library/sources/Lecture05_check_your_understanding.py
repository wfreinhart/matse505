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
# id: Lecture05_check_your_understanding
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## [Check your understanding]
#
# Keep identifying and removing outliers from the dataset until you are satisfied with the point cloud produced by the projection.
#
# Can you automate this process (i.e., determine a cutoff distance and automatically terminate the pruning once that cutoff is reached)?

# %%
fig, ax = plt.subplots()
_ = ax.bar(x.columns, pca.components_[:, 1])
