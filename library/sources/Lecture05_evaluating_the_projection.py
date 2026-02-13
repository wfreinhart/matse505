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
# id: Lecture05_evaluating_the_projection
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Evaluating the projection
#
# So far we have only looked at the first two of 14 components.
# What components should we choose?
# Maybe the 14th one is the best?
# It turns out we have a very systematic way of evaluating these projections using "explained variance":
#
# The explained variance tells us how much of the overall variance in data can be captured by each component.
# From the chart we see that `sklearn` already orders the components by decreasing explained variance.
# Furthermore, the first few components capture most of the variance.
# We can also plot the cumulative explained variance ratio to make it a little easier to decide:
#
# Now we can see that the first 2 components capture 85% of the variance, and the first 3 capture 95%!

# %%
fig, ax = plt.subplots()
ax.plot(pca.explained_variance_, '.-')
ax.set_xlabel('Component #')
ax.set_ylabel('Explained variance')

fig, ax = plt.subplots()
ax.plot(np.cumsum(pca.explained_variance_ratio_), '.-')
ax.set_xlabel('Component #')
ax.set_ylabel('Cumulative explained variance')
