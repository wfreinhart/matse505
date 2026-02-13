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
# id: Lecture04_k_neighbors
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## K Neighbors
#
# Based on these results, we can imagine that K Neighbors will also work fairly well. Let's try it:
#
# Indeed, K Neighbors correctly votes 100% of the time (*on unseen test data!*).
# This means our problem is too easy -- we'll now make it harder to learn more about how the models work.

# %%
from sklearn import neighbors

model = neighbors.KNeighborsClassifier().fit(xtrain, ytrain)

model.score(xtest, ytest)
