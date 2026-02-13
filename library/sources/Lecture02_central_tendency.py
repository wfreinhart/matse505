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
# id: Lecture02_central_tendency
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Central tendency
#
# There are a few key quantities we might calculate for a single `Series`.
# The first category would be "central tendency" like `mean` and `median`:
#
# We can get these same values using the `mean()` and `median` methods of the `Series` object:
#
# Note this is just a wrapper for the `numpy` function that utilizes the data stored in the `Series`.
# We can even see this by accessing the `values` attribute of the `Series`:
#
# There it is -- `Series` is just wrapping a `numpy.ndarray` object under the hood!
#
# What about on the whole `DataFrame`?
#
# Again, `DataFrame` is storing a bunch of `numpy.ndarray` objects.
#
# Anyways, we can continue to perform calculations with either the `numpy` functions directly or using the wrapper methods:
#
# > The warning is telling us that "nuisance columns" (non-numerical data) are being ignored but that future versions of `pandas` will not follow this behavior.

# %%
print(np.mean(x), np.median(x))

print(x.mean(), x.median())

print(type(x.values))
print(x.values)

print(type(data.values))

np.mean(data, axis=0)

data.mean(axis=0)
