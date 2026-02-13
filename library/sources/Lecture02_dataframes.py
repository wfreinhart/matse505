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
# id: Lecture02_dataframes
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## DataFrames
#
# *What is `type` of this variable that Pandas created?*
#
# Let's learn more about how this is stored using the `type` builtin function:
#
# Alright, it looks like this is a `pandas DataFrame`.
# If you Google this it will take you to the [documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
# This lists all the `Attributes` and `Methods` that are built into the `DataFrame` object type.
# For instance, we can see that there are several different `Attributes` that let us access specific data elements:
#
# * `at`: Access a single value for a row/column label pair.
# * `loc`: Access a group of rows and columns by label(s) or a boolean array.
# * `iat`: Access a single value for a row/column pair by integer position.
# * `iloc`: Purely integer-location based indexing for selection by position.
#
# It turns out that `loc` can basically act as `at` by just specifying a single element while the reverse is not true, so we'll just stick to `loc`.
# > Note: it's very common to find multiple different ways to achieve an outcome! Use whichever method you prefer.
#
# Now that we know to use `loc`, we can achieve the result we were looking for:
#
# Inside this `0` index is an entire row (vector) of data (making the entire `data` table above a matrix).
# It turns out we can also use slices just as we did with `list`:

# %%
print(type(data))

data.loc[0]

data.loc[:5]

data.loc[0:10:2]
