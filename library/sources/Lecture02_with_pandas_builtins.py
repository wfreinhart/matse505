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
# id: Lecture02_with_pandas_builtins
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## With `pandas` builtins
#
# There is a huge shortcut we can take using `pandas` again:
#
# This returns all the correlations between pairs of columns.
# It is itself a `DataFrame` so we can filter using `loc`:

# %%
xyz_data.corr()

xyz_data.corr().loc[:, 'Atomic Mass']
