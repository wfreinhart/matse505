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
# id: Lecture02_using_pandas_builtins
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Using `pandas` builtins
#
# I personally don't use `pandas` for plotting, but you should know its objects include handy wrappers for `pyplot`.
#
# For example, a method of the `Series` object:
#
# We can plot multiple `Series` using a method of the `DataFrame`:
#
# And finally we can introduce the (x,y)+color scatter plot from above:
#
# Note the `type` that is returned -- `matplotlib.axes._subplots.AxesSubplot`!
# All this is doing is sending some predefined commands to `pyplot`.
# If you find this syntax helpful you are more than welcome to use it.
# I don't simply because I don't always use `DataFrame` objects to store my data.

# %%
x.hist()

data.plot.bar('Symbol', 'Vacancy Formation Energy (eV)')

data.plot.scatter(x.name, y.name, c=z.name)
