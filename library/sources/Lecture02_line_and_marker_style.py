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
# id: Lecture02_line_and_marker_style
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Line and marker style
#
# We can also switch the type of line and/or marker we use in our chart:
#
# We can also use both lines and markers by specifying their style with keyword arguments `linestyle` and `marker`:
#
# You may have noticed above that we changed the linestyle two different ways:
# * `ax.plot([1, 2, 3], '--')`
# * `ax.plot([1, 2, 3], linestyle='--')`
#
# This is because python supports two different kind of function arguments: **positional** and **keyword** arguments.
# Positional means the variable to set is determined by the position in the list, while keyword indicates this explicitly using `value=`.
# Here's an example of a function signature from `numpy`:
# > `numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)`

# %%
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], '--')  # use dashed line
ax.plot([2, 5, 6, 10], '.')  # use dots instead of line
ax.plot([3, 4, 5, 4], 's')  # use squares instead of line

# i want the chart to have a red line
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], 'r--')

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], linestyle='-.', marker='d')  # dash-dot linesyle with diamond markers
