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
# id: Lecture02_making_a_line_chart
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Making a line chart
#
# We should start be referencing the documentation for `plt.plot`:
#
# Let's create a line plot by manually defining a `list` of $y$ values.
# The $x$ values will be implicit if not provided (as indicated by the `[x]` above).
#
# We can also provide both $x$ and $y$ values:

# %%
help(plt.plot)

y = [1, 2, 4, 8]
plt.plot( y )  # we provide 4 y values, x will go from 0 to 3

plt.plot([1, 2, 3, 10], [2, 4, 6, 10])
