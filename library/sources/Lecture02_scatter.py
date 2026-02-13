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
# id: Lecture02_scatter
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Scatter
#
# We can add color to our charts with `scatter`, by specifying a `c` keyword argument (short for "color").
#
# This isn't helpful without a colorbar indicating the values of the colors and a label to describe it:

# %%
fig, ax = plt.subplots()

ax.scatter(x, y, c=z)

ax.set_xlabel(x.name)
ax.set_ylabel(y.name)

fig, ax = plt.subplots()

im = ax.scatter(x, y, c=z)  # save the output here (scatter object)

cb = plt.colorbar(im)  # save the output again (colorbar object)
cb.set_label(z.name)   # label the colorbar

ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
