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
# id: Lecture02_bar_charts
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Bar charts
#
# **Bar charts** look like histograms but show labeled values instead of the distribution of values.
# The `plt.bar` function takes arguments `(labels, values)` like so:
#
# This is a little crammed so let's expand the width of the figure to read the symbols more clearly.
#
# What happened?
# Green checkmark means the code executed.
# But we didn't ask for it to show us anything -- we need to render the figure again:

# %%
sym = data['Symbol']

fig, ax = plt.subplots()
ax.bar(sym, y)
ax.set_xlabel(sym.name)
ax.set_ylabel(y.name)

fig.set_figwidth(12)

fig
