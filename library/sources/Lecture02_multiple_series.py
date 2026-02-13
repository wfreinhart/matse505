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
# id: Lecture02_multiple_series
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Multiple series
#
# We can plot multiple series by calling `plot` twice:
#
# However, watch what happens if we don't call this in the same cell:

# %%
plt.plot([2, 4, 6, 10], [1, 2, 3, 10])
plt.plot([1, 2, 3, 10], [2, 4, 6, 10])

plt.plot([2, 4, 6, 10], [1, 2, 3, 10])

plt.plot([1, 2, 3, 10], [2, 4, 6, 10])
