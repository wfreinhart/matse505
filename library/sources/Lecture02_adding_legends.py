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
# id: Lecture02_adding_legends
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Adding legends
#
# Legends are essential for good charts (with multiple series).
# Here is my recommended way to add a legend using `ax.legend`:

# %%
fig, ax = plt.subplots()

ax.plot([1, 2, 3, 5], [2, 4, 6, 10], label='Series 1')  # label will be used by legend() later
ax.plot([1, 2, 3, 4], [3, 4, 5, 6], label='Series 2')

ax.legend()  # create the legend based on `label`
