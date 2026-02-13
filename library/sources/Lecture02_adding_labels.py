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
# id: Lecture02_adding_labels
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Adding labels
#
# We can specify labels, titles, and other formatting elements by reading the [documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot) or following a [tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html):

# %%
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 5], [2, 4, 6, 10])
ax.set_title('An example pyplot chart')
ax.set_xlabel('x')
ax.set_ylabel('y')
