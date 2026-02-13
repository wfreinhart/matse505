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
# id: Lecture02_subplots
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Subplots
#
# In order to access the elements of our chart again later, we need to store the `Figure` and/or `Axes` as a variable rather than letting `pyplot` do this with its state machine.
# We can do this using the [`pyplot.subplots`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) command:
#
# Now we have a persistent chart object that we can modify as we go.
# Let's try repeating the examples we've already seen, except this time by calling the same methods on `axes` instead of using the default `pyplot`:
#
# *Where's our figure?*
#
# Because the `Figure` is being stored persistently, `pyplot` doesn't show it to us every time we call a command on it.
# Instead, we can call our variable to show the figure again.

# %%
figure, axes = plt.subplots()
print(figure)
print(axes)

axes.plot([1, 2, 3, 5], [2, 4, 6, 10])

figure
