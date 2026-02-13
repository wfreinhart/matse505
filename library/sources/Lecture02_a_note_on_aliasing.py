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
# id: Lecture02_a_note_on_aliasing
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## A note on aliasing
#
# Before we go any further, let's talk about `aliases` again.
# As you can see from our very first code cell, accessing the functions of the `pyplot` submodule will require us to type `pyplot.<function>` every time.
# In practice, people have all agreed that it's easier to type `plt` instead:
#
# Now we can always use `plt` instead of `pyplot`.
# I bring this up because all the tutorials I link to use this notation!
# You can even see it being done this way on the [official pyplot documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot)!

# %%
from matplotlib import pyplot as plt  # use an alias

plt.plot()  # create an empty plot
