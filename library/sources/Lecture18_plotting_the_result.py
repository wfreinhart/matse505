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
# id: Lecture18_plotting_the_result
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Plotting the result
#
# > Note that I could also choose a similar looking color from [this list](https://matplotlib.org/stable/gallery/color/named_colors.html) instead of defining my own

# %%
from matplotlib import pyplot as plt
import numpy as np

xs, ys = data.columns

light_blue = np.array([26, 110, 177]) / 255  # RGB value from WPD Color Picker

fig, ax = plt.subplots()
ax.plot(data[xs], data[ys], '.', color=light_blue)
ax.set_xlabel(xs)
ax.set_ylabel(ys)
ax.set_title('Compression of architected material')

print(xs)
print(ys)
