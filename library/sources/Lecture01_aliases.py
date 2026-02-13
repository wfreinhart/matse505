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
# id: Lecture01_aliases
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## Aliases
#
# Even though modules have relatively short names, they sometimes need to be referenced a ton of times (hundreds or thousands).
# In this case, it can be handy to change the name to something even shorter.
# When using `import`, you can rename things with the [`as` keyword](https://www.w3schools.com/python/ref_keyword_as.asp):
#
# Now the very short `np` will serve as an alias for the NumPy module.
# You can think of this just like a variable name, where `np` points to the full NumPy module behind the scenes.

# %%
import numpy as np
print(np)
