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
# id: Lecture01_importing_modules
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## Importing modules
#
#
# You can access modules using the `import` keyword followed by the module name.
# Colab has a bunch of modules available by default.
# Let's try importing NumPy:
#
# **Note** modules are supposed to have short, lower-case names.
# Numpy was imported using `numpy` rather than NumPy.
#
# The `print` function gives us some information about the module that indicates it loaded successfully.
# Note that just calling `import` doesn't give any output unless there's an error.
# Let's observe this behavior using Matplotlib:

# %%
import numpy
print(numpy)

import matplotlib
