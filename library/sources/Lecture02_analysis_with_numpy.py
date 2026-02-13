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
# id: Lecture02_analysis_with_numpy
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# # Analysis with `numpy`
#
# NumPy is a popular library for scientific computing that contains fast and convenient functions for linear algebra.
# Let's start by importing the [`numpy` module](https://numpy.org/doc/stable/).
# This will give access to all the features the `numpy` authors have implemented for us.
# Let's use the `as` keyword to keep the module name short again.
# Trust me, this will be worth it -- we are going to reference `numpy` in almost every line!

# %%
import numpy as np
