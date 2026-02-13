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
# id: Lecture01_modules
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# # Modules
#
# *What are modules?*
#
# We've gone over some basic Python functionality, but so far it doesn't seem that powerful.
# We can:
# * define variables
# * perform basic math on them (multiply, divide, add, subtract)
# * convert them to different types
# * print their values
#
# This is less than your graphing calculator can do, so what's the big deal?
# Basically, the answer is **modules**.
# Python modules are pieces of code that other people have written and posted on the internet for free distribution.
# By accessing those modules, you can expand Python's functionality to do nearly anything you can think of.
#
# *What are some commonly used modules?*
#
# **NumPy**
#
# [`numpy`](https://numpy.org/) is so widespread it might as well be considered part of the Python language.
# Basically any program with mathematical equations or data manipulation uses NumPy.
# This includes the other modules we're about to look at!
#
# **Matplotlib**
#
# [`matplotlib`](https://matplotlib.org/) is one of the most popular visualization libraries for Python (although there are others).
# Its name comes from its inspiration from the MATLAB plotting library.
# If you've used MATLAB, this will look familiar.
#
# **SciPy**
#
# [`scipy`](https://docs.scipy.org/doc/scipy/reference/) is a more specialized package for scientific computing.
# It includes things like integration, optimization, and statistical distributions.
#
# **Pandas**
#
# [`pandas`](https://pandas.pydata.org/docs/) is a library for reading, writing, and analyzing data.
# It contains wrappers to do a lot of the things these other libraries do (by calling their functions behind the scenes).
# While we will use all four of these in class, you may end up mostly relying on Pandas "in the wild."
