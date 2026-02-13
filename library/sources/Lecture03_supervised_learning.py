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
# id: Lecture03_supervised_learning
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Supervised learning
#
# Let's say we have a function $f(x) = y$. We have seen equations of this form many times before in math and engineering courses. Most often, we have acquired some data of the form $(x, y)_i$ and tried to find a form of $f$ that we feel explains the trends adequately.
#
# If we don't know the true generating function exactly, we use curve fitting to identify $f$. For instance, we have a vector of $x$ and a vector of $y$ and we write $f(x) = m x + b = y$. Here $f$ is a (univariate) linear regression between $x$ and $y$.
#
# Instead of saying the parameters of the function $f$ are **fit**, we can instead say the function $f$ is **learned**. There, now we are doing machine learning -- our Python programming is effectively learning a relationship between our dependent ($y$) and independent ($X$) variables.
# In ML, we call $X$ the **features** (uppercase indicates a 2D array instead of a 1D vector, each row of $X$ is called a feature vector) and $y$ the **labels**.
