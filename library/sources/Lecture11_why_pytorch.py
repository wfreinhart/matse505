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
# id: Lecture11_why_pytorch
# type: Foundational
# parent_lecture: Lecture11
# ---
#
# ## Why `pytorch`?
#
# We are going to spend a significant amount of time learning to use the `pytorch` framework.
# At the moment it may seem like `sklearn` has everything we need and the additional overhead to use `pytorch` is not worthwhile, but there is simply no way to implement deep learning in `sklearn` or `numpy`.
# In any of these high-level packages, the convenience of having many common models and functions already implemented for you is the very reason why they're limiting.
# We now need lower-level access to the machinery in order to build our own custom models.
#
# The basis of any deep learning package (e.g., `pytorch`, `tensorflow`, or `jax`) is an **automatic differentiation** engine.
# Automatic differentiation is a tool for computing gradients on mathematical operators.
# Here's what it looks like:
#
# <img src="../lectures/assets/lecture11_full_graph.jpg" alt="Computational graph showing operations and gradients" width=600>
#
# Why do we need this?
# Gradient descent is the most practical way to solve high-dimensional optimization problems.
# In this case, the optimization problem is the selection of model parameters (i.e., NN weights) to minimize the model loss function.
# There can easily be millions of parameters to optimize, and the largest models have billions.
# Here's what gradient descent looks like in a 2D parameter space:
#
# <img src="../lectures/assets/lecture11_gradient_descent_2d.jpg" alt="Gradient descent optimization in a 2D parameter space" width=600>
#
# `pytorch` is not unique in its ability to solve these problems.
# However, it has a user-friendly interface and is relatively popular in the community.
# This means there are many well developed features and sample codes for implementing modern ML architectures.
# For this reason, I personally prefer `pytorch` over `tensorflow`, which is a little less polished.
# On the other hand, `keras` is a wrapper built on top of `tensorflow` that abstracts away too many features for my use cases.
# This leaves `pytorch` as a happy medium for me, but you may find a different story in your research.
