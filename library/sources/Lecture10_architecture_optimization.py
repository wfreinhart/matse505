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
# id: Lecture10_architecture_optimization
# type: Foundational
# parent_lecture: Lecture10
# ---
#
# ## Architecture optimization
#
# Deep learning models may have very complex architecture with many hyperparameters to choose:
#
# <img src="../lectures/assets/neural_network_architecture.jpg" width=600 alt="Schematic diagram of a multi-layer perceptron neural network with input, hidden, and output layers">
#
# In this case, we need to be clever in how to encode the many possible options.
# Here are some common shapes for NNs:
#
# <img src="../lectures/assets/neural_network_zoo.jpg" width=400 alt="Graphic showing various neural network architecture types beyond simple feed-forward networks">
#
# You will see that NNs do not typically have wildly oscillating sizes between layers.
# Instead, they vary smoothly and the typical shapes are flat or trapezoidal.
# This means we can reduce the number of parameters from choosing every number of neurons independently to only choosing the "shape" of the network.
