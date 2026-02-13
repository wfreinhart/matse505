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
# id: Lecture12_batches_and_data_i_o
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# # Batches and data I/O
#
# When training a neural network, it's often not feasible to pass the entire dataset through the network at once.
# This is because the dataset might be too large to fit into memory, or because passing the entire dataset through the network in one go might be computationally expensive.
#
# To overcome these issues, we can break the dataset into smaller **batches** and feed one batch at a time to the network.
# This is known as mini-batch training, and it has several benefits:
#
# 1. Memory efficiency: By processing data in small batches, we can work with larger datasets that we wouldn't be able to fit into memory otherwise.
#
# 2. Computation efficiency: Processing one batch at a time is often faster than processing the entire dataset at once, especially if we use a GPU to perform the computations.
#
# 3. Better generalization: Mini-batch training can help prevent overfitting by introducing noise into the optimization process, which can help the network generalize better to new data.
#
# To implement mini-batch training in PyTorch, we can use the `Dataset` and `DataLoader` classes.
