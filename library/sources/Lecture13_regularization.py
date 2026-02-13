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
# id: Lecture13_regularization
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Regularization
#
# **Dropout:** Dropout is a regularization technique used to prevent overfitting in neural networks.
# It randomly drops out (sets to zero) some neurons during training, which forces the network to learn more robust and generalizable features.
#
# **Batch Normalization:** Batch normalization is a technique used to normalize the activations of each layer, which can help stabilize and speed up the training process.
# It normalizes the activations across the batch dimension, which reduces the internal covariate shift and allows the network to learn more efficiently.
#
# While techniques like batch normalization and weight decay have gained popularity, dropout remains a simple and effective technique to prevent overfitting.
#
# However, the use of dropout may not always be necessary or effective in certain scenarios.
# For example, if you have a small dataset or a shallow network, the use of dropout may not be as beneficial. It's also important to note that the effectiveness of dropout can depend on the specific hyperparameters used, such as the dropout rate and the network architecture.
