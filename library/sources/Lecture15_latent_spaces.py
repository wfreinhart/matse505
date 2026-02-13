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
# id: Lecture15_latent_spaces
# type: Foundational
# parent_lecture: Lecture15
# ---
#
# # Latent spaces
#
# In machine learning and deep learning, a latent space is a compressed and abstract representation of the input data that is learned by a neural network during training. The term "latent" refers to the fact that this representation is not directly observable or measurable, but is inferred or deduced from the input data.
#
# The process of learning a latent space involves training a neural network, such as an autoencoder, to encode the input data into a lower-dimensional representation, and then decode this representation back into the original input data. The compressed representation that is learned by the network is often referred to as the encoding or latent space.
#
# The latent space is typically much smaller than the input space, which means that it represents a highly compressed and abstract version of the input data. However, the latent space is also designed to preserve important information about the input data, such as its structure, patterns, and relationships.
#
# The latent space can be thought of as a way to represent the essential features or attributes of the input data in a compact and efficient way. By using a latent space, it is possible to perform tasks such as data compression, data denoising, and data augmentation, as well as generative modeling and unsupervised learning.
