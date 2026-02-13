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
# id: Lecture16_concepts
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## Concepts
#
# Generative modeling is a type of machine learning approach in which a model is trained to learn the underlying distribution of a dataset, and then generate new data samples that are similar to the original data.
# It is called "generative" since we are "generating" new (synthetic) samples.
#
# To use an autoencoder for generative modeling, we first train the autoencoder on a dataset of input data.
# During training, the autoencoder learns to encode the input data into a compressed representation, and then decode the compressed representation back into the original input data.
# Once the autoencoder is trained, **we can sample from the learned encoding or latent space to generate new data samples.**
#
# There are a variety of ways to generate new data samples using a trained autoencoder:
# * Randomly sample from the learned encoding or latent space and then decode the samples to generate new data samples
#   * This requires the specification of a probability distribution to sample from. An arbitrary prior distribution such as uniform or normal can be used, or an empirical PDF based on the training data sample could be used.
# * Interpolate between two data samples in the latent space
#   * This can be done by encoding the two data samples of interest, taking a weighted average of the encodings, and then decoding the weighted average to generate a new data sample that is a combination of the two original data samples.
# * Sample from the latent space on a regular grid
