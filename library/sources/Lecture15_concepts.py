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
# id: Lecture15_concepts
# type: Foundational
# parent_lecture: Lecture15
# ---
#
# ## Concepts
#
# An autoencoder is a type of neural network that is used for unsupervised learning, particularly in the domain of dimensionality reduction. The idea behind an autoencoder is to learn a compressed representation of some input data, and then use that compressed representation to generate a reconstruction of the original data. In essence, the network "encodes" the input data into a compressed representation, and then "decodes" that representation back into a reconstructed version of the original data.
#
# <img src="../lectures/assets/lecture15_autoencoder.jpg" alt="Schematic of an Autoencoder showing Encoder, Bottleneck, and Decoder" width=600>
#
# An autoencoder typically consists of two main parts: an **encoder** and a **decoder**. The encoder is a neural network that takes the input data and maps it to a lower-dimensional representation, while the decoder is a network that takes the lower-dimensional representation and maps it back to the original data space. The encoder and decoder are often symmetric, meaning that the encoder and decoder architectures are mirrored around the middle layer.
# *However, in practice the decoder may benefit from deeper networks.*
#
# <img src="../lectures/assets/lecture15_latent_space.jpg" alt="Visualization of a latent space with mapped data points" width=600>
#
# During training, an autoencoder tries to minimize the reconstruction error between the original input data and the reconstructed output data. This is typically done using a loss function like mean squared error or binary cross-entropy. By minimizing this reconstruction error, the autoencoder learns a compressed representation of the data that is able to capture the most important features of the data.
#
# Autoencoders have a wide range of applications, including in image and signal processing, feature extraction, anomaly detection, and data denoising. They can also be used for generative tasks, such as generating new data that is similar to the input data.
#
# <img src="../lectures/assets/lecture15_cvae_latent.jpg" alt="Latent space visualization of a Conditional VAE" width=600>
#
# How does this work?
# In short, there are not `28 x 28 = 784` unique pieces of information in these images.
#
# <img src="../lectures/assets/lecture15_mnist.jpg" alt="Samples from the MNIST dataset of handwritten digits" width=600>
