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
# id: Lecture17_convolutional_transpose_layer
# type: Foundational
# parent_lecture: Lecture17
# ---
#
# ## Convolutional Transpose Layer
#
# In order to create an autoencoder for convolutional networks, we need to be able to "de-compress" or "de-convolve" an image.
#
# > [!NOTE]
# > The transposed convolution visualization from ResearchGate is currently unavailable due to access restrictions.
#
# A **convolutional transpose layer** is a type of layer in a convolutional neural network that is used for upsampling or increasing the resolution of an input image or feature map.
# It is also sometimes referred to as a "deconvolutional layer," but this is not accurate since the convolutional transpose is *not an inverse of the convolution*.
#
# The main hyperparameters of a convolutional transpose layer include:
#
# * Output channels: The number of channels in the output feature map. This is equivalent to the number of filters used in a standard convolutional layer.
#
# * Kernel size: The size of the convolutional filter that is applied during the transpose operation.
#
# * Stride: The amount of pixels by which the convolutional filter is moved during the transpose operation.
#
# * Padding: The amount of padding that is added to the input image or feature map to ensure that the output size matches the desired size.
#
# * Output padding: The number of pixels to concatenate to the last row and column of the output image.
#
# In convolution, padding and stride are used to control the size of the output feature map. In convolutional transpose, padding and stride are used to control the size of the output feature map as well as the amount of upsampling that is applied to the input.
