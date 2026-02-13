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
# id: Lecture13_image_filters
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Image filters
#
# The basis for a CNN is the concept of **linear image filters**.
#
# Linear image filters are a fundamental concept in digital image processing and computer vision.
# They refer to a class of image filters that operate on an image by computing a weighted sum of the pixel values in a local neighborhood around each pixel.
# The weights used in the sum are determined by a filter kernel, which is a small matrix that specifies the weights for each pixel in the local neighborhood.
#
# The process of applying a linear image filter to an image involves sliding the filter kernel over the entire image, and computing the weighted sum of the pixel values in the local neighborhood for each pixel.
# The resulting output image is a filtered version of the input image, with each pixel value representing a weighted combination of the surrounding pixels.
#
# Linear image filters can be used for a variety of image processing tasks, such as smoothing (also known as blurring), sharpening, edge detection, and noise reduction. Different types of filters can be designed to emphasize different image features or suppress noise.
#
# Some commonly used types of linear filters include:
#
# * Gaussian filter: A filter that applies a Gaussian blur to the image, smoothing out high-frequency noise and details while preserving the overall shape of objects in the image.
# * Sobel filter: A filter that computes the gradient of the image in the x and y directions, which can be used to detect edges in the image.
# * Laplacian filter: A filter that computes the second derivative of the image, which can be used to enhance edges and details in the image.
# * Median filter: A filter that replaces each pixel value with the median value of the pixel values in its local neighborhood, which can be used to remove salt-and-pepper noise from the image.
# * Linear image filters can be implemented using convolutional neural networks (CNNs) in deep learning frameworks such as PyTorch, allowing them to be learned from data and applied to a wide range of computer vision tasks.
#
# Here are some schematics to hep you visualize the process of convolution with filters:
#
# <img src="../lectures/assets/lecture13_convolution_filters.jpg" alt="Schematics of convolution with filters" width=600>
#
# Here are some examples of linear image filters applied to a sample image:
#
# <img src="../lectures/assets/lecture13_filter_examples.jpg" alt="Examples of linear image filters applied to a sample image" width=600>
