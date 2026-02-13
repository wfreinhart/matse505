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
# id: Lecture13_stride
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Stride
#
# The stride refers to the step size used to move the convolutional filter over the input image during the convolution operation. Specifically, the stride determines how many pixels the filter is shifted horizontally and vertically between each application.
#
# When the stride is set to a value greater than 1, the filter "skips" some pixels in the input image, resulting in a smaller output feature map. This can be useful for downsampling the input image and reducing the computational complexity of the network. For example, in the early layers of a CNN, larger strides may be used to capture simple features such as edges and corners, while downsampling the input image to reduce its size.
#
# Conversely, when the stride is set to 1, the filter is applied to every pixel in the input image, resulting in an output feature map with the same size as the input. This can be useful for capturing more fine-grained details in the input image, and is often used in later layers of the CNN to capture more complex features that are specific to the task at hand.
#
# The choice of stride depends on the specific requirements of the task at hand, as well as considerations such as computational complexity and the size of the input image. In general, larger strides can lead to faster processing and reduced computational complexity, but may result in loss of information and detail in the input image. Conversely, smaller strides can capture more fine-grained details in the input image, but may require more computational resources and processing time.
#
# <img src="../lectures/assets/lecture13_stride.jpg" alt="Illustration of stride in convolution" width=400>
