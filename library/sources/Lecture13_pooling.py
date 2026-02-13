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
# id: Lecture13_pooling
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Pooling
#
# Pooling is a technique used to downsample the output feature maps of a convolutional layer. Pooling is typically applied after a convolutional layer to reduce the spatial dimensionality of the output feature maps and decrease the number of parameters in the model.
#
# The most common type of pooling is max pooling, which involves dividing the output feature map into non-overlapping rectangular regions and taking the maximum value of each region as the output. For example, a 2x2 max pooling operation will divide the output feature map into non-overlapping 2x2 regions and output the maximum value within each region.
#
# Max pooling has several benefits in a CNN. Firstly, it reduces the spatial dimensionality of the output feature maps, which can help to decrease the computational complexity of the network and prevent overfitting. Secondly, max pooling helps to capture the most important features of the input image, which can improve the performance of the network on the target task. Finally, max pooling can also increase the invariance of the network to small translations and rotations of the input image.
#
# Other types of pooling include average pooling, which computes the average value of each region instead of the maximum, and L2 pooling, which computes the square root of the sum of squares of each region instead of the maximum.
#
# <img src="../lectures/assets/lecture13_maxpool_sample.jpg" width=600>
#
# While max pooling and average pooling are common techniques used in Convolutional Neural Networks (CNNs) to downsample feature maps and improve computational efficiency, there are some potential pitfalls to be aware of.
#
# One potential issue with max pooling is that it may discard useful information from the input image. By taking only the maximum value within a region of the feature map, max pooling may miss other important features in the region that are not the maximum. This can lead to a loss of detail in the output feature map, which may reduce the performance of the network on certain tasks. Additionally, if the pool size is too large or the stride is too high, the network may lose too much spatial information, which can make it difficult to distinguish between similar objects.
#
# On the other hand, average pooling can smooth out the details in the feature maps and blur out the edges of objects, which can make it harder for the network to distinguish between similar objects. This can reduce the accuracy of the network on certain tasks that require fine-grained detail, such as object detection.
#
# <img src="../lectures/assets/lecture13_avgpool_sample.jpg" width=600>
