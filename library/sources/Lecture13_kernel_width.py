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
# id: Lecture13_kernel_width
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Kernel width
#
# The kernel width refers to the size of the filter that is applied to the input image during the convolution operation. Specifically, the kernel width determines the number of pixels in each row and column of the filter.
#
# The kernel width is a hyperparameter of the CNN that can be tuned to improve the performance of the network for a given task. In general, larger kernel widths can capture more complex features in the input image, but can also lead to a larger number of parameters in the model and increased computational complexity. Conversely, smaller kernel widths can be more computationally efficient, but may not be able to capture as much detail in the input image.
#
# The choice of kernel width also depends on the size of the input image and the depth of the CNN. For example, in the initial layers of the network, smaller kernel widths may be used to capture simple features such as edges and corners. As the image is processed through deeper layers of the network, larger kernel widths may be used to capture more complex features that are specific to the task at hand.
#
# Another factor to consider when choosing the kernel width is the trade-off between local and global information. A small kernel width captures local information in the input image, while a larger kernel width captures more global information. For some tasks, such as image segmentation, it may be important to capture both local and global information, and a combination of different kernel widths may be used.
#
# <img src="../lectures/assets/lecture13_kernel_size.jpg" alt="Illustration of different kernel widths in convolution" width=600>
