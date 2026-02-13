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
# id: Lecture13_architecture
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Architecture
#
# Here are the basic elements of a Convolutional Neural Network (CNN) architecture:
#
# 1. **Convolutional Layers:** The convolutional layer is the core building block of a CNN. It applies a set of learnable filters (also known as kernels or weights) to the input image to extract relevant features. Convolutional layers are usually followed by activation functions (such as ReLU) and pooling layers.
#
# 2. **Pooling Layers:** Pooling layers reduce the spatial dimensions (width and height) of the feature maps produced by the convolutional layers. They do this by downsampling the feature maps, which helps to reduce the computational complexity of the network and make it more efficient. The most common pooling operation is max pooling, which takes the maximum value within a pool of values.
#
# 3. **Activation Functions:** Activation functions introduce nonlinearity into the network and help to model complex relationships between inputs and outputs. The most commonly used activation functions in CNNs are ReLU (Rectified Linear Unit), which sets all negative values to zero, and its variants.
#
# 4. **Fully Connected Layers:** Fully connected layers are typically used at the end of a CNN architecture to map the extracted features to the output class labels. They are also called dense layers, and they have weights that are learned during training.
#
# <img src="../lectures/assets/lecture13_cnn_architecture.jpg" alt="Basic elements of a Convolutional Neural Network architecture" width=600>
#
# In a deep Convolutional Neural Network, the features learned by the filters at different levels of the network become increasingly complex and abstract as the input image is processed through the layers of the network.
#
# At the lowest level of the network, the filters are typically designed to detect simple image features such as edges, corners, and blobs of light or dark pixels. These features are represented as combinations of low-level image gradients, and they form the building blocks for more complex features that are learned in higher layers.
#
# As the image is processed through deeper layers of the network, the learned filters become more complex and specific to the task at hand. For example, in a CNN designed to classify images of animals, the filters in the middle layers might learn to detect shapes and textures that are common to different types of animals, such as fur, scales, or feathers. In the later layers, the filters might learn to detect more specific features that are characteristic of particular types of animals, such as the shape of a bird's beak or the pattern of stripes on a zebra.
#
# One interesting property of learned filters in a CNN is that they tend to be highly selective for specific image features, but also highly robust to variations in those features such as changes in lighting, scale, or orientation. This is because the filters are trained to look for statistical patterns in the image data that are predictive of the target label, and they learn to generalize these patterns to new images that have similar features.
#
# Another important feature of learned filters in a CNN is that they are often highly correlated with the spatial structure of the input image. That is, the filters tend to learn to detect features that are localized in specific regions of the image, and they are often arranged in a way that preserves the spatial structure of the input image. This allows the filters to capture the spatial relationships between different parts of the image, which is important for tasks such as object detection and segmentation.
#
# Here are example activation maps from different layers in a trained CNN:
#
# <img src="../lectures/assets/lecture13_activation_maps.jpg" alt="Example activation maps from different layers in a trained CNN" width=600>
