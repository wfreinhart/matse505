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
# id: Lecture14_resnet
# type: Foundational
# parent_lecture: Lecture14
# ---
#
# ## ResNet
#
# ResNet (short for "Residual Network") is a family of convolutional neural network (CNN) architectures that was introduced in 2015. ResNet was designed to address the problem of vanishing gradients in deep neural networks, which can make it difficult to train models with many layers.
#
# The key innovation of ResNet is the use of **"skip connections"** that allow information to flow more easily between layers. In a traditional CNN, each layer takes the output of the previous layer as its input. In ResNet, some layers take the output of an earlier layer as their input, effectively "skipping over" one or more layers. This allows information to flow directly from the input to the output of the network, bypassing intermediate layers and making it easier to train deep models.
#
# <img src="../lectures/assets/lecture14_skip_connection.jpg" alt="A skip connection in a Residual Network block" width=600>
#
# ResNet comes in several variations, including ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152. The number in each name corresponds to the number of layers in the network. All ResNet models share a similar structure, with several blocks of convolutional layers followed by a global average pooling layer and a fully connected layer. Each block consists of several convolutional layers with skip connections, and the last block is followed by a global average pooling layer and a fully connected layer that produces the final classification output.
#
# One notable feature of ResNet is its ability to achieve state-of-the-art performance on a variety of computer vision tasks with relatively few parameters, making it more efficient to train and deploy than other models. ResNet has been used for image classification, object detection, semantic segmentation, and other computer vision tasks, and has achieved top performance on several benchmark datasets.
#
# > [!NOTE]
# > The ResNet-18 architecture diagram from ResearchGate is currently unavailable due to access restrictions.
#
# We will specifically utilize ResNet18 for expediency in our in-class examples.
# ResNet18 is a convolutional neural network (CNN) architecture that was introduced in 2015 as part of the ResNet family of models. Like other ResNet models, ResNet18 is designed to address the problem of vanishing gradients in deep neural networks by using skip connections that allow information to flow more easily between layers.
#
# ResNet18 consists of 18 layers, including 16 convolutional layers and 2 fully connected layers. The first layer is a convolutional layer that takes as input a 224x224 RGB image, followed by a max pooling layer that reduces the spatial dimensions of the output. The remaining layers are grouped into four blocks, each consisting of several convolutional layers with skip connections. The last block is followed by a global average pooling layer that averages the output of each feature map across its spatial dimensions, and a fully connected layer that produces the final classification output.
#
# Let's try loading the ResNet18 model from `torchvision.models`:

# %%
from torchvision.models import resnet18

model = resnet18(weights="IMAGENET1K_V1")
model.eval()  # set to evaluation mode (freeze the trainable weights)
