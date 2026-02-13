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
# id: Lecture13_setting_up_the_dataset
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Setting up the `Dataset`
#
# `torchvision.datasets.ImageFolder` is a PyTorch dataset class that allows you to load a dataset of images from a directory on disk.
# It assumes that the directory structure is organized such that images of each class are in their own subdirectory.
#
# The `ImageFolder` class makes it easy to load and preprocess a large number of images by defining the following conventions:
# 1. Each subdirectory in the root directory is treated as a separate class. The name of each subdirectory is used as the class label.
# 2. Images in each subdirectory are treated as samples of the corresponding class.
# 3. Images are loaded using `PIL` and are transformed using the provided transform pipeline.
#
# You should think of this as a custom `Dataset` object -- you could achieve a similar result by programming your own `__getitem__` method in a `Dataset` subclass that loads images with `PIL` and transforms them to `tensor`.
#
# Now we can try to send the `Dataset` to a `DataLoader` as before:
#
# What happened?
# We got a `TypeError` that indicates we need `tensors` rather than `PIL.Image.Image` type objects.
# The solution is to use `torchvision.transforms`.
#
# `torchvision.transforms` is a module in the PyTorch library that provides a set of image transformations that can be applied to images or datasets. These transformations are useful for data augmentation, preprocessing, and normalization.
#
# The `torchvision.transforms` module provides several classes of transformations, including:
# * `transforms.Compose`: a class that allows you to chain several transforms together, so they can be applied sequentially to an image or dataset.
# * `transforms.Resize`: a class that resizes an image to a given size.
# * `transforms.ToTensor`: a class that converts an image to a PyTorch tensor.
# * `transforms.CenterCrop`: a class that crops the center of an image to a given size.
# *`transforms.RandomCrop`: a class that randomly crops an image to a given size.
# * `transforms.Normalize`: a class that normalizes an image tensor with given mean and standard deviation values.
#
# Now we are ready to implement and train a model!

# %%
import torchvision.datasets as datasets

dataset = datasets.ImageFolder('NEU-DET-SP/train')

print(f'Number of images: {len(dataset)}')
print(f'Number of classes: {len(dataset.classes)}')

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    print(batch)
    break

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(200),  # Resize the image to 200x200 pixels
    transforms.ToTensor()    # Convert the image to a PyTorch tensor
])

dataset = datasets.ImageFolder('NEU-DET-SP/train', transform=transform)
print(f'Number of images (all classes): {len(dataset)}')

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in dataloader:
    print(batch)
    break
