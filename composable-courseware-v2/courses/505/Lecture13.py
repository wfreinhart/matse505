# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's topics:
# * Convolutional Neural Networks
# * Building a CNN
# * Example: classification of steel defects

# %% [markdown]
# # Convolutional Neural Networks

# %% [markdown]
# ## Motivation
#
# Let me show you a feature of Fully Connected Neural Networks that may not be obvious but has very important consequences.

# %% include: dataset_concrete

# %% [markdown]
# We can train a simple MLP Regressor using `scikit-learn` on the usual set of features:

# %%
from sklearn import neural_network

x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

model = neural_network.MLPRegressor(max_iter=400, random_state=0).fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )

# %% [markdown]
# Now we can try permuting the indices of the features randomly:

# %%
indices = np.arange(0, data.shape[1]-1)
print('before permutation: ', indices)
np.random.shuffle(indices)
print('after permutation: ', indices)

# %% [markdown]
# And retraining the model...

# %%
x = data.iloc[:, indices]
y = data.iloc[:, -1]

model = neural_network.MLPRegressor(max_iter=400, random_state=0).fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )

# %% [markdown]
# There should be no significant difference in the performance after this permutation since the fully connected layers take information from every input.
# We can repeat this several times to get a sense for how much this permutation matters.

# %%
r2 = []
for _ in range(5):
    np.random.shuffle(indices)
    x = data.iloc[:, indices]
    y = data.iloc[:, -1]

    model = neural_network.MLPRegressor(max_iter=400, random_state=0).fit(x, y)
    print( f'R2   = {model.score(x, y):.3f}' )
    r2.append(model.score(x, y))

print(np.mean(r2), np.std(r2))

# %% [markdown]
# And compare it to the effect of changing the random seed:

# %%
r2 = []
for i in range(5):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = neural_network.MLPRegressor(max_iter=400, random_state=i).fit(x, y)
    print( f'R2   = {model.score(x, y):.3f}' )
    r2.append(model.score(x, y))

print(np.mean(r2), np.std(r2))

# %% [markdown]
# We should see that these bands overlap, indicating that the effect of permuting he features is insignificant for the model performance.
#
# *So what's the point of this exercise?*
#
# Fully connected networks do not take into account the spatial structure of images.
# They treat each pixel in the image as an independent feature, which can lead to a large number of parameters and overfitting, and makes it difficult to capture the local structure of images, such as edges and corners.
#
# Here is an example of an image and some permutations of it:
#
# <img src="../lectures/assets/lecture13_image_permutations.jpg" alt="An image and some permutations of it" width=600>
#
# *Would you consider these to be the same data artifact?*
#
#

# %% [markdown]
# In contrast, convolutional layers use a set of learnable filters to perform local operations on the input image, which enables them to capture local patterns and structure in the image. The filters are typically small and applied in a sliding window manner across the input image, which allows them to detect patterns regardless of their position in the image. By sharing the filters across the entire image, convolutional layers are also able to reduce the number of parameters in the model and improve computational efficiency.
#
# Additionally, convolutional layers are often combined with pooling layers to downsample the output feature maps and reduce the spatial dimensionality of the data. This further reduces the number of parameters and helps to prevent overfitting, while still preserving important spatial features in the input image.

# %% [markdown]
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

# %% [markdown]
# Here are some schematics to hep you visualize the process of convolution with filters:
#
# <img src="../lectures/assets/lecture13_convolution_filters.jpg" alt="Schematics of convolution with filters" width=600>

# %% [markdown]
# Here are some examples of linear image filters applied to a sample image:
#
# <img src="../lectures/assets/lecture13_filter_examples.jpg" alt="Examples of linear image filters applied to a sample image" width=600>
#

# %% [markdown]
# # Building a CNN

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
# ## Padding
#
# Padding is a technique used to control the size of the output feature maps produced by the convolutional layers of the network. Specifically, padding involves adding additional pixels around the edges of the input image before applying the convolution operation, which results in an output feature map with the same spatial dimensions as the input.
#
# Padding is important in CNNs for several reasons. First, it helps to preserve the spatial information of the input image by ensuring that the output feature maps have the same size as the input. Without padding, the size of the feature maps would gradually decrease as the input image is processed through the network, which could lead to loss of important spatial information.
#
# Second, padding can help to prevent the "border effect" that can occur when applying convolutional filters to the edges of an input image. This effect can occur when the size of the filter kernel is larger than the size of the input image, and the filter is unable to apply to some pixels at the edges of the image. By adding padding to the input image, the edges are effectively extended, allowing the filter to apply to all pixels in the input.
#
# There are two types of padding that are commonly used in CNNs: "valid" padding and "same" padding. Valid padding refers to the absence of padding, where the filter is only applied to the pixels that are fully contained within the input image. This results in an output feature map that is smaller than the input. Same padding, on the other hand, adds padding to the input such that the output feature map has the same spatial dimensions as the input.
#
# The amount of padding to add to the input image can be controlled by specifying the size of the padding along each dimension. For example, if we want to add 1 pixel of padding to a 2D input image with dimensions 28 x 28, we would use a padding size of 1 along each dimension, resulting in an input image with dimensions 30 x 30.
#
# <img src="../lectures/assets/lecture13_padding_same_full.jpg" alt="Illustration of same and full padding in convolution" width=600>

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
# While max pooling and average pooling are common techniques used in Convolutional Neural Networks (CNNs) to downsample feature maps and improve computational efficiency, there are some potential pitfalls to be aware of.
#
# One potential issue with max pooling is that it may discard useful information from the input image. By taking only the maximum value within a region of the feature map, max pooling may miss other important features in the region that are not the maximum. This can lead to a loss of detail in the output feature map, which may reduce the performance of the network on certain tasks. Additionally, if the pool size is too large or the stride is too high, the network may lose too much spatial information, which can make it difficult to distinguish between similar objects.
#
# On the other hand, average pooling can smooth out the details in the feature maps and blur out the edges of objects, which can make it harder for the network to distinguish between similar objects. This can reduce the accuracy of the network on certain tasks that require fine-grained detail, such as object detection.
#
# <img src="../lectures/assets/lecture13_avgpool_sample.jpg" width=600>

# %% [markdown]
# ## Regularization
#
# **Dropout:** Dropout is a regularization technique used to prevent overfitting in neural networks.
# It randomly drops out (sets to zero) some neurons during training, which forces the network to learn more robust and generalizable features.
#
# **Batch Normalization:** Batch normalization is a technique used to normalize the activations of each layer, which can help stabilize and speed up the training process.
# It normalizes the activations across the batch dimension, which reduces the internal covariate shift and allows the network to learn more efficiently.
#
# While techniques like batch normalization and weight decay have gained popularity, dropout remains a simple and effective technique to prevent overfitting.
#
# However, the use of dropout may not always be necessary or effective in certain scenarios.
# For example, if you have a small dataset or a shallow network, the use of dropout may not be as beneficial. It's also important to note that the effectiveness of dropout can depend on the specific hyperparameters used, such as the dropout rate and the network architecture.

# %% [markdown]
# # Classification of defects

# %% [markdown]
# ## About the data
#
# The [NEU Surface Defect Database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm) is a publicly available dataset of images of various surfaces with different types of defects. It was created by the researchers at Northeastern University, China. The database was released in 2018 and has been widely used in machine learning research.
#
# The database contains 1,800 grayscale images, each of size 200x200 pixels. The images show six types of surface defects on steel plates:
#
# * Rolled-in Scale (RS)
# * Scratches (Scr)
# * Pitted Surface (Pitted)
# * Rolled-in Dirt (RD)
# * Inclusion (In)
# * Crazing (Cr)
#
# Each type of defect has 300 images. The images are labeled with their corresponding defect type, and the labels are provided in a separate file.
#
# The NEU Surface Defect Database is useful for developing and testing image processing and machine learning algorithms for defect detection and classification. It can be used for tasks such as defect detection, classification, segmentation, and recognition.

# %% [markdown]
# You need to download the `zip` file first:

# %%
import zipfile, requests

url = 'https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/wfr5091_psu_edu/ERXYsfbOP4dGm7_M4oIh-0gBV3Ix19fKuSndDu4Ui6zHrQ?e=HOCHNN&download=1'
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open('data.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# %% [markdown]
# Then extract it to individual files with `zipfile`:

# %%
zip_file = zipfile.ZipFile('data.zip')
zip_file.extractall('/content/')
zip_file.close()

# %% [markdown]
# We can load an image using `PIL`:

# %%
from PIL import Image

filepath = 'NEU-DET-SP/train/scratches/scratches_1.jpg'
Image.open(filepath)

# %%
filepath = 'NEU-DET-SP/train/patches/patches_1.jpg'
Image.open(filepath)

# %% [markdown]
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

# %%
import torchvision.datasets as datasets

dataset = datasets.ImageFolder('NEU-DET-SP/train')

print(f'Number of images: {len(dataset)}')
print(f'Number of classes: {len(dataset.classes)}')

# %% [markdown]
# Now we can try to send the `Dataset` to a `DataLoader` as before:

# %%
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in dataloader:
    print(batch)
    break

# %% [markdown]
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

# %%
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

# %% [markdown]
# Now we are ready to implement and train a model!

# %% [markdown]
# ## Basic implementation
#
# We'll start by installing `pytorch-lightning` to simplify the training procedure.

# %%
!pip install pytorch-lightning

# %% [markdown]
# Remember that the basic features of a CNN are the convolutions, activations, and pooling, with fully connected layers at the end.
# We will design a `ConvBlock` object to avoid repeating the first three several times:

# %%
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ClassifierCNN(pl.LightningModule):
    def __init__(self, conv_channels, fc_dim, num_classes):
        super(ClassifierCNN, self).__init__()

        conv_blocks = []
        for i, c in enumerate(conv_channels):
            conv_blocks.append( ConvBlock(c) )

        self.conv = nn.Sequential(*conv_blocks)
        self.fc = nn.Sequential(nn.LazyLinear(fc_dim),
                                nn.LeakyReLU(),
                                nn.LazyLinear(num_classes))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)  # flatten the output for FC layer
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# %% [markdown]
# ## [Check your understanding]
#
# Try training the model with `pytorch_lightning`.

# %%
import pytorch_lightning as pl

model = ClassifierCNN([8, 8], 32, 2)
dl_train = DataLoader(dataset, batch_size=32, shuffle=True)

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model=model, train_dataloaders=dl_train)

# %% [markdown]
# ## Evaluating the model
#
# Let's evaluate the model predictions on the training set.

# %%
import torch
import numpy as np

y_outs = []
y_true = []

with torch.no_grad():
    for x, y in dataloader:
        y_outs += model(x).detach().numpy().tolist()
        y_true += y.detach().numpy().tolist()

y_outs = np.array(y_outs)
y_true = np.array(y_true)

# %% [markdown]
# We can start by investigating this `y_prob` result, the raw model output:

# %%
from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
_ = ax.hist(y_outs[y_true==0], density=True)
ax.set_xlabel('Prediction')
ax.set_ylabel('Probability density')
ax.set_title('True Patches')
ax = axes[1]
_ = ax.hist(y_outs[y_true==1], density=True)
ax.set_xlabel('Prediction')
ax.set_ylabel('Probability density')
ax.set_title('True Scratches')

# %% [markdown]
# We can convert these raw outputs to class probabilities using the softmax function.
# From Wikipedia:
#
# The softmax function takes as input a vector $z$ of $K$ real numbers, and normalizes it into a probability distribution consisting of $K$ probabilities proportional to the exponentials of the input numbers. That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval (0,1), and the components will add up to 1, so that they can be interpreted as probabilities.
#
# $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$

# %%
y_prob = nn.functional.softmax(torch.tensor(y_outs), dim=1).detach().numpy()

fig, ax = plt.subplots()
_ = ax.hist(y_prob[y_true==0][:, 0])
_ = ax.hist(y_prob[y_true==1][:, 1])

# %% [markdown]
# These prediction probabilities can be converted to integer class labels using the `torch.max` function to identify the most likely class:

# %%
y_pred = []
y_true = []

with torch.no_grad():
    for x, y in dataloader:
        outputs = model(x)
        _, label = torch.max(outputs.data, 1)
        y_pred += label.detach().numpy().tolist()
        y_true += y.detach().numpy().tolist()

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# %% [markdown]
# Let's try to turn this into a confusion matrix.

# %%
confusion = np.zeros([2, 2], dtype=int)
for i in range(len(y_pred)):
    row_idx = y_pred[i].round()
    col_idx = y_true[i].round()
    confusion[row_idx.astype(int), col_idx.astype(int)] += 1

fig, ax = plt.subplots()
im = ax.imshow(confusion, 'Blues')
cb = plt.colorbar(im)
cb.set_label('Frequency')
_ = ax.set_xlabel('True label')
_ = ax.set_ylabel('Predicted label')

for i in range(2):
    for j in range(2):
        if confusion[i, j] > 100:
            tc = 'w'
        else:
            tc = 'k'
        ax.text(i, j, confusion[i, j], ha='center', color=tc)

# %% [markdown]
# ## [Check your understanding]
#
# Create a confusion matrix for the validation set.

# %%
# create the validation dataloader
val_dataset = datasets.ImageFolder('NEU-DET-SP/validation', transform=transform)
dl_val = DataLoader(val_dataset, batch_size=32, shuffle=False)

y_pred = []
y_true = []

with torch.no_grad():
    for x, y in dl_val:  # modify this to the correct dataloader
        outputs = model(x)
        _, label = torch.max(outputs.data, 1)
        y_pred += label.detach().numpy().tolist()
        y_true += y.detach().numpy().tolist()

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# %%
confusion = np.zeros([2, 2], dtype=int)
for i in range(len(y_pred)):
    row_idx = y_pred[i].round()
    col_idx = y_true[i].round()
    confusion[row_idx.astype(int), col_idx.astype(int)] += 1

fig, ax = plt.subplots()
im = ax.imshow(confusion, 'Blues')
cb = plt.colorbar(im)
cb.set_label('Frequency')
_ = ax.set_xlabel('True label')
_ = ax.set_ylabel('Predicted label')

for i in range(2):
    for j in range(2):
        if confusion[i, j] > 50:
            tc = 'w'
        else:
            tc = 'k'
        ax.text(i, j, confusion[i, j], ha='center', color=tc)

# %% [markdown]
# ## Batch normalization
#
# Batch normalization is a technique that helps to stabilize the training process of neural networks by normalizing the activations of each layer.
# It works by normalizing the activations across the batch dimension, which reduces the internal covariate shift and allows the network to learn more efficiently.
#
# <img src="../lectures/assets/lecture13_batch_norm_1.jpg" alt="Schematic showing the Batch Normalization process" width=600>
#
# > [!NOTE]
# > A second Batch Normalization diagram is currently unavailable due to access restrictions.
#
# The implementation in PyTorch is called `nn.BatchNorm2d`:

# %%
class ConvBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# note: no need to redefine the ClassifierCNN model!

# %% [markdown]
# In the forward method, we apply batch normalization after each convolutional layer and fully connected layer.
# Note that batch normalization should be applied before the activation function.
#
# Adding batch normalization can improve the convergence speed and generalization performance of the CNN.
# It reduces the internal covariate shift, which helps to stabilize the training process and allows the network to learn more efficiently.
# Additionally, it can act as a regularization technique and prevent overfitting.
#
# Let's try training the model again with batch normalization in place:

# %%
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ClassifierCNN([8, 8], 32, 2)

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model=model, train_dataloaders=dataloader)

# %% [markdown]
# We see right away that the loss is significantly lower than before
# Let's make the confusion matrix:

# %%
y_pred = []
y_true = []

with torch.no_grad():
    for x, y in dataloader:
        outputs = model(x)
        _, label = torch.max(outputs.data, 1)
        y_pred += label.detach().numpy().tolist()
        y_true += y.detach().numpy().tolist()

y_pred = np.array(y_pred)
y_true = np.array(y_true)

confusion = np.zeros([2, 2], dtype=int)
for i in range(len(y_pred)):
    row_idx = y_pred[i].round()
    col_idx = y_true[i].round()
    confusion[row_idx.astype(int), col_idx.astype(int)] += 1

fig, ax = plt.subplots()
im = ax.imshow(confusion, 'Blues')
cb = plt.colorbar(im)
cb.set_label('Frequency')
_ = ax.set_xlabel('True label')
_ = ax.set_ylabel('Predicted label')

for i in range(2):
    for j in range(2):
        if confusion[i, j] > 100:
            tc = 'w'
        else:
            tc = 'k'
        ax.text(i, j, confusion[i, j], ha='center', color=tc)

# %% [markdown]
# ## [Check your understanding]
#
# Try manually tuning the hyperparameters of the CNN model and compare the performance on train and validation sets.
# Which parameters make the greatest difference?

# %%

