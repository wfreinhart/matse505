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
# id: Lecture13_motivation
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Motivation
#
# Let me show you a feature of Fully Connected Neural Networks that may not be obvious but has very important consequences.
#
# We can train a simple MLP Regressor using `scikit-learn` on the usual set of features:
#
# Now we can try permuting the indices of the features randomly:
#
# And retraining the model...
#
# There should be no significant difference in the performance after this permutation since the fully connected layers take information from every input.
# We can repeat this several times to get a sense for how much this permutation matters.
#
# And compare it to the effect of changing the random seed:
#
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
# In contrast, convolutional layers use a set of learnable filters to perform local operations on the input image, which enables them to capture local patterns and structure in the image. The filters are typically small and applied in a sliding window manner across the input image, which allows them to detect patterns regardless of their position in the image. By sharing the filters across the entire image, convolutional layers are also able to reduce the number of parameters in the model and improve computational efficiency.
#
# Additionally, convolutional layers are often combined with pooling layers to downsample the output feature maps and reduce the spatial dimensionality of the data. This further reduces the number of parameters and helps to prevent overfitting, while still preserving important spatial features in the input image.

# %%
import pandas as pd
import numpy as np
import os

# Set the path to the data file
filename = 'concrete.csv'
local_path = f'../datasets/{filename}'
github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    data = pd.read_csv(github_url)
data                            # show a view of the data file

from sklearn import neural_network

x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

model = neural_network.MLPRegressor(max_iter=400, random_state=0).fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )

indices = np.arange(0, data.shape[1]-1)
print('before permutation: ', indices)
np.random.shuffle(indices)
print('after permutation: ', indices)

x = data.iloc[:, indices]
y = data.iloc[:, -1]

model = neural_network.MLPRegressor(max_iter=400, random_state=0).fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )

r2 = []
for _ in range(5):
    np.random.shuffle(indices)
    x = data.iloc[:, indices]
    y = data.iloc[:, -1]

    model = neural_network.MLPRegressor(max_iter=400, random_state=0).fit(x, y)
    print( f'R2   = {model.score(x, y):.3f}' )
    r2.append(model.score(x, y))

print(np.mean(r2), np.std(r2))

r2 = []
for i in range(5):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = neural_network.MLPRegressor(max_iter=400, random_state=i).fit(x, y)
    print( f'R2   = {model.score(x, y):.3f}' )
    r2.append(model.score(x, y))

print(np.mean(r2), np.std(r2))
