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
# id: Lecture08_one_hot_encoding
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## One-hot encoding
#
# Let's take a look at the next categorical column, `Natural Crystal Structure`:
#
# This presents a problem because there are many labels.
# In principle, we can convert them to `int` codes like so:
#
# However, this will be a problem when we use these features in a prediction task because it assumes *ordinality* (recall our discussion on classification labels).
# We can investigate this problem below:
#
# What you can see is that some pairs of crystal structures have large distances while others have small distances.
# Here's another way to visualize this problem:
#
# Imagine using this space for a k-Neighbors regression problem.
# The BCC and CUB phases would be grouped together while being far away from TET.
# It is possible that there is some structure to this space, but we haven't provided it here.
#
# The solution is to use *one-hot encoding*:
#
# <img src="../lectures/assets/one_hot_encoding_diagram.jpg" width=600 alt="Diagram showing the transformation of a categorical 'Color' feature into multiple binary 'One-Hot' feature vectors">
#
# In this scheme, each category gets its own "dummy feature" which is a binary vector indicating whether the observation falls into that category or not.
# Let's see how to achieve this transformation and then look at the advantages.
#
# Here we have a bunch of columns concatenated onto the end that contain the one-hot encodings for each categorical variable.
# Let's focus on only the `Natural Crystal Structure` features for now:
#
# Now let's evaluate how this is different from the above `int`-based scheme:
#
# Note that there are only two distances between different rows: 0 or $\sqrt{2}$.
# This is because the features now live in a k-dimensional space where each label is equidistant from the others.
# Here's a visualization:

# %%
data['Natural Crystal Structure'].value_counts()

cat, labels = np.unique(data['Natural Crystal Structure'], return_inverse=True)

print(cat)
print(labels)
print(cat[labels])

from scipy.spatial import distance
from matplotlib import pyplot as plt

n = len(labels)
dist = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        dist[i, j] = np.abs(labels[i] - labels[j])

fig, ax = plt.subplots()
_ = ax.hist(dist.flatten())
_ = ax.set_xlabel('Distance')
_ = ax.set_ylabel('Frequency')

from plotly import express as px

cat, labels = np.unique(data['Natural Crystal Structure'], return_inverse=True)

px.scatter(x=labels, y=np.random.rand(n), color=labels, hover_name=cat[labels])

pd.get_dummies(data)

# get the augmented DataFrame
data_one_hot = pd.get_dummies(data, columns=['Natural Crystal Structure'])

# drop the original features
n_features = data.shape[1]
data_one_hot = data_one_hot.iloc[:, n_features-1:]
data_one_hot

# convert to numpy array
features = data_one_hot.values.astype(int)

n = len(features)
dist = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        dist[i, j] = np.linalg.norm(features[i] - features[j])

fig, ax = plt.subplots()
_ = ax.hist(dist.flatten())
_ = ax.set_xlabel('Distance')
_ = ax.set_ylabel('Frequency')

px.scatter_3d(x=features[:, 0], y=features[:, 3], z=features[:, 4], color=labels, hover_name=cat[labels])
