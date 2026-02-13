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
# id: Lecture04_dataset
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## Dataset
#
# Let's dive in and see some examples of this in action. We need to switch to a dataset that has categorical labels:
#
# In this dataset, we have elemental compositions at the left and then some mechanical properties at the right.
# Let's try to use the data to predict the `Alloy code`, which is categorical.
#
# > **Note:** Some column names in this dataset have leading spaces (e.g., `' C'`, `' 0.2% Proof Stress (MPa)'`). We include these in the code below.
#
# We can start by looking at the values of `Alloy code`:
#
# This is too many categories for us to keep track of.
# Let's simplify things by taking only the first letter of each code.
# We can call this an `Alloy family`:
#
# Now we have only a few categories:
#
# We can prepare the train/test data by including all the composition columns in our $X$ and the `Alloy family` as our $y$:

# %%
# import requests
import pandas as pd
import os

# Set the path to the data file
filename = 'steels.csv'
local_path = f'../datasets/{filename}'
github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    data = pd.read_csv(github_url)

data                            # show a view of the data file

data['Alloy code'].unique()

data['Alloy family'] = [x[0] for x in data['Alloy code']]
data.head()

data['Alloy family'].unique()

from sklearn.model_selection import train_test_split

x = data.loc[:, ' C':'Nb + Ta']
y = data['Alloy family']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
# note: we make sure to all get the same answer with random_state=0
print(xtrain.shape, xtest.shape)
