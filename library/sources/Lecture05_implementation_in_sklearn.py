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
# id: Lecture05_implementation_in_sklearn
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Implementation in `sklearn`
#
# Let's load the alloys dataset from before...
#
# We start by defining our features, $X$.
#
# Unlike in the supervised case, we have no $y$ labels.
#
# Also, we don't need to do a train/test split because we have no labels to check against!

# %%
import pandas as pd
import numpy as np
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

x = data.loc[:, ' C':'Nb + Ta']
x
