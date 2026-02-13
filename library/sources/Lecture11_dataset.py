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
# id: Lecture11_dataset
# type: Foundational
# parent_lecture: Lecture11
# ---
#
# ## Dataset
#
# Let's start by loading the concrete data:
#
# We'll set up the problem to be a regression task using all the features to predict the compressive strength:

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

x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
