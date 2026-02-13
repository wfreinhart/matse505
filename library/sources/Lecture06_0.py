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
# id: Lecture06_0
# type: Foundational
# parent_lecture: Lecture06
# ---
#
#
#
# Today's topics:
# * Feature scaling
# * Reconstruction
# * Manifold learning
# * Semi-supervised learning
#
# Let's load the alloys dataset from before...
#
# Drop the outlier we found last time:
#
# And define a space to work in:

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

outlier = np.argmax(data[' Tensile Strength (MPa)'])
clean_data = data.drop(index=outlier)

x = clean_data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']
