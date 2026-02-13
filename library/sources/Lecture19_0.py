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
# id: Lecture19_0
# type: Foundational
# parent_lecture: Lecture19
# ---
#
#
#
# Today's topics:
# * Imputation
# * Data augmentation
# * Multi-Task Learning
#
# Let's use this Alloys dataset from before:
#
# There is an outlier in this dataset that needs to be corrected:

# %%
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

import numpy as np

bad_idx = np.argmax( data.loc[:, ' Tensile Strength (MPa)'] )
data.loc[bad_idx, ' Tensile Strength (MPa)'] /= 10.0  # missed a decimal point
