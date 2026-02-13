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
# id: Lecture03_supervised_learning_with_real_world_data
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# # Supervised learning with real-world data
#
# We'll use this dataset of concrete compressive strengths for our supervised learning examples.
# The data were obtained from [this Kaggle page](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set).
#
# > NOTE: Reuse of this database is unlimited with retention of copyright notice for Prof. I-Cheng Yeh and the following published paper:
# I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
# neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)
#
# Let's try to predict `Concrete compressive strength(MPa, megapascals)`.
# We should start by trying to understand the dataset.
# The independent variables are on the left.
# Most are compositions, but the last one is Age in days.
# We can start by checking how the `Concrete compressive strength(MPa, megapascals) ` correlates to the other variables using the `corr()` method of the `DataFrame`:
#
# It looks like `Cement`, `Superplasticizer`, and `Age` are the strongest contributors to the `Concrete compressive strength`. Let's evaluate these trends visually:
#
# > **Note:** Some column names in this dataset have trailing spaces (e.g., `'Concrete compressive strength(MPa, megapascals) '`). Be careful when indexing!

# %%
import requests
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

data.corr()['Concrete compressive strength(MPa, megapascals) ']

ax = data.plot.scatter('Cement (component 1)(kg in a m^3 mixture)', 'Concrete compressive strength(MPa, megapascals) ')
ax = data.plot.scatter('Superplasticizer (component 5)(kg in a m^3 mixture)', 'Concrete compressive strength(MPa, megapascals) ')
ax = data.plot.scatter('Age (day)', 'Concrete compressive strength(MPa, megapascals) ')
