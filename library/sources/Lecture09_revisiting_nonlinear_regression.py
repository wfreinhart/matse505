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
# id: Lecture09_revisiting_nonlinear_regression
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## Revisiting nonlinear regression
#
# Let's pick up where we left off with nonlinear regression: using `sklearn` models to fit a multivariate regression problem for `Concrete compressive strength`.
#
# Split the dataset into train and test sets:

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
data

from sklearn import model_selection

x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
y = data.loc[:, 'Concrete compressive strength(MPa, megapascals) ']

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, random_state=0)
print(xtrain.shape, xtest.shape)
