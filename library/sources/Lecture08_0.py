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
# id: Lecture08_0
# type: Foundational
# parent_lecture: Lecture08
# ---
#
#
#
# Today's topics:
# * Feature representation
# * Feature augmentation
# * Feature selection
#
# Loading a dataset:
#
# import pandas as pd
# import os
#
# Set the path to the data file
# filename = 'elements.csv'
# local_path = f'../datasets/{filename}'
# github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'
#
# Load the data: try local path first, fallback to GitHub for Colab
# if os.path.exists(local_path):
#     data = pd.read_csv(local_path, index_col=0)
# else:
#     data = pd.read_csv(github_url, index_col=0)
# data.head()
#
# This dataset has a combination of continuous labels and categorical features which will help us explore some nuances of feature representation.
# It also contains some missing values that we will have to deal with.
