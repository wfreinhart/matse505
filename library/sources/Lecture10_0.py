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
# id: Lecture10_0
# type: Foundational
# parent_lecture: Lecture10
# ---
#
#
#
# Today's topics:
# * Bayesian optimization for hyperparameter tuning
# * Evolutionary algorithm for feature selection
#
# Start by loading the alloys mechanical properties dataset:
#
# import pandas as pd
# import os
#
# Set the path to the data file
# filename = 'steels.csv'
# local_path = f'../datasets/{filename}'
# github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'
#
# Load the data: try local path first, fallback to GitHub for Colab
# if os.path.exists(local_path):
#     data = pd.read_csv(local_path)
# else:
#     data = pd.read_csv(github_url)
# data
#
# Select the features and regression labels:
#
# Let's train a multivariate linear regression as a baseline:
#
# And if we try a tree-based scheme?
#
# Random Forest is much better!
# Although there is still one fold that gives poor validation performance.
# What about a Neural Network?
#
# I mentioned before that neural networks require more extensive hyperparameter tuning.
# Let's explore methods for doing that now.

# %%
x = data.loc[:, ' C':' Temperature (°C)']
y = data[' Tensile Strength (MPa)']

from sklearn import model_selection
from sklearn import linear_model

model = linear_model.LinearRegression()

folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for train_index, test_index in folds.split(x):
    # split the data
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    # train a model
    model.fit(x_train, y_train)
    # evaluate r2 on validation set
    r2 = model.score(x_val, y_val)
    results.append(r2)

print(results)

from sklearn import model_selection
from sklearn import ensemble

model = ensemble.RandomForestRegressor(random_state=0)

folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for train_index, test_index in folds.split(x):
    # split the data
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    # train a model
    model.fit(x_train, y_train)
    # evaluate r2 on validation set
    r2 = model.score(x_val, y_val)
    results.append(r2)

print(results)

from sklearn import model_selection
from sklearn import neural_network

model = neural_network.MLPRegressor(random_state=0)

folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for train_index, test_index in folds.split(x):
    # split the data
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    # train a model
    model.fit(x_train, y_train)
    # evaluate r2 on validation set
    r2 = model.score(x_val, y_val)
    results.append(r2)

print(results)
