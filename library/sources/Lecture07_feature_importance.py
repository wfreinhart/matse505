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
# id: Lecture07_feature_importance
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# # Feature importance
#
# Now that we have trained several different kinds of models and found some that work well, we might want to understand how they work.
# A common scheme for interrogating a trained model is "feature importance" -- how important each input feature (variable) is in the final output.
#
# Let's revist the mechanical properties of concrete dataset:
#
# Let's say we want to understand how important each of the components is in determining the final strength of the material.
# We previously performed a multiple linear regression on this data and evaluated the coefficients as a proxy for feature importance.
# Let's repeat this now:
#
# We can interrogate the `coef_` attribute of the fitted `LinearRegression` object to find out the linear coefficients in front of each independent variable:
#
# We discussed three notable things about this result:
# * Water is the only component that correlates to a reduced strength
# * Superplasticizer is an additive that should have an outsized effect on strength when measured in kg/m^3, so it is reassuring to see it with the highest coefficient
# * Age is measured in days, so its effect can't be directly compared to the others

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

from sklearn import linear_model

x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
y = data.loc[:, 'Concrete compressive strength(MPa, megapascals) ']

model = linear_model.LinearRegression().fit(x, y)
print( model.score(x, y) )

print('linear model looks like:')
for i in np.argsort(np.abs(model.coef_)):
    print(f'{model.coef_[i]:6.3f} * {x.columns[i]}')
