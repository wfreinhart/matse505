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
# id: Lecture08_new_dataset_3d_printing
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## New dataset: 3d printing
#
# We'll set up a regression problem to predict the surface roughness:
#
# Here is our baseline performance:

# %%
data = pd.read_csv('../datasets/3dprinting.csv')
data

x = data.loc[:, :'fan_speed (%)']
x = pd.get_dummies(x)  # convert categorical to one-hot!

y = data.loc[:, 'roughness (microns)']

model = linear_model.LinearRegression().fit(x, y)
print( 'R2 = ', model.score(x, y) )
