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
# id: Lecture19_missforest
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## MissForest
#
# MissForest is an imputation algorithm for handling missing data that is based on the random forest algorithm. The algorithm works as follows:
#
# 1. Identify the features that contain missing values.
# 2. Split the dataset into two parts: one part that contains the missing values and one part that contains the observed values.
# 3. For each feature that contains missing values, build a random forest regression model using the observed values as the training data and the feature with missing values as the target variable.
# 4. Predict the missing values for each feature using the corresponding random forest regression model.
# 5. Repeat steps 3-4 until the missing values converge.
#
# During each iteration, the random forest model is trained using the observed values for all features except the feature with missing values. The missing values for the feature are then predicted using the trained model. This process is repeated for all features with missing values until the missing values converge, which means that the imputed values no longer change significantly.
#
#
#
# MissForest uses the random forest algorithm because it can handle a mixture of categorical and numerical data, and can handle non-linear relationships between features. The algorithm is also able to capture feature interactions, which is important when imputing missing values.
#
# We will have to start by installing the package:
#
# Once installed, using it is straightforward:
#
# First, let's compare the results from `MissForest` with the real thing.
# > Note that we would never have this opportunity in real life when we have actual missing data. This academic example is only to evaluate the quality of the imputation.
#
# Unfortunately it seems the `MissForest` imputer is not doing a great job at guessing the missing values.
#
# Let's see how our regression model performs with this imputation.
#
# Despite the imperfect imputation above, this is the best performance we found so far.
# Why are we able to get good performance when using values that don't match the real observations?
# Basically it's because only 1-2 of the values in each row are wrong, while all the rest are real data!
# The slight hit we take to reliability of those 1-2 features is worth it to retain all the other columns and double the volume of training data.

# %%
# !pip install MissForest

from missforest.miss_forest import MissForest

mf = MissForest()
imputed = mf.fit_transform(missing)

imputed

from matplotlib import pyplot as plt

missing_cols = [' Si', ' Ni', ' Al', ' Temperature (°C)', ' Elongation (%)']

fig, ax = plt.subplots()
for col in missing_cols:
    scale = 1 / data[col].max()  # scale all the data to [0, 1]
    ax.plot(imputed[col] * scale, data[col] * scale, '.', label=col)
ax.set_xlabel('Imputed')
ax.set_ylabel('Observation')
ax.legend()

x = imputed.iloc[:, 1:-1]
y = imputed.iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')
