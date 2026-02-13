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
# id: Lecture19_sklearn_imputers
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## `sklearn` imputers
#
# The `scikit-learn` module has a submodule called `sklearn.impute` that implements the following:
# * `SimpleImputer`
# * `IterativeImputer`
# * `KNNImputer`
#
# `SimpleImputer` is similar to the schemes we just went over in `pandas`, so we'll skip it.
# The `IterativeImputer` is also "experimental" for now and in heavy development, so we'll skip that one as well.
#
# From the [documentation](https://scikit-learn.org/stable/modules/impute.html):
#
#
# The `KNNImputer` class provides imputation for filling in missing values using the k-Nearest Neighbors approach.
# By default, a Euclidean distance metric that supports missing values, `nan_euclidean_distances`, is used to find the nearest neighbors.
# Each missing feature is imputed using values from `n_neighbors` nearest neighbors that have a value for the feature.
# The feature of the neighbors are averaged uniformly or weighted by distance to each neighbor.
#
# If a sample has more than one feature missing, then the neighbors for that sample can be different depending on the particular feature being imputed.
# When the number of available neighbors is less than `n_neighbors` and there are no defined distances to the training set, the training set average for that feature is used during imputation.
# If there is at least one neighbor with a defined distance, the weighted or unweighted average of the remaining neighbors will be used during imputation.
# If a feature is always missing in training, it is removed during transform.
#
# The imputer returns an array with the missing values imputed.
# Note that we did not include the first column,
# We can use it to train the model as above:
#
# Here you can see that the model performance is substantially higher on the full dataset when the values have been imputed.
# > The warning about feature names is because we trained this Random Forest on an `ndarray` (`result`) but we are testing it on a `DataFrame` (`x_full`)

# %%
from sklearn import impute

imputer = impute.KNNImputer(n_neighbors=5)
result = imputer.fit_transform(missing.iloc[:, 1:])

x = result[:, :-1]
y = result[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

x = result[:, :-1]
y = result[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')
