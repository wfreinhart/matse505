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
# id: Lecture19_pandas_imputation_schemes
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## `pandas` imputation schemes
#
# OK, so missing data hurts the model performance.
# What can we do about it?
#
# One simple method for data imputation is mean imputation.
# This involves filling in missing values with the mean of the non-missing values in that column.
# The basic logic behind mean imputation is that the mean value is often a reasonable estimate of the missing value, especially when the missing data is randomly distributed and not related to other variables.
#
# We can use the `fillna` method with the `mean` argument:
#
# Here the number of rows is increased from 517 to 573 by filling in missing `Temperature (°C)` values.
#
# Interestingly, despite a seemingly reduced performance on the test set, the performance is improved on the full (held-out) dataset when training the model on the mean-imputed temperature values!
# This shows how imputation can help learn from all the other available data even if the value in the imputed column is not perfect.
#
# Another common method for data imputation is forward or backward filling.
# This involves filling in missing values with the value that appears in the previous or next row, respectively.
# The basic logic behind forward fill imputation is that the missing values are sometimes similar to the previous observed values (i.e., there is an ordering to the dataset).
#
# We can use the `fillna` method with the method argument set to either `ffill` (for forward filling) or `bfill` (for backward filling). For example:
#
# Here we see a similar result compared to the mean imputation above.
# It is likely that there is some kind of ordering to the temperatures.

# %%
imputed = missing.copy(deep=True)

imputed[' Temperature (°C)'].fillna(imputed[' Temperature (°C)'].mean(), inplace=True)

print(imputed.dropna().shape)

x = imputed.dropna().iloc[:, 1:-1]
y = imputed.dropna().iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

imputed = missing.copy(deep=True)

imputed[' Temperature (°C)'].fillna(method='ffill', inplace=True)

print(imputed.dropna().shape)

x = imputed.dropna().iloc[:, 1:-1]
y = imputed.dropna().iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')
