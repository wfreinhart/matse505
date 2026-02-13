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
# id: Lecture21_simple_regression_models
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## Simple regression models
#
# We can also try the regularized linear regression models:

# %%
from sklearn import linear_model, model_selection

train_idx, test_idx = model_selection.train_test_split(np.arange(x.shape[0]), random_state=0)

lr = linear_model.LinearRegression().fit(x[train_idx], y[train_idx])
print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')

for model in [linear_model.Lasso, linear_model.Ridge]:
    lr = model().fit(x[train_idx], y[train_idx])
    print(model)
    print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
    print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')
    print()

from sklearn import ensemble, neighbors

for model in [ensemble.RandomForestRegressor, neighbors.KNeighborsRegressor]:
    lr = model().fit(x[train_idx], y[train_idx])
    print(model)
    print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
    print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')
    print()
