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
# id: Lecture09_in_practice
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## In practice
#
# We can implement cross-fold validation using `model_selection.KFold`:
#
# These results are all over the place.
# We can look at the mean and standard deviation to try and get a clearer picture:
#
# We see here that the average test RMSE is both substantially higher than the training RMSE, and also highly variable.

# %%
from sklearn import model_selection

# set up the splitter
folds = model_selection.KFold(n_splits=5)

# run through each subset
k = 0
results = []
for train_index, test_index in folds.split(x):

    # define the train / test split
    # each fold gets its own split!
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # train and evaluate the model
    model = ensemble.RandomForestRegressor().fit(x_train, y_train)
    rmse_train = calc_rmse(model, x_train, y_train)
    rmse_test = calc_rmse(model, x_test, y_test)

    k += 1
    print(f'fold k = {k} RMSE: Train = {rmse_train:.3f}; Test = {rmse_test:.3f}')
    results.append([rmse_train, rmse_test])

mu = np.mean(results, axis=0)
sigma = np.std(results, axis=0)
for i, s in enumerate(['Train', 'Test ']):
    print(f'{s} average: {mu[i]:.3f} +/- {sigma[i]:.3f}')
