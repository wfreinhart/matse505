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
# id: Lecture09_evaluating_the_effect_of_hyperparameters
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## Evaluating the effect of hyperparameters
#
# Let's start by just looking at how changing the hyperparameters influences model performance in an example: K-Neighbors.
#
# Let's push some of these operations into functions to make our life easier going forward.
#
# Now we'll repeat the analysis above with our new convenience functions:
#
# Let's finally get to changing the hyperparameters.
# You'll see why we spent the time on that detour shortly...
#
# The two key hyperparameters for K-Neighbors are `n_neighbors` and `weights`.
# They default to `5` and `uniform` respectively.
# Let's try doubling to `n_neighbors=10` and changing to `weights=distance`:
#
# OK, so that made things slightly worse in both training and testing.
# Let's try the other one:
#
# This improves the test performance a good amount but substantially overfits to the training data!
#
# Of course we could also try doing both together:
#
# Same problem with overfitting but it gives the best test performance so far.

# %%
from sklearn import neighbors

model = neighbors.KNeighborsRegressor()

model.fit(xtrain, ytrain)

y_pred = model.predict(xtrain)
residual = y_pred - ytrain
rmse = np.sqrt(np.mean(residual**2))
print(f'Train  RMSE = {rmse:.3f}')

y_pred = model.predict(xtest)
residual = y_pred - ytest
rmse = np.sqrt(np.mean(residual**2))
print(f'Test RMSE = {rmse:.3f}')

def calc_rmse(model, X, y):
    "Calculate the RMSE from a fitted model"
    y_pred = model.predict(X)
    residuals = y_pred - y
    return np.sqrt(np.mean(residuals**2))


def train_and_report_performance(model, xtrain, ytrain, xtest, ytest):
    "Train a model and print RMSE results for train and test sets"
    model.fit(xtrain, ytrain)
    train_rmse = calc_rmse(model, xtrain, ytrain)
    test_rmse  = calc_rmse(model, xtest, ytest)
    print(f'Train RMSE = {train_rmse:.3f}; Test RMSE = {test_rmse:.3f}')

train_and_report_performance(neighbors.KNeighborsRegressor(),
                             xtrain, ytrain, xtest, ytest)

train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=10),
                             xtrain, ytrain, xtest, ytest)

train_and_report_performance(neighbors.KNeighborsRegressor(weights='distance'),
                             xtrain, ytrain, xtest, ytest)

train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance'),
                             xtrain, ytrain, xtest, ytest)
