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
# id: Lecture09_validation_set
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## Validation set
#
# Our naiive optimization scheme above is flawed:
# in trying all the hyperparemeters and choosing the best performing set, we have ruined our test set!
# The test set is supposed to represent unseen data, but here the model is once again being tuned to achieve performance metric based on the test data.
# What's the solution?
#
# We need another split in our data!
#
# <img src="../lectures/assets/hyperparameter_tuning_bias_variance.jpg" alt="Illustration of the bias-variance tradeoff in hyperparameter tuning showing underfitting and overfitting regions">
#
# Here we see that the performance on truly unseen data is not as good as we hoped based on the validation set -- note that our "test" result printed by `train_and_report_performance` is actually the *validation* set.
# We should update our convenience function:
#
# Here's the new function in action:
#
# Now we can deploy it in our hyperparameter tuning:
#
# The result is much less clear than before.
# We can clearly see that the result with the lowest validation performance ($k = 9$) is not the one with the lowest test performance ($k = 6$).
# This illustrates a similar problem as with overfitting -- you can't trust the performance on data included in the workflow when deploying on unseen data!
#
# In practice, we do have to choose a single set of hyperparameters to use, so we would select $k=9$ and report a test RMSE of 8.230 even though there are other options that would give a lower test RMSE.

# %%
# split off 20% of the data for testing -- it will never be seen by the model
x_trv, x_test, y_trv, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

# remaining data has 80% of total ... want 60% of total for training = 75% of the remainder
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_trv, y_trv, train_size=0.75, shuffle=True, random_state=0)

# use the optimal hyperparameters from above...
model = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
train_and_report_performance(model, x_train, y_train, x_val, y_val)

# check the test result
print(f'Test RMSE = {calc_rmse(model, x_test, y_test):.3f}')

def train_and_report_performance(model, xtrain, ytrain, xtest, ytest, xval=None, yval=None):
    "Train a model and print RMSE results for train and test sets"
    model.fit(xtrain, ytrain)
    train_rmse = calc_rmse(model, xtrain, ytrain)
    test_rmse  = calc_rmse(model, xtest, ytest)
    if xval is not None and yval is not None:
        val_rmse = calc_rmse(model, xval, yval)
        print(f'Train RMSE = {train_rmse:.3f}; Val RMSE = {val_rmse:.3f}; Test RMSE = {test_rmse:.3f}')
    else:
        print(f'Train RMSE = {train_rmse:.3f}; Test RMSE = {test_rmse:.3f}')

model = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
train_and_report_performance(model, x_train, y_train, x_test, y_test, x_val, y_val)

for k in range(1, 20):
    print(f'k = {k:2d}: ', end='')
    train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance'),
                                 x_train, y_train, x_test, y_test, x_val, y_val)
