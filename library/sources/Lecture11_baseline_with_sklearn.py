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
# id: Lecture11_baseline_with_sklearn
# type: Foundational
# parent_lecture: Lecture11
# ---
#
# ## Baseline with `sklearn`
#
# We already know this dataset has important nonlinearities that make it unsuitable for linear regression.
# Here is the baseline performance with multivariate linear regression using `sklearn.linear_model.LinearRegression`:
#
# Note that we aren't even considering test performance here, this is purely to evaluate model expressiveness.
# Even with all the data seen during training, the linear regression model cannot capture the nonlinear behavior.
#
# Now let's evaluate the performance of a `sklearn.neural_network.MLPRegressor` object:
#
# We see that this NN model is more expressive and can achieve greater performance on the regression task *at least when all the data are available for training*.
# We'll return to the issue of validation and test performance soon, after we get the `pytorch` syntax down.

# %%
from sklearn import linear_model

model = linear_model.LinearRegression().fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )

from sklearn import neural_network

model = neural_network.MLPRegressor(random_state=0).fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )
