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
# id: Lecture03_linear_regression_as_an_optimization_problem
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Linear regression as an optimization problem
#
# Certainly none of these attributes explain the data on their own.
# Regardless, let's try fitting a linear regression to the `Cement` and see how well the model does since it had the highest correlation.
#
# Last time we used `stats.linregress` to compute a linear regression.
# Now let's consider the mechanics in greater detail.
# If we look up the documentation, we will see the following:
#
# ```
# scipy.stats.linregress(x, y=None, alternative='two-sided')
# Calculate a linear least-squares regression for two sets of measurements.
# ```
#
# What does "least-squares regression" mean?
# It means the function searches for the parameters $m$ and $b$ in a model $\hat{y} = m x + b$ such that the sum of square residuals is minimized:
#
# $L = \sum_i (\hat{y}_i - y_i)^2$
#
# The function $L$ is called an "objective function" or a "loss function."
# We can solve for the minimum parameters using the `scipy.optimize.minimize` function.
#
# Let's take a look at how this works.
# Here's a still from [this youtube video](https://youtu.be/qAWOnPfZkGM?t=1234) illustrating the idea:
#
# ![image.jpg](../lectures/assets/lecture03_loss_function.jpg)
#
# Let's look at the `minimize` docstring to get started:
# ```
# Help on function minimize in module scipy.optimize._minimize:
#
# minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
#     Minimization of scalar function of one or more variables.
#
#     Parameters
#     ----------
#     fun : callable
#         The objective function to be minimized.
#
#             ``fun(x, *args) -> float``
#
#         where x is an 1-D array with shape (n,) and `args`
#         is a tuple of the fixed parameters needed to completely
#         specify the function.
#     x0 : ndarray, shape (n,)
#         Initial guess. Array of real elements of size (n,),
#         where 'n' is the number of independent variables.
# ```
#
# The arguments we need to pay attention to (for now) are `fun` and `x0`.
#
# * `fun` needs to be a function that takes some `x` as input and returns a value that should be minimized
# * `x0` is an initial guess for the value of `x` that would give minimum
#
# We should first define functions that implement a linear model and the sum of squares objective based on that model:
#
# Then we can use these with `minimize` to identify optimal `m, b` parameters:
#
# The output has a lot of information. Here are the most important ones:
# * `x`: the parameters that gave lowest objective
# * `fun`: the value of the objective function at the end
#
# And some additional ones that may be interesting to look at:
# * `message`: a descriptive message about what happened
# * `nfev`: the number of function evaluations
# * `nit`: the number of solver iterations
# * `success`: a `bool` indicating if convergence was achieved
#
# The rest you can often ignore.
# Here is how we can refer to the `fun` and `x` values:
#
# Let's compare this to the result of `stats.linregress`:
#
# As you can see, the result is identical (to several decimal places).
# Also, the model performance is quite poor.

# %%
def linear_model(x, params):
    m, b = params
    return m * x + b

def least_squares_objective(params, x, y):
    y_model = linear_model(x, params)
    residual = y_model - y
    return np.sum(residual**2)

from scipy import optimize

# define the input and output variables
x = data['Cement (component 1)(kg in a m^3 mixture)']
y = data['Concrete compressive strength(MPa, megapascals) ']

# call the minimize
result = optimize.minimize(least_squares_objective, [1, 1], args=(x, y))
print(result)

print(f'least squares = {result.fun}')
m, b = result.x
print(f'best model: y = {m} * x + {b}')

from scipy import stats

model = stats.linregress(x, y)
print(f'linregress model: y = {model.slope} * x + {model.intercept}')
print(f'model R-squared: {model.rvalue**2}')
