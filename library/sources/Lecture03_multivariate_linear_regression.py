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
# id: Lecture03_multivariate_linear_regression
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Multivariate linear regression
#
# The great part of using `minimize` is that we're directly in control of the function to be minimized, so we can adapt this however we see fit.
# Let's implement a multiple linear regression with least squares:
#
# $\hat{y} = a_0 x_0 + a_1 x_1 + \ldots + a_n x_n$
#
# Now we also need to change our definition of `x` to include the other columns:
#
# How good is this model?
#
# This is much improved compared to the single linear regression ($R^2 = 0.25$).

# %%
def multiple_linear_model(x, params):
    # this takes advantage of numpy element-wise arithmetic:
    return np.sum(x * params, axis=1)

def objective(params, x, y):
    # only one change to make: which model is being called
    y_model = multiple_linear_model(x, params)
    residual = y_model - y
    return np.sum(residual**2)

# define the input and output variables
x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
y = data['Concrete compressive strength(MPa, megapascals) ']

# call the minimize
guess = np.ones(x.shape[1])  # set up the initial guess based on size of inputs
result = optimize.minimize(objective, guess, args=(x, y))
print(result)

# this is a little confusing! x refers to the vector being optimized
params = result.x

# plug into model def
y_model = multiple_linear_model(x, params)
residuals = y_model - y

r2 = 1 - np.var(residuals) / np.var(y - y.mean())

print(f'Rsq = {r2:.3f}')
