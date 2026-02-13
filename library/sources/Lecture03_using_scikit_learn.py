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
# id: Lecture03_using_scikit_learn
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Using scikit-learn
#
# The advantage of using `sklearn` is that we have access to many regression models with a common interface.
# Let's repeat our multiple linear regression using scikit-learn.
#
# `sklearn` is (heavily) object-oriented, so we need to first create a `LinearRegression` object:
#
# Model "fitting" (i.e., parameter optimization) is performed with the `fit()` method:
#
# Note how the `fit()` method returns the fitted model object. In the future, we can just write something like
#
# > `model = linear_model.LinearRegression().fit(x, y)`
#
# Now let's use the `predict()` method to evalute the model:
#
# As you can see, the result is exactly the same as with our manual approach.
# Only the interface for fitting and predicting is different.
# However, you can see a difference between the Statistical way of thinking (where we compute `y_model` using the model equation itself) and the Machine Learning way of thinking (where we call a method called `predict` that computes the result for us).
# If you can adapt to thinking of these functions in the abstract, rather than needing to write down the formulas by hand all the time, you will be able to generalize your workflows much more.

# %%
from sklearn import linear_model

model = linear_model.LinearRegression()
print(model)

model.fit(x, y)

y_pred = model.predict(x)
residuals = y_pred - y

r2 = 1 - np.var(residuals) / np.var(y - y.mean())

print(f'Rsq = {r2:.3f}')
