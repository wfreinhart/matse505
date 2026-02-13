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
# id: Lecture03_polynomial_regression
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Polynomial regression
#
# What recourse do we have when linear regression isn't enough?
#
#
# Consider the following function: $f(x) = a x^2 + b x + c$.
#
# Fitting this with `minimize` would actually be the same as using our new multiple linear regression approach on an array with columns $(x, x^2)$.
#
# We can use `sklearn.preprocessing.PolynomialFeatures` to generate these features automatically from a feature vector.
# It takes the maximum polynomial degree `degree` as a keyword argument.
#
# Consider some 2D features $(x_0, x_1)$:
# * `PolynomialFeatures` with `degree=2` would generate the following features: $(x_0, x_1, x_0^2, x_0 x_1, x_1^2)$.
# * From these, $(x_0, x_1)$ are of degree 1 and $(x_0^2, x_0 x_1, x_1^2)$ are degree 2.
#
# Where did we get 45 features?!?!?
#
# They are basically the combinations of all the 8 features we put into the preprocessor.
#
# We can see their names with the `get_feature_names_out()` method:
#
# We can now use these 45 features in a `LinearRegression` just as if they were any other features:
#
# The results from our regular linear regression were:
#
# ```
# train: Rsq = 0.611, RMSE = 10.553
# test:  Rsq = 0.623, RMSE = 9.793
# ```
#
# So this polynomial preprocessing has given us a significant boost in performance.
#
# We can also check if the `Age` problem is resolved:
#
# Here we see that the systematic deviation of `Age` is at least partially removed by using polynomial features (though perhaps the green points at intermediate Age now drift higher).

# %%
from sklearn.preprocessing import PolynomialFeatures

print('Shape of X:       ', x.shape)

poly = PolynomialFeatures(degree=2)

poly = poly.fit(x)
print('Shape of poly(X): ', poly.transform(x).shape)
print(poly.transform(x))

print(poly.get_feature_names_out()[4:14])

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2).fit(x)
xptrain = poly.transform(xtrain)
xptest = poly.transform(xtest)

# fit on the transformed features!
model = linear_model.LinearRegression().fit(xptrain, ytrain)

y_pred = model.predict(xptrain)
residuals = y_pred - ytrain
r2 = 1 - np.var(residuals) / np.var(ytrain - ytrain.mean())
rmse = np.sqrt(np.mean(residuals**2))
print(f'train:  Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

y_pred = model.predict(xptest)
residuals = y_pred - ytest
r2 = 1 - np.var(residuals) / np.var(ytest - ytest.mean())
rmse = np.sqrt(np.mean(residuals**2))
print(f'test:  Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

fig, ax = plt.subplots(figsize=(5, 5))

# plot the results
ax.plot(ytrain, model.predict(xptrain), '.', label='Train')
ax.plot(ytest, model.predict(xptest), '.', label='Test')

# create reference line along y = x to show the desired behavior
min_max = np.array([y.min(), y.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')  # very helpful to show y = x relationship

# add labels and legend
ax.set_title('Multiple linear regression model of concrete strength')
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')
ax.legend()

fig, ax = plt.subplots(figsize=(5, 5))

# plot the results
y_pred = model.predict(poly.transform(x))
im = ax.scatter(y, y_pred, s=16, c=x['Age (day)'], label='Data')
cb = plt.colorbar(im, ax=ax)
cb.set_label('Age (day)')

# create reference line along y = x to show the desired behavior
min_max = np.array([y.min(), y.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')  # very helpful to show y = x relationship

# add labels and legend
ax.set_title('Linear regression of concrete strength')
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')
ax.legend()
