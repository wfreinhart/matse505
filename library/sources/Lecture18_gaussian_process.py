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
# id: Lecture18_gaussian_process
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Gaussian Process
#
# Gaussian Process (GP) regression is a probabilistic approach to regression that uses a Gaussian Process to model the underlying function. The Gaussian Process defines a prior distribution over functions, which is then updated based on observed data to obtain the posterior distribution over functions. The posterior distribution can then be used to make predictions for new input values.
#
# <img src="../lectures/assets/lecture18_gp_noise.svg" alt="Gaussian Process regression with noise showing uncertainty bands" width=600>
#
# In GP regression, we assume that the underlying function is drawn from a Gaussian Process, which is a collection of random variables, one for each input value. The joint distribution of these random variables is a multivariate normal distribution, which is fully specified by a mean function and a covariance function. The mean function represents the expected value of the function, while the covariance function specifies how correlated the function values are for different input values.
#
# To make predictions for new input values, we first compute the joint distribution of the observed data and the new input values using the mean and covariance functions. We can then condition this joint distribution on the observed data to obtain the posterior distribution of the function values at the new input values. The mean of this posterior distribution gives us the predicted value of the function, while the variance gives us a measure of uncertainty in the prediction.
#
# Let's try randomly sampling from the data points and see how the GP model does at interpolating:
#
# We can make predictions usin gthe `predict` method, and include the uncertainty estimate using `return_std=True`:
#
# We can also try it for extrapolation:
#
# Here we see there is no meaningful extrapolation beyond a "nearest neighbor" type assumption, although there is the added benefit of a reasonable assumption for uncertainty (i.e., the result is highly uncertain).

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF

rng = np.random.RandomState(0)
training_indices = rng.choice(np.arange(y.size), size=18, replace=False)
x_train, y_train = x.values[training_indices].reshape(-1, 1), y.values[training_indices]

kernel = 1 * RBF(length_scale_bounds=(1e-2, 1e2)) + DotProduct()
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0).fit(x_train, y_train)
gpr.score(x.values.reshape(-1, 1), y)

mu, sigma = gpr.predict(x.values.reshape(-1, 1), return_std=True)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x, mu, color='tab:blue', label='GP Model')
ax.fill_between(x, mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')

x_extrap = np.linspace(0.6, 1, 11).reshape(-1, 1)
mu, sigma = gpr.predict(x_extrap, return_std=True)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x_extrap[:, 0], mu, color='tab:blue', label='GP Model')
ax.fill_between(x_extrap[:, 0], mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')
