# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's class topics:
# * WebPlotDigitizer
# * Interpolation
# * Uncertainty estimation
# * Function inversion

# %% [markdown]
# # Interpolation and imputation
#
# Let's start with a quick high-level discussion about missing data.
# **Interpolation** and **imputation** are two techniques used to handle missing data in datasets.
#
# * **Interpolation** refers to the process of estimating missing values in a dataset by using the values of neighboring data points.
# This can be useful when working with time-series data or other types of data that exhibit a pattern over time.
#
# Here is what interpolation looks like in 2D:
#
# <img src="../lectures/assets/lecture18_interpolation.jpg" alt="Illustration of interpolation between discrete data points in 2D" width=600>
#
# * **Imputation** involves filling in missing values in a dataset with estimated values based on the available data. This can be useful when working with datasets that have a significant amount of missing data, and/or when the missing data are high-dimensional, as it allows you to still use the available data to train a machine learning model.
#
# Here is what imputation looks like on tabular data:
#
# <img src="../lectures/assets/lecture18_imputation.jpg" alt="Illustration of data imputation on tabular data" width=600>

# %% [markdown]
# # Digitizing charts
#
# A common source of data is images published in journal articles, textbooks, or on the web. We've been relying on prepared `CSV` files to do our data analysis with Python so far, but this isn't representative of the real world. Let's try using [WebPlotDigitizer](https://apps.automeris.io/wpd/) to obtain raw data from the following chart of mechanical performance of an architected material from [mechanical performance of an architected material](https://doi.org/10.1088/1757-899X/433/1/012078):
#
# <img src="../lectures/assets/lecture18_chart_to_digitize.jpg" alt="Stress-strain chart of an architected material for digitization exercise" width=540>

# %% [markdown]
# ## [Exercise: using WebPlotDigitizer]
#
# Here are the steps:
# * Load image -> select this image from disk
# * 2D (X-Y) Plot -> Align Axes
#   * Select low X, high X, low Y, high Y on the chart
#   * You can use arrow keys to shift the point slightly after each click
#   * Complete -> type in the values (i.e., 0.0, 0.8, 0.0, 80) -> OK
# * Foreground Color (click the colored box on the right)
#   * Color Picker -> Click on the blue line -> Done
# * Automatic Extraction: Box -> drag around the blue line
#   * You will see a yellow box appear
#   * Distance -> 80 (this is a color range) -> Filter Colors
#   * You will see the line highlighted in yellow
# * Averaging Window -> 6 Px, 6 Px -> Run
#   * You will see red dots appear on top of the blue line
# * View Data
#   * You will see the X, Y values in a popup window
#
#

# %% [markdown]
# ## Loading the result
#
# * Save as CSV
#   * Click the "folder" icon on the left side of Colab
#   * Drag and drop the CSV into the Files pane
#   * OR use the "upload" button (file with an up arrow on it)
#
# You will get a message warning you that Colab often resets and you will lose files stored on this instance when that happens. This just means you can't store files permanently on Colab (you can link to Google Drive or OneDrive if you need to do this).
#
# Now load `pandas` and import with `read_csv` as usual:

# %%
import pandas as pd

pd.read_csv('data.csv')

# %% [markdown]
# Note that `pandas` thinks the columns are *named* **0.005129...** and **-0.027810...** because there is no header in the file. We can add proper names by using the `header=None` and `names=` keyword arguments:

# %%
if os.path.exists(local_path):
    data = pd.read_csv(local_path, header=None, names=['Strain', 'Stress (MPa)'])
else:
    try:
        data = pd.read_csv(github_url, header=None, names=['Strain', 'Stress (MPa)'])
    except:
        data = pd.DataFrame(columns=['Strain', 'Stress (MPa)'])
data

# %% [markdown]
# ## Plotting the result

# %%
from matplotlib import pyplot as plt
import numpy as np

xs, ys = data.columns

light_blue = np.array([26, 110, 177]) / 255  # RGB value from WPD Color Picker

fig, ax = plt.subplots()
ax.plot(data[xs], data[ys], '.', color=light_blue)
ax.set_xlabel(xs)
ax.set_ylabel(ys)
ax.set_title('Compression of architected material')

# %%
print(xs)
print(ys)

# %% [markdown]
# > Note that I could also choose a similar looking color from [this list](https://matplotlib.org/stable/gallery/color/named_colors.html) instead of defining my own

# %% [markdown]
# # Interpolation

# %% [markdown]
# What's wrong with this? Let's pretend we're especially interested in the region around 25% strain.

# %%
ax.set_xlim(0.22, 0.28)  # zoom in on 0.22 to 0.28 strain
ax.set_ylim(5, 15)       # zoom in on 5 to 15 MPa stress
fig

# %% [markdown]
# If we wanted to read a value off the chart in between 0.25 and 0.26, for instance, we would need to *interpolate* between the known values. Let's remind ourselves what values we know from the chart:

# %%
print(data[xs][45:55])

# %% [markdown]
# Ok, let's say we want to get the stress at 0.253 % strain. How can we do it? We need to make in informed guess at what it would be based on the values around it. The simplest strategy could be to assume it is the same value as the closest known result. In this case, it would be:
#
#

# %%
idx = np.argmin(np.abs(data[xs] - 0.253))  # find the index of closest x value
print(data.iloc[idx, :])                   # print the corresponding row

# %% [markdown]
# So we're saying that because we know the stress is 11.75 MPa at 0.250 strain, it's also 11.75 MPa at 0.253 strain.
#
# What would this strategy look like in general? I'll plot it below:

# %%
xlist = np.linspace(data[xs].min(), data[xs].max(), 1000)
ylist = np.zeros_like(xlist)   # make an array of zeros the same size as xlist
i = 0
for x in xlist:
    idx = np.argmin(np.abs(data[xs] - x))  # find the index of closest x value
    ylist[i] = data[ys][idx]               # assign the corresponding y value
    i += 1
fig2, ax2 = plt.subplots()
ax2.plot(xlist, ylist, c=light_blue)
ax2.set_xlabel(xs)
ax2.set_ylabel(ys + '(interpolated)')
ax2.set_title('Compression of architected material')

# %% [markdown]
# Hmm, this isn't a very nice looking result. Let's try something else...

# %% [markdown]
# ## Linear interpolation
#
# A better strategy that might be even more intuitive than the one we just described is to assume a straight line between each data point. This is what `pyplot` does when we make a line plot from discrete data points:

# %%
fig2, ax2 = plt.subplots()
ax2.plot(data[xs], data[ys], c=light_blue)
ax2.set_xlabel(xs)
ax2.set_ylabel(ys + '(interpolated)')
ax2.set_title('Compression of architected material')

# %% [markdown]
# That looks a lot smoother and more natural. We can formalize this beyond drawing the chart to actually find arbitrary values, like our 0.253 strain example. Let's use this equation:
#
# $\hat{y} = \frac{y_{i+1} - y_i}{x_{i+1} - x_i} (\hat{x} - x_i) + y_i$

# %%
i = np.argmin(np.abs(data[xs] - 0.253))  # find the index of closest x value
print(data.iloc[idx:(idx+2), :])           # print the corresponding row

# %%
x = data[xs]
y = data[ys]

x_hat = 0.253  # this is our target strain value

slope = (y[i+1] - y[i]) / (x[i+1] - x[i])
y_hat = y[i] + (x_hat - x[i]) * slope
print(y_hat)  # this is our interpolated stress value

# %% [markdown]
# We see that between 0.250 and 0.258 the stress decreases from 11.75 to 10.33 MPa, so this result of 11.24 MPa makes sense.

# %% [markdown]
# ## `interp1d`
#
# Of course there's an easier way that doing this by hand each time, even easier than using a `for` loop. Let's use `scipy.interpolate.interp1d` for this purpose.

# %%
from scipy import interpolate

linear = interpolate.interp1d(data[xs], data[ys])
print(linear)

# %% [markdown]
# We get a *function* back from our `interp1d` call. We can pass it x values and it will return y values:

# %%
print(linear(0.253))

# %% [markdown]
# As we see, this matches our calculation done by hand above. The advantage of this function is we can also pass a whole array at once:

# %%
x_hat = np.linspace(data[xs].min(), data[xs].max(), 1000)
y_hat = linear(x_hat)

fig2, ax2 = plt.subplots()
ax2.plot(x, y, 'o', c=light_blue, label='Data')
ax2.plot(x_hat, y_hat, '.', c=light_blue, label='Linear')
ax2.set_xlabel(xs)
ax2.set_ylabel(ys)
ax2.set_title('Compression of architected material')
ax2.legend()

# %% [markdown]
# If we zoom in as we did before, we see the effect of our interpolation in between the observed data points:

# %%
ax2.set_xlim(0.22, 0.28)  # zoom in on 0.22 to 0.28 strain
ax2.set_ylim(5, 15)       # zoom in on 5 to 15 MPa stress
fig2

# %% [markdown]
# ## Spline interpolation
#
# It turns out there are more sophisticated ways to guess the intermediate values than assuming a linear trend. One such method is called "cubic spline." This is actually one of the most common interpolation schemes, and so it can be accessed using the `kind` keyword argument of `interp1d`:

# %%
spline = interpolate.interp1d(x, y, kind='cubic')
print(spline)

# %% [markdown]
# We can now use it exactly the same way we did with the linear fit:

# %%
x_hat = np.linspace(data[xs].min(), data[xs].max(), 1000)
y_hat = spline(x_hat)

ax2.plot(x_hat, y_hat, '.', c='tab:orange', label='Cubic')
ax2.legend()
fig2

# %% [markdown]
# There is additional "curviness" to the line now, which is calculated by looking at points beyond just the nearest two; the name "cubic spline" means that second and third derivatives are also taken into account.
#
# Just for your information, there are also other options for the `kind` keyword argument, including a "zero-order spline" which is the same as the very first interpolation strategy we tried, just assuming the same as the nearest value:

# %%
zero = interpolate.interp1d(x, y, kind='zero')
y_hat = zero(x_hat)

ax2.plot(x_hat, y_hat, '-', c='tab:green', label='Zero')
ax2.legend()
fig2

# %% [markdown]
# ## When interpolation is not enough
#
# Linear interpolation works well when the underlying relationship between the data points is linear and the distance between the data points is relatively small. However, linear interpolation can break down and require more sophisticated methods in several situations:
#
# 1. *Non-linear relationships:* If the underlying relationship between the data points is non-linear, linear interpolation may produce inaccurate estimates. In these cases, more sophisticated interpolation methods such as polynomial or spline interpolation may be necessary.
#
# 2. *Unevenly spaced data:* If the distance between the data points is not uniform, linear interpolation may not accurately capture the underlying relationship between the data points. In these cases, methods such as spline interpolation or kriging may be more appropriate.
#
# 3. *Extrapolation:* Linear interpolation can only be used to estimate values between existing data points. If you need to estimate values outside of the range of the existing data points, you will need to use more sophisticated methods such as polynomial or spline extrapolation.
#
# 4. *Outliers:* Linear interpolation assumes that the data points are representative of the underlying relationship between the variables. If there are outliers or other unusual data points in the dataset, linear interpolation may produce inaccurate estimates. In these cases, more sophisticated interpolation methods such as robust regression or kernel regression may be necessary.

# %% [markdown]
# # Uncertainty estimation

# %% [markdown]
# ## Gaussian Process
#
# Gaussian Process (GP) regression is a probabilistic approach to regression that uses a Gaussian Process to model the underlying function. The Gaussian Process defines a prior distribution over functions, which is then updated based on observed data to obtain the posterior distribution over functions. The posterior distribution can then be used to make predictions for new input values.
#
# <img src="../lectures/assets/lecture18_gp_noise.svg" alt="Gaussian Process regression with noise showing uncertainty bands" width=600>
#
# In GP regression, we assume that the underlying function is drawn from a Gaussian Process, which is a collection of random variables, one for each input value. The joint distribution of these random variables is a multivariate normal distribution, which is fully specified by a mean function and a covariance function. The mean function represents the expected value of the function, while the covariance function specifies how correlated the function values are for different input values.
#
# To make predictions for new input values, we first compute the joint distribution of the observed data and the new input values using the mean and covariance functions. We can then condition this joint distribution on the observed data to obtain the posterior distribution of the function values at the new input values. The mean of this posterior distribution gives us the predicted value of the function, while the variance gives us a measure of uncertainty in the prediction.

# %% [markdown]
# Let's try randomly sampling from the data points and see how the GP model does at interpolating:

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF

rng = np.random.RandomState(0)
training_indices = rng.choice(np.arange(y.size), size=18, replace=False)
x_train, y_train = x.values[training_indices].reshape(-1, 1), y.values[training_indices]

kernel = 1 * RBF(length_scale_bounds=(1e-2, 1e2)) + DotProduct()
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0).fit(x_train, y_train)
gpr.score(x.values.reshape(-1, 1), y)

# %% [markdown]
# We can make predictions usin gthe `predict` method, and include the uncertainty estimate using `return_std=True`:

# %%
mu, sigma = gpr.predict(x.values.reshape(-1, 1), return_std=True)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x, mu, color='tab:blue', label='GP Model')
ax.fill_between(x, mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')

# %% [markdown]
# We can also try it for extrapolation:

# %%
x_extrap = np.linspace(0.6, 1, 11).reshape(-1, 1)
mu, sigma = gpr.predict(x_extrap, return_std=True)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x_extrap[:, 0], mu, color='tab:blue', label='GP Model')
ax.fill_between(x_extrap[:, 0], mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')

# %% [markdown]
# Here we see there is no meaningful extrapolation beyond a "nearest neighbor" type assumption, although there is the added benefit of a reasonable assumption for uncertainty (i.e., the result is highly uncertain).

# %% [markdown]
# ## Neural function representation

# %%
from sklearn import neural_network

rng = np.random.RandomState(2)
training_indices = rng.choice(np.arange(y.size), size=18, replace=False)
x_train, y_train = x.values[training_indices].reshape(-1, 1), y.values[training_indices]

nn = neural_network.MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=4000, activation='tanh', random_state=0).fit(x_train, y_train)
nn.score(x.values.reshape(-1, 1), y)

# %%
mu = nn.predict(x.values.reshape(-1, 1))

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x, mu, color='tab:blue', label='GP Model')

# %% [markdown]
# ## Bootstrap
#
# **Bootstrap sampling** is a resampling technique used in machine learning to estimate the accuracy of a statistical model or to assess the stability of a model's predictions.
#
# The basic idea behind bootstrap sampling is to create multiple samples of the original dataset by randomly selecting observations with replacement. This means that some observations may be selected multiple times, while others may not be selected at all. The resulting samples, called bootstrap samples, have the same size as the original dataset but are created by sampling with replacement.
#
# Once the bootstrap samples have been created, a statistical model can be trained on each sample and the predictions can be combined to estimate the accuracy of the model or the stability of its predictions. For example, the average of the predictions across all bootstrap samples can be used as an estimate of the model's performance on new data.

# %%
models = []
for i in range(3):
    # do bootstrap sampling
    rng = np.random.RandomState(i)
    training_indices = rng.choice(np.arange(y.size), size=18, replace=False)
    x_train, y_train = x.values[training_indices].reshape(-1, 1), y.values[training_indices]

    # train a model
    nn = neural_network.MLPRegressor(hidden_layer_sizes=(100, 100),
                                     max_iter=4000, activation='tanh',
                                     random_state=0).fit(x_train, y_train)
    print(f'model {i}', nn.score(x.values.reshape(-1, 1), y))
    models.append(nn)


# %% [markdown]
# From the bootstrap procedure, we have 3 different models.
# We can query the discrepancy between the models as an estimate of uncertainty in the function:

# %%
y_hat = []
for nn in models[1:]:
    y_hat.append( nn.predict(x.values.reshape(-1, 1)) )

mu = np.mean(y_hat, axis=0)
sigma = np.std(y_hat, axis=0)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x, mu, color='tab:blue', label='GP Model')
ax.fill_between(x, mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')

# %% [markdown]
# We can also explore the extrapolation behavior:

# %%
x_extrap = np.linspace(0.6, 1, 11).reshape(-1, 1)
y_hat = []
for rf in models:
    y_hat.append( rf.predict(x_extrap) )

mu = np.mean(y_hat, axis=0)
sigma = np.std(y_hat, axis=0)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x_extrap[:, 0], mu, color='tab:blue', label='GP Model')
ax.fill_between(x_extrap[:, 0], mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')

# %% [markdown]
# ## [Check your understanding]
#
# Implement bootstrap resampling using a Random Forest regressor.
# Compare the performance to the neural function representation.

# %%
from sklearn import ensemble

models = []
for i in range(3):
    # do bootstrap sampling
    rng = np.random.RandomState(i)
    training_indices = rng.choice(np.arange(y.size), size=18, replace=False)
    x_train, y_train = x.values[training_indices].reshape(-1, 1), y.values[training_indices]

    # train a model
    rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
    print(f'model {i}', rf.score(x.values.reshape(-1, 1), y))
    models.append(rf)

# %%
y_hat = []
for rf in models[1:]:
    y_hat.append( rf.predict(x.values.reshape(-1, 1)) )

mu = np.mean(y_hat, axis=0)
sigma = np.std(y_hat, axis=0)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x, mu, color='tab:blue', label='GP Model')
ax.fill_between(x, mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')

# %% [markdown]
# # Function inversion
#
# We have now obtained models of the function $f(x) = y$ in a few different ways. What if we want to find the $x$ value to obtain a given $y$? This could be written as $f^{-1}(y) = x$, as the *inverse function*.
#
#

# %% [markdown]
# ## A simple example
#
# Consider the function $y = f(x) = x^3$. Say we want to find $f^{-1}(y)$.
# We know the analytical solution: $f^{-1}(y) = x^{1/3}$.

# %%
x = np.linspace(-1, 1, 21)
y = x**3

fig, ax = plt.subplots()
ax.plot(x, y, '.')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %% [markdown]
# Let's make a spline interpolation and check it against the true values:

# %%
x_hat = np.linspace(-1, 1, 101)
x3_spline = interpolate.interp1d(x, y, kind='cubic')
y_hat = x3_spline(x_hat)
ax.plot(x_hat, y_hat, '-')
fig

# %% [markdown]
# Now it turns out we can actually use our `interp1d` to model $f^{-1}$ too. All we need to do is swap the order of $x$ and $y$!

# %%
# note the order! y then x:
y3_spline = interpolate.interp1d(y, x, kind='cubic')

y_hat = np.linspace(-1, 1, 101)
x_hat = y3_spline(y_hat)  # call on y data!

fig, ax = plt.subplots()
ax.plot(y, x, '.')
ax.set_xlabel('$y$')  # flipped!
ax.set_ylabel('$x$')

ax.plot(y_hat, x_hat, '-')

# %% [markdown]
# It doesn't work quite as well as the other way around because this function is $x = y^{-1/3}$, whereas the cubic spline is using polynomials the $x, x^2, x^3$ of which one of them is the true function!
#
# Anyways, we can still check that it works:

# %%
x_test = 0.67
print(f'Spline f^-1 = {y3_spline(x_test)}')
print(f'True f^-1   = {x_test**(1/3)}')

# %% [markdown]
# We can see there is some error here, but it does get us close without knowing anything about the analytical solution.

# %% [markdown]
# ## Problems with real data
#
# This function will only be well defined when there is a single $y \to x$ mapping. Let's consider what this means graphically:

# %%
x = data[xs]
y = data[ys]

spline = interpolate.interp1d(x, y, kind='cubic')

x_hat = np.linspace(x.min(), x.max(), 1000)
y_hat = spline(x_hat)

fig, ax = plt.subplots()
ax.plot(x_hat, y_hat, '-', c=light_blue)
ax.set_xlabel(xs)
ax.set_ylabel(ys)
ax.set_title('Compression of architected material')

y_target = 30  # in MPa
ax.plot([data[xs].min(), data[xs].max()], [y_target, y_target],
        linestyle='--', c='tab:orange')

# %% [markdown]
# If we ask our function $f^{-1}(y)$ to return $x$ for $y = 30$, which $x$ should it choose? Let's see how `interp1d` behaves here:

# %%
inv_spline = interpolate.interp1d(y, x, kind='cubic')

y_hat = np.linspace(y.min(), y.max(), 1000)
x_hat = inv_spline(y_hat)

fig, ax = plt.subplots()
ax.plot(y, x, '.', c=light_blue)
ax.plot(y_hat, x_hat, '-', c=light_blue)
ax.set_xlabel(ys)
ax.set_ylabel(xs)
ax.set_title('Compression of architected material')

# %% [markdown]
# Ok, that's not what we want at all. If the function is multi-valued, the behavior can be erratic.

# %% [markdown]
# ## Multi-valued functions
#
# To fix this, we need to use logical indexing and create a localized interpolation. Let's say we only want the value up to the first maximum.

# %%
top_strain = x[np.argmax(y)]  # find peak location
print(f'peak occurs at strain = {top_strain}')

before_top = x <= top_strain

inv_spline = interpolate.interp1d(y[before_top], x[before_top], kind='cubic')

y_hat = np.linspace(y[before_top].min(), y[before_top].max(), 1000)
x_hat = inv_spline(y_hat)

fig, ax = plt.subplots()
ax.plot(y, x, '.', c=light_blue)
ax.plot(y_hat, x_hat, '-', c=light_blue)
ax.set_xlabel(ys)
ax.set_ylabel(xs)
ax.set_title('Compression of architected material')

# %% [markdown]
# Now we can reliably invert that first part of the function:

# %%
y_target = 30
print(f'Strain for {y_target} MPa Stress = {inv_spline(30)}')

# %% [markdown]
# ## [Check your understanding]
#
# * Find the Stress for a 24% Strain using "forward" interpolation.
# * Find the Strain for the same Stress using "inverse" interpolation
#
# > Hint: use logical indexing to exclude part of the dataset
#
# > Note: the results will not match exactly due to numerical errors!

# %%
