# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Lecture03

# %%
class Context(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault('data', None)
        self.setdefault('tensors', {})
        self.setdefault('model', None)
        self.setdefault('viz', None)

ctx = Context()

# %% [markdown]
# > Homework 2 posted
#
# Today's topics:
# * Introduction to Machine Learning
# * Linear regression and optimization
# * Multivariate linear regression as ML
# * Nonlinear regression models

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Introduction to Machine Learning
#
# Machine Learning (ML) has become a household term in the last 10 years. One definition that I like is this:
#
# > "Machine learning is a method of data analysis that automates analytical model building."
#
# This graphic gives a good illustration of how ML relates to the overall field of Data Science:
#
# <img src="../lectures/assets/lecture03_data_science_ml_venn.jpg" height=400>
#
# ML is an essential a key part of practical Data Science workflows because it enables us to generate standardized models that can explain a very large class of problems. In turn this enables us to share the models broadly because there is a robust set of common models that the broader community can understand, implement, and use.
#
# Now let's get into the details of what ML is and how to use it.
# ML is often broken down into three categories:
# * Supervised learning
# * Unsupervised learning
# * Reinforcement learning
#
# We will cover Supervised and Unsupervised learning with hands-on examples and discuss applications of Reinforcement learning at the end.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Supervised learning
#
# Let's say we have a function $f(x) = y$. We have seen equations of this form many times before in math and engineering courses. Most often, we have acquired some data of the form $(x, y)_i$ and tried to find a form of $f$ that we feel explains the trends adequately.
#
# If we don't know the true generating function exactly, we use curve fitting to identify $f$. For instance, we have a vector of $x$ and a vector of $y$ and we write $f(x) = m x + b = y$. Here $f$ is a (univariate) linear regression between $x$ and $y$.
#
# Instead of saying the parameters of the function $f$ are **fit**, we can instead say the function $f$ is **learned**. There, now we are doing machine learning -- our Python programming is effectively learning a relationship between our dependent ($y$) and independent ($X$) variables.
# In ML, we call $X$ the **features** (uppercase indicates a 2D array instead of a 1D vector, each row of $X$ is called a feature vector) and $y$ the **labels**.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Statistical modeling versus Supervised ML
#
# *Wait, isn't this just curve fitting? I can do that in Excel*
#
# You can probably recall many times in which you've fitted (or *learned*) functional relationships of the form $f(x) = y$ in your engineering classes.
# Let's talk about how this is different from ML.
#
# “The major difference between machine learning and statistics is their purpose. Machine learning models are designed to make the most accurate predictions possible. Statistical models are designed for inference about the relationships between variables.”
#
# -Matthew Stewart, *The Actual Difference Between Statistics and Machine Learning*
#
# "Machine learning can be defined as the process of solving a practical problem by 1) gathering a dataset, and 2) algorithmically building a statistical model based on that dataset. That statistical model is assumed to be used somehow to solve the practical problem."
#
# -Andriy Burkov, *The Hundred-Page Machine Learning Book*
#
# We can see this in the Venn diagram that we used above. Scroll back up and notice how Machine Learning is the at intersection of Mathematics and Computer Science while Statistical Research is at the intersection of Mathematics and Domain Expertise.
#
# ML typically serves to handle generalizing models beyond the Domain or to extend models which cannot be made sufficiently predictive using Domain knowledge alone. However, Data Scientists require both ML and conventional Statistical Research to function effectively! ML is known to be somewhat brittle and can very easily fail when exposed to unforseen challenges. Domain Expertise is critical in identifying these failure modes and correcting them.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Contrasting vocabulary: Unsupervised learning
#
# Now consider that we have only a collection of $(X)_i$ without corresponding $(y)_i$.
# In other words, we have **unlabeled** data.
# However, we may still want to learn something about the data.
# For instance, do any of the data stick out as outliers?
# Are there discrete groups which can be discerned?
# This is the objective of unsupervised learning.
# The problem can be stated mathematically as $f(X) = \ell$, where $\ell$ is a categorical label belonging to a **class** or **cluster**. Graphically, the problem looks like this:
#
# <img src="../lectures/assets/lecture03_classification_groups.jpg" height=400>
#
# We'll do some unsupervised learning in a future lecture.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Implementation in `sklearn`
#
# [scikit learn](https://scikit-learn.org/stable/index.html), which has the package name `sklearn`, is our go-to Python package for ML implementations. It has modules for:
# * Classification
# * Regression
# * Clustering
# * Dimensionality reduction
# * Model selection
# * Preprocessing
#
# ...and more.
# Essentially all standard ML algorithms are available in this package.
# For deep learning I recommend [pytorch](https://pytorch.org/).

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Supervised learning with real-world data
#
# We'll use this dataset of concrete compressive strengths for our supervised learning examples.
# The data were obtained from [this Kaggle page](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set).
#
# > NOTE: Reuse of this database is unlimited with retention of copyright notice for Prof. I-Cheng Yeh and the following published paper:
# I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
# neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)
#
# Let's try to predict `Concrete compressive strength(MPa, megapascals)`.
# We should start by trying to understand the dataset.
# The independent variables are on the left.
# Most are compositions, but the last one is Age in days.
# We can start by checking how the `Concrete compressive strength(MPa, megapascals) ` correlates to the other variables using the `corr()` method of the `DataFrame`:
#
# It looks like `Cement`, `Superplasticizer`, and `Age` are the strongest contributors to the `Concrete compressive strength`. Let's evaluate these trends visually:
#
# > **Note:** Some column names in this dataset have trailing spaces (e.g., `'Concrete compressive strength(MPa, megapascals) '`). Be careful when indexing!

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    import requests
    import pandas as pd
    import numpy as np
    import os

    # Set the path to the data file
    filename = 'concrete.csv'
    local_path = f'../datasets/{filename}'
    github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

    # Load the data: try local path first, fallback to GitHub for Colab
    if os.path.exists(local_path):
        data = pd.read_csv(local_path)
    else:
        data = pd.read_csv(github_url)

    data                            # show a view of the data file

    data.corr()['Concrete compressive strength(MPa, megapascals) ']

    ax = data.plot.scatter('Cement (component 1)(kg in a m^3 mixture)', 'Concrete compressive strength(MPa, megapascals) ')
    ax = data.plot.scatter('Superplasticizer (component 5)(kg in a m^3 mixture)', 'Concrete compressive strength(MPa, megapascals) ')
    ax = data.plot.scatter('Age (day)', 'Concrete compressive strength(MPa, megapascals) ')
    ctx['data'] = data
    ctx['tensors']['filename'] = filename
    ctx['tensors']['github_url'] = github_url
    ctx['tensors']['local_path'] = local_path
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    least_squares_objective = ctx.get('tensors', {}).get('least_squares_objective')
    optimize = ctx.get('tensors', {}).get('optimize')
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
    ctx['model'] = model
    ctx['tensors']['result'] = result
    ctx['tensors']['x'] = x
    ctx['tensors']['y'] = y
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    multiple_linear_model = ctx.get('tensors', {}).get('multiple_linear_model')
    objective = ctx.get('tensors', {}).get('objective')
    optimize = ctx.get('tensors', {}).get('optimize')
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
    ctx['tensors']['guess'] = guess
    ctx['tensors']['params'] = params
    ctx['tensors']['residuals'] = residuals
    ctx['tensors']['result'] = result
    ctx['tensors']['x'] = x
    ctx['tensors']['y'] = y
    ctx['tensors']['y_model'] = y_model
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    x = ctx.get('tensors', {}).get('x')
    y = ctx.get('tensors', {}).get('y')
    from sklearn import linear_model

    model = linear_model.LinearRegression()
    print(model)

    model.fit(x, y)

    y_pred = model.predict(x)
    residuals = y_pred - y

    r2 = 1 - np.var(residuals) / np.var(y - y.mean())

    print(f'Rsq = {r2:.3f}')
    ctx['model'] = model
    ctx['tensors']['residuals'] = residuals
    ctx['tensors']['y_pred'] = y_pred
run_module(ctx)

# %% [markdown]
# ## Visualizing model performance
#
# How should we visualize the result?
# $x$ is no longer a 1D vector, but a 2D array with many observations of multiple values.
# We can make charts with up to 4 variables using 3D scatter charts plus a color axis, but here we have 8 features!
#
# The standard way is to plot the Prediction against the Observation, called a **parity plot**.
# This shows you whether there is a systematic bias in your model, and the chart looks the same no matter what sort of model you're using.
#
# We can see a slight bias to over-estimate at the bottom and over-estimate at the top. What might be going on here? One thing we could do is make a scatter plot of the results and color the points according to one of the variables. Then we will know if there is a correlation between the variable value and the residuals. Let's try it with `Age`:
#
# Here we see that indeed several of the points furthest from the Reference line are colored yellow, indicating an Age of 365 days. This shows that the model over-estimates the strength of the concrete when aged for much longer than average. We can see the same effect using a `scatter_3d` chart with `plotly`:
#
# We can take another view of the data by plotting the residual against the Age:
#
# If we look back at the `Concrete compressive strength` vs `Age` chart, we would see a highly nonlinear relationship between the two. This is backed up by the nonlinear shape of the residuals shown above. We will resolve this problem using more sophisticated methods very shortly.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    px = ctx.get('tensors', {}).get('px')
    residuals = ctx.get('tensors', {}).get('residuals')
    y = ctx.get('tensors', {}).get('y')
    y_pred = ctx.get('tensors', {}).get('y_pred')
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))

    # plot the results
    ax.plot(y, y_pred, '.', label='Data')

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
    im = ax.scatter(y, y_pred, s=16, c=data['Age (day)'], label='Data')
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

    from plotly import express as px

    px.scatter_3d(x=y, y=y_pred, z=data['Age (day)'], color=data['Age (day)'], width=800, height=600)

    age = data['Age (day)']
    fig, ax = plt.subplots()
    ax.plot(age, residuals, '.')
    ax.plot([age.min(), age.max()], [0, 0], 'k--')
    ax.set_xlabel('Observation')
    ax.set_ylabel('Residual')
    ctx['tensors']['age'] = age
    ctx['tensors']['min_max'] = min_max
run_module(ctx)

# %% [markdown]
# ## [Check your understanding]
#
# * Make a model for the Concrete Compressive Strength using a nonlinear regression model from `sklearn` **that was not covered here**.
# You can refer to the [documentation](https://scikit-learn.org/stable/supervised_learning.html) to find a suitable method.
# * Train the model on a suitable train/test split.
# * Evaluate the R-squared, RMSE, and MAE on train and test data.
# * Make a plot of the predicted vs observed data with train and test data as separate series.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Quantifying model performance

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Challenges with measuring residuals
#
# Remember that when we fit a linear regression, the residuals will sum to zero.
# This also means necessarily that the mean will be zero since the mean is just the sum normalized by the count:
#
# We can visualize the distribution of these residuals around zero by creating a histogram:
#
# From this historgram, you can see a generally Normal distribution around the zero mean.
# We need to measure the deviations instead of the central tendency.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    y = ctx.get('tensors', {}).get('y')
    y_pred = ctx.get('tensors', {}).get('y_pred')
    residual = y - y_pred

    print('sum: ', np.sum(residual))
    print('mean:', np.mean(residual))

    fig, ax = plt.subplots()
    _ = ax.hist(residual)
    ax.set_xlabel('Model residual (eV)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residuals from linear model of concrete strength')
    ctx['tensors']['residual'] = residual
run_module(ctx)

# %% [markdown]
# ## MSE, MAE, and regularization
#
# To get around this problem when evaluating residuals, we discussed several alternate metrics.
# These include the Mean Squared Error (MSE) and Root-MSE (RMSE), which are straightforward to compute:
#
# The RMSE has the same units as $y$, so this means the "typical" residual for the model is about 10 MPa.
#
# > Note that the RMSE formula is actually the same as the standard deviation of the residuals. The difference is just the intent behind the calculation (model error versus variation in a population).
#
# We can compare the RMSE to Mean Absolute Error (MAE):
#
# The MAE applies a power of 1 to the residuals while the RMSE applies a power of 2.
# You can observe that the RMSE leads to a larger value than MAE.
# These normalizations are called $L_1$ and $L_2$ **norms**. Let's make a histogram of the residuals to see what this means visually:
#
# Basically, the L1 residuals are more compact than the L2 residuals (less skewed).
# This means that when fitting the model, the residuals out on the tail will carry much more weight than those close to zero when using L2 regularization.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    residual = ctx.get('tensors', {}).get('residual')
    y = ctx.get('tensors', {}).get('y')
    y_model = ctx.get('tensors', {}).get('y_model')
    mse = np.mean(residual**2)
    rmse = np.sqrt(mse)

    print('MSE: ', mse)
    print('RMSE:', rmse)

    print(np.std(residual))

    mae = np.mean(np.abs(residual))
    print(f'MAE  = {mae}')

    res_l1 = np.abs(y - y_model)
    res_l2 = (y - y_model)**2

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    _ = ax.hist(res_l1)
    ax.set_xlabel('L1 residual')

    ax = axes[1]
    _ = ax.hist(res_l2)
    ax.set_xlabel('L2 residual')
    ctx['tensors']['mae'] = mae
    ctx['tensors']['mse'] = mse
    ctx['tensors']['res_l1'] = res_l1
    ctx['tensors']['res_l2'] = res_l2
    ctx['tensors']['rmse'] = rmse
run_module(ctx)

# %% [markdown]
# ## [Check your understanding]
#
# * Make a model for the Concrete Compressive Strength using a nonlinear regression model from `sklearn` **that was not covered here**.
# You can refer to the [documentation](https://scikit-learn.org/stable/supervised_learning.html) to find a suitable method.
# * Train the model on a suitable train/test split.
# * Evaluate the R-squared, RMSE, and MAE on train and test data.
# * Make a plot of the predicted vs observed data with train and test data as separate series.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Training versus Testing
#
# One of the big differences between ML and statistical modeling is the intended use. We have used linear regression to evaluate how variables are related to each other. With ML we often want to make *predictions*, or *extrapolations*. This requires a change to our workflow.
#
# In order to evaluate the performance of the model on unseen data (i.e., a prediction), we need to *hold out* some data to emulate the effect of the model encountering data it hasn't been fitted to. This will be called the **test** data.
# The data used to fit will be called the **training** data.
#
# As we start using more sophisticated models which are less grounded in physical reality and more on statistical trends, it is vitally important that we evaluate our models in this way to avoid *overfitting*.
# Remember that a model with enough degrees of freedom can perfectly fit any data, but it won't have any predictive power!

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Train/Test split
#
# We'll use the `sklearn.model_selection.train_test_split` function to split our data into a train and test set:
#
# Now we can evaluate the performance on both the `train` and `test` sets separately:
#
# Typically we expect our test performance to be lower than our train performance since it is new data that was not fit to, but it doesn't have to be. The fit to the training data is basically telling you the best case for your model -- you should not expect the performance to be higher on the test set since the model parameters were already optimized for the training set!

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    train_test_split = ctx.get('tensors', {}).get('train_test_split')
    from sklearn.model_selection import train_test_split

    x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
    y = data.loc[:, 'Concrete compressive strength(MPa, megapascals) ']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
    # I use fixed random_state to avoid changing the answer when re-running the code
    print(xtrain.shape, xtest.shape)

    # train on only the (Xtrain, ytrain) data!
    model = linear_model.LinearRegression().fit(xtrain, ytrain)

    y_pred = model.predict(xtrain)
    residuals = y_pred - ytrain

    r2 = 1 - np.var(residuals) / np.var(ytrain - ytrain.mean())

    rmse = np.sqrt(np.mean(residuals**2))

    print(f'train: Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

    y_pred = model.predict(xtest)
    residuals = y_pred - ytest

    r2 = 1 - np.var(residuals) / np.var(ytest - ytest.mean())

    rmse = np.sqrt(np.mean(residuals**2))

    print(f'test:  Rsq = {r2:.3f}, RMSE = {rmse:.3f}')
    ctx['model'] = model
    ctx['tensors']['residuals'] = residuals
    ctx['tensors']['rmse'] = rmse
    ctx['tensors']['x'] = x
    ctx['tensors']['y'] = y
    ctx['tensors']['y_pred'] = y_pred
run_module(ctx)

# %% [markdown]
# ## Investigating features
#
# One additional thing we can do is check which features dominate the response by looking at the coefficients, stored in the `coef_` attribute:
#
# Based on this, we see that `Superplasticizer` has the strongest effect per kg, then `Water` and `Cement`.
# `Age` is also near the top of the list, but it's measured in days, not kg, so it's hard to compare it to the others.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    model = ctx.get('model')
    xtrain = ctx.get('tensors', {}).get('xtrain')
    model.coef_

    for i, col in enumerate(xtrain.columns):
        print(f'{col:55s} {model.coef_[i]:10.4f}')
run_module(ctx)

# %% [markdown]
# # Nonlinear regression models
#
# As we discovered with our `Age` variable, sometimes we need to introduce nonlinear effects into our regression to get a good fit.
# * It's important to realize that **no linear model can ever reproduce a nonlinear generating function**.
# * he best we can hope for is to capture such a function in a very narrow region of space where it is locally linear.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    x = ctx.get('tensors', {}).get('x')
    xtest = ctx.get('tensors', {}).get('xtest')
    xtrain = ctx.get('tensors', {}).get('xtrain')
    y = ctx.get('tensors', {}).get('y')
    ytest = ctx.get('tensors', {}).get('ytest')
    ytrain = ctx.get('tensors', {}).get('ytrain')
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
    ctx['model'] = model
    ctx['tensors']['min_max'] = min_max
    ctx['tensors']['poly'] = poly
    ctx['tensors']['residuals'] = residuals
    ctx['tensors']['rmse'] = rmse
    ctx['tensors']['xptest'] = xptest
    ctx['tensors']['xptrain'] = xptrain
    ctx['tensors']['y_pred'] = y_pred
run_module(ctx)

# %% [markdown]
# ## Some convenience functions
#
# Let's set up some convenience functions so we don't have to repeat the same code all the time...

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    def evaluate_model(model, xtrain, xtest, ytrain, ytest):
        "Evaluate model performance on train and test set, then print the results."

        for name, x, y in [('train', xtrain, ytrain), ('test', xtest, ytest)]:

            y_pred = model.predict(x)
            residuals = y_pred - y

            r2 = 1 - np.var(residuals) / np.var(y - y.mean())

            rmse = np.sqrt(np.mean(residuals**2))

            print(f'{name:5s}: Rsq = {r2:.3f}, RMSE = {rmse:.3f}')


    def plot_model(model, xtrain, xtest, ytrain, ytest, title=None):
        "Create a parity plot using a trained model with train/test split."

        fig, ax = plt.subplots(figsize=(5, 5))

        # plot the results
        ax.plot(ytrain, model.predict(xtrain), '.', label='Training Data')
        ax.plot(ytest, model.predict(xtest), '.', label='Testing Data')

        # create reference line along y = x to show the desired behavior
        min_max = np.array([y.min(), y.max()])
        ax.plot(min_max, min_max, 'k--', label='Reference')
        ax.set_aspect('equal')  # very helpful to show y = x relationship

        # add labels and legend
        ax.set_xlabel('Observation')
        ax.set_ylabel('Prediction')
        ax.legend()

        if title is not None:
            ax.set_title(title)
run_module(ctx)

# %% [markdown]
# ## K-Nearest Neighbors
#
# K-Nearest Neighbors, abbreviated KNN, or called K Neighbors, is a voting algorithm. Essentially we just look at the values of the $k$ nearest points to the unlabeled observation in question $X_i$ in feature space and use the neighbors' labels $\{ y \}_k$ to predict $y_i$. It would look something like this:
#
# <img src="../lectures/assets/lecture03_knn_concept.jpg" alt="K-Nearest Neighbors concept diagram" height=300>
#
# This is a relatively simple idea but it can work quite well for some problems, especially with a lot of training data.
# In some sense it is just an interpolation scheme, but it may work in very high dimensions.
#
# > Note: in contrast to linear regression (a **parametric** or **model-based** scheme), this is an **instance-based** learning scheme.
# Model-based learning involves fitting parameters that can be used to predict new outcomes, while instance-based learning involves comparing new observations to previous ones.
#
# This model performs slightly worse than the polynomial `LinearRegression` on the test set, though in training it appears reasonable. This is a clear case of overfitting, which demonstrates the need for train/test split.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    evaluate_model = ctx.get('tensors', {}).get('evaluate_model')
    plot_model = ctx.get('tensors', {}).get('plot_model')
    xtest = ctx.get('tensors', {}).get('xtest')
    xtrain = ctx.get('tensors', {}).get('xtrain')
    ytest = ctx.get('tensors', {}).get('ytest')
    ytrain = ctx.get('tensors', {}).get('ytrain')
    from sklearn import neighbors

    model = neighbors.KNeighborsRegressor()
    model = model.fit(xtrain, ytrain)

    from sklearn import neighbors

    model = neighbors.KNeighborsRegressor().fit(xtrain, ytrain)

    evaluate_model(model, xtrain, xtest, ytrain, ytest)

    plot_model(model, xtrain, xtest, ytrain, ytest)
    ctx['model'] = model
run_module(ctx)

# %% [markdown]
# ## Decision Trees
#
# Decision Trees are a nonlinear method that identify outcomes based on a separation of the features into discrete domains. The domains are chosen in a hierarchical manner which yields a tree structure, like so:
#
# <img src="../lectures/assets/lecture03_decision_tree.jpg" alt="Visual representation of a decision tree structure" width=600>
#
# In essence, each split while going down the tree defines a mapping of the data from input to output. Building a very deep tree can result in a pretty complex, nonlinear mapping between input and output.
#
# You can see that the tree completely fits the training data here, but is overfitted and does not perform perfectly on the test set. This is because the tree has unlimited depth by default.
# In practice we should limit the depth to increase transferability.
#
# We can provide the `max_depth` keyword argument to the `DecisionTreeRegressor` constructor.
# This is an example of a **hyperparameter** -- a model parameter that is not fitted during the course of learning, but rather chosen outside the learning procedure.
#
# Now we see a more comparable performance on train and test sets, though the overall performance went down on test data. The predictions look a bit odd, not like our other models. Let's investigate why this might be the case.
#
# We can visualize the decision tree itself using `plot_tree`:
#
# This is technically *traceable*, but maybe not *interpretable*. As in, we can see what happened, but probably not understand why. For instance, if we follow the leftmost branch of the tree, we see that at the very end a decision is made which gives one of two values depending on a simple binary cutoff. This sort of decisionmaking is not how we would like to interpret the data since the relationships are probably more like "the `Concrete compressive strength` increases at a rate of $m$ as component $X_i$ increases".
# You can see the striation resulting from these binary decisions in the output of the chart above.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    evaluate_model = ctx.get('tensors', {}).get('evaluate_model')
    plot_model = ctx.get('tensors', {}).get('plot_model')
    xtest = ctx.get('tensors', {}).get('xtest')
    xtrain = ctx.get('tensors', {}).get('xtrain')
    ytest = ctx.get('tensors', {}).get('ytest')
    ytrain = ctx.get('tensors', {}).get('ytrain')
    from sklearn import tree

    model = tree.DecisionTreeRegressor().fit(xtrain, ytrain)

    evaluate_model(model, xtrain, xtest, ytrain, ytest)

    plot_model(model, xtrain, xtest, ytrain, ytest)

    model = tree.DecisionTreeRegressor(max_depth=5).fit(xtrain, ytrain)

    evaluate_model(model, xtrain, xtest, ytrain, ytest)

    plot_model(model, xtrain, xtest, ytrain, ytest)

    fig, ax = plt.subplots(figsize=(24, 8))
    _ = tree.plot_tree(model, ax=ax, fontsize=10, label='root',
                       impurity=False, precision=1, proportion=True)
    ctx['model'] = model
run_module(ctx)

# %% [markdown]
# ## Random Forest
#
# In practical applications, Decision Trees tend to be somewhat weak. There may be too many features to evaluate, or too much variation to provide a strong prediction from a single path through the tree. Decision Trees can be made more robust through aggregation of multiple trees into an **ensemble**. One of the most common forms of ensemble learning is the Random Forest, so named because it uses a collection of Decision Trees (i.e., a Forest):
#
# <img src="../lectures/assets/lecture03_random_forest.jpg" alt="Schematic of a Random Forest ensemble learning architecture" width=600>
#
# Using this collection of many trees with consensus voting can give much stronger results than the single Decision Tree. This method can actually provide very complex mappings and as a result there is a big risk of overfitting. To see how this might happen, you can imagine creating leaf nodes for each individual outcome -- this might be similar to K-Neighbors with only one neighbor.
#
# This model has also suffers from overfitting, like the Decision Trees (since the Forest is made up of Trees).
# From the training data, you would think it predicts the outcome nearly perfectly. However, we get 2-3x increase in RMSE in testing. This is stil the best model we have so far, but it's very important not to believe the training result.
#
# We can greatly reduce overfitting by limiting the depth of the decision trees using the `max_depth` keyword argument:
#
# This model still outperforms the polynomial LinearRegression while not suffering greatly from overfitting. It is substantially more robust than a single Decision Tree. However, it is no longer even traceable, let alone interpretable -- there is no way we can parse the decisions made by 100+ individual trees!

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    evaluate_model = ctx.get('tensors', {}).get('evaluate_model')
    plot_model = ctx.get('tensors', {}).get('plot_model')
    xtest = ctx.get('tensors', {}).get('xtest')
    xtrain = ctx.get('tensors', {}).get('xtrain')
    ytest = ctx.get('tensors', {}).get('ytest')
    ytrain = ctx.get('tensors', {}).get('ytrain')
    from sklearn import ensemble

    model = ensemble.RandomForestRegressor().fit(xtrain, ytrain)

    evaluate_model(model, xtrain, xtest, ytrain, ytest)

    plot_model(model, xtrain, xtest, ytrain, ytest)

    model = ensemble.RandomForestRegressor(max_depth=5).fit(xtrain, ytrain)

    evaluate_model(model, xtrain, xtest, ytrain, ytest)

    plot_model(model, xtrain, xtest, ytrain, ytest)
    ctx['model'] = model
run_module(ctx)

# %% [markdown]
# ## Neural Networks
#
# Neural Networks are perhaps the most famous ML models. They have achieved exceptional results in computer vision and natural language tasks, which have historically been incredibly challenging problems. The model name comes from its similarity to the architecture of a neuron in your brain.
# Here is a schematic of how it works mathematically:
#
# <img src="../lectures/assets/lecture03_neuron.jpg" alt="Mathematical schematic of a single artificial neuron" width=600>
#
# Basically the model alternates between a linear algebra block (which is fast and easy to compute) and a nonlinear function (the activation). By stacking multiple of these "layers," it is mathematically provable that the network can approximate any function. This has led to daring attempts by Deep Learning experts to learn functions with no conceivable alternative functional form, like human face generation (e.g., DeepFakes).
#
# Here we use the simplest version of a NN called a Multi-Layer Perceptron. It is basically a very shallow NN.
#
# > We call this **shallow** learning because there is only one layer (few parameters). Many layers of NN becomes a **deep** network. Deep networks are capable of expressing more complex relationships between variables.
#
# > **Practical Note:** Neural networks are sensitive to the scale of input features. In real-world applications, it is standard practice to scale features (e.g., using `StandardScaler`) before training. We will explore this in detail later.
#
# > We will spend a lot of time with deep neural networks later in the course, using the `pytorch` library.
#
# We see that it does a little better than the LinearRegression, depending on the random seed. It is fairly consistent between train and test performance.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    evaluate_model = ctx.get('tensors', {}).get('evaluate_model')
    neural_network = ctx.get('tensors', {}).get('neural_network')
    plot_model = ctx.get('tensors', {}).get('plot_model')
    xtest = ctx.get('tensors', {}).get('xtest')
    xtrain = ctx.get('tensors', {}).get('xtrain')
    ytest = ctx.get('tensors', {}).get('ytest')
    ytrain = ctx.get('tensors', {}).get('ytrain')
    from sklearn import neural_network

    # we increase the max iterations to ensure convergence
    model = neural_network.MLPRegressor(max_iter=1000).fit(xtrain, ytrain)

    evaluate_model(model, xtrain, xtest, ytrain, ytest)

    plot_model(model, xtrain, xtest, ytrain, ytest)
    ctx['model'] = model
run_module(ctx)

# %% [markdown]
# ## [Check your understanding]
#
# * Make a model for the Concrete Compressive Strength using a nonlinear regression model from `sklearn` **that was not covered here**.
# You can refer to the [documentation](https://scikit-learn.org/stable/supervised_learning.html) to find a suitable method.
# * Train the model on a suitable train/test split.
# * Evaluate the R-squared, RMSE, and MAE on train and test data.
# * Make a plot of the predicted vs observed data with train and test data as separate series.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)
