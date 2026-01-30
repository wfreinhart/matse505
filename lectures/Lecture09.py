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
# > Reminder: detailed presentation plan due 3/14 for discussion.
# Presentations should be 8 minutes + 2 minutes for Q&A.
# Detailed instructions will be posted to Canvas soon.

# %% [markdown]
# Today's topics:
# * Hyperparameter tuning
# * Cross-fold validation
# * Flavors of cross-validation

# %% [markdown]
# ## Revisiting nonlinear regression
#
# Let's pick up where we left off with nonlinear regression: using `sklearn` models to fit a multivariate regression problem for `Concrete compressive strength`.

# %%
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
data

# %% [markdown]
# Split the dataset into train and test sets:

# %%
from sklearn import model_selection

x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
y = data.loc[:, 'Concrete compressive strength(MPa, megapascals) ']

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, random_state=0)
print(xtrain.shape, xtest.shape)

# %% [markdown]
# # Hyperparameters
#
# To reiterate:
# Hyperparameters are the settings of a model that are not fitted.
# The "hyper" is in contrast to the "regular" parameters of the model (which are fitted).
# These are things like the number of neighbors to use in K-Neighbors, maximum tree depth for tree models, and the max iterations in Neural Networks.
#
# So far we have mostly been using the default hyperparameters of the models from `sklearn`, with a few exceptions.
# However, there is nothing special about the default parameters and in most cases we will want to select different parameters.
# Here we'll discuss how to select optimal hyperparameters.

# %% [markdown]
# ## Evaluating the effect of hyperparameters
#
# Let's start by just looking at how changing the hyperparameters influences model performance in an example: K-Neighbors.

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


# %% [markdown]
# Let's push some of these operations into functions to make our life easier going forward.

# %%
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


# %% [markdown]
# Now we'll repeat the analysis above with our new convenience functions:

# %%
train_and_report_performance(neighbors.KNeighborsRegressor(),
                             xtrain, ytrain, xtest, ytest)

# %% [markdown]
# Let's finally get to changing the hyperparameters.
# You'll see why we spent the time on that detour shortly...
#
# The two key hyperparameters for K-Neighbors are `n_neighbors` and `weights`.
# They default to `5` and `uniform` respectively.
# Let's try doubling to `n_neighbors=10` and changing to `weights=distance`:

# %%
train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=10),
                             xtrain, ytrain, xtest, ytest)

# %% [markdown]
# OK, so that made things slightly worse in both training and testing.
# Let's try the other one:

# %%
train_and_report_performance(neighbors.KNeighborsRegressor(weights='distance'),
                             xtrain, ytrain, xtest, ytest)

# %% [markdown]
# This improves the test performance a good amount but substantially overfits to the training data!
#
# Of course we could also try doing both together:

# %%
train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance'),
                             xtrain, ytrain, xtest, ytest)

# %% [markdown]
# Same problem with overfitting but it gives the best test performance so far.

# %% [markdown]
# ## A naiive approach to hyperparameter tuning
#
# Now that we have things well under control, let's just try all the possible values of `n_neighbors`!

# %%
for k in range(1, 20):
    print(f'k = {k:2d}: ', end='')
    train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance'),
                             xtrain, ytrain, xtest, ytest)

# %% [markdown]
# We can see from this that there is a non-monotonic relationship with `n_neighbors` and an optimal value lies in the middle of the range.
# So is that all there is to it?

# %% [markdown]
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


# %% [markdown]
# Here we see that the performance on truly unseen data is not as good as we hoped based on the validation set -- note that our "test" result printed by `train_and_report_performance` is actually the *validation* set.
# We should update our convenience function:

# %%
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


# %% [markdown]
# Here's the new function in action:

# %%
model = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
train_and_report_performance(model, x_train, y_train, x_test, y_test, x_val, y_val)

# %% [markdown]
# Now we can deploy it in our hyperparameter tuning:

# %%
for k in range(1, 20):
    print(f'k = {k:2d}: ', end='')
    train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance'),
                                 x_train, y_train, x_test, y_test, x_val, y_val)

# %% [markdown]
# The result is much less clear than before.
# We can clearly see that the result with the lowest validation performance ($k = 9$) is not the one with the lowest test performance ($k = 6$).
# This illustrates a similar problem as with overfitting -- you can't trust the performance on data included in the workflow when deploying on unseen data!
#
# In practice, we do have to choose a single set of hyperparameters to use, so we would select $k=9$ and report a test RMSE of 8.230 even though there are other options that would give a lower test RMSE.

# %% [markdown]
# ## Grid search
#
# We typically have more than one hyperparameter to tune.
# In fact, we already dealt with 2 implicitly by ignoring the `weights` option of `KNeighborsRegressor`.
# Now we'll deal with this issue directly via "grid search":
#
# <img src="../lectures/assets/grid_search_diagram.jpg" width=500 alt="Schematic of Grid Search: systematically testing combinations of hyperparameters across a grid">
#
# Here the "grid" just means looping over all combinations of different parameters and trying them.
# The contours represent the value of the RMSE from before.
# We'll see how this looks in just a minute.
#
# For now, the easiest way to implement grid search is to use nested `for` loops:

# %%
results = []
for k in range(2, 12):
    for w in ['uniform', 'distance']:
        print(f'Evaluating ({k}, {w})...')
        # set up the model
        model = neighbors.KNeighborsRegressor(n_neighbors=k, weights=w)
        model.fit(x_train, y_train)
        # evaluate rmse on all the splits
        train = calc_rmse(model, x_train, y_train)
        val = calc_rmse(model, x_val, y_val)
        test = calc_rmse(model, x_test, y_test)
        # save results
        results.append( [k, w, train, val, test] )

# %% [markdown]
# Based on the stored results, we can visualize the grid of possible models:

# %%
from plotly import express as px

# create a DataFrame with results
df = pd.DataFrame(results, columns=['k', 'w', 'Train', 'Validation', 'Test'])

# make a scatter plot of the grid
px.scatter(df, x='k', y='w', color='Validation')

# %% [markdown]
# This shows the same result from above, that optimal validation performance is achieve with `weights=distance` and $k = 9$.

# %% [markdown]
# ## In more dimensions
#
# Many models have more than two hyperparameters.
# The grid search can easily be extended to additional dimensions using nested `for` loops.
# For instance, consider the `max_depth`, `n_estimators`, and `min_samples_split` hyperparameters of the `RandomForestRegressor`:

# %%
from sklearn import ensemble

results = []
for md in np.arange(10, 31, 10):
    for ne in np.arange(20, 61, 20):
        for mss in np.arange(2, 5):
            print(f'Evaluating ({md}, {ne}, {mss})...')
            model = ensemble.RandomForestRegressor(max_depth=md, n_estimators=ne, min_samples_split=mss, random_state=0)
            model.fit(x_train, y_train)
            # evaluate rmse on all the splits
            train = calc_rmse(model, x_train, y_train)
            val = calc_rmse(model, x_val, y_val)
            test = calc_rmse(model, x_test, y_test)
            # save results
            results.append( [md, ne, mss, train, val, test] )

# %%
df = pd.DataFrame(results, columns=['md', 'ne', 'mss', 'Train', 'Validation', 'Test'])
px.scatter_3d(df, x='md', y='ne', z='mss', color='Validation')

# %% [markdown]
# Note once again that even though the best validation RMSE is `4.84` at `(10, 40, 4)`, the test RMSE is `5.19` here and as low as `5.08` elsewhere (with validation RMSE slightly higher at `4.89`):

# %%
df

# %% [markdown]
# ## [Check your understanding]
#
# Find the best set of hyperparameters for the decision tree based on validation RMSE using grid search.
# Does it outperform the other options on the test set?

# %%

# %% [markdown]
# # Cross-fold validation
#
# How do we deal with the issue of the validation set being unreliable?
# We can't keep taking data out of the training set or we won't have any left.

# %% [markdown]
# ## In theory
#
# The solution is to use "cross-fold validation."
# In this scheme, we separate the data into some $k$ number of "folds" and then train $k$ different models using different sub-splits:
#
# <img src="../lectures/assets/hyperparameter_validation_set.jpg" width=600 alt="Diagram of the Train, Validation, and Test set split strategy">
#
# This has the great advantage of not needing to remove any more data from the training set while averaging out some of the variance in the test performance.

# %% [markdown]
# ## In practice
#
# We can implement cross-fold validation using `model_selection.KFold`:

# %%
from sklearn import model_selection

# set up the splitter
folds = model_selection.KFold(n_splits=5)

# run through each subset
k = 0
results = []
for train_index, test_index in folds.split(x):

    # define the train / test split
    # each fold gets its own split!
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # train and evaluate the model
    model = ensemble.RandomForestRegressor().fit(x_train, y_train)
    rmse_train = calc_rmse(model, x_train, y_train)
    rmse_test = calc_rmse(model, x_test, y_test)

    k += 1
    print(f'fold k = {k} RMSE: Train = {rmse_train:.3f}; Test = {rmse_test:.3f}')
    results.append([rmse_train, rmse_test])

# %% [markdown]
# These results are all over the place.
# We can look at the mean and standard deviation to try and get a clearer picture:

# %%
mu = np.mean(results, axis=0)
sigma = np.std(results, axis=0)
for i, s in enumerate(['Train', 'Test ']):
    print(f'{s} average: {mu[i]:.3f} +/- {sigma[i]:.3f}')

# %% [markdown]
# We see here that the average test RMSE is both substantially higher than the training RMSE, and also highly variable.

# %% [markdown]
# ## Holding out a test set
#
# What if we want to do cross-fold-validation-based hyperparameter tuning?
# Of course we will need to hold out a test set as well.
# We can do it as shown in this diagram:
#
# <img src="../lectures/assets/grid_search_cv.jpg" width=500 alt="Workflow combining Grid Search with Cross-Validation for robust hyperparameter optimization">
#
# All the models trained on the different fold-specific train/test splits can then be tested using the held-out test set.

# %%
# first split out 20% for testing at the end
x_kf, x_test, y_kf, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True)

# now do cross-fold validation on the remaining data
folds = model_selection.KFold(n_splits=4, shuffle=True)
results = []
k = 0
for train_index, val_index in folds.split(x_kf):
    # define the train / test split for this fold
    x_train, x_val = x_kf.iloc[train_index], x_kf.iloc[val_index]
    y_train, y_val = y_kf.iloc[train_index], y_kf.iloc[val_index]

    # train and evaluate the model
    model = ensemble.RandomForestRegressor().fit(x_train, y_train)
    rmse_train = calc_rmse(model, x_train, y_train)
    rmse_val = calc_rmse(model, x_val, y_val)
    rmse_test = calc_rmse(model, x_test, y_test)

    results.append( [rmse_train, rmse_val, rmse_test] )

    k += 1
    print(f'fold k = {k} RMSE: Train = {rmse_train:.3f}; Val = {rmse_val:.3f}; Test = {rmse_test:.3f}')

# %% [markdown]
# Again, we can aggregate these results into a mean and standard deviation:

# %%
mu = np.mean(results, axis=0)
sigma = np.std(results, axis=0)
for i, kind in enumerate(['Train', 'Val  ', 'Test ']):
    print(f'{kind} RMSE = {mu[i]:.3f} +/- {sigma[i]:.3f}')


# %% [markdown]
# From the aggregated results we actually see that the performance is slightly worse in testing compared to validation, but the standard deviations are relatively low so we can have some confidence about the values.
#
# We can use the stabilization from cross-fold validation on the hyperparameter optimization problem from before:

# %%
def cv_rmse(model, X_train, y_train, X_test, y_test, X_val, y_val):
    # now do cross-fold validation on the remaining data
    folds = model_selection.KFold(n_splits=4, shuffle=True)
    results = []
    k = 0
    for train_index, val_index in folds.split(x_kf):
        # define the train / test split for this fold
        x_train, x_val = x_kf.iloc[train_index], x_kf.iloc[val_index]
        y_train, y_val = y_kf.iloc[train_index], y_kf.iloc[val_index]

        # train and evaluate the model
        model.fit(x_train, y_train)
        rmse_train = calc_rmse(model, x_train, y_train)
        rmse_val = calc_rmse(model, x_val, y_val)
        rmse_test = calc_rmse(model, x_test, y_test)

        results.append( [rmse_train, rmse_val, rmse_test] )

    return results


# %%
all_mu = []
all_sigma = []
for k in range(1, 20):
    print(f'k = {k:2d}: ', end='')
    model = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance')
    results = cv_rmse(model, x_train, y_train, x_test, y_test, x_val, y_val)
    mu = np.mean(results, axis=0)
    sigma = np.std(results, axis=0)
    for i, kind in enumerate(['Train', 'Val', 'Test']):
        print(f'{kind} = {mu[i]:.2f} +/- {sigma[i]:.2f}', end='; ')
    print()
    all_mu.append(mu)  # save these for later
    all_sigma.append(sigma)

# %%
from matplotlib import pyplot as plt

# convert to numpy arrays
mu = np.array(all_mu)
sigma = np.array(all_sigma)
k = np.arange(1, 20)
# set up the plot
fig, ax = plt.subplots()
ax.bar(k, mu[:, 1], yerr=sigma[:, 1], label='Val', width=0.5)
ax.bar(k, mu[:, 2], label='Test', width=0.5, align='edge', zorder=-1)
# labels and legend
ax.set_xlabel('$k$')
ax.set_ylabel('RMSE')
ax.legend()
# zoom in
ax.set_ylim(7.5, 10)

# %% [markdown]
# The chart shows that although validation does not always match test performance, the error bars provided by cross-fold validation do typically encompass the average test performance.

# %% [markdown]
# # Flavors of cross-validation
#
# There are many more sophisticated schemes that can be applied when the situation calls for it.
# Let's review some of these variations.

# %% [markdown]
# ## K-Fold
#
# The basic implementation of cross-fold validation is the k-fold scheme.
# Here we divide the data directly into $k$ number of evenly sized folds like so:
#
# <img src="../lectures/assets/kfold_cv.jpg" width=600 alt="Schematic of K-Fold Cross-Validation showing the data split into k folds for iterative training and testing">
#
# What is the problem with this scheme?
#
# Let's explore this on the cement dataset:

# %%
# define the fold splitting strategy
folds = model_selection.KFold(n_splits=5)

in_fold = np.zeros([x.shape[0], folds.n_splits])
this_fold = np.zeros(x.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

# %% [markdown]
# Here we see that the `Age (day)` variable has high and low values distributed throughout the folds.
# What if the data were entered in ascending order (such as in a spreadsheet or lab notebook while the cement was curing)?

# %%
# create a sorted dataset to illustrate the pathological cases
x_sort = x.sort_values(by='Age (day)')

# define the fold splitting strategy
folds = model_selection.KFold(n_splits=5)

in_fold = np.zeros([x_sort.shape[0], folds.n_splits])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

# %% [markdown]
# ## [Check your understanding]
#
# Calculate validation RMSE for regression models trained on each fold using each of the `x` and `x_sort` dataset above.
# How much worse is the result when using the sorted data?
# Why?

# %%
# define the fold splitting strategy
folds = model_selection.KFold(n_splits=5)

order = np.arange(x.shape[0])
print('unsorted:')

results = []
for k, (train_index, test_index) in enumerate(folds.split(np.arange(x_sort.shape[0]))):
    # define the train / test split for this fold
    x_train, x_val = x.iloc[order].iloc[train_index], x.iloc[order].iloc[val_index]
    y_train, y_val = y[order].iloc[train_index], y[order].iloc[val_index]

    print(x_train['Age (day)'].min(), x_train['Age (day)'].max())

    # train and evaluate the model
    model = ensemble.RandomForestRegressor().fit(x_train, y_train)
    rmse_train = calc_rmse(model, x_train, y_train)
    rmse_val = calc_rmse(model, x_val, y_val)
    rmse_test = calc_rmse(model, x_test, y_test)

    results.append( [rmse_train, rmse_val, rmse_test] )

    k += 1
    print(f'fold k = {k} RMSE: Train = {rmse_train:.3f}; Val = {rmse_val:.3f}; Test = {rmse_test:.3f}')

# %%
# define the fold splitting strategy
folds = model_selection.KFold(n_splits=5)

order = np.argsort(x['Age (day)'].values)
print('sorted:')

results = []
for k, (train_index, test_index) in enumerate(folds.split(np.arange(x_sort.shape[0]))):
    # define the train / test split for this fold
    x_train, x_val = x.iloc[order].iloc[train_index], x.iloc[order].iloc[val_index]
    y_train, y_val = y[order].iloc[train_index], y[order].iloc[val_index]

    print(x_train['Age (day)'].min(), x_train['Age (day)'].max())

    # train and evaluate the model
    model = ensemble.RandomForestRegressor().fit(x_train, y_train)
    rmse_train = calc_rmse(model, x_train, y_train)
    rmse_val = calc_rmse(model, x_val, y_val)
    rmse_test = calc_rmse(model, x_test, y_test)

    results.append( [rmse_train, rmse_val, rmse_test] )

    k += 1
    print(f'fold k = {k} RMSE: Train = {rmse_train:.3f}; Val = {rmse_val:.3f}; Test = {rmse_test:.3f}')

# %% [markdown]
# ## Shuffle and split
#
# We can get around some of the limitations of the above using `shuffle=True` to get something more like this:
#
# <img src="../lectures/assets/shuffle_split_cv.jpg" width=600 alt="Illustration of the Shuffle-Split cross-validation strategy">
#
# Let's see how this applies to the cement data:

# %%
# define the fold splitting strategy
folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

in_fold = np.zeros([x_sort.shape[0], folds.n_splits])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

# %% [markdown]
# ## Leave-one-out
#
# An extreme version of the cross-fold validation is to use a number of folds equal to the number of observations.
# This effectively uses each and every sample as a test set.
# It would look like so:
#
# <img src="../lectures/assets/cross_validation_diagram.jpg" width=600 alt="Generalized flowchart of the cross-validation process">
#
# What are the downsides of this scheme?
#
# Let's see what this looks like on our cement data:

# %%
# define the fold splitting strategy
folds = model_selection.LeaveOneOut()

in_fold = np.zeros([x_sort.shape[0], folds.get_n_splits(x_sort)])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

# %% [markdown]
# ## Stratified CV
#
# We may want to take care to get a similar distribution in each fold.
# Here's what it looks like:
#
# <img src="../lectures/assets/stratified_cv.jpg" width=600 alt="Schematic of Stratified K-Fold CV ensuring each fold has the same class distribution as the original data">
#
# This can improve model performance but is not always a good assumption.
# Why would this be a problem?
#
# Let's see what it looks like on the cement data.
#

# %%
# define the fold splitting strategy
folds = model_selection.StratifiedKFold(n_splits=5)

in_fold = np.zeros([x_sort.shape[0], folds.n_splits])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort, x_sort['Age (day)'])):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

# %% [markdown]
# Note this error message:
#
# `The least populated class in y has only 2 members, which is less than n_splits=5.`
#
# This is an indication that the `StratifiedKFold` object is trying to split the observations according to unique `Age (day)` values, but because those values are continuous, it is having trouble performing even splits.
# If we want to use stratification for continuous values, we should first transform the values to discrete bins.
# We did this before using the `KBinsDiscretizer`:
#

# %%
from sklearn import preprocessing

# create the discretizer object
discretizer = preprocessing.KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
classes = discretizer.fit_transform(x_sort['Age (day)'].values.reshape(-1, 1))

fig, ax = plt.subplots()
_ = ax.plot(x_sort['Age (day)'], classes, '.')
_ = ax.set_xlabel('Age (day)')
_ = ax.set_ylabel('Discrete bin')

# %% [markdown]
# Now we try again to implement the `StratifiedKFold` but reference the `classes` instead of the `Age (day)`:

# %%
# define the fold splitting strategy
folds = model_selection.StratifiedKFold(n_splits=5)

in_fold = np.zeros([x_sort.shape[0], folds.n_splits])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort, classes)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

# %% [markdown]
# As you can see, the high and low values are more evenly distributed between different folds.
# This will reduce variance between *replicas* (repeated splits).

# %% [markdown]
# ## Group K Fold
#
# Sometimes there are additional identifiers in the data beyond discrete or continuous labels.
# For instance, imagine making multiple measurements on different physical material samples.
#
# <img src="../lectures/assets/group_cv_concept.jpg" width=600 alt="Visual explanation of Group-based Cross-Validation where samples are grouped by a shared identifier like session or patient">
#
# Some of these tensile bars might have been made on different days, or different machines, or tested by different personnel.
# It may be important to evaluate the effect of these variations through "grouping" the samples into specific folds.
# In this case, the group is extra information not already included in the model.
#
# Here is a schematic illustrating this concept:
#
# <img src="../lectures/assets/group_kfold.jpg" width=600 alt="Schematic of Group K-Fold Cross-Validation ensuring no group is split across training and validation folds">
#
# Note that unlike stratification, the classes are not balanced here, as only the groups are evaluated to create the folds.
# Of course there are "group" variants of all the other schemes, including `StratifiedGroupKFold`, `LeaveOneGroupOut`, etc.
#
# Let's try using `GroupKFold` CV to evaluate different groups in the cement dataset.
# Imagine there are discrete groups according to the `Superplasticizer` content:

# %%
discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

y = x_sort['Superplasticizer (component 5)(kg in a m^3 mixture)']
groups = discretizer.fit_transform(y.values.reshape(-1, 1))

fig, ax = plt.subplots()
_ = ax.plot(y, groups, '.')
_ = ax.set_xlabel(y.name)
_ = ax.set_ylabel('Discrete bin')

# %% [markdown]
# Now we'll use these bins to create group-based folds:

# %%
# define the fold splitting strategy
folds = model_selection.GroupKFold(n_splits=5)

in_group = np.zeros([x_sort.shape[0], discretizer.n_bins])
for i, g in enumerate(groups.astype(int)):
    in_group[i, g] = 1

in_fold = np.zeros([x_sort.shape[0], folds.get_n_splits(x_sort, classes, groups=groups)])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort, classes, groups=groups)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(groups.flatten())
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]
in_group = in_group[order]

fig, axes = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_group.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Group')
ax = axes[2]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

# %% [markdown]
# What happened to our uniform group size and `Age` distribution?
# Creating folds by group naturally leads to imbalance in the size and data distribution in each fold as it becomes difficult to assign equally sized groups.
# While it will likely lead to worse model performance, it is also more realistic!
# How often do you get to decide what your new data will look like?

# %% [markdown]
# ## The problem of extrapolation
#
# The schemes we have employed so far are a little idealistic in that they assume the same data distribution in the test set as seen during training.
# Actually this can be unusual in the physical sciences and engineering since we are often interested in making discoveries (i.e., exploring new data domains) with our models.
#
# Here's a schematic illustrating the difference between **interpolation** and **extrapolation**:
#
# <img src="../lectures/assets/interpolation_vs_extrapolation.jpg" width=600 alt="Plot comparing interpolation (making predictions within the range of training data) versus extrapolation (predicting outside the range)">
#
# How can we address this?
# By using **groups** to investigate how the models perform on data towards the center of the distribution versus the edges!
#
#

# %% [markdown]
# ## [Check your understanding]
#
# Apply Group K-Fold CV to split the data by `Age (day)` feature groups.
# This will allow you to test the performance when exposed to new values not seen in training.
# Evaluate the model validation performance on interpolation and extrapolation tasks.

# %%

# %% [markdown]
# > Note: the [`scikit-learn` documentation](https://scikit-learn.org/stable/modules/cross_validation.html) has a great resource explaning different kinds of CV.
