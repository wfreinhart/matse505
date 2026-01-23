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
# Today's topics:
# * Feature importance
#   * Concept
#   * Permutation
#   * Drop column
#   * Regressors versus classifiers

# %% [markdown]
# # Feature importance
#
# Now that we have trained several different kinds of models and found some that work well, we might want to understand how they work.
# A common scheme for interrogating a trained model is "feature importance" -- how important each input feature (variable) is in the final output.

# %% [markdown]
# Let's revist the mechanical properties of concrete dataset:

# %%
import pandas as pd
import numpy as np

data = pd.read_csv('../datasets/concrete.csv')
data

# %% [markdown]
# Let's say we want to understand how important each of the components is in determining the final strength of the material.
# We previously performed a multiple linear regression on this data and evaluated the coefficients as a proxy for feature importance.
# Let's repeat this now:

# %%
from sklearn import linear_model

x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
y = data.loc[:, 'Concrete compressive strength(MPa, megapascals) ']

model = linear_model.LinearRegression().fit(x, y)
print( model.score(x, y) )

# %% [markdown]
# We can interrogate the `coef_` attribute of the fitted `LinearRegression` object to find out the linear coefficients in front of each independent variable:

# %%
print('linear model looks like:')
for i in np.argsort(np.abs(model.coef_)):
    print(f'{model.coef_[i]:6.3f} * {x.columns[i]}')


# %% [markdown]
# We discussed three notable things about this result:
# * Water is the only component that correlates to a reduced strength
# * Superplasticizer is an additive that should have an outsized effect on strength when measured in kg/m^3, so it is reassuring to see it with the highest coefficient
# * Age is measured in days, so its effect can't be directly compared to the others

# %% [markdown]
# ## Permutation importance
#
# There are two outstanding questions from the analysis above:
# 1. How to compare variables with different units
# 2. How to evaluate non-linear models
#
# We can address these using other strategies for measuring importance aside from linear coefficients.
# The first method will be called permutation importance.
# We will first train a model, then shuffle (permute) each column one at a time to see how badly the predictions suffer.
# This is basically checking how much the model relies on each feature to make the predictions.
#
# The scheme is illustrated schematically below:
#
# <img src="./assets/permutation_importance_diagram.jpg" width=600 alt="Visual explanation of the permutation importance algorithm: shuffling a single feature to measure its impact on model error">

# %%
def calc_rmse(y, y_pred):
    residuals = y - y_pred
    return np.sqrt(np.mean(residuals**2))

model = linear_model.LinearRegression().fit(x, y)
baseline = calc_rmse(y, model.predict(x))  # first score the baseline model with all columns
print(f'baseline rmse is {baseline}')

permuted = np.zeros_like(x.columns)  # create empty array to store values
for i, col in enumerate(x.columns):
    x_permuted = x.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(x[col])
    permuted[i] = calc_rmse(y, model.predict(x_permuted))  # score on the permuted column
print('rmse on permuted columns is:', permuted)

# %% [markdown]
# We can clean up this result to view it more clearly as a sorted `DataFrame`:

# %%
result = pd.DataFrame({'Feature': x.columns, 'Coefficient': model.coef_,
                       'Permutation Importance': permuted - baseline})
result.sort_values('Permutation Importance')

# %% [markdown]
# This shows a number of interesting results:
# * Cement is by far the most important despite having an average coefficient
# * Superplasticizer has a small importance despite having the largest coefficient
# * Age is more important than it appeared from the coefficient
#
# The data above might be clearer when visualized as a bar chart:

# %%
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.bar(x.columns, permuted, label='Permuted')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend()

# %% [markdown]
# Remember that higher RMSE indicates a worse result, so `Cement` is the most impactful and `Coarse Aggregate` and `Fine Aggregate` are close to tied for least impactful.
#
# We can add the coefficients to the same chart to compare them head-to-head:

# %%
axR = ax.twinx()  # set up a second y axis on the same x axis (different scale)
axR.plot(np.arange(model.coef_.shape[0]), model.coef_, label='Coefficient',
         marker='s', linestyle='-', color='tab:orange', zorder=2)
axR.set_ylabel('Coefficient')
axR.set_ylim(-0.25, 0.3)  # make the zero near baseline RMSE
fig

# %% [markdown]
# This chart clearly shows that coefficients and permutation importance measure different things.

# %% [markdown]
# ## Rescaling features
#
# Let's rescale the features and see how it compares to PCA (or any other dimensionality reduction approach).
# First, the original (unscaled) data:

# %%
from sklearn import decomposition

# do pca
pca = decomposition.PCA()
z = pca.fit_transform(x)

# plot the result
fig, ax = plt.subplots()
_ = ax.scatter(*z[:, :2].T)

# plot the components
fig, ax = plt.subplots()
_ = ax.bar(x.columns, pca.components_[0])
_ = ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)

# %% [markdown]
# Here we see that Cement is the dominant feature in the input data.
# Why?

# %%
print( x.std(axis=0) )

# %% [markdown]
# Very simply, that column had the largest absolute variance.
# Thus it was the dominant column in the first component of the PCA.
#
# Now we can move on to the rescaled data for comparison:

# %%
from sklearn import preprocessing

# use a scaler to normalize the feature magnitude
x_sc = preprocessing.StandardScaler().fit_transform( x.values )
x_sc = pd.DataFrame( x_sc, columns=x.columns )  # make it back into a DataFrame
pca = decomposition.PCA()
z = pca.fit_transform(x_sc)

# plot the result
fig, ax = plt.subplots()
_ = ax.scatter(*z[:, :2].T)

# plot the components
fig, ax = plt.subplots()
_ = ax.bar(x_sc.columns, pca.components_[0])
_ = ax.set_xticklabels([it.split('(')[0].strip() for it in x_sc.columns], rotation=90)

# %% [markdown]
# Once the columns are normalized, the variance in every column is 1 and the PCA becomes more balanced.
# Finally, we can see what effect this has on the permutation importance:

# %%
# compute baseline
model = linear_model.LinearRegression().fit(x_sc, y)
baseline = calc_rmse(y, model.predict(x_sc))  # first score the baseline model with all columns

# compute permutation importance
permuted_sc = np.zeros_like(x_sc.columns)  # create empty array to store values
for i, col in enumerate(x_sc.columns):
    x_permuted = x_sc.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(x_sc[col])
    permuted_sc[i] = calc_rmse(y, model.predict(x_permuted))  # score on the permuted column

# plot result
fig, ax = plt.subplots()
ax.bar(x.columns, permuted, label='Permuted')
ax.bar(x_sc.columns, permuted_sc, label='Permuted (S)', edgecolor='tab:orange', facecolor='none')
ax.set_xticklabels([it.split('(')[0].strip() for it in x_sc.columns], rotation=90)
ax.hlines(baseline, 0, len(x_sc.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend()

# %% [markdown]
# We can see here that aside from some minor fluctuations (possibly due to uncontrolled random seed in `np.random.permutation`), there is no difference in the permutation feature importance before and after scaling.
# This illustrates an important difference between unsupervised learning and supervised learning!

# %% [markdown]
# ## Spurious features
#
# Let's add a fake feature to the data and compare results between PCA and feature importance:

# %%
x_aug = x.copy()
x_aug['Random'] = np.random.rand(x_aug.shape[0]) * 1e6
x_aug

# %% [markdown]
# Just to make this totally unambiguous, let's take a look at the resulting PCA decomposition:

# %%
# do pca
pca = decomposition.PCA()
z = pca.fit_transform(x_aug)

# plot the result
fig, ax = plt.subplots()
_ = ax.scatter(*z[:, :2].T)

# plot the components
fig, ax = plt.subplots()
_ = ax.bar(x_aug.columns, pca.components_[0])
_ = ax.set_xticklabels([it.split('(')[0].strip() for it in x_aug.columns], rotation=90)

# %% [markdown]
# As you can see the Random feature totally dominates the space due to its large magnitude.
# Now let's pass this augmented data through permutation importance:

# %%
# compute baseline
model = linear_model.LinearRegression().fit(x_aug, y)
baseline = calc_rmse(y, model.predict(x_aug))  # first score the baseline model with all columns

# compute permutation importance
permuted_aug = np.zeros_like(x_aug.columns)  # create empty array to store values
for i, col in enumerate(x_aug.columns):
    x_permuted = x_aug.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(x_aug[col])
    permuted_aug[i] = calc_rmse(y, model.predict(x_permuted))  # score on the permuted column

# plot result
fig, ax = plt.subplots()
ax.bar(x_aug.columns, permuted_aug, label='Permuted')
ax.set_xticklabels([it.split('(')[0].strip() for it in x_aug.columns], rotation=90)
ax.hlines(baseline, 0, len(x_aug.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend()

# %% [markdown]
# You'll see here that the Random feature is assigned practically no importance in terms of changing the model score.

# %% [markdown]
# ## Drop-column importance
#
# Permutation feature importance tells us something very specific: how much  **this trained model** depends on the particular feature we are permuting.
# It does not tell us how necessary the given column is for predicting the outcome.
#
# To be more holistic we can actually train a series of models while leaving out each column and see how well the model can do.
# This called drop-column importance because we are dropping each column in the training.

# %%
import copy

x = np.random.rand(5)
print(x)
y = copy.deepcopy(x)
y += 1
print(x)

# %%
model.fit(x, y)
baseline = calc_rmse(y, model.predict(x))  # first score the baseline model with all columns
print(f'baseline rmse is {baseline}')

dropped = np.zeros_like(x.columns)  # create empty array to store values
for i, col in enumerate(x.columns):
    x_dropped = x.copy().drop(columns=col)  # remember to copy!
    model.fit(x_dropped, y)
    dropped[i] = calc_rmse(y, model.predict(x_dropped))  # score with the dropped column
print('rmse on dropped columns is:', dropped)

# %% [markdown]
# The results here appear much closer than in the permutation importance.
# Let's reuse our plotting code from above:

# %%
fig, ax = plt.subplots()
ax.bar(x.columns, dropped, label='Dropped')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend(loc='lower center')

# %% [markdown]
# We can show both Permutation and Drop-Column importance on the same chart to really get a sense for it:

# %%
fig, ax = plt.subplots()
ax.bar(x.columns, permuted, width=0.5, align='edge', label='Permuted')
ax.bar(x.columns, dropped, width=0.5, align='center', label='Dropped')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend(loc='lower center')

# %% [markdown]
# Here we see that not only are the quantitative results different between the two, but the general trends are even different.
# For instance, the model does not suffer nearly as much as we thought when it loses access to Cement *and can retrain without it*.
# This is probably because the same information is available from the rest of the columns (i.e., the total density is similar for all instances).
# Instead, Age of the same becomes the most influential variable because it cannot be inferred from the other data.

# %% [markdown]
# ## Using the test set
#
# If you were paying attention last week, all these `model.fit(x, y)` calls should be bothering you.
# Instead, we should be using `train_test_split` to assess the performance on only data not seen during train time.
# Let's fix this now.

# %%
from sklearn import model_selection

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

model = linear_model.LinearRegression().fit(xtrain, ytrain)  # train on train data
base_train = calc_rmse(ytrain, model.predict(xtrain))
base_test = calc_rmse(ytest, model.predict(xtest))  # evaluations will be on test data
print(f'baseline rmse is {base_train} / {base_test}')

permuted = np.zeros_like(x.columns)  # create empty array to store values
for i, col in enumerate(x.columns):
    x_permuted = xtest.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(xtest[col])
    permuted[i] = calc_rmse(ytest, model.predict(x_permuted))  # score on the permuted column
print('rmse on permuted columns is:', permuted)

# %% [markdown]
# Of course we'll visualize this result like we did above:

# %%
fig, ax = plt.subplots()
ax.bar(x.columns, permuted, label='Permuted')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(base_train, 0, len(x.columns)-1, linestyles='dashed', label='Base, Train')
ax.hlines(base_test, 0, len(x.columns)-1, linestyles='dotted', label='Base, Test')
ax.set_ylabel('Model RMSE')
ax.legend()

# %% [markdown]
# The validation set doesn't change much compared to our analysis on the train set, which is good.
# In this case, the test performance is incidentally better than training.
# This doesn't usually happen, but the model is pretty poor ($R^2 \approx 0.6$) so there is plenty of room for some random fluctutations depending on the particular data in each set.

# %% [markdown]
# ## [Check your understanding]
#
# Measure *drop-column importance* using a train / test split instead of training on all the available data.

# %%

# %%
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=1)

model = linear_model.LinearRegression().fit(xtrain, ytrain)  # train on train data
base_train = calc_rmse(ytrain, model.predict(xtrain))
base_test = calc_rmse(ytest, model.predict(xtest))  # evaluations will be on test data
print(f'baseline rmse is {base_train} / {base_test}')

dropped = np.zeros_like(x.columns)
for i, col in enumerate(x.columns):
    x_dropped_train = xtrain.copy().drop(columns=col)
    x_dropped_test = xtest.copy().drop(columns=col)
    model.fit(x_dropped_train, ytrain)
    dropped[i] = calc_rmse(ytest, model.predict(x_dropped_test))
print('rmse on dropped columns is:', dropped)

fig, ax = plt.subplots()
ax.bar(x.columns, dropped, label='Dropped, Test')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(base_train, 0, len(x.columns)-1, linestyles='dashed', label='Base, Train')
ax.hlines(base_test, 0, len(x.columns)-1, linestyles='dotted', label='Base, Test')
ax.set_ylabel('Model RMSE')
ax.legend()


# %% [markdown]
# # Evaluating nonlinear models
#
# Let's investigate how this works on nonlinear models (really, any model that's not `LinearRegression`).
# Before we get into it let's make some convenience functions for our repeated operations:

# %%
def permutation_importance(model, x, y, metric=calc_rmse):
    """Compute the permutation importance on a trained model."""
    baseline = metric(y, model.predict(x))

    permuted = np.zeros_like(x.columns)
    for i, col in enumerate(x.columns):
        x_permuted = x.copy()
        x_permuted[col] = np.random.permutation(x[col])
        permuted[i] = metric(y, model.predict(x_permuted))

    return baseline, permuted


# %% [markdown]
# ## Random Forest
#
# Now we can start with the best go-to model, the Random Forest.

# %%
from sklearn import ensemble

model = ensemble.RandomForestRegressor(random_state=0)
model.fit(xtrain, ytrain)

baseline_rf, permuted_rf = permutation_importance(model, xtest, ytest)
print('rmse on baseline data is:', baseline_rf)
print('rmse on permuted columns is:', permuted_rf)

# %% [markdown]
# It looks like these have a lot more variance compared to our linear result.

# %%
fig, ax = plt.subplots()
ax.bar(x.columns, permuted_rf, label='Random Forest')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline_rf, 0, len(x.columns)-1, linestyles='dashed', label='Baseline (RF)')
ax.set_ylabel('Model RMSE')
ax.legend()

# %% [markdown]
# We see that Cement and Age continue to play important roles in this new model, although there are appear to be some stronger influences from the other features compared to the linear model.
# Let's check...

# %% [markdown]
# ## Comparison to Linear Regression
#
# We can compare these results directly to the linear regression result:

# %%
model = linear_model.LinearRegression()
model.fit(xtrain, ytrain)
baseline_lin, permuted_lin = permutation_importance(model, xtest, ytest)

# %% [markdown]
# Plotting for clarity:

# %%
fig, ax = plt.subplots()

ax.bar(x.columns, permuted_rf, width=0.5, align='center', label='Random Forest')
ax.hlines(baseline_rf, 0, len(x.columns)-1, linestyles='dashed', color='tab:blue', label='Baseline (RF)')

ax.bar(np.arange(x.columns.shape[0]), permuted_lin, width=0.5, align='edge', label='Linear')
ax.hlines(baseline_lin, 0, len(x.columns)-1, linestyles='dashed', color='tab:orange', label='Baseline (LR)')

ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Model RMSE')
ax.legend()

# %% [markdown]
# This chart might be easier to read as the change from baseline since the two models have very different baseline RMSE:

# %%
fig, ax = plt.subplots()

delta_percent_rf = 100 * (permuted_rf - baseline_rf) / baseline_rf
ax.bar(x.columns, delta_percent_rf, width=0.5, align='center', label='Random Forest')

delta_percent_lin = 100 * (permuted_lin - baseline_lin) / baseline_lin
ax.bar(np.arange(x.columns.shape[0]), delta_percent_lin, width=0.5, align='edge', label='Linear')

ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model RMSE (%)')
ax.legend()

# %% [markdown]
# From this chart we can see that Cement and Age are very important for both models.
# However, Water and Superplasticizer are also key features for the Random Forest despite not being influential in the linear model.
# Another difference that is clear when plotting this as a percent difference is that Random Forest in general experiences stronger effects from permuting any given column -- even the Aggregates show a notable effect here, while they were completely inconsequential for linear regression.

# %% [markdown]
# ## Evaluating a bunch of models
#
# Let's try this on some additional regressors and see if we can find some common trends.

# %%
from sklearn import neighbors, tree, neural_network

# make a list of model constructors that can be called like constructor().fit(x, y)
constructors = [linear_model.LinearRegression,
                ensemble.RandomForestRegressor,
                neighbors.KNeighborsRegressor,
                tree.DecisionTreeRegressor,
                neural_network.MLPRegressor,
                ]

results = {}
for constructor in constructors:
    try:
        model = constructor(random_state=0).fit(xtrain, ytrain)
    except:
        model = constructor().fit(xtrain, ytrain)
    b, p = permutation_importance(model, xtest, ytest)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'permuted': p}

# %% [markdown]
# Let's quickly check what the baseline performance looked like:

# %%
baseline = []
names = []
for model_name, scores in results.items():
    baseline.append( scores['baseline'] )
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    names.append( short_model_name )

xticks = np.arange(len(constructors))

fig, ax = plt.subplots()
ax.bar(xticks, baseline)
ax.set_xticks(xticks)
ax.set_xticklabels(names, rotation=45, horizontalalignment='right')
ax.set_ylabel('Baseline RMSE')

# %% [markdown]
# We should keep this in mind going into the results that Random Forest was by far the most successful model.
# It is important to keep in mind this is probably due to using default hyperparameters!
# Feature importance of a bad (poorly tuned) model might not mean anything.
#
# We'll use a line plot for the delta RMSE this time since there will be so many series:

# %%
fig, ax = plt.subplots()

xticks = np.arange(x.columns.shape[0])
for model_name, scores in results.items():
    delta_percent = 100 * (scores['permuted'] - scores['baseline']) / scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta_percent, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model RMSE (%)')
ax.legend()


# %% [markdown]
# It seems all the models depend heavily on Cement and Age, although the tree-based models experience the strongest influence of Age.
# Water and Superplasticizer also appear to have the strongest influence on the tree-based models.
# Again, this could be due to the fact that the other models are not tuned well.
# Things could change if they were optimized properly.

# %% [markdown]
# ## Doing the same with drop columns
#
# Remember that the permutation feature importance only measures how much the model relies on those features.
# Dropping columns gives the model a chance to learn the same information somewhere else in the data.
# Let's try it with all those models above:

# %%
def drop_column_importance(model, x, y, metric=calc_rmse):
    """Compute the drop-column importance on a trained model."""
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

    model.fit(xtrain, ytrain)
    baseline = metric(ytest, model.predict(xtest))

    dropped = np.zeros_like(xtest.columns)
    for i, col in enumerate(xtest.columns):
        x_dropped = xtest.copy().drop(columns=col)
        model.fit(x_dropped, ytest)
        dropped[i] = metric(ytest, model.predict(x_dropped))

    return baseline, dropped


# %% [markdown]
# This will take longer since we have to train the models many times:

# %%
results = {}
for constructor in constructors:
    try:
        model = constructor(random_state=0)  # instantiate the model object from class name
    except:
        model = constructor()
    b, d = drop_column_importance(model, x, y)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'dropped': d}

# %% [markdown]
# Finally we evaluate the result:

# %%
fig, ax = plt.subplots()

xticks = np.arange(x.columns.shape[0])
for model_name, scores in results.items():
    delta_percent = 100 * (scores['dropped'] - scores['baseline']) / scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta_percent, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model RMSE (%)')
ax.legend()

# %% [markdown]
# Note how differently these models behave in the drop column test compared to permtuations!
#
# This result is a little crazy because it says that tree-based models perform **better** when dropping features and retraining -- for all except Age.
# This confirms the claim above that any other component can be inferred from the rest, while it also shows that the model gets less confused without additional variables to make decisions with.
#
# For K-Neighbors and Linear Regression, Cement and Age show up again as important.
#
# The MLP result is crazy and probably indicates poorly designed and poorly fitted models.
# We shouldn't read too much into this without additional tuning (neural networks need a lot of tuning!)

# %% [markdown]
# ## [Check your understanding]
#
# (a) Analyze one or more additional regression models using this scheme.
#
# (b) Try switching the metric to $R^2$ and make a plot similar to the one above.
# > You will need to specify a different `metric` keyword argument in the `drop_column_importance` function call.

# %%

# %%
from sklearn import svm

# part a
constructors = [svm.SVR]

results = {}
for constructor in constructors:
    try:
        model = constructor(random_state=0)  # instantiate the model object from class name
    except:
        model = constructor()
    b, d = drop_column_importance(model, x, y)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'dropped': d}

fig, ax = plt.subplots()

xticks = np.arange(x.columns.shape[0])
for model_name, scores in results.items():
    delta_percent = 100 * (scores['dropped'] - scores['baseline']) / scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta_percent, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model RMSE (%)')
ax.legend()

# %%
from sklearn import metrics

# part b
results = {}
for constructor in constructors:
    try:
        model = constructor(random_state=0)  # instantiate the model object from class name
    except:
        model = constructor()
    b, d = drop_column_importance(model, x, y, metric=metrics.r2_score)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'dropped': d}

fig, ax = plt.subplots()

xticks = np.arange(x.columns.shape[0])
for model_name, scores in results.items():
    delta_percent = 100 * (scores['dropped'] - scores['baseline']) / scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta_percent, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model $R^2$ (%)')
ax.legend()

# %% [markdown]
# # Classification
#
# The great thing about these feature importance schemes is they work exactly the same way with classification problems as regression problems.
# We just need to choose a suitable metric (e.g., accuracy) and perform the same procedure.
# Let's try it on the dataset from last time:

# %% [markdown]
# ## A new dataset

# %%
data_cl = pd.read_csv('../datasets/steels.csv')
data_cl['Alloy family'] = [x[0] for x in data_cl['Alloy code']]
data_cl

# %% [markdown]
# ## Modification for classifiers
#
# We need to use new `constructors_cl` and specify `metric=metrics.accuracy_score` as a kwarg of `permutation_importance`:

# %%
from sklearn import metrics, preprocessing

x_cl = data_cl.loc[:, ' C':'Nb + Ta']
y_cl = preprocessing.LabelEncoder().fit_transform(data_cl['Alloy family'])

xtrain_cl, xtest_cl, ytrain_cl, ytest_cl = model_selection.train_test_split(x_cl, y_cl, test_size=0.20, shuffle=True, random_state=0)

# make a list of model constructors that can be called like constructor().fit(x, y)
constructors_cl = [ensemble.RandomForestClassifier,
                  neighbors.KNeighborsClassifier,
                  tree.DecisionTreeClassifier,
                  neural_network.MLPClassifier,
                  ]

results = {}
for constructor in constructors_cl:
    try:
        model = constructor(random_state=0).fit(xtrain_cl, ytrain_cl)
    except:
        model = constructor().fit(xtrain_cl, ytrain_cl)
    b, p = permutation_importance(model, xtest_cl, ytest_cl, metric=metrics.accuracy_score)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'permuted': p}

# %% [markdown]
# Again, let's check the baseline accuracy of our models:

# %%
baseline = []
names = []
for model_name, scores in results.items():
    baseline.append( scores['baseline'] )
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    names.append( short_model_name )

xticks = np.arange(len(constructors_cl))

fig, ax = plt.subplots()
ax.bar(xticks, baseline)
ax.set_xticks(xticks)
ax.set_xticklabels(names, rotation=45, horizontalalignment='right')
ax.set_ylabel('Baseline Accuracy')

# %% [markdown]
# Here all the models perform at nearly 100% accuracy -- even on test data.
# Remember this was a trivial classification problem.
#
# And now we can visualize the change in performance with permuted columns:

# %%
fig, ax = plt.subplots()

xticks = np.arange(x_cl.columns.shape[0])
for model_name, scores in results.items():
    delta = scores['permuted'] - scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it for it in x_cl.columns], rotation=90)
ax.set_ylabel('Delta Model Accuracy')
ax.legend()

# %% [markdown]
# We can see that Mn, Ni, Cr, and Mo are the key elements that control the classification decisions for most of the models.
# Decision Tree picks up V and K-Neighbors picks up Mn, but the others don't.
#
# Let's repeat this with drop column importance:

# %%
results = {}
for constructor in constructors_cl:
    try:
        model = constructor(random_state=0)  # instantiate the model object from class name
    except:
        model = constructor()
    b, d = drop_column_importance(model, x_cl, y_cl, metric=metrics.accuracy_score)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'dropped': d}

# %% [markdown]
# Plotting the drop-column feature importance:

# %%
fig, ax = plt.subplots()

xticks = np.arange(x_cl.columns.shape[0])
for model_name, scores in results.items():
    delta = scores['dropped'] - scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it for it in x_cl.columns], rotation=90)
ax.set_ylabel('Delta Model Accuracy')
ax.legend()

# %% [markdown]
# Overall this looks similar to the results from permutation importance, except much of the noise for elements other than Cr and Mo is removed.
# In other words, none of the models mistakenly attribute much weight to elements other than Cr and Mo.

# %% [markdown]
# # Repeatability
#
# These methods include a significant amount of randomness.
# We should repeat them several times and average the result to get the best picture of what's going on.

# %%
n_repeats = 10

b_list = np.zeros([n_repeats, 1])
p_list = np.zeros([n_repeats, x_cl.columns.shape[0]])

for k in range(n_repeats):

    split_data = model_selection.train_test_split(x_cl, y_cl, test_size=0.20,
                                                shuffle=True, random_state=k)
    xtrain_cl, xtest_cl, ytrain_cl, ytest_cl = split_data

    model = ensemble.RandomForestClassifier(random_state=k)
    model.fit(xtrain_cl, ytrain_cl)

    b_list[k], p_list[k] = permutation_importance(model, xtest_cl, ytest_cl, metric=metrics.accuracy_score)

# %% [markdown]
# Now we'll make the same chart but with [`plt.errorbar`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html)

# %%
fig, ax = plt.subplots()

delta = (p_list - b_list).mean(axis=0)
sigma = (p_list - b_list).std(axis=0)

xticks = np.arange(x_cl.columns.shape[0])
short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
ax.errorbar(xticks, delta, yerr=sigma, marker='s')

ax.set_xticks(xticks)
ax.set_xticklabels([it for it in x_cl.columns], rotation=90)
ax.set_ylabel('Delta Model Accuracy')

# %% [markdown]
# Here we find that repeated trials average out any result from elements other than Cr and Mo and confirm that the drop in classification accuracy for these two are indeed significant.

# %% [markdown]
# ## [Check your understanding]
#
# (a) It turns our `scikit-learn` has these built into the `inspection` submodule: [inspection.permutation_importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
# Use this function to create a plot like the one above for the `RandomForestClassifier`.
#
# (b) Repeat this analysis for one or more additional classifiers.
# Does the trend hold up?
# What can you learn about the dataset if the classifiers are consistent or inconsistent?

# %%
