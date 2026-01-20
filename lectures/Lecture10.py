# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's topics:
# * Bayesian optimization for hyperparameter tuning
# * Evolutionary algorithm for feature selection
#

# %% [markdown]
# Start by loading the alloys mechanical properties dataset:

# %%
import pandas as pd

data = pd.read_csv('../datasets/steels.csv')
data

# %% [markdown]
# Select the features and regression labels:

# %%
x = data.loc[:, ' C':' Temperature (°C)']
y = data[' Tensile Strength (MPa)']

# %% [markdown]
# Let's train a multivariate linear regression as a baseline:

# %%
from sklearn import model_selection
from sklearn import linear_model

model = linear_model.LinearRegression()

folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for train_index, test_index in folds.split(x):
    # split the data
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    # train a model
    model.fit(x_train, y_train)
    # evaluate r2 on validation set
    r2 = model.score(x_val, y_val)
    results.append(r2)

print(results)

# %% [markdown]
# And if we try a tree-based scheme?

# %%
from sklearn import model_selection
from sklearn import ensemble

model = ensemble.RandomForestRegressor(random_state=0)

folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for train_index, test_index in folds.split(x):
    # split the data
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    # train a model
    model.fit(x_train, y_train)
    # evaluate r2 on validation set
    r2 = model.score(x_val, y_val)
    results.append(r2)

print(results)

# %% [markdown]
# Random Forest is much better!
# Although there is still one fold that gives poor validation performance.
# What about a Neural Network?

# %%
from sklearn import model_selection
from sklearn import neural_network

model = neural_network.MLPRegressor(random_state=0)

folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for train_index, test_index in folds.split(x):
    # split the data
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    # train a model
    model.fit(x_train, y_train)
    # evaluate r2 on validation set
    r2 = model.score(x_val, y_val)
    results.append(r2)

print(results)

# %% [markdown]
# I mentioned before that neural networks require more extensive hyperparameter tuning.
# Let's explore methods for doing that now.

# %% [markdown]
# # Bayesian optimization
#
# <img src="./assets/bayesian_optimization_workflow.jpg" width=600 alt="Flowchart of the Bayesian Optimization process: surrogate model, acquisition function, and objective evaluation">

# %% [markdown]
# ## Gaussian process
#
# <img src="./assets/gp_noise_gp.jpg" width=600 alt="Visualization of a Gaussian Process regression with confidence intervals and noisy data points">
#
# <img src="./assets/kernel_types.jpg" width=600 alt="Visual gallery of different Gaussian Process kernels and their resulting functions">

# %% [markdown]
# ## Basic hyperparameter tuning
#
# We will use the `ax-platform` package for hyperparameter tuning.
# It is more convenient than implementing this ourselves.

# %%
# !pip install ax-platform

# %% [markdown]
# Now we need to do a train/validation/test split and an objective function.

# %%
x_trv, x_test, y_trv, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_trv, y_trv, train_size=0.75, shuffle=True, random_state=0)

def mlp_fitness(parameterization):
    try:
        model = neural_network.MLPRegressor(**parameterization, random_state=0).fit(x_train, y_train)
        score = model.score(x_val, y_val)
    except:
        score = -1
    return score


# %% [markdown]
# Now we set up the optimization problem in the format specified by `ax-platform`.

# %%
from ax.service.managed_loop import optimize

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "learning_rate_init", "type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
        {"name": "max_iter", "type": "range", "bounds": [10, 1000]},
    ],
    evaluation_function=mlp_fitness,
    objective_name='r-squared',
)

# %% [markdown]
# With Gaussian Process, we can evaluate the results in terms of both mean and variance:

# %%
print( best_parameters )
means, covariances = values
print( means, covariances )

# %% [markdown]
# It is very helpful to visualize the loss surface:

# %%
from ax.plot.contour import plot_contour
from ax.utils.notebook.plotting import render

render(plot_contour(model=model, param_x='learning_rate_init', param_y='max_iter', metric_name='r-squared'))

# %% [markdown]
# And we can also view it as a function of iterations.

# %%
from plotly import express as px
import numpy as np

best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
results = np.maximum.accumulate(best_objectives, axis=1).tolist()
px.line(y=results)

# %% [markdown]
# Now that we have the optimal result, we should train a model with those hyperparameters.

# %%
model = neural_network.MLPRegressor(**best_parameters, random_state=0).fit(x_train, y_train)
print( 'r-squared, val: ', model.score(x_val, y_val))
print( 'r-squared, test:', model.score(x_test, y_test))

# %% [markdown]
# ## Categorical hyperparameters
#
# We can also specify discrete options such as the activation functions.
# These can have a huge effect on the results:
#
# <img src="./assets/activation_functions.jpg" width=600 alt="Plots of common neural network activation functions including Sigmoid, Tanh, and ReLU">

# %%
best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "learning_rate_init", "type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
        {"name": "max_iter", "type": "range", "bounds": [10, 1000]},
        {"name": "activation", "type": "choice", "values": ["identity", "logistic", "tanh", "relu"]}
    ],
    evaluation_function=mlp_fitness,
    objective_name='r-squared',
)

# %%
print(best_parameters)

# %% [markdown]
# ## [Check your understanding]
#
# Use `ax-platform` to optimize the `solver`.
# Refer to the documentation for the options.

# %%

# %% [markdown]
# ## Architecture optimization
#
# Deep learning models may have very complex architecture with many hyperparameters to choose:
#
# <img src="./assets/neural_network_architecture.jpg" width=600 alt="Schematic diagram of a multi-layer perceptron neural network with input, hidden, and output layers">
#
# In this case, we need to be clever in how to encode the many possible options.
# Here are some common shapes for NNs:
#
# <img src="./assets/neural_network_zoo.jpg" width=400 alt="Graphic showing various neural network architecture types beyond simple feed-forward networks">
#
# You will see that NNs do not typically have wildly oscillating sizes between layers.
# Instead, they vary smoothly and the typical shapes are flat or trapezoidal.
# This means we can reduce the number of parameters from choosing every number of neurons independently to only choosing the "shape" of the network.

# %% [markdown]
# ## [Check your understanding]
#
# First, write a function that gives you a list `hidden_layer_sizes` that encodes the number of neurons in each layer from a simpler parameterization (e.g., number of layers `n`, number of neurons in the first layer `n_init`, and number of neurons in the last layer, `n_last`).
#
# Now use `ax-platform` to optimize the `hidden_layer_sizes` using your parameterization above.
#
# Refer to the `MLPRegressor` documentation for syntax.

# %%

# %% [markdown]
# # Evolutionary algorithm

# %% [markdown]
# ## Concepts
#
# "Evolutionary algorithm," "genetic algorithm," or "evolutionary optimization" is a scheme that utilizes the idea of natural selection to perform numerical optimization:
#
# <img src="./assets/genetic_algorithm_concept.jpg" width=600 alt="Visual metaphor for Genetic Algorithms based on biological evolution: selection, crossover, and mutation">
#
# In each "generation," traits from the best individuals are combined in a process analagous to gene transfer between DNA of parents in biological organisms:
#
# <img src="./assets/genetic_algorithm_flow.jpg" width=600 alt="Step-by-step flowchart of the Genetic Algorithm iterative loop">
#
# We also include mutations to permit new traits to arise in the population:
#
# <img src="./assets/genetic_algorithm_operators.jpg" width=600 alt="Visual detail of Genetic Algorithm operators: Crossover (recombination) and Mutation">
#
# If the new traits lead to greater fitness, they persist and are passed on to future generations.
#
# Why use evolutionary algorithm (EA) over Gaussian Process (GP)?
# One answer is that GP stops working well in higher dimensions due to the ambiguity of distances in those high-dimensional spaces.
# Another is that GP is meant for continuous spaces, while we often have discrete choices in model tuning.
# EA has no problem with high-dimensional spaces as the crossover and mutation can occur in any dimension.
# In addition, GP scales like $\mathcal{O}(N^3)$, which can get out of hand quickly.
# EA has no fitting so it's simply $\mathcal{O}(N)$ -- although it may converge less quickly.

# %% [markdown]
# ## Feature selection
#
# Let's consider an example

# %%
# !pip install pygad

# %%
from sklearn import neighbors

# set up train/validation/test sets
x_trv, x_test, y_trv, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_trv, y_trv, train_size=0.75, shuffle=True, random_state=0)

# set up fitness function
def fitness(features, solution_index):
    """A fitness function that selects features based on the input array and trains a KNeighborsRegressor.
    It returns the model R-squared on validation data."""
    f = np.argwhere(features > 0.5).flatten()
    model = neighbors.KNeighborsRegressor().fit(x_train.iloc[:, f], y_train)
    r_squared = model.score(x_val.iloc[:, f], y_val)
    return r_squared


# %% [markdown]
# Let's make sure this does what we expect:

# %%
all_features = np.ones(x.shape[1])
print( fitness(all_features, None) )

# %% [markdown]
# Compared to the full feature set:

# %%
model = neighbors.KNeighborsRegressor().fit(x_train, y_train)
print('validation: ', model.score(x_val, y_val))
print('test: ', model.score(x_test, y_test))

# %% [markdown]
# Set up the `pygad` optimization problem (it has a lot of parameters!!!):

# %%
import pygad

features = np.zeros(x.shape[1], dtype=int)
fitness_function = fitness

num_generations = 16
num_parents_mating = 2

sol_per_pop = 8
num_genes = len(features)

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       random_seed=0, save_solutions=True,
                       )

# %% [markdown]
# Run it:

# %%
ga_instance.run()

# %% [markdown]
# View the best solution and its fitness:

# %%
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : ", x.columns[solution>0.5])
print("Fitness value of the best solution = ", solution_fitness)

# %% [markdown]
# We can plot the result as a function of iteration:

# %%
_ = ga_instance.plot_fitness()

# %% [markdown]
# We can visualize which features were selected by looking back at the `solutions` attribute of the `ga_instance`:

# %%
from matplotlib import pyplot as plt

solutions = np.array(ga_instance.solutions) > 0.5

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

ax = axes[0]
_ = ax.imshow(np.array(ga_instance.solutions_fitness).reshape(1, -1))

ax = axes[1]
_ = ax.set_yticks(np.arange(x.shape[1]))
_ = ax.set_yticklabels(x.columns)
_ = ax.imshow(solutions.T)
_ = ax.set_xlabel('Solution index')
_ = ax.set_ylabel('Feature index')
_ = ax.set_aspect('auto')

plt.subplots_adjust(hspace=0)

# %% [markdown]
# And like with `ax-platform`, we can also check test performance in this case:

# %%
f = np.argwhere(solution > 0.5).flatten()
model = neighbors.KNeighborsRegressor().fit(x_train.iloc[:, f], y_train)
print('validation: ', model.score(x_val.iloc[:, f], y_val))
print('test: ', model.score(x_test.iloc[:, f], y_test))


# %% [markdown]
# We got lucky here in that our test set actually performs better than our validation set!

# %% [markdown]
# Let's also quickly compare to what we would get from recursive drop-column feature importance.

# %%
def drop_column_importance(model, xtrain, ytrain, xtest, ytest):
    """Compute the drop-column importance on a trained model."""

    model.fit(xtrain, ytrain)
    baseline = model.score(xtest, ytest)

    dropped = np.zeros_like(xtest.columns)
    for i, col in enumerate(xtest.columns):
        x_dropped_train = xtrain.copy().drop(columns=col)
        x_dropped_test = xtest.copy().drop(columns=col)
        model.fit(x_dropped_train, ytrain)
        dropped[i] = model.score(x_dropped_test, ytest)

    return baseline, dropped


def choose_worst_feature(model, x_train, y_train, x_val, y_val):
    baseline, dropped = drop_column_importance(model, x_train, y_train, x_val, y_val)
    return baseline, dropped, np.argmax(dropped)


model = neighbors.KNeighborsRegressor()

x_train_trim = x_train.copy()
x_val_trim = x_val.copy()

r2 = []
feature_order = []

for k in range(x_train_trim.shape[1]-1):
    baseline, dropped, worst = choose_worst_feature(model, x_train_trim, y_train, x_val_trim, y_val)
    r2.append(baseline)
    feature_order.append(x_train_trim.columns[worst])
    x_train_trim = x_train_trim.drop(columns=[x_train_trim.columns[worst]])
    x_val_trim = x_val_trim.drop(columns=[x_val_trim.columns[worst]])

# we have one feature left that we didn't drop:
last_feature = x_train_trim.columns[0]
feature_order.append( last_feature )

print(np.round(r2, 2))
print(feature_order)  # later is better (dropped last)

# %% [markdown]
# If we compare that to the results from EA we may see something interesting:

# %%
f = np.argwhere(solution > 0.5).flatten()

print( x.columns[f] )

model = neighbors.KNeighborsRegressor().fit(x_train.iloc[:, f], y_train)
print('validation: ', model.score(x_val.iloc[:, f], y_val))
print('test: ', model.score(x_test.iloc[:, f], y_test))

# %% [markdown]
# The features are different -- how do they compare in performance?

# %%
top_features_rfe = feature_order[-len(f):]
print(top_features_rfe)

model = model.fit(x_train.loc[:, top_features_rfe], y_train)

print( 'validation: ', model.score(x_val.loc[:, top_features_rfe], y_val) )
print( 'test: ', model.score(x_test.loc[:, top_features_rfe], y_test) )

# %% [markdown]
# ## Interaction between feature selection and hyperparameter tuning
#
# This feature selection scheme is only half of a complete workflow -- each of these feature subsets will have different optimal hyperparameters!
# For instance, imagine the extreme cases where we prune all but one feature.
# Of course the optimal $k$ could be different here than in the case where we include all the features.
# Likewise with distance weighting.
# To be sure you have the best possible model, you need to optimize features and hyperparameters together.
# This often requires training thousands of models!
#
# For this reason, there is always a balance between practicality and performance.
# If you will spend hundreds or thousands of compute hours and gain only 1% improved performance you are probably wasting your time and resources.
# As a result, it's best to start small and work your way up in complexity if you are seeing improvement.

# %% [markdown]
# ## [Check your understanding]
#
# (a) Implement cross-fold validation inside the GA fitness function and use the mean validation $R^2$ as the feature fitness.

# %%

# %% [markdown]
# (b) Alternatively, add an option to perform hyperparameter tuning for the `n_neighbors` and `weighting` of `KNeighborsRegressor` (together with feature selection).
# > You will need to split apart your `features` array into different parts when you get inside the `fitness` function. For instance,
# ```
# k = features[0]
# w = features[1]
# sel = features[2:]
# ```
# > You may also need to manipulate the values such as rounding to `int`, mapping to discrete options for `str`, etc.

# %%

# %% [markdown]
# (c) Alternatively, add an option to select both which features will be used and which model.
# > You will need to map an integer index to a model definition. One way is to use a predefined list like so:
#
# ```
# # before the fitness function:
# possible_models = [linear_model.LinearRegression(), ensemble.RandomForestRegressor(), neighbors.KNeighborsRegressor(), ...]
#
# # then inside the fitness function:
# model = possible_models[i]
# ```

# %%
