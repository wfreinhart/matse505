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
# Today's topics:
# * Search and screening
# * Conditional data generation
# * Inverse function approximation

# %% [markdown]
# # Requirements
# 
# Let's install the necessary packages first so we don't have to restart the runtime later!

# %%
!pip install sdv==1.0.0 pygad ax-platform nflows

# %% [markdown]
# # Dataset
# 
# We'll use the alloys dataset again for much of the lesson, then switch to another dataset later on.

# %% include: dataset_steels

# %% [markdown]
# There is an outlier in this dataset that needs to be corrected:

# %%
import numpy as np

bad_idx = np.argmax( data.loc[:, ' Tensile Strength (MPa)'] )
data.loc[bad_idx, ' Tensile Strength (MPa)'] /= 10.0  # missed a decimal point

# %% [markdown]
# # Screening and search
# 
# Let's consider how to identify and evaluate optimal designs using Machine Learning.

# %% [markdown]
# ## Design space
# 
# Our design space will be the composition and temperature of the alloys.
# We can evaluate our existing database to see what the known mechanical properties are.

# %%
from matplotlib import pyplot as plt

x = data.iloc[:, 1:-4]
y = data.iloc[:, -4:]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        if i == j:
            ax.hist(y.iloc[:, i])
        else:
            ax.scatter(y.iloc[:, i], y.iloc[:, j])
        if i == 3:
            ax.set_xlabel(y.columns[j])
        if j == 0:
            ax.set_ylabel(y.columns[i])

# %% [markdown]
# Let's imagine that we want to maximize Tensile Strength and Elongation (some proxy for ductility) at the same time.
# From the chart above you can see that these two properties compete with each other and lead to tradeoffs.

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(y.iloc[:, 1], y.iloc[:, 2])
ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

# %% [markdown]
# We should consider the "Pareto front" AKA "efficient frontier" like so:
# 
# <img src="../lectures/assets/lecture20_pareto_frontier.jpg" alt="Pareto efficient frontier showing trade-off between two objectives" width=400>
# 
# Here is a short code to compute the efficient frontier:

# %%
def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

# find the efficient frontier
costs = y.iloc[:, [1, 2]].values
on_frontier = is_pareto_efficient(-costs)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier', markerfacecolor='none')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# If we want to maximize both properties, the best solution would be one of these.
# Which solution is best depends on the relative value of each of the two objectives, strength or ductility.

# %% [markdown]
# ## Surrogate modeling
# 
# If we want to find new solutions, we need to start with a surrogate model.
# We'll use the Random Forest Regressor for simplicity.
# > It will be important to scale our properties since their scales vary by an order of magnitude or more!

# %%
from sklearn import preprocessing, model_selection

idx_train, idx_test = model_selection.train_test_split(np.arange(x.shape[0]))

# fit the scaler on only train data to avoid biases
x_scaler = preprocessing.StandardScaler().fit(x.iloc[idx_train])
y_scaler = preprocessing.StandardScaler().fit(y.iloc[idx_train])

# prepare scaled data
xs = x_scaler.transform(x)
ys = y_scaler.transform(y)

# %%
from sklearn import ensemble

# fit model
model = ensemble.RandomForestRegressor(random_state=0)
_ = model.fit(xs[idx_train], ys[idx_train])

# report performance
print(f'Train {model.score(xs[idx_train], ys[idx_train]):.3f}')
print(f'Test  {model.score(xs[idx_test],  ys[idx_test]):.3f}')

# %% [markdown]
# ## Screening
# 
# The most straightforward way to find good solutions from the forward model $f: x \to y$ is to try a lot of $x$ candidates and see if any give a $y$ we like.
# How would we do that?
# One way would be to draw from an empirical distribution from the input features.
# We should use Kernel Density Estimation for this.
# 
# If you recall, KDE approximates each point as a Gaussian:
# 
# <img src="../lectures/assets/lecture20_hist_vs_smooth.jpg" alt="Comparison between a histogram and a smooth density estimate" width=600>
# 
# This can easily be extended to 2+ dimensions like so:
# 
# <img src="../lectures/assets/lecture20_kde_example.jpg" alt="Kernel Density Estimation applied to the Old Faithful Geyser dataset" width=400>

# %% [markdown]
# Here is a short code to perform the KDE and sample from the fitted distribution:

# %%
from scipy.stats import gaussian_kde

kde = gaussian_kde(xs[idx_train].T)

xs_fake = kde.resample(100).T
ys_fake = model.predict(xs_fake)
y_fake = y_scaler.inverse_transform(ys_fake)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# So far none of these 100 samples outperformed the original data.
# However, there's nothing stopping us from generating many more samples.
# Let's try 10,000 samples:

# %%
# perform sampling
xs_fake = kde.resample(10000).T
ys_fake = model.predict(xs_fake)
y_fake = y_scaler.inverse_transform(ys_fake)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# While some of these solutions are close to the frontier, especially at high Tensile Strength, none of them are substantially better than what we already know.
# We need a more efficient scheme.

# %% [markdown]
# ## Search
# 
# Instead of sampling randomly, let's try searching near parts of the design space that we know are good.
# We already know how to do search using Evolutionary Algorithm with `pygad` and Bayesian Optimization with `ax-platform`.
# 
# We'll start with `pygad`.
# The first decison to make is the fitness or objective function.
# Let's just try to maximize both objectives by adding them togther:

# %%
# set up fitness function
def fitness(features, solution_index):
    out = model.predict(features.reshape(1, -1))
    return out[0, 1] + out[0, 2]

# %% [markdown]
# Then we can run the experiment like so:

# %%
import pygad

features = np.zeros(x.shape[1], dtype=int)
fitness_function = fitness

num_generations = 32
num_parents_mating = 2

sol_per_pop = 16
num_genes = len(features)

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "two_points"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=-3,
                       init_range_high=3,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       random_seed=0, save_solutions=True,
                       )

ga_instance.run()
_ = ga_instance.plot_fitness()

# %% [markdown]
# We see that over time our fitness improved slightly, but not dramatically.
# We can look at the result back in the property space:

# %%
ys_out = model.predict(ga_instance.solutions)
y_out = y_scaler.inverse_transform(ys_out)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_out[:, 1], y_out[:, 2], c=np.arange(y_out.shape[0]), label='Optimized')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# This does not give us a very satisfactory result, in that none of the solutions is superior to any of the Pareto-efficient solutions from the database.
# 
# We could also try with Bayesian Optimization.
# Actually `ax-platform` implements several proper Multi-Objective Optimization (MOO) schemes.
# Here we'll set up the experiment:

# %%
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

ax_client = AxClient()
ax_client.create_experiment(
    name="moo_experiment",
    parameters=[
        {"name": col, "type": "range", "bounds": [float(xs[:, i].min()), float(xs[:, i].max())]}
        for i, col in enumerate(data.columns[1:-4])
    ],
    objectives={
        "TS": ObjectiveProperties(minimize=False),
        "El": ObjectiveProperties(minimize=False)
    },
    overwrite_existing_experiment=True,
    is_test=True,
)

# %% [markdown]
# Then define the evaluation function:

# %%
coefs = np.polyfit(ys[on_frontier, 1], ys[on_frontier, 2], deg=3)
poly_func = np.poly1d(coefs)
def evaluate(parameterization):
    # construct array from dictionary
    features = np.array([parameterization[it] for it in data.columns[1:-4]])
    # perform prediction, etc.
    out = model.predict(features.reshape(1, -1))
    return {"TS": (out[0, 1], 0.0), "El": (out[0, 2], 0.0)}

# %% [markdown]
# Finally we run 50 trials:

# %%
for i in range(40):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

# %% [markdown]
# We can evaluate the outcome of these trials like so:

# %%
ys_out = np.zeros([0, 4])
for i, trial in ax_client.experiment.trials.items():
    parameterization = trial.arm.parameters
    features = np.array([parameterization[it] for it in data.columns[1:-4]])
    out = model.predict(features.reshape(1, -1))
    ys_out = np.vstack([ys_out, out])

# unscale the model output
y_out = y_scaler.inverse_transform(ys_out)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_out[:, 1], y_out[:, 2], c=np.arange(y_out.shape[0]), label='Optimized')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# You will see that even with Bayesian Optimization it is very difficult to improve upon the existing solutions.

# %% [markdown]
# # Synthetic data generation

# %% [markdown]
# ## Data generation with SDV
# 
# Let's use Synthetic Data Vault to train a conditional probabilistic model on our dataset.
# This will be similar to the use of KDE above, except that SDV permits conditional generation using rejection sampling (we'll see that in a minute).

# %%
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

df_meta = SingleTableMetadata()
df_meta.detect_from_dataframe(data)

# create a synthetic data generator using GaussianCopula model
synthesizer = GaussianCopulaSynthesizer(df_meta)
synthesizer.fit(data)

# generate synthetic data with no missing values
synthetic_data = synthesizer.sample(num_rows=10)
synthetic_data

# %% [markdown]
# We can repeat a similar process compared to what we did above by generating many thousands of samples and testing to see if any are like what we want:

# %%
# perform sampling
x_fake = synthesizer.sample(10000).iloc[:, 1:-4]
xs_fake = x_scaler.transform(x_fake)

ys_fake = model.predict(xs_fake)
y_fake = y_scaler.inverse_transform(ys_fake)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# These results are no better than using the simpler KDE above (may actually be worse).

# %% [markdown]
# ## Conditional (rejection) sampling
# 
# SDV uses rejection sampling to achieve conditional samples on the distributions.

# %%
from sdv.sampling import Condition

target = Condition(num_rows=100, column_values={' Tensile Strength (MPa)': 800})
designs = synthesizer.sample_from_conditions(conditions=[target])

designs

# %% [markdown]
# Are any of these realistic?
# To find out, we need to run the inputs through the predictive model.

# %%
xs_fake = x_scaler.transform(designs.iloc[:, 1:-4])

ys_fake = model.predict(xs_fake)
y_fake = y_scaler.inverse_transform(ys_fake)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# Unfortunately it seems like these designs give us an above-average TS but one that is within the 600-700 range rather than 800 as requested.
# 
# What if we reduce our TS requirement but add a Elongation requirement as well?

# %%
target = Condition(num_rows=100, column_values={' Tensile Strength (MPa)': 500, ' Elongation (%)': 40})
designs = synthesizer.sample_from_conditions(conditions=[target])

xs_fake = x_scaler.transform(designs.iloc[:, 1:-4])

ys_fake = model.predict(xs_fake)
y_fake = y_scaler.inverse_transform(ys_fake)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

# %% [markdown]
# Now it seems that the results are all over the place.
# The additional condition made it even harder to approximate this.

# %% [markdown]
# # A simple synthetic dataset
# 
# Let's create a simpler dataset and see if we can achieve good results using any of these methods.

# %%
def sample_tri(Ns):
    """Use rejection sampling to find `Ns` points inside an equilateral triangle."""
    # draw samples
    xy = np.random.rand(Ns*4, 2)
    xy[:, 1] *= np.sqrt(3)/2
    # constrain to the triangle
    c1 = xy[:, 1] < xy[:, 0]*np.sqrt(3)
    c2 = xy[:, 1] < np.sqrt(3)*(1-xy[:, 0])
    xy = xy[np.logical_and(c1,c2)]
    # return only the number requested
    xy = xy[:Ns]
    return xy

out = sample_tri(1000)
fig, ax = plt.subplots()
ax.plot(*out.T, '.')
ax.set_aspect('equal')

# %% [markdown]
# This is a fictitious ternary system.
# We could think of it like a 3-component metal alloy.
# 
# The above is a 2D representation in the phase diagram, but we need to be able to convert between this and the full 3D representation:

# %%
def xy_to_comp(xy):
    """Convert 2D coordinates to a ternary composition."""
    x, y = xy.T
    a = 2/np.sqrt(3)*y
    b = 1 - a/2 - x
    c = 2*x + b - 1
    return np.vstack([a,b,c]).T

def comp_to_xy(abc):
    """Convert ternary compositions to 2D coordinates."""
    a, b, c = abc.T
    x = ( 1 + c - b ) /2
    y = np.sqrt(3) * a / 2

    return np.vstack([x, y]).T

a, b, c = xy_to_comp(sample_tri(1000)).T

# draw histograms
fig, ax = plt.subplots()
_ = ax.hist(a, alpha=0.5)
_ = ax.hist(b, alpha=0.5)
_ = ax.hist(c, alpha=0.5)

# %% [markdown]
# Let's define some "property" of this system.

# %%
def quadratic_fom(xy, noise=0):
    """Define a figure of merit to model."""
    abc = xy_to_comp( xy )
    lab = np.array([[1,2,3]])
    metric = np.sum(abc**2 * lab, axis=1)
    metric += noise * np.random.standard_normal(metric.shape)
    return metric

x = sample_tri(1000)
y = quadratic_fom(x, 0.10).reshape(-1, 1)
fig, ax = plt.subplots()
im = ax.scatter(*x.T, s=4, c=y)
ax.set_aspect('equal')
cb = plt.colorbar(im)

# %% [markdown]
# The true minimum of this function is $f = 6/11$ at the point $(x, y, z) = (6/11, 3/11, 2/11)$, or approximately $(0.545, 0.273, 0.182)$.
# This corresponds to the point $(0.455, 0.472)$.
# Without the equation we wouldn't be able to determine this analytically.

# %% [markdown]
# ## Surrogate modeling
# 
# Let's generate a surrogate model for this simplified system:

# %%
# split data
idx_train, idx_test = model_selection.train_test_split(np.arange(x.shape[0]))

# fit model
model = ensemble.RandomForestRegressor(random_state=0)
_ = model.fit(x[idx_train], y[idx_train].flatten())

# report performance
print(f'Train {model.score(x[idx_train], y[idx_train]):.3f}')
print(f'Test  {model.score(x[idx_test],  y[idx_test]):.3f}')

# %% [markdown]
# ## Screening
# 
# And now we repeat our screening approach by KDE:

# %%
from scipy.stats import gaussian_kde

# fit kde model
kde = gaussian_kde(x[idx_train].T)

# perform sampling
x_fake = kde.resample(10000).T

# predict on these new samples
y_fake = model.predict(x_fake)

# make figure
fig, ax = plt.subplots()
im = ax.scatter(*x_fake.T, s=4, c=y_fake)
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
ax.set_aspect('equal')
cb = plt.colorbar(im)

# plot the true minimum
ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

# plot the minimum by screening
min_idx = np.argmin(y_fake)
ax.plot(*x_fake[min_idx], '^', color='tab:orange', label='Best Result')
print(f'min value is {y_fake[min_idx]} at {xy_to_comp( x_fake[min_idx] )}')

ax.legend()

# %% [markdown]
# We see that the value obtained by screening is close but not equal to the true minimum.

# %% [markdown]
# ## Search
# 
# We can employ Bayesian Optimization to search for the minimum:

# %%
from ax.service.managed_loop import optimize

def ax_fitness(parameterization):
    xy = np.array([parameterization[p] for p in 'xy']).reshape(1, -1)
    return model.predict(xy)[0]

best_parameters, values, experiment, ax_model = optimize(
    parameters=[
        {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
    ],
    evaluation_function=ax_fitness,
    objective_name='objective',
    minimize=True,
)

# %%
from ax.utils.notebook.plotting import render
from ax.plot.contour import plot_contour

render(plot_contour(model=ax_model, param_x='x', param_y='y', metric_name='objective'))

# %%
# process the experiments
x_out = []
y_out = []
for i, trial in experiment.trials.items():
    parameterization = trial.arm.parameters
    features = np.array([parameterization[p] for p in 'xy'])
    x_out.append(features)
    out = model.predict(features.reshape(1, -1))[0]
    y_out.append(out)

# make figure
fig, ax = plt.subplots()
im = ax.scatter(*np.array(x_out).T, s=4, c=y_out)
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
ax.set_aspect('equal')
cb = plt.colorbar(im)

# plot the true minimum
ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

# plot the minimum by Bayesian Optimization
best_features = np.array([best_parameters[p] for p in 'xy'])
best_outcome = model.predict(best_features.reshape(1, -1))[0]
ax.plot(*best_features, '^', color='tab:orange', label='Best Result')
print(f'min value is {best_outcome} at {xy_to_comp( best_features )}')

ax.legend()

# %% [markdown]
# Your result will vary here.
# In my trials I sometimes got very close and other times much farther away.

# %% [markdown]
# ## Rejection sampling
# 
# We can also try the rejection sampling approach with SDV to get values close to some target:

# %%
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

ternary_df = pd.DataFrame({'x': x[:, 0], 'y': x[:, 1], 'obj': y.flatten()})

df_meta = SingleTableMetadata()
df_meta.detect_from_dataframe(ternary_df)

# create a synthetic data generator using GaussianCopula model
synthesizer = GaussianCopulaSynthesizer(df_meta)
synthesizer.fit(ternary_df)

# generate synthetic data with no missing values
synthetic_data = synthesizer.sample(num_rows=10)
synthetic_data

# %%
from sdv.sampling import Condition

low_objective = Condition(num_rows=100, column_values={'obj': 0.45})
synthetic_data = synthesizer.sample_from_conditions(conditions=[low_objective])

synthetic_data

# %%
x_fake = synthetic_data.iloc[:, 0:2].values
y_fake = model.predict(x_fake)

# make figure
fig, ax = plt.subplots()
im = ax.scatter(*x_fake.T, s=4, c=y_fake)
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
ax.set_aspect('equal')
cb = plt.colorbar(im)

# plot the true minimum
ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

# plot the minimum by screening
min_idx = np.argmin(y_fake)
ax.plot(*x_fake[min_idx], '^', color='tab:orange', label='Best Result')
print(f'min value is {y_fake[min_idx]} at {xy_to_comp( x_fake[min_idx] )}')

ax.legend()

# %% [markdown]
# You will see here that the sampled points are all over the place.
# While they are generally biased towards the true minimum, they aren't any better than screening.

# %% [markdown]
# # Inverse function approximation

# %% [markdown]
# ## Normalizing flows
# 
# <img src="../lectures/assets/lecture20_normalizing_flow.jpg" alt="Conceptual diagram of Normalizing Flows for density estimation" width=600>

# %%
import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                          hidden_features=4))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

# %%
def plot_flow(flow, x):
    xline = torch.linspace(0, 1, 100)
    yline = torch.linspace(0, np.sqrt(3)/2, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

    fig, ax = plt.subplots()
    ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
    ax.plot(*x.T, 'w.')
    return fig

out = sample_tri(128)
fig = plot_flow(flow, out)

# %%
import tqdm
import numpy as np

num_iter = 1000
for i in tqdm.tqdm(np.arange(num_iter)):
    x_out = sample_tri(256)
    x_out = torch.tensor(x_out, dtype=torch.float32)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x_out).mean()
    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        fig = plot_flow(flow, x_out)
        fig.axes[0].set_title('iteration {}'.format(i + 1))
        plt.show()

# %%
x_out, y_out = flow.sample(1000).detach().numpy().T
fig, ax = plt.subplots()
ax.plot(x_out, y_out, '.')
ax.set_aspect('equal')

# %% [markdown]
# ## Conditional Normalizing Flows

# %%
import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet


num_layers = 5
base_dist = ConditionalDiagonalNormal(shape=[2],
                                      context_encoder=nn.Linear(1, 4))

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                          hidden_features=4,
                                                          context_features=1))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

# %%
def plot_cond_flow(flow, c=None):
    xline = torch.linspace(0, 1, 100)
    yline = torch.linspace(0, np.sqrt(3)/2, 100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    if c is None:
        c = torch.rand_like(xyinput[:, :1]) * 2 + 1

    with torch.no_grad():
        zgrid = flow.log_prob(xyinput, c*torch.ones_like(xyinput[:, :1])).exp().reshape(100, 100)

    fig, ax = plt.subplots()
    ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
    return fig

fig = plot_cond_flow(flow, 0.5)
fig = plot_cond_flow(flow, 1.0)
fig = plot_cond_flow(flow, 1.5)

# %%
import tqdm
import numpy as np

num_iter = 2000

x_out = torch.tensor(x, dtype=torch.float32)
y_out = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

for i in tqdm.tqdm(np.arange(num_iter)):

    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x_out, context=y_out).mean()
    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        fig = plot_cond_flow(flow, 1.0)
        fig.axes[0].set_title('iteration {}'.format(i + 1))
        plt.show()
        flow = flow

# %%
fig = plot_cond_flow(flow, 0.5)
fig = plot_cond_flow(flow, 1.0)
fig = plot_cond_flow(flow, 1.5)

# %%
# generate samples
targets = np.arange(0.5, 3.0, 0.25)
out = flow.sample(1024, context=torch.tensor(targets, dtype=torch.float32).reshape(-1, 1))
xy = out.detach().numpy()

# plot the samples
fig, ax = plt.subplots()
for i, xi in enumerate(xy):
    ax.plot(*xi.T, '.', label=f'c={targets[i]}', zorder=len(targets)-i, alpha=0.33)
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
ax.set_aspect('equal')
ax.legend()

# %%
# evaluate the results
metrics = np.array([model.predict(it) for it in xy])

# create parity plot
fig, ax = plt.subplots()
ax.errorbar(targets, metrics.mean(axis=1), yerr=metrics.std(axis=1), ls='none', marker='o', label='Generated')
ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', label='Reference')

ax.set_aspect('equal')
ax.set_xlabel('Condition')
ax.set_ylabel('Sampled')
ax.legend()

# %% [markdown]
# ## conditional Generative Adversarial Networks
# 
# <img src="../lectures/assets/lecture20_generative_models.jpg" alt="Different types of generative models like VAE, GAN, Flow-based" width=600>
# 
# Details for a future lecture...

# %%

