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
# id: Lecture20_design_space
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Design space
#
# Our design space will be the composition and temperature of the alloys.
# We can evaluate our existing database to see what the known mechanical properties are.
#
# Let's imagine that we want to maximize Tensile Strength and Elongation (some proxy for ductility) at the same time.
# From the chart above you can see that these two properties compete with each other and lead to tradeoffs.
#
# We should consider the "Pareto front" AKA "efficient frontier" like so:
#
# <img src="../lectures/assets/lecture20_pareto_frontier.jpg" alt="Pareto efficient frontier showing trade-off between two objectives" width=400>
#
# Here is a short code to compute the efficient frontier:
#
# If we want to maximize both properties, the best solution would be one of these.
# Which solution is best depends on the relative value of each of the two objectives, strength or ductility.

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

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(y.iloc[:, 1], y.iloc[:, 2])
ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

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
