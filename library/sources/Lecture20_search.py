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
# id: Lecture20_search
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Search
#
# We can employ Bayesian Optimization to search for the minimum:
#
# Your result will vary here.
# In my trials I sometimes got very close and other times much farther away.

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

from ax.utils.notebook.plotting import render
from ax.plot.contour import plot_contour

render(plot_contour(model=ax_model, param_x='x', param_y='y', metric_name='objective'))

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
