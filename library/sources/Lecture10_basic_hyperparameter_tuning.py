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
# id: Lecture10_basic_hyperparameter_tuning
# type: Foundational
# parent_lecture: Lecture10
# ---
#
# ## Basic hyperparameter tuning
#
# We will use the `ax-platform` package for hyperparameter tuning.
# It is more convenient than implementing this ourselves.
#
# Now we need to do a train/validation/test split and an objective function.
#
# Now we set up the optimization problem in the format specified by `ax-platform`.
#
# With Gaussian Process, we can evaluate the results in terms of both mean and variance:
#
# It is very helpful to visualize the loss surface:
#
# And we can also view it as a function of iterations.
#
# Now that we have the optimal result, we should train a model with those hyperparameters.

# %%
# !pip install ax-platform

x_trv, x_test, y_trv, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_trv, y_trv, train_size=0.75, shuffle=True, random_state=0)

def mlp_fitness(parameterization):
    try:
        model = neural_network.MLPRegressor(**parameterization, random_state=0).fit(x_train, y_train)
        score = model.score(x_val, y_val)
    except:
        score = -1
    return score

from ax.service.managed_loop import optimize

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "learning_rate_init", "type": "range", "bounds": [1e-6, 1e-1], "log_scale": True},
        {"name": "max_iter", "type": "range", "bounds": [10, 1000]},
    ],
    evaluation_function=mlp_fitness,
    objective_name='r-squared',
)

print( best_parameters )
means, covariances = values
print( means, covariances )

from ax.plot.contour import plot_contour
from ax.utils.notebook.plotting import render

render(plot_contour(model=model, param_x='learning_rate_init', param_y='max_iter', metric_name='r-squared'))

from plotly import express as px
import numpy as np

best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
results = np.maximum.accumulate(best_objectives, axis=1).tolist()
px.line(y=results)

model = neural_network.MLPRegressor(**best_parameters, random_state=0).fit(x_train, y_train)
print( 'r-squared, val: ', model.score(x_val, y_val))
print( 'r-squared, test:', model.score(x_test, y_test))
