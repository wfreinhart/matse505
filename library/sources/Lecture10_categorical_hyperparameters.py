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
# id: Lecture10_categorical_hyperparameters
# type: Foundational
# parent_lecture: Lecture10
# ---
#
# ## Categorical hyperparameters
#
# We can also specify discrete options such as the activation functions.
# These can have a huge effect on the results:
#
# <img src="../lectures/assets/activation_functions.jpg" width=600 alt="Plots of common neural network activation functions including Sigmoid, Tanh, and ReLU">

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

print(best_parameters)
