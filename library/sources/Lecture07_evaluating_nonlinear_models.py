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
# id: Lecture07_evaluating_nonlinear_models
# type: Foundational
# parent_lecture: Lecture07
# ---
#
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
