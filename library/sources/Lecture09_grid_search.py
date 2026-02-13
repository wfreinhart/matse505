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
# id: Lecture09_grid_search
# type: Foundational
# parent_lecture: Lecture09
# ---
#
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
#
# Based on the stored results, we can visualize the grid of possible models:
#
# This shows the same result from above, that optimal validation performance is achieve with `weights=distance` and $k = 9$.

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

from plotly import express as px

# create a DataFrame with results
df = pd.DataFrame(results, columns=['k', 'w', 'Train', 'Validation', 'Test'])

# make a scatter plot of the grid
px.scatter(df, x='k', y='w', color='Validation')
