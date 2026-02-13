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
# id: Lecture09_in_more_dimensions
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## In more dimensions
#
# Many models have more than two hyperparameters.
# The grid search can easily be extended to additional dimensions using nested `for` loops.
# For instance, consider the `max_depth`, `n_estimators`, and `min_samples_split` hyperparameters of the `RandomForestRegressor`:
#
# Note once again that even though the best validation RMSE is `4.84` at `(10, 40, 4)`, the test RMSE is `5.19` here and as low as `5.08` elsewhere (with validation RMSE slightly higher at `4.89`):

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

df = pd.DataFrame(results, columns=['md', 'ne', 'mss', 'Train', 'Validation', 'Test'])
px.scatter_3d(df, x='md', y='ne', z='mss', color='Validation')

df
