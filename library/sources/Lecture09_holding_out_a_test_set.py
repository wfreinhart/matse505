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
# id: Lecture09_holding_out_a_test_set
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## Holding out a test set
#
# What if we want to do cross-fold-validation-based hyperparameter tuning?
# Of course we will need to hold out a test set as well.
# We can do it as shown in this diagram:
#
# <img src="../lectures/assets/grid_search_cv.jpg" width=500 alt="Workflow combining Grid Search with Cross-Validation for robust hyperparameter optimization">
#
# All the models trained on the different fold-specific train/test splits can then be tested using the held-out test set.
#
# Again, we can aggregate these results into a mean and standard deviation:
#
# From the aggregated results we actually see that the performance is slightly worse in testing compared to validation, but the standard deviations are relatively low so we can have some confidence about the values.
#
# We can use the stabilization from cross-fold validation on the hyperparameter optimization problem from before:
#
# The chart shows that although validation does not always match test performance, the error bars provided by cross-fold validation do typically encompass the average test performance.

# %%
# first split out 20% for testing at the end
x_kf, x_test, y_kf, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True)

# now do cross-fold validation on the remaining data
folds = model_selection.KFold(n_splits=4, shuffle=True)
results = []
k = 0
for train_index, val_index in folds.split(x_kf):
    # define the train / test split for this fold
    x_train, x_val = x_kf.iloc[train_index], x_kf.iloc[val_index]
    y_train, y_val = y_kf.iloc[train_index], y_kf.iloc[val_index]

    # train and evaluate the model
    model = ensemble.RandomForestRegressor().fit(x_train, y_train)
    rmse_train = calc_rmse(model, x_train, y_train)
    rmse_val = calc_rmse(model, x_val, y_val)
    rmse_test = calc_rmse(model, x_test, y_test)

    results.append( [rmse_train, rmse_val, rmse_test] )

    k += 1
    print(f'fold k = {k} RMSE: Train = {rmse_train:.3f}; Val = {rmse_val:.3f}; Test = {rmse_test:.3f}')

mu = np.mean(results, axis=0)
sigma = np.std(results, axis=0)
for i, kind in enumerate(['Train', 'Val  ', 'Test ']):
    print(f'{kind} RMSE = {mu[i]:.3f} +/- {sigma[i]:.3f}')

def cv_rmse(model, X_train, y_train, X_test, y_test, X_val, y_val):
    # now do cross-fold validation on the remaining data
    folds = model_selection.KFold(n_splits=4, shuffle=True)
    results = []
    k = 0
    for train_index, val_index in folds.split(x_kf):
        # define the train / test split for this fold
        x_train, x_val = x_kf.iloc[train_index], x_kf.iloc[val_index]
        y_train, y_val = y_kf.iloc[train_index], y_kf.iloc[val_index]

        # train and evaluate the model
        model.fit(x_train, y_train)
        rmse_train = calc_rmse(model, x_train, y_train)
        rmse_val = calc_rmse(model, x_val, y_val)
        rmse_test = calc_rmse(model, x_test, y_test)

        results.append( [rmse_train, rmse_val, rmse_test] )

    return results

all_mu = []
all_sigma = []
for k in range(1, 20):
    print(f'k = {k:2d}: ', end='')
    model = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance')
    results = cv_rmse(model, x_train, y_train, x_test, y_test, x_val, y_val)
    mu = np.mean(results, axis=0)
    sigma = np.std(results, axis=0)
    for i, kind in enumerate(['Train', 'Val', 'Test']):
        print(f'{kind} = {mu[i]:.2f} +/- {sigma[i]:.2f}', end='; ')
    print()
    all_mu.append(mu)  # save these for later
    all_sigma.append(sigma)

from matplotlib import pyplot as plt

# convert to numpy arrays
mu = np.array(all_mu)
sigma = np.array(all_sigma)
k = np.arange(1, 20)
# set up the plot
fig, ax = plt.subplots()
ax.bar(k, mu[:, 1], yerr=sigma[:, 1], label='Val', width=0.5)
ax.bar(k, mu[:, 2], label='Test', width=0.5, align='edge', zorder=-1)
# labels and legend
ax.set_xlabel('$k$')
ax.set_ylabel('RMSE')
ax.legend()
# zoom in
ax.set_ylim(7.5, 10)
