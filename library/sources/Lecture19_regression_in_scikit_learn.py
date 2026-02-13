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
# id: Lecture19_regression_in_scikit_learn
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## Regression in `scikit-learn`
#
# Let's evaluate the performance of 4 single-task models compared to a model that simultaneously predicts all 4 labels using `sklearn.neural_network`
#
# ### 1 model, 1 target
#
# We'll start by training 4 individual models, one for each target label:
#
# You see here that the model is highly overfitted to the training set in each task.
#
# ### 1 model, 4 targets
#
# What if we train a single model to predict all 4 targets at once?
#
# This is a tiny dataset!
# We can't expect good performance.
# Nevertheless, let's evaluate it:
#
# We can visualize this using a parity plot:
#
# You can see here that the result is quite bad.
# It is interesting to note that the performance on the "full" dataset (before making missing values) is much worse than the "test" dataset.
# This reveals that the generalization error is quite high and the model is highly overfitted (as expected from a NN with so few training data).

# %%
from sklearn import neural_network
from sklearn import preprocessing, model_selection

x_full = data.iloc[:, 1:-4].values.astype(float)
y_full = data.iloc[:, -4:].values.astype(float)

for i in range(4):
    print(f'Task {i}')
    print('------------------')

    # set up the feature and labels
    x_arr = missing.iloc[:, 1:-4].values.astype(float)
    y_arr = missing.iloc[:, i-4].values.astype(float).reshape(-1, 1)

    # filter out missing labels
    not_nan = np.isfinite(y_arr).flatten()

    # create scalers for both input and output features
    x_scaler = preprocessing.StandardScaler()
    x_sc = x_scaler.fit_transform(x_arr[not_nan])

    y_scaler = preprocessing.StandardScaler()
    y_sc = y_scaler.fit_transform(y_arr[not_nan]).flatten()

    # train / test split
    idx_train, idx_test = model_selection.train_test_split(np.arange(x_sc.shape[0]), random_state=0)

    # train a model
    model_skl = neural_network.MLPRegressor(hidden_layer_sizes=(128, 64, ),
                                            max_iter=1000, random_state=0)
    model_skl.fit(x_sc[idx_train], y_sc[idx_train])

    # calculate error on these train data
    residuals = model_skl.predict(x_sc[idx_train]) - y_sc[idx_train]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Train RMSE {i}: {rmse:.3f}')

    # calculate error on these test data
    residuals = model_skl.predict(x_sc[idx_test]) - y_sc[idx_test]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Test  RMSE {i}: {rmse:.3f}')

    # calculate error on the full dataset
    xf_sc = x_scaler.transform(x_full)
    yf_sc = y_scaler.transform(y_full[:, i-4].reshape(-1, 1))

    y_hat = model_skl.predict(xf_sc)
    residuals = y_hat - yf_sc
    print(f'Full  RMSE {i}: {np.sqrt(np.mean(residuals**2)):.3f}')
    print()

from sklearn import neural_network
from sklearn import preprocessing

x_arr = missing.iloc[:, 1:-4].values.astype(float)
y_arr = missing.iloc[:, -4:].values.astype(float)

not_nan = np.isfinite(y_arr.sum(axis=1)).flatten()

x_scaler = preprocessing.StandardScaler()
x_sc = x_scaler.fit_transform(x_arr[not_nan])

y_scaler = preprocessing.StandardScaler()
y_sc = y_scaler.fit_transform(y_arr[not_nan])

idx_train, idx_test = model_selection.train_test_split(np.arange(x_sc.shape[0]), random_state=0)

print(f'{len(idx_train)} train data, {len(idx_test)} test data')

model_skl = neural_network.MLPRegressor(hidden_layer_sizes=(128, 64, ), max_iter=2000, random_state=0)
_ = model_skl.fit(x_sc[idx_train], y_sc[idx_train])

for i in range(4):
    print(f'Task {i}')
    print('------------------')

    # calculate error on these train data
    residuals = model_skl.predict(x_sc[idx_train])[:, i] - y_sc[idx_train, i]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Train RMSE {i}: {rmse:.3f}')

    # calculate error on these test data
    residuals = model_skl.predict(x_sc[idx_test])[:, i] - y_sc[idx_test, i]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Test  RMSE {i}: {rmse:.3f}')

    # calculate error on the full dataset
    xf_sc = x_scaler.transform(x_full)
    yf_sc = y_scaler.transform(y_full)[:, i-4].reshape(-1, 1)

    y_hat = model_skl.predict(xf_sc)
    residuals = y_hat[:, i] - yf_sc
    print(f'Full  RMSE {i}: {np.sqrt(np.mean(residuals**2)):.3f}')
    print()

from matplotlib import pyplot as plt

# calculate error on the full dataset
xf_sc = x_scaler.transform(x_full)
yf_sc = y_scaler.transform(y_full)[:, -4:]

y_hat = model_skl.predict(xf_sc)

# make parity plot
fig, ax = plt.subplots()
ax.plot([-3, 3], [-3, 3], 'k--')
for i in range(y_hat.shape[1]):
    ax.scatter(y_hat[:, i], yf_sc[:, i], label=f'{missing.columns[i-4]}')

ax.set_xlabel('Predicted')
ax.set_ylabel('Observed')
ax.legend()
