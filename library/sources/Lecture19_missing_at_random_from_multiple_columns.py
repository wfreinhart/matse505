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
# id: Lecture19_missing_at_random_from_multiple_columns
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## Missing at random from multiple columns
#
# Let's set up a `DataFrame` with more missing values.
#
# Just to prove to you that we have dropped a bunch of columns:
#
# Here is the problem: if we want to use all the columns for training a model, we now lost half our training data despite missing less than 3% of the values (because if one column is missing the entire row is invalidated).
#
# Also note that we are not missing `5 x 100 = 500` rows, instead only 398.
# This shows that a bunch of columns have multiple missing entries.
#
# The performance is degraded compared to the larger dataset without missing values.
# We can also check its performance on the held-out data from our artifical introduction of missing data:
#
# Here we can see the performance is even lower than implied by the test set from our reduced `missing` DataFrame.

# %%
import numpy as np

missing = data.copy(deep=True)

n_missing = 100
rng = np.random.default_rng(0)

miss_idx = []
for i, col in enumerate([' Si', ' Ni', ' Al', ' Temperature (°C)', ' Elongation (%)']):
    rand_idx = rng.choice(np.arange(missing.shape[0]), n_missing, replace=False)
    missing.loc[rand_idx, col] = np.nan
    miss_idx.append(rand_idx)

missing

print( missing.dropna().shape )

x = missing.dropna().iloc[:, 1:-1]
y = missing.dropna().iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

full_idx = np.sort(np.hstack(miss_idx))
x_full = data.iloc[full_idx, 1:-1]
y_full = data.iloc[full_idx, -1]

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')
