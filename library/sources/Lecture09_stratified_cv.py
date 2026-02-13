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
# id: Lecture09_stratified_cv
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## Stratified CV
#
# We may want to take care to get a similar distribution in each fold.
# Here's what it looks like:
#
# <img src="../lectures/assets/stratified_cv.jpg" width=600 alt="Schematic of Stratified K-Fold CV ensuring each fold has the same class distribution as the original data">
#
# This can improve model performance but is not always a good assumption.
# Why would this be a problem?
#
# Let's see what it looks like on the cement data.
#
# Note this error message:
#
# `The least populated class in y has only 2 members, which is less than n_splits=5.`
#
# This is an indication that the `StratifiedKFold` object is trying to split the observations according to unique `Age (day)` values, but because those values are continuous, it is having trouble performing even splits.
# If we want to use stratification for continuous values, we should first transform the values to discrete bins.
# We did this before using the `KBinsDiscretizer`:
#
# Now we try again to implement the `StratifiedKFold` but reference the `classes` instead of the `Age (day)`:
#
# As you can see, the high and low values are more evenly distributed between different folds.
# This will reduce variance between *replicas* (repeated splits).

# %%
# define the fold splitting strategy
folds = model_selection.StratifiedKFold(n_splits=5)

in_fold = np.zeros([x_sort.shape[0], folds.n_splits])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort, x_sort['Age (day)'])):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)

from sklearn import preprocessing

# create the discretizer object
discretizer = preprocessing.KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
classes = discretizer.fit_transform(x_sort['Age (day)'].values.reshape(-1, 1))

fig, ax = plt.subplots()
_ = ax.plot(x_sort['Age (day)'], classes, '.')
_ = ax.set_xlabel('Age (day)')
_ = ax.set_ylabel('Discrete bin')

# define the fold splitting strategy
folds = model_selection.StratifiedKFold(n_splits=5)

in_fold = np.zeros([x_sort.shape[0], folds.n_splits])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort, classes)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)
