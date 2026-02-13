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
# id: Lecture09_k_fold
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## K-Fold
#
# The basic implementation of cross-fold validation is the k-fold scheme.
# Here we divide the data directly into $k$ number of evenly sized folds like so:
#
# <img src="../lectures/assets/kfold_cv.jpg" width=600 alt="Schematic of K-Fold Cross-Validation showing the data split into k folds for iterative training and testing">
#
# What is the problem with this scheme?
#
# Let's explore this on the cement dataset:
#
# Here we see that the `Age (day)` variable has high and low values distributed throughout the folds.
# What if the data were entered in ascending order (such as in a spreadsheet or lab notebook while the cement was curing)?

# %%
# define the fold splitting strategy
folds = model_selection.KFold(n_splits=5)

in_fold = np.zeros([x.shape[0], folds.n_splits])
this_fold = np.zeros(x.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x.iloc[order]
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

# create a sorted dataset to illustrate the pathological cases
x_sort = x.sort_values(by='Age (day)')

# define the fold splitting strategy
folds = model_selection.KFold(n_splits=5)

in_fold = np.zeros([x_sort.shape[0], folds.n_splits])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort)):
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
