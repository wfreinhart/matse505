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
# id: Lecture09_leave_one_out
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## Leave-one-out
#
# An extreme version of the cross-fold validation is to use a number of folds equal to the number of observations.
# This effectively uses each and every sample as a test set.
# It would look like so:
#
# <img src="../lectures/assets/cross_validation_diagram.jpg" width=600 alt="Generalized flowchart of the cross-validation process">
#
# What are the downsides of this scheme?
#
# Let's see what this looks like on our cement data:

# %%
# define the fold splitting strategy
folds = model_selection.LeaveOneOut()

in_fold = np.zeros([x_sort.shape[0], folds.get_n_splits(x_sort)])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(this_fold)
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]

fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
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
