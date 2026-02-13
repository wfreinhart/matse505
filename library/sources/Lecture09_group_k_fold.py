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
# id: Lecture09_group_k_fold
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## Group K Fold
#
# Sometimes there are additional identifiers in the data beyond discrete or continuous labels.
# For instance, imagine making multiple measurements on different physical material samples.
#
# <img src="../lectures/assets/group_cv_concept.jpg" width=600 alt="Visual explanation of Group-based Cross-Validation where samples are grouped by a shared identifier like session or patient">
#
# Some of these tensile bars might have been made on different days, or different machines, or tested by different personnel.
# It may be important to evaluate the effect of these variations through "grouping" the samples into specific folds.
# In this case, the group is extra information not already included in the model.
#
# Here is a schematic illustrating this concept:
#
# <img src="../lectures/assets/group_kfold.jpg" width=600 alt="Schematic of Group K-Fold Cross-Validation ensuring no group is split across training and validation folds">
#
# Note that unlike stratification, the classes are not balanced here, as only the groups are evaluated to create the folds.
# Of course there are "group" variants of all the other schemes, including `StratifiedGroupKFold`, `LeaveOneGroupOut`, etc.
#
# Let's try using `GroupKFold` CV to evaluate different groups in the cement dataset.
# Imagine there are discrete groups according to the `Superplasticizer` content:
#
# Now we'll use these bins to create group-based folds:
#
# What happened to our uniform group size and `Age` distribution?
# Creating folds by group naturally leads to imbalance in the size and data distribution in each fold as it becomes difficult to assign equally sized groups.
# While it will likely lead to worse model performance, it is also more realistic!
# How often do you get to decide what your new data will look like?

# %%
discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

y = x_sort['Superplasticizer (component 5)(kg in a m^3 mixture)']
groups = discretizer.fit_transform(y.values.reshape(-1, 1))

fig, ax = plt.subplots()
_ = ax.plot(y, groups, '.')
_ = ax.set_xlabel(y.name)
_ = ax.set_ylabel('Discrete bin')

# define the fold splitting strategy
folds = model_selection.GroupKFold(n_splits=5)

in_group = np.zeros([x_sort.shape[0], discretizer.n_bins])
for i, g in enumerate(groups.astype(int)):
    in_group[i, g] = 1

in_fold = np.zeros([x_sort.shape[0], folds.get_n_splits(x_sort, classes, groups=groups)])
this_fold = np.zeros(x_sort.shape[0])
for i, (train_index, test_index) in enumerate(folds.split(x_sort, classes, groups=groups)):
    in_fold[test_index, i] = 1
    this_fold[test_index] = i

order = np.argsort(groups.flatten())
x_sort_local = x_sort.iloc[order]
in_fold = in_fold[order]
in_group = in_group[order]

fig, axes = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
ax = axes[0]
_ = ax.imshow(x_sort_local['Age (day)'].values.reshape(1, -1), interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Age')
ax = axes[1]
_ = ax.imshow(in_group.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_ylabel('Group')
ax = axes[2]
_ = ax.imshow(in_fold.T, interpolation='none')
_ = ax.set_aspect('auto')
_ = ax.set_xlabel('Observation')
_ = ax.set_ylabel('Fold')
plt.subplots_adjust(hspace=0.05)
