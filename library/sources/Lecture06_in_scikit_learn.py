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
# id: Lecture06_in_scikit_learn
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## In `scikit-learn`
#
# Let's try out one of the `sklearn.semi_supervised` builtins:
#
# Let's remove 99% of the data by replacing the class labels with `-1` (per the instructions in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading)):
#
# We can visualize the result of this "un-labeling" by splitting the dataset up in the scatter plot:
#
# Now we fit the model and check its performance against the full labeled dataset:
#
# This performance may surprise you.
# Let's look at the result:

# %%
from sklearn import semi_supervised

x = data.loc[:, ' C':'Nb + Ta']
p = decomposition.PCA().fit_transform(x)

import numpy as np

remove_n = int(0.99 * x.shape[0])  # remove 98% of the labels!

sparse_y = np.array(y)  # create a copy of the labels

rng = np.random.RandomState(0)  # set random state so we always get same result
remove_idx = rng.choice(np.arange(y.shape[0]), remove_n, replace=False)

sparse_y[remove_idx] = -1  # remove some labels

idx = sparse_y > 0  # separate the plotted points into labeled/unlabeled

fig, ax = plt.subplots()
ax.scatter(p[idx, 0], p[idx, 1], c=sparse_y[idx])
ax.plot(p[~idx, 0], p[~idx, 1], '.', color=np.ones(3)*0.8, zorder=0)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

model = semi_supervised.LabelSpreading().fit(p, sparse_y)
print(model.score(p, y))

predicted_class = model.predict(p)
incorrect = (predicted_class^y).astype(bool)

fig, ax = plt.subplots()
ax.scatter(p[:, 0], p[:, 1], c=predicted_class)
ax.plot(p[incorrect, 0], p[incorrect, 1], 'rx')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
