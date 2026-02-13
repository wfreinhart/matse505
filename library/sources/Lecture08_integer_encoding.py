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
# id: Lecture08_integer_encoding
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## Integer encoding
#
# Let's take a look at the first categorical column, `STP Phase`:
#
# Since there are only two possible values, we can represent this with a binary label using .
#
# Here's another way to achieve this result using `scikit-learn` functions:
#
# Once the `LabelEncoder` is fitted, you can obtain the original `str` with the `inverse_transform`:
#
# Finally, you can use the `numpy.unique` function to achieve a similar result:

# %%
data['STP Phase'].value_counts()

data['STP Phase'].astype("category").cat.codes

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
labels = encoder.fit_transform(data['STP Phase'])

print(labels)

print(encoder.inverse_transform(labels))

import numpy as np

cat, labels = np.unique(data['STP Phase'], return_inverse=True)

print(cat)
print(labels)
print(cat[labels])
