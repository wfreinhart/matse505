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
# id: Lecture19_practical_example
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## Practical example
#
# Let's randomly remove labels from some of the columns that correspond to mechanical properties:
#
# How many data are left if we remove rows without any target labels?
#
# Now let's check how much data is left in each combination of properties:
#
# You see here that with 400 entries from each property, we have only 45 entries with all 4 properties available.
# This will make it difficult to leverage all the data to train an accurate model.

# %%
import numpy as np

missing = data.copy(deep=True)

n_missing = 515
rng = np.random.default_rng(0)

miss_idx = []
for i, col in enumerate([' 0.2% Proof Stress (MPa)', ' Tensile Strength (MPa)',
                         ' Elongation (%)', ' Reduction in Area (%)']):
    rand_idx = rng.choice(np.arange(missing.shape[0]), n_missing, replace=False)
    missing.loc[rand_idx, col] = np.nan
    miss_idx.append(rand_idx)

missing

missing.iloc[:, -4:].dropna(how='all').shape

import itertools

for i in range(1, 5):
    for cols in itertools.combinations(range(4), i):
        sub_df = missing.iloc[:, -4:].iloc[:, list(cols)].dropna()
        print(f'{str(cols):12s} -> {sub_df.shape[0]} entries')
