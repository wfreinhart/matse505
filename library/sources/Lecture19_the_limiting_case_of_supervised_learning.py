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
# id: Lecture19_the_limiting_case_of_supervised_learning
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## The limiting case of supervised learning
#
# In the limit of missing entries from only one column, the imputation problem becomes supervised learning.
#
# This is the same as having 229 missing entries from the `Reduction in Area (%)` column.
#
# Keep this performance in mind as we move ahead to talk about multi-column imputation.

# %%
from sklearn import model_selection

x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

print('feature names:\n', x.columns)
print()
print('label name:\n', y.name)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

print('dataset sizes:')
print(x_train.shape, x_test.shape)

from sklearn import ensemble

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')
