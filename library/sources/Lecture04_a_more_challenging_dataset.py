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
# id: Lecture04_a_more_challenging_dataset
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## A more challenging dataset
#
# Instead of using the alloy compositions, let's try to predict the family based on the properties. This corresponds to the columns `0.2% Proof Stress (MPa)` through `Reduction in Area (%)`.
#
# Now we can test our `DecisionTreeClassifier` again.
#
# We see here the model accuracy is less than perfect. Does this hold up for the `KNeighborsClassifier`?
#
# Yes -- we are no longer dealing with a "definition" of the `Alloy code`.

# %%
x = data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']
y = data['Alloy family']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
print(xtrain.shape)

from sklearn import tree

model = tree.DecisionTreeClassifier().fit(xtrain, ytrain)

model.score(xtest, ytest)

from sklearn import neighbors

model = neighbors.KNeighborsClassifier().fit(xtrain, ytrain)

model.score(xtest, ytest)
