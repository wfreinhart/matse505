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
# id: Lecture04_decision_tree
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## Decision tree
#
# Now let's try with a Decision Tree.
# The interface is very similar to the code we gave you for regression, we just use a `DecisionTreeClassifier` instead of a `DecisionTreeRegressor`:
#
# Our simple model scored a perfect 100% accuracy on the test set, which it hadn't seen before! Let's explore this by drawing the tree itself:
#
# As you can see, the tree is very shallow and requires few decisions.
# This is actually because the `Alloy code` is defined by the composition of the alloy -- our decision tree is actually learning the real definition of the codes.
# In cases such as this, decision tree is actually a great option!
#
# We can also use the fitted tree to visualize the compositions -- we see that features 7, 9, and 6 completely define the tree!
#
# So another way to think about the decision tree algorithm is to imagine it dividing up the space into boxes which belong to a single class.

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=0).fit(xtrain, ytrain)

model.score(xtest, ytest)  # this gives the accuracy of the classifier

from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(24, 8))
_ = tree.plot_tree(model, ax=ax, fontsize=10, label='root', class_names=['C', 'L', 'M', 'V'],
                   impurity=False, precision=2, proportion=True)

from plotly import express as px
from sklearn import preprocessing

# can do this in one go with `fit_transform` if we don't need the encoder object
labels = preprocessing.LabelEncoder().fit_transform(y)

px.scatter_3d(x=x.iloc[:, 6], y=x.iloc[:, 7], z=x.iloc[:, 9], color=labels)
