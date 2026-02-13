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
# id: Lecture03_decision_trees
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Decision Trees
#
# Decision Trees are a nonlinear method that identify outcomes based on a separation of the features into discrete domains. The domains are chosen in a hierarchical manner which yields a tree structure, like so:
#
# <img src="../lectures/assets/lecture03_decision_tree.jpg" alt="Visual representation of a decision tree structure" width=600>
#
# In essence, each split while going down the tree defines a mapping of the data from input to output. Building a very deep tree can result in a pretty complex, nonlinear mapping between input and output.
#
# You can see that the tree completely fits the training data here, but is overfitted and does not perform perfectly on the test set. This is because the tree has unlimited depth by default.
# In practice we should limit the depth to increase transferability.
#
# We can provide the `max_depth` keyword argument to the `DecisionTreeRegressor` constructor.
# This is an example of a **hyperparameter** -- a model parameter that is not fitted during the course of learning, but rather chosen outside the learning procedure.
#
# Now we see a more comparable performance on train and test sets, though the overall performance went down on test data. The predictions look a bit odd, not like our other models. Let's investigate why this might be the case.
#
# We can visualize the decision tree itself using `plot_tree`:
#
# This is technically *traceable*, but maybe not *interpretable*. As in, we can see what happened, but probably not understand why. For instance, if we follow the leftmost branch of the tree, we see that at the very end a decision is made which gives one of two values depending on a simple binary cutoff. This sort of decisionmaking is not how we would like to interpret the data since the relationships are probably more like "the `Concrete compressive strength` increases at a rate of $m$ as component $X_i$ increases".
# You can see the striation resulting from these binary decisions in the output of the chart above.

# %%
from sklearn import tree

model = tree.DecisionTreeRegressor().fit(xtrain, ytrain)

evaluate_model(model, xtrain, xtest, ytrain, ytest)

plot_model(model, xtrain, xtest, ytrain, ytest)

model = tree.DecisionTreeRegressor(max_depth=5).fit(xtrain, ytrain)

evaluate_model(model, xtrain, xtest, ytrain, ytest)

plot_model(model, xtrain, xtest, ytrain, ytest)

fig, ax = plt.subplots(figsize=(24, 8))
_ = tree.plot_tree(model, ax=ax, fontsize=10, label='root',
                   impurity=False, precision=1, proportion=True)
