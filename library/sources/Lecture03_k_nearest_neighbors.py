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
# id: Lecture03_k_nearest_neighbors
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## K-Nearest Neighbors
#
# K-Nearest Neighbors, abbreviated KNN, or called K Neighbors, is a voting algorithm. Essentially we just look at the values of the $k$ nearest points to the unlabeled observation in question $X_i$ in feature space and use the neighbors' labels $\{ y \}_k$ to predict $y_i$. It would look something like this:
#
# <img src="../lectures/assets/lecture03_knn_concept.jpg" alt="K-Nearest Neighbors concept diagram" height=300>
#
# This is a relatively simple idea but it can work quite well for some problems, especially with a lot of training data.
# In some sense it is just an interpolation scheme, but it may work in very high dimensions.
#
# > Note: in contrast to linear regression (a **parametric** or **model-based** scheme), this is an **instance-based** learning scheme.
# Model-based learning involves fitting parameters that can be used to predict new outcomes, while instance-based learning involves comparing new observations to previous ones.
#
# This model performs slightly worse than the polynomial `LinearRegression` on the test set, though in training it appears reasonable. This is a clear case of overfitting, which demonstrates the need for train/test split.

# %%
from sklearn import neighbors

model = neighbors.KNeighborsRegressor()
model = model.fit(xtrain, ytrain)

from sklearn import neighbors

model = neighbors.KNeighborsRegressor().fit(xtrain, ytrain)

evaluate_model(model, xtrain, xtest, ytrain, ytest)

plot_model(model, xtrain, xtest, ytrain, ytest)
