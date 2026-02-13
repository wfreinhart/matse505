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
# id: Lecture04_support_vector_machines
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## Support Vector Machines
#
# Support Vector Machine (SVM) is a ML method that identifies an *optimal boundary between different classes in the feature space*.
#
# step 1: the boundary is identified
#
# step 2: new observations are easily classified by checking which side of the boundary they fall on.
#
# While this is a very common method for classification, it can also be extended to do regression.
#
# Here is a simple classification example in 2D with only 2 classes:
#
# <img src="../lectures/assets/lecture04_svm_boundary_linear.jpg" width=500>
#
# This approach can be extended to multiple classes and higher dimensions.
# It can also be used with nonlinear decision boundaries (illustrated in the [SVM classification documentation](https://scikit-learn.org/stable/modules/svm.html#classification)):
#
# <img src="../lectures/assets/lecture04_svm_boundary_nonlinear.jpg">
#
# The nonlinearity is introduced through a function called a **kernel**.
#
# From [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method):
#
# > Kernel functions enable a method to operate in a   high-dimensional, implicit feature space without ever computing the coordinates of the data in that space, but rather by simply computing the inner products between the images of all pairs of data in the feature space.
# >
# >This operation is often computationally cheaper than the explicit computation of the coordinates.
# >
# > This approach is called the "kernel trick".
#
# We will discuss this further in our lesson on unsupervised learning.
#
# For now, let's test the performance of the SVM
#
# From these scores, it looks like adding nonlinearity to the decision boundary actually reduces performance.
#
# We can't tell from this simple analysis if this is because:
#
# *  the decision boundary is really linear, or
#
# * if we don't have enough training data to fit the boundary, or
#
# * if the boundary is nonlinear in a different way from any of these kernels.
#
# For more information on choosing kernels: https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html

# %%
from sklearn import svm

model = svm.SVC(kernel='linear', random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)

model = svm.SVC(kernel='rbf', random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)

model = svm.SVC(kernel='poly', random_state=0).fit(xtrain, ytrain)
model.score(xtest, ytest)
