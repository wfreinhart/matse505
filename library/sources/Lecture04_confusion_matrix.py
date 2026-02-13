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
# id: Lecture04_confusion_matrix
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# ## Confusion matrix
#
# We can also evaluate the performance of the model graphically using a confusion matrix.
#
# The confusion matrix shows the number of labels in each category for both the predictions and the true observations.
#
# It can communicate a lot of information quickly without resorting to technical definitions such as the Accuracy, Precision, and Recall.
#
# Here's a reference:
#
# <img src="../lectures/assets/lecture04_confusion_matrix.jpg" alt="Confusion matrix diagram showing true positives, false positives, etc." height=300>
#
# We can create this confusion matrix using the convenient `sklearn.metrics.ConfusionMatrixDisplay` function:
#
# We can see from this chart that the `V` family has relatively few mistakes, while the `L` family seems to be very difficult to distinguish from `C` and `M`.
#
# At this moment it's difficult for us to determine which features are responsible for this.
#
# In the following lessons we will learn several tools that could help with this problem:
#
# * **representation learning** can help us understand how high-dimensional observations differ from each other
#
# * **feature importance** can help us quantify which features are responsible for certain decisions

# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, xtest, ytest)
