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
# id: Lecture04_classification_metrics
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# # Classification Metrics
#
# Let's take a moment to discuss metrics for classification problems. Because the labels are not numerical, we can't use RMSE anymore.
# Basically, if the true label is `0`, we should penalize `1` or `2` the same amount.
#
# ![image](../lectures/assets/lecture04_categorical_vs_continuous.jpg)
#
# **Accuracy** is the most obvious metric for classification. Accuracy is simply the number of correct predictions divided by the total predictions; when we use `model.score`, it is telling us the fraction of correct predictions. We can also compute accuracy using `sklearn.metrics.accuracy_score`:
#
# **Precision** and **Recall** are the next most common metrics.
#
# Precision is the fraction of times the predicted label is correct, out of all the times that label is predicted.
#
# Recall is the fraction of times the correct label is predicted out of all the true labels of that type.
#
# It's possible for a model to have a poor accuracy with high precision, or poor accuracy with high recall. These values can also vary significantly by class - we'll see this in a bit.
#
# Finally, we can get what's called the **F1 Score**, which is the harmonic mean of the Precision and Recall:
#
# $F_1 = 2 \frac{\mathrm{precision} \cdot \mathrm{recall}}{ \mathrm{precision} + \mathrm{recall} }$
#
# This is basically just a way to balance the two measures in a single, combined measure that is more discriminating than accuracy.
#
# We can get all of this information from `sklearn` using `metrics.classification_report`:
#
# You will see a bunch of numbers here. Basically we get the precision, recall, and f1-score for each class individually, with the number of labels in the class (called "support"). We also get the overall accuracy underneath, and the precision, recall, and f1-scores again with different weighting schemes (macro and weighted avg).
#
# > If I ask you for the precision or recall score of a model, you can pick either of the two weighting schemes. I'll always mean the overall score for the model rather than for an individual class.

# %%
from sklearn import metrics

ypred = model.predict(xtest)
print(f'accuracy = {metrics.accuracy_score(ytest, ypred)}')

print(metrics.classification_report(ytest, ypred))
