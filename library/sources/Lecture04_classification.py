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
# id: Lecture04_classification
# type: Foundational
# parent_lecture: Lecture04
# ---
#
# # Classification
#
# With regression we are predicting *continuous* labels, basically floating point numbers. However, some problems have *categorical* labels, which correspond to discrete groups rather than numbers.
#
# <img src="../lectures/assets/lecture04_regression_vs_classification.jpg" width=600>
#
# At first glance you might think we could just do regression with `int` in place of `float`. For instance, consider data falling in 3 groups: `Apple, Banana, Orange`. If we convert these to `int` so they are discrete, we would get `0, 1, 2`. Technically we can then use a regressor to predict the values.
#
# Aside from having to round off the outputs, this simple strategy makes a HUGE assumption in the math: that `Apple` is closer to `Banana` than it is to `Orange`, since they are represented by `0, 1, 2`. This will lead to systematic bias in the predictions and reward the wrong types of predictions, while having no basis in reality for the problem.
