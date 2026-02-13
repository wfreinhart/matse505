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
# id: Lecture09_the_problem_of_extrapolation
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## The problem of extrapolation
#
# The schemes we have employed so far are a little idealistic in that they assume the same data distribution in the test set as seen during training.
# Actually this can be unusual in the physical sciences and engineering since we are often interested in making discoveries (i.e., exploring new data domains) with our models.
#
# Here's a schematic illustrating the difference between **interpolation** and **extrapolation**:
#
# <img src="../lectures/assets/interpolation_vs_extrapolation.jpg" width=600 alt="Plot comparing interpolation (making predictions within the range of training data) versus extrapolation (predicting outside the range)">
#
# How can we address this?
# By using **groups** to investigate how the models perform on data towards the center of the distribution versus the edges!
