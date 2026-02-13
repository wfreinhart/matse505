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
# id: Lecture18_interpolation_and_imputation
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# # Interpolation and imputation
#
# Let's start with a quick high-level discussion about missing data.
# **Interpolation** and **imputation** are two techniques used to handle missing data in datasets.
#
# * **Interpolation** refers to the process of estimating missing values in a dataset by using the values of neighboring data points.
# This can be useful when working with time-series data or other types of data that exhibit a pattern over time.
#
# Here is what interpolation looks like in 2D:
#
# <img src="../lectures/assets/lecture18_interpolation.jpg" alt="Illustration of interpolation between discrete data points in 2D" width=600>
#
# * **Imputation** involves filling in missing values in a dataset with estimated values based on the available data. This can be useful when working with datasets that have a significant amount of missing data, and/or when the missing data are high-dimensional, as it allows you to still use the available data to train a machine learning model.
#
# Here is what imputation looks like on tabular data:
#
# <img src="../lectures/assets/lecture18_imputation.jpg" alt="Illustration of data imputation on tabular data" width=600>
