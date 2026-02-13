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
# id: Lecture18_when_interpolation_is_not_enough
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## When interpolation is not enough
#
# Linear interpolation works well when the underlying relationship between the data points is linear and the distance between the data points is relatively small. However, linear interpolation can break down and require more sophisticated methods in several situations:
#
# 1. *Non-linear relationships:* If the underlying relationship between the data points is non-linear, linear interpolation may produce inaccurate estimates. In these cases, more sophisticated interpolation methods such as polynomial or spline interpolation may be necessary.
#
# 2. *Unevenly spaced data:* If the distance between the data points is not uniform, linear interpolation may not accurately capture the underlying relationship between the data points. In these cases, methods such as spline interpolation or kriging may be more appropriate.
#
# 3. *Extrapolation:* Linear interpolation can only be used to estimate values between existing data points. If you need to estimate values outside of the range of the existing data points, you will need to use more sophisticated methods such as polynomial or spline extrapolation.
#
# 4. *Outliers:* Linear interpolation assumes that the data points are representative of the underlying relationship between the variables. If there are outliers or other unusual data points in the dataset, linear interpolation may produce inaccurate estimates. In these cases, more sophisticated interpolation methods such as robust regression or kernel regression may be necessary.
